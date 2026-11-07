"""Train/evaluate on official splits, or refit train+val and automatically infer test."""

import argparse
import json
import logging
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .data import FeedDataset, fit_numeric_bins, make_loader, split_feed_batch, to_device
from .infer import predict
from .losses import listmle, pairwise_loss, prepare_ranking_layout, sample_pairwise, target_rank_loss
from .models import ModelConfig, build_model, compile_for_training
from .runtime import (
    RunMonitor,
    autocast,
    device_from_name,
    evaluate,
    lcfn_metrics,
    lcfn_totals,
    new_run,
    point_current_run,
    save_checkpoint,
    system_metrics,
    write_json,
    write_report,
)


def epoch_steps(dataset, batch_size, workers, epoch, max_steps=0):
    order = np.arange(len(dataset.shards))
    if dataset.shuffle:
        np.random.default_rng(dataset.seed + epoch).shuffle(order)
    workers = max(workers, 1)
    # IterableDataset batches each worker's stream independently, including its own tail batch.
    steps = sum(
        math.ceil(sum(dataset.shards[index][2] for index in order[worker::workers]) / batch_size)
        for worker in range(workers)
    )
    return min(steps, max_steps) if max_steps else steps


def schedule_plan(config, dataset):
    total = sum(
        epoch_steps(dataset, config["batch_size"], config["workers"], epoch, config["max_steps"])
        for epoch in range(config["epochs"])
    )
    return {
        "kind": config["lr_schedule"],
        "base_lr": config["lr"],
        "total_steps": total,
        "warmup_steps": min(max(total - 1, 0), math.ceil(total * config["warmup_ratio"])),
        "min_lr_ratio": config["min_lr_ratio"],
    }


def learning_rate(plan, step):
    if plan["kind"] == "constant":
        return plan["base_lr"]
    warmup = plan["warmup_steps"]
    if step < warmup:
        return plan["base_lr"] * (step + 1) / warmup
    progress = min(max((step - warmup) / max(plan["total_steps"] - warmup - 1, 1), 0.0), 1.0)
    ratio = plan["min_lr_ratio"] + (1 - plan["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
    return plan["base_lr"] * ratio


def make_optimizers(model, config, device):
    tables = [module for module in model.modules() if isinstance(module, torch.nn.Embedding)]
    if any(table.sparse for table in tables):
        raise ValueError("RMSprop requires dense embedding gradients")
    embedding_ids = {id(parameter) for table in tables for parameter in table.parameters(recurse=False)}
    split = config["embedding_optimizer"] == "rmsprop"
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    dense = [(name, parameter) for name, parameter in named if not split or id(parameter) not in embedding_ids]
    embeddings = [(name, parameter) for name, parameter in named if split and id(parameter) in embedding_ids]
    if len(dense) + len(embeddings) != len(named) or not dense:
        raise ValueError("Optimizer parameter partition is incomplete")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in dense],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        fused=device.type == "cuda",
    )
    embedding_optimizer = (
        torch.optim.RMSprop(
            [parameter for _, parameter in embeddings],
            lr=config["embedding_lr"],
            alpha=0.99,
            eps=1e-8,
            momentum=0,
            weight_decay=0,
            foreach=device.type == "cuda",
        )
        if embeddings
        else None
    )
    names = {"adamw": [name for name, _ in dense], "embedding": [name for name, _ in embeddings]}
    return optimizer, embedding_optimizer, names


def restore_optimizers(optimizer, embedding_optimizer, restored, config):
    previous_kind = restored["config"].get("embedding_optimizer", "adamw")
    if previous_kind != config["embedding_optimizer"]:
        raise ValueError("Changing the embedding optimizer requires a fresh run without --resume")
    optimizer.load_state_dict(restored["optimizer"])
    if embedding_optimizer is not None:
        embedding_optimizer.load_state_dict(restored["embedding_optimizer"])


class ReplicaTrainer:
    """Two devices process one global batch; merged gradients receive one optimizer update each."""

    def __init__(self, model, model_config, config, device, restored=None):
        if model_config.name == "onetrans":
            from .onetrans import flash_varlen, prepare_fused_norm

            flash_varlen()
            prepare_fused_norm()
        self.models = [model]
        self.devices = [next(model.parameters()).device, torch.device(device)]
        self.config = config
        self.sequential_replica = self.devices[0] == self.devices[1]
        cpu_rng = torch.get_rng_state()
        replica = build_model(model_config).to(self.devices[1])
        torch.set_rng_state(cpu_rng)
        # Cross-socket CUDA peer copies on this host returned corrupted data in the preflight.
        # All inter-device transfers use explicit host staging, including initialization.
        replica.load_state_dict({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
        self.master_parameters = list(model.parameters())
        self.replica_parameters = list(replica.parameters())
        self.replica_weights = torch.nn.utils.parameters_to_vector(self.replica_parameters).detach()
        self.host_weights = torch.empty_like(
            self.replica_weights, device="cpu", pin_memory=self.devices[1].type == "cuda"
        )
        self.host_gradients = torch.empty_like(self.host_weights, pin_memory=self.devices[1].type == "cuda")
        if config.get("compile"):
            replica = compile_for_training(replica, config.get("mixer_ffn_kernels", "native"))
        self.models.append(replica)
        if not self.sequential_replica and restored is not None and restored.get("replica_cuda_rng") is not None:
            torch.cuda.set_rng_state(restored["replica_cuda_rng"], self.devices[1])
        self.pool = None if self.sequential_replica else ThreadPoolExecutor(max_workers=2)
        self.warm_shapes = set()
        self.debug_backward = os.environ.get("AVITO_ANOMALY") == "1"
        self.backward_steps = 0
        self.timings = {}
        self.work_seconds = [0.0, 0.0]

    def backward(self, batch):
        begin = perf_counter()
        master, replica = self.models
        with torch.no_grad():
            if self.sequential_replica:
                for target, source in zip(self.replica_parameters, self.master_parameters):
                    target.copy_(source)
            else:
                weights = torch.nn.utils.parameters_to_vector(self.master_parameters)
                self.host_weights.copy_(weights, non_blocking=False)
                self.replica_weights.copy_(self.host_weights, non_blocking=False)
                offset = 0
                for target in self.replica_parameters:
                    target.copy_(self.replica_weights[offset : offset + target.numel()].view_as(target))
                    offset += target.numel()
        if self.devices[1].type == "cuda":
            torch.cuda.synchronize(self.devices[1])
        replica.zero_grad(set_to_none=True)
        self.timings["broadcast_seconds"] = perf_counter() - begin
        halves = split_feed_batch(batch)
        total_candidates = int(batch["counts"].sum())
        total_target_rank = 0
        if self.config.get("target_rank_weight", 0):
            counts = np.asarray(batch["counts"], dtype=np.int64)
            offsets = np.cumsum(np.r_[0, counts[:-1]])
            positives = np.add.reduceat(batch["target"].numpy(), offsets)
            eligible_target = positives > 0
            if self.config.get("target_rank_kind", "softmax") == "ndcg_pairwise":
                eligible_target &= positives < counts
            total_target_rank = int(np.count_nonzero(eligible_target))
        total_aux = int((batch["position_available"] & (batch["counts"] > 1)).sum())
        total_pairwise = int(batch["pairwise_available"].sum()) if self.config.get("pairwise_weight", 0) else 0

        def work(index):
            work_start = perf_counter()
            model, device = self.models[index], self.devices[index]
            with (
                torch.cuda.device(device) if device.type == "cuda" else nullcontext(),
                torch.autograd.detect_anomaly() if self.debug_backward else nullcontext(),
            ):
                profile = (
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                    )
                    if os.environ.get("AVITO_PROFILE") == "1" and index == 0 and not self.warm_shapes
                    else None
                )
                if profile:
                    profile.start()
                model.train()
                ranking_layout = prepare_ranking_layout(
                    halves[index], device, self.config.get("target_rank_kind", "softmax")
                )
                local = to_device(halves[index], device)
                with autocast(device, self.config["amp"]):
                    outputs = model(local)
                    logits = outputs["main"].float()
                    main = F.binary_cross_entropy_with_logits(logits, local["target"])
                    if total_target_rank:
                        target_rank, eligible_target_rank = target_rank_loss(
                            logits,
                            local["target"],
                            local["counts"],
                            self.config.get("target_rank_kind", "softmax"),
                            ranking_layout,
                        )
                    else:
                        target_rank, eligible_target_rank = main.new_zeros(()), 0
                    aux, eligible = listmle(
                        outputs["aux"].float(),
                        local["position"],
                        local["counts"],
                        local["position_available"],
                        ranking_layout,
                    )
                    main = main * (logits.numel() / total_candidates)
                    target_rank = (
                        target_rank * (eligible_target_rank / total_target_rank)
                        if eligible_target_rank and total_target_rank
                        else main.new_zeros(())
                    )
                    aux = aux * (eligible / total_aux) if aux is not None and total_aux else main.new_zeros(())
                    pair = main.new_zeros(())
                    if total_pairwise:
                        pair, eligible_pairs = pairwise_loss(
                            outputs["pairwise"].float(),
                            local["pairwise_indices"],
                            local["counts"],
                            local["pairwise_available"],
                            ranking_layout,
                        )
                        pair = pair * (eligible_pairs / total_pairwise) if pair is not None else main.new_zeros(())
                    loss = (
                        main
                        + self.config.get("target_rank_weight", 0) * target_rank
                        + self.config["aux_weight"] * aux
                        + self.config.get("pairwise_weight", 0) * pair
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite parallel loss")
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                self.work_seconds[index] = perf_counter() - work_start
                if profile:
                    profile.stop()
                    logging.getLogger(f"avito.{Path(self.config['output']).resolve()}").info(
                        "Actual training operator timings (includes first graph initialization):\n%s",
                        profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30),
                    )
                return tuple(value.detach().cpu() for value in (logits, main, target_rank, aux, pair))

        shape = tuple((len(half["counts"]), half["history_item"].shape[1]) for half in halves)
        if (
            self.sequential_replica
            or not self.warm_shapes
            or (self.config.get("compile") and shape not in self.warm_shapes)
        ):
            # PyTorch 2.5 FX tracing is process-global. Compile each device's forward/backward
            # serially once per feed/history shape. Native Triton RMSNorm also needs one serial
            # initialization before threads, because its autotuner stores process-global state.
            # Same-GPU replicas are always serial so their activations do not overlap.
            results = [work(index) for index in range(2)]
            self.warm_shapes.add(shape)
        else:
            futures = [self.pool.submit(work, index) for index in range(2)]
            results = [future.result() for future in futures]
        self.backward_steps += 1
        self.debug_backward = os.environ.get("AVITO_ANOMALY") == "1" and self.backward_steps < 3
        if self.devices[0].type == "cuda":
            torch.cuda.synchronize(self.devices[0])
        active = [
            (target, source)
            for target, source in zip(self.master_parameters, self.replica_parameters)
            if source.grad is not None
        ]
        merge_start = perf_counter()
        if active and self.sequential_replica:
            for target, source in active:
                if target.grad is None:
                    target.grad = source.grad.clone()
                else:
                    target.grad.add_(source.grad)
        elif active:
            gradients = torch.cat([source.grad.reshape(-1) for _, source in active])
            host = self.host_gradients[: gradients.numel()]
            host.copy_(gradients, non_blocking=False)
            additions = host.to(self.devices[0], non_blocking=False)
            offset = 0
            for target, source in active:
                addition = additions[offset : offset + target.numel()].view_as(target)
                if target.grad is None:
                    target.grad = addition
                else:
                    target.grad.add_(addition)
                offset += target.numel()
        self.timings.update(
            gradient_merge_seconds=perf_counter() - merge_start,
            replica_compute_seconds=sum(self.work_seconds) if self.sequential_replica else max(self.work_seconds),
        )
        logits = torch.cat([result[0] for result in results])
        main = sum(result[1] for result in results)
        self.target_rank_loss = sum(result[2] for result in results)
        self.target_rank_feeds = total_target_rank
        aux = sum(result[3] for result in results)
        self.pairwise_loss = sum(result[4] for result in results)
        self.pairwise_feeds = total_pairwise
        return (
            logits,
            main,
            self.target_rank_loss,
            total_target_rank,
            aux,
            total_aux,
            main
            + self.config.get("target_rank_weight", 0) * self.target_rank_loss
            + self.config["aux_weight"] * aux
            + self.config.get("pairwise_weight", 0) * self.pairwise_loss,
        )

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        self.models.pop()
        self.replica_parameters.clear()
        del self.replica_weights
        if self.devices[1].type == "cuda":
            with torch.cuda.device(self.devices[1]):
                torch.cuda.empty_cache()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("train-eval", "full"), default="train-eval")
    result.add_argument("--model", choices=("mlp", "tokenmixer", "onetrans"), default="onetrans")
    result.add_argument("--data", type=Path, default=Path("data"))
    result.add_argument("--output", type=Path)
    result.add_argument(
        "--from-eval", type=Path, help="Full refit: inherit a complete evaluation run's recipe; default to two epochs"
    )
    result.add_argument("--resume", type=Path, help="Restore model/optimizer/scaler/RNG and replay the data cursor")
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--batch-size", type=int, default=32, help="Feeds per batch; candidate counts vary")
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--read-batch-size", type=int, default=256)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--device", default="auto")
    result.add_argument("--replica-device", help="Optional second CUDA device; keeps the global batch and LR plan")
    result.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    result.add_argument("--eval-amp", choices=("none", "bf16"), default="none", help="FP32 evaluation by default")
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--embedding-optimizer", choices=("adamw", "rmsprop"), default="rmsprop")
    result.add_argument("--embedding-lr", type=float, help="Embedding peak LR; defaults to --lr")
    result.add_argument("--lr-schedule", choices=("warmup_cosine", "constant"), default="warmup_cosine")
    result.add_argument("--warmup-ratio", type=float, default=0.05)
    result.add_argument("--min-lr-ratio", type=float, default=0.05)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--grad-clip", type=float, default=1.0)
    result.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    result.add_argument("--category-dim", type=int, default=64)
    result.add_argument("--event-dim", type=int, default=32)
    result.add_argument("--source-dim", type=int, default=32)
    result.add_argument("--numeric-bins", type=int, default=128)
    result.add_argument("--numeric-embedding-dim", type=int, default=32)
    result.add_argument("--numeric-residual", action="store_true", help="Append raw f1-f5 to bucket embeddings")
    result.add_argument("--relative-numeric", action="store_true", help="Append within-feed f1-f5 population zscores")
    result.add_argument(
        "--scale-embedding-fields", action="store_true", help="Scale enhanced embedding fields by 1/sqrt(dimension)"
    )
    result.add_argument(
        "--similarity-residual", action="store_true", help="Append continuous UI/history cosine residuals to DIN"
    )
    result.add_argument("--item-id-dim", type=int, default=32, help="Shared quotient/remainder item-ID embedding")
    result.add_argument(
        "--numeric-bin-file", type=Path, help="Reuse train-only bucket boundaries from a preflight JSON"
    )
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--width", type=int, default=256)
    result.add_argument("--depth", type=int, default=3)
    result.add_argument(
        "--block-residual", action="store_true", help="Add an input-output residual around each mixer block"
    )
    result.add_argument("--checkpoint-mixer", action="store_true", help="Recompute mixer activations on backward")
    result.add_argument(
        "--bf16-mixer-norm", action="store_true", help="Retain BF16 Mixer normalization outputs during BF16 training"
    )
    result.add_argument("--mixer-norm", choices=("layernorm", "rmsnorm"), default="layernorm")
    result.add_argument(
        "--mixer-norm-backend",
        choices=("flash_attn_triton", "parallel_triton", "parallel_triton_fused_mix"),
        default="flash_attn_triton",
    )
    result.add_argument("--expansion", type=int, default=4)
    result.add_argument("--token-ffn-activation", choices=("gelu", "swiglu"), default="swiglu")
    result.add_argument("--tokens", type=int, default=4)
    result.add_argument("--din", action="store_true", help="Add target-conditioned DIN interest to TokenMixer features")
    result.add_argument("--din-attention-dim", type=int, default=32)
    result.add_argument("--history-position-dim", type=int, default=0, help="DIN newest-first history rank embedding")
    result.add_argument("--recent-history-window", type=int, default=0, help="Append recent-history cosine inputs")
    result.add_argument("--fast-execution", action="store_true", help="Use grouped DIN and batched token projections")
    result.add_argument(
        "--din-softmax-context", action="store_true", help="Append normalized DIN context alongside its weighted sum"
    )
    result.add_argument("--din-chunk-size", type=int, default=2048)
    result.add_argument(
        "--din-query-context", action="store_true", help="Condition DIN on user/item/category/QR fields"
    )
    result.add_argument("--din-history-bucket", type=int, default=0, help="Group DIN histories by length; 0 disables")
    result.add_argument("--lcfn", action="store_true", help="Append category/item history hit and count embeddings")
    result.add_argument("--lcfn-embedding-dim", type=int, default=32)
    result.add_argument("--lcfn-fields", choices=("category", "item"), nargs="+", default=["category"])
    result.add_argument(
        "--din-recompute", choices=("on", "off"), default="off", help="Recompute DIN activations backward"
    )
    result.add_argument("--attention-heads", type=int, default=4)
    result.add_argument(
        "--candidate-context-depth", type=int, default=0, help="Feed-local candidate self-attention layers"
    )
    result.add_argument("--attention-layout", choices=("cross", "unified_prefix"), default="unified_prefix")
    result.add_argument("--aux-weight", type=float, default=0.1)
    result.add_argument("--target-rank-weight", type=float, default=0.0, help="Main-tower target ranking loss weight")
    result.add_argument("--target-rank-kind", choices=("softmax", "ndcg_pairwise"), default="softmax")
    result.add_argument("--pairwise-weight", type=float, default=0.1, help="Independent pairwise task weight")
    result.add_argument(
        "--pairwise-pairs", type=int, default=64, help="Random eligible pairs per feed, with replacement"
    )
    result.add_argument("--pairwise-gap", type=int, default=4, help="Minimum train.pos difference")
    result.add_argument("--user-embedding-dropout", type=float, default=0.1)
    result.add_argument("--item-embedding-dropout", type=float, default=0.1)
    result.add_argument("--embedding-corruption", choices=("zero", "random"), default="zero")
    result.add_argument("--monitor", choices=("ndcg", "ndcg_at_10", "ndcg_at_30"), default="ndcg")
    result.add_argument("--patience", type=int, default=2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument("--checkpoint-every", type=int, default=1000)
    result.add_argument("--max-feeds", type=int, default=0)
    result.add_argument("--val-max-feeds", type=int, default=0)
    result.add_argument("--test-max-feeds", type=int, default=0)
    result.add_argument("--max-steps", type=int, default=0, help="Per-epoch diagnostic limit; 0 means complete")
    result.add_argument("--no-tensorboard", action="store_true")
    result.add_argument(
        "--test-infer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer test after complete training; use --no-test-infer to disable",
    )
    result.add_argument("--compile", action="store_true", help="Compile the model with dynamic candidate counts")
    result.add_argument("--mixer-ffn-kernels", choices=("native", "compiled"), default="native")
    return result


def configuration(argv):
    args = parser().parse_args(argv)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    explicit = {token.split("=")[0].removeprefix("--").replace("-", "_") for token in argv if token.startswith("--")}
    state = None
    if args.resume and args.from_eval:
        raise ValueError("Choose resume or from-eval")
    if args.resume or args.from_eval:
        allowed = {
            "data",
            "output",
            "device",
            "workers",
            "threads",
            "log_every",
            "checkpoint_every",
            "no_tensorboard",
            "replica_device",
            "test_infer",
            "compile",
            "batch_size",
            "embedding_optimizer",
            "embedding_lr",
            "fast_execution",
            "din_recompute",
            "din_chunk_size",
            "mixer_ffn_kernels",
            "mixer_norm_backend",
        }
        if args.resume:
            # DataLoader worker count affects batch order; retain it during exact replay.
            allowed.remove("workers")
            allowed |= {"resume", "epochs"}
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            source = state["config"]
            if "embedding_optimizer" in explicit and args.embedding_optimizer != source.get(
                "embedding_optimizer", "adamw"
            ):
                raise ValueError("Changing the embedding optimizer requires a fresh run without --resume")
            if state["global_step"] == 0 and state["batch_cursor"] == 0:
                allowed.add("batch_size")
                if "batch_size" in explicit and args.batch_size != source["batch_size"]:
                    state["lr_plan"] = None
        else:
            allowed |= {"from_eval", "mode", "epochs"}
            source_run = json.loads((args.from_eval / "run.json").read_text())
            if source_run["mode"] != "train-eval" or source_run["status"] != "complete" or source_run["limited"]:
                raise ValueError("from-eval requires a completed evaluation run with complete train and val")
            if "mode" in explicit and args.mode != "full":
                raise ValueError("from-eval is only for full refitting")
            source = source_run["config"]
        if unexpected := explicit - allowed:
            raise ValueError(f"Recipe overrides are not allowed here: {sorted(unexpected)}")
        config = {**source, **{name: config[name] for name in explicit}}
        config["resume"], config["from_eval"] = str(args.resume) if args.resume else None, (
            str(args.from_eval) if args.from_eval else None
        )
        config["output"] = str(args.output) if args.output else str(args.resume.parent) if args.resume else None
        if args.from_eval:
            config.update(mode="full", epochs=args.epochs, max_feeds=0, val_max_feeds=0, max_steps=0)
            config.pop("embedding_optimizer_switch", None)
    config.setdefault("lr_schedule", "constant" if args.resume else "warmup_cosine")
    config.setdefault("warmup_ratio", 0.05)
    config.setdefault("min_lr_ratio", 0.05)
    config.setdefault("embedding_optimizer", "adamw")
    if config.get("embedding_lr") is None:
        config["embedding_lr"] = config["lr"]
    if config["embedding_lr"] <= 0:
        raise ValueError("Embedding learning rate must be positive")
    if not 0 <= config["warmup_ratio"] < 1 or not 0 <= config["min_lr_ratio"] <= 1:
        raise ValueError("warmup-ratio must be in [0,1); min-lr-ratio must be in [0,1]")
    positive = (
        "epochs",
        "batch_size",
        "read_batch_size",
        "threads",
        "log_every",
        "checkpoint_every",
        "grad_clip",
        "lr",
        "patience",
    )
    if (
        any(config[name] <= 0 for name in positive)
        or config["workers"] < 0
        or min(config["weight_decay"], config["aux_weight"]) < 0
    ):
        raise ValueError("Invalid training or logging parameters")
    if any(config[name] < 0 for name in ("max_feeds", "val_max_feeds", "test_max_feeds", "max_steps")):
        raise ValueError("Diagnostic limits cannot be negative")
    if config["mode"] == "full" and config["val_max_feeds"]:
        raise ValueError("Full refitting does not use a separate validation limit")
    config["output"] = config["output"] or str(new_run(config["model"]))
    config.setdefault("compile", False)
    config.setdefault("mixer_ffn_kernels", "native")
    if config["mixer_ffn_kernels"] == "compiled" and (
        not config["compile"] or config["model"] != "tokenmixer" or config.get("checkpoint_mixer", False)
    ):
        raise ValueError("Compiled Mixer FFN requires compiled TokenMixer with stored activations")
    config.pop("cuda_graphs", None)
    config.setdefault("eval_amp", "none")
    config.setdefault("attention_heads", 8)
    config.setdefault("attention_layout", "cross")
    config.setdefault("numeric_bins", 0)
    config.setdefault("numeric_embedding_dim", 32)
    config.setdefault("numeric_residual", False)
    config.setdefault("relative_numeric", False)
    config.setdefault("scale_embedding_fields", False)
    config.setdefault("similarity_residual", False)
    config.setdefault("history_position_dim", 0)
    config.setdefault("recent_history_window", 0)
    config.setdefault("fast_execution", False)
    config.setdefault("candidate_context_depth", 0)
    config.setdefault("numeric_bin_file", None)
    config.setdefault("item_id_dim", 0)
    config.setdefault("token_ffn_activation", "gelu")
    config.setdefault("din", False)
    config.setdefault("din_softmax_context", False)
    config.setdefault("din_query_context", False)
    config.setdefault("lcfn", False)
    config.setdefault("lcfn_embedding_dim", 32)
    config.setdefault("lcfn_fields", ["category", "item"])
    config.setdefault("din_attention_dim", 32)
    config.setdefault("din_chunk_size", 2048)
    config.setdefault("din_history_bucket", 0)
    config.setdefault("din_recompute", "on")
    config.setdefault("block_residual", False)
    config.setdefault("checkpoint_mixer", False)
    config.setdefault("bf16_mixer_norm", False)
    config.setdefault("mixer_norm", "layernorm")
    if config["bf16_mixer_norm"] and config["amp"] != "bf16":
        raise ValueError("BF16 mixer normalization requires BF16 training")
    if config["bf16_mixer_norm"]:
        config.setdefault("mixer_norm_backend", "flash_attn_triton")
    else:
        if config.get("mixer_norm_backend") in {"parallel_triton", "parallel_triton_fused_mix"}:
            raise ValueError("Parallel RMSNorm requires --bf16-mixer-norm")
        config["mixer_norm_backend"] = "native"
    config.setdefault("pairwise_weight", 0.0)
    config.setdefault("target_rank_weight", 0.0)
    config.setdefault("target_rank_kind", "softmax")
    config.setdefault("pairwise_pairs", 64)
    config.setdefault("pairwise_gap", 4)
    config.setdefault("pairwise_tower", bool(config["pairwise_weight"]))
    if (
        config["pairwise_weight"] < 0
        or config["target_rank_weight"] < 0
        or config["pairwise_pairs"] <= 0
        or config["pairwise_gap"] <= 0
    ):
        raise ValueError("Invalid ranking loss weight, pair count or position gap")
    if not (
        (config["model"] == "onetrans" and config["attention_layout"] == "unified_prefix")
        or (config["model"] == "tokenmixer" and config["din"])
    ):
        config["numeric_bins"] = 0
        config["item_id_dim"] = 0
        if config["scale_embedding_fields"]:
            raise ValueError("Embedding field scaling requires unified OneTrans or TokenMixer with DIN")
    config.setdefault("replica_device", None)
    if config["replica_device"] == "none":
        config["replica_device"] = None
    config.setdefault("test_infer", True)
    return config, state


def train(config, restored=None):
    # Set this before CUDA initializes; tmux servers do not inherit arbitrary client variables.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config["cuda_allocator_config"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    if config.get("compile"):
        config.pop("compile_disabled_reason", None)
    device = device_from_name(config["device"])
    if config["replica_device"] and (device.type != "cuda" or config["amp"] == "fp16"):
        raise ValueError("A replica needs CUDA and BF16 or FP32 training")
    config["sequential_replica"] = bool(config["replica_device"] and device == torch.device(config["replica_device"]))
    with autocast(device, config["amp"]):
        pass
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(config["seed"])
    model_config = ModelConfig(
        **{
            key: config[key]
            for key in (
                "hidden",
                "category_dim",
                "event_dim",
                "source_dim",
                "dropout",
                "width",
                "depth",
                "expansion",
                "tokens",
                "attention_heads",
                "attention_layout",
                "numeric_bins",
                "numeric_embedding_dim",
                "numeric_residual",
                "relative_numeric",
                "scale_embedding_fields",
                "similarity_residual",
                "history_position_dim",
                "recent_history_window",
                "fast_execution",
                "candidate_context_depth",
                "item_id_dim",
                "pairwise_tower",
                "token_ffn_activation",
                "din",
                "din_softmax_context",
                "din_query_context",
                "lcfn",
                "lcfn_embedding_dim",
                "lcfn_fields",
                "din_attention_dim",
                "din_chunk_size",
                "din_history_bucket",
                "din_recompute",
                "block_residual",
                "checkpoint_mixer",
                "bf16_mixer_norm",
                "mixer_norm",
                "mixer_norm_backend",
                "user_embedding_dropout",
                "item_embedding_dropout",
                "embedding_corruption",
            )
        },
        name=config["model"],
    )
    model = build_model(model_config).to(device)
    if config["numeric_bins"] and not restored:
        if config.get("numeric_bin_edges") is not None:
            edges = config["numeric_bin_edges"]
        elif config.get("numeric_bin_file"):
            prepared = json.loads(Path(config["numeric_bin_file"]).read_text())
            metadata = prepared["numeric_bin_fit"]
            if metadata["source"] != "train_only" or metadata["bins"] != config["numeric_bins"]:
                raise ValueError("Bucket boundaries must be fitted on train with the configured bucket count")
            edges = prepared["numeric_bin_edges"]
            config["numeric_bin_edges"], config["numeric_bin_fit"] = edges, metadata
        else:
            print("Preparing train-only quantile buckets for f1-f5...", flush=True)
            edges, metadata = fit_numeric_bins(config["data"], config["numeric_bins"], config["seed"])
            config["numeric_bin_edges"], config["numeric_bin_fit"] = edges.tolist(), metadata
        model.features.set_numeric_bins(edges)
    optimizer, embedding_optimizer, optimizer_names = make_optimizers(model, config, device)
    optimizers = [optimizer, *([embedding_optimizer] if embedding_optimizer is not None else [])]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["amp"] == "fp16")
    data, output = Path(config["data"]), Path(config["output"])
    if (
        output.exists()
        and {p.name for p in output.iterdir()} - {"compiler_cache", "preflight.json"}
        and restored is None
    ):
        raise FileExistsError(f"Use a fresh run directory: {output}")
    splits = [data / "train", *([data / "val"] if config["mode"] == "full" else [])]
    training = FeedDataset(splits, config["max_feeds"], True, config["seed"], config["read_batch_size"])
    validation = (
        FeedDataset([data / "val"], config["val_max_feeds"], read_batch_size=config["read_batch_size"])
        if config["mode"] == "train-eval"
        else None
    )
    if not training.labeled or (validation and not validation.labeled):
        raise ValueError("Training and validation require labels")
    lr_plan = restored.get("lr_plan") if restored else None
    lr_plan = lr_plan or schedule_plan(config, training)
    lr_plan.setdefault("reference_batch_size", restored["config"]["batch_size"] if restored else config["batch_size"])
    global_step, start_epoch, cursor = 0, 0, 0
    lr_position, data_cursor = 0.0, None
    best_score, best_epoch, bad_epochs, history, saved_stats = None, 0, 0, [], None
    if restored:
        previous_model_config = asdict(ModelConfig(**restored["model_config"]))
        current_model_config = asdict(model_config)
        # Execution kernels do not change parameter shapes, features, losses or optimizer states.
        previous_model_config.pop("fast_execution")
        current_model_config.pop("fast_execution")
        # DIN has no RNG-dependent operations inside its recomputation region.
        previous_model_config.pop("din_recompute")
        current_model_config.pop("din_recompute")
        # Candidate chunking changes execution grouping only; DIN has no per-group RNG.
        previous_model_config.pop("din_chunk_size")
        current_model_config.pop("din_chunk_size")
        # Switching kernels preserves the normalization definition and parameter keys.
        previous_model_config.pop("mixer_norm_backend")
        current_model_config.pop("mixer_norm_backend")
        if previous_model_config != current_model_config:
            raise ValueError("Resumed model configuration differs from the checkpoint")
        model.load_state_dict(restored["model"])
        restore_optimizers(optimizer, embedding_optimizer, restored, config)
        scaler.load_state_dict(restored["scaler"])
        global_step, start_epoch, cursor = restored["global_step"], restored["epoch"], restored["batch_cursor"]
        lr_position = restored.get("lr_position", float(global_step))
        data_cursor = restored.get("data_cursor")
        best_score, best_epoch, bad_epochs = restored["best_score"], restored["best_epoch"], restored["bad_epochs"]
        history, saved_stats = restored["history"], restored["epoch_stats"]
        torch.set_rng_state(restored["torch_rng"])
        if device.type == "cuda" and restored["cuda_rng"] is not None:
            torch.cuda.set_rng_state(restored["cuda_rng"], device)
    if config["compile"]:
        config["compile_scope"] = (
            "event_mlp_only"
            if model_config.name == "tokenmixer" and (model_config.checkpoint_mixer or model_config.din)
            else "event_mlp_and_blocks"
        )
        if config["mixer_ffn_kernels"] == "compiled":
            config["compile_scope"] = "event_mlp_and_mixer_ffn"
        cache = (
            os.environ.get("TORCHINDUCTOR_CACHE_DIR") or config.get("compiler_cache") or str(output / "compiler_cache")
        )
        config["compiler_cache"] = str(Path(cache).resolve())
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = config["compiler_cache"]
        model = compile_for_training(model, config["mixer_ffn_kernels"])
    parallel = (
        ReplicaTrainer(model, model_config, config, config["replica_device"], restored)
        if config["replica_device"]
        else None
    )
    monitor = RunMonitor(output, not config["no_tensorboard"], global_step + 1 if restored else None)
    previous_warning_handler = warnings.showwarning
    if os.environ.get("AVITO_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: monitor.logger.warning(
            "%s:%s %s: %s", filename, lineno, category.__name__, message
        )

        def check_logit_gradient(gradient):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Order/main loss produced a nonfinite logit gradient")
            return gradient

    limited = bool(config["max_feeds"] or config["max_steps"] or (validation and config["val_max_feeds"]))
    summary = {
        "mode": config["mode"],
        "status": "running",
        "limited": limited,
        "config": config,
        "model_config": asdict(model_config),
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "lr_plan": lr_plan,
    }

    def checkpoint(epoch, batch_cursor, epoch_stats):
        return {
            "config": config,
            "lr_plan": lr_plan,
            "lr_position": lr_position,
            "data_cursor": data_cursor if batch_cursor else None,
            "model_config": asdict(model_config),
            "model": getattr(model, "_orig_mod", model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "embedding_optimizer": embedding_optimizer.state_dict() if embedding_optimizer is not None else None,
            "optimizer_parameter_names": optimizer_names,
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "batch_cursor": batch_cursor,
            "epoch_stats": epoch_stats,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "replica_cuda_rng": (
                torch.cuda.get_rng_state(parallel.devices[1]) if parallel and not parallel.sequential_replica else None
            ),
        }

    profile_step = int(os.environ.get("AVITO_PROFILE_STEP", "0"))
    if profile_step and profile_step <= global_step:
        raise ValueError("AVITO_PROFILE_STEP must be a future global optimizer step")
    operator_profile = None
    try:
        write_json(output / "config.json", config)
        write_json(output / "run.json", summary)
        write_report(output, summary)
        if not limited:
            point_current_run(output, config["model"])
        monitor.record(
            "start",
            global_step,
            {
                "mode": config["mode"],
                "model": config["model"],
                "train_feeds": training.feed_count,
                "val_feeds": validation.feed_count if validation else 0,
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(device),
                "limited": limited,
                "resumed": restored is not None,
                "embedding_optimizer": config["embedding_optimizer"],
                "embedding_lr": config["embedding_lr"],
                "embedding_tables": len(optimizer_names["embedding"]),
                "embedding_optimizer_switch": config.get("embedding_optimizer_switch"),
                "target_rank_kind": config["target_rank_kind"],
                "target_rank_weight": config["target_rank_weight"],
                "compile_scope": config.get("compile_scope", "none"),
                "sequential_replica": config["sequential_replica"],
                "cuda_allocator_config": config["cuda_allocator_config"],
                "bf16_mixer_norm": config["bf16_mixer_norm"],
                "mixer_norm_backend": config["mixer_norm_backend"],
                "mixer_norm": config["mixer_norm"],
            },
        )
        if restored is None:
            save_checkpoint(output / "last.pt", checkpoint(0, 0, None))
        for epoch in range(start_epoch, config["epochs"]):
            if validation and bad_epochs >= config["patience"]:
                break
            training.epoch = epoch
            loader = make_loader(
                training,
                config["batch_size"],
                config["workers"],
                device,
                resume_batches=cursor if epoch == start_epoch else 0,
                resume_cursor=data_cursor if epoch == start_epoch and cursor else None,
                cursor_batch_size=restored["config"]["batch_size"] if restored else config["batch_size"],
            )
            data_cursor = {"rows": list(training.resume_rows), "next_worker": training.resume_worker_offset}
            model.train()
            totals = (
                saved_stats.copy()
                if saved_stats
                else {
                    "loss_sum": 0.0,
                    "main_sum": 0.0,
                    "aux_sum": 0.0,
                    "aux_feeds": 0,
                    "candidates": 0,
                    "feeds": 0,
                    "seconds": 0.0,
                }
            )
            saved_stats = None
            for key in ("target_rank_sum", "target_rank_feeds", "pairwise_sum", "pairwise_feeds", "pairwise_pairs"):
                totals.setdefault(key, 0)
            window_count, window_feeds, window_loss = 0, 0, 0.0
            window_data, window_compute = 0.0, 0.0
            window_main, window_target_rank, window_target_rank_feeds = 0.0, 0.0, 0
            window_aux, window_aux_feeds = 0.0, 0
            window_pairwise, window_pairwise_feeds = 0.0, 0
            window_features = {}
            start = end = window_start = perf_counter()
            prior_seconds = totals["seconds"]
            for batch_index, batch in enumerate(loader, start=cursor if epoch == start_epoch else 0):
                if config["max_steps"] and batch_index >= config["max_steps"]:
                    break
                received = perf_counter()
                if global_step + 1 == profile_step:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    operator_profile = torch.profiler.profile(activities=activities, record_shapes=True)
                    operator_profile.start()
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(lr_plan, lr_position)
                if embedding_optimizer is not None:
                    for group in embedding_optimizer.param_groups:
                        group["lr"] = learning_rate(lr_plan, lr_position) * config["embedding_lr"] / lr_plan["base_lr"]
                for active_optimizer in optimizers:
                    active_optimizer.zero_grad(set_to_none=True)
                if config["pairwise_weight"]:
                    sample_pairwise(batch, config["pairwise_pairs"], config["pairwise_gap"])
                if parallel and len(batch["counts"]) >= 2:
                    logits, main_loss, target_rank, target_rank_feeds, aux_loss, aux_feeds, loss = parallel.backward(
                        batch
                    )
                    pair_loss, pair_feeds = parallel.pairwise_loss, parallel.pairwise_feeds
                    batch["target"] = batch["target"].to(device, non_blocking=True)
                else:
                    ranking_layout = prepare_ranking_layout(batch, device, config["target_rank_kind"])
                    batch = to_device(batch, device)
                    with autocast(device, config["amp"]):
                        outputs = model(batch)
                        if os.environ.get("AVITO_ANOMALY") == "1":
                            for scores in outputs.values():
                                scores.register_hook(check_logit_gradient)
                        logits = outputs["main"].float()
                        main_loss = F.binary_cross_entropy_with_logits(logits, batch["target"])
                        if config["target_rank_weight"]:
                            target_rank, target_rank_feeds = target_rank_loss(
                                logits, batch["target"], batch["counts"], config["target_rank_kind"], ranking_layout
                            )
                        else:
                            target_rank, target_rank_feeds = main_loss.new_zeros(()), 0
                        aux_loss, aux_feeds = listmle(
                            outputs["aux"].float(),
                            batch["position"],
                            batch["counts"],
                            batch["position_available"],
                            ranking_layout,
                        )
                        loss = (
                            main_loss + config["aux_weight"] * aux_loss
                            if aux_loss is not None and config["aux_weight"]
                            else main_loss
                        )
                        if target_rank_feeds:
                            loss = loss + config["target_rank_weight"] * target_rank
                        pair_loss, pair_feeds = None, 0
                        if config["pairwise_weight"]:
                            pair_loss, pair_feeds = pairwise_loss(
                                outputs["pairwise"].float(),
                                batch["pairwise_indices"],
                                batch["counts"],
                                batch["pairwise_available"],
                                ranking_layout,
                            )
                            if pair_loss is not None:
                                loss = loss + config["pairwise_weight"] * pair_loss
                    scaler.scale(loss).backward()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                for active_optimizer in optimizers:
                    scaler.unscale_(active_optimizer)
                try:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["grad_clip"], error_if_nonfinite=True
                    )
                except RuntimeError:
                    invalid = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    ]
                    monitor.logger.error(
                        "Nonfinite gradient parameters=%s loss=%s main=%s aux=%s pair=%s",
                        invalid,
                        loss.detach().item(),
                        main_loss.detach().item(),
                        aux_loss.detach().item() if aux_loss is not None else None,
                        pair_loss.detach().item() if pair_loss is not None else None,
                    )
                    if parallel:
                        invalid_replica = [
                            name
                            for name, parameter in parallel.models[1].named_parameters()
                            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                        ]
                        monitor.logger.error("Nonfinite replica gradients=%s", invalid_replica)
                    raise
                for active_optimizer in optimizers:
                    scaler.step(active_optimizer)
                scaler.update()
                loss_value = loss.detach().item()
                main_value = main_loss.detach().item()
                target_rank_value = target_rank.detach().item() if target_rank is not None else 0.0
                aux_value = aux_loss.detach().item() if aux_loss is not None else 0.0
                pair_value = pair_loss.detach().item() if pair_loss is not None else 0.0
                feature_models = parallel.models if parallel and len(batch["counts"]) >= 2 else [model]
                for name, value in lcfn_totals(feature_models).items():
                    totals[name] = totals.get(name, 0) + value
                    window_features[name] = window_features.get(name, 0) + value
                if operator_profile is not None:
                    operator_profile.stop()
                    events = operator_profile.key_averages()
                    monitor.logger.info(
                        "Actual training step %s operator profile (profiling overhead included):\n%s\n%s",
                        global_step + 1,
                        events.table(sort_by="self_cuda_time_total", row_limit=25),
                        events.table(sort_by="self_cpu_time_total", row_limit=20),
                    )
                    monitor.record(
                        "profile", global_step + 1, {"scope": "actual_training_update", "timing_overhead": True}
                    )
                    operator_profile = None
                finished = perf_counter()
                count, feeds = logits.numel(), len(batch["counts"])
                worker_id = data_cursor["next_worker"]
                while data_cursor["rows"][worker_id] == training.worker_total_rows[worker_id]:
                    worker_id = (worker_id + 1) % len(data_cursor["rows"])
                data_cursor["rows"][worker_id] += feeds
                data_cursor["next_worker"] = (worker_id + 1) % len(data_cursor["rows"])
                totals["loss_sum"] += loss_value * count
                totals["main_sum"] += main_value * count
                totals["target_rank_sum"] += target_rank_value * target_rank_feeds
                totals["target_rank_feeds"] += target_rank_feeds
                totals["aux_sum"] += aux_value * aux_feeds
                totals["aux_feeds"] += aux_feeds
                totals["pairwise_sum"] += pair_value * pair_feeds
                totals["pairwise_feeds"] += pair_feeds
                totals["pairwise_pairs"] += pair_feeds * config["pairwise_pairs"]
                totals["candidates"] += count
                totals["feeds"] += feeds
                totals["seconds"] = prior_seconds + finished - start
                global_step += 1
                lr_position += config["batch_size"] / lr_plan["reference_batch_size"]
                window_count += count
                window_feeds += feeds
                window_loss += loss_value * count
                window_main += main_value * count
                window_target_rank += target_rank_value * target_rank_feeds
                window_target_rank_feeds += target_rank_feeds
                window_aux += aux_value * aux_feeds
                window_aux_feeds += aux_feeds
                window_pairwise += pair_value * pair_feeds
                window_pairwise_feeds += pair_feeds
                window_data += received - end
                window_compute += finished - received
                if global_step <= 3 or global_step % config["log_every"] == 0:
                    monitor.record(
                        "train",
                        global_step,
                        {
                            "epoch": epoch + 1,
                            "batch": batch_index + 1,
                            "loss": window_loss / window_count,
                            "main_bce": window_main / window_count,
                            "target_rank": (
                                window_target_rank / window_target_rank_feeds if window_target_rank_feeds else None
                            ),
                            "target_rank_feeds": window_target_rank_feeds,
                            "aux_listmle": window_aux / window_aux_feeds if window_aux_feeds else None,
                            "aux_feeds": window_aux_feeds,
                            "pairwise": window_pairwise / window_pairwise_feeds if window_pairwise_feeds else None,
                            "pairwise_feeds": window_pairwise_feeds,
                            "pairwise_pairs": window_pairwise_feeds * config["pairwise_pairs"],
                            "epoch_loss": totals["loss_sum"] / totals["candidates"],
                            "lr": optimizer.param_groups[0]["lr"],
                            "embedding_lr": (
                                embedding_optimizer.param_groups[0]["lr"]
                                if embedding_optimizer is not None
                                else optimizer.param_groups[0]["lr"]
                            ),
                            "gradient_norm": float(gradient_norm),
                            "positive_rate_batch": batch["target"].mean().item(),
                            "mean_prediction_batch": logits.detach().sigmoid().mean().item(),
                            **lcfn_metrics(window_features),
                            "candidates_per_second": window_count / (finished - window_start),
                            "feeds_per_second": window_feeds / (finished - window_start),
                            "data_seconds": window_data,
                            "compute_seconds": window_compute,
                            "epoch_progress": totals["feeds"] / training.feed_count,
                            "eta_epoch_seconds": max(training.feed_count - totals["feeds"], 0)
                            / (window_feeds / (finished - window_start)),
                            **system_metrics(device),
                            **(
                                {
                                    "replica_gpu_allocated_gb": torch.cuda.memory_allocated(parallel.devices[1])
                                    / 2**30,
                                    "replica_gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(
                                        parallel.devices[1]
                                    )
                                    / 2**30,
                                    **parallel.timings,
                                }
                                if parallel
                                else {}
                            ),
                        },
                    )
                    window_count, window_feeds, window_loss, window_data, window_compute = 0, 0, 0.0, 0.0, 0.0
                    window_main, window_target_rank, window_target_rank_feeds = 0.0, 0.0, 0
                    window_aux, window_aux_feeds = 0.0, 0
                    window_pairwise, window_pairwise_feeds = 0.0, 0
                    window_features = {}
                    window_start = perf_counter()
                    summary.update(global_step=global_step, epoch=epoch + 1, processed_feeds=totals["feeds"])
                    write_json(output / "run.json", summary)
                if global_step == 1 or global_step % config["checkpoint_every"] == 0:
                    save_checkpoint(output / "last.pt", checkpoint(epoch, batch_index + 1, totals))
                end = perf_counter()
            cursor = 0
            if not totals["candidates"]:
                raise RuntimeError("Training yielded no candidates; check worker sharding and resume cursor")
            if not limited and totals["feeds"] != training.feed_count:
                raise RuntimeError(f"Incomplete epoch coverage: {totals['feeds']} / {training.feed_count}")
            record = {
                "epoch": epoch + 1,
                "loss": totals["loss_sum"] / totals["candidates"],
                "main_bce": totals["main_sum"] / totals["candidates"],
                "target_rank": (
                    totals["target_rank_sum"] / totals["target_rank_feeds"] if totals["target_rank_feeds"] else None
                ),
                "target_rank_feeds": totals["target_rank_feeds"],
                "aux_listmle": totals["aux_sum"] / totals["aux_feeds"] if totals["aux_feeds"] else None,
                "aux_feeds": totals["aux_feeds"],
                "pairwise": totals["pairwise_sum"] / totals["pairwise_feeds"] if totals["pairwise_feeds"] else None,
                "pairwise_feeds": totals["pairwise_feeds"],
                "pairwise_pairs": totals["pairwise_pairs"],
                "feeds": totals["feeds"],
                "candidates": totals["candidates"],
                "seconds": totals["seconds"],
                **lcfn_metrics(totals),
            }
            monitor.record("train_epoch", global_step, record)
            improved = False
            if validation:
                quality = evaluate(
                    model,
                    make_loader(
                        validation,
                        (
                            min(config["batch_size"], 256 if config["attention_layout"] == "unified_prefix" else 1024)
                            if config["model"] == "onetrans"
                            else min(config["batch_size"], 1024)
                        ),
                        config["workers"],
                        device,
                    ),
                    device,
                    config["eval_amp"],
                    monitor,
                    global_step,
                    config["log_every"],
                    output / f"validation_epoch{epoch + 1}.parquet",
                )
                record["validation"] = quality
                score = quality[config["monitor"]]
                improved = best_score is None or score > best_score
                best_score, best_epoch, bad_epochs = (
                    (score, epoch + 1, 0) if improved else (best_score, best_epoch, bad_epochs + 1)
                )
            history.append(record)
            state = checkpoint(epoch + 1, 0, None)
            save_checkpoint(output / "last.pt", state)
            if improved:
                save_checkpoint(output / "best.pt", state)
            summary.update(history=history, best_score=best_score, best_epoch=best_epoch, epochs_completed=epoch + 1)
            write_json(output / "run.json", summary)
            write_report(output, summary)
        summary.update(
            status="training_complete" if not limited and config["test_infer"] else "complete",
            global_step=global_step,
            test_infer_enabled=config["test_infer"],
        )
        write_json(output / "run.json", summary)
        write_report(output, summary)
        monitor.logger.info("Training complete; output=%s best_epoch=%s", output, best_epoch)
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
        write_json(output / "run.json", summary)
        write_report(output, summary)
        monitor.logger.exception("Training interrupted/failed; last.pt contains the latest completed checkpoint")
        raise
    finally:
        if operator_profile is not None:
            operator_profile.stop()
        warnings.showwarning = previous_warning_handler
        if os.environ.get("AVITO_ANOMALY") == "1":
            torch.autograd.set_detect_anomaly(False)
        monitor.close()
        if parallel:
            parallel.close()
    if not limited and config["test_infer"]:
        try:
            summary["inference"] = predict(
                output / ("best.pt" if validation else "last.pt"),
                data,
                output,
                str(device),
                min(config["batch_size"], 1024),
                config["workers"],
                config["test_max_feeds"],
                config["log_every"],
                not config["no_tensorboard"],
                fast_execution=config["fast_execution"],
            )
            summary["status"] = "complete"
        except BaseException as error:
            summary.update(status="failed", error=f"Test inference failed: {error!r}")
            raise
        finally:
            write_json(output / "run.json", summary)
            write_report(output, summary)
    return summary


def main(argv=None):
    config, restored = configuration(sys.argv[1:] if argv is None else argv)
    return train(config, restored)


if __name__ == "__main__":
    main()
