"""Measure end-to-end native training throughput without changing official data or keeping diagnostic models."""

import argparse
import gc
import json
from dataclasses import asdict
from time import perf_counter

import torch
from torch.nn import functional as F

from .data import FeedDataset, fit_numeric_bins, make_loader, to_device
from .losses import listmle, pairwise_loss, sample_pairwise
from .models import ModelConfig, build_model, compile_for_training
from .runtime import autocast, device_from_name


def measure(
    data,
    device,
    batch_size,
    workers,
    steps,
    warmup,
    compile_model=False,
    model_name="onetrans",
    attention_heads=8,
    model_options=None,
    replica_device=None,
    numeric_bin_file=None,
    pairwise_weight=0.1,
    pairwise_pairs=64,
    pairwise_gap=4,
):
    device = device_from_name(device)
    with torch.cuda.device(device):
        free, total = torch.cuda.mem_get_info(device)
        budget = free * 0.85
        torch.cuda.set_per_process_memory_fraction(budget / total, device)
    config = ModelConfig(
        **{
            "name": model_name,
            "width": 128,
            "attention_heads": attention_heads,
            "pairwise_tower": bool(pairwise_weight),
            **(model_options or {}),
        }
    )
    model = build_model(config).to(device).train()
    if config.name == "onetrans" and config.attention_layout == "unified_prefix" and config.numeric_bins:
        edges = (
            json.loads(numeric_bin_file.read_text())["numeric_bin_edges"]
            if numeric_bin_file
            else fit_numeric_bins(data, config.numeric_bins)[0]
        )
        model.features.set_numeric_bins(edges)
    if compile_model:
        model = compile_for_training(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, fused=True)
    parallel = None
    if replica_device:
        from .train import ReplicaTrainer

        parallel = ReplicaTrainer(
            model,
            config,
            {"amp": "bf16", "aux_weight": 0.1, "compile": compile_model, "pairwise_weight": pairwise_weight},
            replica_device,
        )
    dataset = FeedDataset([data / "train"], shuffle=True)
    loader = iter(make_loader(dataset, batch_size, workers, device))
    torch.cuda.reset_peak_memory_stats(device)
    count, feeds, start = 0, 0, None
    data_seconds, compute_seconds = 0.0, 0.0
    try:
        for index in range(steps + warmup):
            if index == warmup:
                torch.cuda.synchronize(device)
                start = perf_counter()
            begin = perf_counter()
            cpu_batch = next(loader)
            if pairwise_weight:
                sample_pairwise(cpu_batch, pairwise_pairs, pairwise_gap)
            received = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if parallel:
                _, _, _, _, loss = parallel.backward(cpu_batch)
            else:
                batch = to_device(cpu_batch, device)
                with autocast(device, "bf16"):
                    outputs = model(batch)
                    main = F.binary_cross_entropy_with_logits(outputs["main"], batch["target"])
                    auxiliary, _ = listmle(
                        outputs["aux"], batch["position"], batch["counts"], batch["position_available"]
                    )
                    loss = main + 0.1 * auxiliary
                    if pairwise_weight:
                        pair, _ = pairwise_loss(
                            outputs["pairwise"], batch["pairwise_indices"], batch["counts"], batch["pairwise_available"]
                        )
                        if pair is not None:
                            loss = loss + pairwise_weight * pair
                loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite benchmark loss")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            if index >= warmup:
                count += len(cpu_batch["item_id"])
                feeds += len(cpu_batch["counts"])
                data_seconds += received - begin
                compute_seconds += perf_counter() - received
        torch.cuda.synchronize(device)
        seconds = perf_counter() - start
        return {
            "status": "ok",
            "device": str(device),
            "replica_device": replica_device,
            "batch_size": batch_size,
            "workers": workers,
            "amp": "bf16",
            "model_config": asdict(config),
            "measured_steps": steps,
            "candidates_per_second": count / seconds,
            "feeds_per_second": feeds / seconds,
            "estimated_epoch_seconds": dataset.feed_count / (feeds / seconds),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "memory_budget_gib": budget / 2**30,
            "compiled": compile_model,
            "compile_scope": (
                ("token_blocks_and_event_mlp" if model_name == "tokenmixer" else "event_mlp_ffns_final_query_mixer")
                if compile_model
                else None
            ),
            "evaluation_policy": "uncompiled FP32; this benchmark measures only training throughput",
            "data_seconds": data_seconds,
            "compute_seconds": compute_seconds,
        }
    finally:
        try:
            loader._shutdown_workers()
        except RuntimeError:
            for worker in loader._workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
        del loader, model, optimizer
        if parallel:
            parallel.close()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--model", choices=("tokenmixer", "onetrans", "mlp"), default="onetrans")
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--category-dim", type=int, default=64)
    parser.add_argument("--event-dim", type=int, default=32)
    parser.add_argument("--source-dim", type=int, default=32)
    parser.add_argument("--attention-layout", choices=("cross", "unified_prefix"), default="unified_prefix")
    parser.add_argument("--numeric-bins", type=int, default=128)
    parser.add_argument("--numeric-embedding-dim", type=int, default=32)
    parser.add_argument("--item-id-dim", type=int, default=32)
    parser.add_argument("--token-ffn-activation", choices=("gelu", "swiglu"), default="swiglu")
    parser.add_argument("--pairwise-weight", type=float, default=0.1)
    parser.add_argument("--pairwise-pairs", type=int, default=64)
    parser.add_argument("--pairwise-gap", type=int, default=4)
    parser.add_argument("--numeric-bin-file", type=Path)
    parser.add_argument("--replica-device")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        result = measure(
            args.data,
            args.device,
            args.batch_size,
            args.workers,
            args.steps,
            args.warmup,
            args.compile,
            args.model,
            args.attention_heads,
            {
                name: getattr(args, name)
                for name in (
                    "width",
                    "category_dim",
                    "event_dim",
                    "source_dim",
                    "attention_layout",
                    "numeric_bins",
                    "numeric_embedding_dim",
                    "item_id_dim",
                    "token_ffn_activation",
                )
            },
            args.replica_device,
            args.numeric_bin_file,
            args.pairwise_weight,
            args.pairwise_pairs,
            args.pairwise_gap,
        )
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "device": args.device, "batch_size": args.batch_size, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
