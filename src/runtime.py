"""Run logging, checkpoint I/O, device selection, and validation shared by both models."""

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .data import to_device
from .metrics import RankingMetrics
from .models import ModelConfig, build_model

LOCAL_TIME = timezone(timedelta(hours=8))

LCFN_COUNTERS = ("lcfn_category_hits", "lcfn_item_hits", "lcfn_category_count_sum", "lcfn_item_count_sum")
LCFN_METRICS = ("lcfn_category_hit_rate", "lcfn_item_hit_rate", "lcfn_category_mean_count", "lcfn_item_mean_count")


def lcfn_totals(models):
    """Read additive candidate-weighted counters; replicas transfer through CPU."""
    result = {}
    for model in models:
        module = getattr(getattr(model, "features", None), "lcfn", None)
        if module is None or module.statistics is None:
            continue
        result["lcfn_candidates"] = result.get("lcfn_candidates", 0) + module.candidates
        for name, value in zip(module.statistic_names, module.statistics.detach().cpu().tolist()):
            result[name] = result.get(name, 0) + int(value)
    return result


def lcfn_metrics(totals):
    candidates = totals.get("lcfn_candidates", 0)
    if not candidates:
        return {}
    return {
        "lcfn_candidates": candidates,
        **{
            metric: totals[counter] / candidates
            for counter, metric in zip(LCFN_COUNTERS, LCFN_METRICS)
            if counter in totals
        },
    }


TB_TAGS = {
    "train": {
        "loss": "loss/train_total",
        "main_bce": "loss/train_main_bce",
        "target_rank": "loss/train_target_rank",
        "aux_listmle": "loss/train_aux_listmle",
        "pairwise": "loss/train_pairwise",
        "lr": "engineering/optimizer/learning_rate",
        "embedding_lr": "engineering/optimizer/embedding_learning_rate",
        "gradient_norm": "engineering/optimizer/gradient_norm",
        "candidates_per_second": "engineering/throughput/candidates_per_second",
        "feeds_per_second": "engineering/throughput/feeds_per_second",
        "data_seconds": "engineering/timing/data_seconds",
        "compute_seconds": "engineering/timing/compute_seconds",
        "pair_sampling_seconds": "engineering/timing/pair_sampling_seconds",
        "item_vector_transfer_mib": "engineering/io/item_vector_transfer_mib",
        "item_vector_transfer_ratio": "engineering/io/item_vector_transfer_ratio",
        "broadcast_seconds": "engineering/sync/broadcast_seconds",
        "gradient_merge_seconds": "engineering/sync/gradient_merge_seconds",
        "replica_compute_seconds": "engineering/timing/replica_compute_seconds",
        "gpu_allocated_gb": "engineering/memory/gpu_allocated_gib",
        "gpu_reserved_gb": "engineering/memory/gpu_reserved_gib",
        "gpu_peak_allocated_gb": "engineering/memory/gpu_peak_allocated_gib",
        "replica_gpu_allocated_gb": "engineering/memory/replica_gpu_allocated_gib",
        "replica_gpu_peak_allocated_gb": "engineering/memory/replica_gpu_peak_allocated_gib",
        "cpu_rss_gb": "engineering/memory/cpu_rss_gib",
        "epoch_progress": "engineering/progress/epoch_fraction",
        "eta_epoch_seconds": "engineering/progress/eta_seconds",
    },
    "validation": {
        "logloss": "loss/validation_main_bce",
        "ndcg": "effect/ndcg",
        "ndcg_at_10": "effect/ndcg_at_10",
        "ndcg_at_30": "effect/ndcg_at_30",
        "roc_auc": "effect/roc_auc",
        "average_precision": "effect/average_precision",
        "brier": "effect/brier",
    },
}


def new_run(model):
    return Path("output") / model / datetime.now(LOCAL_TIME).strftime("%Y%m%d_%H%M%S_%f")


def point_current_run(folder, model):
    folder, parent = Path(folder).resolve(), (Path("output") / model).resolve()
    if folder.parent != parent:
        return  # Custom/test directories must not replace the live dashboard's run.
    # Independent training processes can start together; their staging symlinks
    # must not unlink or rename one another before the atomic pointer replacement.
    current, temporary = parent / "current", parent / f".current.{os.getpid()}.tmp"
    if current.exists() and not current.is_symlink():
        raise FileExistsError(f"Current-run pointer is not a symlink: {current}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(folder.name, target_is_directory=True)
    temporary.replace(current)
    current, temporary = parent.parent / "current", parent.parent / f".current.{os.getpid()}.tmp"
    if current.exists() and not current.is_symlink():
        raise FileExistsError(f"Current-run pointer is not a symlink: {current}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(folder.relative_to(parent.parent), target_is_directory=True)
    temporary.replace(current)


def write_json(path, content):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_report(folder, summary):
    """One report per run, refreshed after every validation and test inference."""
    folder = Path(folder)
    config, model = summary["config"], summary["model_config"]
    history = summary.get("history", [])
    lines = ["# 运行报告", "", f"状态：{summary['status']}；模式：{summary['mode']}；模型：{model['name']}。", ""]

    def value(item):
        if item is None:
            return "—"
        return f"{item:.8f}" if isinstance(item, float) else str(item)

    def table(title, headers, rows):
        lines.extend(
            [f"## {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        )
        lines.extend("| " + " | ".join(value(item) for item in row) + " |" for row in rows)
        lines.append("")

    table(
        "模型与训练配置",
        ["项目", "设置"],
        [
            ("attention_layout", model.get("attention_layout", "cross")),
            ("width / tokens / depth", f"{model['width']} / {model['tokens']} / {model['depth']}"),
            ("Token FFN activation", model.get("token_ffn_activation", "gelu")),
            ("TokenMixer block residual", model.get("block_residual", False)),
            ("TokenMixer activation checkpoint", model.get("checkpoint_mixer", False)),
            ("BF16 Mixer normalization outputs (training only)", model.get("bf16_mixer_norm", False)),
            ("Mixer normalization backend", config.get("mixer_norm_backend", "not recorded")),
            ("Mixer normalization", model.get("mixer_norm", "layernorm")),
            ("DIN activation kernels", config.get("din_activation_kernels", "native")),
            ("additional normalized DIN context", model.get("din_softmax_context", False)),
            ("DIN query includes user/category/QR", model.get("din_query_context", False)),
            ("LCFN hit+count embeddings", model.get("lcfn", False)),
            ("LCFN fields", ", ".join(model.get("lcfn_fields", ("category", "item")))),
            ("LCFN dimension per table", model.get("lcfn_embedding_dim", 32)),
            ("DIN history length bucket (0 disables)", model.get("din_history_bucket", 0)),
            ("DIN backward recomputation", model.get("din_recompute", "on")),
            (
                "DIN / attention dim / candidate chunk",
                f"{model.get('din', False)} / {model.get('din_attention_dim', 32)} / {model.get('din_chunk_size', 2048)}",
            ),
            ("training compile", config.get("compile", False)),
            ("compiled components", config.get("compile_scope", "legacy partial compile")),
            ("CUDA allocator configuration", config.get("cuda_allocator_config", "not recorded")),
            (
                "category / event_type / event_source",
                f"{model['category_dim']} / {model['event_dim']} / {model['source_dim']}",
            ),
            ("numeric_bins", model.get("numeric_bins", 0)),
            ("f1-f5 embedding dim", model.get("numeric_embedding_dim", 32)),
            ("raw f1-f5 alongside buckets", model.get("numeric_residual", False)),
            ("within-feed f1-f5 zscores", model.get("relative_numeric", False)),
            ("embedding field scale 1/sqrt(d)", model.get("scale_embedding_fields", False)),
            ("continuous UI / history max / history max zscore", model.get("similarity_residual", False)),
            ("history rank embedding dim (newest first, not elapsed time)", model.get("history_position_dim", 0)),
            ("recent history cosine input window", model.get("recent_history_window", 0)),
            ("fast execution (grouped DIN / batched projections)", model.get("fast_execution", False)),
            ("feed-local candidate attention layers", model.get("candidate_context_depth", 0)),
            ("item ID embedding", f"QR / {model.get('item_id_dim', 0)}"),
            (
                "pairwise tower / weight / pairs / min gap",
                f"{model.get('pairwise_tower', False)} / "
                f"{config.get('pairwise_weight', 0)} / {config.get('pairwise_pairs', 64)} / {config.get('pairwise_gap', 4)}",
            ),
            ("main target-rank weight", config.get("target_rank_weight", 0)),
            ("main target-rank kind", config.get("target_rank_kind", "softmax")),
            ("epochs / global batch / workers", f"{config['epochs']} / {config['batch_size']} / {config['workers']}"),
            ("device / replica / amp", f"{config['device']} / {config.get('replica_device')} / {config['amp']}"),
            (
                "replica execution",
                (
                    "sequential same-GPU"
                    if config.get("sequential_replica")
                    else "parallel two-GPU" if config.get("replica_device") else "single device"
                ),
            ),
            ("lr / schedule", f"{config['lr']} / {config.get('lr_schedule','constant')}"),
            ("dense / embedding optimizer", f"AdamW / {config.get('embedding_optimizer', 'adamw')}"),
            ("embedding peak LR", config.get("embedding_lr", config["lr"])),
            (
                "embedding RMSprop alpha / eps / momentum / weight decay",
                "0.99 / 1e-8 / 0 / 0" if config.get("embedding_optimizer") == "rmsprop" else "not applicable",
            ),
            ("embedding optimizer switch", config.get("embedding_optimizer_switch", "none")),
            ("warmup_ratio / min_lr_ratio", f"{config.get('warmup_ratio')} / {config.get('min_lr_ratio')}"),
            ("aux_weight", config["aux_weight"]),
            ("best epoch / NDCG", f"{summary.get('best_epoch')} / {summary.get('best_score')}"),
        ],
    )
    quality_keys = ("ndcg", "ndcg_at_10", "ndcg_at_30", "roc_auc", "average_precision", "logloss", "brier")
    table(
        "每轮完整验证",
        ["Epoch", "NDCG", "NDCG@10", "NDCG@30", "AUC", "AP", "Logloss", "Brier"],
        [(row["epoch"], *(row.get("validation", {}).get(key) for key in quality_keys)) for row in history],
    )
    validation_predictions = [
        (row["epoch"], row.get("validation", {}).get("prediction_path"))
        for row in history
        if row.get("validation", {}).get("prediction_path")
    ]
    if validation_predictions:
        table(
            "验证逐候选预测",
            ["Epoch", "文件"],
            [(epoch, f"[{Path(path).name}]({Path(path).name})") for epoch, path in validation_predictions],
        )
    table(
        "训练损失与耗时",
        [
            "Epoch",
            "总 loss",
            "主 BCE",
            "主 target-rank",
            "target-rank feeds",
            "辅助 ListMLE",
            "Pairwise",
            "抽样 pair 数",
            "训练秒",
            "验证秒",
            "训练候选/秒",
        ],
        [
            (
                row["epoch"],
                row["loss"],
                row["main_bce"],
                row.get("target_rank"),
                row.get("target_rank_feeds", 0),
                row["aux_listmle"],
                row.get("pairwise"),
                row.get("pairwise_pairs", 0),
                row["seconds"],
                row.get("validation", {}).get("seconds"),
                row["candidates"] / max(row["seconds"], 1e-9),
            )
            for row in history
        ],
    )
    table(
        "样本覆盖与预测统计",
        ["Epoch", "训练 feeds", "训练候选", "辅助 feeds", "验证 feeds", "验证候选", "正例率", "平均预测概率"],
        [
            (
                row["epoch"],
                row["feeds"],
                row["candidates"],
                row["aux_feeds"],
                row.get("validation", {}).get("feeds"),
                row.get("validation", {}).get("candidates"),
                row.get("validation", {}).get("positive_rate"),
                row.get("validation", {}).get("mean_prediction"),
            )
            for row in history
        ],
    )
    peaks = {}
    if model.get("lcfn", False):
        table(
            "LCFN 匹配监控（分母为全部候选）",
            ["Epoch", "Split", "候选数", "类别匹配率", "商品匹配率", "类别平均次数", "商品平均次数"],
            [
                (row["epoch"], split, stats.get("lcfn_candidates"), *(stats.get(key) for key in LCFN_METRICS))
                for row in history
                for split, stats in (("train", row), ("val", row.get("validation", {})))
            ],
        )
    events = folder / "metrics.jsonl"
    if events.exists():
        for raw in events.read_text().splitlines():
            event = json.loads(raw)
            if event["stage"] == "train":
                entry = peaks.setdefault(event["epoch"], {})
                for key in ("gpu_peak_allocated_gb", "replica_gpu_peak_allocated_gb", "cpu_rss_gb"):
                    if key in event:
                        entry[key] = max(entry.get(key, 0), event[key])
    table(
        "工程峰值",
        ["Epoch", "主卡显存 GiB", "副卡显存 GiB", "进程 RSS GiB"],
        [
            (
                row["epoch"],
                *(
                    peaks.get(row["epoch"], {}).get(key)
                    for key in ("gpu_peak_allocated_gb", "replica_gpu_peak_allocated_gb", "cpu_rss_gb")
                ),
            )
            for row in history
        ],
    )
    inference = summary.get("inference")
    if inference:
        table("Test 推理与导出", ["项目", "结果"], list(inference.items()))
    else:
        lines.extend(["Test 推理尚未完成。", ""])
    lines.extend(
        ["原始完整精度及逐步 LR、梯度范数、吞吐和显存见 [run.json](run.json)、[metrics.jsonl](metrics.jsonl)。", ""]
    )
    temporary = folder / "report.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(folder / "report.md")


def save_checkpoint(path, state):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model.to(device), state


def device_from_name(name):
    device = torch.device(
        "cuda:0" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly")
    if device.type == "cuda":
        # Bind before autocast/backend initialization, which otherwise uses the
        # default GPU and can create an unrelated context on GPU 0.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def autocast(device, precision):
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("Use none/bf16 precision on CPU")
    return torch.autocast(
        device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16, enabled=precision != "none"
    )


def system_metrics(device):
    result = {"cpu_rss_gb": psutil.Process(os.getpid()).memory_info().rss / 2**30}
    if device.type == "cuda":
        result.update(
            gpu_allocated_gb=torch.cuda.memory_allocated(device) / 2**30,
            gpu_reserved_gb=torch.cuda.memory_reserved(device) / 2**30,
            gpu_peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
        )
    return result


class RunMonitor:
    def __init__(self, folder, tensorboard=True, purge_step=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"avito.{self.folder.resolve()}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s+0800 %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
        formatter.converter = lambda timestamp: datetime.fromtimestamp(timestamp, LOCAL_TIME).timetuple()
        handlers = [logging.StreamHandler()]
        if not os.environ.get("AVITO_STDERR_IN_RUNLOG"):
            handlers.append(logging.FileHandler(self.folder / "run.log"))
        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.events = (self.folder / "metrics.jsonl").open("a", buffering=1)
        self.writer = None
        if tensorboard:
            try:
                # Select TensorBoard's bundled TF-free backend; do not import TensorFlow or initialize its CUDA runtime.
                from tensorboard import compat
                from tensorboard.compat import tensorflow_stub

                compat.tf = tensorflow_stub
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.folder / "tensorboard"), purge_step=purge_step)
                self.writer.add_custom_scalars(
                    {
                        "Loss": {
                            "主任务 BCE": ["Multiline", ["loss/train_main_bce", "loss/validation_main_bce"]],
                            "主任务 TargetRank": ["Multiline", ["loss/train_target_rank"]],
                            "辅助 ListMLE": ["Multiline", ["loss/train_aux_listmle"]],
                            "Pairwise": ["Multiline", ["loss/train_pairwise"]],
                            "训练总损失": ["Multiline", ["loss/train_total"]],
                        },
                        "效果指标": {
                            "NDCG": ["Multiline", ["effect/ndcg", "effect/ndcg_at_10", "effect/ndcg_at_30"]],
                            "AUC": ["Multiline", ["effect/roc_auc"]],
                            "AP": ["Multiline", ["effect/average_precision"]],
                            "Brier": ["Multiline", ["effect/brier"]],
                        },
                        "工程指标": {
                            "候选吞吐": ["Multiline", ["engineering/throughput/candidates_per_second"]],
                            "feed 吞吐": ["Multiline", ["engineering/throughput/feeds_per_second"]],
                            "GPU 显存 GiB": [
                                "Multiline",
                                [
                                    "engineering/memory/gpu_allocated_gib",
                                    "engineering/memory/gpu_reserved_gib",
                                    "engineering/memory/gpu_peak_allocated_gib",
                                ],
                            ],
                            "CPU 内存 GiB": ["Multiline", ["engineering/memory/cpu_rss_gib"]],
                            "数据与计算耗时": [
                                "Multiline",
                                ["engineering/timing/data_seconds", "engineering/timing/compute_seconds"],
                            ],
                            "学习率": [
                                "Multiline",
                                [
                                    "engineering/optimizer/learning_rate",
                                    "engineering/optimizer/embedding_learning_rate",
                                ],
                            ],
                            "双卡同步耗时": [
                                "Multiline",
                                ["engineering/sync/broadcast_seconds", "engineering/sync/gradient_merge_seconds"],
                            ],
                            "梯度范数": ["Multiline", ["engineering/optimizer/gradient_norm"]],
                            "epoch 进度": ["Multiline", ["engineering/progress/epoch_fraction"]],
                            "epoch ETA 秒": ["Multiline", ["engineering/progress/eta_seconds"]],
                        },
                        "特征监控": {
                            name: [
                                "Multiline",
                                [
                                    f"features/lcfn/{stage}/{key}"
                                    for stage in ("train", "validation", "inference_complete")
                                ],
                            ]
                            for name, key in (
                                ("类别匹配率", "category_hit_rate"),
                                ("商品匹配率", "item_hit_rate"),
                                ("类别平均匹配次数", "category_mean_count"),
                                ("商品平均匹配次数", "item_mean_count"),
                            )
                        },
                    }
                )
            except ImportError:
                self.logger.warning("TensorBoard unavailable; run.log and metrics.jsonl remain enabled")

    def record(self, stage, step, values):
        event = {"time": datetime.now(LOCAL_TIME).isoformat(), "stage": stage, "step": step, **values}
        self.events.write(json.dumps(event, allow_nan=False) + "\n")
        self.logger.info("%s step=%s %s", stage, step, json.dumps(values, allow_nan=False))
        if self.writer:
            for name, value in values.items():
                tag = TB_TAGS.get(stage, {}).get(name)
                if name in LCFN_METRICS:
                    tag = f"features/lcfn/{stage}/{name.removeprefix('lcfn_')}"
                if tag and isinstance(value, (int, float)) and math.isfinite(value):
                    self.writer.add_scalar(tag, value, step)
            self.writer.flush()

    def close(self):
        self.events.close()
        if self.writer:
            self.writer.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


@torch.inference_mode()
def evaluate(model, loader, device, precision, monitor=None, step=0, log_every=100, prediction_path=None):
    model = getattr(model, "_orig_mod", model)
    model.eval()
    metric = RankingMetrics()
    feature_totals = {}
    start = perf_counter()
    prediction_path = Path(prediction_path) if prediction_path is not None else None
    temporary = Path(str(prediction_path) + ".tmp") if prediction_path is not None else None
    writer = None
    schema = pa.schema(
        [
            pa.field("feed_id", pa.int32()),
            pa.field("item_id", pa.int32()),
            pa.field("target", pa.float32()),
            pa.field("score", pa.float32()),
        ]
    )
    try:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(temporary, schema)
        for index, batch in enumerate(loader, 1):
            feed_id, item_id = batch["feed_id"], batch["item_id"]
            # Labels and auxiliary supervision are consumed by CPU metrics only.
            # Keep them off the accelerator instead of copying target to GPU and back.
            batch = to_device(batch, device, features_only=True)
            with autocast(device, precision):
                scores = model(batch, main_only=True)["main"].float()
            score_values = scores.cpu().numpy().astype(np.float32, copy=False)
            target_values = batch["target"].cpu().numpy().astype(np.float32, copy=False)
            metric.update(score_values, target_values, batch["counts"])
            for name, value in lcfn_totals([model]).items():
                feature_totals[name] = feature_totals.get(name, 0) + value
            if writer is not None:
                writer.write_table(
                    pa.table(
                        {
                            "feed_id": pa.array(feed_id, type=pa.int32()),
                            "item_id": pa.array(item_id, type=pa.int32()),
                            "target": pa.array(target_values, type=pa.float32()),
                            "score": pa.array(score_values, type=pa.float32()),
                        },
                        schema=schema,
                    )
                )
            if monitor and index % log_every == 0:
                monitor.logger.info(
                    "validation batches=%s feeds=%s elapsed_seconds=%.1f", index, metric.feeds, perf_counter() - start
                )
        expected_feeds = getattr(getattr(loader, "dataset", None), "feed_count", None)
        if expected_feeds is not None and metric.feeds != expected_feeds:
            raise ValueError(f"Validation feed coverage mismatch: {metric.feeds} != {expected_feeds}")
        result = metric.compute()
        result.update(lcfn_metrics(feature_totals))
        if writer is not None:
            writer.close()
            writer = None
            temporary.replace(prediction_path)
    except BaseException:
        if writer is not None:
            writer.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    seconds = perf_counter() - start
    result.update(
        seconds=seconds,
        candidates_per_second=result["candidates"] / seconds,
        auxiliary_listmle=None,
        **system_metrics(device),
    )
    if prediction_path is not None:
        result["prediction_path"] = str(prediction_path)
    if monitor:
        monitor.record("validation", step, result)
    return result
