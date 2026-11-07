"""Stream test submissions or labeled official-val predictions for error analysis."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits

from .data import FeedDataset, make_loader, to_device
from .metrics import RankingMetrics
from .models import enable_fast_execution
from .runtime import (
    RunMonitor,
    autocast,
    device_from_name,
    evaluate,
    lcfn_metrics,
    lcfn_totals,
    load_model,
    new_run,
    system_metrics,
    write_json,
)


def validate_predictions(path, template_path, complete=True):
    frame, template = pl.read_parquet(path), pl.read_parquet(template_path)
    keys = ["feed_id", "item_id"]
    if dict(frame.schema) != {"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32}:
        raise ValueError(f"Invalid prediction schema: {frame.schema}")
    if frame.is_empty() or frame.null_count().to_numpy().any() or not frame["score"].is_finite().all():
        raise ValueError("Predictions must be nonempty, non-null, and finite")
    if frame.select(keys).n_unique() != len(frame):
        raise ValueError("Duplicate prediction keys")
    if not frame.select(keys).join(template.select(keys), on=keys, how="anti").is_empty():
        raise ValueError("Predictions contain keys outside the official template")
    if complete and len(frame) != len(template):
        raise ValueError("Predictions do not cover the complete official template")
    return {"candidates": len(frame), "feeds": frame["feed_id"].n_unique(), "submission_ready": complete}


@torch.inference_mode()
def predict(
    checkpoint,
    data,
    output=None,
    device="auto",
    batch_size=32,
    workers=2,
    max_feeds=0,
    log_every=100,
    tensorboard=True,
    fast_execution=False,
):
    device = device_from_name(str(device))
    model, state = load_model(checkpoint, device)
    if fast_execution:
        enable_fast_execution(model)
    model.eval()
    precision = state["config"].get("eval_amp", "none")
    if (
        state["model_config"]["name"] == "onetrans"
        and state["model_config"].get("attention_layout") == "unified_prefix"
    ):
        batch_size = min(batch_size, 256)
    data, output = Path(data), Path(output) if output else Path(checkpoint).parent
    dataset = FeedDataset([data / "test"], max_feeds=max_feeds)
    loader = make_loader(dataset, batch_size, workers, device)
    monitor = RunMonitor(output, tensorboard)
    destination = output / ("diagnostic_predictions.parquet" if max_feeds else "submission.parquet")
    temporary = output / ("." + destination.name + ".partial")
    start, candidates, feeds = perf_counter(), 0, 0
    feature_totals = {}
    created = False
    try:
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Use a fresh output directory: {destination}")
        schema = pa.schema([("feed_id", pa.int32()), ("item_id", pa.int32()), ("score", pa.float32())])
        created = True
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for index, batch in enumerate(loader, 1):
                inputs = to_device(batch, device, features_only=True)
                with autocast(device, precision):
                    scores = model(inputs, main_only=True)["main"].float().cpu().numpy()
                for name, value in lcfn_totals([model]).items():
                    feature_totals[name] = feature_totals.get(name, 0) + value
                if not np.isfinite(scores).all():
                    raise ValueError("Model produced nonfinite test scores")
                writer.write_table(
                    pa.Table.from_pydict(
                        {"feed_id": batch["feed_id"], "item_id": batch["item_id"], "score": scores}, schema=schema
                    )
                )
                candidates += len(scores)
                feeds += len(batch["counts"])
                if index % log_every == 0:
                    monitor.record(
                        "inference",
                        index,
                        {
                            "candidates": candidates,
                            "feeds": feeds,
                            **lcfn_metrics(feature_totals),
                            "candidates_per_second": candidates / (perf_counter() - start),
                            **system_metrics(device),
                        },
                    )
        result = validate_predictions(temporary, data / "sample_submission.parquet", complete=not max_feeds)
        temporary.replace(destination)
        result.update(
            seconds=perf_counter() - start,
            checkpoint=str(checkpoint),
            path=str(destination),
            amp=precision,
            fast_execution=bool(getattr(model, "fast_execution", False)),
            batch_size=batch_size,
            **lcfn_metrics(feature_totals),
        )
        monitor.record("inference_complete", state["global_step"], {**result, **system_metrics(device)})
        write_json(output / "inference.json", result)
        return result
    except Exception:
        monitor.logger.exception("Inference failed")
        if created:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        monitor.close()


@torch.inference_mode()
def predict_validation(
    checkpoint,
    data,
    output=None,
    device="auto",
    batch_size=1024,
    workers=8,
    max_feeds=0,
    log_every=100,
    tensorboard=False,
    fast_execution=False,
):
    device = device_from_name(str(device))
    model, state = load_model(checkpoint, device)
    if fast_execution:
        enable_fast_execution(model)
    output = Path(output) if output else Path(checkpoint).parent
    destination = output / ("validation_diagnostic.parquet" if max_feeds else "validation_predictions.parquet")
    metadata = output / ("validation_diagnostic.json" if max_feeds else "validation_predictions.json")
    if destination.exists() or metadata.exists():
        raise FileExistsError(f"Validation predictions already exist: {destination}")
    if state["model_config"]["name"] == "onetrans":
        batch_size = min(batch_size, 256)
    dataset = FeedDataset([Path(data) / "val"], max_feeds=max_feeds)
    if not dataset.labeled:
        raise ValueError("Validation predictions require official val labels")
    monitor = RunMonitor(output, tensorboard)
    try:
        result = evaluate(
            model,
            make_loader(dataset, batch_size, workers, device),
            device,
            "none",
            monitor,
            state["global_step"],
            log_every,
            prediction_path=destination,
        )
        if result["feeds"] != dataset.feed_count:
            raise ValueError("Validation inference did not cover all requested feeds")
        result.update(
            checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_epoch=state["epoch"],
            limited=bool(max_feeds),
            split="val",
            amp="none",
            fast_execution=bool(getattr(model, "fast_execution", False)),
            batch_size=batch_size,
        )
        write_json(metadata, result)
        return result
    finally:
        monitor.close()


def _read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _resolved(path):
    return str(Path(path).resolve())


def _validation_frame(path):
    frame = pl.read_parquet(path)
    expected = {"feed_id": pl.Int32, "item_id": pl.Int32, "target": pl.Float32, "score": pl.Float32}
    if dict(frame.schema) != expected:
        raise ValueError(f"Invalid validation prediction schema: {path}: {frame.schema}")
    if frame.is_empty() or frame.null_count().to_numpy().any() or not frame["score"].is_finite().all():
        raise ValueError(f"Validation predictions must be nonempty, non-null, and finite: {path}")
    if frame.select(["feed_id", "item_id"]).n_unique() != len(frame):
        raise ValueError(f"Duplicate validation prediction keys: {path}")
    return frame.sort(["feed_id", "item_id"])


def _source_run(folder, data):
    folder = Path(folder).resolve()
    run = _read_json(folder / "run.json")
    if run.get("status") != "complete" or run.get("limited") or run.get("mode") != "train-eval":
        raise ValueError(f"Source run must be complete, unlimited train-eval: {folder}")
    inference = _read_json(folder / "inference.json")
    validation_path = folder / "validation_predictions.parquet"
    if (folder / "validation_predictions.json").exists():
        metadata = _read_json(folder / "validation_predictions.json")
    else:
        selected = next((row for row in run["history"] if row["epoch"] == run["best_epoch"]), None)
        quality = selected.get("validation", {}) if selected else {}
        if not quality.get("prediction_path"):
            raise FileNotFoundError(f"Generate full val predictions with --split val first: {folder}")
        validation_path = Path(quality["prediction_path"])
        if not validation_path.exists():
            validation_path = folder / validation_path.name
        metadata = {
            **quality,
            "limited": False,
            "split": "val",
            "checkpoint_epoch": run["best_epoch"],
            "checkpoint": str(folder / "best.pt"),
        }
    if metadata.get("limited") or metadata.get("split") != "val":
        raise ValueError(f"Source validation predictions must be full official val: {folder}")
    if metadata.get("checkpoint_epoch") != run.get("best_epoch"):
        raise ValueError(f"Validation checkpoint epoch is not the source best epoch: {folder}")
    if not inference.get("submission_ready"):
        raise ValueError(f"Source test inference is not submission-ready: {folder}")
    validation_checkpoint = _resolved(metadata["checkpoint"])
    test_checkpoint = _resolved(inference["checkpoint"])
    if validation_checkpoint != test_checkpoint:
        raise ValueError(f"Validation/test checkpoints differ: {folder}")
    validation = _validation_frame(validation_path)
    if len(validation) != metadata.get("candidates") or validation["feed_id"].n_unique() != metadata.get("feeds"):
        raise ValueError(f"Validation file coverage differs from its completed evaluation: {folder}")
    submission = folder / "submission.parquet"
    validate_predictions(submission, Path(data) / "sample_submission.parquet")
    return {
        "folder": str(folder),
        "checkpoint": validation_checkpoint,
        "best_epoch": run["best_epoch"],
        "validation_path": str(validation_path.resolve()),
        "validation": validation,
        "submission": pl.read_parquet(submission).sort(["feed_id", "item_id"]),
    }


def _atomic_parquet(frame, destination):
    destination = Path(destination)
    temporary = Path(str(destination) + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _history_max_correct(frame, data, split, weight):
    """Apply an unlabeled per-feed history/item cosine-max correction in sorted key order."""
    if weight == 0:
        return frame
    keys = ["feed_id", "item_id"]
    if frame.select(keys).n_unique() != len(frame):
        raise ValueError("History-max correction requires unique feed/item keys")
    feed_ids = frame["feed_id"].to_numpy()
    item_ids = frame["item_id"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, feed_ids[1:] != feed_ids[:-1], True])
    offsets = {int(feed_ids[start]): (start, end) for start, end in zip(boundaries[:-1], boundaries[1:])}
    history_max = np.empty(len(frame), dtype=np.float32)
    seen = set()
    dataset = FeedDataset([Path(data) / split])
    with threadpool_limits(limits=1):
        for row in dataset:
            feed_id = int(row["feed_id"])
            if feed_id not in offsets or feed_id in seen:
                raise ValueError(f"History-max feed mismatch or duplicate feed: {feed_id}")
            start, end = offsets[feed_id]
            order = np.argsort(row["item_id"], kind="stable")
            if not np.array_equal(row["item_id"][order], item_ids[start:end]):
                raise ValueError(f"History-max item keys do not align for feed {feed_id}")
            candidates = np.asarray(row["item"], dtype=np.float32)
            history = np.asarray(row["history_item"], dtype=np.float32)
            candidate_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
            candidate_unit = candidates / np.maximum(candidate_norm, 1e-6)
            history_norm = np.linalg.norm(history, axis=1)
            history = history[history_norm > 0]
            if len(history):
                history = history / np.maximum(np.linalg.norm(history, axis=1, keepdims=True), 1e-6)
                local_max = (candidate_unit @ history.T).max(axis=1).astype(np.float32, copy=False)
            else:
                local_max = np.zeros(len(candidates), dtype=np.float32)
            history_max[start:end] = local_max[order]
            seen.add(feed_id)
    if len(seen) != len(offsets):
        missing = len(offsets) - len(seen)
        raise ValueError(f"History-max did not cover all prediction feeds: missing={missing}")
    scores = frame["score"].to_numpy().astype(np.float32, copy=True)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        base, history = scores[start:end], history_max[start:end]
        base_scale = max(float(base.std()), 1e-6)
        history_scale = max(float(history.std()), 1e-6)
        scores[start:end] = base + weight * base_scale * (history - history.mean()) / history_scale
    if not np.isfinite(scores).all():
        raise ValueError("History-max correction produced nonfinite scores")
    return frame.with_columns(pl.Series("score", scores))


def ensemble(runs, weights=None, output=None, data="data", history_max_weight=0.0):
    """Fuse already-complete run predictions with a weighted average of main-tower logits."""
    if not runs or (len(runs) == 1 and history_max_weight == 0):
        raise ValueError("Use at least two source runs, or one source with history-max correction")
    if not np.isfinite(history_max_weight) or history_max_weight < 0:
        raise ValueError("History-max weight must be finite and nonnegative")
    weights = np.ones(len(runs), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    if len(weights) != len(runs) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(
            "Ensemble weights must be finite, nonnegative, match source count, and sum to a positive value"
        )
    weights /= weights.sum(dtype=np.float64)
    output = Path(output) if output else new_run("ensemble")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use an empty ensemble output directory: {output}")
    sources = [_source_run(folder, data) for folder in runs]
    keys = ["feed_id", "item_id"]
    reference = sources[0]["validation"]
    reference_keys, reference_target = reference.select(keys), reference["target"]
    official = (
        pl.scan_parquet(str(Path(data) / "val" / "impressions" / "*.parquet"))
        .select("feed_id", "item_id", "target")
        .explode("item_id", "target")
        .collect()
        .sort(keys)
    )
    if not reference_keys.equals(official.select(keys)) or not reference_target.equals(
        official["target"].cast(pl.Float32)
    ):
        raise ValueError("Validation predictions do not exactly cover official val keys and labels")
    for source in sources[1:]:
        candidate = source["validation"]
        if not reference_keys.equals(candidate.select(keys)) or not reference_target.equals(candidate["target"]):
            raise ValueError("Validation source keys or labels do not align exactly")
    validation_scores = sum(
        weight * source["validation"]["score"].to_numpy() for weight, source in zip(weights, sources)
    )
    validation = reference.select([*keys, "target"]).with_columns(
        pl.Series("score", validation_scores.astype(np.float32, copy=False))
    )
    validation = _history_max_correct(validation, data, "val", history_max_weight)
    metric = RankingMetrics()
    counts = validation.group_by("feed_id", maintain_order=True).len()["len"].to_numpy()
    metric.update(validation["score"].to_numpy(), validation["target"].to_numpy(), counts)
    metrics = metric.compute()
    test_reference = sources[0]["submission"]
    test_keys = test_reference.select(keys)
    for source in sources[1:]:
        if not test_keys.equals(source["submission"].select(keys)):
            raise ValueError("Test submission keys do not align exactly")
    test_scores = sum(weight * source["submission"]["score"].to_numpy() for weight, source in zip(weights, sources))
    submission = test_reference.select(keys).with_columns(
        pl.Series("score", test_scores.astype(np.float32, copy=False))
    )
    submission = _history_max_correct(submission, data, "test", history_max_weight)
    submission_path = output / "submission.parquet"
    validation_path = output / "validation_predictions.parquet"
    output.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(validation, validation_path)
    try:
        _atomic_parquet(submission, submission_path)
        submission_info = validate_predictions(submission_path, Path(data) / "sample_submission.parquet")
        result = {
            "sources": [
                {key: source[key] for key in ("folder", "checkpoint", "best_epoch", "validation_path")}
                for source in sources
            ],
            "weights": weights.tolist(),
            "history_max_weight": history_max_weight,
            "history_max_formula": (
                "base_logits + weight * max(std(base_logits), 1e-6) * "
                "(history_max - mean(history_max)) / max(std(history_max), 1e-6), per feed"
            ),
            "metrics": metrics,
            "validation_predictions": str(validation_path),
            "submission": {**submission_info, "path": str(submission_path)},
            "selection_boundary": "Weights were selected on this same official validation split; not independent validation or online evidence.",
        }
        write_json(output / "ensemble.json", result)
        report = [
            "# 离线融合报告",
            "",
            "验证指标和权重均来自同一份官方 val，属于同验证集选权重，不是独立验证或线上成绩。",
            "history-max 权重同样是该 val 上的探索结果，不是线上或独立验证证据。",
            "",
            "## 来源与权重",
            "",
            "| Run | Checkpoint | Weight |",
            "| --- | --- | ---: |",
            *[
                f"| {source['folder']} | {source['checkpoint']} | {weight:.8f} |"
                for source, weight in zip(result["sources"], weights)
            ],
            "",
            "## 官方 val 指标",
            "",
            *[f"- {name}: {value:.10f}" for name, value in metrics.items()],
            f"- history-max weight: {history_max_weight:.8f}",
            "- history-max 公式：`base_logits + weight × max(std(base_logits), 1e-6) × "
            "(history_max - mean(history_max)) / max(std(history_max), 1e-6)`（逐 feed）。",
            "",
            f"- 验证逐候选预测：[validation_predictions.parquet]({validation_path.name})",
            f"- 本地提交文件：[submission.parquet]({submission_path.name})",
            "",
        ]
        temporary = output / "report.md.tmp"
        temporary.write_text("\n".join(report))
        temporary.replace(output / "report.md")
        return result
    except BaseException:
        validation_path.unlink(missing_ok=True)
        submission_path.unlink(missing_ok=True)
        (output / "ensemble.json").unlink(missing_ok=True)
        (output / "report.md").unlink(missing_ok=True)
        (output / "report.md.tmp").unlink(missing_ok=True)
        raise


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--ensemble-runs", type=Path, nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--history-max-weight", type=float, default=0.0)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32, help="Feeds per batch")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-feeds", type=int, default=0, help="Limited diagnostics; never marked submission-ready")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--fast-execution", action="store_true")
    args = parser.parse_args()
    if args.log_every <= 0 or args.threads <= 0:
        parser.error("log-every and threads must be positive")
    if args.weights is not None and not args.ensemble_runs:
        parser.error("--weights requires --ensemble-runs")
    if args.history_max_weight and not args.ensemble_runs:
        parser.error("--history-max-weight requires --ensemble-runs")
    torch.set_num_threads(args.threads)
    if args.ensemble_runs:
        if args.split != "test" or args.max_feeds:
            parser.error("Ensemble only supports complete test submissions")
        try:
            ensemble(args.ensemble_runs, args.weights, args.output, args.data, args.history_max_weight)
        except ValueError as error:
            parser.error(str(error))
        return
    inference = predict if args.split == "test" else predict_validation
    inference(
        args.checkpoint,
        args.data,
        args.output,
        args.device,
        args.batch_size,
        args.workers,
        args.max_feeds,
        args.log_every,
        not args.no_tensorboard if args.split == "test" else False,
        fast_execution=args.fast_execution,
    )


if __name__ == "__main__":
    main()
