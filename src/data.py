"""Stream complete feeds; keep candidate order and split-local embedding indices."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

NUMERIC = ("f1", "f2", "f3", "f4", "f5")
COLUMNS = (
    "feed_id",
    "timestamp",
    "item_id",
    "category_id",
    *NUMERIC,
    "user_embedding",
    "item_id__embedding_idx",
    "history__item_id__embedding_idx",
    "history__item_id",
    "history__category_id",
    "history__event_type",
    "history__event_source",
)


def fit_numeric_bins(data, bins, seed=42, sample_size=1_000_000):
    """Fit numerical quantiles from train only, sampling across all training shards."""
    paths = sorted((Path(data) / "train/impressions").glob("part-*.parquet"))
    if bins < 2 or not paths:
        raise ValueError("Numeric bins need at least two buckets and nonempty training shards")
    quota = math.ceil(sample_size / len(paths))

    def sample_shard(index_path):
        index, path = index_path
        table = pq.read_table(path, columns=list(NUMERIC), use_threads=False).combine_chunks()
        values = np.column_stack([table[name].chunk(0).flatten().to_numpy() for name in NUMERIC]).astype(np.float32)
        if len(values) > quota:
            indices = np.random.default_rng(seed + index).choice(len(values), quota, replace=False)
            values = values[indices]
        return values

    with ThreadPoolExecutor(max_workers=4) as pool:
        sampled = np.concatenate(list(pool.map(sample_shard, enumerate(paths))))
    if not np.isfinite(sampled).all():
        raise ValueError("Nonfinite numerical values in training bucket sample")
    edges = np.quantile(sampled, np.arange(1, bins) / bins, axis=0).T.astype(np.float32)
    metadata = {
        "source": "train_only",
        "method": "per_shard_uniform_sample_quantiles",
        "bins": bins,
        "sample_candidates": len(sampled),
        "train_shards": len(paths),
        "seed": seed,
    }
    return edges, metadata


def item_vectors(table, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if ((indices < -1) | (indices >= len(table))).any():
        raise ValueError("Embedding index outside the current split's table")
    present = indices >= 0
    if present.all():
        return table[indices].astype(np.float32), present
    vectors = np.zeros((len(indices), 128), dtype=np.float32)
    vectors[present] = table[indices[present]].astype(np.float32)
    return vectors, present


def encode_arrow_batch(batch, embeddings):
    """Convert nested Arrow columns in bulk, retaining zero-copy row slices where possible."""

    def flatten(name, dtype):
        array = batch.column(batch.schema.get_field_index(name))
        offsets = array.offsets.to_numpy().astype(np.int64)
        values = array.values.slice(int(offsets[0]), int(offsets[-1] - offsets[0]))
        return values.to_numpy(zero_copy_only=False).astype(dtype, copy=False), offsets - offsets[0]

    ids, candidate_offsets = flatten("item_id", np.int32)
    category, offsets = flatten("category_id", np.int64)
    if not np.array_equal(offsets, candidate_offsets) or ((category < 0) | (category > 53)).any():
        raise ValueError("Invalid candidate categories")
    numerical = []
    for name in NUMERIC:
        values, offsets = flatten(name, np.float32)
        if not np.array_equal(offsets, candidate_offsets):
            raise ValueError("Misaligned numerical lists")
        numerical.append(values)
    numerical = np.column_stack(numerical)
    if not np.isfinite(numerical).all():
        raise ValueError("Nonfinite numerical features")
    indices, offsets = flatten("item_id__embedding_idx", np.int64)
    if not np.array_equal(offsets, candidate_offsets):
        raise ValueError("Misaligned item embedding indices")
    items, present = item_vectors(embeddings, indices)
    indices, history_offsets = flatten("history__item_id__embedding_idx", np.int64)
    history_items, history_present = item_vectors(embeddings, indices)
    history_fields = {}
    history_ids, offsets = flatten("history__item_id", np.int64)
    if not np.array_equal(offsets, history_offsets) or (history_ids < 0).any():
        raise ValueError("Invalid or misaligned historical item IDs")
    history_fields["history_item_id"] = history_ids
    for name, column, maximum in (
        ("history_category", "category_id", 53),
        ("history_event_type", "event_type", 10),
        ("history_event_source", "event_source", 30),
    ):
        values, offsets = flatten("history__" + column, np.int64)
        if not np.array_equal(offsets, history_offsets) or ((values < 0) | (values > maximum)).any():
            raise ValueError("Invalid history fields")
        history_fields[name] = values
    users, offsets = flatten("user_embedding", np.float32)
    if not np.all(np.diff(offsets) == 128):
        raise ValueError("Invalid user embedding dimensions")
    users = users.reshape(len(batch), 128)
    counts = np.diff(history_offsets)
    valid_prefix = np.r_[0, np.cumsum(history_present)]
    valid_counts = valid_prefix[history_offsets[1:]] - valid_prefix[history_offsets[:-1]]
    stats = np.column_stack((counts / 256, valid_counts / np.maximum(counts, 1))).astype(np.float32)
    labels, positions = None, None
    for name in ("target", "pos"):
        if name in batch.schema.names:
            values, offsets = flatten(name, np.float32 if name == "target" else np.int64)
            if not np.array_equal(offsets, candidate_offsets):
                raise ValueError("Misaligned labels or original positions")
            if name == "target":
                labels = values
            else:
                if (values < 0).any():
                    raise ValueError("Invalid original positions")
                positions = values
    feed_ids = batch.column(batch.schema.get_field_index("feed_id")).to_numpy()
    timestamps = batch.column(batch.schema.get_field_index("timestamp")).to_numpy()
    missing = (~present).astype(np.float32)[:, None]
    history_missing = (~history_present).astype(np.float32)[:, None]
    rows = []
    for index in range(len(batch)):
        candidate = slice(candidate_offsets[index], candidate_offsets[index + 1])
        history = slice(history_offsets[index], history_offsets[index + 1])
        rows.append(
            {
                "feed_id": int(feed_ids[index]),
                "timestamp": int(timestamps[index]),
                "item_id": ids[candidate],
                "category": category[candidate],
                "numerical": numerical[candidate],
                "item": items[candidate],
                "item_missing": missing[candidate],
                "user": users[index],
                "history_item": history_items[history],
                "history_item_missing": history_missing[history],
                **{name: values[history] for name, values in history_fields.items()},
                "history_stats": stats[index],
                "target": labels[candidate] if labels is not None else None,
                "position": positions[candidate] if positions is not None else None,
            }
        )
    return rows


def encode_feed(row, embeddings):
    items, present = item_vectors(embeddings, row["item_id__embedding_idx"])
    history, valid_history = item_vectors(embeddings, row["history__item_id__embedding_idx"])
    history_fields = {
        name: np.asarray(row["history__" + column], dtype=np.int64)
        for name, column in (
            ("history_category", "category_id"),
            ("history_event_type", "event_type"),
            ("history_event_source", "event_source"),
        )
    }
    history_fields["history_item_id"] = np.asarray(row["history__item_id"], dtype=np.int64)
    if len(history_fields["history_item_id"]) != len(history) or (history_fields["history_item_id"] < 0).any():
        raise ValueError("Invalid or misaligned historical item IDs")
    for name, maximum in (("history_category", 53), ("history_event_type", 10), ("history_event_source", 30)):
        values = history_fields[name]
        if len(values) != len(history) or ((values < 0) | (values > maximum)).any():
            raise ValueError(f"Invalid or misaligned {name}")
    count = len(row["item_id"])
    numerical = np.column_stack([row[name] for name in NUMERIC]).astype(np.float32)
    category = np.asarray(row["category_id"], dtype=np.int64)
    if numerical.shape != (count, 5) or len(category) != count or len(items) != count:
        raise ValueError("Misaligned candidate lists")
    if ((category < 0) | (category > 53)).any() or not np.isfinite(numerical).all():
        raise ValueError("Unexpected categories or nonfinite numerical features")
    position = np.asarray(row["pos"], dtype=np.int64) if "pos" in row else None
    if position is not None and (len(position) != count or (position < 0).any()):
        raise ValueError("Invalid original candidate positions")
    return {
        "feed_id": row["feed_id"],
        "timestamp": row["timestamp"],
        "item_id": np.asarray(row["item_id"], dtype=np.int32),
        "numerical": numerical,
        "category": category,
        "item": items,
        "item_missing": (~present).astype(np.float32)[:, None],
        "user": np.asarray(row["user_embedding"], dtype=np.float32),
        "history_item": history,
        "history_item_missing": (~valid_history).astype(np.float32)[:, None],
        **history_fields,
        "history_stats": np.asarray(
            [len(valid_history) / 256, valid_history.sum() / max(len(valid_history), 1)], dtype=np.float32
        ),
        "target": np.asarray(row["target"], dtype=np.float32) if "target" in row else None,
        "position": position,
    }


class FeedDataset(IterableDataset):
    """Globally limit feeds before worker sharding; never load the expanded train into RAM."""

    def __init__(self, roots, max_feeds=0, shuffle=False, seed=42, read_batch_size=256):
        super().__init__()
        if max_feeds < 0 or read_batch_size <= 0:
            raise ValueError("Invalid feed limit or read batch size")
        self.shuffle, self.seed, self.epoch, self.read_batch_size = shuffle, seed, 0, read_batch_size
        self.shards, self.feed_count = [], 0
        remaining = max_feeds or float("inf")
        labeled = set()
        for root in map(Path, roots):
            files = sorted((root / "impressions").glob("part-*.parquet"))
            if not files:
                raise FileNotFoundError(f"No impression parts in {root}")
            for path in files:
                reader = pq.ParquetFile(path)
                take = int(min(remaining, reader.metadata.num_rows))
                if take:
                    labeled.add("target" in reader.schema_arrow.names)
                    self.shards.append((root, path, take, "pos" in reader.schema_arrow.names))
                    self.feed_count += take
                    remaining -= take
                if not remaining:
                    break
            if not remaining:
                break
        if len(labeled) != 1 or not self.feed_count:
            raise ValueError("Dataset must be nonempty with consistently available labels")
        self.labeled = labeled.pop()

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        worker_id = (worker_id + getattr(self, "resume_worker_offset", 0)) % workers
        skip = getattr(self, "resume_rows", [0] * workers)[worker_id]
        order = np.arange(len(self.shards))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + worker_id)
        tables = {}
        for shard in order[worker_id::workers]:
            root, path, remaining, has_position = self.shards[shard]
            columns = [*COLUMNS, *(["target"] if self.labeled else []), *(["pos"] if has_position else [])]
            if skip >= remaining:
                columns = ["feed_id"]
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=self.read_batch_size, columns=columns, use_threads=False
            ):
                take = min(remaining, len(batch))
                if skip >= take:
                    if self.shuffle:
                        # Match the original Python-list shuffle RNG consumption without fetching item vectors.
                        rng.shuffle(list(range(take)))
                    skip -= take
                    remaining -= take
                    if not remaining:
                        break
                    continue
                if root not in tables:
                    folder = root / "item_embeddings"
                    meta = json.loads((folder / "data.meta").read_text())
                    tables[root] = np.memmap(
                        folder / "data.memmap",
                        mode="r",
                        dtype=meta["dtype"],
                        shape=tuple(meta["shape"]),
                        order=meta["order"],
                    )
                rows = encode_arrow_batch(batch.slice(0, take), tables[root])
                if self.shuffle:
                    rng.shuffle(rows)
                for row in rows[skip:]:
                    yield row
                skip = 0
                remaining -= take
                if remaining == 0:
                    break


def collate_feeds(rows):
    counts = np.asarray([len(row["item_id"]) for row in rows], dtype=np.int64)
    result = {
        name: torch.from_numpy(np.concatenate([r[name] for r in rows]))
        for name in ("numerical", "category", "item", "item_missing")
    }
    for name in ("user", "history_stats"):
        result[name] = torch.from_numpy(np.stack([r[name] for r in rows]))
    seconds = np.asarray([r["timestamp"] for r in rows], dtype=np.int64)
    # Unix time in UTC; weekday 0=Monday. Workday means Monday-Friday, without a holiday calendar.
    weekday = (seconds // 86400 + 3) % 7
    result["time_bins"] = torch.from_numpy(
        np.column_stack(
            (
                seconds // 3600 % 24,
                seconds // 60 % 60,
                weekday,
                (weekday < 5).astype(np.int64),
                weekday * 24 + seconds // 3600 % 24,
            )
        )
    )
    lengths = [len(row["history_item"]) for row in rows]
    result["history_lengths"] = np.asarray(lengths, dtype=np.int32)
    maximum = max(lengths)
    history_mask = np.zeros((len(rows), maximum), dtype=bool)
    for name in (
        "history_item",
        "history_item_missing",
        "history_item_id",
        "history_category",
        "history_event_type",
        "history_event_source",
    ):
        example = rows[0][name]
        padded = np.zeros((len(rows), maximum, *example.shape[1:]), dtype=example.dtype)
        for index, (row, length) in enumerate(zip(rows, lengths)):
            padded[index, :length] = row[name]
            history_mask[index, :length] = True
        result[name] = torch.from_numpy(padded)
    result["history_mask"] = torch.from_numpy(history_mask)
    result["candidate_feed"] = torch.from_numpy(np.repeat(np.arange(len(rows)), counts))
    result["target"] = (
        torch.from_numpy(np.concatenate([r["target"] for r in rows])) if rows[0]["target"] is not None else None
    )
    result["counts"] = counts
    result["position_available"] = np.asarray([row["position"] is not None for row in rows], dtype=bool)
    result["position"] = torch.from_numpy(
        np.concatenate(
            [
                row["position"] if row["position"] is not None else np.full(count, -1, dtype=np.int64)
                for row, count in zip(rows, counts)
            ]
        )
    )
    result["feed_id"] = np.repeat(np.asarray([r["feed_id"] for r in rows], dtype=np.int32), counts)
    result["item_id"] = np.concatenate([r["item_id"] for r in rows])
    result["candidate_item_id"] = torch.from_numpy(result["item_id"].astype(np.int64))
    return result


def make_loader(dataset, batch_size, workers, device, resume_batches=0, resume_cursor=None, cursor_batch_size=None):
    if batch_size <= 0 or workers < 0:
        raise ValueError("Invalid feed batch size or worker count")
    generator = torch.Generator().manual_seed(dataset.seed + dataset.epoch)
    count = max(workers, 1)
    order = np.arange(len(dataset.shards))
    if dataset.shuffle:
        np.random.default_rng(dataset.seed + dataset.epoch).shuffle(order)
    feed_counts = [sum(dataset.shards[i][2] for i in order[w::count]) for w in range(count)]
    if resume_cursor is None:
        cursor_batch_size = cursor_batch_size or batch_size
        batch_counts = [math.ceil(n / cursor_batch_size) for n in feed_counts]
        if not 0 <= resume_batches <= sum(batch_counts):
            raise ValueError("Resume cursor exceeds available worker batches")
        used, next_worker = [0] * count, 0
        for _ in range(resume_batches):
            while used[next_worker] == batch_counts[next_worker]:
                next_worker = (next_worker + 1) % count
            used[next_worker] += 1
            next_worker = (next_worker + 1) % count
        dataset.resume_rows = [min(n * cursor_batch_size, total) for n, total in zip(used, feed_counts)]
    else:
        dataset.resume_rows, next_worker = list(resume_cursor["rows"]), resume_cursor["next_worker"]
        if len(dataset.resume_rows) != count or any(
            not 0 <= n <= total for n, total in zip(dataset.resume_rows, feed_counts)
        ):
            raise ValueError("Invalid worker row cursor")
    dataset.worker_total_rows = feed_counts
    dataset.resume_worker_offset = next_worker
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate_feeds,
        pin_memory=torch.device(device).type == "cuda",
        generator=generator,
        **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}),
    )


def to_device(batch, device, *, features_only=False):
    cpu_fields = {"target", "position", "position_available", "pairwise_indices", "pairwise_available"}
    return {
        name: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor) and not (features_only and name in cpu_fields)
            else value
        )
        for name, value in batch.items()
    }


def split_feed_batch(batch):
    """Split on a feed boundary, preserving each feed's complete ranking list and history."""
    middle = len(batch["counts"]) // 2
    if not middle:
        raise ValueError("Parallel batches need at least two feeds")
    candidate_middle = int(batch["counts"][:middle].sum())
    feed_fields = {
        "user",
        "time_bins",
        "history_stats",
        "history_item",
        "history_item_missing",
        "history_item_id",
        "pairwise_indices",
        "pairwise_available",
        "history_category",
        "history_event_type",
        "history_event_source",
        "history_mask",
        "history_lengths",
        "counts",
        "position_available",
    }
    halves = []
    for start, end, first, last in (
        (0, middle, 0, candidate_middle),
        (middle, len(batch["counts"]), candidate_middle, None),
    ):
        half = {
            name: None if value is None else value[start:end] if name in feed_fields else value[first:last]
            for name, value in batch.items()
        }
        half["candidate_feed"] = half["candidate_feed"] - start
        halves.append(half)
    return halves
