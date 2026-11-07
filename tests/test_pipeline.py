"""CPU integration tests on synthetic data; these are not competition experiments."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from catboost.utils import eval_metric
from src import train
from src.data import (
    FeedDataset,
    collate_feeds,
    fit_numeric_bins,
    item_vectors,
    make_loader,
    encode_arrow_batch,
    encode_feed,
)
from src.infer import predict, validate_predictions
from src.metrics import RankingMetrics
from src.models import FeatureEncoder, ModelConfig, PerTokenFFN, build_model, token_mix
from src.onetrans import OneTransBlock, QREmbedding
from src.losses import listmle, pairwise_loss, sample_pairwise
from src.train import epoch_steps, learning_rate
from src.train import ReplicaTrainer
from src.runtime import LOCAL_TIME, RunMonitor


def fixture(root):
    for number, (split, count) in enumerate((("train", 6), ("val", 3), ("test", 4))):
        folder = root / split
        (folder / "impressions").mkdir(parents=True)
        (folder / "item_embeddings").mkdir()
        embeddings = np.random.default_rng(number).integers(-127, 128, (8, 128), dtype=np.int8)
        embeddings[0] = 127
        embeddings.tofile(folder / "item_embeddings/data.memmap")
        (folder / "item_embeddings/data.meta").write_text(
            json.dumps({"dtype": "int8", "shape": [8, 128], "order": "C"})
        )
        rows = []
        for i in range(count):
            size = 5 + i % 3
            history = [] if i == 0 else [-1, -1] if i == 1 else [0, -1, 2]
            row = {
                "feed_id": number * 100 + i + 1,
                "timestamp": 1788739200 + i * 3661,
                "item_id": [100 + j for j in range(size)],
                "category_id": [0, 1, 2, 3, 53, 1, 2][:size],
                "user_embedding": [127] * 128,
                "item_id__embedding_idx": [-1, *range(size - 1)],
                "history__item_id__embedding_idx": history,
                "history__item_id": [100, 101, 102][: len(history)],
                "history__category_id": [1, 53, 0][: len(history)],
                "history__event_type": [1, 10, 0][: len(history)],
                "history__event_source": [1, 30, 0][: len(history)],
                **{f"f{j}": [float(k + i) / 10 for k in range(size)] for j in range(1, 6)},
            }
            if split != "test":
                row["target"] = [int(j == i % size) for j in range(size)]
            if split == "train" and i == 0:
                row["item_id"][1] = row["item_id"][0]  # Preserve duplicate training candidates.
            if split == "train":
                row["pos"] = list(reversed(range(size)))
            rows.append(row)
        for index, group in enumerate((rows[:2], rows[2:])):
            if group:
                pl.DataFrame(group).write_parquet(folder / f"impressions/part-{index:05d}.parquet")
        if split == "test":
            sample = pl.DataFrame(rows).select("feed_id", "item_id").explode("item_id")
            sample.with_columns(
                pl.col("feed_id").cast(pl.Int32),
                pl.col("item_id").cast(pl.Int32),
                pl.lit(0.0, dtype=pl.Float32).alias("score"),
            ).write_parquet(root / "sample_submission.parquet")


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        parent = Path("output/mlp")
        parent.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(
            dir=parent, prefix=datetime.now(LOCAL_TIME).strftime("%Y%m%d_%H%M%S_test_")
        )
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        fixture(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_training(self, output, *extra):
        arguments = [
            "--data",
            str(self.data),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--threads",
            "1",
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--hidden",
            "8",
            "--width",
            "8",
            "--depth",
            "1",
            "--expansion",
            "2",
            "--log-every",
            "1",
            "--checkpoint-every",
            "1",
            "--no-tensorboard",
            *extra,
        ]
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            return train.main(arguments)

    def test_streaming_workers_and_missing_embeddings(self):
        dataset = FeedDataset([self.data / "train"], max_feeds=4, read_batch_size=1)
        for workers in (0, 2):
            batches = list(make_loader(dataset, 2, workers, "cpu"))
            self.assertEqual(sum(len(b["counts"]) for b in batches), 4)
            self.assertEqual(sorted(set(np.concatenate([b["feed_id"] for b in batches]))), [1, 2, 3, 4])
        rows = list(dataset)
        self.assertEqual(len(rows[0]["item_id"]), 5)
        self.assertEqual(len(np.unique(rows[0]["item_id"])), 4)
        self.assertEqual(rows[0]["history_item"].shape, (0, 128))
        np.testing.assert_array_equal(rows[1]["history_item"], np.zeros((2, 128)))
        np.testing.assert_array_equal(rows[0]["item"][0], np.zeros(128))
        np.testing.assert_array_equal(rows[0]["item"][1], np.full(128, 127.0))
        with self.assertRaises(ValueError):
            item_vectors(np.zeros((1, 128), dtype=np.int8), [-2])

    def test_arrow_batch_matches_row_encoder_with_sliced_offsets(self):
        import pyarrow.parquet as pq

        for split in ("train", "val", "test"):
            root = self.data / split
            embeddings = np.memmap(root / "item_embeddings/data.memmap", mode="r", dtype="int8", shape=(8, 128))
            for path in (root / "impressions").glob("*.parquet"):
                for batch in pq.ParquetFile(path).iter_batches():
                    for sliced in (batch, batch.slice(1, 1)):
                        for actual, row in zip(encode_arrow_batch(sliced, embeddings), sliced.to_pylist()):
                            expected = encode_feed(row, embeddings)
                            self.assertEqual(set(actual), set(expected))
                            for name in actual:
                                if expected[name] is None:
                                    self.assertIsNone(actual[name])
                                else:
                                    np.testing.assert_array_equal(actual[name], expected[name])

    def test_tokenmixer_math_and_gradients(self):
        tokens = torch.arange(2 * 8 * 16).reshape(2, 8, 16).float()
        expected = torch.stack(
            [torch.cat([tokens[:, t, h * 2 : (h + 1) * 2] for t in range(8)], dim=-1) for h in range(8)], dim=1
        )
        torch.testing.assert_close(token_mix(tokens), expected, rtol=0, atol=0)
        ffn = PerTokenFFN(8, 16, 2, 0.0)
        expected = torch.stack(
            [torch.nn.functional.gelu(tokens[:, t] @ ffn.w1[t] + ffn.b1[t]) @ ffn.w2[t] + ffn.b2[t] for t in range(8)],
            dim=1,
        )
        torch.testing.assert_close(ffn(tokens), expected)
        gated = PerTokenFFN(8, 16, 2, 0.0, "swiglu")
        expected = torch.stack(
            [
                torch.nn.functional.silu(tokens[:, t] @ gated.w_gate[t] + gated.b_gate[t])
                * (tokens[:, t] @ gated.w1[t] + gated.b1[t])
                for t in range(8)
            ],
            dim=1,
        )
        expected = torch.stack([expected[:, t] @ gated.w2[t] + gated.b2[t] for t in range(8)], dim=1)
        torch.testing.assert_close(gated(tokens), expected)
        gated(tokens).square().mean().backward()
        self.assertGreater(gated.w_gate.grad.abs().sum().item(), 0)
        batch = collate_feeds(list(FeedDataset([self.data / "train"], max_feeds=2)))
        for name in ("mlp", "tokenmixer"):
            model = build_model(ModelConfig(name=name, hidden=(8,), width=8, depth=1))
            outputs = model(batch)
            scores = outputs["main"]
            self.assertEqual(tuple(scores.shape), (len(batch["item_id"]),))
            aux, _ = listmle(outputs["aux"], batch["position"], batch["counts"], batch["position_available"])
            (torch.nn.functional.binary_cross_entropy_with_logits(scores, batch["target"]) + 0.1 * aux).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_metrics_match_official_ties_and_cutoffs(self):
        rng = np.random.default_rng(7)
        counts = [5, 19, 45]
        labels = rng.integers(0, 2, sum(counts))
        labels[:5] = 0
        scores = rng.integers(-2, 3, len(labels)).astype(float)
        accumulator = RankingMetrics()
        accumulator.update(scores[:24], labels[:24], counts[:2])
        accumulator.update(scores[24:], labels[24:], counts[2:])
        actual = accumulator.compute()
        for name, metric in (
            ("ndcg", "NDCG:type=Base"),
            ("ndcg_at_10", "NDCG:top=10;type=Base"),
            ("ndcg_at_30", "NDCG:top=30;type=Base"),
        ):
            reference = eval_metric(
                labels.tolist(),
                scores.tolist(),
                metric,
                group_id=np.repeat(np.arange(3), counts).tolist(),
                thread_count=1,
            )[0]
            self.assertAlmostEqual(actual[name], reference, places=12)

    def test_onetrans_causality_shared_history_and_padding(self):
        config = ModelConfig(name="onetrans", hidden=(8,), width=16, depth=2, attention_heads=4, dropout=0)
        model = build_model(config).eval()
        batch = collate_feeds(list(FeedDataset([self.data / "train"], max_feeds=4)))
        base, events, mask = model.features(batch)
        self.assertEqual(tuple(base.shape), (len(batch["item_id"]), 408))
        self.assertEqual(tuple(events.shape), (4, 3, 16))
        original = model.features.encode_events(batch)
        torch.testing.assert_close(events[2], original[2].flip(0))
        self.assertEqual(model.chunk_width, 51)
        self.assertEqual(model.padding, 0)

        block = model.blocks[0]
        sequence = torch.randn(2, 4, 16)
        valid = torch.ones(2, 4, dtype=torch.bool)
        changed = sequence.clone()
        changed[:, 2:] += 20 * torch.randn_like(changed[:, 2:])
        first = block.advance_sequence(sequence, valid)
        second = block.advance_sequence(changed, valid)
        torch.testing.assert_close(first[:, :2], second[:, :2])

        query = torch.randn(len(base), 8, 16)
        feed = batch["candidate_feed"]
        offsets = torch.as_tensor(np.cumsum(np.r_[0, batch["counts"][:-1]]))
        positions = torch.arange(len(base)) - offsets[feed]
        packed = block.read_history(query, events, mask, feed, positions, int(batch["counts"].max()))
        q = block.heads_view(block.query_q(block.query_norm(query)))
        k, v = block.context_kv(block.context_norm(events)).chunk(2, dim=-1)
        valid = mask.clone()
        valid[:, 0] |= ~valid.any(dim=1)
        reference = (
            torch.nn.functional.scaled_dot_product_attention(
                q, block.heads_view(k[feed]), block.heads_view(v[feed]), attn_mask=valid[feed, None, None, :]
            )
            .transpose(1, 2)
            .flatten(2)
        )
        reference = query + block.query_out(reference)
        reference = reference + block.query_ffn(block.query_ffn_norm(reference))
        torch.testing.assert_close(packed, reference, rtol=1e-5, atol=1e-6)

        scores = model(batch)["main"]
        calls = []
        hooks = [b.context_kv.register_forward_hook(lambda *args: calls.append(1)) for b in model.blocks]
        cache = model.prepare_history_cache(batch)
        self.assertTrue(cache.layers[0].value.requires_grad)
        torch.testing.assert_close(model(batch, history_cache=cache)["main"], scores)
        for start in (0, 1):
            indices = torch.arange(start, len(scores), 2)
            chunk = dict(batch)
            for name in ("numerical", "category", "item", "item_missing", "candidate_feed"):
                chunk[name] = batch[name][indices]
            chunk["counts"] = np.bincount(chunk["candidate_feed"].numpy(), minlength=len(batch["counts"]))
            torch.testing.assert_close(model(chunk, history_cache=cache)["main"], scores[indices])
        self.assertEqual(len(calls), len(model.blocks))
        for hook in hooks:
            hook.remove()
        altered = dict(batch)
        altered["history_item"] = batch["history_item"].clone()
        altered["history_item"][~batch["history_mask"]] = 57
        torch.testing.assert_close(model(altered)["main"], scores)
        self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.features.event_type.weight.grad.abs().sum().item(), 0)
        empty = collate_feeds(list(FeedDataset([self.data / "train"], max_feeds=1)))
        self.assertTrue(torch.isfinite(model(empty)["main"]).all())

    def test_train_eval_full_inference_and_monitoring(self):
        for name in ("mlp", "tokenmixer", "onetrans"):
            evaluation = self.root / f"eval_{name}"
            result = self.run_training(evaluation, "--model", name)
            self.assertEqual(result["status"], "complete")
            self.assertTrue((evaluation / "best.pt").exists())
            self.assertTrue(result["inference"]["submission_ready"])
            self.assertEqual(result["inference"]["checkpoint"], str(evaluation / "best.pt"))
            self.assertTrue((evaluation / "submission.parquet").exists())
            report = (evaluation / "report.md").read_text()
            self.assertIn("NDCG@30", report)
            self.assertIn("Logloss", report)
            self.assertIn(f"{result['best_score']:.8f}", report)
            events = [json.loads(s) for s in (evaluation / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(any(e["stage"] == "train" and "gradient_norm" in e for e in events))
            self.assertTrue(any(e["stage"] == "validation" and "roc_auc" in e for e in events))
            full = self.root / f"full_{name}"
            with redirect_stderr(io.StringIO()):
                result = train.main(
                    ["--mode", "full", "--from-eval", str(evaluation), "--output", str(full), "--device", "cpu"]
                )
            self.assertEqual(result["history"][0]["feeds"], 9)
            self.assertEqual(result["history"][0]["aux_feeds"], 6)
            self.assertEqual(result["model_config"]["name"], name)
            self.assertTrue(result["inference"]["submission_ready"])
            with redirect_stderr(io.StringIO()):
                repeated = predict(
                    full / "last.pt", self.data, self.root / f"predict_{name}", "cpu", 1, 0, tensorboard=False
                )
            keys = ["feed_id", "item_id"]
            left = pl.read_parquet(full / "submission.parquet").sort(keys)
            right = pl.read_parquet(repeated["path"]).sort(keys)
            self.assertTrue(left.select(keys).equals(right.select(keys)))
            np.testing.assert_allclose(left["score"], right["score"], rtol=1e-5, atol=1e-6)
            bad = self.root / f"bad_{name}.parquet"
            left.head(1).write_parquet(bad)
            with self.assertRaises(ValueError):
                validate_predictions(bad, self.data / "sample_submission.parquet")

    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        complete, interrupted = self.root / "uninterrupted", self.root / "interrupted"
        self.run_training(complete, "--epochs", "2")
        save = train.save_checkpoint

        def stop_after_checkpoint(path, state):
            save(path, state)
            if state["global_step"] == 1:
                raise RuntimeError("simulated interruption")

        with patch.object(train, "save_checkpoint", side_effect=stop_after_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_training(interrupted, "--epochs", "2")
        with redirect_stderr(io.StringIO()):
            train.main(["--resume", str(interrupted / "last.pt"), "--device", "cpu"])
        left = torch.load(complete / "last.pt", weights_only=True)
        right = torch.load(interrupted / "last.pt", weights_only=True)
        self.assertEqual(left["global_step"], right["global_step"])
        for name in left["model"]:
            torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)

    def test_warmup_cosine_and_worker_tail_step_counts(self):
        dataset = FeedDataset([self.data / "train"])
        self.assertEqual(epoch_steps(dataset, 3, 0, 0), 2)
        self.assertEqual(epoch_steps(dataset, 3, 2, 0), 3)
        plan = {"kind": "warmup_cosine", "base_lr": 0.001, "total_steps": 20, "warmup_steps": 2, "min_lr_ratio": 0.05}
        rates = [learning_rate(plan, step) for step in range(20)]
        self.assertAlmostEqual(rates[0], 0.0005)
        self.assertAlmostEqual(rates[1], 0.001)
        self.assertAlmostEqual(rates[2], 0.001)
        self.assertAlmostEqual(rates[-1], 0.00005)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))
        self.assertEqual(learning_rate(plan, 20), rates[-1])

    def test_replica_gradient_weights_match_full_feed_batch(self):
        config = ModelConfig(
            name="onetrans",
            hidden=(8,),
            width=16,
            depth=1,
            attention_heads=4,
            dropout=0,
            user_embedding_dropout=0,
            item_embedding_dropout=0,
            pairwise_tower=True,
        )
        reference, master = build_model(config), build_model(config)
        master.load_state_dict(reference.state_dict())
        batch = collate_feeds(list(FeedDataset([self.data / "train"])))
        batch["position_available"][-2:] = False
        sample_pairwise(batch, pairs_per_feed=16)
        outputs = reference(batch)
        main = torch.nn.functional.binary_cross_entropy_with_logits(outputs["main"], batch["target"])
        aux, count = listmle(outputs["aux"], batch["position"], batch["counts"], batch["position_available"])
        pair, pair_feeds = pairwise_loss(
            outputs["pairwise"], batch["pairwise_indices"], batch["counts"], batch["pairwise_available"]
        )
        expected = main + 0.1 * aux + 0.2 * pair
        expected.backward()
        parallel = ReplicaTrainer(
            master, config, {"amp": "none", "aux_weight": 0.1, "pairwise_weight": 0.2, "compile": False}, "cpu"
        )
        try:
            _, actual_main, actual_aux, actual_count, actual = parallel.backward(batch)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_main, main)
            torch.testing.assert_close(actual_aux, aux)
            self.assertEqual(actual_count, count)
            self.assertEqual(parallel.pairwise_feeds, pair_feeds)
            torch.testing.assert_close(parallel.pairwise_loss, pair)
            for first, second in zip(reference.parameters(), master.parameters()):
                if first.grad is None:
                    self.assertIsNone(second.grad)
                else:
                    torch.testing.assert_close(first.grad, second.grad, rtol=1e-4, atol=1e-6)
        finally:
            parallel.close()

    def test_unified_features_qr_time_buckets_and_removed_flags(self):
        batch = collate_feeds(list(FeedDataset([self.data / "train"])))
        config = ModelConfig(
            name="onetrans",
            attention_layout="unified_prefix",
            width=16,
            depth=1,
            category_dim=64,
            event_dim=32,
            source_dim=32,
            numeric_bins=8,
            numeric_embedding_dim=32,
            item_id_dim=32,
            pairwise_tower=True,
            user_embedding_dropout=0,
            item_embedding_dropout=0,
        )
        model = build_model(config).eval()
        edges, metadata = fit_numeric_bins(self.data, 8, sample_size=100)
        self.assertEqual(metadata["source"], "train_only")
        model.features.set_numeric_bins(edges)
        base, events, mask = model.features(batch)
        self.assertEqual(tuple(base.shape), (len(batch["item_id"]), 864))
        self.assertEqual(model.chunk_width, 108)
        self.assertEqual(model.features.event_mlp[0].in_features, 288)
        self.assertEqual(events.shape[-1], config.width)
        torch.testing.assert_close(events[:, 0], model.features.separator.expand(events.shape[0], -1))
        torch.testing.assert_close(mask.sum(dim=1), torch.as_tensor(batch["history_lengths"] + 1, dtype=torch.int64))
        # Fixture starts on Monday 2026-09-07 00:00 UTC.
        torch.testing.assert_close(batch["time_bins"][0], torch.tensor([0, 0, 0, 1, 0]))
        altered = dict(batch)
        for name in ("item_missing", "history_item_missing", "history_stats"):
            altered[name] = torch.randn_like(batch[name]) * 100
        changed_base, changed_events, _ = model.features(altered)
        torch.testing.assert_close(base, changed_base)
        torch.testing.assert_close(events, changed_events)
        qr = model.features.item_id
        ids = torch.tensor([0, 1, 65536, 65537, 2**31 - 1])
        expected = qr.quotient(torch.tensor([0, 1, 2, 2, 32768])) + qr.remainder(torch.tensor([0, 2, 1, 2, 65536]))
        torch.testing.assert_close(qr(ids), expected)
        torch.testing.assert_close(qr(ids)[0], torch.zeros(32))
        restored = build_model(config).eval()
        restored.load_state_dict(model.state_dict())
        outputs = model(batch)
        self.assertEqual(set(outputs), {"main", "aux", "pairwise"})
        torch.testing.assert_close(outputs["main"], restored(batch)["main"])

    def test_sampled_pairwise_gap_availability_and_gradients(self):
        batch = {
            "counts": np.array([5, 3, 2]),
            "position": torch.tensor([8, 1, 5, 0, 4, 0, 1, 2, -1, -1]),
            "position_available": np.array([True, True, False]),
        }
        sample_pairwise(batch, 256, 4, torch.Generator().manual_seed(7))
        np.testing.assert_array_equal(batch["pairwise_available"], [True, False, False])
        pairs = batch["pairwise_indices"][0]
        pos = batch["position"]
        self.assertTrue(((pos[pairs[:, 1]] - pos[pairs[:, 0]]) >= 4).all())
        self.assertGreater(torch.unique(pairs, dim=0).shape[0], 1)
        good = (-pos.float()).requires_grad_()
        loss, count = pairwise_loss(good, batch["pairwise_indices"], batch["counts"], batch["pairwise_available"])
        bad, _ = pairwise_loss(-good, batch["pairwise_indices"], batch["counts"], batch["pairwise_available"])
        self.assertEqual(count, 1)
        self.assertLess(loss.item(), bad.item())
        loss.backward()
        torch.testing.assert_close(good.grad[5:], torch.zeros(5))
        self.assertIsNone(pairwise_loss(good, batch["pairwise_indices"], batch["counts"], np.zeros(3, dtype=bool))[0])

    def test_unified_query_prefix_matches_literal_causal_self_attention(self):
        config = ModelConfig(
            name="onetrans",
            attention_layout="unified_prefix",
            width=16,
            attention_heads=4,
            depth=1,
            hidden=(8,),
            dropout=0,
            user_embedding_dropout=0,
            item_embedding_dropout=0,
        )
        model = build_model(config).eval()
        batch = collate_feeds(list(FeedDataset([self.data / "train"], max_feeds=4)))
        _, sequence, mask = model.features(batch)
        block = model.blocks[0]
        queries = torch.randn(len(batch["item_id"]), config.tokens, config.width)
        feed = batch["candidate_feed"]
        offsets = torch.as_tensor(np.cumsum(np.r_[0, batch["counts"][:-1]]))
        positions = torch.arange(len(queries)) - offsets[feed]
        actual_sequence, cache = block.advance_sequence_with_cache(sequence, mask)
        actual = block.read_cached_history(queries, cache, feed, positions, int(batch["counts"].max()))
        norm = block.query_norm(queries)
        sq, sk, sv = block.sequence_qkv(block.sequence_norm(sequence)).chunk(3, dim=-1)
        q = torch.cat((block.query_q(norm), sq[feed]), dim=1)
        k = torch.cat((block.query_k(norm), sk[feed]), dim=1)
        v = torch.cat((block.query_v(norm), sv[feed]), dim=1)
        length = q.shape[1]
        valid = torch.cat((torch.ones(len(q), config.tokens, dtype=torch.bool), mask[feed]), dim=1)
        visible = torch.ones(length, length, dtype=torch.bool).triu()[None, None] & valid[:, None, None, :]
        joined = (
            torch.nn.functional.scaled_dot_product_attention(
                block.heads_view(q), block.heads_view(k), block.heads_view(v), attn_mask=visible
            )
            .transpose(1, 2)
            .flatten(2)
        )
        expected = queries + block.query_out(joined[:, : config.tokens])
        expected = expected + block.query_ffn(block.query_ffn_norm(expected))
        torch.testing.assert_close(actual, expected)
        expected_s = sequence[feed] + block.sequence_out(joined[:, config.tokens :])
        expected_s = (expected_s + block.sequence_ffn(block.sequence_ffn_norm(expected_s))) * mask[feed, :, None]
        torch.testing.assert_close(actual_sequence[feed], expected_s)
        altered_queries = queries.clone()
        altered_queries[:, 0] += 5
        altered = block.read_cached_history(altered_queries, cache, feed, positions, int(batch["counts"].max()))
        torch.testing.assert_close(altered[:, 1:], actual[:, 1:])
        changed = sequence.clone()
        changed[:, 0] += 7
        torch.testing.assert_close(block.advance_sequence(changed, mask)[:, 1:], actual_sequence[:, 1:])
        actual.square().mean().backward()
        self.assertGreater(block.query_k.weight.grad.abs().sum().item(), 0)
        self.assertGreater(block.query_v.weight.grad.abs().sum().item(), 0)

    def test_shared_history_encoder_pooling_and_augmentation(self):
        batch = collate_feeds(list(FeedDataset([self.data / "train"], max_feeds=3)))
        encoder = FeatureEncoder(ModelConfig(user_embedding_dropout=1.0, item_embedding_dropout=1.0)).eval()
        history = encoder.encode_history(batch)
        torch.testing.assert_close(history[0], torch.zeros(128))
        padded = {k: v for k, v in batch.items()}
        for key in (
            "history_item",
            "history_item_missing",
            "history_category",
            "history_event_type",
            "history_event_source",
            "history_mask",
        ):
            value = batch[key]
            shape = list(value.shape)
            shape[1] = 2
            padding = torch.zeros(shape, dtype=value.dtype)
            if key == "history_category":
                padding.fill_(53)
            padded[key] = torch.cat([value, padding], dim=1)
        torch.testing.assert_close(encoder.encode_history(padded), history)
        base = encoder(batch)
        self.assertEqual(base.shape[1], 792)
        history.sum().backward()
        self.assertGreater(encoder.category.weight.grad[1].abs().sum().item(), 0.0)
        vector = torch.ones(3, 128)
        encoder.train()
        zero, mask = encoder.corrupt(vector, 1.0)
        self.assertTrue(mask.all())
        self.assertEqual(zero.abs().sum().item(), 0.0)
        encoder.eval()
        unchanged, _ = encoder.corrupt(vector, 1.0)
        torch.testing.assert_close(unchanged, vector)
        encoder.config.embedding_corruption = "random"
        encoder.train()
        random, _ = encoder.corrupt(vector, 1.0)
        torch.testing.assert_close(random.norm(dim=-1), torch.ones(3))

    def test_listmle_order_invariance_and_missing_pos(self):
        scores = torch.tensor([3.0, 2.0, -1.0, 0.5, 1.0], requires_grad=True)
        position = torch.tensor([0, 4, 9, -1, -1])
        loss, feeds = listmle(scores, position, [3, 2], [True, False])
        expected = sum(torch.logsumexp(scores[i:3], dim=0) - scores[i] for i in range(2)) / 2
        torch.testing.assert_close(loss, expected)
        shifted, _ = listmle(scores + 1000, position, [3, 2], [True, False])
        torch.testing.assert_close(loss, shifted, atol=1e-4, rtol=1e-4)
        loss.backward()
        torch.testing.assert_close(scores.grad[3:], torch.zeros(2))
        self.assertIsNone(listmle(scores, position, [5], [False])[0])

    def test_tensorboard_scalars(self):
        already_loaded = "tensorflow" in sys.modules
        folder = self.root / "monitor"
        with redirect_stderr(io.StringIO()):
            monitor = RunMonitor(folder, tensorboard=True)
            monitor.record("train", 1, {"main_bce": 0.3, "aux_listmle": 0.5})
            monitor.close()
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(folder / "tensorboard")).Reload()
        self.assertEqual(events.Scalars("loss/train_main_bce")[0].value, np.float32(0.3))
        self.assertFalse(any(tag.startswith("start/") for tag in events.Tags()["scalars"]))
        self.assertEqual("tensorflow" in sys.modules, already_loaded)


if __name__ == "__main__":
    unittest.main()
