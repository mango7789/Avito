"""Feed-equal pessimistic-tie NDCG and exact validation classification metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class RankingMetrics:
    def __init__(self):
        self.scores, self.labels = [], []
        self.ndcg = np.zeros(3, dtype=np.float64)
        self.feeds = 0

    def update(self, scores, labels, counts):
        scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
        if sum(counts) != len(scores) or len(labels) != len(scores) or not np.isfinite(scores).all():
            raise ValueError("Nonfinite scores or misaligned evaluation batches")
        if not np.isin(labels, [0, 1]).all() or any(count <= 0 for count in counts):
            raise ValueError("Evaluation requires binary labels and nonempty feeds")
        labels = labels.astype(np.int8)
        self.scores.append(scores)
        self.labels.append(labels)
        offset = 0
        for count in counts:
            target, score = labels[offset : offset + count], scores[offset : offset + count]
            ranked = target[np.lexsort((target, -score))]
            discounts = 1 / np.log2(np.arange(count) + 2)
            positives = int(target.sum())
            for j, cutoff in enumerate((count, 10, 30)):
                limit = min(count, cutoff)
                ideal = discounts[: min(positives, limit)].sum()
                self.ndcg[j] += np.dot(ranked[:limit], discounts[:limit]) / ideal if ideal else 1.0
            self.feeds += 1
            offset += count

    def compute(self):
        if not self.feeds:
            raise ValueError("No validation feeds were evaluated")
        scores, labels = np.concatenate(self.scores), np.concatenate(self.labels)
        probability = np.exp(-np.logaddexp(0, -scores))
        return {
            **dict(zip(("ndcg", "ndcg_at_10", "ndcg_at_30"), (self.ndcg / self.feeds).tolist())),
            "logloss": float(np.mean(np.logaddexp(0, scores) - labels * scores)),
            "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "average_precision": float(average_precision_score(labels, scores)) if labels.any() else None,
            "brier": float(np.mean((probability - labels) ** 2)),
            "positive_rate": float(labels.mean()),
            "mean_prediction": float(probability.mean()),
            "feeds": self.feeds,
            "candidates": len(labels),
        }
