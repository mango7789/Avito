"""Target-based main ranking losses and auxiliary order supervision from train.pos."""

import numpy as np
import torch
from torch.nn import functional as F


def prepare_ranking_layout(batch, device, kind="softmax"):
    """Transfer shared ranking indices before forward, without post-forward CPU/GPU fences."""
    counts = np.asarray(batch["counts"], dtype=np.int64)
    if not len(counts):
        return None
    width = int(counts.max())
    offsets = np.cumsum(np.r_[0, counts[:-1]])
    target = batch["target"].detach().cpu().numpy()
    positives = np.add.reduceat(target, offsets)
    eligible = positives > 0
    if kind == "ndcg_pairwise":
        eligible &= positives < counts
    arrays = {
        "indices": np.minimum(offsets[:, None] + np.arange(width)[None], len(target) - 1).reshape(-1),
        "counts": counts,
        "offsets": offsets,
        "target": np.flatnonzero(eligible),
        "aux": np.flatnonzero(np.asarray(batch["position_available"]) & (counts > 1)),
        "pair": np.flatnonzero(batch.get("pairwise_available", np.zeros(len(counts), dtype=bool))),
    }
    packed = torch.from_numpy(np.concatenate(list(arrays.values())))
    device = torch.device(device)
    packed = packed.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else packed.to(device)
    layout, cursor = {}, 0
    for name, array in arrays.items():
        layout[name] = packed[cursor : cursor + len(array)]
        cursor += len(array)
    layout["indices"] = layout["indices"].reshape(len(counts), width)
    columns = torch.arange(width, device=device)[None]
    layout["valid"] = columns < layout["counts"][:, None]
    layout["decisions"] = columns < (layout["counts"] - 1)[:, None]
    return layout


def sample_pairwise(batch, pairs_per_feed=64, min_gap=4, generator=None):
    """Uniformly draw eligible unordered pairs with replacement, without building an M x M matrix.

    Store relative indices per feed so data-parallel splits retain the exact same sampled pairs.
    Sampling uses the checkpointed CPU Torch RNG before work is dispatched to either GPU.
    """
    if pairs_per_feed <= 0 or min_gap <= 0:
        raise ValueError("Pair count and minimum original-position gap must be positive")
    counts = batch["counts"]
    positions = batch["position"].detach().cpu().numpy()
    sampled = np.zeros((len(counts), pairs_per_feed, 2), dtype=np.int64)
    available = np.zeros(len(counts), dtype=bool)
    offset = 0
    for feed, count in enumerate(counts):
        local = positions[offset : offset + count]
        offset += count
        if not batch["position_available"][feed] or count < 2:
            continue
        order = np.argsort(local, kind="stable")
        ranked = local[order]
        starts = np.searchsorted(ranked, ranked + min_gap, side="left")
        cumulative = np.cumsum(count - starts)
        if not cumulative[-1]:
            continue
        ranks = torch.randint(int(cumulative[-1]), (pairs_per_feed,), generator=generator).numpy()
        better = np.searchsorted(cumulative, ranks, side="right")
        before = cumulative[better] - (count - starts[better])
        worse = starts[better] + ranks - before
        sampled[feed, :, 0], sampled[feed, :, 1] = order[better], order[worse]
        available[feed] = True
    batch["pairwise_indices"] = torch.from_numpy(sampled)
    batch["pairwise_available"] = available
    return batch


def pairwise_loss(scores, indices, counts, available, layout=None):
    """Feed-equal logistic loss on pre-sampled (better, worse) pairs for the separate pairwise tower."""
    eligible = np.flatnonzero(available)
    if not len(eligible):
        return None, 0
    selected = layout["pair"] if layout is not None else torch.as_tensor(eligible, device=scores.device)
    offsets = (
        layout["offsets"]
        if layout is not None
        else torch.as_tensor(np.cumsum(np.r_[0, counts[:-1]]), device=scores.device)
    )
    pairs = indices[selected] + offsets[selected, None, None]
    loss = F.softplus(scores[pairs[..., 1]] - scores[pairs[..., 0]])
    return loss.mean(dim=1).mean(), len(eligible)


def target_rank_loss(scores, target, counts, kind="softmax", layout=None):
    """Feed-equal target-based ranking loss for the main tower.

    Each eligible feed assigns equal probability mass to all clicked candidates:
    ``-sum(target * log_softmax(scores)) / positive_count``. Feeds without a
    positive target have no defined target distribution and are excluded.
    ``ndcg_pairwise`` instead samples 64 positive/negative pairs per eligible
    feed and normalizes detached delta-NDCG weights within that sampled set.
    This is a sampled, feed-normalized variant, not an exact LambdaLoss reproduction.
    """
    if kind not in {"softmax", "ndcg_pairwise"}:
        raise ValueError(f"Unknown target ranking loss: {kind}")
    counts = np.asarray(counts)
    if not len(counts):
        return None, 0
    width = int(counts.max())
    if layout is None:
        offsets = np.cumsum(np.r_[0, counts[:-1]])
        indices = offsets[:, None] + np.arange(width)[None, :]
        valid = torch.as_tensor(np.arange(width)[None, :] < counts[:, None], device=scores.device)
        indices = torch.as_tensor(np.minimum(indices, len(scores) - 1), device=scores.device)
    else:
        indices, valid = layout["indices"], layout["valid"]
    padded_scores = torch.where(valid, scores[indices], torch.finfo(scores.dtype).min / 4)
    padded_target = torch.where(valid, target[indices].to(scores.dtype), 0.0)
    positives = padded_target.sum(dim=1)
    if layout is None:
        eligible = positives > 0
        if kind == "ndcg_pairwise":
            eligible &= positives < torch.as_tensor(counts, device=scores.device)
        eligible_count = int(eligible.sum().item())
    else:
        eligible = layout["target"]
        eligible_count = eligible.numel()
    if not eligible_count:
        return None, 0
    if kind == "ndcg_pairwise":
        local_scores, labels, local_valid = padded_scores[eligible], padded_target[eligible], valid[eligible]
        with torch.no_grad():
            # Stable sorting matches the metric's pessimistic handling of tied scores.
            label_order = torch.argsort(labels, dim=1, stable=True)
            score_order = torch.argsort(local_scores.gather(1, label_order), dim=1, descending=True, stable=True)
            order = label_order.gather(1, score_order)
            ranks = torch.empty_like(order)
            positions = torch.arange(width, device=scores.device).expand_as(order)
            ranks.scatter_(1, order, positions)
            positive = torch.multinomial(labels, 64, replacement=True)
            negative = torch.multinomial((1 - labels) * local_valid, 64, replacement=True)
            discount = 1 / torch.log2(torch.arange(width, device=scores.device, dtype=torch.float32) + 2)
            ideal = discount.cumsum(dim=0)[labels.sum(dim=1).long() - 1]
            delta = (discount[ranks.gather(1, positive)] - discount[ranks.gather(1, negative)]).abs()
            delta = delta / ideal[:, None]
            weights = delta / delta.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pair_losses = F.softplus(local_scores.gather(1, negative) - local_scores.gather(1, positive))
        return (weights * pair_losses).sum(dim=1).mean(), eligible_count
    losses = -(padded_target * F.log_softmax(padded_scores, dim=1)).sum(dim=1) / positives.clamp_min(1)
    return losses[eligible].mean(), eligible_count


def listmle(scores, positions, counts, available, layout=None):
    """Average per-decision ListMLE across feeds with observed original positions.

    The final one-item choice contributes zero. Tied teacher positions keep their input order.
    Return None when a batch has no observed order; this leaves the auxiliary tower without gradients.
    """
    counts = np.asarray(counts)
    eligible = np.flatnonzero(np.asarray(available) & (counts > 1))
    if not len(eligible):
        return None, 0
    width = int(counts.max())
    if layout is None:
        offsets = np.cumsum(np.r_[0, counts[:-1]])
        indices = offsets[:, None] + np.arange(width)[None, :]
        valid = torch.as_tensor(np.arange(width)[None, :] < counts[:, None], device=scores.device)
        indices = torch.as_tensor(np.minimum(indices, len(scores) - 1), device=scores.device)
    else:
        indices, valid = layout["indices"], layout["valid"]
    teacher = torch.where(valid, positions[indices], torch.iinfo(torch.int64).max)
    order = torch.argsort(teacher, dim=1, stable=True)
    # A finite constant avoids -inf - -inf and NaN gradients in padded suffixes.
    padded = torch.where(valid, scores[indices], torch.finfo(scores.dtype).min / 4)
    ranked = padded.gather(1, order)
    log_denominator = torch.logcumsumexp(ranked.flip(1), dim=1).flip(1)
    decisions = (
        layout["decisions"]
        if layout is not None
        else torch.as_tensor(np.arange(width)[None, :] < (counts - 1)[:, None], device=scores.device)
    )
    losses = torch.where(decisions, log_denominator - ranked, 0.0).sum(dim=1)
    denominator = (
        (layout["counts"] - 1).clamp_min(1)
        if layout is not None
        else torch.as_tensor(np.maximum(counts - 1, 1), device=scores.device)
    )
    selected = layout["aux"] if layout is not None else torch.as_tensor(eligible, device=scores.device)
    return (losses / denominator)[selected].mean(), len(eligible)
