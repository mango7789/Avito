"""Target-conditioned DIN interest features for the TokenMixer backbone.

Local activation uses [query, key, query-key, query*key], PReLU and an
unnormalized weighted sum, following https://arxiv.org/abs/1706.06978.
The smaller attention projection and retained mean/product features are Avito adaptations.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .onetrans import UnifiedFeatures


class LightCrossFeatures(nn.Module):
    """Embed exact candidate/history ID matches, with no label or exposure-position input.

    Avito supplies category and item IDs, not the shop or category hierarchy IDs
    in the reference. Each field has separate trainable hit/count tables; zero
    is a real non-match value, so these tables deliberately have no padding_idx.
    """

    def __init__(self, dimension, scale_fields, fields):
        super().__init__()
        self.fields = tuple(fields)
        self.hit = nn.ModuleDict({name: nn.Embedding(2, dimension) for name in self.fields})
        self.count = nn.ModuleDict({name: nn.Embedding(257, dimension) for name in self.fields})
        self.statistic_names = tuple(f"lcfn_{name}_hits" for name in self.fields) + tuple(
            f"lcfn_{name}_count_sum" for name in self.fields
        )
        self.scale = dimension**-0.5 if scale_fields else 1.0
        self.statistics = None
        self.candidates = 0

    @staticmethod
    @torch.no_grad()
    def match_counts(batch, fields):
        mask = batch["history_mask"]
        candidate_feed = batch["candidate_feed"]
        if mask.shape[1] > 256:
            raise ValueError("LCFN count embeddings support at most 256 history events")
        if mask.shape[1] == 0:
            zero = torch.zeros_like(candidate_feed)
            return (zero,) * len(fields)
        counts = np.asarray(batch["counts"], dtype=np.int64)
        starts = np.r_[0, np.cumsum(counts[:-1])]
        width = int(counts.max())
        indices = np.minimum(starts[:, None] + np.arange(width)[None], len(candidate_feed) - 1)
        positions = np.arange(len(candidate_feed)) - np.repeat(starts, counts)
        packed = torch.from_numpy(np.concatenate((indices.reshape(-1), positions)))
        device = candidate_feed.device
        packed = packed.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else packed.to(device)
        indices = packed[: indices.size].reshape(len(counts), width)
        positions = packed[len(counts) * width :]
        matches = []
        names = {"category": ("category", "history_category"), "item": ("candidate_item_id", "history_item_id")}
        for field in fields:
            candidate_name, history_name = names[field]
            history = batch[history_name]
            candidate = batch[candidate_name]
            # Unknown IDs and padded positions cannot match. Missing official
            # item vectors do not invalidate an otherwise known item ID.
            history = history.masked_fill(~mask | (history <= 0), torch.iinfo(torch.int64).max)
            ordered = history.sort(dim=1).values.contiguous()
            queries = candidate[indices].contiguous()
            frequencies = torch.searchsorted(ordered, queries, right=True) - torch.searchsorted(ordered, queries)
            matches.append(frequencies[candidate_feed, positions] * (candidate > 0))
        return tuple(matches)

    def forward(self, batch):
        counts = self.match_counts(batch, self.fields)
        self.statistics = torch.stack([(count > 0).sum() for count in counts] + [count.sum() for count in counts])
        self.candidates = len(counts[0])
        embeddings = []
        for name, count in zip(self.fields, counts):
            embeddings.extend((self.hit[name]((count > 0).long()), self.count[name](count)))
        return torch.cat(embeddings, dim=-1) * self.scale


class DINFeatures(UnifiedFeatures):
    def __init__(self, config):
        super().__init__(config)
        del self.separator
        self.fast_execution = config.fast_execution
        self.interest_width = 256 if config.din_softmax_context else 128
        event_width = (
            128
            + config.category_dim
            + config.event_dim
            + config.source_dim
            + config.item_id_dim
            + config.history_position_dim
        )
        if config.history_position_dim:
            self.history_position = nn.Embedding(257, config.history_position_dim, padding_idx=0)
        self.event_mlp = nn.Sequential(nn.Linear(event_width, 256), nn.GELU(), nn.Linear(256, 256))
        self.history_projection = nn.Linear(256, 128, bias=False)
        query_width = 256 + config.category_dim + config.item_id_dim if config.din_query_context else 128
        self.query_projection = nn.Linear(query_width, config.din_attention_dim)
        self.key_projection = nn.Linear(128, config.din_attention_dim)
        self.activation_input = nn.Linear(4 * config.din_attention_dim, 32)
        self.activation_hidden = nn.Linear(32, 16)
        self.activation_output = nn.Linear(16, 1)
        self.alpha_input = nn.Parameter(torch.tensor([0.25]))
        self.alpha_hidden = nn.Parameter(torch.tensor([0.25]))
        nn.init.normal_(self.activation_output.weight, std=0.01)
        nn.init.zeros_(self.activation_output.bias)
        if config.numeric_bins and config.numeric_residual:
            self.dimensions = (*self.dimensions[:-1], 128, 128, 128, self.interest_width, self.dimensions[-1])
        else:
            self.dimensions = (*self.dimensions, 128, 128, 128, self.interest_width)
        if config.similarity_residual:
            self.dimensions = (*self.dimensions, 5 if config.recent_history_window else 3)
        if config.relative_numeric:
            self.dimensions = (*self.dimensions, 5)
        self.lcfn = (
            LightCrossFeatures(config.lcfn_embedding_dim, config.scale_embedding_fields, config.lcfn_fields)
            if config.lcfn
            else None
        )
        if self.lcfn is not None:
            self.dimensions = (*self.dimensions, 2 * len(config.lcfn_fields) * config.lcfn_embedding_dim)

    def pool_interest(self, weights, values, mask):
        interest = torch.bmm(weights * mask, values)
        if not self.config.din_softmax_context:
            return interest
        # Retain DIN's original unnormalized sum. The additional context shares
        # its logits and values, with FP32 softmax over valid history positions.
        # A finite minimum followed by masking gives zero for entirely empty feeds.
        logits = weights.float().masked_fill(~mask, torch.finfo(torch.float32).min)
        probability = F.softmax(logits, dim=-1) * mask
        normalized = torch.bmm(probability.to(values.dtype), values)
        return torch.cat((interest, normalized), dim=-1)

    def attend_chunk(self, query, query_part, key, key_part, values, feed, mask):
        # Factor linear query/key terms before candidate expansion; only the product
        # term needs a candidate-by-event activation. Histories remain shared by feed.
        product_weight = self.activation_input.weight.chunk(4, dim=1)[3]
        product = F.linear(query.unsqueeze(1) * key[feed], product_weight)
        hidden = product + query_part.unsqueeze(1) + key_part[feed]
        hidden = F.prelu(hidden, self.alpha_input.to(hidden.dtype))
        hidden = self.activation_hidden(hidden)
        hidden = F.prelu(hidden, self.alpha_hidden.to(hidden.dtype))
        weights = self.activation_output(hidden).squeeze(-1)
        return self.pool_interest(weights.unsqueeze(1), values[feed], mask[feed, None]).squeeze(1)

    def attend_training(self, *args):
        if self.config.din_recompute == "off":
            return self.attend_chunk(*args)
        # Keep recomputation native: the compiled combinations failed with
        # saved-tensor metadata and partitioning errors on this PyTorch version.
        return checkpoint(self.attend_chunk, *args, use_reentrant=False, preserve_rng_state=False)

    def attend_group(self, query, query_part, key, key_part, values, mask):
        """Factor the bilinear term and reuse each feed's values for all its candidates."""
        product_weight = self.activation_input.weight.chunk(4, dim=1)[3]
        # [feed, event, attention, hidden], rather than candidate-repeated history vectors.
        weighted_key = key.unsqueeze(-1) * product_weight.t().to(key.dtype)[None, None]
        feeds, events, attention, hidden_width = weighted_key.shape
        weighted_key = weighted_key.permute(0, 2, 1, 3).reshape(feeds, attention, events * hidden_width)
        product = torch.bmm(query, weighted_key).reshape(feeds, query.shape[1], events, hidden_width)
        hidden = product + query_part.unsqueeze(2) + key_part.unsqueeze(1)
        hidden = F.prelu(hidden, self.alpha_input.to(hidden.dtype))
        hidden = self.activation_hidden(hidden)
        hidden = F.prelu(hidden, self.alpha_hidden.to(hidden.dtype))
        weights = self.activation_output(hidden).squeeze(-1)
        return self.pool_interest(weights, values, mask[:, None])

    def attend_grouped(self, batch, query, query_part, key, key_part, values, mask):
        counts = np.asarray(batch["counts"], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(counts)]
        bucket_size = self.config.din_history_bucket
        if bucket_size:
            lengths = np.asarray(batch["history_lengths"], dtype=np.int64)
            buckets = (np.maximum(lengths, 1) - 1) // bucket_size
            order = np.lexsort((counts, buckets))
        else:
            order = np.argsort(counts, kind="stable")
        groups, pending = [], []
        for feed in order:
            count = int(counts[feed])
            if not count:
                continue
            new_bucket = bool(pending) and bucket_size and buckets[feed] != buckets[pending[0]]
            if pending and (
                new_bucket or count * (len(pending) + 1) > self.config.din_chunk_size or len(pending) >= 64
            ):
                groups.append(pending)
                pending = []
            pending.append(int(feed))
        if pending:
            groups.append(pending)
        # Build all CPU grouping metadata first. A single pinned transfer avoids
        # a device synchronization for every small index array inside the loop.
        plans, original_indices, packed_arrays = [], [], []
        packed_size = 0

        def pack(array):
            nonlocal packed_size
            array = np.asarray(array, dtype=np.int64).reshape(-1)
            section = slice(packed_size, packed_size + len(array))
            packed_arrays.append(array)
            packed_size += len(array)
            return section

        for feeds in groups:
            maximum = int(counts[feeds].max())
            if maximum > self.config.din_chunk_size:
                feed = feeds[0]
                start, stop = int(offsets[feed]), int(offsets[feed + 1])
                for first in range(start, stop, self.config.din_chunk_size):
                    last = min(first + self.config.din_chunk_size, stop)
                    plans.append((first, last))
                    original_indices.append(np.arange(first, last))
                continue
            columns = np.arange(maximum)
            valid = columns[None] < counts[feeds, None]
            padded_indices = offsets[feeds, None] + columns[None]
            original_indices.append(padded_indices[valid])
            # Drop masked suffixes only; keep one masked event for empty groups.
            events = max(1, int(lengths[feeds].max())) if bucket_size else key.shape[1]
            plans.append(
                (
                    len(feeds),
                    maximum,
                    events,
                    pack(np.minimum(padded_indices, len(query) - 1)),
                    pack(feeds),
                    pack(np.flatnonzero(valid.reshape(-1))),
                )
            )
        indices = np.concatenate(original_indices)
        inverse = np.empty(len(query), dtype=np.int64)
        inverse[indices] = np.arange(len(indices))
        inverse_section = pack(inverse)
        packed = torch.from_numpy(np.concatenate(packed_arrays))
        if query.is_cuda:
            packed = packed.pin_memory().to(query.device, non_blocking=True)
        else:
            packed = packed.to(query.device)
        regular = [plan for plan in plans if len(plan) != 2]
        grouped_inputs = iter(())
        if regular:
            candidate_sizes = [plan[0] * plan[1] for plan in regular]
            feed_sizes = [plan[0] for plan in regular]
            candidate_indices = torch.cat([packed[plan[3]] for plan in regular])
            feed_indices = torch.cat([packed[plan[4]] for plan in regular])
            # One gather and one split per source tensor. Per-group advanced
            # indexing made every backward allocate a full-batch zero gradient;
            # SplitWithSizesBackward instead combines the group gradients once.
            grouped_inputs = iter(
                zip(
                    query.index_select(0, candidate_indices).split(candidate_sizes),
                    query_part.index_select(0, candidate_indices).split(candidate_sizes),
                    key.index_select(0, feed_indices).split(feed_sizes),
                    key_part.index_select(0, feed_indices).split(feed_sizes),
                    values.index_select(0, feed_indices).split(feed_sizes),
                    mask.index_select(0, feed_indices).split(feed_sizes),
                )
            )
        outputs = []
        for plan in plans:
            if len(plan) == 2:
                # Outlier feeds retain the bounded chunks of the reference path.
                first, last = plan
                args = (
                    query[first:last],
                    query_part[first:last],
                    key,
                    key_part,
                    values,
                    batch["candidate_feed"][first:last],
                    mask,
                )
                function = self.attend_training if self.training and torch.is_grad_enabled() else self.attend_chunk
                outputs.append(function(*args))
                continue
            feed_count, maximum, events, _, _, selected_section = plan
            local_query, local_query_part, local_key, local_key_part, local_values, local_mask = next(grouped_inputs)
            args = (
                local_query.reshape(feed_count, maximum, -1),
                local_query_part.reshape(feed_count, maximum, -1),
                local_key[:, :events],
                local_key_part[:, :events],
                local_values[:, :events],
                local_mask[:, :events],
            )
            if self.training and torch.is_grad_enabled() and self.config.din_recompute == "on":
                output = checkpoint(self.attend_group, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                output = self.attend_group(*args)
            outputs.append(output.reshape(-1, self.interest_width)[packed[selected_section]])
        return torch.cat(outputs, dim=0)[packed[inverse_section]]

    @staticmethod
    def similarity_residuals(batch, user, item, history_item, recent_window=0):
        """Candidate UI/history cosine residuals without materializing candidate-history vectors."""
        feeds = batch["history_mask"].shape[0]
        candidate_feed = batch["candidate_feed"]
        counts = np.asarray(batch["counts"], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(counts)]
        output = item.new_zeros((len(item), 5 if recent_window else 3), dtype=torch.float32)
        layout = torch.from_numpy(np.stack((counts, offsets[:-1])))
        layout = layout.pin_memory().to(item.device, non_blocking=True) if item.is_cuda else layout.to(item.device)
        with torch.no_grad(), torch.autocast(item.device.type, enabled=False):
            users = user.float()
            candidates = item.float()
            histories = history_item.float()
            mask = batch["history_mask"]
            ui = (users * candidates).sum(dim=-1)
            output[:, 0] = ui
            for feed_start in range(0, feeds, 128):
                feed_stop = min(feed_start + 128, feeds)
                candidate_start, candidate_stop = int(offsets[feed_start]), int(offsets[feed_stop])
                local_counts_cpu = counts[feed_start:feed_stop]
                local_counts = layout[0, feed_start:feed_stop]
                local_starts = layout[1, feed_start:feed_stop] - candidate_start
                local_feeds = candidate_feed[candidate_start:candidate_stop] - feed_start
                positions = (
                    torch.arange(candidate_stop - candidate_start, device=item.device) - local_starts[local_feeds]
                )
                maximum = int(local_counts_cpu.max())
                padded = candidates.new_zeros((feed_stop - feed_start, maximum, candidates.shape[-1]))
                padded[local_feeds, positions] = candidates[candidate_start:candidate_stop]
                valid_history = mask[feed_start:feed_stop] & histories[feed_start:feed_stop].ne(0).any(dim=-1)
                candidate_mask = torch.arange(maximum, device=item.device)[None] < local_counts[:, None]
                if histories.shape[1]:
                    cosine = torch.bmm(padded, histories[feed_start:feed_stop].transpose(1, 2))
                    cosine = cosine.masked_fill(~valid_history[:, None], float("-inf"))
                    maximum_cosine = cosine.max(dim=-1).values
                    available = valid_history.any(dim=-1, keepdim=True) & padded.ne(0).any(dim=-1)
                    maximum_cosine = torch.where(available, maximum_cosine, torch.zeros_like(maximum_cosine))
                else:
                    maximum_cosine = padded.new_zeros(padded.shape[:2])
                flat_maximum = maximum_cosine[local_feeds, positions]
                mean = (maximum_cosine * candidate_mask).sum(dim=1, keepdim=True) / local_counts[:, None]
                centered = maximum_cosine - mean
                variance = (centered.square() * candidate_mask).sum(dim=1, keepdim=True) / local_counts[:, None]
                output[candidate_start:candidate_stop, 1] = flat_maximum
                output[candidate_start:candidate_stop, 2] = (
                    flat_maximum - mean.squeeze(1)[local_feeds]
                ) / variance.sqrt().squeeze(1)[local_feeds].clamp_min(1e-6)
                if recent_window:
                    if histories.shape[1]:
                        recent_maximum = cosine[..., :recent_window].max(dim=-1).values
                        recent_available = valid_history[:, :recent_window].any(dim=-1, keepdim=True)
                        recent_available = recent_available & padded.ne(0).any(dim=-1)
                        recent_maximum = torch.where(recent_available, recent_maximum, torch.zeros_like(recent_maximum))
                    else:
                        recent_maximum = padded.new_zeros(padded.shape[:2])
                    recent_mean = (recent_maximum * candidate_mask).sum(dim=1, keepdim=True) / local_counts[:, None]
                    recent_variance = ((recent_maximum - recent_mean).square() * candidate_mask).sum(
                        dim=1, keepdim=True
                    ) / local_counts[:, None]
                    recent_flat = recent_maximum[local_feeds, positions]
                    output[candidate_start:candidate_stop, 3] = recent_flat
                    output[candidate_start:candidate_stop, 4] = (
                        recent_flat - recent_mean.squeeze(1)[local_feeds]
                    ) / recent_variance.sqrt().squeeze(1)[local_feeds].clamp_min(1e-6)
        return output

    @staticmethod
    @torch.no_grad()
    def relative_numerical(batch):
        """Within-feed population zscores; constant columns remain exactly zero."""
        numerical = batch["numerical"]
        counts = np.asarray(batch["counts"], dtype=np.int64)
        layout = torch.from_numpy(np.stack((np.r_[0, np.cumsum(counts[:-1])], counts)))
        layout = (
            layout.pin_memory().to(numerical.device, non_blocking=True)
            if numerical.is_cuda
            else layout.to(numerical.device)
        )
        with torch.autocast(numerical.device.type, enabled=False):
            numerical = numerical.float()
            feed = batch["candidate_feed"]
            starts = layout[0]
            denominator = layout[1].float()[:, None]
            shifted = numerical - numerical[starts][feed]
            mean = numerical.new_zeros((len(counts), numerical.shape[-1]))
            mean.index_add_(0, feed, shifted)
            mean = mean / denominator
            centered = shifted - mean[feed]
            variance = torch.zeros_like(mean)
            variance.index_add_(0, feed, centered.square())
            deviation = (variance / denominator).sqrt().clamp_min(1e-6)
            return centered / deviation[feed]

    def forward(self, batch):
        base = self.encode_nonsequential(batch)
        # Read the same corrupted vectors used in the base, without a second RNG draw.
        offset = (5 * self.config.numeric_embedding_dim if self.config.numeric_bins else 5) + self.config.category_dim
        user, item = base[:, offset : offset + 128], base[:, offset + 128 : offset + 256]
        raw_numerical = base[:, -5:] if self.config.numeric_bins and self.config.numeric_residual else None
        if raw_numerical is not None:
            base = base[:, :-5]
        mask = batch["history_mask"]
        if self.config.similarity_residual:
            events, history_item = self.encode_events(batch, return_items=True)
        else:
            events = self.encode_events(batch)
            history_item = None
        values = self.history_projection(events) * mask.unsqueeze(-1)
        mean = values.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        feed = batch["candidate_feed"]
        mean = mean[feed]
        if values.shape[1] == 0:
            interest = item.new_zeros((len(item), self.interest_width))
        else:
            if self.config.din_query_context:
                # Reuse the already encoded/corrupted fields: no new embedding
                # tables or random masks. Raw numerical residuals were removed
                # above, leaving QR IDs at the end of the nonsequential base.
                category = base[:, offset - self.config.category_dim : offset]
                query_fields = [user, item, category]
                if self.config.item_id_dim:
                    query_fields.append(base[:, -self.config.item_id_dim :])
                query_input = torch.cat(query_fields, dim=-1)
            else:
                query_input = item
            query, key = self.query_projection(query_input), self.key_projection(values)
            w_query, w_key, w_difference, _ = self.activation_input.weight.chunk(4, dim=1)
            query_part = F.linear(query, w_query + w_difference)
            key_part = F.linear(key, w_key - w_difference, self.activation_input.bias)
            if self.fast_execution:
                interest = self.attend_grouped(batch, query, query_part, key, key_part, values, mask)
            else:
                chunks = []
                for start in range(0, len(item), self.config.din_chunk_size):
                    stop = start + self.config.din_chunk_size
                    args = (query[start:stop], query_part[start:stop], key, key_part, values, feed[start:stop], mask)
                    if self.training and torch.is_grad_enabled():
                        chunks.append(self.attend_training(*args))
                    else:
                        chunks.append(self.attend_chunk(*args))
                interest = torch.cat(chunks, dim=0)
        features = (base, mean, mean * item, user * mean, interest)
        if raw_numerical is not None:
            features = (*features, raw_numerical)
        if history_item is not None:
            features = (
                *features,
                self.similarity_residuals(batch, user, item, history_item, self.config.recent_history_window).to(
                    base.dtype
                ),
            )
        if self.config.relative_numeric:
            features = (*features, self.relative_numerical(batch).to(base.dtype))
        if self.lcfn is not None:
            features = (*features, self.lcfn(batch).to(base.dtype))
        return torch.cat(features, dim=-1)
