"""Unified right-causal attention on [query prefix; newest-first history], with feed-level shared KV.

The legacy separate cross-attention model remains available for its saved checkpoints.
Reference: https://arxiv.org/html/2510.26104v3; this adaptation keeps all historical events.
"""

from dataclasses import dataclass
import os
import threading

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import FeatureEncoder, PerTokenFFN, TaskHeads, TokenMixerBlock

_FLASH_VARLEN = None
_FLASH_CHECKED = False
_NORM_FUNCTION = None
_LAYER_NORM_FUNCTION = None
_NORM_LAUNCH_LOCK = threading.RLock()


def prepare_fused_norm(rms=True):
    """Serialize Triton's process-global autotuner metadata, while CUDA kernels run asynchronously."""
    global _NORM_FUNCTION, _LAYER_NORM_FUNCTION
    with _NORM_LAUNCH_LOCK:
        if _NORM_FUNCTION is None:
            from flash_attn.ops.triton import layer_norm

            for name in ("_layer_norm_fwd_1pass_kernel", "_layer_norm_bwd_kernel"):
                kernel = getattr(layer_norm, name)
                original = kernel.run

                def guarded(*args, _launch=original, **kwargs):
                    with _NORM_LAUNCH_LOCK:
                        return _launch(*args, **kwargs)

                kernel.run = guarded
            _NORM_FUNCTION = layer_norm.rms_norm_fn
            _LAYER_NORM_FUNCTION = layer_norm.layer_norm_fn
    return _NORM_FUNCTION if rms else _LAYER_NORM_FUNCTION


def flash_varlen():
    global _FLASH_VARLEN, _FLASH_CHECKED
    if not _FLASH_CHECKED:
        try:
            from flash_attn import flash_attn_varlen_func

            _FLASH_VARLEN = flash_attn_varlen_func
        except (ImportError, OSError):
            pass
        _FLASH_CHECKED = True
    return _FLASH_VARLEN


class OneTransFeatures(FeatureEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.event_mlp[-1] = nn.Linear(128, config.width)
        self.dimensions = (8, config.category_dim, 128, 128, 128)

    def encode_nonsequential(self, batch):
        index = batch["candidate_feed"]
        user, _ = self.corrupt(F.normalize(batch["user"], dim=-1), self.config.user_embedding_dropout)
        user = user[index]
        item, missing = self.corrupt(F.normalize(batch["item"], dim=-1), self.config.item_embedding_dropout)
        return torch.cat(
            (
                batch["numerical"],
                torch.maximum(batch["item_missing"], missing.unsqueeze(-1).float()),
                batch["history_stats"][index],
                self.category(batch["category"]),
                user,
                item,
                user * item,
            ),
            dim=-1,
        )

    def encode_sequence(self, batch):
        events = self.encode_events(batch)
        mask = batch["history_mask"]
        # Official histories are newest-first. Reverse valid events only, retaining right padding,
        # so a triangular causal mask means older -> newer and never exposes a future event.
        if events.shape[1]:
            positions = torch.arange(events.shape[1], device=events.device)
            reverse = (mask.sum(dim=1, keepdim=True) - 1 - positions).clamp_min(0)
            events = events.gather(1, reverse.unsqueeze(-1).expand(-1, -1, events.shape[-1]))
            events = events * mask.unsqueeze(-1)
        else:
            events = F.pad(events, (0, 0, 0, 1))
            mask = F.pad(mask, (0, 1))
        return events, mask

    def forward(self, batch):
        base = self.encode_nonsequential(batch)
        events, mask = self.encode_sequence(batch)
        return base, events, mask


@dataclass
class HistoryKV:
    """One layer's feed-level cache, kept on the autograd graph during training."""

    key: torch.Tensor | None
    value: torch.Tensor | None
    valid: torch.Tensor
    flat_key: torch.Tensor | None = None
    flat_value: torch.Tensor | None = None
    cu_lengths: torch.Tensor | None = None
    max_length: int = 0
    has_history: torch.Tensor | None = None


@dataclass
class HistoryCache:
    """All layers for the same immutable feed/history batch and model parameters.

    Lifetime is one forward/backward in training; eval callers may reuse it for candidate chunks.
    Candidate chunks must retain original feed indices and B-length counts (zeros are allowed).
    Never reuse after an optimizer step, a changed history, or a train/eval mode switch.
    """

    layers: tuple


class PerQueryLinear(nn.Module):
    """Independent bias-free projection for each nonsequential query token."""

    def __init__(self, tokens, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(tokens, width, width))
        for matrix in self.weight:
            nn.init.xavier_uniform_(matrix)

    def forward(self, queries):
        return torch.einsum("ntd,tdh->nth", queries, self.weight)


class FastRMSNorm(nn.RMSNorm):
    """Fused BF16 training normalization; native FP32 evaluation and unchanged state keys."""

    fused = True

    def forward(self, values):
        if self.training and values.is_cuda and values.dtype in (torch.bfloat16, torch.float16):
            if not self.fused:
                with torch.autocast(values.device.type, enabled=False):
                    return F.rms_norm(values.float(), self.normalized_shape, self.weight, self.eps).to(values.dtype)
            function = _NORM_FUNCTION or prepare_fused_norm()
            token_major = values.ndim == 3 and values.stride(0) == values.shape[-1]
            normalized = function(values.transpose(0, 1) if token_major else values, self.weight, None, eps=self.eps)
            return normalized.transpose(0, 1) if token_major else normalized
        return super().forward(values)


class OneTransBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.heads, self.width = config.attention_heads, width
        self.attention_dropout = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        self.sequence_norm = FastRMSNorm(width, eps=1e-6)
        self.sequence_qkv = nn.Linear(width, 3 * width, bias=False)
        self.sequence_out = nn.Linear(width, width, bias=False)
        self.sequence_ffn_norm = FastRMSNorm(width, eps=1e-6)
        self.sequence_ffn = nn.Sequential(
            nn.Linear(width, width * config.expansion),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width * config.expansion, width),
        )
        self.context_norm = FastRMSNorm(width, eps=1e-6)
        self.context_kv = nn.Linear(width, 2 * width, bias=False)
        self.query_norm = FastRMSNorm(width, eps=1e-6)
        self.query_q = PerQueryLinear(config.tokens, width)
        self.query_out = PerQueryLinear(config.tokens, width)
        self.query_ffn_norm = FastRMSNorm(width, eps=1e-6)
        self.query_ffn = PerTokenFFN(
            config.tokens, width, config.expansion, config.dropout, config.token_ffn_activation
        )
        self._compiled_sequence_ffn = None
        self._compiled_query_ffn = None

    def heads_view(self, values):
        batch, length, _ = values.shape
        return values.reshape(batch, length, self.heads, self.width // self.heads).transpose(1, 2)

    def advance_sequence(self, sequence, mask):
        q, k, v = self.sequence_qkv(self.sequence_norm(sequence)).chunk(3, dim=-1)
        output = F.scaled_dot_product_attention(
            self.heads_view(q),
            self.heads_view(k),
            self.heads_view(v),
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).flatten(2)
        sequence = sequence + self.dropout(self.sequence_out(output))
        ffn = self._compiled_sequence_ffn if self.training and self._compiled_sequence_ffn else self.sequence_ffn
        sequence = sequence + self.dropout(ffn(self.sequence_ffn_norm(sequence)))
        return sequence * mask.unsqueeze(-1)

    def cache_history(self, sequence, mask, layout=None):
        k, v = self.context_kv(self.context_norm(sequence)).chunk(2, dim=-1)
        valid = mask.clone()
        valid[:, 0] |= ~valid.any(dim=1)
        cache = HistoryKV(self.heads_view(k), self.heads_view(v), valid)
        if layout is not None:
            indices, cache.cu_lengths, cache.max_length = layout
            shape = (-1, self.heads, self.width // self.heads)
            cache.flat_key = k.reshape(shape).index_select(0, indices)
            cache.flat_value = v.reshape(shape).index_select(0, indices)
        return cache

    def read_cached_history(self, queries, cache, feed_index, candidate_position, max_candidates, cu_queries=None):
        # Pack all queries of a feed into one attention query axis. K/V are projected and stored
        # once per feed; do not replicate an L x D history for every candidate item.
        feeds = cache.key.shape[0]
        tokens = queries.shape[1]
        projected = self.query_q(self.query_norm(queries))
        if cache.flat_key is not None and cu_queries is not None:
            # Varlen FlashAttention consumes only real candidate queries and valid history tokens.
            # Each feed has M_b*T queries but just L_b keys, with no M_max or history padding work.
            output = flash_varlen()(
                projected.reshape(-1, self.heads, self.width // self.heads),
                cache.flat_key,
                cache.flat_value,
                cu_queries,
                cache.cu_lengths,
                max_candidates * tokens,
                cache.max_length,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=False,
            ).reshape_as(queries)
        else:
            output = self.padded_cross_attention(projected, cache, feed_index, candidate_position, max_candidates)
        queries = queries + self.dropout(self.query_out(output))
        ffn = self._compiled_query_ffn if self.training and self._compiled_query_ffn else self.query_ffn
        return queries + self.dropout(ffn(self.query_ffn_norm(queries)))

    def padded_cross_attention(self, projected, cache, feed_index, candidate_position, max_candidates):
        feeds = cache.key.shape[0]
        tokens = projected.shape[1]
        packed = projected.new_zeros((feeds, max_candidates, tokens, self.width))
        packed[feed_index, candidate_position] = projected
        q = self.heads_view(packed.reshape(feeds, max_candidates * tokens, self.width))
        # An empty history gets one masked-to-zero dummy value: no all-masked attention/NaN.
        output = F.scaled_dot_product_attention(
            q,
            cache.key,
            cache.value,
            attn_mask=cache.valid[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(feeds, max_candidates, tokens, self.width)
        output = output[feed_index, candidate_position]
        return output

    def read_history(self, queries, sequence, mask, feed_index, candidate_position, max_candidates):
        return self.read_cached_history(
            queries, self.cache_history(sequence, mask), feed_index, candidate_position, max_candidates
        )

    def forward(self, queries, sequence, mask, feed_index, candidate_position, max_candidates):
        sequence = self.advance_sequence(sequence, mask)
        queries = self.read_history(queries, sequence, mask, feed_index, candidate_position, max_candidates)
        return queries, sequence


class OneTrans(nn.Module):
    features_type = OneTransFeatures
    block_type = OneTransBlock

    def __init__(self, config):
        super().__init__()
        if (
            min(config.width, config.depth, config.tokens, config.attention_heads, config.expansion) <= 0
            or config.width % config.tokens
            or config.width % config.attention_heads
        ):
            raise ValueError("OneTrans width must be divisible by query-token count and attention-head count")
        self.features = self.features_type(config)
        self.base_width = sum(self.features.dimensions)
        self.chunk_width = (self.base_width + config.tokens - 1) // config.tokens
        self.padding = config.tokens * self.chunk_width - self.base_width
        self.projections = nn.ModuleList(nn.Linear(self.chunk_width, config.width) for _ in range(config.tokens))
        self.blocks = nn.ModuleList(self.block_type(config) for _ in range(config.depth))
        self.final_mixer = TokenMixerBlock(config.tokens, config)
        self._compiled_final_mixer = None
        self.heads = TaskHeads(config.tokens * config.width, config)

    def prepare_history_cache(self, batch):
        sequence, mask = self.features.encode_sequence(batch)
        layout = None
        if sequence.is_cuda and sequence.dtype in (torch.float16, torch.bfloat16) and flash_varlen() is not None:
            lengths = np.maximum(batch["history_lengths"], 1).astype(np.int32)
            indices = np.concatenate([np.arange(size) + feed * sequence.shape[1] for feed, size in enumerate(lengths)])
            layout = (
                torch.as_tensor(indices, device=sequence.device),
                torch.as_tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.int32, device=sequence.device),
                int(lengths.max()),
            )
        caches = []
        for block in self.blocks:
            sequence = block.advance_sequence(sequence, mask)
            caches.append(block.cache_history(sequence, mask, layout))
        return HistoryCache(tuple(caches))

    def forward(self, batch, history_cache=None, main_only=False):
        base = self.features.encode_nonsequential(batch)
        history_cache = history_cache if history_cache is not None else self.prepare_history_cache(batch)
        chunks = F.pad(base, (0, self.padding)).reshape(len(base), len(self.projections), self.chunk_width)
        weights = torch.stack([project.weight.t() for project in self.projections])
        biases = torch.stack([project.bias for project in self.projections])
        queries = torch.einsum("ntc,tcd->ntd", chunks, weights)
        queries = queries + biases.to(queries.dtype)
        feed_index = batch["candidate_feed"]
        offsets = torch.as_tensor(np.cumsum(np.r_[0, batch["counts"][:-1]]), device=base.device)
        candidate_position = torch.arange(len(base), device=base.device) - offsets[feed_index]
        cu_queries = None
        if history_cache.layers[0].flat_key is not None:
            cu_queries = torch.as_tensor(
                np.r_[0, np.cumsum(batch["counts"]) * len(self.projections)], dtype=torch.int32, device=base.device
            )
        for block, cache in zip(self.blocks, history_cache.layers):
            queries = block.read_cached_history(
                queries, cache, feed_index, candidate_position, int(np.max(batch["counts"])), cu_queries
            )
        mixer = self._compiled_final_mixer if self.training and self._compiled_final_mixer else self.final_mixer
        return self.heads(mixer(queries).flatten(1), main_only=main_only)


def reverse_valid(values, mask):
    positions = torch.arange(values.shape[1], device=values.device)
    indices = (mask.sum(dim=1, keepdim=True) - 1 - positions).clamp_min(0)
    return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])) * mask.unsqueeze(-1)


class QREmbedding(nn.Module):
    """Positive int32 IDs use unique (quotient, remainder) index pairs; zero is padding.

    Two shared codebooks sum to one vector. Unique index pairs do not imply mathematically unique
    output vectors or the capacity of a separate parameter vector for every item.
    """

    def __init__(self, dimension, base=65536):
        super().__init__()
        self.base = base
        self.quotient = nn.Embedding((2**31 - 1) // base + 2, dimension, padding_idx=0)
        self.remainder = nn.Embedding(base + 1, dimension, padding_idx=0)
        for table in (self.quotient, self.remainder):
            nn.init.normal_(table.weight, std=2**-0.5)
            with torch.no_grad():
                table.weight[0].zero_()

    def forward(self, ids):
        valid = ids > 0
        safe = ids.clamp_min(0)
        quotient = torch.where(valid, safe // self.base + 1, 0)
        remainder = torch.where(valid, safe % self.base + 1, 0)
        return self.quotient(quotient) + self.remainder(remainder)


class UnifiedFeatures(OneTransFeatures):
    def __init__(self, config):
        super().__init__(config)
        event_width = 128 + config.category_dim + config.event_dim + config.source_dim + config.item_id_dim
        self.event_mlp = nn.Sequential(
            nn.Linear(event_width, config.width), nn.GELU(), nn.Linear(config.width, config.width)
        )
        self._bins_ready = False
        self.time_embeddings = nn.ModuleList(nn.Embedding(count, 32) for count in (24, 60, 7, 2, 168))
        self.history_length = nn.Embedding(10, 32)
        self.register_buffer("history_length_edges", torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128]))
        self.ui_similarity = nn.Embedding(128, 32)
        self.item_id = QREmbedding(config.item_id_dim) if config.item_id_dim else None
        self.separator = nn.Parameter(torch.randn(config.width) * 0.02)
        self.dimensions = (
            5 * config.numeric_embedding_dim if config.numeric_bins else 5,
            config.category_dim,
            128,
            128,
            128,
            32,
            160,
            32,
            config.item_id_dim,
        )
        if config.numeric_bins and config.numeric_residual:
            self.dimensions = (*self.dimensions, 5)
        if config.numeric_bins:
            if config.numeric_bins < 2:
                raise ValueError("Numerical bucket count must be at least two")
            self.numeric = nn.ModuleList(
                nn.Embedding(config.numeric_bins, config.numeric_embedding_dim) for _ in range(5)
            )
            self.register_buffer("numeric_edges", torch.zeros(5, config.numeric_bins - 1))

    def set_numeric_bins(self, edges):
        values = torch.as_tensor(edges, device=self.numeric_edges.device, dtype=self.numeric_edges.dtype)
        if (
            values.shape != self.numeric_edges.shape
            or not torch.isfinite(values).all()
            or (values[:, 1:] < values[:, :-1]).any()
        ):
            raise ValueError("Invalid train-fitted numerical bucket boundaries")
        self.numeric_edges.copy_(values)
        self._bins_ready = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if prefix + "numeric_edges" in state_dict:
            self._bins_ready = True

    def scale_embedding(self, values):
        if not self.config.scale_embedding_fields:
            return values
        return values * (values.shape[-1] ** -0.5)

    def encode_nonsequential(self, batch):
        index = batch["candidate_feed"]
        user, _ = self.corrupt(F.normalize(batch["user"], dim=-1), self.config.user_embedding_dropout)
        user = user[index]
        item, _ = self.corrupt(F.normalize(batch["item"], dim=-1), self.config.item_embedding_dropout)
        if self.config.numeric_bins:
            if not self._bins_ready:
                raise ValueError("Fit numerical buckets on train before training or inference")
            numerical = torch.cat(
                [
                    self.scale_embedding(
                        embedding(
                            torch.bucketize(batch["numerical"][:, column].contiguous(), self.numeric_edges[column])
                        )
                    )
                    for column, embedding in enumerate(self.numeric)
                ],
                dim=-1,
            )
        else:
            numerical = batch["numerical"]
        length_bins = torch.bucketize(batch["history_mask"].sum(dim=1), self.history_length_edges)
        time = torch.cat(
            [
                self.scale_embedding(table(batch["time_bins"][:, column]))
                for column, table in enumerate(self.time_embeddings)
            ],
            dim=-1,
        )
        similarity_bins = ((user * item).sum(dim=-1).clamp(-1, 1).add(1) * 64).long().clamp_max(127)
        features = [
            numerical,
            self.scale_embedding(self.category(batch["category"])),
            user,
            item,
            user * item,
            self.scale_embedding(self.history_length(length_bins))[index],
            time[index],
            self.scale_embedding(self.ui_similarity(similarity_bins)),
        ]
        if self.item_id is not None:
            ids, _ = self.corrupt(self.item_id(batch["candidate_item_id"]), self.config.item_embedding_dropout)
            features.append(self.scale_embedding(ids))
        if self.config.numeric_bins and self.config.numeric_residual:
            features.append(batch["numerical"])
        return torch.cat(features, dim=-1)

    def encode_events(self, batch, return_items=False):
        item, _ = self.corrupt(F.normalize(batch["history_item"], dim=-1), self.config.item_embedding_dropout)
        features = [
            item,
            self.scale_embedding(self.category(batch["history_category"])),
            self.scale_embedding(self.event_type(batch["history_event_type"])),
            self.scale_embedding(self.event_source(batch["history_event_source"])),
        ]
        if self.item_id is not None:
            ids, _ = self.corrupt(self.item_id(batch["history_item_id"]), self.config.item_embedding_dropout)
            features.append(self.scale_embedding(ids))
        if self.config.history_position_dim:
            # Published history order is newest first; these are ranks, not elapsed times.
            mask = batch["history_mask"]
            positions = torch.arange(1, mask.shape[1] + 1, device=item.device)[None].expand_as(mask)
            positions = positions.masked_fill(~mask, 0)
            features.append(self.scale_embedding(self.history_position(positions)))
        mlp = self._compiled_event_mlp if self.training and self._compiled_event_mlp is not None else self.event_mlp
        events = mlp(torch.cat(features, dim=-1)) * batch["history_mask"].unsqueeze(-1)
        return (events, item) if return_items else events

    def encode_sequence(self, batch):
        events, mask = self.encode_events(batch), batch["history_mask"]
        separator = self.separator.to(events.dtype).expand(events.shape[0], 1, -1)
        valid_separator = torch.ones(mask.shape[0], 1, device=mask.device, dtype=torch.bool)
        return torch.cat((separator, events), dim=1), torch.cat((valid_separator, mask), dim=1)


def _flash_forward(q, k, v, cu_q, cu_k, max_q, max_k, dropout, causal):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_forward

    out, lse, _, rng = _wrapped_flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        dropout,
        q.shape[-1] ** -0.5,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
    )
    return out, lse, rng


def _flash_backward(dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, dropout, causal, rng):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward

    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    _wrapped_flash_attn_varlen_backward(
        dout.contiguous(),
        q,
        k,
        v,
        out,
        lse.contiguous(),
        dq,
        dk,
        dv,
        cu_q,
        cu_k,
        max_q,
        max_k,
        dropout,
        q.shape[-1] ** -0.5,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
        rng_state=rng,
        zero_tensors=True,
    )
    return dq, dk, dv


def _reverse_query(values, tokens):
    return values.reshape(-1, tokens, *values.shape[1:]).flip(1).reshape_as(values).contiguous()


class UnifiedCachedAttention(torch.autograd.Function):
    """Exact joint softmax over query-prefix keys and shared historical keys.

    The two FlashAttention tiles use one global log-normalizer. Backward supplies that global
    normalizer AND the global output to both kernels, so gradients include the normalization
    interaction between query and history keys. This is not a sum of separate attentions.
    """

    query_candidates_per_tile = 8192

    @staticmethod
    def forward(ctx, q, history_k, history_v, query_k, query_v, cu_q, cu_k, max_q, max_k, tokens, dropout, has_history):
        count, heads, width = q.shape
        history_out, history_lse, history_rng = _flash_forward(
            q, history_k, history_v, cu_q, cu_k, max_q, max_k, dropout, False
        )
        chunk_size = UnifiedCachedAttention.query_candidates_per_tile * tokens
        query_outputs, query_normalizers, query_states = [], [], []
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            cu_self = torch.arange(0, stop - start + 1, tokens, device=q.device, dtype=torch.int32)
            tile_out, tile_lse, tile_rng = _flash_forward(
                _reverse_query(q[start:stop], tokens),
                _reverse_query(query_k[start:stop], tokens),
                _reverse_query(query_v[start:stop], tokens),
                cu_self,
                cu_self,
                tokens,
                tokens,
                dropout,
                True,
            )
            query_outputs.append(_reverse_query(tile_out, tokens))
            query_normalizers.append(tile_lse.reshape(heads, -1, tokens).flip(-1).reshape(heads, stop - start))
            query_states.append(tile_rng)
        query_out = torch.cat(query_outputs)
        query_lse = torch.cat(query_normalizers, dim=1).contiguous()
        query_rng = torch.stack(query_states)
        history_lse = torch.where(has_history.unsqueeze(0), history_lse, -torch.inf)
        lse = torch.logaddexp(history_lse, query_lse)
        weight = torch.exp(history_lse - lse).transpose(0, 1).unsqueeze(-1)
        output = (weight * history_out.float() + (1 - weight) * query_out.float()).to(q.dtype)
        ctx.save_for_backward(
            q,
            history_k,
            history_v,
            query_k,
            query_v,
            output,
            lse,
            cu_q,
            cu_k,
            history_rng,
            query_rng,
            has_history,
        )
        ctx.options = max_q, max_k, tokens, dropout
        return output

    @staticmethod
    def backward(ctx, dout):
        q, hk, hv, qk, qv, out, lse, cu_q, cu_k, hrng, qrng, valid = ctx.saved_tensors
        max_q, max_k, tokens, dropout = ctx.options
        dq_history, dhk, dhv = _flash_backward(
            dout * valid[:, None, None], q, hk, hv, out, lse, cu_q, cu_k, max_q, max_k, dropout, False, hrng
        )
        if os.environ.get("AVITO_ANOMALY") == "1" and any(not torch.isfinite(g).all() for g in (dq_history, dhk, dhv)):
            raise FloatingPointError("Nonfinite shared-history attention gradients")
        chunk_size = UnifiedCachedAttention.query_candidates_per_tile * tokens
        query_gradients, key_gradients, value_gradients = [], [], []
        for tile, start in enumerate(range(0, len(q), chunk_size)):
            stop = min(start + chunk_size, len(q))
            cu_self = torch.arange(0, stop - start + 1, tokens, device=q.device, dtype=torch.int32)
            local_lse = lse[:, start:stop]
            reverse_lse = local_lse.reshape(lse.shape[0], -1, tokens).flip(-1).reshape_as(local_lse).contiguous()
            tile_gradients = _flash_backward(
                _reverse_query(dout[start:stop], tokens),
                _reverse_query(q[start:stop], tokens),
                _reverse_query(qk[start:stop], tokens),
                _reverse_query(qv[start:stop], tokens),
                _reverse_query(out[start:stop], tokens),
                reverse_lse,
                cu_self,
                cu_self,
                tokens,
                tokens,
                dropout,
                True,
                qrng[tile],
            )
            if os.environ.get("AVITO_ANOMALY") == "1" and any(not torch.isfinite(g).all() for g in tile_gradients):
                raise FloatingPointError("Nonfinite query-prefix attention gradients")
            for target, gradient in zip((query_gradients, key_gradients, value_gradients), tile_gradients):
                target.append(_reverse_query(gradient, tokens))
        return (
            dq_history + torch.cat(query_gradients),
            dhk,
            dhv,
            torch.cat(key_gradients),
            torch.cat(value_gradients),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class UnifiedOneTransBlock(OneTransBlock):
    """One right-causal self-attention layer on [query tokens; newest-first historical tokens]."""

    def __init__(self, config):
        super().__init__(config)
        del self.context_norm, self.context_kv
        self.query_k = PerQueryLinear(config.tokens, config.width)
        self.query_v = PerQueryLinear(config.tokens, config.width)

    def _make_cache(self, k, v, mask, layout):
        cache = HistoryKV(self.heads_view(k), self.heads_view(v), mask, has_history=mask.any(dim=1))
        if layout is not None:
            indices, cache.cu_lengths, cache.max_length = layout
            shape = (-1, self.heads, self.width // self.heads)
            cache.flat_key = k.reshape(shape).index_select(0, indices)
            cache.flat_value = v.reshape(shape).index_select(0, indices)
        return cache

    def advance_sequence_with_cache(self, sequence, mask, layout=None):
        # K/V belong to the INPUT of this same unified attention layer, before its residual/FFN.
        projected = reverse_valid(self.sequence_qkv(self.sequence_norm(sequence)), mask)
        q, k, v = projected.chunk(3, dim=-1)
        cache = self._make_cache(k, v, mask, layout)
        output = (
            F.scaled_dot_product_attention(
                self.heads_view(q),
                self.heads_view(k),
                self.heads_view(v),
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
            )
            .transpose(1, 2)
            .flatten(2)
        )
        sequence = sequence + self.dropout(self.sequence_out(reverse_valid(output, mask)))
        ffn = self._compiled_sequence_ffn if self.training and self._compiled_sequence_ffn else self.sequence_ffn
        sequence = sequence + self.dropout(ffn(self.sequence_ffn_norm(sequence)))
        return sequence * mask.unsqueeze(-1), cache

    def advance_sequence(self, sequence, mask):
        return self.advance_sequence_with_cache(sequence, mask)[0]

    def cache_history(self, sequence, mask, layout=None):
        _, k, v = reverse_valid(self.sequence_qkv(self.sequence_norm(sequence)), mask).chunk(3, dim=-1)
        return self._make_cache(k, v, mask, layout)

    def read_cached_history(self, queries, cache, feed_index, candidate_position, max_candidates, cu_queries=None):
        count, tokens, width = queries.shape
        normalized = self.query_norm(queries)
        qkv = torch.einsum(
            "ntd,tdh->nth",
            normalized,
            torch.cat((self.query_q.weight, self.query_k.weight, self.query_v.weight), dim=-1),
        )
        q, k, v = qkv.chunk(3, dim=-1)
        dropout = self.attention_dropout if self.training else 0.0
        if cache.flat_key is not None and cu_queries is not None and width // self.heads % 8 == 0:
            shape = (-1, self.heads, width // self.heads)
            has_history = cache.has_history[feed_index].repeat_interleave(tokens)
            output = UnifiedCachedAttention.apply(
                q.reshape(shape),
                cache.flat_key,
                cache.flat_value,
                k.reshape(shape),
                v.reshape(shape),
                cu_queries,
                cache.cu_lengths,
                max_candidates * tokens,
                cache.max_length,
                tokens,
                dropout,
                has_history,
            ).reshape_as(queries)
        else:
            # Literal [query-prefix; history] self-attention reference, also used for FP32 export.
            query_k, query_v = self.heads_view(k), self.heads_view(v)
            keys = torch.cat((query_k, cache.key[feed_index]), dim=2)
            values = torch.cat((query_v, cache.value[feed_index]), dim=2)
            causal = torch.ones(tokens, tokens, device=q.device, dtype=torch.bool).triu()
            visible = torch.cat(
                (causal[None].expand(count, -1, -1), cache.valid[feed_index, None, :].expand(-1, tokens, -1)), dim=-1
            )
            output = (
                F.scaled_dot_product_attention(
                    self.heads_view(q),
                    keys,
                    values,
                    attn_mask=visible[:, None],
                    dropout_p=dropout,
                )
                .transpose(1, 2)
                .flatten(2)
            )
        queries = queries + self.dropout(self.query_out(output))
        ffn = self._compiled_query_ffn if self.training and self._compiled_query_ffn else self.query_ffn
        return queries + self.dropout(ffn(self.query_ffn_norm(queries)))

    def forward(self, queries, sequence, mask, feed_index, candidate_position, max_candidates):
        sequence, cache = self.advance_sequence_with_cache(sequence, mask)
        return self.read_cached_history(queries, cache, feed_index, candidate_position, max_candidates), sequence


class UnifiedOneTrans(OneTrans):
    features_type = UnifiedFeatures
    block_type = UnifiedOneTransBlock

    def prepare_history_cache(self, batch):
        sequence, mask = self.features.encode_sequence(batch)
        layout = None
        if (
            sequence.is_cuda
            and sequence.dtype in (torch.float16, torch.bfloat16)
            and self.blocks[0].width // self.blocks[0].heads % 8 == 0
            and flash_varlen() is not None
        ):
            lengths = (batch["history_lengths"] + 1).astype(np.int32)
            indices = np.concatenate(
                [np.arange(size - 1, -1, -1) + feed * sequence.shape[1] for feed, size in enumerate(lengths)]
            )
            flat = sequence.flatten(0, 1).index_select(0, torch.as_tensor(indices, device=sequence.device))
            cu = torch.as_tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.int32, device=sequence.device)
            maximum = int(lengths.max())
            caches = []
            for block in self.blocks:
                qkv = block.sequence_qkv(block.sequence_norm(flat)).reshape(
                    -1, 3, block.heads, block.width // block.heads
                )
                q, k, v = qkv.unbind(dim=1)
                caches.append(
                    HistoryKV(
                        None,
                        None,
                        mask,
                        flat_key=k,
                        flat_value=v,
                        cu_lengths=cu,
                        max_length=maximum,
                        has_history=mask.any(dim=1),
                    )
                )
                attended = flash_varlen()(
                    q,
                    k,
                    v,
                    cu,
                    cu,
                    maximum,
                    maximum,
                    dropout_p=block.attention_dropout if self.training else 0.0,
                    causal=True,
                )
                flat = flat + block.dropout(block.sequence_out(attended.flatten(1)))
                ffn = (
                    block._compiled_sequence_ffn
                    if self.training and block._compiled_sequence_ffn
                    else block.sequence_ffn
                )
                flat = flat + block.dropout(ffn(block.sequence_ffn_norm(flat)))
            return HistoryCache(tuple(caches))
        layers = []
        for block in self.blocks:
            sequence, cache = block.advance_sequence_with_cache(sequence, mask, layout)
            layers.append(cache)
        return HistoryCache(tuple(layers))
