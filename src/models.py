"""Concat feature base -> TokenMixer -> task tower, with an MLP baseline.

TokenMixer core follows RankMixer equations (1), (3)-(7): https://arxiv.org/html/2507.15551v3
The concat tokenizer, shared event-MLP/mean-pooling encoder and task tower are Avito adaptations.
"""

from dataclasses import dataclass
from functools import wraps
import math
import threading

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


_COMPILED_CALL_LOCK = threading.RLock()


def compile_shared(function):
    """Serialize compiler entry across replicas.

    FX tracing state is process-global in PyTorch 2.5. A new candidate chunk
    shape can recompile after the first batch. Protect every compiled entry
    rather than only warming feed/history shapes; CUDA execution stays async.
    """
    compiled = torch.compile(function, dynamic=True)

    @wraps(function)
    def call(*args, **kwargs):
        with _COMPILED_CALL_LOCK:
            return compiled(*args, **kwargs)

    return call


@dataclass
class ModelConfig:
    name: str = "tokenmixer"
    hidden: tuple = (256, 128)
    category_dim: int = 16
    event_dim: int = 8
    source_dim: int = 8
    dropout: float = 0.1
    width: int = 128
    depth: int = 3
    expansion: int = 4
    tokens: int = 8
    user_embedding_dropout: float = 0.1
    item_embedding_dropout: float = 0.1
    embedding_corruption: str = "zero"
    attention_heads: int = 8
    attention_layout: str = "cross"
    numeric_bins: int = 0
    numeric_embedding_dim: int = 32
    numeric_residual: bool = False
    relative_numeric: bool = False
    scale_embedding_fields: bool = False
    similarity_residual: bool = False
    history_position_dim: int = 0
    recent_history_window: int = 0
    fast_execution: bool = False
    item_id_dim: int = 0
    pairwise_tower: bool = False
    token_ffn_activation: str = "gelu"
    din: bool = False
    din_softmax_context: bool = False
    din_query_context: bool = False
    lcfn: bool = False
    lcfn_embedding_dim: int = 32
    # Older LCFN checkpoints contain both fields. Fresh CLI runs default to category only.
    lcfn_fields: tuple = ("category", "item")
    din_attention_dim: int = 32
    din_chunk_size: int = 2048
    din_history_bucket: int = 0
    din_recompute: str = "on"
    block_residual: bool = False
    checkpoint_mixer: bool = False
    candidate_context_depth: int = 0
    bf16_mixer_norm: bool = False
    mixer_norm: str = "layernorm"
    mixer_norm_backend: str = "flash_attn_triton"

    def __post_init__(self):
        self.lcfn_fields = tuple(self.lcfn_fields)


class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._compiled_event_mlp = None
        self.category = nn.Embedding(54, config.category_dim, padding_idx=0)
        self.event_type = nn.Embedding(11, config.event_dim, padding_idx=0)
        self.event_source = nn.Embedding(31, config.source_dim, padding_idx=0)
        self.event_mlp = nn.Sequential(
            nn.Linear(129 + config.category_dim + config.event_dim + config.source_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.dimensions = (8, config.category_dim, 128, 128, 128, 128, 128, 128)

    def corrupt(self, vectors, probability):
        selected = torch.zeros(vectors.shape[:-1], dtype=torch.bool, device=vectors.device)
        if self.training and probability:
            selected = (torch.rand(vectors.shape[:-1], device=vectors.device) < probability) & vectors.ne(0).any(dim=-1)
            replacement = (
                torch.zeros_like(vectors)
                if self.config.embedding_corruption == "zero"
                else F.normalize(torch.randn_like(vectors), dim=-1)
            )
            vectors = torch.where(selected.unsqueeze(-1), replacement, vectors)
        return vectors, selected if self.config.embedding_corruption == "zero" else torch.zeros_like(selected)

    def encode_events(self, batch):
        # Embedding tables and the position-wise MLP are shared across all sequence positions.
        item, missing = self.corrupt(F.normalize(batch["history_item"], dim=-1), self.config.item_embedding_dropout)
        events = torch.cat(
            (
                item,
                self.category(batch["history_category"]),
                self.event_type(batch["history_event_type"]),
                self.event_source(batch["history_event_source"]),
                torch.maximum(batch["history_item_missing"], missing.unsqueeze(-1).float()),
            ),
            dim=-1,
        )
        mask = batch["history_mask"].unsqueeze(-1)
        event_mlp = (
            self._compiled_event_mlp if self.training and self._compiled_event_mlp is not None else self.event_mlp
        )
        return event_mlp(events) * mask

    def encode_history(self, batch):
        events = self.encode_events(batch)
        return events.sum(dim=1) / batch["history_mask"].sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, batch):
        index = batch["candidate_feed"]
        user, _ = self.corrupt(F.normalize(batch["user"], dim=-1), self.config.user_embedding_dropout)
        user = user[index]
        history = self.encode_history(batch)[index]
        item, missing = self.corrupt(F.normalize(batch["item"], dim=-1), self.config.item_embedding_dropout)
        dense = torch.cat(
            (
                batch["numerical"],
                torch.maximum(batch["item_missing"], missing.unsqueeze(-1).float()),
                batch["history_stats"][index],
            ),
            dim=-1,
        )
        return torch.cat(
            (
                dense,
                self.category(batch["category"]),
                user,
                item,
                history,
                user * item,
                history * item,
                user * history,
            ),
            dim=-1,
        )


def task_tower(input_width, config):
    layers, current = [], input_width
    for width in config.hidden:
        layers.extend((nn.Linear(current, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(config.dropout)))
        current = width
    return TaskTower(*layers, nn.Linear(current, 1))


class TaskTower(nn.Sequential):
    def forward(self, representation):
        for layer in list(self.children())[:-1]:
            representation = layer(representation)
        # Keep ranking logits in FP32; BF16 output rounding otherwise creates avoidable score ties.
        with torch.autocast(representation.device.type, enabled=False):
            return self[-1](representation.float())


class TaskHeads(nn.Module):
    def __init__(self, input_width, config):
        super().__init__()
        self.main = task_tower(input_width, config)
        self.aux = task_tower(input_width, config)
        if config.pairwise_tower:
            self.pairwise = task_tower(input_width, config)

    def forward(self, representation, main_only=False):
        outputs = {"main": self.main(representation).squeeze(-1)}
        if main_only:
            return outputs
        outputs["aux"] = self.aux(representation).squeeze(-1)
        if hasattr(self, "pairwise"):
            outputs["pairwise"] = self.pairwise(representation).squeeze(-1)
        return outputs


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = FeatureEncoder(config)
        self.heads = TaskHeads(sum(self.features.dimensions), config)

    def forward(self, batch, main_only=False):
        return self.heads(self.features(batch), main_only=main_only)


def token_mix(tokens):
    batch, count, width = tokens.shape
    if width % count:
        raise ValueError("TokenMixer width must be divisible by token count (H=T)")
    return tokens.reshape(batch, count, count, width // count).transpose(1, 2).reshape(batch, count, width)


def bias_gelu(hidden, bias):
    return F.gelu(hidden + bias.to(hidden.dtype))


def bias_gelu_dropout_backward(gradient, mask, hidden, bias, scale):
    # Keep the original operators and dtype boundaries. Inductor can fuse the
    # deterministic backward without replacing PyTorch's dropout RNG/mask.
    dropped_gradient = torch.ops.aten.native_dropout_backward.default(gradient, mask, scale)
    preactivation = hidden + bias.to(hidden.dtype)
    hidden_gradient = torch.ops.aten.gelu_backward.default(dropped_gradient, preactivation, approximate="none")
    return hidden_gradient, hidden_gradient.sum(0).to(bias.dtype)


class _BiasGELUDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, bias, probability, forward_function, backward_function):
        activated = forward_function(hidden, bias)
        output, mask = torch.ops.aten.native_dropout.default(activated, probability, True)
        ctx.save_for_backward(hidden, bias, mask)
        ctx.scale = 1.0 / (1.0 - probability) if probability < 1.0 else 0.0
        ctx.backward_function = backward_function
        return output

    @staticmethod
    def backward(ctx, gradient):
        hidden, bias, mask = ctx.saved_tensors
        hidden_gradient, bias_gradient = ctx.backward_function(gradient, mask, hidden, bias, ctx.scale)
        return hidden_gradient, bias_gradient, None, None, None


class PerTokenFFN(nn.Module):
    def __init__(self, tokens, width, expansion, dropout, activation="gelu"):
        super().__init__()
        if activation not in {"gelu", "swiglu"}:
            raise ValueError("Token FFN activation must be gelu or swiglu")
        self.activation = activation
        hidden = width * expansion if activation == "gelu" else math.ceil(width * expansion * 2 / 3 / 64) * 64
        self.w1 = nn.Parameter(torch.empty(tokens, width, hidden))
        self.w2 = nn.Parameter(torch.empty(tokens, hidden, width))
        self.b1 = nn.Parameter(torch.zeros(tokens, hidden))
        self.b2 = nn.Parameter(torch.zeros(tokens, width))
        if activation == "swiglu":
            self.w_gate = nn.Parameter(torch.empty(tokens, width, hidden))
            self.b_gate = nn.Parameter(torch.zeros(tokens, hidden))
        self.dropout = nn.Dropout(dropout)
        self._compiled_up = None
        self._compiled_down = None
        self._compiled_dropout_forward = None
        self._compiled_dropout_backward = None
        for token in range(tokens):
            nn.init.xavier_uniform_(self.w1[token])
            nn.init.xavier_uniform_(self.w2[token])
            if activation == "swiglu":
                nn.init.xavier_uniform_(self.w_gate[token])

    def input_activation(self, tokens):
        if self.activation == "swiglu":
            packed = torch.einsum("btd,tdh->bth", tokens, torch.cat((self.w1, self.w_gate), dim=-1))
            hidden, gate = packed.chunk(2, dim=-1)
            hidden = F.silu(gate + self.b_gate.to(gate.dtype)) * (hidden + self.b1.to(hidden.dtype))
        else:
            hidden = torch.einsum("btd,tdh->bth", tokens, self.w1)
            hidden = hidden + self.b1.to(hidden.dtype)
            hidden = F.gelu(hidden)
        return hidden

    def output_projection(self, hidden):
        output = torch.einsum("bth,thd->btd", hidden, self.w2)
        return output + self.b2.to(output.dtype)

    def forward(self, tokens):
        up = self._compiled_up if self.training and self._compiled_up is not None else self.input_activation
        down = self._compiled_down if self.training and self._compiled_down is not None else self.output_projection
        if self.training and self._compiled_dropout_backward is not None:
            hidden = torch.einsum("btd,tdh->bth", tokens, self.w1)
            dropped = _BiasGELUDropout.apply(
                hidden, self.b1, self.dropout.p, self._compiled_dropout_forward, self._compiled_dropout_backward
            )
            return down(dropped)
        return down(self.dropout(up(tokens)))


class TokenMixerBlock(nn.Module):
    def __init__(self, tokens, config):
        super().__init__()
        self.ffn = PerTokenFFN(tokens, config.width, config.expansion, config.dropout, config.token_ffn_activation)
        norm = nn.RMSNorm if config.mixer_norm == "rmsnorm" else nn.LayerNorm
        self.mix_norm = norm(config.width, eps=1e-5)
        self.ffn_norm = norm(config.width, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)
        self.block_residual = config.block_residual
        self.bf16_mixer_norm = config.bf16_mixer_norm
        self._bf16_normalize = None
        self._fused_mix_normalize = None
        if self.bf16_mixer_norm:
            if config.mixer_norm_backend in {"parallel_triton", "parallel_triton_fused_mix"}:
                from .mixer_norm import mix_rms_norm, rms_norm

                self._bf16_normalize = rms_norm
                if config.mixer_norm_backend == "parallel_triton_fused_mix":
                    self._fused_mix_normalize = mix_rms_norm
            else:
                from .onetrans import prepare_fused_norm

                self._bf16_normalize = prepare_fused_norm(rms=config.mixer_norm == "rmsnorm")

    def normalize(self, norm, tokens):
        # Keep FP32 affine parameters/reductions and BF16 training activations.
        # Evaluation uses the corresponding native FP32 LayerNorm or RMSNorm.
        if self.training and self.bf16_mixer_norm and tokens.is_cuda and tokens.dtype == torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                return self._bf16_normalize(tokens, norm.weight, getattr(norm, "bias", None), eps=norm.eps)
        return norm(tokens)

    def forward(self, tokens):
        residual = tokens
        if (
            self.training
            and self._fused_mix_normalize is not None
            and tokens.is_cuda
            and tokens.dtype == torch.bfloat16
        ):
            with torch.autocast("cuda", enabled=False):
                tokens = self._fused_mix_normalize(tokens, self.mix_norm.weight, eps=self.mix_norm.eps)
        else:
            tokens = self.normalize(self.mix_norm, tokens + token_mix(tokens))
        tokens = self.normalize(self.ffn_norm, tokens + self.dropout(self.ffn(tokens)))
        return residual + tokens if self.block_residual else tokens

    def forward_checkpointed(self, tokens):
        return checkpoint(self.forward, tokens, use_reentrant=False)


class CandidateContextBlock(nn.Module):
    """Bidirectional candidate attention inside each feed, without position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.width = config.width
        self.heads = config.attention_heads
        self.attention_norm = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, config.width * config.expansion),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width * config.expansion, config.width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values, cu_lengths, maximum, groups, flash):
        qkv = self.qkv(self.attention_norm(values)).reshape(-1, 3, self.heads, self.width // self.heads)
        query, key, value = qkv.unbind(dim=1)
        dropout = self.dropout.p if self.training else 0.0
        if cu_lengths is not None:
            attended = flash(
                query, key, value, cu_lengths, cu_lengths, maximum, maximum, dropout_p=dropout, causal=False
            )
        else:
            outputs = []
            for indices, valid in groups:
                local = F.scaled_dot_product_attention(
                    query[indices].transpose(1, 2),
                    key[indices].transpose(1, 2),
                    value[indices].transpose(1, 2),
                    attn_mask=valid[:, None, None, :],
                    dropout_p=dropout,
                )
                # Boolean indexing restores real candidates in the original feed/list order.
                outputs.append(local.transpose(1, 2)[valid])
            attended = torch.cat(outputs, dim=0)
        values = values + self.dropout(self.output(attended.reshape(-1, self.width)))
        return values + self.dropout(self.ffn(self.ffn_norm(values)))


class CandidateContext(nn.Module):
    def __init__(self, input_width, config):
        super().__init__()
        from .onetrans import flash_varlen

        # Resolve the optional backend during construction, before replica worker threads run.
        self.flash = flash_varlen()
        self.projection = nn.Linear(input_width, config.width)
        self.blocks = nn.ModuleList(CandidateContextBlock(config) for _ in range(config.candidate_context_depth))

    def forward(self, representation, counts):
        counts = np.asarray(counts, dtype=np.int64)
        if not len(counts) or counts.min() <= 0 or counts.sum() != len(representation):
            raise ValueError("Candidate context requires complete nonempty feeds")
        values = self.projection(representation)
        offsets = np.r_[0, np.cumsum(counts)]
        maximum = int(counts.max())
        cu_lengths, groups = None, None
        if self.flash is not None and values.is_cuda and values.dtype in (torch.float16, torch.bfloat16):
            cu_lengths = torch.as_tensor(offsets, dtype=torch.int32, device=values.device)
        else:
            # FP32 evaluation retains the full candidate set; padding is bounded to 128 feeds.
            groups = []
            for first in range(0, len(counts), 128):
                last = min(first + 128, len(counts))
                columns = np.arange(int(counts[first:last].max()))
                valid = columns[None] < counts[first:last, None]
                indices = np.minimum(offsets[first:last, None] + columns[None], len(values) - 1)
                groups.append(
                    (
                        torch.as_tensor(indices, device=values.device),
                        torch.as_tensor(valid, device=values.device),
                    )
                )
        for block in self.blocks:
            values = block(values, cu_lengths, maximum, groups, self.flash)
        return values


class TokenMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.din:
            from .din import DINFeatures

            self.features = DINFeatures(config)
        else:
            self.features = FeatureEncoder(config)
        self._compiled_blocks = None
        self.fast_execution = config.fast_execution
        self.checkpoint_mixer = config.checkpoint_mixer
        count = config.tokens
        if count <= 0 or config.width <= 0 or config.width % count or config.depth <= 0 or config.expansion <= 0:
            raise ValueError("TokenMixer needs positive dimensions and width divisible by token count")
        self.base_width = sum(self.features.dimensions)
        self.chunk_width = (self.base_width + count - 1) // count
        self.padding = count * self.chunk_width - self.base_width
        self.projections = nn.ModuleList(nn.Linear(self.chunk_width, config.width) for _ in range(count))
        self.blocks = nn.Sequential(*(TokenMixerBlock(count, config) for _ in range(config.depth)))
        self.candidate_context = (
            CandidateContext(count * config.width, config) if config.candidate_context_depth else None
        )
        head_width = count * config.width + (config.width if self.candidate_context is not None else 0)
        self.heads = TaskHeads(head_width, config)

    def forward(self, batch, main_only=False):
        base = F.pad(self.features(batch), (0, self.padding))
        chunks = base.reshape(len(base), len(self.projections), self.chunk_width)
        if self.fast_execution:
            weights = torch.stack([project.weight.t() for project in self.projections])
            biases = torch.stack([project.bias for project in self.projections])
            tokens = torch.einsum("ntc,tcd->ntd", chunks, weights)
            tokens = tokens + biases.to(tokens.dtype)
        else:
            tokens = torch.stack([project(chunks[:, index]) for index, project in enumerate(self.projections)], dim=1)
        blocks = self._compiled_blocks if self.training and self._compiled_blocks is not None else self.blocks
        if self.training and self.checkpoint_mixer:
            for block in self.blocks:
                tokens = block.forward_checkpointed(tokens)
        else:
            tokens = blocks(tokens)
        representation = tokens.flatten(start_dim=1)
        if self.candidate_context is not None:
            context = self.candidate_context(representation, batch["counts"])
            representation = torch.cat((representation, context), dim=-1)
        return self.heads(representation, main_only=main_only)


def enable_fast_execution(model):
    """Select parameter-compatible TokenMixer kernels without changing checkpoint tensors."""
    if not isinstance(model, TokenMixer):
        raise ValueError("Fast execution currently supports TokenMixer")
    model.fast_execution = True
    if hasattr(model.features, "fast_execution"):
        model.features.fast_execution = True


def compile_for_training(model, mixer_ffn_kernels="native", din_activation_kernels="native"):
    from .onetrans import OneTrans

    # Keep optimized native embedding backward; generic compiled scatter can be slow on tiny vocabularies.
    model.features._compiled_event_mlp = compile_shared(model.features.event_mlp.forward)
    if isinstance(model, TokenMixer):
        # PyTorch 2.5 failed both eager-checkpoint/compiled-body resume and
        # compiled-checkpoint partitioning. Keep grouping and recomputed regions
        # native; independent deterministic activation islands avoid those combinations.
        native_blocks = model.checkpoint_mixer or model.features.config.din or mixer_ffn_kernels != "native"
        model._compiled_blocks = None if native_blocks else compile_shared(model.blocks.forward)
        if mixer_ffn_kernels != "native":
            if model.checkpoint_mixer:
                raise ValueError("Compiled Mixer FFN requires stored Mixer activations")
            for block in model.blocks:
                # Random dropout still uses PyTorch's native mask generation.
                if mixer_ffn_kernels == "compiled_dropout":
                    if block.ffn.activation != "gelu":
                        raise ValueError("Fused dropout backward currently requires GELU")
                    block.ffn._compiled_dropout_forward = compile_shared(bias_gelu)
                    block.ffn._compiled_dropout_backward = compile_shared(bias_gelu_dropout_backward)
                else:
                    block.ffn._compiled_up = compile_shared(block.ffn.input_activation)
                block.ffn._compiled_down = compile_shared(block.ffn.output_projection)
        if din_activation_kernels == "compiled":
            if not model.features.config.din or model.features.config.din_recompute != "off":
                raise ValueError("Compiled DIN activation requires DIN with stored activations")
            model.features._compiled_group_activation = compile_shared(model.features.group_activation)
    elif isinstance(model, OneTrans):
        for block in model.blocks:
            block._compiled_sequence_ffn = compile_shared(block.sequence_ffn.forward)
            block._compiled_query_ffn = compile_shared(block.query_ffn.forward)
        model._compiled_final_mixer = compile_shared(model.final_mixer.forward)
    return model


def build_model(config):
    from .onetrans import OneTrans, UnifiedOneTrans

    if config.mixer_norm not in {"layernorm", "rmsnorm"}:
        raise ValueError("Mixer normalization must be layernorm or rmsnorm")
    if config.mixer_norm_backend not in {"native", "flash_attn_triton", "parallel_triton", "parallel_triton_fused_mix"}:
        raise ValueError("Unknown Mixer normalization backend")
    if config.mixer_norm_backend in {"parallel_triton", "parallel_triton_fused_mix"} and (
        config.mixer_norm != "rmsnorm" or not config.bf16_mixer_norm or config.width > 1024
    ):
        raise ValueError("Parallel RMSNorm requires BF16 Mixer RMSNorm with width <= 1024")
    if config.fast_execution and config.name != "tokenmixer":
        raise ValueError("Fast execution currently supports TokenMixer")
    if config.bf16_mixer_norm and config.name != "tokenmixer":
        raise ValueError("BF16 mixer normalization currently supports TokenMixer")
    if config.candidate_context_depth < 0 or (
        config.candidate_context_depth
        and (config.name != "tokenmixer" or config.attention_heads <= 0 or config.width % config.attention_heads)
    ):
        raise ValueError("Candidate context requires TokenMixer, nonnegative depth and heads dividing the width")

    if config.din and (config.name != "tokenmixer" or min(config.din_attention_dim, config.din_chunk_size) <= 0):
        raise ValueError("DIN requires TokenMixer and positive attention/chunk dimensions")
    if config.din_softmax_context and not config.din:
        raise ValueError("Normalized DIN context requires DIN")
    if config.din_query_context and not config.din:
        raise ValueError("Contextual DIN query requires DIN")
    if config.lcfn_embedding_dim <= 0 or (config.lcfn and not config.din):
        raise ValueError("LCFN requires DIN and a positive embedding dimension")
    if (
        not config.lcfn_fields
        or len(set(config.lcfn_fields)) != len(config.lcfn_fields)
        or any(field not in {"category", "item"} for field in config.lcfn_fields)
    ):
        raise ValueError("LCFN fields must be distinct category/item fields")
    if config.din_recompute not in {"on", "off"}:
        raise ValueError("DIN recomputation must be on or off")
    if not 0 <= config.din_history_bucket <= 256 or (
        config.din_history_bucket and not (config.din and config.fast_execution)
    ):
        raise ValueError("DIN history bucketing requires fast DIN and a bucket size in [0, 256]")
    if config.similarity_residual and not config.din:
        raise ValueError("Similarity residual requires DIN")
    if config.relative_numeric and not (
        config.din and config.numeric_bins > 0 and config.numeric_residual and config.similarity_residual
    ):
        raise ValueError("Relative numeric features require DIN with numeric and similarity residuals")
    if config.history_position_dim < 0 or (config.history_position_dim and not config.din):
        raise ValueError("History position embeddings require DIN and a nonnegative dimension")
    if not 0 <= config.recent_history_window <= 256 or (
        config.recent_history_window and not config.similarity_residual
    ):
        raise ValueError("Recent history window requires similarity residual and must be in [0, 256]")

    if (
        min(config.category_dim, config.event_dim, config.source_dim) <= 0
        or any(size <= 0 for size in config.hidden)
        or not 0 <= config.dropout < 1
    ):
        raise ValueError("Invalid model dimensions/dropout")
    if (
        not 0 <= config.user_embedding_dropout <= 1
        or not 0 <= config.item_embedding_dropout <= 1
        or config.embedding_corruption not in {"zero", "random"}
    ):
        raise ValueError("Invalid user/item embedding corruption settings")
    if config.attention_layout not in {"cross", "unified_prefix"}:
        raise ValueError("Unknown attention layout")
    models = {
        "mlp": MLP,
        "tokenmixer": TokenMixer,
        "onetrans": UnifiedOneTrans if config.attention_layout == "unified_prefix" else OneTrans,
    }
    return models[config.name](config)
