"""Stride-aware RMSNorm kernels for narrow, large-row-count Mixer activations.

Input and output gradients may use different batch/token layouts. Address both
through their strides instead of materializing a contiguous matrix. Each program
handles a small row tile; affine gradients use FP32 partials and a final reduction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_forward(
    X,
    W,
    Y,
    RSTD,
    M,
    INNER,
    SX0,
    SX1,
    SX2,
    N: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    Z,
    MIX: tl.constexpr,
    TOKENS: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = (rows // INNER) * SX0 + (rows % INNER) * SX1
    x = tl.load(X + offsets[:, None] + cols[None, :] * SX2, valid, 0).to(tl.float32)
    if MIX:
        # P(x)[batch, token, other * chunk + lane] = x[batch, other, token * chunk + lane].
        if TOKEN_MAJOR:
            token, batch = rows // INNER, rows % INNER
            batch_stride, token_stride = SX1, SX0
        else:
            batch, token = rows // INNER, rows % INNER
            batch_stride, token_stride = SX0, SX1
        chunk = N // TOKENS
        mixed_offsets = (
            batch[:, None] * batch_stride
            + (cols[None, :] // chunk) * token_stride
            + (token[:, None] * chunk + cols[None, :] % chunk) * SX2
        )
        mixed = tl.load(X + mixed_offsets, valid, 0).to(tl.float32)
        # Preserve the BF16 addition boundary of the unfused x + token_mix(x).
        x = (x + mixed).to(X.dtype.element_ty).to(tl.float32)
        tl.store(Z + rows[:, None] * N + cols[None, :], x, valid)
    weight = tl.load(W + cols, cols < N, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / N + EPS)
    y = x * rstd[:, None] * weight[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], y, valid)
    tl.store(RSTD + rows, rstd, rows < M)


@triton.jit
def _rms_backward(
    X,
    W,
    DY,
    RSTD,
    DX,
    DW,
    M,
    INNER,
    SX0,
    SX1,
    SX2,
    SY0,
    SY1,
    SY2,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    x_offsets = (rows // INNER) * SX0 + (rows % INNER) * SX1
    dy_offsets = (rows // INNER) * SY0 + (rows % INNER) * SY1
    x = tl.load(X + x_offsets[:, None] + cols[None, :] * SX2, valid, 0).to(tl.float32)
    dy = tl.load(DY + dy_offsets[:, None] + cols[None, :] * SY2, valid, 0).to(tl.float32)
    weight = tl.load(W + cols, cols < N, 0).to(tl.float32)
    rstd = tl.load(RSTD + rows, rows < M, 0)
    xhat = x * rstd[:, None]
    wdy = dy * weight[None, :]
    correction = tl.sum(xhat * wdy, 1) / N
    dx = (wdy - xhat * correction[:, None]) * rstd[:, None]
    tl.store(DX + rows[:, None] * N + cols[None, :], dx, valid)
    dw = tl.sum(dy * xhat, 0)
    tl.store(DW + block * N + cols, dw, cols < N)


@triton.jit
def _residual_rms_forward(
    X,
    R,
    S,
    W,
    Y,
    Z,
    RSTD,
    M,
    INNER,
    SX0,
    SX1,
    SX2,
    SR0,
    SR1,
    SR2,
    SS0,
    SS1,
    SS2,
    N: tl.constexpr,
    EPS: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    outer, inner = rows // INNER, rows % INNER
    x = tl.load(X + outer[:, None] * SX0 + inner[:, None] * SX1 + cols[None, :] * SX2, valid, 0)
    residual = tl.load(R + outer[:, None] * SR0 + inner[:, None] * SR1 + cols[None, :] * SR2, valid, 0)
    z = (x.to(tl.float32) + residual.to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(z * z, 1) / N + EPS)
    weight = tl.load(W + cols, cols < N, 0).to(tl.float32)
    # Keep the standalone normalization's BF16 output rounding before the block residual.
    output = (z * rstd[:, None] * weight[None, :]).to(X.dtype.element_ty).to(tl.float32)
    if HAS_SKIP:
        skip = tl.load(S + outer[:, None] * SS0 + inner[:, None] * SS1 + cols[None, :] * SS2, valid, 0)
        output += skip.to(tl.float32)
    offsets = rows[:, None] * N + cols[None, :]
    tl.store(Z + offsets, z, valid)
    tl.store(Y + offsets, output, valid)
    tl.store(RSTD + rows, rstd, rows < M)


class _ResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, residual, skip, weight, eps):
        token_major = values.stride(0) < values.stride(1)
        x = values.transpose(0, 1) if token_major else values
        r = residual.transpose(0, 1) if token_major else residual
        s = (skip.transpose(0, 1) if token_major else skip) if skip is not None else x
        outer, inner, width = x.shape
        rows = outer * inner
        output = torch.empty((outer, inner, width), device=x.device, dtype=x.dtype)
        norm_input = torch.empty_like(output)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        _residual_rms_forward[(triton.cdiv(rows, 16),)](
            x,
            r,
            s,
            weight,
            output,
            norm_input,
            rstd,
            rows,
            inner,
            *x.stride(),
            *r.stride(),
            *s.stride(),
            width,
            eps,
            skip is not None,
            16,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(norm_input, weight, rstd)
        ctx.token_major, ctx.has_skip = token_major, skip is not None
        return output.transpose(0, 1) if token_major else output

    @staticmethod
    def backward(ctx, gradient):
        x, weight, rstd = ctx.saved_tensors
        dy = gradient.transpose(0, 1) if ctx.token_major else gradient
        outer, inner, width = x.shape
        rows = outer * inner
        blocks = triton.cdiv(rows, 16)
        dx = torch.empty_like(x)
        partial = torch.empty((blocks, width), device=x.device, dtype=torch.float32)
        _rms_backward[(blocks,)](
            x,
            weight,
            dy,
            rstd,
            dx,
            partial,
            rows,
            inner,
            *x.stride(),
            *dy.stride(),
            width,
            16,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        dx = dx.transpose(0, 1) if ctx.token_major else dx
        return dx, dx, gradient if ctx.has_skip else None, partial.sum(0).to(weight.dtype), None


def residual_rms_norm(values, residual, skip, weight, eps=1e-5):
    """Fuse optional block residual + RMSNorm(values + residual), with native sum rounding."""
    if values.ndim != 3 or not values.is_cuda or not 0 < values.shape[-1] <= 1024:
        raise ValueError("Fused residual RMSNorm requires CUDA [batch, token, width<=1024]")
    if residual.shape != values.shape or (skip is not None and skip.shape != values.shape):
        raise ValueError("RMSNorm residuals must have matching shapes")
    return _ResidualRMSNorm.apply(values, residual, skip, weight, eps)


@triton.jit
def _mix_rms_tiled_forward(
    X,
    W,
    Y,
    Z,
    RSTD,
    BATCH,
    SX0,
    SX1,
    SX2,
    N: tl.constexpr,
    TOKENS: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
    EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    token = tl.arange(0, TOKENS)
    cols = tl.arange(0, N)
    if TOKEN_MAJOR:
        offsets = token[:, None] * SX0 + batch * SX1 + cols[None, :] * SX2
        rows = token * BATCH + batch
    else:
        offsets = batch * SX0 + token[:, None] * SX1 + cols[None, :] * SX2
        rows = batch * TOKENS + token
    x = tl.load(X + offsets).to(tl.float32)
    # All tokens of one candidate fit in a tile. Permute within that tile,
    # reusing the first load instead of issuing scattered reads of P(x).
    mixed = tl.reshape(tl.permute(tl.reshape(x, (TOKENS, TOKENS, N // TOKENS)), (1, 0, 2)), (TOKENS, N))
    z = (x + mixed).to(X.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(z * z, 1) / N + EPS)
    weight = tl.load(W + cols).to(tl.float32)
    output_offsets = rows[:, None] * N + cols[None, :]
    tl.store(Z + output_offsets, z)
    tl.store(Y + output_offsets, z * rstd[:, None] * weight[None, :])
    tl.store(RSTD + rows, rstd)


@triton.jit
def _mix_tiled_backward(DZ, DX, BATCH, N: tl.constexpr, TOKENS: tl.constexpr, TOKEN_MAJOR: tl.constexpr):
    batch = tl.program_id(0)
    token = tl.arange(0, TOKENS)
    cols = tl.arange(0, N)
    rows = token * BATCH + batch if TOKEN_MAJOR else batch * TOKENS + token
    offsets = rows[:, None] * N + cols[None, :]
    gradient = tl.load(DZ + offsets).to(tl.float32)
    mixed = tl.reshape(tl.permute(tl.reshape(gradient, (TOKENS, TOKENS, N // TOKENS)), (1, 0, 2)), (TOKENS, N))
    tl.store(DX + offsets, gradient + mixed)


@triton.jit
def _mix_backward(
    DZ,
    DX,
    M,
    INNER,
    N: tl.constexpr,
    TOKENS: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    chunk = N // TOKENS
    if TOKEN_MAJOR:
        token, batch = rows // INNER, rows % INNER
        mixed_rows = (cols[None, :] // chunk) * INNER + batch[:, None]
    else:
        batch, token = rows // INNER, rows % INNER
        mixed_rows = batch[:, None] * INNER + cols[None, :] // chunk
    mixed_cols = token[:, None] * chunk + cols[None, :] % chunk
    own = tl.load(DZ + rows[:, None] * N + cols[None, :], valid, 0).to(tl.float32)
    mixed = tl.load(DZ + mixed_rows * N + mixed_cols, valid, 0).to(tl.float32)
    # P is an involution, so d(x + P(x))/dx = dz + P(dz).
    tl.store(DX + rows[:, None] * N + cols[None, :], own + mixed, valid)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, eps, mix):
        # Einsum commonly produces token-major storage with logical [batch, token, width].
        token_major = values.stride(0) < values.stride(1)
        x = values.transpose(0, 1) if token_major else values
        outer, inner, width = x.shape
        rows = outer * inner
        output = torch.empty((outer, inner, width), device=x.device, dtype=x.dtype)
        norm_input = torch.empty_like(output) if mix else x
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        tiled = (
            mix
            and triton.next_power_of_2(width) == width
            and triton.next_power_of_2(values.shape[1]) == values.shape[1]
        )
        if tiled:
            _mix_rms_tiled_forward[(values.shape[0],)](
                x,
                weight,
                output,
                norm_input,
                rstd,
                values.shape[0],
                *x.stride(),
                width,
                values.shape[1],
                token_major,
                eps,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _rms_forward[(triton.cdiv(rows, 16),)](
                x,
                weight,
                output,
                rstd,
                rows,
                inner,
                *x.stride(),
                width,
                eps,
                16,
                triton.next_power_of_2(width),
                norm_input,
                mix,
                values.shape[1],
                token_major,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(norm_input, weight, rstd)
        ctx.token_major = token_major
        ctx.mix, ctx.tokens = mix, values.shape[1]
        ctx.tiled, ctx.batch_size = tiled, values.shape[0]
        return output.transpose(0, 1) if token_major else output

    @staticmethod
    def backward(ctx, gradient):
        x, weight, rstd = ctx.saved_tensors
        dy = gradient.transpose(0, 1) if ctx.token_major else gradient
        outer, inner, width = x.shape
        rows = outer * inner
        blocks = triton.cdiv(rows, 16)
        dx = torch.empty((outer, inner, width), device=x.device, dtype=x.dtype)
        partial = torch.empty((blocks, width), device=x.device, dtype=torch.float32)
        _rms_backward[(blocks,)](
            x,
            weight,
            dy,
            rstd,
            dx,
            partial,
            rows,
            inner,
            *x.stride(),
            *dy.stride(),
            width,
            16,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        if ctx.mix:
            mixed_dx = torch.empty_like(dx)
            if ctx.tiled:
                _mix_tiled_backward[(ctx.batch_size,)](
                    dx,
                    mixed_dx,
                    ctx.batch_size,
                    width,
                    ctx.tokens,
                    ctx.token_major,
                    num_warps=4,
                )
            else:
                _mix_backward[(blocks,)](
                    dx,
                    mixed_dx,
                    rows,
                    inner,
                    width,
                    ctx.tokens,
                    ctx.token_major,
                    16,
                    triton.next_power_of_2(width),
                    num_warps=4,
                )
            dx = mixed_dx
        return (dx.transpose(0, 1) if ctx.token_major else dx), partial.sum(0).to(weight.dtype), None, None


def rms_norm(values, weight, bias=None, eps=1e-5):
    """BF16/FP16 training path; callers retain native FP32 evaluation."""
    if bias is not None or values.ndim != 3 or not values.is_cuda or not 0 < values.shape[-1] <= 1024:
        raise ValueError("Mixer RMSNorm requires CUDA [batch, token, width<=1024] and no bias")
    return _RMSNorm.apply(values, weight, eps, False)


def mix_rms_norm(values, weight, eps=1e-5):
    """Fuse RMSNorm(x + token_mix(x)), retaining the original BF16 sum rounding."""
    if values.ndim != 3 or not values.is_cuda or not 0 < values.shape[-1] <= 1024:
        raise ValueError("Fused Mixer RMSNorm requires CUDA [batch, token, width<=1024]")
    if values.shape[-1] % values.shape[1]:
        raise ValueError("Mixer width must be divisible by the token count")
    return _RMSNorm.apply(values, weight, eps, True)
