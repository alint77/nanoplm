"""Triton kernels for varlen Canon depthwise convolution.

Kernels cover the full forward + backward, plus fused LayerNorm variants:

  _canon_fwd_kernel                   — forward: depthwise conv with seq-boundary masking
  _canon_bwd_dx_kernel                — backward: grad_x via transpose conv (flipped weights)
  _canon_bwd_dw_db_kernel             — backward: grad_weight (reduction over T) + grad_bias
  _canon_ln_fwd_kernel                — fused: LN normalize + depthwise conv + LN skip
  _ln_stats_kernel                    — one-pass mean + rstd per row (FlashAttention-style)
  _ln_bwd_kernel                      — LN backward: dx per-row + partial dgamma
  _canon_ln_bwd_dw_db_partial_kernel  — conv bwd_dw_db with partial-buffer reduction (no atomics)

Conv kernels tile over (T, C) with a 2-D grid.  The inner K-tap loop is
statically unrolled via ``tl.static_range``.  LN stats and backward kernels
use a 1-D grid (one program per row / multi-row chunk) with BLOCK_N covering
the full hidden dim, following the FlashAttention Triton LayerNorm design.

``FP32_ACCUM`` constexpr controls accumulation dtype: when True (bf16/fp16
inputs), accumulation is done in fp32; when False (fp32/fp64 inputs), the
native dtype is preserved.
"""

from __future__ import annotations

import triton
import triton.language as tl


# ── Autotune configurations ──────────────────────────────────────────────


def _autotune_configs(
    *,
    block_ts: tuple[int, ...],
    block_cs: tuple[int, ...],
    warps: tuple[int, ...],
    stages: tuple[int, ...],
) -> list[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_T": bt, "BLOCK_C": bc},
            num_warps=nw,
            num_stages=ns,
        )
        for bt in block_ts
        for bc in block_cs
        for nw in warps
        for ns in stages
    ]


def _streaming_configs(configs: tuple[tuple[int, int, int, int], ...]) -> list[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_T": bt, "BLOCK_C": bc},
            num_warps=nw,
            num_stages=ns,
        )
        for bt, bc, nw, ns in configs
    ]


def _rowwise_configs(
    configs: tuple[tuple[int, int], ...],
    *,
    pre_hook=None,
) -> list[triton.Config]:
    return [
        triton.Config(
            {"PROGRAM_MULTIPLIER": pm},
            num_warps=nw,
            pre_hook=pre_hook,
        )
        for pm, nw in configs
    ]


_AUTOTUNE_FWD_CONFIGS = _streaming_configs(
    (
        (16, 64, 1, 2),
        (16, 64, 1, 3),
        (16, 64, 1, 4),
        (16, 128, 2, 1),
        (32, 64, 1, 1),
        (32, 64, 1, 4),
        (32, 64, 4, 2),
        (64, 128, 4, 2),
    )
)

_AUTOTUNE_BWD_DX_CONFIGS = _streaming_configs(
    (
        (16, 64, 1, 2),
        (16, 64, 1, 3),
        (32, 64, 1, 1),
        (32, 64, 1, 3),
        (32, 64, 1, 4),
        (32, 64, 4, 2),
        (16, 128, 2, 4),
        (64, 128, 4, 2),
    )
)

def _bwd_dw_db_pre_hook(nargs):
    """Zero accumulator buffers before each autotune trial."""
    nargs["grad_w_ptr"].zero_()
    nargs["grad_b_ptr"].zero_()


_AUTOTUNE_BWD_DW_DB_CONFIGS = [
    triton.Config(
        {"BLOCK_T": bt, "BLOCK_C": bc},
        num_warps=nw,
        num_stages=ns,
        pre_hook=_bwd_dw_db_pre_hook,
    )
    for bt, bc, nw, ns in (
        (256, 64, 4, 2),
        (256, 64, 4, 1),
        (256, 64, 4, 3),
        (256, 32, 4, 2),
        (256, 32, 4, 1),
        (128, 64, 4, 2),
        (128, 64, 2, 2),
        (128, 32, 2, 2),
    )
]


# ── Forward kernel ────────────────────────────────────────────────────────


@triton.jit
def _canon_fwd_kernel(
    # Pointers
    x_ptr,
    seq_id_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    # Dimensions
    T,
    C,
    # Strides for x / out  (same layout)
    stride_t,
    stride_c,
    # Compile-time
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """out[t, c] = bias[c] + Σ_k weight[c, k] * x[t + k - r, c] * valid"""
    K: tl.constexpr = 2 * RADIUS + 1

    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offs < T
    c_mask = c_offs < C

    my_seq = tl.load(seq_id_ptr + t_offs, mask=t_mask, other=-1)

    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for k in tl.static_range(K):
        offset = k - RADIUS
        nbr_t = t_offs + offset
        in_bounds = (nbr_t >= 0) & (nbr_t < T)

        nbr_seq = tl.load(seq_id_ptr + nbr_t, mask=in_bounds & t_mask, other=-1)
        valid = in_bounds & (my_seq == nbr_seq)

        x_ptrs = x_ptr + nbr_t[:, None] * stride_t + c_offs[None, :] * stride_c
        x_val = tl.load(x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)
        if FP32_ACCUM:
            x_val = x_val.to(tl.float32)

        w_k = tl.load(weight_ptr + c_offs * K + k, mask=c_mask, other=0.0)
        if FP32_ACCUM:
            w_k = w_k.to(tl.float32)

        acc += x_val * w_k[None, :]

    b = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
    if FP32_ACCUM:
        b = b.to(tl.float32)
    acc += b[None, :]

    out_ptrs = out_ptr + t_offs[:, None] * stride_t + c_offs[None, :] * stride_c
    tl.store(out_ptrs, acc, mask=t_mask[:, None] & c_mask[None, :])


# ── Backward grad_x kernel ───────────────────────────────────────────────


@triton.jit
def _canon_bwd_dx_kernel(
    grad_out_ptr,
    seq_id_ptr,
    weight_ptr,
    grad_x_ptr,
    T,
    C,
    stride_t,
    stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """grad_x[s, c] = Σ_k weight[c, K-1-k] * grad_out[s + k - r, c] * valid"""
    K: tl.constexpr = 2 * RADIUS + 1

    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offs < T
    c_mask = c_offs < C

    my_seq = tl.load(seq_id_ptr + t_offs, mask=t_mask, other=-1)

    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for k in tl.static_range(K):
        offset = k - RADIUS
        nbr_t = t_offs + offset
        in_bounds = (nbr_t >= 0) & (nbr_t < T)

        nbr_seq = tl.load(seq_id_ptr + nbr_t, mask=in_bounds & t_mask, other=-1)
        valid = in_bounds & (my_seq == nbr_seq)

        go_ptrs = (
            grad_out_ptr + nbr_t[:, None] * stride_t + c_offs[None, :] * stride_c
        )
        go_val = tl.load(go_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)
        if FP32_ACCUM:
            go_val = go_val.to(tl.float32)

        # Flipped weight index: K-1-k
        w_k = tl.load(
            weight_ptr + c_offs * K + (K - 1 - k), mask=c_mask, other=0.0
        )
        if FP32_ACCUM:
            w_k = w_k.to(tl.float32)

        acc += go_val * w_k[None, :]

    out_ptrs = grad_x_ptr + t_offs[:, None] * stride_t + c_offs[None, :] * stride_c
    tl.store(out_ptrs, acc, mask=t_mask[:, None] & c_mask[None, :])


# ── Backward grad_weight + grad_bias kernel ───────────────────────────────


@triton.jit
def _canon_bwd_dw_db_kernel(
    grad_out_ptr,
    x_ptr,
    seq_id_ptr,
    grad_w_ptr,  # (C, K) fp32 — must be pre-zeroed
    grad_b_ptr,  # (C,)   fp32 — must be pre-zeroed
    T,
    C,
    stride_t,
    stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """grad_w[c,k] = Σ_t go[t,c] * x[t+k-r, c] * valid;  grad_b[c] = Σ_t go[t,c]

    Each block computes partial sums over its BLOCK_T tile, then uses
    atomic_add to accumulate into the global (C, K) / (C,) outputs.
    """
    K: tl.constexpr = 2 * RADIUS + 1

    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offs < T
    c_mask = c_offs < C

    go_ptrs = grad_out_ptr + t_offs[:, None] * stride_t + c_offs[None, :] * stride_c
    go = tl.load(go_ptrs, mask=t_mask[:, None] & c_mask[None, :], other=0.0)
    if FP32_ACCUM:
        go = go.to(tl.float32)

    my_seq = tl.load(seq_id_ptr + t_offs, mask=t_mask, other=-1)

    # Bias grad: sum grad_out over T-tile
    pb = tl.sum(go, axis=0)  # (BLOCK_C,)
    tl.atomic_add(grad_b_ptr + c_offs, pb, mask=c_mask)

    # Weight grad: per-tap reduction
    for k in tl.static_range(K):
        offset = k - RADIUS
        nbr_t = t_offs + offset
        in_bounds = (nbr_t >= 0) & (nbr_t < T)

        nbr_seq = tl.load(seq_id_ptr + nbr_t, mask=in_bounds & t_mask, other=-1)
        valid = in_bounds & (my_seq == nbr_seq)

        x_ptrs = x_ptr + nbr_t[:, None] * stride_t + c_offs[None, :] * stride_c
        x_val = tl.load(
            x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0
        )
        if FP32_ACCUM:
            x_val = x_val.to(tl.float32)

        pw_k = tl.sum(go * x_val, axis=0)  # (BLOCK_C,)
        tl.atomic_add(grad_w_ptr + c_offs * K + k, pw_k, mask=c_mask)


# ── Autotuned wrappers ──────────────────────────────────────────────────
# Key includes T and C (problem size) so Triton caches per shape.
# RADIUS is a constexpr passed through — each radius compiles a separate
# kernel binary regardless, so it does not need to be in the key.


@triton.autotune(configs=_AUTOTUNE_FWD_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _canon_fwd_kernel_autotuned(
    x_ptr, seq_id_ptr, weight_ptr, bias_ptr, out_ptr,
    T, C, stride_t, stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    _canon_fwd_kernel(
        x_ptr, seq_id_ptr, weight_ptr, bias_ptr, out_ptr,
        T, C, stride_t, stride_c,
        RADIUS, BLOCK_T, BLOCK_C, FP32_ACCUM,
    )


@triton.autotune(configs=_AUTOTUNE_BWD_DX_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _canon_bwd_dx_kernel_autotuned(
    grad_out_ptr, seq_id_ptr, weight_ptr, grad_x_ptr,
    T, C, stride_t, stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    _canon_bwd_dx_kernel(
        grad_out_ptr, seq_id_ptr, weight_ptr, grad_x_ptr,
        T, C, stride_t, stride_c,
        RADIUS, BLOCK_T, BLOCK_C, FP32_ACCUM,
    )


@triton.autotune(configs=_AUTOTUNE_BWD_DW_DB_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _canon_bwd_dw_db_kernel_autotuned(
    grad_out_ptr, x_ptr, seq_id_ptr, grad_w_ptr, grad_b_ptr,
    T, C, stride_t, stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    _canon_bwd_dw_db_kernel(
        grad_out_ptr, x_ptr, seq_id_ptr, grad_w_ptr, grad_b_ptr,
        T, C, stride_t, stride_c,
        RADIUS, BLOCK_T, BLOCK_C, FP32_ACCUM,
    )


# ── Fused LayerNorm + Conv kernels ──────────────────────────────────────
#
# These kernels fuse nn.LayerNorm normalization into the conv forward/backward
# to eliminate a (T, C) intermediate materialization.  Stats (mean, rstd) are
# pre-computed by the caller; the kernels apply normalize-and-scale inline.


@triton.jit
def _canon_ln_fwd_kernel(
    # Raw (un-normalized) input
    x_ptr,
    seq_id_ptr,
    # LN params / stats (pre-computed)
    mean_ptr,       # (T,)   fp32
    rstd_ptr,       # (T,)   fp32
    ln_w_ptr,       # (C,)   LN gamma
    # Conv params
    conv_w_ptr,     # (C, K)
    conv_b_ptr,     # (C,)
    # Output
    out_ptr,
    # Dims
    T, C,
    stride_t, stride_c,
    # Compile-time
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """out[t,c] = gamma[c] * (x_hat[t,c] + Σ_k conv_w[c,k] * x_hat[t+k-r,c]) + conv_bias[c]

    x_hat[t,c] = (x[t,c] - mean[t]) * rstd[t]

    Gamma is factored out of the K-tap loop to shorten live ranges and
    reduce register pressure (enabling higher occupancy on H100).
    """
    K: tl.constexpr = 2 * RADIUS + 1

    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offs < T
    c_mask = c_offs < C

    my_seq = tl.load(seq_id_ptr + t_offs, mask=t_mask, other=-1)

    # LN gamma — shared across all K taps, load once
    gamma = tl.load(ln_w_ptr + c_offs, mask=c_mask, other=1.0)
    if FP32_ACCUM:
        gamma = gamma.to(tl.float32)

    # Skip connection: x_hat[t, c] for the residual (gamma applied after loop)
    x_self_ptrs = x_ptr + t_offs[:, None] * stride_t + c_offs[None, :] * stride_c
    x_self = tl.load(x_self_ptrs, mask=t_mask[:, None] & c_mask[None, :], other=0.0)
    if FP32_ACCUM:
        x_self = x_self.to(tl.float32)
    my_mean = tl.load(mean_ptr + t_offs, mask=t_mask, other=0.0)
    my_rstd = tl.load(rstd_ptr + t_offs, mask=t_mask, other=1.0)
    if FP32_ACCUM:
        my_mean = my_mean.to(tl.float32)
        my_rstd = my_rstd.to(tl.float32)
    x_hat_self = (x_self - my_mean[:, None]) * my_rstd[:, None]

    # Center-tap specialization: the center tap (k=RADIUS) reads the same
    # row as x_self, so fuse it with the skip connection to save 1 x load
    # and 1 normalization per CTA.
    w_center = tl.load(conv_w_ptr + c_offs * K + RADIUS, mask=c_mask, other=0.0)
    if FP32_ACCUM:
        w_center = w_center.to(tl.float32)
    acc = (1.0 + w_center[None, :]) * x_hat_self

    # Non-center taps: two half-loops around RADIUS (gamma factored out)
    for k in tl.range(0, RADIUS):
        offset = k - RADIUS
        nbr_t = t_offs + offset
        in_bounds = (nbr_t >= 0) & (nbr_t < T)

        nbr_seq = tl.load(seq_id_ptr + nbr_t, mask=in_bounds & t_mask, other=-1)
        valid = in_bounds & (my_seq == nbr_seq)

        x_ptrs = x_ptr + nbr_t[:, None] * stride_t + c_offs[None, :] * stride_c
        x_val = tl.load(x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)
        if FP32_ACCUM:
            x_val = x_val.to(tl.float32)

        nbr_mean = tl.load(mean_ptr + nbr_t, mask=in_bounds & t_mask, other=0.0)
        nbr_rstd = tl.load(rstd_ptr + nbr_t, mask=in_bounds & t_mask, other=1.0)
        if FP32_ACCUM:
            nbr_mean = nbr_mean.to(tl.float32)
            nbr_rstd = nbr_rstd.to(tl.float32)
        x_hat = (x_val - nbr_mean[:, None]) * nbr_rstd[:, None]
        x_hat = tl.where(valid[:, None], x_hat, 0.0)

        w_k = tl.load(conv_w_ptr + c_offs * K + k, mask=c_mask, other=0.0)
        if FP32_ACCUM:
            w_k = w_k.to(tl.float32)

        acc += x_hat * w_k[None, :]

    for k in tl.range(RADIUS + 1, K):
        offset = k - RADIUS
        nbr_t = t_offs + offset
        in_bounds = (nbr_t >= 0) & (nbr_t < T)

        nbr_seq = tl.load(seq_id_ptr + nbr_t, mask=in_bounds & t_mask, other=-1)
        valid = in_bounds & (my_seq == nbr_seq)

        x_ptrs = x_ptr + nbr_t[:, None] * stride_t + c_offs[None, :] * stride_c
        x_val = tl.load(x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)
        if FP32_ACCUM:
            x_val = x_val.to(tl.float32)

        nbr_mean = tl.load(mean_ptr + nbr_t, mask=in_bounds & t_mask, other=0.0)
        nbr_rstd = tl.load(rstd_ptr + nbr_t, mask=in_bounds & t_mask, other=1.0)
        if FP32_ACCUM:
            nbr_mean = nbr_mean.to(tl.float32)
            nbr_rstd = nbr_rstd.to(tl.float32)
        x_hat = (x_val - nbr_mean[:, None]) * nbr_rstd[:, None]
        x_hat = tl.where(valid[:, None], x_hat, 0.0)

        w_k = tl.load(conv_w_ptr + c_offs * K + k, mask=c_mask, other=0.0)
        if FP32_ACCUM:
            w_k = w_k.to(tl.float32)

        acc += x_hat * w_k[None, :]

    # Apply gamma and conv bias after the loop
    cb = tl.load(conv_b_ptr + c_offs, mask=c_mask, other=0.0)
    if FP32_ACCUM:
        cb = cb.to(tl.float32)
    acc = acc * gamma[None, :] + cb[None, :]

    out_ptrs = out_ptr + t_offs[:, None] * stride_t + c_offs[None, :] * stride_c
    tl.store(out_ptrs, acc, mask=t_mask[:, None] & c_mask[None, :])


# ── One-pass LayerNorm stats kernel ──────────────────────────────────────


@triton.jit
def _ln_stats_kernel(
    x_ptr,
    mean_ptr,      # (T,) output
    rstd_ptr,      # (T,) output
    T, C,
    stride_t, stride_c,
    eps,
    BLOCK_N: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """One-pass mean + rstd per row.  Grid = (T,).

    BLOCK_N >= C (covers full hidden dim in registers, like FlashAttention).
    """
    row = tl.program_id(0)
    if row >= T:
        return
    cols = tl.arange(0, BLOCK_N)
    mask = cols < C

    x = tl.load(x_ptr + row * stride_t + cols * stride_c,
                 mask=mask, other=0.0)
    if FP32_ACCUM:
        x = x.to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    xbar = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


# ── LayerNorm backward kernel (FlashAttention-style) ─────────────────────


@triton.jit
def _ln_bwd_kernel(
    grad_ln_out_ptr,    # (T, C)
    x_ptr,              # (T, C)
    mean_ptr,           # (T,)
    rstd_ptr,           # (T,)
    ln_w_ptr,           # (C,)
    grad_x_ptr,         # (T, C) output
    partial_dgamma_ptr, # (num_programs, C) output
    T, C,
    stride_t, stride_c,
    rows_per_program,
    BLOCK_N: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """Fused LN backward: dx per-row + partial dgamma in registers.

    Grid = (num_programs,) where num_programs = SM_count * 8.
    Each program handles ``rows_per_program`` consecutive rows.
    """
    pid = tl.program_id(0)
    row_start = pid * rows_per_program
    row_end = min(row_start + rows_per_program, T)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < C

    gamma = tl.load(ln_w_ptr + cols, mask=mask, other=0.0)
    if FP32_ACCUM:
        gamma = gamma.to(tl.float32)
    dgamma_acc = tl.zeros((BLOCK_N,), dtype=gamma.dtype)

    for row in range(row_start, row_end):
        row_off = row * stride_t
        glo = tl.load(grad_ln_out_ptr + row_off + cols * stride_c,
                       mask=mask, other=0.0)
        x = tl.load(x_ptr + row_off + cols * stride_c,
                     mask=mask, other=0.0)
        if FP32_ACCUM:
            glo = glo.to(tl.float32)
            x = x.to(tl.float32)
        m = tl.load(mean_ptr + row)
        r = tl.load(rstd_ptr + row)
        if FP32_ACCUM:
            m = m.to(tl.float32)
            r = r.to(tl.float32)

        x_hat = (x - m) * r
        x_hat = tl.where(mask, x_hat, 0.0)

        dgamma_acc += glo * x_hat

        wdy = glo * gamma
        c1 = tl.sum(x_hat * wdy, axis=0) / C
        c2 = tl.sum(wdy, axis=0) / C
        dx = (wdy - x_hat * c1 - c2) * r

        tl.store(grad_x_ptr + row_off + cols * stride_c, dx, mask=mask)

    tl.store(partial_dgamma_ptr + pid * C + cols, dgamma_acc, mask=mask)


# ── Fused conv_bwd_dx + LN backward kernel ────────────────────────────────
#
# Fuses three operations into one kernel (1D grid, BLOCK_N >= C):
#   1. conv_bwd_dx: transpose conv to get grad w.r.t. LN output
#   2. elementwise add: grad_ln_out = grad_out + grad_conv  (skip connection)
#   3. LN backward: dx + partial dgamma
# Eliminates two intermediate (T, C) tensors (grad_conv, grad_ln_out).


@triton.jit
def _fused_conv_bwd_dx_ln_bwd_kernel(
    grad_out_ptr,        # (T, C) input
    x_ptr,               # (T, C) input
    seq_id_ptr,          # (T,)   input
    mean_ptr,            # (T,)   input
    rstd_ptr,            # (T,)   input
    ln_w_ptr,            # (C,)   input — LN gamma
    conv_w_ptr,          # (C, K) input — conv weight
    grad_x_ptr,          # (T, C) output
    partial_dgamma_ptr,  # (num_programs, C) output
    T, C,
    stride_t, stride_c,
    rows_per_program,
    RADIUS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """Fused conv transpose + residual add + LN backward.

    Grid = (num_programs,) where num_programs = SM_count * 8.
    Each program handles ``rows_per_program`` consecutive rows.
    Per row: compute conv_bwd_dx via K-tap transpose conv in registers,
    add to grad_out for grad_ln_out, then run LN backward for dx + dgamma.
    """
    K: tl.constexpr = 2 * RADIUS + 1

    pid = tl.program_id(0)
    row_start = pid * rows_per_program
    row_end = min(row_start + rows_per_program, T)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < C

    gamma = tl.load(ln_w_ptr + cols, mask=mask, other=0.0)
    if FP32_ACCUM:
        gamma = gamma.to(tl.float32)
    dgamma_acc = tl.zeros((BLOCK_N,), dtype=gamma.dtype)

    # Pre-load conv weights (K values per channel, all fit in registers)
    # We need flipped weights for transpose conv: index K-1-k
    # Load all K taps upfront to avoid repeated global loads
    # conv_w layout: (C, K) — conv_w_ptr[c * K + k]

    for row in range(row_start, row_end):
        row_off = row * stride_t

        # Load grad_out for this row
        go_self = tl.load(grad_out_ptr + row_off + cols * stride_c,
                          mask=mask, other=0.0)
        if FP32_ACCUM:
            go_self = go_self.to(tl.float32)

        my_seq = tl.load(seq_id_ptr + row)

        # ── Step 1: conv_bwd_dx — transpose conv with flipped weights ──
        conv_acc = tl.zeros((BLOCK_N,), dtype=go_self.dtype)
        for k in tl.range(0, K):
            offset = k - RADIUS
            nbr = row + offset
            in_bounds = (nbr >= 0) & (nbr < T)

            nbr_seq = tl.load(seq_id_ptr + nbr, mask=in_bounds, other=-1)
            valid = in_bounds & (my_seq == nbr_seq)

            go_nbr = tl.load(
                grad_out_ptr + nbr * stride_t + cols * stride_c,
                mask=valid & mask, other=0.0,
            )
            if FP32_ACCUM:
                go_nbr = go_nbr.to(tl.float32)

            # Flipped weight: K-1-k
            w_k = tl.load(conv_w_ptr + cols * K + (K - 1 - k),
                          mask=mask, other=0.0)
            if FP32_ACCUM:
                w_k = w_k.to(tl.float32)

            conv_acc += go_nbr * w_k

        # ── Step 2: grad_ln_out = grad_out + conv_grad (skip + conv) ──
        grad_ln_out = go_self + conv_acc

        # ── Step 3: LN backward ──
        x_row = tl.load(x_ptr + row_off + cols * stride_c,
                        mask=mask, other=0.0)
        if FP32_ACCUM:
            x_row = x_row.to(tl.float32)
        m = tl.load(mean_ptr + row)
        r = tl.load(rstd_ptr + row)
        if FP32_ACCUM:
            m = m.to(tl.float32)
            r = r.to(tl.float32)

        x_hat = (x_row - m) * r
        x_hat = tl.where(mask, x_hat, 0.0)

        dgamma_acc += grad_ln_out * x_hat

        wdy = grad_ln_out * gamma
        c1 = tl.sum(x_hat * wdy, axis=0) / C
        c2 = tl.sum(wdy, axis=0) / C
        dx = (wdy - x_hat * c1 - c2) * r

        tl.store(grad_x_ptr + row_off + cols * stride_c, dx, mask=mask)

    tl.store(partial_dgamma_ptr + pid * C + cols, dgamma_acc, mask=mask)


# ── Fused LN+Conv backward dw/db with partial buffers ───────────────────


@triton.jit
def _canon_ln_bwd_dw_db_partial_kernel(
    grad_out_ptr,
    x_ptr,
    seq_id_ptr,
    mean_ptr,       # (T,) fp32
    rstd_ptr,       # (T,) fp32
    ln_w_ptr,       # (C,) LN gamma
    partial_w_ptr,  # (max_t_blocks, C * K) fp32 — pre-zeroed
    partial_b_ptr,  # (max_t_blocks, C) fp32 — pre-zeroed
    T, C,
    stride_t, stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """Conv weight/bias grads with partial-buffer reduction (no atomics).

    Same 2-D (T, C) grid as the atomic version, but each T-block writes
    its partial sums to ``partial_w[pid_t, :]`` and ``partial_b[pid_t, :]``.
    The caller sums over dim-0 afterward.
    """
    K: tl.constexpr = 2 * RADIUS + 1

    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offs < T
    c_mask = c_offs < C

    gamma = tl.load(ln_w_ptr + c_offs, mask=c_mask, other=1.0)
    if FP32_ACCUM:
        gamma = gamma.to(tl.float32)

    go_ptrs = grad_out_ptr + t_offs[:, None] * stride_t + c_offs[None, :] * stride_c
    go = tl.load(go_ptrs, mask=t_mask[:, None] & c_mask[None, :], other=0.0)
    if FP32_ACCUM:
        go = go.to(tl.float32)

    my_seq = tl.load(seq_id_ptr + t_offs, mask=t_mask, other=-1)

    # Bias grad partial — store to per-block row
    pb = tl.sum(go, axis=0)
    tl.store(partial_b_ptr + pid_t * C + c_offs, pb, mask=c_mask)

    # Center-tap specialization: k=RADIUS reads t_offs (self), same as x_self.
    # Compute it once without redundant x/mean/rstd loads.
    x_self_ptrs = x_ptr + t_offs[:, None] * stride_t + c_offs[None, :] * stride_c
    x_self = tl.load(x_self_ptrs, mask=t_mask[:, None] & c_mask[None, :], other=0.0)
    if FP32_ACCUM:
        x_self = x_self.to(tl.float32)
    my_mean = tl.load(mean_ptr + t_offs, mask=t_mask, other=0.0)
    my_rstd = tl.load(rstd_ptr + t_offs, mask=t_mask, other=1.0)
    if FP32_ACCUM:
        my_mean = my_mean.to(tl.float32)
        my_rstd = my_rstd.to(tl.float32)
    x_hat_self = (x_self - my_mean[:, None]) * my_rstd[:, None]

    pw_center = tl.sum(go * x_hat_self, axis=0) * gamma
    tl.store(partial_w_ptr + pid_t * C * K + c_offs * K + RADIUS, pw_center, mask=c_mask)

    # Non-center taps: two half-loops around RADIUS
    for k in tl.range(0, RADIUS):
        offset = k - RADIUS
        nbr_t = t_offs + offset
        in_bounds = (nbr_t >= 0) & (nbr_t < T)

        nbr_seq = tl.load(seq_id_ptr + nbr_t, mask=in_bounds & t_mask, other=-1)
        valid = in_bounds & (my_seq == nbr_seq)

        x_ptrs = x_ptr + nbr_t[:, None] * stride_t + c_offs[None, :] * stride_c
        x_val = tl.load(x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)
        if FP32_ACCUM:
            x_val = x_val.to(tl.float32)

        nbr_mean = tl.load(mean_ptr + nbr_t, mask=in_bounds & t_mask, other=0.0)
        nbr_rstd = tl.load(rstd_ptr + nbr_t, mask=in_bounds & t_mask, other=1.0)
        if FP32_ACCUM:
            nbr_mean = nbr_mean.to(tl.float32)
            nbr_rstd = nbr_rstd.to(tl.float32)
        x_hat = (x_val - nbr_mean[:, None]) * nbr_rstd[:, None]
        x_hat = tl.where(valid[:, None], x_hat, 0.0)

        pw_k = tl.sum(go * x_hat, axis=0) * gamma
        tl.store(partial_w_ptr + pid_t * C * K + c_offs * K + k, pw_k, mask=c_mask)

    for k in tl.range(RADIUS + 1, K):
        offset = k - RADIUS
        nbr_t = t_offs + offset
        in_bounds = (nbr_t >= 0) & (nbr_t < T)

        nbr_seq = tl.load(seq_id_ptr + nbr_t, mask=in_bounds & t_mask, other=-1)
        valid = in_bounds & (my_seq == nbr_seq)

        x_ptrs = x_ptr + nbr_t[:, None] * stride_t + c_offs[None, :] * stride_c
        x_val = tl.load(x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)
        if FP32_ACCUM:
            x_val = x_val.to(tl.float32)

        nbr_mean = tl.load(mean_ptr + nbr_t, mask=in_bounds & t_mask, other=0.0)
        nbr_rstd = tl.load(rstd_ptr + nbr_t, mask=in_bounds & t_mask, other=1.0)
        if FP32_ACCUM:
            nbr_mean = nbr_mean.to(tl.float32)
            nbr_rstd = nbr_rstd.to(tl.float32)
        x_hat = (x_val - nbr_mean[:, None]) * nbr_rstd[:, None]
        x_hat = tl.where(valid[:, None], x_hat, 0.0)

        pw_k = tl.sum(go * x_hat, axis=0) * gamma
        tl.store(partial_w_ptr + pid_t * C * K + c_offs * K + k, pw_k, mask=c_mask)


# ── Fused LN+Conv autotune configs ─────────────────────────────────────

_AUTOTUNE_LN_FWD_CONFIGS = _streaming_configs(
    (
        # H100 160-sweep winners (C=1024, center-tap + gamma factoring)
        (64, 32, 2, 1),    # best: 0.2050 ms
        (32, 32, 1, 3),    # 0.2078 ms
        (32, 32, 1, 2),    # 0.2081 ms
        (32, 32, 1, 1),    # 0.2083 ms
        (64, 32, 2, 2),    # 0.2114 ms
        (64, 64, 4, 2),    # 0.2120 ms
        (32, 128, 1, 1),   # 0.2141 ms (step2 winner)
        # Anchors (different T shapes)
        (16, 64, 2, 1),
        (16, 128, 4, 2),
        (64, 128, 4, 2),
    )
)

_AUTOTUNE_LN_STATS_CONFIGS = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw, ns in (
        # H100 160-sweep winners (C=1024)
        (1, 1),   # best: 0.0569 ms
        (2, 2),   # 0.0581 ms
        (2, 3),   # 0.0581 ms
        (1, 3),   # 0.0582 ms
        (1, 4),   # 0.0585 ms
    )
]


def _ln_bwd_pre_hook(nargs):
    nargs["partial_dgamma_ptr"].zero_()


_AUTOTUNE_LN_BWD_CONFIGS = _rowwise_configs(
    (
        (8, 1),
        (8, 4),
        (10, 1),
        (10, 2),
        (24, 1),
        (18, 4),
    ),
    pre_hook=_ln_bwd_pre_hook,
)

_AUTOTUNE_FUSED_CONV_LN_BWD_CONFIGS = _rowwise_configs(
    (
        # H100 160-sweep winners (C=1024, tl.range K-loop + center-tap)
        (64, 8),   # best: 0.3799 ms
        (72, 8),   # 0.3809 ms
        (76, 8),   # 0.3811 ms
        (80, 8),   # 0.3819 ms
        (88, 8),   # 0.3819 ms
        (56, 8),   # 0.3833 ms
        (4, 8),    # 0.3892 ms
        # Anchors (different T shapes)
        (12, 8),
        (24, 8),
        (64, 4),
        (64, 2),
    ),
    pre_hook=_ln_bwd_pre_hook,
)


def _ln_bwd_dw_db_partial_pre_hook(nargs):
    """Zero partial buffers before each autotune trial."""
    nargs["partial_w_ptr"].zero_()
    nargs["partial_b_ptr"].zero_()


_AUTOTUNE_LN_BWD_DW_DB_PARTIAL_CONFIGS = [
    triton.Config(
        {"BLOCK_T": bt, "BLOCK_C": bc},
        num_warps=nw,
        num_stages=ns,
        pre_hook=_ln_bwd_dw_db_partial_pre_hook,
    )
    for bt, bc, nw, ns in (
        # H100 160-sweep winners (C=1024, center-tap + gamma factoring)
        (64, 32, 1, 3),    # best: 0.2402 ms
        (64, 32, 1, 1),    # 0.2407 ms
        (64, 32, 1, 4),    # 0.2411 ms
        (64, 32, 1, 2),    # 0.2417 ms
        (64, 64, 1, 2),    # 0.2652 ms
        (64, 64, 1, 1),    # 0.2667 ms (step2 winner)
        (128, 64, 2, 1),   # 0.2583 ms
        # Anchors (different T shapes)
        (64, 64, 2, 1),
        (128, 32, 2, 2),
        (256, 32, 4, 2),
        (256, 64, 4, 2),
    )
]


# ── Fused LN+Conv autotuned wrappers ───────────────────────────────────


@triton.autotune(configs=_AUTOTUNE_LN_FWD_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _canon_ln_fwd_kernel_autotuned(
    x_ptr, seq_id_ptr, mean_ptr, rstd_ptr, ln_w_ptr,
    conv_w_ptr, conv_b_ptr, out_ptr,
    T, C, stride_t, stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    _canon_ln_fwd_kernel(
        x_ptr, seq_id_ptr, mean_ptr, rstd_ptr, ln_w_ptr,
        conv_w_ptr, conv_b_ptr, out_ptr,
        T, C, stride_t, stride_c,
        RADIUS, BLOCK_T, BLOCK_C, FP32_ACCUM,
    )


@triton.autotune(configs=_AUTOTUNE_LN_STATS_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _ln_stats_kernel_autotuned(
    x_ptr, mean_ptr, rstd_ptr,
    T, C, stride_t, stride_c,
    eps,
    BLOCK_N: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    _ln_stats_kernel(
        x_ptr, mean_ptr, rstd_ptr,
        T, C, stride_t, stride_c,
        eps, BLOCK_N, FP32_ACCUM,
    )


@triton.autotune(configs=_AUTOTUNE_LN_BWD_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _ln_bwd_kernel_autotuned(
    grad_ln_out_ptr, x_ptr, mean_ptr, rstd_ptr, ln_w_ptr,
    grad_x_ptr, partial_dgamma_ptr,
    T, C, stride_t, stride_c,
    num_sms,
    BLOCK_N: tl.constexpr,
    PROGRAM_MULTIPLIER: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    rows_per_program = tl.cdiv(T, num_sms * PROGRAM_MULTIPLIER)
    _ln_bwd_kernel(
        grad_ln_out_ptr, x_ptr, mean_ptr, rstd_ptr, ln_w_ptr,
        grad_x_ptr, partial_dgamma_ptr,
        T, C, stride_t, stride_c,
        rows_per_program, BLOCK_N, FP32_ACCUM,
    )


@triton.autotune(configs=_AUTOTUNE_FUSED_CONV_LN_BWD_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _fused_conv_bwd_dx_ln_bwd_kernel_autotuned(
    grad_out_ptr, x_ptr, seq_id_ptr, mean_ptr, rstd_ptr, ln_w_ptr, conv_w_ptr,
    grad_x_ptr, partial_dgamma_ptr,
    T, C, stride_t, stride_c,
    num_sms,
    RADIUS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PROGRAM_MULTIPLIER: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    rows_per_program = tl.cdiv(T, num_sms * PROGRAM_MULTIPLIER)
    _fused_conv_bwd_dx_ln_bwd_kernel(
        grad_out_ptr, x_ptr, seq_id_ptr, mean_ptr, rstd_ptr, ln_w_ptr, conv_w_ptr,
        grad_x_ptr, partial_dgamma_ptr,
        T, C, stride_t, stride_c,
        rows_per_program, RADIUS, BLOCK_N, FP32_ACCUM,
    )


@triton.autotune(configs=_AUTOTUNE_LN_BWD_DW_DB_PARTIAL_CONFIGS, key=["T", "C"], cache_results=True)
@triton.jit
def _canon_ln_bwd_dw_db_partial_kernel_autotuned(
    grad_out_ptr, x_ptr, seq_id_ptr, mean_ptr, rstd_ptr, ln_w_ptr,
    partial_w_ptr, partial_b_ptr,
    T, C, stride_t, stride_c,
    RADIUS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    _canon_ln_bwd_dw_db_partial_kernel(
        grad_out_ptr, x_ptr, seq_id_ptr, mean_ptr, rstd_ptr, ln_w_ptr,
        partial_w_ptr, partial_b_ptr,
        T, C, stride_t, stride_c,
        RADIUS, BLOCK_T, BLOCK_C, FP32_ACCUM,
    )
