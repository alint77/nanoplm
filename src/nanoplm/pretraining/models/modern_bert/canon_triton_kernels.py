"""Triton kernels for varlen Canon depthwise convolution.

Three kernels cover the full forward + backward:

  _canon_fwd_kernel       — forward: depthwise conv with seq-boundary masking
  _canon_bwd_dx_kernel    — backward: grad_x via transpose conv (flipped weights)
  _canon_bwd_dw_db_kernel — backward: grad_weight (reduction over T) + grad_bias

All kernels tile over (T, C) with a 2-D grid.  The inner K-tap loop is
statically unrolled via ``tl.static_range`` so the compiler can schedule
all neighbor loads at once.

``FP32_ACCUM`` constexpr controls accumulation dtype: when True (bf16/fp16
inputs), accumulation is done in fp32; when False (fp32/fp64 inputs), the
native dtype is preserved.
"""

from __future__ import annotations

import triton
import triton.language as tl


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
