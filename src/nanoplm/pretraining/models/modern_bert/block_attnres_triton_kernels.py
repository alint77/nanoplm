"""Fused Triton kernels for Block Attention Residuals.

Two-pass design for both forward and backward:
  Forward:  Pass 1 — D-reduction per source (RMS norm + dot → logits → softmax)
            Pass 2 — weighted sum of sources
  Backward: Pass 1 — D-reduction (grad_alpha, qw_dot → softmax bwd)
            Pass 2 — write grad_stacked + accumulate R for param grads

Each source is read exactly twice (once per pass) — the theoretical minimum.
N (number of sources) is tl.constexpr; Triton JIT-compiles a variant per N value.

Intermediates (alpha, inv_rms) are stored to global memory between passes.
At (N, T) scalars this is negligible bandwidth vs (N, T, D) source data.

No autotune — these are memory-bound kernels where fixed configs are optimal.
"""

import triton
import triton.language as tl

from .mhc_triton_kernels import _get_hw_config


@triton.jit
def _load_dense_tile(
    base_ptr,
    t_start,
    d_start,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block = tl.make_block_ptr(
        base=base_ptr,
        shape=(T, D),
        strides=(D, 1),
        offsets=(t_start, d_start),
        block_shape=(BLOCK_T, BLOCK_D),
        order=(1, 0),
    )
    return tl.load(block, boundary_check=(0, 1)).to(tl.float32)


@triton.jit
def _load_state_completed_tile(
    j: tl.constexpr,
    ptr0,
    ptr1,
    ptr2,
    ptr3,
    ptr4,
    ptr5,
    ptr6,
    ptr7,
    t_start,
    d_start,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    if j == 0:
        return _load_dense_tile(ptr0, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
    if j == 1:
        return _load_dense_tile(ptr1, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
    if j == 2:
        return _load_dense_tile(ptr2, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
    if j == 3:
        return _load_dense_tile(ptr3, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
    if j == 4:
        return _load_dense_tile(ptr4, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
    if j == 5:
        return _load_dense_tile(ptr5, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
    if j == 6:
        return _load_dense_tile(ptr6, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
    return _load_dense_tile(ptr7, t_start, d_start, T, D, BLOCK_T, BLOCK_D)


# ═══════════════════════════════════════════════════════════════════════════
# Forward kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _block_attnres_fwd_kernel(
    stacked_ptr,    # (N, T, D) contiguous
    qw_ptr,         # (D,) = query * norm_weight, precomputed
    result_ptr,     # (T, D) output
    alpha_ptr,      # (N, T) output — also used as scratch for logits
    inv_rms_ptr,    # (N, T) output
    T, D: tl.constexpr, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    TD = T * D

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # ── Pass 1: D-reduction per source → inv_rms, logits ─────────
        # Process each source sequentially.  Store inv_rms and logits
        # (in alpha_ptr as scratch) to global memory for softmax.
        for j in tl.static_range(N):
            sum_sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw = tl.zeros((BLOCK_T,), dtype=tl.float32)

            for d_start in tl.range(0, D, BLOCK_D):
                qw_tile = tl.load(
                    qw_ptr + d_start + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) + d_start < D,
                    other=0.0,
                ).to(tl.float32)

                src_block = tl.make_block_ptr(
                    base=stacked_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                src_tile = tl.load(src_block, boundary_check=(0, 1)).to(tl.float32)

                sum_sq += tl.sum(src_tile * src_tile, axis=1)
                dot_qw += tl.sum(src_tile * qw_tile[None, :], axis=1)

            ir_j = tl.rsqrt(sum_sq / D + eps)
            logit_j = ir_j * dot_qw
            tl.store(inv_rms_ptr + j * T + t_offs, ir_j, mask=t_mask)
            tl.store(alpha_ptr + j * T + t_offs, logit_j, mask=t_mask)  # scratch

        # ── Softmax over N logits ─────────────────────────────────────
        # Load logits back from alpha_ptr, compute stable softmax, overwrite.
        max_val = tl.full((BLOCK_T,), float('-inf'), dtype=tl.float32)
        for j in tl.static_range(N):
            logit_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=float('-inf'))
            max_val = tl.maximum(max_val, logit_j)

        sum_exp = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for j in tl.static_range(N):
            logit_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=float('-inf'))
            exp_j = tl.exp(logit_j - max_val)
            sum_exp += exp_j
            tl.store(alpha_ptr + j * T + t_offs, exp_j, mask=t_mask)  # unnormalized

        for j in tl.static_range(N):
            exp_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            alpha_j = exp_j / sum_exp
            tl.store(alpha_ptr + j * T + t_offs, alpha_j, mask=t_mask)

        # ── Pass 2: weighted sum ──────────────────────────────────────
        for d_start in tl.range(0, D, BLOCK_D):
            acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for j in tl.static_range(N):
                alpha_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)

                src_block = tl.make_block_ptr(
                    base=stacked_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                src_tile = tl.load(src_block, boundary_check=(0, 1)).to(tl.float32)
                acc += alpha_j[:, None] * src_tile

            out_block = tl.make_block_ptr(
                base=result_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            tl.store(out_block, acc.to(tl.bfloat16), boundary_check=(0, 1))


@triton.jit
def _block_attnres_state_fwd_kernel(
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    partial_ptr,
    qw_ptr,
    result_ptr,
    alpha_ptr,
    inv_rms_ptr,
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    PARTIAL_IDX = NC

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        for j in tl.static_range(NC):
            sum_sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw = tl.zeros((BLOCK_T,), dtype=tl.float32)

            for d_start in tl.range(0, D, BLOCK_D):
                qw_tile = tl.load(
                    qw_ptr + d_start + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) + d_start < D,
                    other=0.0,
                ).to(tl.float32)
                src_tile = _load_state_completed_tile(
                    j,
                    completed0_ptr,
                    completed1_ptr,
                    completed2_ptr,
                    completed3_ptr,
                    completed4_ptr,
                    completed5_ptr,
                    completed6_ptr,
                    completed7_ptr,
                    t_start,
                    d_start,
                    T,
                    D,
                    BLOCK_T,
                    BLOCK_D,
                )
                sum_sq += tl.sum(src_tile * src_tile, axis=1)
                dot_qw += tl.sum(src_tile * qw_tile[None, :], axis=1)

            ir_j = tl.rsqrt(sum_sq / D + eps)
            logit_j = ir_j * dot_qw
            tl.store(inv_rms_ptr + j * T + t_offs, ir_j, mask=t_mask)
            tl.store(alpha_ptr + j * T + t_offs, logit_j, mask=t_mask)

        sum_sq_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        dot_qw_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)
            partial_tile = _load_dense_tile(partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
            sum_sq_p += tl.sum(partial_tile * partial_tile, axis=1)
            dot_qw_p += tl.sum(partial_tile * qw_tile[None, :], axis=1)

        ir_p = tl.rsqrt(sum_sq_p / D + eps)
        logit_p = ir_p * dot_qw_p
        tl.store(inv_rms_ptr + PARTIAL_IDX * T + t_offs, ir_p, mask=t_mask)
        tl.store(alpha_ptr + PARTIAL_IDX * T + t_offs, logit_p, mask=t_mask)

        max_val = tl.full((BLOCK_T,), float("-inf"), dtype=tl.float32)
        for j in tl.static_range(NC + 1):
            logit_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=float("-inf"))
            max_val = tl.maximum(max_val, logit_j)

        sum_exp = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for j in tl.static_range(NC + 1):
            logit_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=float("-inf"))
            exp_j = tl.exp(logit_j - max_val)
            sum_exp += exp_j
            tl.store(alpha_ptr + j * T + t_offs, exp_j, mask=t_mask)

        for j in tl.static_range(NC + 1):
            exp_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            alpha_j = exp_j / sum_exp
            tl.store(alpha_ptr + j * T + t_offs, alpha_j, mask=t_mask)

        for d_start in tl.range(0, D, BLOCK_D):
            acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for j in tl.static_range(NC):
                alpha_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                src_tile = _load_state_completed_tile(
                    j,
                    completed0_ptr,
                    completed1_ptr,
                    completed2_ptr,
                    completed3_ptr,
                    completed4_ptr,
                    completed5_ptr,
                    completed6_ptr,
                    completed7_ptr,
                    t_start,
                    d_start,
                    T,
                    D,
                    BLOCK_T,
                    BLOCK_D,
                )
                acc += alpha_j[:, None] * src_tile

            alpha_p = tl.load(alpha_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0)
            partial_tile = _load_dense_tile(partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D)
            acc += alpha_p[:, None] * partial_tile

            out_block = tl.make_block_ptr(
                base=result_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            tl.store(out_block, acc.to(tl.bfloat16), boundary_check=(0, 1))


# ═══════════════════════════════════════════════════════════════════════════
# Backward kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _block_attnres_bwd_kernel(
    grad_result_ptr,  # (T, D)
    stacked_ptr,      # (N, T, D)
    qw_ptr,           # (D,) = query * norm_weight
    alpha_ptr,        # (N, T)
    inv_rms_ptr,      # (N, T)
    grad_stacked_ptr, # (N, T, D) output
    R_ptr,            # (D,) output — atomic accumulator for param grads
    # scratch buffers reused across passes:
    #   pass 1: grad_alpha_ptr=grad_alpha, qw_dot_ptr=qw_dot
    #   pass 2: grad_alpha_ptr=grad_logit, qw_dot_ptr=dot_term
    grad_alpha_ptr,   # (N, T) scratch
    qw_dot_ptr,       # (N, T) scratch
    T, D: tl.constexpr, N: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    TD = T * D

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # ── Pass 1: D-reduction per source → grad_alpha, qw_dot ──────
        # Zero the scratch rows once, then accumulate across D tiles. This
        # keeps grad_result/qw loads outside the source loop, which matters
        # much more than the tiny scratch traffic.
        for j in tl.static_range(N):
            zeros = tl.zeros((BLOCK_T,), dtype=tl.float32)
            tl.store(grad_alpha_ptr + j * T + t_offs, zeros, mask=t_mask)
            tl.store(qw_dot_ptr + j * T + t_offs, zeros, mask=t_mask)

        for d_start in tl.range(0, D, BLOCK_D):
            go_block = tl.make_block_ptr(
                base=grad_result_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            go_tile = tl.load(go_block, boundary_check=(0, 1)).to(tl.float32)

            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)

            for j in tl.static_range(N):
                ga_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                qwd_j = tl.load(qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                src_block = tl.make_block_ptr(
                    base=stacked_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                src_tile = tl.load(src_block, boundary_check=(0, 1)).to(tl.float32)

                ga_j += tl.sum(go_tile * src_tile, axis=1)
                qwd_j += tl.sum(qw_tile[None, :] * src_tile, axis=1)
                tl.store(grad_alpha_ptr + j * T + t_offs, ga_j, mask=t_mask)
                tl.store(qw_dot_ptr + j * T + t_offs, qwd_j, mask=t_mask)

        # ── Softmax backward ─────────────────────────────────────────
        # grad_logit[j] = alpha[j] * (grad_alpha[j] - v)
        # v = sum_j(alpha[j] * grad_alpha[j])
        v = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for j in tl.static_range(N):
            a_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ga_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            v += a_j * ga_j

        for j in tl.static_range(N):
            a_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ga_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ir_j = tl.load(inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            qwd_j = tl.load(qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0)

            gl_j = a_j * (ga_j - v)
            dt_j = gl_j * ir_j * qwd_j

            tl.store(grad_alpha_ptr + j * T + t_offs, gl_j, mask=t_mask)
            tl.store(qw_dot_ptr + j * T + t_offs, dt_j, mask=t_mask)

        # ── Pass 2: write grad_stacked + accumulate R ─────────────────
        for d_start in tl.range(0, D, BLOCK_D):
            go_block = tl.make_block_ptr(
                base=grad_result_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            go_tile = tl.load(go_block, boundary_check=(0, 1)).to(tl.float32)

            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)

            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            R_chunk = tl.zeros((BLOCK_D,), dtype=tl.float32)

            for j in tl.static_range(N):
                alpha_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                inv_rms_j = tl.load(inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                grad_logit_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                dot_term_j = tl.load(qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0)

                src_block = tl.make_block_ptr(
                    base=stacked_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                src_tile = tl.load(src_block, boundary_check=(0, 1)).to(tl.float32)

                # RMS-norm backward
                raw_j = src_tile * inv_rms_j[:, None]
                grad_raw = grad_logit_j[:, None] * qw_tile[None, :]
                gs_norm = inv_rms_j[:, None] * (grad_raw - raw_j * (dot_term_j / D)[:, None])

                # Weighted-sum backward
                gs_result = alpha_j[:, None] * go_tile

                gs_block = tl.make_block_ptr(
                    base=grad_stacked_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                tl.store(gs_block, (gs_norm + gs_result).to(tl.bfloat16), boundary_check=(0, 1))

                # Accumulate R for param grads
                contribution = (grad_logit_j * inv_rms_j)[:, None] * src_tile
                R_chunk += tl.sum(contribution, axis=0)

            tl.atomic_add(R_ptr + d_offs, R_chunk, mask=d_mask)


@triton.jit
def _block_attnres_state_bwd_kernel(
    grad_result_ptr,    # (T, D)
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    partial_ptr,        # (T, D)
    qw_ptr,             # (D,) = query * norm_weight
    alpha_ptr,          # (NC + 1, T)
    inv_rms_ptr,        # (NC + 1, T)
    grad_completed_ptr, # (NC, T, D) output
    grad_partial_ptr,   # (T, D) output
    R_ptr,              # (D,) output — atomic accumulator for param grads
    grad_alpha_ptr,     # (NC + 1, T) scratch
    qw_dot_ptr,         # (NC + 1, T) scratch
    T, D: tl.constexpr, NC: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    TD = T * D
    PARTIAL_IDX = NC

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        for j in tl.static_range(NC + 1):
            zeros = tl.zeros((BLOCK_T,), dtype=tl.float32)
            tl.store(grad_alpha_ptr + j * T + t_offs, zeros, mask=t_mask)
            tl.store(qw_dot_ptr + j * T + t_offs, zeros, mask=t_mask)

        for d_start in tl.range(0, D, BLOCK_D):
            go_block = tl.make_block_ptr(
                base=grad_result_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            go_tile = tl.load(go_block, boundary_check=(0, 1)).to(tl.float32)

            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)

            for j in tl.static_range(NC):
                ga_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                qwd_j = tl.load(qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                src_tile = _load_state_completed_tile(
                    j,
                    completed0_ptr,
                    completed1_ptr,
                    completed2_ptr,
                    completed3_ptr,
                    completed4_ptr,
                    completed5_ptr,
                    completed6_ptr,
                    completed7_ptr,
                    t_start,
                    d_start,
                    T,
                    D,
                    BLOCK_T,
                    BLOCK_D,
                )

                ga_j += tl.sum(go_tile * src_tile, axis=1)
                qwd_j += tl.sum(qw_tile[None, :] * src_tile, axis=1)
                tl.store(grad_alpha_ptr + j * T + t_offs, ga_j, mask=t_mask)
                tl.store(qw_dot_ptr + j * T + t_offs, qwd_j, mask=t_mask)

            ga_p = tl.load(grad_alpha_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0)
            qwd_p = tl.load(qw_dot_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0)
            partial_block = tl.make_block_ptr(
                base=partial_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            partial_tile = tl.load(partial_block, boundary_check=(0, 1)).to(tl.float32)

            ga_p += tl.sum(go_tile * partial_tile, axis=1)
            qwd_p += tl.sum(qw_tile[None, :] * partial_tile, axis=1)
            tl.store(grad_alpha_ptr + PARTIAL_IDX * T + t_offs, ga_p, mask=t_mask)
            tl.store(qw_dot_ptr + PARTIAL_IDX * T + t_offs, qwd_p, mask=t_mask)

        v = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for j in tl.static_range(NC + 1):
            a_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ga_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            v += a_j * ga_j

        for j in tl.static_range(NC + 1):
            a_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ga_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ir_j = tl.load(inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            qwd_j = tl.load(qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0)

            gl_j = a_j * (ga_j - v)
            dt_j = gl_j * ir_j * qwd_j

            tl.store(grad_alpha_ptr + j * T + t_offs, gl_j, mask=t_mask)
            tl.store(qw_dot_ptr + j * T + t_offs, dt_j, mask=t_mask)

        for d_start in tl.range(0, D, BLOCK_D):
            go_block = tl.make_block_ptr(
                base=grad_result_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            go_tile = tl.load(go_block, boundary_check=(0, 1)).to(tl.float32)

            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)

            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            R_chunk = tl.zeros((BLOCK_D,), dtype=tl.float32)

            for j in tl.static_range(NC):
                alpha_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                inv_rms_j = tl.load(inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                grad_logit_j = tl.load(grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                dot_term_j = tl.load(qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0)

                src_tile = _load_state_completed_tile(
                    j,
                    completed0_ptr,
                    completed1_ptr,
                    completed2_ptr,
                    completed3_ptr,
                    completed4_ptr,
                    completed5_ptr,
                    completed6_ptr,
                    completed7_ptr,
                    t_start,
                    d_start,
                    T,
                    D,
                    BLOCK_T,
                    BLOCK_D,
                )

                raw_j = src_tile * inv_rms_j[:, None]
                grad_raw = grad_logit_j[:, None] * qw_tile[None, :]
                gs_norm = inv_rms_j[:, None] * (grad_raw - raw_j * (dot_term_j / D)[:, None])
                gs_result = alpha_j[:, None] * go_tile

                gs_block = tl.make_block_ptr(
                    base=grad_completed_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                tl.store(gs_block, (gs_norm + gs_result).to(tl.bfloat16), boundary_check=(0, 1))

                contribution = (grad_logit_j * inv_rms_j)[:, None] * src_tile
                R_chunk += tl.sum(contribution, axis=0)

            alpha_p = tl.load(alpha_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0)
            inv_rms_p = tl.load(inv_rms_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0)
            grad_logit_p = tl.load(grad_alpha_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0)
            dot_term_p = tl.load(qw_dot_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0)

            partial_block = tl.make_block_ptr(
                base=partial_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            partial_tile = tl.load(partial_block, boundary_check=(0, 1)).to(tl.float32)

            raw_p = partial_tile * inv_rms_p[:, None]
            grad_raw_p = grad_logit_p[:, None] * qw_tile[None, :]
            gs_norm_p = inv_rms_p[:, None] * (grad_raw_p - raw_p * (dot_term_p / D)[:, None])
            gs_result_p = alpha_p[:, None] * go_tile

            grad_partial_block = tl.make_block_ptr(
                base=grad_partial_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            tl.store(
                grad_partial_block,
                (gs_norm_p + gs_result_p).to(tl.bfloat16),
                boundary_check=(0, 1),
            )

            contribution_p = (grad_logit_p * inv_rms_p)[:, None] * partial_tile
            R_chunk += tl.sum(contribution_p, axis=0)
            tl.atomic_add(R_ptr + d_offs, R_chunk, mask=d_mask)
