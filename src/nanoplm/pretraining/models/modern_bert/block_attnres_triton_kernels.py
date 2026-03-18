"""Fused Triton kernels for Block Attention Residuals.

Two-pass design for both forward and backward:
  Forward:  Pass 1 — D-reduction per source (RMS norm + dot → logits → softmax)
            Pass 2 — weighted sum of sources
  Backward: Pass 1 — D-reduction (grad_alpha, qw_dot → softmax bwd)
            Pass 2 — write grad_stacked + accumulate R for param grads

Each source is read exactly twice (once per pass) — the theoretical minimum.
N (number of sources) is tl.constexpr; Triton JIT-compiles a variant per N value.

Forward kernels keep softmax alpha values in registers between Pass 1 and
Pass 2 — alpha_ptr / inv_rms_ptr are written as outputs (for the backward
pass) but never re-read within the forward kernel.  This avoids a Triton
3.x compiler bug on SM120 where store-then-load to the same global address
within a persistent-kernel loop can return stale values.

Phase 1 batched D-reduction:
  _block_attnres_batched_completed_dreduction_kernel reads each completed ref
  once for Q sublayers' D-reduction work, producing shared inv_rms (NC, T) and
  per-query logits (Q, NC, T).

  _block_attnres_state_fwd_precomputed_kernel uses precomputed completed
  logits/inv_rms, only computing D-reduction for the partial source.

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
    stacked_ptr,  # (N, T, D) contiguous
    qw_ptr,  # (D,) = query * norm_weight, precomputed
    result_ptr,  # (T, D) output
    alpha_ptr,  # (N, T) output (written once, never re-read)
    inv_rms_ptr,  # (N, T) output
    T,
    D: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
        # Logits are kept in register variables (never re-read from global
        # memory) to avoid SM120 store→load non-determinism.
        logit_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_3 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_4 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_5 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_6 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_7 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_8 = tl.zeros((BLOCK_T,), dtype=tl.float32)

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

            # Save logit to register variable (static if-chain)
            if j == 0:
                logit_0 = logit_j
            if j == 1:
                logit_1 = logit_j
            if j == 2:
                logit_2 = logit_j
            if j == 3:
                logit_3 = logit_j
            if j == 4:
                logit_4 = logit_j
            if j == 5:
                logit_5 = logit_j
            if j == 6:
                logit_6 = logit_j
            if j == 7:
                logit_7 = logit_j
            if j == 8:
                logit_8 = logit_j

        # ── Softmax over N logits (entirely in registers) ─────────────
        max_val = logit_0
        if N > 1:
            max_val = tl.maximum(max_val, logit_1)
        if N > 2:
            max_val = tl.maximum(max_val, logit_2)
        if N > 3:
            max_val = tl.maximum(max_val, logit_3)
        if N > 4:
            max_val = tl.maximum(max_val, logit_4)
        if N > 5:
            max_val = tl.maximum(max_val, logit_5)
        if N > 6:
            max_val = tl.maximum(max_val, logit_6)
        if N > 7:
            max_val = tl.maximum(max_val, logit_7)
        if N > 8:
            max_val = tl.maximum(max_val, logit_8)

        exp_0 = tl.exp(logit_0 - max_val)
        exp_1 = (
            tl.exp(logit_1 - max_val)
            if N > 1
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_2 = (
            tl.exp(logit_2 - max_val)
            if N > 2
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_3 = (
            tl.exp(logit_3 - max_val)
            if N > 3
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_4 = (
            tl.exp(logit_4 - max_val)
            if N > 4
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_5 = (
            tl.exp(logit_5 - max_val)
            if N > 5
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_6 = (
            tl.exp(logit_6 - max_val)
            if N > 6
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_7 = (
            tl.exp(logit_7 - max_val)
            if N > 7
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_8 = (
            tl.exp(logit_8 - max_val)
            if N > 8
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )

        sum_exp = exp_0
        if N > 1:
            sum_exp += exp_1
        if N > 2:
            sum_exp += exp_2
        if N > 3:
            sum_exp += exp_3
        if N > 4:
            sum_exp += exp_4
        if N > 5:
            sum_exp += exp_5
        if N > 6:
            sum_exp += exp_6
        if N > 7:
            sum_exp += exp_7
        if N > 8:
            sum_exp += exp_8

        alpha_0 = exp_0 / sum_exp
        alpha_1 = exp_1 / sum_exp if N > 1 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        alpha_2 = exp_2 / sum_exp if N > 2 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        alpha_3 = exp_3 / sum_exp if N > 3 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        alpha_4 = exp_4 / sum_exp if N > 4 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        alpha_5 = exp_5 / sum_exp if N > 5 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        alpha_6 = exp_6 / sum_exp if N > 6 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        alpha_7 = exp_7 / sum_exp if N > 7 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        alpha_8 = exp_8 / sum_exp if N > 8 else tl.zeros((BLOCK_T,), dtype=tl.float32)

        # Write alpha to output (for backward pass — never re-read here)
        if N > 0:
            tl.store(alpha_ptr + 0 * T + t_offs, alpha_0, mask=t_mask)
        if N > 1:
            tl.store(alpha_ptr + 1 * T + t_offs, alpha_1, mask=t_mask)
        if N > 2:
            tl.store(alpha_ptr + 2 * T + t_offs, alpha_2, mask=t_mask)
        if N > 3:
            tl.store(alpha_ptr + 3 * T + t_offs, alpha_3, mask=t_mask)
        if N > 4:
            tl.store(alpha_ptr + 4 * T + t_offs, alpha_4, mask=t_mask)
        if N > 5:
            tl.store(alpha_ptr + 5 * T + t_offs, alpha_5, mask=t_mask)
        if N > 6:
            tl.store(alpha_ptr + 6 * T + t_offs, alpha_6, mask=t_mask)
        if N > 7:
            tl.store(alpha_ptr + 7 * T + t_offs, alpha_7, mask=t_mask)
        if N > 8:
            tl.store(alpha_ptr + 8 * T + t_offs, alpha_8, mask=t_mask)

        # ── Pass 2: weighted sum (alpha read from registers) ──────────
        for d_start in tl.range(0, D, BLOCK_D):
            acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for j in tl.static_range(N):
                # Select alpha from register variable
                if j == 0:
                    a_j = alpha_0
                if j == 1:
                    a_j = alpha_1
                if j == 2:
                    a_j = alpha_2
                if j == 3:
                    a_j = alpha_3
                if j == 4:
                    a_j = alpha_4
                if j == 5:
                    a_j = alpha_5
                if j == 6:
                    a_j = alpha_6
                if j == 7:
                    a_j = alpha_7
                if j == 8:
                    a_j = alpha_8

                src_block = tl.make_block_ptr(
                    base=stacked_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                src_tile = tl.load(src_block, boundary_check=(0, 1)).to(tl.float32)
                acc += a_j[:, None] * src_tile

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

        # ── Pass 1: D-reduction → inv_rms, logits (in registers) ─────
        # Logits kept in register variables to avoid SM120 store→load bug.
        logit_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_3 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_4 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_5 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_6 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_7 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_8 = tl.zeros((BLOCK_T,), dtype=tl.float32)

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

            # Save logit to register variable
            if j == 0:
                logit_0 = logit_j
            if j == 1:
                logit_1 = logit_j
            if j == 2:
                logit_2 = logit_j
            if j == 3:
                logit_3 = logit_j
            if j == 4:
                logit_4 = logit_j
            if j == 5:
                logit_5 = logit_j
            if j == 6:
                logit_6 = logit_j
            if j == 7:
                logit_7 = logit_j

        # Partial source
        sum_sq_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        dot_qw_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)
            partial_tile = _load_dense_tile(
                partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D
            )
            sum_sq_p += tl.sum(partial_tile * partial_tile, axis=1)
            dot_qw_p += tl.sum(partial_tile * qw_tile[None, :], axis=1)

        ir_p = tl.rsqrt(sum_sq_p / D + eps)
        logit_p = ir_p * dot_qw_p
        tl.store(inv_rms_ptr + PARTIAL_IDX * T + t_offs, ir_p, mask=t_mask)

        # Save partial logit to register (index NC)
        if NC == 0:
            logit_0 = logit_p
        if NC == 1:
            logit_1 = logit_p
        if NC == 2:
            logit_2 = logit_p
        if NC == 3:
            logit_3 = logit_p
        if NC == 4:
            logit_4 = logit_p
        if NC == 5:
            logit_5 = logit_p
        if NC == 6:
            logit_6 = logit_p
        if NC == 7:
            logit_7 = logit_p
        if NC == 8:
            logit_8 = logit_p

        # ── Softmax over NC + 1 logits (entirely in registers) ────────
        max_val = logit_0
        if NC + 1 > 1:
            max_val = tl.maximum(max_val, logit_1)
        if NC + 1 > 2:
            max_val = tl.maximum(max_val, logit_2)
        if NC + 1 > 3:
            max_val = tl.maximum(max_val, logit_3)
        if NC + 1 > 4:
            max_val = tl.maximum(max_val, logit_4)
        if NC + 1 > 5:
            max_val = tl.maximum(max_val, logit_5)
        if NC + 1 > 6:
            max_val = tl.maximum(max_val, logit_6)
        if NC + 1 > 7:
            max_val = tl.maximum(max_val, logit_7)
        if NC + 1 > 8:
            max_val = tl.maximum(max_val, logit_8)

        exp_0 = tl.exp(logit_0 - max_val)
        exp_1 = (
            tl.exp(logit_1 - max_val)
            if NC + 1 > 1
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_2 = (
            tl.exp(logit_2 - max_val)
            if NC + 1 > 2
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_3 = (
            tl.exp(logit_3 - max_val)
            if NC + 1 > 3
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_4 = (
            tl.exp(logit_4 - max_val)
            if NC + 1 > 4
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_5 = (
            tl.exp(logit_5 - max_val)
            if NC + 1 > 5
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_6 = (
            tl.exp(logit_6 - max_val)
            if NC + 1 > 6
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_7 = (
            tl.exp(logit_7 - max_val)
            if NC + 1 > 7
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_8 = (
            tl.exp(logit_8 - max_val)
            if NC + 1 > 8
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )

        sum_exp = exp_0
        if NC + 1 > 1:
            sum_exp += exp_1
        if NC + 1 > 2:
            sum_exp += exp_2
        if NC + 1 > 3:
            sum_exp += exp_3
        if NC + 1 > 4:
            sum_exp += exp_4
        if NC + 1 > 5:
            sum_exp += exp_5
        if NC + 1 > 6:
            sum_exp += exp_6
        if NC + 1 > 7:
            sum_exp += exp_7
        if NC + 1 > 8:
            sum_exp += exp_8

        alpha_0 = exp_0 / sum_exp
        alpha_1 = (
            exp_1 / sum_exp if NC + 1 > 1 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_2 = (
            exp_2 / sum_exp if NC + 1 > 2 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_3 = (
            exp_3 / sum_exp if NC + 1 > 3 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_4 = (
            exp_4 / sum_exp if NC + 1 > 4 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_5 = (
            exp_5 / sum_exp if NC + 1 > 5 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_6 = (
            exp_6 / sum_exp if NC + 1 > 6 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_7 = (
            exp_7 / sum_exp if NC + 1 > 7 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_8 = (
            exp_8 / sum_exp if NC + 1 > 8 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )

        # Write alpha to output (for backward — never re-read here)
        if NC + 1 > 0:
            tl.store(alpha_ptr + 0 * T + t_offs, alpha_0, mask=t_mask)
        if NC + 1 > 1:
            tl.store(alpha_ptr + 1 * T + t_offs, alpha_1, mask=t_mask)
        if NC + 1 > 2:
            tl.store(alpha_ptr + 2 * T + t_offs, alpha_2, mask=t_mask)
        if NC + 1 > 3:
            tl.store(alpha_ptr + 3 * T + t_offs, alpha_3, mask=t_mask)
        if NC + 1 > 4:
            tl.store(alpha_ptr + 4 * T + t_offs, alpha_4, mask=t_mask)
        if NC + 1 > 5:
            tl.store(alpha_ptr + 5 * T + t_offs, alpha_5, mask=t_mask)
        if NC + 1 > 6:
            tl.store(alpha_ptr + 6 * T + t_offs, alpha_6, mask=t_mask)
        if NC + 1 > 7:
            tl.store(alpha_ptr + 7 * T + t_offs, alpha_7, mask=t_mask)
        if NC + 1 > 8:
            tl.store(alpha_ptr + 8 * T + t_offs, alpha_8, mask=t_mask)

        # ── Pass 2: weighted sum (alpha from registers) ───────────────
        for d_start in tl.range(0, D, BLOCK_D):
            acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for j in tl.static_range(NC):
                # Select alpha from register
                if j == 0:
                    a_j = alpha_0
                if j == 1:
                    a_j = alpha_1
                if j == 2:
                    a_j = alpha_2
                if j == 3:
                    a_j = alpha_3
                if j == 4:
                    a_j = alpha_4
                if j == 5:
                    a_j = alpha_5
                if j == 6:
                    a_j = alpha_6
                if j == 7:
                    a_j = alpha_7

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
                acc += a_j[:, None] * src_tile

            # Partial alpha from register
            if NC == 0:
                a_p = alpha_0
            if NC == 1:
                a_p = alpha_1
            if NC == 2:
                a_p = alpha_2
            if NC == 3:
                a_p = alpha_3
            if NC == 4:
                a_p = alpha_4
            if NC == 5:
                a_p = alpha_5
            if NC == 6:
                a_p = alpha_6
            if NC == 7:
                a_p = alpha_7
            if NC == 8:
                a_p = alpha_8

            partial_tile = _load_dense_tile(
                partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D
            )
            acc += a_p[:, None] * partial_tile

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
# Phase 1: Batched completed-block D-reduction
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _load_qw_tile(
    q: tl.constexpr,
    qw0_ptr,
    qw1_ptr,
    qw2_ptr,
    qw3_ptr,
    qw4_ptr,
    qw5_ptr,
    qw6_ptr,
    qw7_ptr,
    d_start,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Load a (BLOCK_D,) tile from the q-th qw pointer via static dispatch."""
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    if q == 0:
        return tl.load(qw0_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    if q == 1:
        return tl.load(qw1_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    if q == 2:
        return tl.load(qw2_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    if q == 3:
        return tl.load(qw3_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    if q == 4:
        return tl.load(qw4_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    if q == 5:
        return tl.load(qw5_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    if q == 6:
        return tl.load(qw6_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    return tl.load(qw7_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)


@triton.jit
def _block_attnres_batched_completed_dreduction_kernel(
    # Completed ref pointers (up to 8)
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    # Per-query qw pointers (up to 8)
    qw0_ptr,
    qw1_ptr,
    qw2_ptr,
    qw3_ptr,
    qw4_ptr,
    qw5_ptr,
    qw6_ptr,
    qw7_ptr,
    # Outputs
    logits_ptr,  # (Q, NC, T) output
    inv_rms_ptr,  # (NC, T) output — shared across queries
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    Q: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Batched completed-block D-reduction for Phase 1.

    Reads each completed ref once, computing:
      - shared inv_rms[j, t] = rsqrt(sum(src_j^2) / D + eps)  for j in [0, NC)
      - logits[q, j, t] = inv_rms[j, t] * dot(src_j, qw_q)   for q in [0, Q)

    Register pressure: Q scalar (BLOCK_T,) accumulators on top of one (BLOCK_T, BLOCK_D)
    source tile. For Q=4, BLOCK_T=32: 4*32 + 32 = 160 extra floats. Trivial.
    """
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    NC_T = NC * T  # stride for Q dimension in logits

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        for j in tl.static_range(NC):
            # Shared accumulator for inv_rms
            sum_sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
            # Per-query accumulators for dot products
            dot_qw_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw_2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw_3 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw_4 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw_5 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw_6 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            dot_qw_7 = tl.zeros((BLOCK_T,), dtype=tl.float32)

            for d_start in tl.range(0, D, BLOCK_D):
                # Load source tile once — shared across all Q queries
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

                # Accumulate Q dot products — one qw load per query
                if Q > 0:
                    qw_tile = _load_qw_tile(
                        0,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_0 += tl.sum(src_tile * qw_tile[None, :], axis=1)
                if Q > 1:
                    qw_tile = _load_qw_tile(
                        1,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_1 += tl.sum(src_tile * qw_tile[None, :], axis=1)
                if Q > 2:
                    qw_tile = _load_qw_tile(
                        2,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_2 += tl.sum(src_tile * qw_tile[None, :], axis=1)
                if Q > 3:
                    qw_tile = _load_qw_tile(
                        3,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_3 += tl.sum(src_tile * qw_tile[None, :], axis=1)
                if Q > 4:
                    qw_tile = _load_qw_tile(
                        4,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_4 += tl.sum(src_tile * qw_tile[None, :], axis=1)
                if Q > 5:
                    qw_tile = _load_qw_tile(
                        5,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_5 += tl.sum(src_tile * qw_tile[None, :], axis=1)
                if Q > 6:
                    qw_tile = _load_qw_tile(
                        6,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_6 += tl.sum(src_tile * qw_tile[None, :], axis=1)
                if Q > 7:
                    qw_tile = _load_qw_tile(
                        7,
                        qw0_ptr,
                        qw1_ptr,
                        qw2_ptr,
                        qw3_ptr,
                        qw4_ptr,
                        qw5_ptr,
                        qw6_ptr,
                        qw7_ptr,
                        d_start,
                        D,
                        BLOCK_D,
                    )
                    dot_qw_7 += tl.sum(src_tile * qw_tile[None, :], axis=1)

            # Compute shared inv_rms and per-query logits
            ir_j = tl.rsqrt(sum_sq / D + eps)
            tl.store(inv_rms_ptr + j * T + t_offs, ir_j, mask=t_mask)

            # Store logits[q, j, t] = inv_rms_j * dot_qw_q
            # Layout: logits_ptr shape (Q, NC, T), stride = (NC*T, T, 1)
            if Q > 0:
                tl.store(
                    logits_ptr + 0 * NC_T + j * T + t_offs, ir_j * dot_qw_0, mask=t_mask
                )
            if Q > 1:
                tl.store(
                    logits_ptr + 1 * NC_T + j * T + t_offs, ir_j * dot_qw_1, mask=t_mask
                )
            if Q > 2:
                tl.store(
                    logits_ptr + 2 * NC_T + j * T + t_offs, ir_j * dot_qw_2, mask=t_mask
                )
            if Q > 3:
                tl.store(
                    logits_ptr + 3 * NC_T + j * T + t_offs, ir_j * dot_qw_3, mask=t_mask
                )
            if Q > 4:
                tl.store(
                    logits_ptr + 4 * NC_T + j * T + t_offs, ir_j * dot_qw_4, mask=t_mask
                )
            if Q > 5:
                tl.store(
                    logits_ptr + 5 * NC_T + j * T + t_offs, ir_j * dot_qw_5, mask=t_mask
                )
            if Q > 6:
                tl.store(
                    logits_ptr + 6 * NC_T + j * T + t_offs, ir_j * dot_qw_6, mask=t_mask
                )
            if Q > 7:
                tl.store(
                    logits_ptr + 7 * NC_T + j * T + t_offs, ir_j * dot_qw_7, mask=t_mask
                )


@triton.jit
def _block_attnres_state_fwd_precomputed_kernel(
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    partial_ptr,
    qw_ptr,  # (D,) for THIS sublayer only
    precomputed_logits_ptr,  # (NC, T) — this sublayer's completed logits
    precomputed_inv_rms_ptr,  # (NC, T) — shared completed inv_rms
    result_ptr,
    alpha_ptr,  # (NC + 1, T) output
    inv_rms_ptr,  # (NC + 1, T) output
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Forward state kernel with precomputed completed-ref logits/inv_rms.

    Same as _block_attnres_state_fwd_kernel but skips pass 1 for completed
    refs — reads precomputed logits and inv_rms from the batched D-reduction.
    Only computes D-reduction for the partial source.  Pass 2 (weighted sum)
    is unchanged: still reads completed sources per sublayer.
    """
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    PARTIAL_IDX = NC

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # ── Load precomputed completed logits/inv_rms into registers ──
        logit_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_3 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_4 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_5 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_6 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_7 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        logit_8 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        for j in tl.static_range(NC):
            logit_j = tl.load(
                precomputed_logits_ptr + j * T + t_offs, mask=t_mask, other=0.0
            )
            ir_j = tl.load(
                precomputed_inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0
            )
            tl.store(inv_rms_ptr + j * T + t_offs, ir_j, mask=t_mask)

            # Save logit to register variable
            if j == 0:
                logit_0 = logit_j
            if j == 1:
                logit_1 = logit_j
            if j == 2:
                logit_2 = logit_j
            if j == 3:
                logit_3 = logit_j
            if j == 4:
                logit_4 = logit_j
            if j == 5:
                logit_5 = logit_j
            if j == 6:
                logit_6 = logit_j
            if j == 7:
                logit_7 = logit_j

        # ── Pass 1 for partial only ───────────────────────────────────
        sum_sq_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        dot_qw_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)
            partial_tile = _load_dense_tile(
                partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D
            )
            sum_sq_p += tl.sum(partial_tile * partial_tile, axis=1)
            dot_qw_p += tl.sum(partial_tile * qw_tile[None, :], axis=1)

        ir_p = tl.rsqrt(sum_sq_p / D + eps)
        logit_p = ir_p * dot_qw_p
        tl.store(inv_rms_ptr + PARTIAL_IDX * T + t_offs, ir_p, mask=t_mask)

        # Save partial logit to register (index NC)
        if NC == 0:
            logit_0 = logit_p
        if NC == 1:
            logit_1 = logit_p
        if NC == 2:
            logit_2 = logit_p
        if NC == 3:
            logit_3 = logit_p
        if NC == 4:
            logit_4 = logit_p
        if NC == 5:
            logit_5 = logit_p
        if NC == 6:
            logit_6 = logit_p
        if NC == 7:
            logit_7 = logit_p
        if NC == 8:
            logit_8 = logit_p

        # ── Softmax over NC + 1 logits (entirely in registers) ────────
        max_val = logit_0
        if NC + 1 > 1:
            max_val = tl.maximum(max_val, logit_1)
        if NC + 1 > 2:
            max_val = tl.maximum(max_val, logit_2)
        if NC + 1 > 3:
            max_val = tl.maximum(max_val, logit_3)
        if NC + 1 > 4:
            max_val = tl.maximum(max_val, logit_4)
        if NC + 1 > 5:
            max_val = tl.maximum(max_val, logit_5)
        if NC + 1 > 6:
            max_val = tl.maximum(max_val, logit_6)
        if NC + 1 > 7:
            max_val = tl.maximum(max_val, logit_7)
        if NC + 1 > 8:
            max_val = tl.maximum(max_val, logit_8)

        exp_0 = tl.exp(logit_0 - max_val)
        exp_1 = (
            tl.exp(logit_1 - max_val)
            if NC + 1 > 1
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_2 = (
            tl.exp(logit_2 - max_val)
            if NC + 1 > 2
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_3 = (
            tl.exp(logit_3 - max_val)
            if NC + 1 > 3
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_4 = (
            tl.exp(logit_4 - max_val)
            if NC + 1 > 4
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_5 = (
            tl.exp(logit_5 - max_val)
            if NC + 1 > 5
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_6 = (
            tl.exp(logit_6 - max_val)
            if NC + 1 > 6
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_7 = (
            tl.exp(logit_7 - max_val)
            if NC + 1 > 7
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        exp_8 = (
            tl.exp(logit_8 - max_val)
            if NC + 1 > 8
            else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )

        sum_exp = exp_0
        if NC + 1 > 1:
            sum_exp += exp_1
        if NC + 1 > 2:
            sum_exp += exp_2
        if NC + 1 > 3:
            sum_exp += exp_3
        if NC + 1 > 4:
            sum_exp += exp_4
        if NC + 1 > 5:
            sum_exp += exp_5
        if NC + 1 > 6:
            sum_exp += exp_6
        if NC + 1 > 7:
            sum_exp += exp_7
        if NC + 1 > 8:
            sum_exp += exp_8

        alpha_0 = exp_0 / sum_exp
        alpha_1 = (
            exp_1 / sum_exp if NC + 1 > 1 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_2 = (
            exp_2 / sum_exp if NC + 1 > 2 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_3 = (
            exp_3 / sum_exp if NC + 1 > 3 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_4 = (
            exp_4 / sum_exp if NC + 1 > 4 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_5 = (
            exp_5 / sum_exp if NC + 1 > 5 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_6 = (
            exp_6 / sum_exp if NC + 1 > 6 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_7 = (
            exp_7 / sum_exp if NC + 1 > 7 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )
        alpha_8 = (
            exp_8 / sum_exp if NC + 1 > 8 else tl.zeros((BLOCK_T,), dtype=tl.float32)
        )

        # Write alpha to output (for backward — never re-read here)
        if NC + 1 > 0:
            tl.store(alpha_ptr + 0 * T + t_offs, alpha_0, mask=t_mask)
        if NC + 1 > 1:
            tl.store(alpha_ptr + 1 * T + t_offs, alpha_1, mask=t_mask)
        if NC + 1 > 2:
            tl.store(alpha_ptr + 2 * T + t_offs, alpha_2, mask=t_mask)
        if NC + 1 > 3:
            tl.store(alpha_ptr + 3 * T + t_offs, alpha_3, mask=t_mask)
        if NC + 1 > 4:
            tl.store(alpha_ptr + 4 * T + t_offs, alpha_4, mask=t_mask)
        if NC + 1 > 5:
            tl.store(alpha_ptr + 5 * T + t_offs, alpha_5, mask=t_mask)
        if NC + 1 > 6:
            tl.store(alpha_ptr + 6 * T + t_offs, alpha_6, mask=t_mask)
        if NC + 1 > 7:
            tl.store(alpha_ptr + 7 * T + t_offs, alpha_7, mask=t_mask)
        if NC + 1 > 8:
            tl.store(alpha_ptr + 8 * T + t_offs, alpha_8, mask=t_mask)

        # ── Pass 2: weighted sum (alpha from registers) ───────────────
        for d_start in tl.range(0, D, BLOCK_D):
            acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for j in tl.static_range(NC):
                # Select alpha from register
                if j == 0:
                    a_j = alpha_0
                if j == 1:
                    a_j = alpha_1
                if j == 2:
                    a_j = alpha_2
                if j == 3:
                    a_j = alpha_3
                if j == 4:
                    a_j = alpha_4
                if j == 5:
                    a_j = alpha_5
                if j == 6:
                    a_j = alpha_6
                if j == 7:
                    a_j = alpha_7

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
                acc += a_j[:, None] * src_tile

            # Partial alpha from register
            if NC == 0:
                a_p = alpha_0
            if NC == 1:
                a_p = alpha_1
            if NC == 2:
                a_p = alpha_2
            if NC == 3:
                a_p = alpha_3
            if NC == 4:
                a_p = alpha_4
            if NC == 5:
                a_p = alpha_5
            if NC == 6:
                a_p = alpha_6
            if NC == 7:
                a_p = alpha_7
            if NC == 8:
                a_p = alpha_8

            partial_tile = _load_dense_tile(
                partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D
            )
            acc += a_p[:, None] * partial_tile

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
# Phase 2: Online partial merge — completed weighted sum + merge kernels
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _block_attnres_completed_wsum_kernel(
    # Completed ref pointers (up to 8)
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    # Precomputed from Phase 1 batched D-reduction
    precomputed_logits_ptr,  # (NC, T) — this query's completed logits
    precomputed_inv_rms_ptr,  # (NC, T) — shared completed inv_rms
    # Running-state outputs
    running_m_ptr,  # (T,) output — running max logit
    running_l_ptr,  # (T,) output — running sum-exp
    running_acc_ptr,  # (T, D) output — weighted accumulator
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Completed-block weighted sum with online softmax (Phase 2).

    Reads each completed ref once, producing per-query running state
    (m, l, acc) using precomputed logits from the Phase 1 batched
    D-reduction.  The running state captures the partial softmax over
    just the NC completed sources — the per-sublayer merge_partial
    kernel will incorporate the partial source contribution.

    Online softmax update for source j:
        new_m = max(m, logit_j)
        correction = exp(m - new_m)
        exp_j = exp(logit_j - new_m)
        acc = acc * correction + exp_j[:, None] * src_j    (per D-tile)
        l = l * correction + exp_j
        m = new_m
    """
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # Initialise running state: m = -inf, l = 0
        # acc is initialised per D-tile below (written out each D-tile)
        NEG_INF = -1e30
        m = tl.full((BLOCK_T,), NEG_INF, dtype=tl.float32)
        l = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # ── Process completed sources with online softmax ─────────────
        # Two nested loops: outer over sources, inner over D-tiles.
        # For each source we first load its logit, update (m, l), then
        # accumulate the weighted contribution across D-tiles.
        for j in tl.static_range(NC):
            # Load precomputed logit for this source
            logit_j = tl.load(
                precomputed_logits_ptr + j * T + t_offs, mask=t_mask, other=NEG_INF
            )

            # Online softmax update
            new_m = tl.maximum(m, logit_j)
            correction = tl.exp(m - new_m)
            exp_j = tl.exp(logit_j - new_m)

            l = l * correction + exp_j
            m = new_m

            # Accumulate weighted source across D-tiles
            for d_start in tl.range(0, D, BLOCK_D):
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

                # Load current acc tile
                acc_block = tl.make_block_ptr(
                    base=running_acc_ptr,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                if j == 0:
                    # First source — no previous acc to load/correct
                    new_acc = exp_j[:, None] * src_tile
                else:
                    old_acc = tl.load(acc_block, boundary_check=(0, 1))
                    new_acc = old_acc * correction[:, None] + exp_j[:, None] * src_tile

                tl.store(acc_block, new_acc, boundary_check=(0, 1))

        # ── Store running state ───────────────────────────────────────
        tl.store(running_m_ptr + t_offs, m, mask=t_mask)
        tl.store(running_l_ptr + t_offs, l, mask=t_mask)


@triton.jit
def _block_attnres_batched_completed_wsum_kernel(
    # Completed ref pointers (up to 8)
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    # Precomputed logits/inv_rms from Phase 1 batched D-reduction
    precomputed_logits_ptr,  # (Q_BATCH, NC, T) — logits for this batch of queries
    precomputed_inv_rms_ptr,  # (NC, T) — shared completed inv_rms (unused for wsum, kept for API symmetry)
    # Running-state outputs — (Q_BATCH, T) for m/l, (Q_BATCH, T, D) for acc
    running_m_ptr,  # (Q_BATCH, T) output
    running_l_ptr,  # (Q_BATCH, T) output
    running_acc_ptr,  # (Q_BATCH, T, D) output
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    Q_BATCH: tl.constexpr,  # 1 or 2
    eps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Batched completed-block weighted sum with online softmax (Phase 3).

    Processes Q_BATCH queries simultaneously over NC completed sources.
    Each source is read once per T-tile per D-tile — shared across Q_BATCH
    queries.  Uses precomputed logits from Phase 1 batched D-reduction.

    At Q_BATCH=2, BLOCK_D=128, BLOCK_T=32: each query holds a
    (32, 128) = 4096 f32 accumulator.  Two queries = 8192 f32 values —
    same register footprint as the single-query kernel at BLOCK_D=256.

    Online softmax update for source j, query q:
        new_m_q = max(m_q, logit_j_q)
        correction_q = exp(m_q - new_m_q)
        exp_j_q = exp(logit_j_q - new_m_q)
        acc_q = acc_q * correction_q + exp_j_q[:, None] * src_j
        l_q = l_q * correction_q + exp_j_q
        m_q = new_m_q
    """
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    NC_T = NC * T  # stride for Q dimension in logits layout (Q_BATCH, NC, T)

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # Initialise per-query running state: m = -inf, l = 0
        NEG_INF = -1e30
        m_0 = tl.full((BLOCK_T,), NEG_INF, dtype=tl.float32)
        l_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        if Q_BATCH > 1:
            m_1 = tl.full((BLOCK_T,), NEG_INF, dtype=tl.float32)
            l_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # ── Process completed sources with online softmax ─────────────
        for j in tl.static_range(NC):
            # Load precomputed logits for this source, per query
            logit_j_0 = tl.load(
                precomputed_logits_ptr + 0 * NC_T + j * T + t_offs,
                mask=t_mask,
                other=NEG_INF,
            )
            # Online softmax update — query 0
            new_m_0 = tl.maximum(m_0, logit_j_0)
            correction_0 = tl.exp(m_0 - new_m_0)
            exp_j_0 = tl.exp(logit_j_0 - new_m_0)
            l_0 = l_0 * correction_0 + exp_j_0
            m_0 = new_m_0

            if Q_BATCH > 1:
                logit_j_1 = tl.load(
                    precomputed_logits_ptr + 1 * NC_T + j * T + t_offs,
                    mask=t_mask,
                    other=NEG_INF,
                )
                new_m_1 = tl.maximum(m_1, logit_j_1)
                correction_1 = tl.exp(m_1 - new_m_1)
                exp_j_1 = tl.exp(logit_j_1 - new_m_1)
                l_1 = l_1 * correction_1 + exp_j_1
                m_1 = new_m_1

            # Accumulate weighted source across D-tiles — shared source load
            for d_start in tl.range(0, D, BLOCK_D):
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

                # ── Query 0 accumulator ──────────────────────────
                acc_block_0 = tl.make_block_ptr(
                    base=running_acc_ptr + 0 * T * D,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                if j == 0:
                    new_acc_0 = exp_j_0[:, None] * src_tile
                else:
                    old_acc_0 = tl.load(acc_block_0, boundary_check=(0, 1))
                    new_acc_0 = (
                        old_acc_0 * correction_0[:, None] + exp_j_0[:, None] * src_tile
                    )
                tl.store(acc_block_0, new_acc_0, boundary_check=(0, 1))

                # ── Query 1 accumulator (if Q_BATCH > 1) ─────────
                if Q_BATCH > 1:
                    acc_block_1 = tl.make_block_ptr(
                        base=running_acc_ptr + 1 * T * D,
                        shape=(T, D),
                        strides=(D, 1),
                        offsets=(t_start, d_start),
                        block_shape=(BLOCK_T, BLOCK_D),
                        order=(1, 0),
                    )
                    if j == 0:
                        new_acc_1 = exp_j_1[:, None] * src_tile
                    else:
                        old_acc_1 = tl.load(acc_block_1, boundary_check=(0, 1))
                        new_acc_1 = (
                            old_acc_1 * correction_1[:, None]
                            + exp_j_1[:, None] * src_tile
                        )
                    tl.store(acc_block_1, new_acc_1, boundary_check=(0, 1))

        # ── Store per-query running state ─────────────────────────────
        tl.store(running_m_ptr + 0 * T + t_offs, m_0, mask=t_mask)
        tl.store(running_l_ptr + 0 * T + t_offs, l_0, mask=t_mask)
        if Q_BATCH > 1:
            tl.store(running_m_ptr + 1 * T + t_offs, m_1, mask=t_mask)
            tl.store(running_l_ptr + 1 * T + t_offs, l_1, mask=t_mask)


@triton.jit
def _block_attnres_merge_partial_kernel(
    partial_ptr,  # (T, D) — current partial block
    qw_ptr,  # (D,) — query * norm_weight for this sublayer
    # Running state from completed-wsum (or previous merge)
    running_m_ptr,  # (T,) input — running max logit from completed sources
    running_l_ptr,  # (T,) input — running sum-exp from completed sources
    running_acc_ptr,  # (T, D) input — weighted accumulator from completed sources
    # Precomputed from Phase 1
    precomputed_logits_ptr,  # (NC, T) — this query's completed logits
    precomputed_inv_rms_ptr,  # (NC, T) — shared completed inv_rms
    # Outputs
    result_ptr,  # (T, D) output — final weighted sum (bf16)
    alpha_ptr,  # (NC + 1, T) output — softmax weights for backward
    inv_rms_ptr,  # (NC + 1, T) output — inv_rms values for backward
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Merge partial source into completed running state (Phase 2).

    1. D-reduce partial → logit_p, inv_rms_p
    2. Online merge: incorporate logit_p into (m, l, acc)
    3. result = acc_final / l_final
    4. Reconstruct alpha for all NC+1 sources (for backward)
    5. Write alpha, inv_rms outputs

    The merge touches only the partial source data (cheap, single source),
    plus the running-state tensors and precomputed logits (scalars per token).
    """
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    PARTIAL_IDX = NC

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # ── D-reduce partial source ───────────────────────────────────
        sum_sq_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        dot_qw_p = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            qw_tile = tl.load(
                qw_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)
            partial_tile = _load_dense_tile(
                partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D
            )
            sum_sq_p += tl.sum(partial_tile * partial_tile, axis=1)
            dot_qw_p += tl.sum(partial_tile * qw_tile[None, :], axis=1)

        ir_p = tl.rsqrt(sum_sq_p / D + eps)
        logit_p = ir_p * dot_qw_p

        # ── Load completed running state ──────────────────────────────
        m_c = tl.load(running_m_ptr + t_offs, mask=t_mask, other=-1e30)
        l_c = tl.load(running_l_ptr + t_offs, mask=t_mask, other=0.0)

        # ── Online merge: incorporate partial ─────────────────────────
        new_m = tl.maximum(m_c, logit_p)
        correction = tl.exp(m_c - new_m)
        exp_p = tl.exp(logit_p - new_m)
        l_final = l_c * correction + exp_p
        m_final = new_m

        # ── Reconstruct alpha for backward ────────────────────────────
        # alpha[j] = exp(logit_j - m_final) / l_final for completed refs
        # alpha[NC] = exp_p / l_final for partial
        for j in tl.static_range(NC):
            logit_j = tl.load(
                precomputed_logits_ptr + j * T + t_offs, mask=t_mask, other=0.0
            )
            alpha_j = tl.exp(logit_j - m_final) / l_final
            tl.store(alpha_ptr + j * T + t_offs, alpha_j, mask=t_mask)

            # Copy precomputed inv_rms to output (for backward)
            ir_j = tl.load(
                precomputed_inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0
            )
            tl.store(inv_rms_ptr + j * T + t_offs, ir_j, mask=t_mask)

        # Partial alpha and inv_rms
        alpha_p = exp_p / l_final
        tl.store(alpha_ptr + PARTIAL_IDX * T + t_offs, alpha_p, mask=t_mask)
        tl.store(inv_rms_ptr + PARTIAL_IDX * T + t_offs, ir_p, mask=t_mask)

        # ── Weighted sum: acc * correction + exp_p * partial, / l_final ─
        for d_start in tl.range(0, D, BLOCK_D):
            acc_block = tl.make_block_ptr(
                base=running_acc_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            old_acc = tl.load(acc_block, boundary_check=(0, 1))
            partial_tile = _load_dense_tile(
                partial_ptr, t_start, d_start, T, D, BLOCK_T, BLOCK_D
            )
            final_acc = (
                old_acc * correction[:, None] + exp_p[:, None] * partial_tile
            ) / l_final[:, None]

            out_block = tl.make_block_ptr(
                base=result_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            tl.store(out_block, final_acc.to(tl.bfloat16), boundary_check=(0, 1))


# ═══════════════════════════════════════════════════════════════════════════
# Backward kernel
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _block_attnres_bwd_kernel(
    grad_result_ptr,  # (T, D)
    stacked_ptr,  # (N, T, D)
    qw_ptr,  # (D,) = query * norm_weight
    alpha_ptr,  # (N, T)
    inv_rms_ptr,  # (N, T)
    grad_stacked_ptr,  # (N, T, D) output
    R_local_ptr,  # (NUM_SMS, D) output — per-SM buffer for param grads
    # scratch buffers for softmax backward intermediates:
    grad_alpha_ptr,  # (N, T) scratch
    qw_dot_ptr,  # (N, T) scratch
    T,
    D: tl.constexpr,
    N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    TD = T * D

    # Zero this SM's R_local row once at start
    for d_start in tl.range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        tl.store(
            R_local_ptr + pid * D + d_offs,
            tl.zeros((BLOCK_D,), dtype=tl.float32),
            mask=d_mask,
        )

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # ── Pass 1: D-reduction per source → grad_alpha, qw_dot ──────
        # D-tile outer loop keeps go_tile/qw_tile loads outside the source
        # loop — this matters much more than the per-source scratch traffic.
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
                inv_rms_j = tl.load(
                    inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )
                grad_logit_j = tl.load(
                    grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )
                dot_term_j = tl.load(
                    qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )

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
                gs_norm = inv_rms_j[:, None] * (
                    grad_raw - raw_j * (dot_term_j / D)[:, None]
                )

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
                tl.store(
                    gs_block,
                    (gs_norm + gs_result).to(tl.bfloat16),
                    boundary_check=(0, 1),
                )

                # Accumulate R for param grads
                contribution = (grad_logit_j * inv_rms_j)[:, None] * src_tile
                R_chunk += tl.sum(contribution, axis=0)

            # Write to this SM's private R_local row — no atomics needed
            R_prev = tl.load(R_local_ptr + pid * D + d_offs, mask=d_mask, other=0.0)
            tl.store(R_local_ptr + pid * D + d_offs, R_prev + R_chunk, mask=d_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Batched backward kernel (Q_BATCH=1 or 2)
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _block_attnres_batched_state_bwd_kernel(
    # Per-sublayer inputs — sublayer 0
    grad_result0_ptr,  # (T, D)
    partial0_ptr,  # (T, D)
    qw0_ptr,  # (D,)
    alpha0_ptr,  # (NC + 1, T)
    inv_rms0_ptr,  # (NC + 1, T)
    # Per-sublayer inputs — sublayer 1 (ignored when Q_BATCH=1)
    grad_result1_ptr,
    partial1_ptr,
    qw1_ptr,
    alpha1_ptr,
    inv_rms1_ptr,
    # Shared completed refs (up to 8)
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    # Outputs: summed grad_completed (NC, T, D)
    grad_completed_ptr,  # (NC, T, D) — sum of per-sublayer contributions
    # Per-sublayer outputs
    grad_partial0_ptr,  # (T, D) output
    grad_partial1_ptr,  # (T, D) output (unused when Q_BATCH=1)
    R_local0_ptr,  # (NUM_SMS, D) output
    R_local1_ptr,  # (NUM_SMS, D) output (unused when Q_BATCH=1)
    # Per-sublayer scratch buffers
    grad_alpha0_ptr,  # (NC + 1, T)
    qw_dot0_ptr,  # (NC + 1, T)
    grad_alpha1_ptr,  # (NC + 1, T) (unused when Q_BATCH=1)
    qw_dot1_ptr,  # (NC + 1, T) (unused when Q_BATCH=1)
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    Q_BATCH: tl.constexpr,  # 1 or 2
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Batched backward for Q_BATCH sublayers sharing completed refs.

    Produces:
    - grad_completed: (NC, T, D) — SUM of per-sublayer grad contributions
    - grad_partial_{0,1}: per-sublayer (T, D)
    - R_local_{0,1}: per-sublayer (NUM_SMS, D) for param grad decomposition
    """
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    TD = T * D
    PARTIAL_IDX = NC

    # Zero this SM's R_local rows once at start
    for d_start in tl.range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        tl.store(
            R_local0_ptr + pid * D + d_offs,
            tl.zeros((BLOCK_D,), dtype=tl.float32),
            mask=d_mask,
        )
        if Q_BATCH > 1:
            tl.store(
                R_local1_ptr + pid * D + d_offs,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=d_mask,
            )

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # ── Zero scratch buffers for this tile ────────────────────────
        for j in tl.static_range(NC + 1):
            zeros = tl.zeros((BLOCK_T,), dtype=tl.float32)
            tl.store(grad_alpha0_ptr + j * T + t_offs, zeros, mask=t_mask)
            tl.store(qw_dot0_ptr + j * T + t_offs, zeros, mask=t_mask)
            if Q_BATCH > 1:
                tl.store(grad_alpha1_ptr + j * T + t_offs, zeros, mask=t_mask)
                tl.store(qw_dot1_ptr + j * T + t_offs, zeros, mask=t_mask)

        # ── Pass 1: D-reduction per source → grad_alpha, qw_dot ──────
        for d_start in tl.range(0, D, BLOCK_D):
            # Load go_tile and qw_tile for sublayer 0
            go_block_0 = tl.make_block_ptr(
                base=grad_result0_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            go_tile_0 = tl.load(go_block_0, boundary_check=(0, 1)).to(tl.float32)

            qw_tile_0 = tl.load(
                qw0_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)

            # Load go_tile and qw_tile for sublayer 1 (if Q_BATCH > 1)
            if Q_BATCH > 1:
                go_block_1 = tl.make_block_ptr(
                    base=grad_result1_ptr,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                go_tile_1 = tl.load(go_block_1, boundary_check=(0, 1)).to(tl.float32)

                qw_tile_1 = tl.load(
                    qw1_ptr + d_start + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) + d_start < D,
                    other=0.0,
                ).to(tl.float32)

            # Completed sources — shared read, per-sublayer accumulation
            for j in tl.static_range(NC):
                ga_j_0 = tl.load(
                    grad_alpha0_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )
                qwd_j_0 = tl.load(qw_dot0_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                if Q_BATCH > 1:
                    ga_j_1 = tl.load(
                        grad_alpha1_ptr + j * T + t_offs, mask=t_mask, other=0.0
                    )
                    qwd_j_1 = tl.load(
                        qw_dot1_ptr + j * T + t_offs, mask=t_mask, other=0.0
                    )

                # Shared source tile load
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

                # Sublayer 0 accumulation
                ga_j_0 += tl.sum(go_tile_0 * src_tile, axis=1)
                qwd_j_0 += tl.sum(qw_tile_0[None, :] * src_tile, axis=1)
                tl.store(grad_alpha0_ptr + j * T + t_offs, ga_j_0, mask=t_mask)
                tl.store(qw_dot0_ptr + j * T + t_offs, qwd_j_0, mask=t_mask)

                # Sublayer 1 accumulation
                if Q_BATCH > 1:
                    ga_j_1 += tl.sum(go_tile_1 * src_tile, axis=1)
                    qwd_j_1 += tl.sum(qw_tile_1[None, :] * src_tile, axis=1)
                    tl.store(grad_alpha1_ptr + j * T + t_offs, ga_j_1, mask=t_mask)
                    tl.store(qw_dot1_ptr + j * T + t_offs, qwd_j_1, mask=t_mask)

            # Partial source — per-sublayer (different partial for each sublayer)
            # Sublayer 0 partial
            ga_p_0 = tl.load(
                grad_alpha0_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            qwd_p_0 = tl.load(
                qw_dot0_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            partial_block_0 = tl.make_block_ptr(
                base=partial0_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            partial_tile_0 = tl.load(partial_block_0, boundary_check=(0, 1)).to(
                tl.float32
            )

            ga_p_0 += tl.sum(go_tile_0 * partial_tile_0, axis=1)
            qwd_p_0 += tl.sum(qw_tile_0[None, :] * partial_tile_0, axis=1)
            tl.store(grad_alpha0_ptr + PARTIAL_IDX * T + t_offs, ga_p_0, mask=t_mask)
            tl.store(qw_dot0_ptr + PARTIAL_IDX * T + t_offs, qwd_p_0, mask=t_mask)

            # Sublayer 1 partial
            if Q_BATCH > 1:
                ga_p_1 = tl.load(
                    grad_alpha1_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
                )
                qwd_p_1 = tl.load(
                    qw_dot1_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
                )
                partial_block_1 = tl.make_block_ptr(
                    base=partial1_ptr,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                partial_tile_1 = tl.load(partial_block_1, boundary_check=(0, 1)).to(
                    tl.float32
                )

                ga_p_1 += tl.sum(go_tile_1 * partial_tile_1, axis=1)
                qwd_p_1 += tl.sum(qw_tile_1[None, :] * partial_tile_1, axis=1)
                tl.store(
                    grad_alpha1_ptr + PARTIAL_IDX * T + t_offs, ga_p_1, mask=t_mask
                )
                tl.store(qw_dot1_ptr + PARTIAL_IDX * T + t_offs, qwd_p_1, mask=t_mask)

        # ── Softmax backward — per sublayer ───────────────────────────
        # Sublayer 0
        v_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for j in tl.static_range(NC + 1):
            a_j = tl.load(alpha0_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ga_j = tl.load(grad_alpha0_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            v_0 += a_j * ga_j

        for j in tl.static_range(NC + 1):
            a_j = tl.load(alpha0_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ga_j = tl.load(grad_alpha0_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            ir_j = tl.load(inv_rms0_ptr + j * T + t_offs, mask=t_mask, other=0.0)
            qwd_j = tl.load(qw_dot0_ptr + j * T + t_offs, mask=t_mask, other=0.0)

            gl_j = a_j * (ga_j - v_0)
            dt_j = gl_j * ir_j * qwd_j

            tl.store(grad_alpha0_ptr + j * T + t_offs, gl_j, mask=t_mask)
            tl.store(qw_dot0_ptr + j * T + t_offs, dt_j, mask=t_mask)

        # Sublayer 1
        if Q_BATCH > 1:
            v_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
            for j in tl.static_range(NC + 1):
                a_j = tl.load(alpha1_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                ga_j = tl.load(grad_alpha1_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                v_1 += a_j * ga_j

            for j in tl.static_range(NC + 1):
                a_j = tl.load(alpha1_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                ga_j = tl.load(grad_alpha1_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                ir_j = tl.load(inv_rms1_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                qwd_j = tl.load(qw_dot1_ptr + j * T + t_offs, mask=t_mask, other=0.0)

                gl_j = a_j * (ga_j - v_1)
                dt_j = gl_j * ir_j * qwd_j

                tl.store(grad_alpha1_ptr + j * T + t_offs, gl_j, mask=t_mask)
                tl.store(qw_dot1_ptr + j * T + t_offs, dt_j, mask=t_mask)

        # ── Pass 2: write grad_completed (summed) + grad_partial + R ──
        for d_start in tl.range(0, D, BLOCK_D):
            # Load go_tile for sublayer 0
            go_block_0 = tl.make_block_ptr(
                base=grad_result0_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            go_tile_0 = tl.load(go_block_0, boundary_check=(0, 1)).to(tl.float32)

            qw_tile_0 = tl.load(
                qw0_ptr + d_start + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) + d_start < D,
                other=0.0,
            ).to(tl.float32)

            if Q_BATCH > 1:
                go_block_1 = tl.make_block_ptr(
                    base=grad_result1_ptr,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                go_tile_1 = tl.load(go_block_1, boundary_check=(0, 1)).to(tl.float32)

                qw_tile_1 = tl.load(
                    qw1_ptr + d_start + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) + d_start < D,
                    other=0.0,
                ).to(tl.float32)

            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            R_chunk_0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
            if Q_BATCH > 1:
                R_chunk_1 = tl.zeros((BLOCK_D,), dtype=tl.float32)

            # Completed sources — shared read, summed grad_completed
            for j in tl.static_range(NC):
                # Sublayer 0 scalars
                alpha_j_0 = tl.load(alpha0_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                inv_rms_j_0 = tl.load(
                    inv_rms0_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )
                grad_logit_j_0 = tl.load(
                    grad_alpha0_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )
                dot_term_j_0 = tl.load(
                    qw_dot0_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )

                if Q_BATCH > 1:
                    alpha_j_1 = tl.load(
                        alpha1_ptr + j * T + t_offs, mask=t_mask, other=0.0
                    )
                    inv_rms_j_1 = tl.load(
                        inv_rms1_ptr + j * T + t_offs, mask=t_mask, other=0.0
                    )
                    grad_logit_j_1 = tl.load(
                        grad_alpha1_ptr + j * T + t_offs, mask=t_mask, other=0.0
                    )
                    dot_term_j_1 = tl.load(
                        qw_dot1_ptr + j * T + t_offs, mask=t_mask, other=0.0
                    )

                # Shared source tile load
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

                # Sublayer 0: grad for completed source j
                raw_j_0 = src_tile * inv_rms_j_0[:, None]
                grad_raw_0 = grad_logit_j_0[:, None] * qw_tile_0[None, :]
                gs_norm_0 = inv_rms_j_0[:, None] * (
                    grad_raw_0 - raw_j_0 * (dot_term_j_0 / D)[:, None]
                )
                gs_result_0 = alpha_j_0[:, None] * go_tile_0
                gs_total = gs_norm_0 + gs_result_0

                # Sublayer 1: grad for completed source j
                if Q_BATCH > 1:
                    raw_j_1 = src_tile * inv_rms_j_1[:, None]
                    grad_raw_1 = grad_logit_j_1[:, None] * qw_tile_1[None, :]
                    gs_norm_1 = inv_rms_j_1[:, None] * (
                        grad_raw_1 - raw_j_1 * (dot_term_j_1 / D)[:, None]
                    )
                    gs_result_1 = alpha_j_1[:, None] * go_tile_1
                    gs_total = gs_total + gs_norm_1 + gs_result_1

                # Write summed grad_completed
                gs_block = tl.make_block_ptr(
                    base=grad_completed_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                tl.store(
                    gs_block,
                    gs_total.to(tl.bfloat16),
                    boundary_check=(0, 1),
                )

                # R accumulation — per sublayer
                contribution_0 = (grad_logit_j_0 * inv_rms_j_0)[:, None] * src_tile
                R_chunk_0 += tl.sum(contribution_0, axis=0)

                if Q_BATCH > 1:
                    contribution_1 = (grad_logit_j_1 * inv_rms_j_1)[:, None] * src_tile
                    R_chunk_1 += tl.sum(contribution_1, axis=0)

            # Partial source — sublayer 0
            alpha_p_0 = tl.load(
                alpha0_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            inv_rms_p_0 = tl.load(
                inv_rms0_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            grad_logit_p_0 = tl.load(
                grad_alpha0_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            dot_term_p_0 = tl.load(
                qw_dot0_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )

            partial_block_0 = tl.make_block_ptr(
                base=partial0_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            partial_tile_0 = tl.load(partial_block_0, boundary_check=(0, 1)).to(
                tl.float32
            )

            raw_p_0 = partial_tile_0 * inv_rms_p_0[:, None]
            grad_raw_p_0 = grad_logit_p_0[:, None] * qw_tile_0[None, :]
            gs_norm_p_0 = inv_rms_p_0[:, None] * (
                grad_raw_p_0 - raw_p_0 * (dot_term_p_0 / D)[:, None]
            )
            gs_result_p_0 = alpha_p_0[:, None] * go_tile_0

            grad_partial_block_0 = tl.make_block_ptr(
                base=grad_partial0_ptr,
                shape=(T, D),
                strides=(D, 1),
                offsets=(t_start, d_start),
                block_shape=(BLOCK_T, BLOCK_D),
                order=(1, 0),
            )
            tl.store(
                grad_partial_block_0,
                (gs_norm_p_0 + gs_result_p_0).to(tl.bfloat16),
                boundary_check=(0, 1),
            )

            contribution_p_0 = (grad_logit_p_0 * inv_rms_p_0)[:, None] * partial_tile_0
            R_chunk_0 += tl.sum(contribution_p_0, axis=0)

            # Write R_local for sublayer 0
            R_prev_0 = tl.load(R_local0_ptr + pid * D + d_offs, mask=d_mask, other=0.0)
            tl.store(R_local0_ptr + pid * D + d_offs, R_prev_0 + R_chunk_0, mask=d_mask)

            # Partial source — sublayer 1
            if Q_BATCH > 1:
                alpha_p_1 = tl.load(
                    alpha1_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
                )
                inv_rms_p_1 = tl.load(
                    inv_rms1_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
                )
                grad_logit_p_1 = tl.load(
                    grad_alpha1_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
                )
                dot_term_p_1 = tl.load(
                    qw_dot1_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
                )

                partial_block_1 = tl.make_block_ptr(
                    base=partial1_ptr,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                partial_tile_1 = tl.load(partial_block_1, boundary_check=(0, 1)).to(
                    tl.float32
                )

                raw_p_1 = partial_tile_1 * inv_rms_p_1[:, None]
                grad_raw_p_1 = grad_logit_p_1[:, None] * qw_tile_1[None, :]
                gs_norm_p_1 = inv_rms_p_1[:, None] * (
                    grad_raw_p_1 - raw_p_1 * (dot_term_p_1 / D)[:, None]
                )
                gs_result_p_1 = alpha_p_1[:, None] * go_tile_1

                grad_partial_block_1 = tl.make_block_ptr(
                    base=grad_partial1_ptr,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                tl.store(
                    grad_partial_block_1,
                    (gs_norm_p_1 + gs_result_p_1).to(tl.bfloat16),
                    boundary_check=(0, 1),
                )

                contribution_p_1 = (grad_logit_p_1 * inv_rms_p_1)[
                    :, None
                ] * partial_tile_1
                R_chunk_1 += tl.sum(contribution_p_1, axis=0)

                # Write R_local for sublayer 1
                R_prev_1 = tl.load(
                    R_local1_ptr + pid * D + d_offs, mask=d_mask, other=0.0
                )
                tl.store(
                    R_local1_ptr + pid * D + d_offs, R_prev_1 + R_chunk_1, mask=d_mask
                )


@triton.jit
def _block_attnres_state_bwd_kernel(
    grad_result_ptr,  # (T, D)
    completed0_ptr,
    completed1_ptr,
    completed2_ptr,
    completed3_ptr,
    completed4_ptr,
    completed5_ptr,
    completed6_ptr,
    completed7_ptr,
    partial_ptr,  # (T, D)
    qw_ptr,  # (D,) = query * norm_weight
    alpha_ptr,  # (NC + 1, T)
    inv_rms_ptr,  # (NC + 1, T)
    grad_completed_ptr,  # (NC, T, D) output
    grad_partial_ptr,  # (T, D) output
    R_local_ptr,  # (NUM_SMS, D) output — per-SM buffer for param grads
    grad_alpha_ptr,  # (NC + 1, T) scratch
    qw_dot_ptr,  # (NC + 1, T) scratch
    T,
    D: tl.constexpr,
    NC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    TD = T * D
    PARTIAL_IDX = NC

    # Zero this SM's R_local row once at start
    for d_start in tl.range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        tl.store(
            R_local_ptr + pid * D + d_offs,
            tl.zeros((BLOCK_D,), dtype=tl.float32),
            mask=d_mask,
        )

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_start = tile_id * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # ── Pass 1: D-reduction per source → grad_alpha, qw_dot ──────
        # D-tile outer loop keeps go_tile/qw_tile loads outside the source
        # loop — this matters much more than the per-source scratch traffic.
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

            ga_p = tl.load(
                grad_alpha_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            qwd_p = tl.load(
                qw_dot_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
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

        # ── Softmax backward ─────────────────────────────────────────
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

            for j in tl.static_range(NC):
                alpha_j = tl.load(alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0)
                inv_rms_j = tl.load(
                    inv_rms_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )
                grad_logit_j = tl.load(
                    grad_alpha_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )
                dot_term_j = tl.load(
                    qw_dot_ptr + j * T + t_offs, mask=t_mask, other=0.0
                )

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
                gs_norm = inv_rms_j[:, None] * (
                    grad_raw - raw_j * (dot_term_j / D)[:, None]
                )
                gs_result = alpha_j[:, None] * go_tile

                gs_block = tl.make_block_ptr(
                    base=grad_completed_ptr + j * TD,
                    shape=(T, D),
                    strides=(D, 1),
                    offsets=(t_start, d_start),
                    block_shape=(BLOCK_T, BLOCK_D),
                    order=(1, 0),
                )
                tl.store(
                    gs_block,
                    (gs_norm + gs_result).to(tl.bfloat16),
                    boundary_check=(0, 1),
                )

                contribution = (grad_logit_j * inv_rms_j)[:, None] * src_tile
                R_chunk += tl.sum(contribution, axis=0)

            alpha_p = tl.load(
                alpha_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            inv_rms_p = tl.load(
                inv_rms_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            grad_logit_p = tl.load(
                grad_alpha_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )
            dot_term_p = tl.load(
                qw_dot_ptr + PARTIAL_IDX * T + t_offs, mask=t_mask, other=0.0
            )

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
            gs_norm_p = inv_rms_p[:, None] * (
                grad_raw_p - raw_p * (dot_term_p / D)[:, None]
            )
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

            # Write to this SM's private R_local row — no atomics needed
            R_prev = tl.load(R_local_ptr + pid * D + d_offs, mask=d_mask, other=0.0)
            tl.store(R_local_ptr + pid * D + d_offs, R_prev + R_chunk, mask=d_mask)
