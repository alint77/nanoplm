"""Fused Triton kernels for mHC-lite residual connections.

Implements fused kernels (K3, K4) that replace separate PyTorch ops.
K2 (coefficient computation) stays in PyTorch since it operates on only
32 values per token — the overhead is in launch, not compute.

Design follows the same persistent-scheduling pattern as triton_kernels.py.

v2: Eliminated atomic_add in backward kernels by splitting into:
  - Kernel A: grad_x/grad_lo (tiles over T×C, no reductions needed)
  - Kernel B: grad_H/grad_hp (tiles over T only, reduces over ALL of C internally)
"""

import math
import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════════
# Hardware detection (shared with triton_kernels.py)
# ═══════════════════════════════════════════════════════════════════════════
_NUM_SMS = None


def _get_num_sms():
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    return _NUM_SMS


# ═══════════════════════════════════════════════════════════════════════════
# Kernel 4: fused_post_res — THE CRITICAL KERNEL
#
# output[t, i, c] = Σ_j H_merged[t,i,j] * x[t,j,c]  +  h_post[t,i] * lo[t,c]
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_post_res_fwd_kernel(
    x_ptr, lo_ptr, H_ptr, hp_ptr, out_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        t_mask = t_offs < T
        tc_mask = t_mask[:, None] & (c_offs[None, :] < C)

        # Load layer_output[t, c]: (BLOCK_T, BLOCK_C)
        lo_idx = t_offs[:, None] * C + c_offs[None, :]
        lo = tl.load(lo_ptr + lo_idx, mask=tc_mask, other=0.0).to(tl.float32)

        for i in tl.static_range(n):
            hp_i = tl.load(hp_ptr + t_offs * n + i, mask=t_mask, other=0.0)
            acc = hp_i[:, None] * lo

            for j in tl.static_range(n):
                h_ij = tl.load(H_ptr + t_offs * (n * n) + i * n + j, mask=t_mask, other=0.0)
                x_idx = t_offs[:, None] * (n * C) + j * C + c_offs[None, :]
                x_j = tl.load(x_ptr + x_idx, mask=tc_mask, other=0.0).to(tl.float32)
                acc += h_ij[:, None] * x_j

            out_idx = t_offs[:, None] * (n * C) + i * C + c_offs[None, :]
            tl.store(out_ptr + out_idx, acc.to(tl.bfloat16), mask=tc_mask)


# ---------- K4 Backward: Kernel A (grad_x, grad_lo) ----------
# Tiles over (T, C). No reductions, no atomics.

@triton.jit
def _fused_post_res_bwd_xlo_kernel(
    H_ptr, hp_ptr, grad_out_ptr, lo_ptr,
    grad_x_ptr, grad_lo_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Compute ∂L/∂x_streams and ∂L/∂layer_output (no reductions needed)."""
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        t_mask = t_offs < T
        tc_mask = t_mask[:, None] & (c_offs[None, :] < C)

        # ∂L/∂layer_output[t,c] = Σ_i h_post[t,i] * grad_out[t,i,c]
        grad_lo_acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

        for i in tl.static_range(n):
            hp_i = tl.load(hp_ptr + t_offs * n + i, mask=t_mask, other=0.0)
            go_idx = t_offs[:, None] * (n * C) + i * C + c_offs[None, :]
            grad_i = tl.load(grad_out_ptr + go_idx, mask=tc_mask, other=0.0).to(tl.float32)
            grad_lo_acc += hp_i[:, None] * grad_i

        lo_idx = t_offs[:, None] * C + c_offs[None, :]
        tl.store(grad_lo_ptr + lo_idx, grad_lo_acc.to(tl.bfloat16), mask=tc_mask)

        # ∂L/∂x[t,j,c] = Σ_i H_merged[t,i,j] * grad_out[t,i,c]
        for j in tl.static_range(n):
            grad_x_j = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
            for i in tl.static_range(n):
                h_ij = tl.load(H_ptr + t_offs * (n * n) + i * n + j, mask=t_mask, other=0.0)
                go_idx = t_offs[:, None] * (n * C) + i * C + c_offs[None, :]
                grad_i = tl.load(grad_out_ptr + go_idx, mask=tc_mask, other=0.0).to(tl.float32)
                grad_x_j += h_ij[:, None] * grad_i
            x_out_idx = t_offs[:, None] * (n * C) + j * C + c_offs[None, :]
            tl.store(grad_x_ptr + x_out_idx, grad_x_j.to(tl.bfloat16), mask=tc_mask)


# ---------- K4 Backward: Kernel B (grad_H, grad_hp) ----------
# Tiles over T only. Each program reduces over ALL of C — NO atomics.

@triton.jit
def _fused_post_res_bwd_Hhp_kernel(
    x_ptr, lo_ptr, grad_out_ptr,
    grad_H_ptr, grad_hp_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Compute ∂L/∂H_merged and ∂L/∂h_post by reducing over ALL of C internally."""
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_offs = tile_id * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # Accumulators for the C-reduction (all scalars per token)
        # grad_H[t, i, j] = Σ_c grad_out[t,i,c] * x[t,j,c]  → n*n accums
        # grad_hp[t, i] = Σ_c grad_out[t,i,c] * lo[t,c]      → n accums
        # Total: n*n + n = 20 accumulators per token, each is (BLOCK_T,)

        # We accumulate inside BLOCK_C-chunked loop over C
        # Initialize all accumulators to zero
        # For n=4: 20 separate (BLOCK_T,) accumulators

        # grad_hp accumulators: n of them
        ghp_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp_2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp_3 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # grad_H accumulators: n*n = 16 of them
        gH_00 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_01 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_02 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_03 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_10 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_11 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_12 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_13 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_20 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_21 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_22 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_23 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_30 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_31 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_32 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_33 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # Iterate over C in chunks
        for c_start in tl.range(0, C, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C
            tc_mask = t_mask[:, None] & c_mask[None, :]

            # Load lo[t, c]
            lo_idx = t_offs[:, None] * C + c_offs[None, :]
            lo = tl.load(lo_ptr + lo_idx, mask=tc_mask, other=0.0).to(tl.float32)

            # Load x[t, j, c] for all j
            x0 = tl.load(x_ptr + t_offs[:, None] * (n * C) + 0 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + t_offs[:, None] * (n * C) + 1 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + t_offs[:, None] * (n * C) + 2 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + t_offs[:, None] * (n * C) + 3 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)

            # Load grad_out[t, i, c] for all i
            go0 = tl.load(grad_out_ptr + t_offs[:, None] * (n * C) + 0 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)
            go1 = tl.load(grad_out_ptr + t_offs[:, None] * (n * C) + 1 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)
            go2 = tl.load(grad_out_ptr + t_offs[:, None] * (n * C) + 2 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)
            go3 = tl.load(grad_out_ptr + t_offs[:, None] * (n * C) + 3 * C + c_offs[None, :], mask=tc_mask, other=0.0).to(tl.float32)

            # grad_hp[t,i] += Σ_c grad_out[t,i,c] * lo[t,c]
            ghp_0 += tl.sum(go0 * lo, axis=1)
            ghp_1 += tl.sum(go1 * lo, axis=1)
            ghp_2 += tl.sum(go2 * lo, axis=1)
            ghp_3 += tl.sum(go3 * lo, axis=1)

            # grad_H[t,i,j] += Σ_c grad_out[t,i,c] * x[t,j,c]
            gH_00 += tl.sum(go0 * x0, axis=1)
            gH_01 += tl.sum(go0 * x1, axis=1)
            gH_02 += tl.sum(go0 * x2, axis=1)
            gH_03 += tl.sum(go0 * x3, axis=1)
            gH_10 += tl.sum(go1 * x0, axis=1)
            gH_11 += tl.sum(go1 * x1, axis=1)
            gH_12 += tl.sum(go1 * x2, axis=1)
            gH_13 += tl.sum(go1 * x3, axis=1)
            gH_20 += tl.sum(go2 * x0, axis=1)
            gH_21 += tl.sum(go2 * x1, axis=1)
            gH_22 += tl.sum(go2 * x2, axis=1)
            gH_23 += tl.sum(go2 * x3, axis=1)
            gH_30 += tl.sum(go3 * x0, axis=1)
            gH_31 += tl.sum(go3 * x1, axis=1)
            gH_32 += tl.sum(go3 * x2, axis=1)
            gH_33 += tl.sum(go3 * x3, axis=1)

        # Store grad_hp[t, i]
        tl.store(grad_hp_ptr + t_offs * n + 0, ghp_0, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 1, ghp_1, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 2, ghp_2, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 3, ghp_3, mask=t_mask)

        # Store grad_H[t, i, j] — 16 stores
        H_base = t_offs * (n * n)
        tl.store(grad_H_ptr + H_base + 0,  gH_00, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 1,  gH_01, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 2,  gH_02, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 3,  gH_03, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 4,  gH_10, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 5,  gH_11, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 6,  gH_12, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 7,  gH_13, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 8,  gH_20, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 9,  gH_21, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 10, gH_22, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 11, gH_23, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 12, gH_30, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 13, gH_31, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 14, gH_32, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 15, gH_33, mask=t_mask)


class FusedPostRes(torch.autograd.Function):
    """output = H_merged @ x_streams + h_post[:,:,None] * layer_output[:,None,:]"""

    @staticmethod
    def forward(ctx, x_streams, layer_output, H_merged, h_post):
        T, n, C = x_streams.shape
        out = torch.empty_like(x_streams)
        NUM_SMS = _get_num_sms()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)
        _fused_post_res_fwd_kernel[grid](
            x_streams, layer_output, H_merged, h_post, out,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
        )
        ctx.save_for_backward(x_streams, layer_output, H_merged, h_post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_streams, layer_output, H_merged, h_post = ctx.saved_tensors
        T, n, C = x_streams.shape
        grad_out = grad_output.contiguous()
        NUM_SMS = _get_num_sms()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)

        # Kernel A: grad_x and grad_lo (tiles over T×C, no atomics)
        grad_x = torch.empty_like(x_streams)
        grad_lo = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
        _fused_post_res_bwd_xlo_kernel[grid](
            H_merged, h_post, grad_out, layer_output,
            grad_x, grad_lo,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
        )

        # Kernel B: grad_H and grad_hp (tiles over T, reduces over C, no atomics)
        grad_H = torch.empty((T, n, n), device=x_streams.device, dtype=torch.float32)
        grad_hp = torch.empty((T, n), device=x_streams.device, dtype=torch.float32)
        _fused_post_res_bwd_Hhp_kernel[grid](
            x_streams, layer_output, grad_out,
            grad_H, grad_hp,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
        )

        return grad_x, grad_lo, grad_H, grad_hp


# ═══════════════════════════════════════════════════════════════════════════
# Kernel 3: fused_pre_map
# layer_input[t, c] = Σ_j h_pre[t, j] * x_streams[t, j, c]
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_pre_map_fwd_kernel(
    x_ptr, h_pre_ptr, out_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles
        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        t_mask = t_offs < T
        tc_mask = t_mask[:, None] & (c_offs[None, :] < C)

        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(n):
            hp_j = tl.load(h_pre_ptr + t_offs * n + j, mask=t_mask, other=0.0)
            x_idx = t_offs[:, None] * (n * C) + j * C + c_offs[None, :]
            x_j = tl.load(x_ptr + x_idx, mask=tc_mask, other=0.0).to(tl.float32)
            acc += hp_j[:, None] * x_j

        out_idx = t_offs[:, None] * C + c_offs[None, :]
        tl.store(out_ptr + out_idx, acc.to(tl.bfloat16), mask=tc_mask)


@triton.jit
def _fused_pre_map_bwd_kernel(
    x_ptr, h_pre_ptr, grad_out_ptr,
    grad_x_ptr, grad_hp_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles
        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        t_mask = t_offs < T
        tc_mask = t_mask[:, None] & (c_offs[None, :] < C)

        go_idx = t_offs[:, None] * C + c_offs[None, :]
        grad_out = tl.load(grad_out_ptr + go_idx, mask=tc_mask, other=0.0).to(tl.float32)

        for j in tl.static_range(n):
            hp_j = tl.load(h_pre_ptr + t_offs * n + j, mask=t_mask, other=0.0)
            # ∂L/∂x[t,j,c] = h_pre[j] * grad_out[c]
            grad_x_j = hp_j[:, None] * grad_out
            x_out_idx = t_offs[:, None] * (n * C) + j * C + c_offs[None, :]
            tl.store(grad_x_ptr + x_out_idx, grad_x_j.to(tl.bfloat16), mask=tc_mask)
            # ∂L/∂h_pre[t,j] += Σ_c x[t,j,c] * grad_out[c]
            x_j = tl.load(x_ptr + x_out_idx, mask=tc_mask, other=0.0).to(tl.float32)
            dot = tl.sum(x_j * grad_out, axis=1)  # (BLOCK_T,)
            tl.atomic_add(grad_hp_ptr + t_offs * n + j, dot, mask=t_mask)


class FusedPreMap(torch.autograd.Function):
    """layer_input = h_pre @ x_streams (weighted stream aggregation)."""

    @staticmethod
    def forward(ctx, x_streams, h_pre):
        T, n, C = x_streams.shape
        out = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
        NUM_SMS = _get_num_sms()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)
        _fused_pre_map_fwd_kernel[grid](
            x_streams, h_pre, out,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
        )
        ctx.save_for_backward(x_streams, h_pre)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_streams, h_pre = ctx.saved_tensors
        T, n, C = x_streams.shape
        grad_out = grad_output.contiguous()
        
        grad_x = torch.empty_like(x_streams)
        grad_hp = torch.zeros((T, n), device=x_streams.device, dtype=torch.float32)
        NUM_SMS = _get_num_sms()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)

        _fused_pre_map_bwd_kernel[grid](
            x_streams, h_pre, grad_out,
            grad_x, grad_hp,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
        )
        return grad_x, grad_hp


# ═══════════════════════════════════════════════════════════════════════════
# K2: fused_coefficients (PyTorch — tiny 32-dim ops, not worth a kernel)
# ═══════════════════════════════════════════════════════════════════════════

def fused_coefficients_pytorch(proj_out, alpha_pre, alpha_post, alpha_res, bias, perm_mat, n):
    """Compute h_pre, h_post, H_merged from projection output."""
    import torch.nn.functional as F

    pre_proj = proj_out[:, :n]
    post_proj = proj_out[:, n:2*n]
    res_proj = proj_out[:, 2*n:]

    pre_bias = bias[:n]
    post_bias = bias[n:2*n]
    res_bias = bias[2*n:]

    h_pre = torch.sigmoid(alpha_pre * pre_proj + pre_bias)
    h_post = 2.0 * torch.sigmoid(alpha_post * post_proj + post_bias)
    a_res = F.softmax(alpha_res * res_proj + res_bias, dim=-1)

    H_res = torch.matmul(a_res, perm_mat).unflatten(-1, (n, n))
    H_merged = H_res - h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)

    return h_pre, h_post, H_merged
