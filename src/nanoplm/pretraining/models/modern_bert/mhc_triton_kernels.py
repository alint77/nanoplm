"""Fused Triton kernels for mHC-lite residual connections.

Implements 3 fused kernels (K1, K3, K4) that replace separate PyTorch ops.
K2 (coefficient computation) stays in PyTorch since it operates on only
32 values per token — the overhead is in launch, not compute.

Design follows the same persistent-scheduling pattern as triton_kernels.py.
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
#
# Each tile processes BLOCK_T tokens × full C channels.
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

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # (BLOCK_T,)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # (BLOCK_C,)
        t_mask = t_offs < T
        c_mask = c_offs < C

        # Load layer_output[t, c]: (BLOCK_T, BLOCK_C)
        lo_idx = t_offs[:, None] * C + c_offs[None, :]
        lo_mask = t_mask[:, None] & c_mask[None, :]
        lo = tl.load(lo_ptr + lo_idx, mask=lo_mask, other=0.0).to(tl.float32)

        for i in tl.static_range(n):
            # Load h_post[t, i]: (BLOCK_T,) scalar per token
            hp_i = tl.load(hp_ptr + t_offs * n + i, mask=t_mask, other=0.0)

            # acc = h_post[t,i] * lo[t,c]: (BLOCK_T, BLOCK_C)
            acc = hp_i[:, None] * lo

            for j in tl.static_range(n):
                # Load H_merged[t, i, j]: (BLOCK_T,) scalar per token
                h_ij = tl.load(H_ptr + t_offs * (n * n) + i * n + j, mask=t_mask, other=0.0)
                # Load x[t, j, c]: (BLOCK_T, BLOCK_C)
                x_idx = t_offs[:, None] * (n * C) + j * C + c_offs[None, :]
                x_j = tl.load(x_ptr + x_idx, mask=lo_mask, other=0.0).to(tl.float32)
                acc += h_ij[:, None] * x_j

            # Store output[t, i, c]
            out_idx = t_offs[:, None] * (n * C) + i * C + c_offs[None, :]
            tl.store(out_ptr + out_idx, acc.to(tl.bfloat16), mask=lo_mask)


@triton.jit
def _fused_post_res_bwd_kernel(
    x_ptr, lo_ptr, H_ptr, hp_ptr, grad_out_ptr,
    grad_x_ptr, grad_lo_ptr, grad_H_ptr, grad_hp_ptr,
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
        c_mask = c_offs < C
        tc_mask = t_mask[:, None] & c_mask[None, :]

        # Load layer_output
        lo_idx = t_offs[:, None] * C + c_offs[None, :]
        lo = tl.load(lo_ptr + lo_idx, mask=tc_mask, other=0.0).to(tl.float32)

        # ∂L/∂layer_output[t,c] = Σ_i h_post[t,i] * grad_out[t,i,c]
        grad_lo_acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

        for i in tl.static_range(n):
            hp_i = tl.load(hp_ptr + t_offs * n + i, mask=t_mask, other=0.0)
            go_idx = t_offs[:, None] * (n * C) + i * C + c_offs[None, :]
            grad_i = tl.load(grad_out_ptr + go_idx, mask=tc_mask, other=0.0).to(tl.float32)

            grad_lo_acc += hp_i[:, None] * grad_i

            # ∂L/∂h_post[t,i] += Σ_c grad_out[t,i,c] * lo[t,c]
            dot_hp = tl.sum(grad_i * lo, axis=1)  # (BLOCK_T,)
            tl.atomic_add(grad_hp_ptr + t_offs * n + i, dot_hp, mask=t_mask)

            # ∂L/∂H_merged[t,i,j] += Σ_c grad_out[t,i,c] * x[t,j,c]
            for j in tl.static_range(n):
                x_idx = t_offs[:, None] * (n * C) + j * C + c_offs[None, :]
                x_j = tl.load(x_ptr + x_idx, mask=tc_mask, other=0.0).to(tl.float32)
                dot_H = tl.sum(grad_i * x_j, axis=1)
                tl.atomic_add(grad_H_ptr + t_offs * (n * n) + i * n + j, dot_H, mask=t_mask)

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
        grad_x = torch.empty_like(x_streams)
        grad_lo = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
        grad_H = torch.zeros((T, n, n), device=x_streams.device, dtype=torch.float32)
        grad_hp = torch.zeros((T, n), device=x_streams.device, dtype=torch.float32)
        NUM_SMS = _get_num_sms()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)
        _fused_post_res_bwd_kernel[grid](
            x_streams, layer_output, H_merged, h_post,
            grad_output.contiguous(),
            grad_x, grad_lo, grad_H, grad_hp,
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
        grad_x = torch.empty_like(x_streams)
        grad_hp = torch.zeros((T, n), device=x_streams.device, dtype=torch.float32)
        NUM_SMS = _get_num_sms()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)
        _fused_pre_map_bwd_kernel[grid](
            x_streams, h_pre, grad_output.contiguous(),
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
