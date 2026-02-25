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
_HW_CONFIG = None

def _get_hw_config():
    global _HW_CONFIG
    if _HW_CONFIG is None:
        props = torch.cuda.get_device_properties("cuda")
        num_sms = props.multi_processor_count
        cc = (props.major, props.minor)
        
        # SM90 (Hopper) has 227KB shared memory per block
        if cc == (9, 0):
            # (num_sms, num_warps, num_stages)
            _HW_CONFIG = (num_sms, 8, 4)
        elif cc[0] >= 12 or cc == (8, 9):
            # SM120 (Blackwell consumer/workstation: RTX 5090, RTX 6000 Blackwell) 
            # and SM89 (Ada Lovelace) have strict ~99-100KB shared memory limits per block.
            _HW_CONFIG = (num_sms, 4, 2)
        else:
            # Safe default for older/untested architectures
            _HW_CONFIG = (num_sms, 4, 2)
            
    return _HW_CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# Kernel 1: fused_rmsnorm_project
#
# proj_out[t, d] = (x_flat[t] / rms[t]) @ W[d]
# Dimensions: x_flat is (T, nC), W is (D_out, nC), proj_out is (T, D_out)
# D_out is tiny (e.g. 32), nC is large (e.g. 8192). 
# This is a vectorized dot-product kernel, not a standard GEMM.
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_rmsnorm_project_fwd_kernel(
    x_ptr, W_ptr,
    out_ptr, inv_rms_ptr,
    T, nC: tl.constexpr, D_out: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    t_start = pid * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    
    d_offs = tl.arange(0, D_out)
    
    sum_sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, D_out), dtype=tl.float32)
    
    for k_start in tl.range(0, nC, BLOCK_K):
        # Load x_tile: (BLOCK_T, BLOCK_K) via TMA
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(T, nC),
            strides=(nC, 1),
            offsets=(t_start, k_start),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0)
        )
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1))
        x_tile_f32 = x_tile.to(tl.float32)
        sum_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)
        
        # NOTE: TMA descriptors cannot be transposed. Load W in its native layout
        # (D_out, nC) and transpose in registers for the dot.
        w_block_ptr = tl.make_block_ptr(
            base=W_ptr,
            shape=(D_out, nC),
            strides=(nC, 1),
            offsets=(0, k_start),
            block_shape=(D_out, BLOCK_K),
            order=(1, 0),
        )
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1))

        # Tensor Core matmul: (BLOCK_T, BLOCK_K) @ (BLOCK_K, D_out)
        acc += tl.dot(x_tile, w_tile.T)

    inv_rms = tl.rsqrt(sum_sq / nC + 1e-6)
    tl.store(inv_rms_ptr + t_offs, inv_rms, mask=t_mask)

    # proj_out = (x @ W_T) / rms = (x @ W_T) * inv_rms
    out_vals = acc * inv_rms[:, None]

    out_ptrs = out_ptr + t_offs[:, None] * D_out + d_offs[None, :]
    tl.store(out_ptrs, out_vals.to(tl.bfloat16), mask=t_mask[:, None])


@triton.jit
def _fused_rmsnorm_project_bwd_dx_kernel(
    x_ptr, W_ptr, grad_out_ptr, proj_out_ptr, inv_rms_ptr,
    grad_x_ptr,
    T, nC: tl.constexpr, D_out: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute ∂L/∂x_flat in a single pass!
    dx = c1 * dx_norm - scale * x
    where dx_norm = grad_proj @ W
    c1 = 1/rms.
    dot_term = sum_d(grad_proj_d * proj_out_d) * rms
    scale = dot_term / (rms^3 * nC) = sum_d(grad_proj_d * proj_out_d) / (rms^2 * nC)
    """
    pid = tl.program_id(0)
    t_start = pid * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T

    inv_rms = tl.load(inv_rms_ptr + t_offs, mask=t_mask, other=1.0)
    
    d_offs = tl.arange(0, D_out)
    go_ptrs = grad_out_ptr + t_offs[:, None] * D_out + d_offs[None, :]
    proj_ptrs = proj_out_ptr + t_offs[:, None] * D_out + d_offs[None, :]
    
    go = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0)
    proj_out = tl.load(proj_ptrs, mask=t_mask[:, None], other=0.0)

    # Compute scale directly from outputs without iterating x again.
    # scale = sum_d(go_d * proj_out_d) / (rms^2 * nC) = dot * inv_rms^2 / nC
    dot_term_fast = tl.sum(go.to(tl.float32) * proj_out.to(tl.float32), axis=1)  # (BLOCK_T,)
    scale = dot_term_fast * (inv_rms * inv_rms) / nC

    # Single pass to compute dx
    for k_start in tl.range(0, nC, BLOCK_K):
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(T, nC),
            strides=(nC, 1),
            offsets=(t_start, k_start),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0)
        )
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        
        # Load W_tile: need (D_out, BLOCK_K)
        w_block_ptr = tl.make_block_ptr(
            base=W_ptr,
            shape=(D_out, nC),
            strides=(nC, 1),
            offsets=(0, k_start),
            block_shape=(D_out, BLOCK_K),
            order=(1, 0)
        )
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1))
        
        # Tensor Core matmul: (BLOCK_T, D_out) @ (D_out, BLOCK_K)
        dx_norm_tile = tl.dot(go, w_tile)  # (BLOCK_T, BLOCK_K)

        dx_tile = inv_rms[:, None] * dx_norm_tile - scale[:, None] * x_tile
        
        gx_block_ptr = tl.make_block_ptr(
            base=grad_x_ptr,
            shape=(T, nC),
            strides=(nC, 1),
            offsets=(t_start, k_start),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0)
        )
        tl.store(gx_block_ptr, dx_tile.to(tl.bfloat16), boundary_check=(0, 1))

class FusedRMSNormProject(torch.autograd.Function):
    """RMSNorm(x_flat) @ W.T -> proj_out (T, D_out).
    Expects x_flat (T, nC) and W (D_out, nC).
    No biases — they are added later in the Python coefficients func.
    """

    @staticmethod
    def forward(ctx, x_flat, W):
        T, nC = x_flat.shape
        D_out = W.shape[0]
        out = torch.empty((T, D_out), device=x_flat.device, dtype=x_flat.dtype)
        inv_rms = torch.empty((T,), device=x_flat.device, dtype=torch.float32)
        
        BLOCK_K = min(128, triton.next_power_of_2(nC))
        BLOCK_T = 128
        _, nw, ns = _get_hw_config()
        
        grid = (triton.cdiv(T, BLOCK_T),)
        # SM120 limit: 99KB shared memory. BLOCK_T=128, BLOCK_K=128 needs ~32KB per stage for x.
        # num_stages=2 ensures we stay well under 99KB. SM90 can use 4 stages.
        _fused_rmsnorm_project_fwd_kernel[grid](
            x_flat, W,
            out, inv_rms,
            T, nC, D_out,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
            num_warps=nw, num_stages=ns
        )
        ctx.save_for_backward(x_flat, W, out, inv_rms)
        ctx.BLOCK_T = BLOCK_T
        ctx.BLOCK_K = BLOCK_K
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, W, proj_out, inv_rms = ctx.saved_tensors
        T, nC = x_flat.shape
        D_out = W.shape[0]
        BLOCK_T = getattr(ctx, 'BLOCK_T', 128)
        BLOCK_K = getattr(ctx, 'BLOCK_K', 128)
        grad_out = grad_output.contiguous()
        
        grad_x = torch.empty_like(x_flat)
        grid = (triton.cdiv(T, BLOCK_T),)
        _, nw, ns = _get_hw_config()
        # SMEM constraints based on HW Config
        _fused_rmsnorm_project_bwd_dx_kernel[grid](
            x_flat, W, grad_out, proj_out, inv_rms,
            grad_x,
            T, nC, D_out,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
            num_warps=nw, num_stages=ns
        )
        
        # dW natively with cuBLAS without instantiating x_norm!
        # dW = grad_out^T @ (x_flat / rms) = (grad_out * inv_rms)^T @ x_flat
        grad_out_scaled = (grad_out * inv_rms[:, None]).to(x_flat.dtype)
        grad_W = torch.matmul(grad_out_scaled.transpose(0, 1), x_flat)
        
        return grad_x, grad_W


# ═══════════════════════════════════════════════════════════════════════════
# Kernel 4: fused_post_res — THE CRITICAL KERNEL
#
# output[t, i, c] = Σ_j H_merged[t,i,j] * x[t,j,c]  +  h_post[t,i] * lo[t,c]
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_post_res_fwd_kernel_n4(
    x_ptr, lo_ptr, H_ptr, hp_ptr, out_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """n=4 specialization that reuses each x[j] tile across all 4 outputs."""
    tl.static_assert(n == 4)
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        lo_ptrs = tl.make_block_ptr(
            base=lo_ptr,
            shape=(T, C),
            strides=(C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        lo = tl.load(lo_ptrs, boundary_check=(0, 1)).to(tl.float32)

        hp0 = tl.load(hp_ptr + t_offs * n + 0, mask=t_mask, other=0.0).to(tl.float32)
        hp1 = tl.load(hp_ptr + t_offs * n + 1, mask=t_mask, other=0.0).to(tl.float32)
        hp2 = tl.load(hp_ptr + t_offs * n + 2, mask=t_mask, other=0.0).to(tl.float32)
        hp3 = tl.load(hp_ptr + t_offs * n + 3, mask=t_mask, other=0.0).to(tl.float32)

        acc0 = hp0[:, None] * lo
        acc1 = hp1[:, None] * lo
        acc2 = hp2[:, None] * lo
        acc3 = hp3[:, None] * lo

        for j in tl.static_range(4):
            x_ptrs = tl.make_block_ptr(
                base=x_ptr + j * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            xj = tl.load(x_ptrs, boundary_check=(0, 1)).to(tl.float32)

            h0j = tl.load(H_ptr + t_offs * (n * n) + 0 * n + j, mask=t_mask, other=0.0).to(tl.float32)
            h1j = tl.load(H_ptr + t_offs * (n * n) + 1 * n + j, mask=t_mask, other=0.0).to(tl.float32)
            h2j = tl.load(H_ptr + t_offs * (n * n) + 2 * n + j, mask=t_mask, other=0.0).to(tl.float32)
            h3j = tl.load(H_ptr + t_offs * (n * n) + 3 * n + j, mask=t_mask, other=0.0).to(tl.float32)

            acc0 += h0j[:, None] * xj
            acc1 += h1j[:, None] * xj
            acc2 += h2j[:, None] * xj
            acc3 += h3j[:, None] * xj

        out0 = tl.make_block_ptr(
            base=out_ptr + 0 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        out1 = tl.make_block_ptr(
            base=out_ptr + 1 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        out2 = tl.make_block_ptr(
            base=out_ptr + 2 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        out3 = tl.make_block_ptr(
            base=out_ptr + 3 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        tl.store(out0, acc0.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(out1, acc1.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(out2, acc2.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(out3, acc3.to(tl.bfloat16), boundary_check=(0, 1))


# ---------- K4 Backward: Kernel A (grad_x, grad_lo) ----------
# Tiles over (T, C). No reductions, no atomics.

@triton.jit
def _fused_post_res_bwd_xlo_kernel_n4(
    H_ptr, hp_ptr, grad_out_ptr, lo_ptr,
    grad_x_ptr, grad_lo_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """n=4 specialization: reuse grad_out[i] across all grad_x[j] updates."""
    tl.static_assert(n == 4)
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        grad_lo_acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        gx0 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        gx1 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        gx2 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        gx3 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

        for i in tl.static_range(4):
            hp_i = tl.load(hp_ptr + t_offs * n + i, mask=t_mask, other=0.0).to(tl.float32)

            go_ptr = tl.make_block_ptr(
                base=grad_out_ptr + i * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            go_i = tl.load(go_ptr, boundary_check=(0, 1)).to(tl.float32)

            grad_lo_acc += hp_i[:, None] * go_i

            h_i0 = tl.load(H_ptr + t_offs * (n * n) + i * n + 0, mask=t_mask, other=0.0).to(tl.float32)
            h_i1 = tl.load(H_ptr + t_offs * (n * n) + i * n + 1, mask=t_mask, other=0.0).to(tl.float32)
            h_i2 = tl.load(H_ptr + t_offs * (n * n) + i * n + 2, mask=t_mask, other=0.0).to(tl.float32)
            h_i3 = tl.load(H_ptr + t_offs * (n * n) + i * n + 3, mask=t_mask, other=0.0).to(tl.float32)

            gx0 += h_i0[:, None] * go_i
            gx1 += h_i1[:, None] * go_i
            gx2 += h_i2[:, None] * go_i
            gx3 += h_i3[:, None] * go_i

        glo_ptr = tl.make_block_ptr(
            base=grad_lo_ptr,
            shape=(T, C),
            strides=(C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        tl.store(glo_ptr, grad_lo_acc.to(tl.bfloat16), boundary_check=(0, 1))

        gx0_ptr = tl.make_block_ptr(
            base=grad_x_ptr + 0 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        gx1_ptr = tl.make_block_ptr(
            base=grad_x_ptr + 1 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        gx2_ptr = tl.make_block_ptr(
            base=grad_x_ptr + 2 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        gx3_ptr = tl.make_block_ptr(
            base=grad_x_ptr + 3 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        tl.store(gx0_ptr, gx0.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(gx1_ptr, gx1.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(gx2_ptr, gx2.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(gx3_ptr, gx3.to(tl.bfloat16), boundary_check=(0, 1))


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
    tl.static_assert(n == 4)
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
            # Load lo[t, c]
            lo_block_ptr = tl.make_block_ptr(
                base=lo_ptr,
                shape=(T, C),
                strides=(C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0)
            )
            lo = tl.load(lo_block_ptr, boundary_check=(0, 1)).to(tl.float32)

            # Load x[t, j, c] for all j
            x0_ptr = tl.make_block_ptr(base=x_ptr + 0 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            x1_ptr = tl.make_block_ptr(base=x_ptr + 1 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            x2_ptr = tl.make_block_ptr(base=x_ptr + 2 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            x3_ptr = tl.make_block_ptr(base=x_ptr + 3 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            
            x0 = tl.load(x0_ptr, boundary_check=(0, 1)).to(tl.float32)
            x1 = tl.load(x1_ptr, boundary_check=(0, 1)).to(tl.float32)
            x2 = tl.load(x2_ptr, boundary_check=(0, 1)).to(tl.float32)
            x3 = tl.load(x3_ptr, boundary_check=(0, 1)).to(tl.float32)

            # Load grad_out[t, i, c] for all i
            go0_ptr = tl.make_block_ptr(base=grad_out_ptr + 0 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            go1_ptr = tl.make_block_ptr(base=grad_out_ptr + 1 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            go2_ptr = tl.make_block_ptr(base=grad_out_ptr + 2 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            go3_ptr = tl.make_block_ptr(base=grad_out_ptr + 3 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))

            go0 = tl.load(go0_ptr, boundary_check=(0, 1)).to(tl.float32)
            go1 = tl.load(go1_ptr, boundary_check=(0, 1)).to(tl.float32)
            go2 = tl.load(go2_ptr, boundary_check=(0, 1)).to(tl.float32)
            go3 = tl.load(go3_ptr, boundary_check=(0, 1)).to(tl.float32)

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
        if n != 4:
            raise ValueError("FusedPostRes currently supports n=4 only")
        NUM_SMS, nw_default, ns = _get_hw_config()
        cc_major, _ = torch.cuda.get_device_capability()

        # Smaller tiles enable fusing all 4 outputs while keeping register pressure reasonable.
        BLOCK_T = 64 if cc_major >= 9 else 32
        BLOCK_C = 128 if C >= 128 else triton.next_power_of_2(C)
        nw = 8 if cc_major >= 9 else nw_default
        grid = (NUM_SMS,)
        _fused_post_res_fwd_kernel_n4[grid](
            x_streams, layer_output, H_merged, h_post, out,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
            num_warps=nw, num_stages=ns
        )
        ctx.save_for_backward(x_streams, layer_output, H_merged, h_post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_streams, layer_output, H_merged, h_post = ctx.saved_tensors
        T, n, C = x_streams.shape
        if n != 4:
            raise ValueError("FusedPostRes currently supports n=4 only")
        grad_out = grad_output.contiguous()
        NUM_SMS, nw_default, ns = _get_hw_config()
        cc_major, _ = torch.cuda.get_device_capability()

        BLOCK_T = 64 if cc_major >= 9 else 32
        BLOCK_C = 128 if C >= 128 else triton.next_power_of_2(C)
        nw = 8 if cc_major >= 9 else nw_default
        grid = (NUM_SMS,)

        grad_x = torch.empty_like(x_streams)
        grad_lo = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
        grad_H = torch.empty((T, n, n), device=x_streams.device, dtype=torch.float32)
        grad_hp = torch.empty((T, n), device=x_streams.device, dtype=torch.float32)

        _fused_post_res_bwd_xlo_kernel_n4[grid](
            H_merged, h_post, grad_out, layer_output,
            grad_x, grad_lo,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
            num_warps=nw, num_stages=ns,
        )

        BLOCK_T_B = 64
        BLOCK_C_B = min(256, triton.next_power_of_2(C))
        _fused_post_res_bwd_Hhp_kernel[grid](
            x_streams, layer_output, grad_out,
            grad_H, grad_hp,
            T, C, n,
            BLOCK_T=BLOCK_T_B, BLOCK_C=BLOCK_C_B, NUM_SMS=NUM_SMS,
            num_warps=nw_default, num_stages=ns,
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
            
            x_base_offset = j * C
            x_block_ptr = tl.make_block_ptr(
                base=x_ptr + x_base_offset,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0)
            )
            x_j = tl.load(x_block_ptr, boundary_check=(0, 1)).to(tl.float32)
            acc += hp_j[:, None] * x_j

        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(T, C),
            strides=(C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0)
        )
        tl.store(out_block_ptr, acc.to(tl.bfloat16), boundary_check=(0, 1))


@triton.jit
def _fused_pre_map_bwd_dx_kernel(
    h_pre_ptr, grad_out_ptr,
    grad_x_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Compute ∂L/∂x_streams. No reductions, no atomics."""
    tl.static_assert(n == 4)
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        go_ptr = tl.make_block_ptr(
            base=grad_out_ptr,
            shape=(T, C),
            strides=(C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        go = tl.load(go_ptr, boundary_check=(0, 1)).to(tl.float32)

        for j in tl.static_range(4):
            hp_j = tl.load(h_pre_ptr + t_offs * n + j, mask=t_mask, other=0.0).to(tl.float32)
            gx = hp_j[:, None] * go

            gx_ptr = tl.make_block_ptr(
                base=grad_x_ptr + j * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            tl.store(gx_ptr, gx.to(tl.bfloat16), boundary_check=(0, 1))


@triton.jit
def _fused_pre_map_bwd_hpre_kernel(
    x_ptr, grad_out_ptr,
    grad_hp_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Compute ∂L/∂h_pre by reducing over ALL of C. No atomics."""
    tl.static_assert(n == 4)
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_offs = tile_id * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        ghp0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp3 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        for c_start in tl.range(0, C, BLOCK_C):
            go_ptr = tl.make_block_ptr(
                base=grad_out_ptr,
                shape=(T, C),
                strides=(C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            go = tl.load(go_ptr, boundary_check=(0, 1)).to(tl.float32)

            x0_ptr = tl.make_block_ptr(
                base=x_ptr + 0 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            x1_ptr = tl.make_block_ptr(
                base=x_ptr + 1 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            x2_ptr = tl.make_block_ptr(
                base=x_ptr + 2 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            x3_ptr = tl.make_block_ptr(
                base=x_ptr + 3 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )

            x0 = tl.load(x0_ptr, boundary_check=(0, 1)).to(tl.float32)
            x1 = tl.load(x1_ptr, boundary_check=(0, 1)).to(tl.float32)
            x2 = tl.load(x2_ptr, boundary_check=(0, 1)).to(tl.float32)
            x3 = tl.load(x3_ptr, boundary_check=(0, 1)).to(tl.float32)

            ghp0 += tl.sum(x0 * go, axis=1)
            ghp1 += tl.sum(x1 * go, axis=1)
            ghp2 += tl.sum(x2 * go, axis=1)
            ghp3 += tl.sum(x3 * go, axis=1)

        tl.store(grad_hp_ptr + t_offs * n + 0, ghp0, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 1, ghp1, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 2, ghp2, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 3, ghp3, mask=t_mask)


class FusedPreMap(torch.autograd.Function):
    """layer_input = h_pre @ x_streams (weighted stream aggregation)."""

    @staticmethod
    def forward(ctx, x_streams, h_pre):
        T, n, C = x_streams.shape
        if n != 4:
            raise ValueError("FusedPreMap currently supports n=4 only")
        out = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
        NUM_SMS, nw, ns = _get_hw_config()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)
        # SM120 99KB limit
        _fused_pre_map_fwd_kernel[grid](
            x_streams, h_pre, out,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
            num_warps=nw, num_stages=ns
        )
        ctx.save_for_backward(x_streams, h_pre)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_streams, h_pre = ctx.saved_tensors
        T, n, C = x_streams.shape
        if n != 4:
            raise ValueError("FusedPreMap currently supports n=4 only")
        grad_out = grad_output.contiguous()
        
        NUM_SMS, nw, ns = _get_hw_config()
        BLOCK_T = 64
        BLOCK_C = min(256, triton.next_power_of_2(C))
        grid = (NUM_SMS,)

        grad_x = torch.empty_like(x_streams)
        _fused_pre_map_bwd_dx_kernel[grid](
            h_pre, grad_out,
            grad_x,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
            num_warps=nw, num_stages=ns
        )
        grad_hp = torch.empty((T, n), device=x_streams.device, dtype=torch.float32)
        _fused_pre_map_bwd_hpre_kernel[grid](
            x_streams, grad_out,
            grad_hp,
            T, C, n,
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, NUM_SMS=NUM_SMS,
            num_warps=nw, num_stages=ns
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
