"""Fused Triton kernels for Manifold-Constrained Hyper-Connections (mHC).

Following the DeepSeek paper's optimization strategy (§4.3.1):
  Kernel Group A (Eq 5-6): matmul x@phi done via cuBLAS (tensor cores), and RMS
      norm of x computed alongside. Both share a single read of x.
  Kernel Group B (Eq 7-10): Fused kernel for all lightweight coefficient ops:
      invr scaling + alpha + bias + sigmoid (pre/post) + Sinkhorn (res).
      All 24 scalars per token processed in one kernel launch.
  Kernel 2: Fused pre-aggregate: h_pre-weighted sum over n streams → (T, C)
  Kernel 3: Fused post+res merge: h_res @ x + h_post * f_out → (T, n, C)

All kernels are designed for n=4 (hard-coded for maximum unrolling/perf).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel B: Fused coefficient computation (Eq 7-10)
# Input: raw mix logits (T, 24), invr (T,)
# Output: h_pre (T,4), h_post (T,4), h_res (T,16)
# Applies: invr * alpha * mix + bias → sigmoid/Sinkhorn
# One program per token — purely scalar ops on 24 values + 20 Sinkhorn iters.
# ---------------------------------------------------------------------------

@triton.jit
def _mhc_coeffs_kernel(
    Mix_ptr,         # (T, M) float32 — raw x@phi results
    Invr_ptr,        # (T,) float32 — 1/RMS
    Alpha_pre_ptr, Alpha_post_ptr, Alpha_res_ptr,
    B_ptr,           # (M,) float32
    H_pre_ptr,       # (T, N) float32 output
    H_post_ptr,      # (T, N) float32 output
    H_res_ptr,       # (T, N*N) float32 output
    stride_mix_t,
    stride_hpre_t, stride_hpost_t, stride_hres_t,
    post_mult: tl.constexpr,
    TMAX: tl.constexpr,
    EPS: tl.constexpr,
    N: tl.constexpr,  # 4
):
    pid = tl.program_id(0)

    invr = tl.load(Invr_ptr + pid).to(tl.float32)
    alpha_pre = tl.load(Alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(Alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(Alpha_res_ptr).to(tl.float32)

    mix_base = pid * stride_mix_t

    # h_pre = sigmoid(mix[0:4] * invr * alpha_pre + bias[0:4])
    h_pre_base = pid * stride_hpre_t
    for i in tl.static_range(4):
        val = tl.load(Mix_ptr + mix_base + i).to(tl.float32)
        val = val * invr * alpha_pre + tl.load(B_ptr + i).to(tl.float32)
        tl.store(H_pre_ptr + h_pre_base + i, tl.sigmoid(val))

    # h_post = post_mult * sigmoid(mix[4:8] * invr * alpha_post + bias[4:8])
    h_post_base = pid * stride_hpost_t
    for i in tl.static_range(4):
        val = tl.load(Mix_ptr + mix_base + N + i).to(tl.float32)
        val = val * invr * alpha_post + tl.load(B_ptr + N + i).to(tl.float32)
        tl.store(H_post_ptr + h_post_base + i, tl.sigmoid(val) * post_mult)

    # h_res: Sinkhorn on mix[8:24] * invr * alpha_res + bias[8:24]
    r00 = tl.load(Mix_ptr + mix_base + 8).to(tl.float32)  * invr * alpha_res + tl.load(B_ptr + 8).to(tl.float32)
    r01 = tl.load(Mix_ptr + mix_base + 9).to(tl.float32)  * invr * alpha_res + tl.load(B_ptr + 9).to(tl.float32)
    r02 = tl.load(Mix_ptr + mix_base + 10).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 10).to(tl.float32)
    r03 = tl.load(Mix_ptr + mix_base + 11).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 11).to(tl.float32)
    r10 = tl.load(Mix_ptr + mix_base + 12).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 12).to(tl.float32)
    r11 = tl.load(Mix_ptr + mix_base + 13).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 13).to(tl.float32)
    r12 = tl.load(Mix_ptr + mix_base + 14).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 14).to(tl.float32)
    r13 = tl.load(Mix_ptr + mix_base + 15).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 15).to(tl.float32)
    r20 = tl.load(Mix_ptr + mix_base + 16).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 16).to(tl.float32)
    r21 = tl.load(Mix_ptr + mix_base + 17).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 17).to(tl.float32)
    r22 = tl.load(Mix_ptr + mix_base + 18).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 18).to(tl.float32)
    r23 = tl.load(Mix_ptr + mix_base + 19).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 19).to(tl.float32)
    r30 = tl.load(Mix_ptr + mix_base + 20).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 20).to(tl.float32)
    r31 = tl.load(Mix_ptr + mix_base + 21).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 21).to(tl.float32)
    r32 = tl.load(Mix_ptr + mix_base + 22).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 22).to(tl.float32)
    r33 = tl.load(Mix_ptr + mix_base + 23).to(tl.float32) * invr * alpha_res + tl.load(B_ptr + 23).to(tl.float32)

    # exp with row-max subtraction
    max0 = tl.maximum(tl.maximum(r00, r01), tl.maximum(r02, r03))
    max1 = tl.maximum(tl.maximum(r10, r11), tl.maximum(r12, r13))
    max2 = tl.maximum(tl.maximum(r20, r21), tl.maximum(r22, r23))
    max3 = tl.maximum(tl.maximum(r30, r31), tl.maximum(r32, r33))

    r00 = tl.exp(r00 - max0); r01 = tl.exp(r01 - max0); r02 = tl.exp(r02 - max0); r03 = tl.exp(r03 - max0)
    r10 = tl.exp(r10 - max1); r11 = tl.exp(r11 - max1); r12 = tl.exp(r12 - max1); r13 = tl.exp(r13 - max1)
    r20 = tl.exp(r20 - max2); r21 = tl.exp(r21 - max2); r22 = tl.exp(r22 - max2); r23 = tl.exp(r23 - max2)
    r30 = tl.exp(r30 - max3); r31 = tl.exp(r31 - max3); r32 = tl.exp(r32 - max3); r33 = tl.exp(r33 - max3)

    for _t in tl.static_range(TMAX):
        s0 = r00 + r01 + r02 + r03 + EPS
        s1 = r10 + r11 + r12 + r13 + EPS
        s2 = r20 + r21 + r22 + r23 + EPS
        s3 = r30 + r31 + r32 + r33 + EPS
        r00 /= s0; r01 /= s0; r02 /= s0; r03 /= s0
        r10 /= s1; r11 /= s1; r12 /= s1; r13 /= s1
        r20 /= s2; r21 /= s2; r22 /= s2; r23 /= s2
        r30 /= s3; r31 /= s3; r32 /= s3; r33 /= s3
        c0 = r00 + r10 + r20 + r30 + EPS
        c1 = r01 + r11 + r21 + r31 + EPS
        c2 = r02 + r12 + r22 + r32 + EPS
        c3 = r03 + r13 + r23 + r33 + EPS
        r00 /= c0; r01 /= c1; r02 /= c2; r03 /= c3
        r10 /= c0; r11 /= c1; r12 /= c2; r13 /= c3
        r20 /= c0; r21 /= c1; r22 /= c2; r23 /= c3
        r30 /= c0; r31 /= c1; r32 /= c2; r33 /= c3

    h_res_base = pid * stride_hres_t
    tl.store(H_res_ptr + h_res_base + 0,  r00)
    tl.store(H_res_ptr + h_res_base + 1,  r01)
    tl.store(H_res_ptr + h_res_base + 2,  r02)
    tl.store(H_res_ptr + h_res_base + 3,  r03)
    tl.store(H_res_ptr + h_res_base + 4,  r10)
    tl.store(H_res_ptr + h_res_base + 5,  r11)
    tl.store(H_res_ptr + h_res_base + 6,  r12)
    tl.store(H_res_ptr + h_res_base + 7,  r13)
    tl.store(H_res_ptr + h_res_base + 8,  r20)
    tl.store(H_res_ptr + h_res_base + 9,  r21)
    tl.store(H_res_ptr + h_res_base + 10, r22)
    tl.store(H_res_ptr + h_res_base + 11, r23)
    tl.store(H_res_ptr + h_res_base + 12, r30)
    tl.store(H_res_ptr + h_res_base + 13, r31)
    tl.store(H_res_ptr + h_res_base + 14, r32)
    tl.store(H_res_ptr + h_res_base + 15, r33)


# ---------------------------------------------------------------------------
# Kernel 2: Fused pre-aggregate — h_pre-weighted sum: (T, n, C) → (T, C)
# ---------------------------------------------------------------------------

@triton.jit
def _mhc_pre_kernel(
    X_ptr, H_pre_ptr, Out_ptr,
    stride_x_t, stride_x_n, stride_x_c,
    stride_hp_t, stride_hp_n,
    stride_o_t, stride_o_c,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    hp0 = tl.load(H_pre_ptr + pid_t * stride_hp_t + 0 * stride_hp_n).to(tl.float32)
    hp1 = tl.load(H_pre_ptr + pid_t * stride_hp_t + 1 * stride_hp_n).to(tl.float32)
    hp2 = tl.load(H_pre_ptr + pid_t * stride_hp_t + 2 * stride_hp_n).to(tl.float32)
    hp3 = tl.load(H_pre_ptr + pid_t * stride_hp_t + 3 * stride_hp_n).to(tl.float32)

    x_base = pid_t * stride_x_t
    x0 = tl.load(X_ptr + x_base + 0 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_ptr + x_base + 1 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)
    x2 = tl.load(X_ptr + x_base + 2 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)
    x3 = tl.load(X_ptr + x_base + 3 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)

    out = hp0 * x0 + hp1 * x1 + hp2 * x2 + hp3 * x3
    tl.store(Out_ptr + pid_t * stride_o_t + c_offsets * stride_o_c, out.to(X_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Kernel 3: Fused post+res merge — h_res @ x + h_post * f_out
# ---------------------------------------------------------------------------

@triton.jit
def _mhc_post_res_kernel(
    X_ptr, F_ptr, H_res_ptr, H_post_ptr, Out_ptr,
    stride_x_t, stride_x_n, stride_x_c,
    stride_f_t, stride_f_c,
    stride_hr_t, stride_hp_t,
    stride_o_t, stride_o_n, stride_o_c,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_nc = tl.program_id(1)
    n_blocks_c = tl.cdiv(C, BLOCK_C)
    stream_idx = pid_nc // n_blocks_c
    c_block = pid_nc % n_blocks_c

    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    hr_base = pid_t * stride_hr_t + stream_idx * N
    hr0 = tl.load(H_res_ptr + hr_base + 0).to(tl.float32)
    hr1 = tl.load(H_res_ptr + hr_base + 1).to(tl.float32)
    hr2 = tl.load(H_res_ptr + hr_base + 2).to(tl.float32)
    hr3 = tl.load(H_res_ptr + hr_base + 3).to(tl.float32)

    hp = tl.load(H_post_ptr + pid_t * stride_hp_t + stream_idx).to(tl.float32)

    x_base = pid_t * stride_x_t
    x0 = tl.load(X_ptr + x_base + 0 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_ptr + x_base + 1 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)
    x2 = tl.load(X_ptr + x_base + 2 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)
    x3 = tl.load(X_ptr + x_base + 3 * stride_x_n + c_offsets * stride_x_c, mask=c_mask, other=0.0).to(tl.float32)

    res = hr0 * x0 + hr1 * x1 + hr2 * x2 + hr3 * x3

    f = tl.load(F_ptr + pid_t * stride_f_t + c_offsets * stride_f_c, mask=c_mask, other=0.0).to(tl.float32)
    out = res + hp * f

    o_base = pid_t * stride_o_t + stream_idx * stride_o_n
    tl.store(Out_ptr + o_base + c_offsets * stride_o_c, out.to(X_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def mhc_coeffs_fused(
    mix: torch.Tensor,         # (T, m) float32 — already computed x @ phi
    invr: torch.Tensor,        # (T,) float32 — 1/RMS
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    bias: torch.Tensor,        # (m,) float32
    n: int = 4,
    tmax: int = 20,
    eps: float = 1e-6,
    post_mult: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused: invr*alpha*mix + bias → sigmoid/Sinkhorn."""
    T = mix.shape[0]
    h_pre = torch.empty(T, n, device=mix.device, dtype=torch.float32)
    h_post = torch.empty(T, n, device=mix.device, dtype=torch.float32)
    h_res = torch.empty(T, n * n, device=mix.device, dtype=torch.float32)

    _mhc_coeffs_kernel[(T,)](
        mix, invr, alpha_pre, alpha_post, alpha_res, bias,
        h_pre, h_post, h_res,
        mix.stride(0),
        h_pre.stride(0), h_post.stride(0), h_res.stride(0),
        post_mult=post_mult,
        TMAX=tmax,
        EPS=eps,
        N=n,
    )
    return h_pre, h_post, h_res.view(T, n, n)


def mhc_pre_aggregate(
    x: torch.Tensor,       # (T, n, C)
    h_pre: torch.Tensor,   # (T, n)
) -> torch.Tensor:
    T, n, C = x.shape
    out = torch.empty(T, C, device=x.device, dtype=x.dtype)
    BLOCK_C = min(triton.next_power_of_2(C), 1024)
    grid = (T, triton.cdiv(C, BLOCK_C))
    _mhc_pre_kernel[grid](
        x, h_pre, out,
        x.stride(0), x.stride(1), x.stride(2),
        h_pre.stride(0), h_pre.stride(1),
        out.stride(0), out.stride(1),
        C=C, N=n, BLOCK_C=BLOCK_C,
    )
    return out


def mhc_post_res_merge(
    x: torch.Tensor,        # (T, n, C)
    f_out: torch.Tensor,     # (T, C)
    h_res: torch.Tensor,     # (T, n, n)
    h_post: torch.Tensor,    # (T, n)
) -> torch.Tensor:
    T, n, C = x.shape
    out = torch.empty_like(x)
    BLOCK_C = min(triton.next_power_of_2(C), 1024)
    n_blocks_c = triton.cdiv(C, BLOCK_C)
    grid = (T, n * n_blocks_c)
    _mhc_post_res_kernel[grid](
        x, f_out,
        h_res.view(T, n * n), h_post,
        out,
        x.stride(0), x.stride(1), x.stride(2),
        f_out.stride(0), f_out.stride(1),
        n * n, h_post.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        C=C, N=n, BLOCK_C=BLOCK_C,
    )
    return out


# ---------------------------------------------------------------------------
# Drop-in MHCLayer replacement using fused kernels
# ---------------------------------------------------------------------------

class FusedMHCLayer(nn.Module):
    """Drop-in replacement for MHCLayer using fused Triton kernels.

    Optimization strategy (following DeepSeek paper §4.3.1):
    - x @ phi uses cuBLAS (tensor cores) for the matmul
    - RMS norm of x is computed alongside
    - All coefficient ops (invr scaling, sigmoid, Sinkhorn) are fused in one kernel
    - Pre-aggregate and post+res merge each use a single fused kernel
    """

    def __init__(
        self,
        layer: nn.Module,
        n: int,
        c: int,
        tmax: int = 20,
        rms_eps: float = 1e-6,
        sinkhorn_eps: float = 1e-6,
        post_mult: float = 2.0,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.n = n
        self.c = c
        self.tmax = tmax
        self.rms_eps = rms_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.post_mult = post_mult

        k = n * c
        m = n * n + 2 * n

        self.phi = nn.Parameter(torch.empty(k, m))
        self.b = nn.Parameter(torch.zeros(m))
        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.phi, std=0.02)

    def _coeffs(self, x: torch.Tensor):
        n, c = self.n, self.c
        lead = x.shape[:-2]
        x_mat = x.reshape(-1, n * c)  # (T, k)

        # Paper Eq 5-6: matmul via cuBLAS + RMS norm
        invr = torch.rsqrt(x_mat.pow(2).float().mean(dim=-1) + self.rms_eps)  # (T,)
        mix = (x_mat @ self.phi.to(x_mat.dtype)).float()  # (T, m)

        # Paper Eq 7-10: fused coefficient kernel (sigmoid + Sinkhorn)
        h_pre, h_post, h_res = mhc_coeffs_fused(
            mix, invr,
            self.alpha_pre, self.alpha_post, self.alpha_res,
            self.b,
            n=n, tmax=self.tmax, eps=self.sinkhorn_eps, post_mult=self.post_mult,
        )

        h_pre = h_pre.to(x.dtype).reshape(*lead, n)
        h_post = h_post.to(x.dtype).reshape(*lead, n)
        h_res = h_res.to(x.dtype).reshape(*lead, n, n)
        return h_pre, h_post, h_res

    def forward(self, x: torch.Tensor, **layer_kwargs) -> torch.Tensor:
        h_pre, h_post, h_res = self._coeffs(x)

        # Fused pre-aggregate
        lead = x.shape[:-2]
        x_flat = x.reshape(-1, self.n, self.c)
        h_pre_flat = h_pre.reshape(-1, self.n)
        x_in = mhc_pre_aggregate(x_flat, h_pre_flat)
        x_in = x_in.reshape(*lead, self.c)

        # Run sublayer
        f_out = self.layer(x_in, **layer_kwargs)

        # Fused post+res merge
        f_out_flat = f_out.reshape(-1, self.c)
        h_res_flat = h_res.reshape(-1, self.n, self.n)
        h_post_flat = h_post.reshape(-1, self.n)
        out = mhc_post_res_merge(x_flat, f_out_flat, h_res_flat, h_post_flat)
        return out.reshape(*lead, self.n, self.c)
