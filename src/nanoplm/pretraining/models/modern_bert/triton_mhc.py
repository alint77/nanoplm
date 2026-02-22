"""Triton-backed fused kernels for mHC pre/post routing.

This module implements fused kernels for:
- F_pre:  x_agg = H_pre @ x
- F_post,res: H_res @ x + H_post^T * f_out

Both ops have custom autograd functions with Triton kernels on CUDA (N=4)
and a numerically equivalent pure-PyTorch fallback for unsupported cases.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - covered by fallback logic
    triton = None
    tl = None
    _HAS_TRITON = False


_SUPPORTED_N = 4
_MHC_BACKEND = os.environ.get("NANOPLM_MHC_BACKEND", "auto").strip().lower()


# ---------------------------------------------------------------------------
# Coefficients (K1+K2+K3) helpers
# ---------------------------------------------------------------------------


def _mhc_sinkhorn_unrolled(logits: torch.Tensor, tmax: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn-Knopp for N=4 written in pure torch for stable autograd."""
    r0 = logits[..., 0:4]
    r1 = logits[..., 4:8]
    r2 = logits[..., 8:12]
    r3 = logits[..., 12:16]

    r0 = torch.exp(r0 - torch.amax(r0, dim=-1, keepdim=True))
    r1 = torch.exp(r1 - torch.amax(r1, dim=-1, keepdim=True))
    r2 = torch.exp(r2 - torch.amax(r2, dim=-1, keepdim=True))
    r3 = torch.exp(r3 - torch.amax(r3, dim=-1, keepdim=True))

    for _ in range(tmax):
        s0 = r0[..., 0:1] + r0[..., 1:2] + r0[..., 2:3] + r0[..., 3:4]
        s1 = r1[..., 0:1] + r1[..., 1:2] + r1[..., 2:3] + r1[..., 3:4]
        s2 = r2[..., 0:1] + r2[..., 1:2] + r2[..., 2:3] + r2[..., 3:4]
        s3 = r3[..., 0:1] + r3[..., 1:2] + r3[..., 2:3] + r3[..., 3:4]

        r0 = r0 / (s0 + eps)
        r1 = r1 / (s1 + eps)
        r2 = r2 / (s2 + eps)
        r3 = r3 / (s3 + eps)

        c = r0 + r1 + r2 + r3 + eps
        r0 = r0 / c
        r1 = r1 / c
        r2 = r2 / c
        r3 = r3 / c

    return torch.cat([r0, r1, r2, r3], dim=-1).view(*logits.shape[:-1], 4, 4)


def mhc_coeffs(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    b: torch.Tensor,
    tmax: int,
    rms_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (h_pre, h_post, h_res) with paper's norm reordering trick.

    Uses the K1 reordering in pure torch (single read of flattened x for the
    projection path): mix = (x @ phi) / rms(x).
    """
    n = x.shape[-2]
    c = x.shape[-1]
    lead = x.shape[:-2]

    x_mat = x.reshape(-1, n * c)
    invr = torch.rsqrt(x_mat.pow(2).float().mean(dim=-1, keepdim=True) + rms_eps)
    mix = (x_mat @ phi.to(x_mat.dtype)).float() * invr

    pre_logits = mix[..., :n] * alpha_pre.to(mix.dtype) + b[:n].to(mix.dtype)
    post_logits = mix[..., n : 2 * n] * alpha_post.to(mix.dtype) + b[n : 2 * n].to(mix.dtype)
    res_logits = mix[..., 2 * n :] * alpha_res.to(mix.dtype) + b[2 * n :].to(mix.dtype)

    h_pre = torch.sigmoid(pre_logits.float()).to(x.dtype)
    h_post = (torch.sigmoid(post_logits.float()) * post_mult).to(x.dtype)
    h_res = _mhc_sinkhorn_unrolled(res_logits.float(), tmax=tmax, eps=sinkhorn_eps).to(x.dtype)

    h_pre = h_pre.reshape(*lead, n)
    h_post = h_post.reshape(*lead, n)
    h_res = h_res.reshape(*lead, n, n)
    return h_pre, h_post, h_res


# ---------------------------------------------------------------------------
# Pure torch references
# ---------------------------------------------------------------------------


def _mhc_pre_torch(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    return (x * h_pre.unsqueeze(-1)).sum(dim=-2)


def _mhc_post_torch(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    is_compiling = (
        hasattr(torch, "compiler")
        and hasattr(torch.compiler, "is_compiling")
        and torch.compiler.is_compiling()
    )

    if x.shape[-2] == _SUPPORTED_N and is_compiling:
        # N=4 specialization that compiles into a single fast fused graph and
        # avoids the higher activation footprint of the explicit-loop baseline.
        x0, x1, x2, x3 = x.unbind(dim=-2)

        h00 = h_res[..., 0, 0:1]
        h01 = h_res[..., 0, 1:2]
        h02 = h_res[..., 0, 2:3]
        h03 = h_res[..., 0, 3:4]
        h10 = h_res[..., 1, 0:1]
        h11 = h_res[..., 1, 1:2]
        h12 = h_res[..., 1, 2:3]
        h13 = h_res[..., 1, 3:4]
        h20 = h_res[..., 2, 0:1]
        h21 = h_res[..., 2, 1:2]
        h22 = h_res[..., 2, 2:3]
        h23 = h_res[..., 2, 3:4]
        h30 = h_res[..., 3, 0:1]
        h31 = h_res[..., 3, 1:2]
        h32 = h_res[..., 3, 2:3]
        h33 = h_res[..., 3, 3:4]

        p0 = h_post[..., 0:1]
        p1 = h_post[..., 1:2]
        p2 = h_post[..., 2:3]
        p3 = h_post[..., 3:4]

        y0 = h00 * x0 + h01 * x1 + h02 * x2 + h03 * x3 + p0 * f_out
        y1 = h10 * x0 + h11 * x1 + h12 * x2 + h13 * x3 + p1 * f_out
        y2 = h20 * x0 + h21 * x1 + h22 * x2 + h23 * x3 + p2 * f_out
        y3 = h30 * x0 + h31 * x1 + h32 * x2 + h33 * x3 + p3 * f_out
        return torch.stack((y0, y1, y2, y3), dim=-2)

    res = torch.einsum("...ij,...jc->...ic", h_res, x)
    return res + h_post.unsqueeze(-1) * f_out.unsqueeze(-2)


def _mhc_post_legacy(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    n = x.shape[-2]
    res_gather = torch.zeros_like(x)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc = acc + h_res[..., i : i + 1, j : j + 1] * x[..., j : j + 1, :]
        res_gather[..., i : i + 1, :] = acc
    return res_gather + h_post.unsqueeze(-1) * f_out.unsqueeze(-2)


def _resolve_backend() -> str:
    """Resolve mHC backend mode.

    Supported values for NANOPLM_MHC_BACKEND:
      - auto   : torch path under torch.compile, Triton otherwise
      - triton : force Triton autograd kernels for pre/post
      - torch  : force pure-torch pre/post equations
      - legacy : force original explicit-loop post path
    """
    mode = _MHC_BACKEND
    if mode in {"triton", "torch", "legacy"}:
        return mode

    # auto/default: keep compile path in torch equations to maximize
    # inductor fusion opportunities; use Triton in eager mode.
    is_compiling = (
        hasattr(torch, "compiler")
        and hasattr(torch.compiler, "is_compiling")
        and torch.compiler.is_compiling()
    )
    return "torch" if is_compiling else "triton"


def _can_use_triton(x: torch.Tensor, n: int, use_triton: Optional[bool]) -> bool:
    if use_triton is False:
        return False

    return (
        _HAS_TRITON
        and x.is_cuda
        and n == _SUPPORTED_N
        and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
    )


# ---------------------------------------------------------------------------
# Triton kernels: F_pre
# ---------------------------------------------------------------------------


if _HAS_TRITON:

    @triton.jit
    def _mhc_pre_fwd_kernel(
        x_ptr,
        h_pre_ptr,
        out_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_cb = tl.program_id(1)
        offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs_c < C

        base_x_t = pid_t * N * C
        base_h_t = pid_t * N

        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for i in range(N):
            h = tl.load(h_pre_ptr + base_h_t + i).to(tl.float32)
            xv = tl.load(x_ptr + base_x_t + i * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            acc += h * xv

        tl.store(out_ptr + pid_t * C + offs_c, acc, mask=mask)


    @triton.jit
    def _mhc_pre_bwd_dx_kernel(
        grad_out_ptr,
        h_pre_ptr,
        grad_x_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_cb = tl.program_id(2)

        offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs_c < C

        h = tl.load(h_pre_ptr + pid_t * N + pid_n).to(tl.float32)
        go = tl.load(grad_out_ptr + pid_t * C + offs_c, mask=mask, other=0.0).to(tl.float32)
        gx = h * go

        tl.store(grad_x_ptr + (pid_t * N + pid_n) * C + offs_c, gx, mask=mask)


    @triton.jit
    def _mhc_pre_bwd_h_kernel(
        x_ptr,
        grad_out_ptr,
        grad_h_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_n = tl.program_id(1)

        acc = tl.zeros((), dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C

            x = tl.load(x_ptr + (pid_t * N + pid_n) * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            go = tl.load(grad_out_ptr + pid_t * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(x * go, axis=0)

        tl.store(grad_h_ptr + pid_t * N + pid_n, acc)


    # -----------------------------------------------------------------------
    # Triton kernels: F_post,res
    # -----------------------------------------------------------------------

    @triton.jit
    def _mhc_post_fwd_kernel(
        x_ptr,
        f_ptr,
        h_post_ptr,
        h_res_ptr,
        out_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_i = tl.program_id(1)
        pid_cb = tl.program_id(2)

        offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs_c < C

        base_x_t = pid_t * N * C
        base_out_t = pid_t * N * C
        base_f_t = pid_t * C
        base_hp_t = pid_t * N
        base_hr_t = pid_t * N * N

        hp = tl.load(h_post_ptr + base_hp_t + pid_i).to(tl.float32)
        f = tl.load(f_ptr + base_f_t + offs_c, mask=mask, other=0.0).to(tl.float32)

        acc = hp * f
        for j in range(N):
            h = tl.load(h_res_ptr + base_hr_t + pid_i * N + j).to(tl.float32)
            xv = tl.load(x_ptr + base_x_t + j * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            acc += h * xv

        tl.store(out_ptr + base_out_t + pid_i * C + offs_c, acc, mask=mask)


    @triton.jit
    def _mhc_post_bwd_x_kernel(
        grad_out_ptr,
        h_res_ptr,
        grad_x_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_j = tl.program_id(1)
        pid_cb = tl.program_id(2)

        offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs_c < C

        base_go_t = pid_t * N * C
        base_hr_t = pid_t * N * N

        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for i in range(N):
            h = tl.load(h_res_ptr + base_hr_t + i * N + pid_j).to(tl.float32)
            go = tl.load(grad_out_ptr + base_go_t + i * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            acc += h * go

        tl.store(grad_x_ptr + (pid_t * N + pid_j) * C + offs_c, acc, mask=mask)


    @triton.jit
    def _mhc_post_bwd_f_kernel(
        grad_out_ptr,
        h_post_ptr,
        grad_f_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_cb = tl.program_id(1)

        offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs_c < C

        base_go_t = pid_t * N * C
        base_hp_t = pid_t * N

        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for i in range(N):
            hp = tl.load(h_post_ptr + base_hp_t + i).to(tl.float32)
            go = tl.load(grad_out_ptr + base_go_t + i * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            acc += hp * go

        tl.store(grad_f_ptr + pid_t * C + offs_c, acc, mask=mask)


    @triton.jit
    def _mhc_post_bwd_h_post_kernel(
        grad_out_ptr,
        f_ptr,
        grad_h_post_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_i = tl.program_id(1)

        acc = tl.zeros((), dtype=tl.float32)
        base_go_t = pid_t * N * C
        base_f_t = pid_t * C

        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            go = tl.load(grad_out_ptr + base_go_t + pid_i * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            f = tl.load(f_ptr + base_f_t + offs_c, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(go * f, axis=0)

        tl.store(grad_h_post_ptr + pid_t * N + pid_i, acc)


    @triton.jit
    def _mhc_post_bwd_h_res_kernel(
        grad_out_ptr,
        x_ptr,
        grad_h_res_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_i = tl.program_id(1)
        pid_j = tl.program_id(2)

        acc = tl.zeros((), dtype=tl.float32)
        base_go_t = pid_t * N * C
        base_x_t = pid_t * N * C

        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            go = tl.load(grad_out_ptr + base_go_t + pid_i * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            xv = tl.load(x_ptr + base_x_t + pid_j * C + offs_c, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(go * xv, axis=0)

        tl.store(grad_h_res_ptr + (pid_t * N + pid_i) * N + pid_j, acc)


def _default_block_c(c: int) -> int:
    if c >= 128:
        return 128
    if c >= 64:
        return 64
    return 32


def _mhc_pre_triton_fwd(x_2d: torch.Tensor, h_pre_2d: torch.Tensor) -> torch.Tensor:
    t, n, c = x_2d.shape
    out = torch.empty((t, c), device=x_2d.device, dtype=x_2d.dtype)
    block_c = _default_block_c(c)
    grid = (t, triton.cdiv(c, block_c))
    _mhc_pre_fwd_kernel[grid](
        x_2d,
        h_pre_2d,
        out,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return out


def _mhc_pre_triton_bwd(
    x_2d: torch.Tensor,
    h_pre_2d: torch.Tensor,
    grad_out_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t, n, c = x_2d.shape
    block_c = _default_block_c(c)

    grad_x = torch.empty_like(x_2d)
    grid_dx = (t, n, triton.cdiv(c, block_c))
    _mhc_pre_bwd_dx_kernel[grid_dx](
        grad_out_2d,
        h_pre_2d,
        grad_x,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )

    grad_h = torch.empty_like(h_pre_2d)
    grid_h = (t, n)
    _mhc_pre_bwd_h_kernel[grid_h](
        x_2d,
        grad_out_2d,
        grad_h,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return grad_x, grad_h


def _mhc_post_triton_fwd(
    x_2d: torch.Tensor,
    f_out_2d: torch.Tensor,
    h_post_2d: torch.Tensor,
    h_res_3d: torch.Tensor,
) -> torch.Tensor:
    t, n, c = x_2d.shape
    out = torch.empty_like(x_2d)
    block_c = _default_block_c(c)

    grid = (t, n, triton.cdiv(c, block_c))
    _mhc_post_fwd_kernel[grid](
        x_2d,
        f_out_2d,
        h_post_2d,
        h_res_3d,
        out,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return out


def _mhc_post_triton_bwd(
    x_2d: torch.Tensor,
    f_out_2d: torch.Tensor,
    h_post_2d: torch.Tensor,
    h_res_3d: torch.Tensor,
    grad_out_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t, n, c = x_2d.shape
    block_c = _default_block_c(c)

    grad_x = torch.empty_like(x_2d)
    grid_x = (t, n, triton.cdiv(c, block_c))
    _mhc_post_bwd_x_kernel[grid_x](
        grad_out_2d,
        h_res_3d,
        grad_x,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )

    grad_f = torch.empty_like(f_out_2d)
    grid_f = (t, triton.cdiv(c, block_c))
    _mhc_post_bwd_f_kernel[grid_f](
        grad_out_2d,
        h_post_2d,
        grad_f,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )

    grad_h_post = torch.empty_like(h_post_2d)
    grid_hp = (t, n)
    _mhc_post_bwd_h_post_kernel[grid_hp](
        grad_out_2d,
        f_out_2d,
        grad_h_post,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )

    grad_h_res = torch.empty_like(h_res_3d)
    grid_hr = (t, n, n)
    _mhc_post_bwd_h_res_kernel[grid_hr](
        grad_out_2d,
        x_2d,
        grad_h_res,
        C=c,
        N=n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )

    return grad_x, grad_f, grad_h_post, grad_h_res


class MHCPreFunction(torch.autograd.Function):
    """Autograd wrapper for fused pre-aggregation F_pre."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, h_pre: torch.Tensor, use_triton: Optional[bool] = None):
        n = x.shape[-2]
        c = x.shape[-1]
        lead = x.shape[:-2]

        x_2d = x.reshape(-1, n, c).contiguous()
        h_pre_2d = h_pre.reshape(-1, n).contiguous()

        use_triton_impl = _can_use_triton(x_2d, n=n, use_triton=use_triton)
        if use_triton_impl:
            out_2d = _mhc_pre_triton_fwd(x_2d, h_pre_2d)
        else:
            out_2d = _mhc_pre_torch(x_2d, h_pre_2d)

        ctx.use_triton_impl = use_triton_impl
        ctx.n = n
        ctx.c = c
        ctx.lead = lead
        ctx.save_for_backward(x_2d, h_pre_2d)

        return out_2d.reshape(*lead, c)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_2d, h_pre_2d = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, ctx.c).contiguous()

        if ctx.use_triton_impl:
            grad_x_2d, grad_h_pre_2d = _mhc_pre_triton_bwd(x_2d, h_pre_2d, grad_out_2d)
        else:
            grad_x_2d = h_pre_2d.unsqueeze(-1) * grad_out_2d.unsqueeze(-2)
            grad_h_pre_2d = (x_2d * grad_out_2d.unsqueeze(-2)).sum(dim=-1)

        grad_x = grad_x_2d.reshape(*ctx.lead, ctx.n, ctx.c)
        grad_h_pre = grad_h_pre_2d.reshape(*ctx.lead, ctx.n)
        return grad_x, grad_h_pre, None


class MHCPostFunction(torch.autograd.Function):
    """Autograd wrapper for fused post-distribution F_post,res."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        f_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
        use_triton: Optional[bool] = None,
    ):
        n = x.shape[-2]
        c = x.shape[-1]
        lead = x.shape[:-2]

        x_2d = x.reshape(-1, n, c).contiguous()
        f_out_2d = f_out.reshape(-1, c).contiguous()
        h_post_2d = h_post.reshape(-1, n).contiguous()
        h_res_3d = h_res.reshape(-1, n, n).contiguous()

        use_triton_impl = _can_use_triton(x_2d, n=n, use_triton=use_triton)
        if use_triton_impl:
            out_2d = _mhc_post_triton_fwd(x_2d, f_out_2d, h_post_2d, h_res_3d)
        else:
            out_2d = _mhc_post_torch(x_2d, f_out_2d, h_post_2d, h_res_3d)

        ctx.use_triton_impl = use_triton_impl
        ctx.n = n
        ctx.c = c
        ctx.lead = lead
        ctx.save_for_backward(x_2d, f_out_2d, h_post_2d, h_res_3d)

        return out_2d.reshape(*lead, n, c)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_2d, f_out_2d, h_post_2d, h_res_3d = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, ctx.n, ctx.c).contiguous()

        if ctx.use_triton_impl:
            grad_x_2d, grad_f_2d, grad_h_post_2d, grad_h_res_3d = _mhc_post_triton_bwd(
                x_2d,
                f_out_2d,
                h_post_2d,
                h_res_3d,
                grad_out_2d,
            )
        else:
            grad_x_2d = torch.einsum("tij,tic->tjc", h_res_3d, grad_out_2d)
            grad_f_2d = (h_post_2d.unsqueeze(-1) * grad_out_2d).sum(dim=1)
            grad_h_post_2d = (grad_out_2d * f_out_2d.unsqueeze(1)).sum(dim=-1)
            grad_h_res_3d = torch.einsum("tic,tjc->tij", grad_out_2d, x_2d)

        grad_x = grad_x_2d.reshape(*ctx.lead, ctx.n, ctx.c)
        grad_f = grad_f_2d.reshape(*ctx.lead, ctx.c)
        grad_h_post = grad_h_post_2d.reshape(*ctx.lead, ctx.n)
        grad_h_res = grad_h_res_3d.reshape(*ctx.lead, ctx.n, ctx.n)
        return grad_x, grad_f, grad_h_post, grad_h_res, None


def mhc_pre_aggregate(
    x: torch.Tensor,
    h_pre: torch.Tensor,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """F_pre wrapper with Triton + autograd fallback."""
    mode = _resolve_backend()
    if mode in {"torch", "legacy"}:
        return _mhc_pre_torch(x, h_pre)
    return MHCPreFunction.apply(x, h_pre, use_triton)


def mhc_post_distribute(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """F_post,res wrapper with Triton + autograd fallback."""
    mode = _resolve_backend()
    if mode == "torch":
        return _mhc_post_torch(x, f_out, h_post, h_res)
    if mode == "legacy":
        return _mhc_post_legacy(x, f_out, h_post, h_res)
    return MHCPostFunction.apply(x, f_out, h_post, h_res, use_triton)
