"""Torch custom ops for Block Attention Residuals Triton kernels.

Exposes fused Triton kernels as dispatcher ops with FakeTensor support and
registered autograd formulas, compatible with torch.compile.

Follows the same patterns as mhc_triton_ops.py.
No autotune — these are memory-bound kernels where fixed configs are optimal.
"""

from __future__ import annotations

import os

import torch
import torch.library

_lib = torch.library.Library("nanoplm_bar", "DEF")

_lib.define(
    "fused_block_attnres("
    "  Tensor stacked, Tensor query, Tensor norm_weight, float eps"
    ") -> (Tensor result, Tensor alpha, Tensor inv_rms)"
)

_lib.define(
    "fused_block_attnres_bwd("
    "  Tensor grad_result, Tensor stacked, Tensor query, Tensor norm_weight,"
    "  Tensor alpha, Tensor inv_rms"
    ") -> (Tensor grad_stacked, Tensor R)"
)


def _triton_enabled() -> bool:
    v = os.getenv("NANOPLM_BLOCK_ATTNRES_TRITON", "1").strip().lower()
    return v not in {"0", "false", "off", "no"}


_STATE_MAX_COMPLETED_REFS = 8


def _pick_block_attnres_fwd_meta(
    D: int,
    *,
    default_warps: int,
    default_stages: int,
) -> tuple[int, int, int, int]:
    """Return (BLOCK_T, BLOCK_D, num_warps, num_stages) for forward.

    A100/SM80 benefits materially from a narrower token tile and higher warp
    count on the large packed-token shapes Block AttnRes uses in training.
    """
    props = torch.cuda.get_device_properties("cuda")
    cc = (props.major, props.minor)
    block_d = min(256, _next_power_of_2(D))
    if cc == (8, 0) and D >= 256:
        return 32, block_d, 8, 3
    if cc[0] >= 12 and D >= 256:
        # RTX 5090 / SM120 tuning on the real training shape (T=32768, D=768)
        # favored a narrower token tile while keeping the wide D tile.
        return 32, block_d, default_warps, default_stages
    return 64, block_d, default_warps, default_stages


def _pick_block_attnres_bwd_meta(
    D: int,
    *,
    default_warps: int,
    default_stages: int,
) -> tuple[int, int, int, int]:
    """Return (BLOCK_T, BLOCK_D, num_warps, num_stages) for backward."""
    props = torch.cuda.get_device_properties("cuda")
    cc = (props.major, props.minor)
    block_d = min(256, _next_power_of_2(D))
    if cc == (8, 0) and D >= 256:
        return 64, block_d, 8, 3
    if cc[0] >= 12 and D >= 256:
        # On SM120 the backward kernel is register-limited at 64x256.
        # A 32x128 tile reduced weighted runtime the most on the current
        # packed-token training shape while preserving the generic 4w/2s setup.
        return 32, min(128, _next_power_of_2(D)), default_warps, default_stages
    return 64, block_d, default_warps, default_stages


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ═══════════════════════════════════════════════════════════════════════════
# FakeTensor implementations
# ═══════════════════════════════════════════════════════════════════════════

@torch.library.register_fake("nanoplm_bar::fused_block_attnres")
def _fused_block_attnres_fake(
    stacked: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
):
    N, T, D = stacked.shape
    result = torch.empty((T, D), device=stacked.device, dtype=stacked.dtype)
    alpha = torch.empty((N, T), device=stacked.device, dtype=torch.float32)
    inv_rms = torch.empty((N, T), device=stacked.device, dtype=torch.float32)
    return result, alpha, inv_rms


@torch.library.register_fake("nanoplm_bar::fused_block_attnres_bwd")
def _fused_block_attnres_bwd_fake(
    grad_result: torch.Tensor,
    stacked: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    alpha: torch.Tensor,
    inv_rms: torch.Tensor,
):
    grad_stacked = torch.empty_like(stacked)
    D = stacked.shape[2]
    R = torch.empty((D,), device=stacked.device, dtype=torch.float32)
    return grad_stacked, R


# ═══════════════════════════════════════════════════════════════════════════
# CUDA implementations — fixed configs, no autotune
# ═══════════════════════════════════════════════════════════════════════════

@torch.library.impl(_lib, "fused_block_attnres", "CUDA")
def _fused_block_attnres_cuda(
    stacked: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    N, T, D = stacked.shape
    result = torch.empty((T, D), device=stacked.device, dtype=stacked.dtype)
    alpha = torch.empty((N, T), device=stacked.device, dtype=torch.float32)
    inv_rms = torch.empty((N, T), device=stacked.device, dtype=torch.float32)

    qw = (query.float() * norm_weight.float()).to(stacked.dtype)

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, BLOCK_D, nw, ns = _pick_block_attnres_fwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    k._block_attnres_fwd_kernel[(NUM_SMS,)](
        stacked, qw,
        result, alpha, inv_rms,
        T, D, N, eps,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )
    return result, alpha, inv_rms


@torch.library.impl(_lib, "fused_block_attnres_bwd", "CUDA")
def _fused_block_attnres_bwd_cuda(
    grad_result: torch.Tensor,
    stacked: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    alpha: torch.Tensor,
    inv_rms: torch.Tensor,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    N, T, D = stacked.shape
    grad_result = grad_result.contiguous()
    grad_stacked = torch.empty_like(stacked)

    scratch_f32 = torch.empty((2, N, T), device=stacked.device, dtype=torch.float32)
    grad_alpha_buf = scratch_f32[0]
    qw_dot_buf = scratch_f32[1]

    qw = (query.float() * norm_weight.float()).to(stacked.dtype)

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, BLOCK_D, nw, ns = _pick_block_attnres_bwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    R_local = torch.zeros((NUM_SMS, D), device=stacked.device, dtype=torch.float32)

    k._block_attnres_bwd_kernel[(NUM_SMS,)](
        grad_result, stacked, qw,
        alpha, inv_rms,
        grad_stacked, R_local,
        grad_alpha_buf, qw_dot_buf,
        T, D, N,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )
    R = R_local.sum(dim=0)
    return grad_stacked, R


def _pad_completed_refs(
    completed_refs: tuple[torch.Tensor, ...],
    partial: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    padded = list(completed_refs)
    if not padded:
        padded.append(partial)
    while len(padded) < _STATE_MAX_COMPLETED_REFS:
        padded.append(partial)
    return tuple(padded[:_STATE_MAX_COMPLETED_REFS])


def _fused_block_attnres_state_cuda(
    completed_refs: tuple[torch.Tensor, ...],
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    NC = len(completed_refs)
    T, D = partial.shape
    result = torch.empty((T, D), device=partial.device, dtype=partial.dtype)
    alpha = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)
    inv_rms = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)

    qw = (query.float() * norm_weight.float()).to(partial.dtype)

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, BLOCK_D, nw, ns = _pick_block_attnres_fwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    padded_refs = _pad_completed_refs(completed_refs, partial)
    k._block_attnres_state_fwd_kernel[(NUM_SMS,)](
        *padded_refs,
        partial,
        qw,
        result,
        alpha,
        inv_rms,
        T,
        D,
        NC,
        eps,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )
    return result, alpha, inv_rms


def _fused_block_attnres_state_bwd_cuda(
    grad_result: torch.Tensor,
    completed_refs: tuple[torch.Tensor, ...],
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    alpha: torch.Tensor,
    inv_rms: torch.Tensor,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    NC = len(completed_refs)
    T, D = partial.shape
    grad_result = grad_result.contiguous()
    grad_completed = torch.empty(
        (NC, T, D),
        device=partial.device,
        dtype=partial.dtype,
    )
    grad_partial = torch.empty_like(partial)

    scratch_f32 = torch.empty((2, NC + 1, T), device=partial.device, dtype=torch.float32)
    grad_alpha_buf = scratch_f32[0]
    qw_dot_buf = scratch_f32[1]

    qw = (query.float() * norm_weight.float()).to(partial.dtype)

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, BLOCK_D, nw, ns = _pick_block_attnres_bwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    R_local = torch.zeros((NUM_SMS, D), device=partial.device, dtype=torch.float32)

    padded_refs = _pad_completed_refs(completed_refs, partial)
    k._block_attnres_state_bwd_kernel[(NUM_SMS,)](
        grad_result,
        *padded_refs,
        partial,
        qw,
        alpha,
        inv_rms,
        grad_completed,
        grad_partial,
        R_local,
        grad_alpha_buf,
        qw_dot_buf,
        T,
        D,
        NC,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )
    R = R_local.sum(dim=0)
    return grad_completed, grad_partial, R


# ═══════════════════════════════════════════════════════════════════════════
# Autograd registration
# ═══════════════════════════════════════════════════════════════════════════

def _setup_block_attnres_context(ctx, inputs, output):
    """Save only tensor inputs (skip eps float) plus all outputs."""
    stacked, query, norm_weight, _eps = inputs
    if isinstance(output, tuple):
        ctx.save_for_backward(stacked, query, norm_weight, *output)
    else:
        ctx.save_for_backward(stacked, query, norm_weight, output)


def _fused_block_attnres_backward(ctx, grad_result, _grad_alpha, _grad_inv_rms):
    stacked, query, norm_weight, _result, alpha, inv_rms = ctx.saved_tensors

    grad_stacked, R = torch.ops.nanoplm_bar.fused_block_attnres_bwd(
        grad_result, stacked, query, norm_weight, alpha, inv_rms,
    )

    grad_query = norm_weight.float() * R
    grad_norm_weight = query.float() * R

    return grad_stacked, grad_query.to(query.dtype), grad_norm_weight.to(norm_weight.dtype), None


torch.library.register_autograd(
    "nanoplm_bar::fused_block_attnres",
    _fused_block_attnres_backward,
    setup_context=_setup_block_attnres_context,
)


class _BlockAttnResStateFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        partial: torch.Tensor,
        query: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        *completed_refs: torch.Tensor,
    ) -> torch.Tensor:
        result, alpha, inv_rms = _fused_block_attnres_state_cuda(
            completed_refs,
            partial,
            query,
            norm_weight,
            eps,
        )
        ctx.save_for_backward(
            partial,
            query,
            norm_weight,
            alpha,
            inv_rms,
            *completed_refs,
        )
        return result

    @staticmethod
    def backward(ctx, grad_result: torch.Tensor):
        partial, query, norm_weight, alpha, inv_rms, *completed_refs = ctx.saved_tensors
        grad_completed, grad_partial, R = _fused_block_attnres_state_bwd_cuda(
            grad_result,
            tuple(completed_refs),
            partial,
            query,
            norm_weight,
            alpha,
            inv_rms,
        )

        grad_query = norm_weight.float() * R
        grad_norm_weight = query.float() * R
        grad_completed_refs = tuple(
            grad_completed[i] for i in range(len(completed_refs))
        )
        return (
            grad_partial,
            grad_query.to(query.dtype),
            grad_norm_weight.to(norm_weight.dtype),
            None,
            *grad_completed_refs,
        )


def fused_block_attnres_from_state(
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    completed_refs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return _BlockAttnResStateFn.apply(
        partial,
        query,
        norm_weight,
        eps,
        *completed_refs,
    )
