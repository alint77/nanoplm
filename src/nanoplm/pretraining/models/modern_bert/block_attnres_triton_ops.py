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

# ═══════════════════════════════════════════════════════════════════════════
# Op definitions — stacked (non-state) path
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# Op definitions — state path (Phase 5: registered as custom ops for
# torch.compile compatibility, replacing old autograd.Function subclasses)
# ═══════════════════════════════════════════════════════════════════════════

_lib.define(
    "fused_block_attnres_state("
    "  Tensor completed_refs_stacked, Tensor partial, Tensor query,"
    "  Tensor norm_weight, float eps"
    ") -> (Tensor result, Tensor alpha, Tensor inv_rms)"
)

_lib.define(
    "fused_block_attnres_state_bwd("
    "  Tensor grad_result, Tensor completed_refs_stacked, Tensor partial,"
    "  Tensor query, Tensor norm_weight, Tensor alpha, Tensor inv_rms"
    ") -> (Tensor grad_completed, Tensor grad_partial, Tensor R)"
)

_lib.define(
    "fused_block_attnres_state_precomputed("
    "  Tensor completed_refs_stacked, Tensor partial, Tensor query,"
    "  Tensor norm_weight, Tensor precomputed_logits,"
    "  Tensor precomputed_inv_rms, float eps"
    ") -> (Tensor result, Tensor alpha, Tensor inv_rms)"
)

_lib.define(
    "fused_block_attnres_merge_partial("
    "  Tensor completed_refs_stacked, Tensor partial, Tensor query,"
    "  Tensor norm_weight, Tensor running_m, Tensor running_l,"
    "  Tensor running_acc, Tensor precomputed_logits,"
    "  Tensor precomputed_inv_rms, float eps"
    ") -> (Tensor result, Tensor alpha, Tensor inv_rms)"
)

_lib.define(
    "batched_completed_dreduction("
    "  Tensor completed_refs_stacked, Tensor[] qw_list, int NC, int Q, float eps"
    ") -> (Tensor logits, Tensor inv_rms)"
)

_lib.define(
    "batched_completed_dreduction_async("
    "  Tensor completed_refs_stacked, Tensor[] qw_list, int NC, int Q, float eps"
    ") -> (Tensor logits, Tensor inv_rms)"
)

_lib.define(
    "batched_completed_wsum("
    "  Tensor completed_refs_stacked, Tensor precomputed_logits,"
    "  Tensor precomputed_inv_rms, int NC, int Q, float eps"
    ") -> (Tensor batch_m, Tensor batch_l, Tensor batch_acc)"
)

_lib.define(
    "batched_completed_precompute("
    "  Tensor completed_refs_stacked, Tensor[] qw_list, int NC, int Q, float eps"
    ") -> (Tensor logits, Tensor inv_rms, Tensor batch_m, Tensor batch_l, Tensor batch_acc)"
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


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


_MAX_SUBLAYERS_PER_BLOCK = 8
_ASYNC_PRECOMPUTE_STREAMS: dict[tuple[str, int], torch.cuda.Stream] = {}
_ASYNC_PRECOMPUTE_EVENTS: dict[int, tuple[torch.cuda.Event, tuple[torch.Tensor, ...]]] = {}


def _pad_completed_refs(
    completed_refs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dummy: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    padded = list(completed_refs)
    if not padded:
        padded.append(dummy)
    while len(padded) < _STATE_MAX_COMPLETED_REFS:
        padded.append(dummy)
    return tuple(padded[:_STATE_MAX_COMPLETED_REFS])


def _ensure_completed_refs_stacked(
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    if isinstance(completed_refs, torch.Tensor):
        if completed_refs.ndim != 3:
            raise ValueError(
                f"expected completed_refs_stacked to have shape (NC, T, D), got {tuple(completed_refs.shape)}"
            )
        return completed_refs
    refs = tuple(completed_refs)
    if len(refs) == 0:
        raise ValueError("completed_refs must be non-empty")
    return torch.stack(refs, dim=0)


def _completed_refs_list_from_stacked(
    completed_refs_stacked: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> list[torch.Tensor]:
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    if completed_refs_stacked.ndim != 3:
        raise ValueError(
            f"expected completed_refs_stacked to have shape (NC, T, D), got {tuple(completed_refs_stacked.shape)}"
        )
    return list(completed_refs_stacked.unbind(0))


def _pad_qw_list(
    qw_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Pad qw_list to exactly _MAX_SUBLAYERS_PER_BLOCK entries."""
    padded = list(qw_list)
    if not padded:
        raise ValueError("qw_list must be non-empty")
    while len(padded) < _MAX_SUBLAYERS_PER_BLOCK:
        padded.append(padded[0])  # value is ignored since Q constrains iteration
    return padded[:_MAX_SUBLAYERS_PER_BLOCK]


def _get_async_precompute_stream(device: torch.device) -> torch.cuda.Stream:
    key = (device.type, device.index or 0)
    stream = _ASYNC_PRECOMPUTE_STREAMS.get(key)
    if stream is None:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
        _ASYNC_PRECOMPUTE_STREAMS[key] = stream
    return stream


def _async_precompute_key(tensor: torch.Tensor) -> int:
    # Python Tensor wrapper identity is not stable across compiled graph
    # boundaries; use the underlying CUDA storage pointer instead.
    return int(tensor.untyped_storage().data_ptr())


def _register_async_precompute(
    key_tensor: torch.Tensor,
    *,
    event: torch.cuda.Event,
    tensors: tuple[torch.Tensor, ...],
) -> None:
    _ASYNC_PRECOMPUTE_EVENTS[_async_precompute_key(key_tensor)] = (event, tensors)


def _wait_for_async_precompute(key_tensor: torch.Tensor) -> None:
    pending = _ASYNC_PRECOMPUTE_EVENTS.pop(_async_precompute_key(key_tensor), None)
    if pending is None:
        return
    event, tensors = pending
    current_stream = torch.cuda.current_stream(device=key_tensor.device)
    current_stream.wait_event(event)
    for tensor in tensors:
        tensor.record_stream(current_stream)


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


@torch.library.register_fake("nanoplm_bar::fused_block_attnres_state")
def _fused_block_attnres_state_fake(
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
):
    NC = completed_refs_stacked.shape[0]
    T, D = partial.shape
    result = torch.empty((T, D), device=partial.device, dtype=partial.dtype)
    alpha = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)
    inv_rms = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)
    return result, alpha, inv_rms


@torch.library.register_fake("nanoplm_bar::fused_block_attnres_state_bwd")
def _fused_block_attnres_state_bwd_fake(
    grad_result: torch.Tensor,
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    alpha: torch.Tensor,
    inv_rms: torch.Tensor,
):
    NC = completed_refs_stacked.shape[0]
    T, D = partial.shape
    grad_completed = torch.empty((NC, T, D), device=partial.device, dtype=partial.dtype)
    grad_partial = torch.empty_like(partial)
    R = torch.empty((D,), device=partial.device, dtype=torch.float32)
    return grad_completed, grad_partial, R


@torch.library.register_fake("nanoplm_bar::fused_block_attnres_state_precomputed")
def _fused_block_attnres_state_precomputed_fake(
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    eps: float,
):
    NC = completed_refs_stacked.shape[0]
    T, D = partial.shape
    result = torch.empty((T, D), device=partial.device, dtype=partial.dtype)
    alpha = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)
    inv_rms = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)
    return result, alpha, inv_rms


@torch.library.register_fake("nanoplm_bar::fused_block_attnres_merge_partial")
def _fused_block_attnres_merge_partial_fake(
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    running_m: torch.Tensor,
    running_l: torch.Tensor,
    running_acc: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    eps: float,
):
    NC = completed_refs_stacked.shape[0]
    T, D = partial.shape
    result = torch.empty((T, D), device=partial.device, dtype=partial.dtype)
    alpha = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)
    inv_rms = torch.empty((NC + 1, T), device=partial.device, dtype=torch.float32)
    return result, alpha, inv_rms


@torch.library.register_fake("nanoplm_bar::batched_completed_dreduction")
def _batched_completed_dreduction_fake(
    completed_refs_stacked: torch.Tensor,
    qw_list: list[torch.Tensor],
    NC: int,
    Q: int,
    eps: float,
):
    _nc, T, D = completed_refs_stacked.shape
    logits = torch.empty(
        (Q, NC, T), device=completed_refs_stacked.device, dtype=torch.float32
    )
    inv_rms = torch.empty((NC, T), device=completed_refs_stacked.device, dtype=torch.float32)
    return logits, inv_rms


@torch.library.register_fake("nanoplm_bar::batched_completed_dreduction_async")
def _batched_completed_dreduction_async_fake(
    completed_refs_stacked: torch.Tensor,
    qw_list: list[torch.Tensor],
    NC: int,
    Q: int,
    eps: float,
):
    return _batched_completed_dreduction_fake(
        completed_refs_stacked, qw_list, NC, Q, eps
    )


@torch.library.register_fake("nanoplm_bar::batched_completed_wsum")
def _batched_completed_wsum_fake(
    completed_refs_stacked: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    NC: int,
    Q: int,
    eps: float,
):
    _nc, T, D = completed_refs_stacked.shape
    batch_m = torch.empty((Q, T), device=completed_refs_stacked.device, dtype=torch.float32)
    batch_l = torch.empty((Q, T), device=completed_refs_stacked.device, dtype=torch.float32)
    batch_acc = torch.empty(
        (Q, T, D), device=completed_refs_stacked.device, dtype=torch.float32
    )
    return batch_m, batch_l, batch_acc


@torch.library.register_fake("nanoplm_bar::batched_completed_precompute")
def _batched_completed_precompute_fake(
    completed_refs_stacked: torch.Tensor,
    qw_list: list[torch.Tensor],
    NC: int,
    Q: int,
    eps: float,
):
    _nc, T, D = completed_refs_stacked.shape
    logits = torch.empty(
        (Q, NC, T), device=completed_refs_stacked.device, dtype=torch.float32
    )
    inv_rms = torch.empty((NC, T), device=completed_refs_stacked.device, dtype=torch.float32)
    batch_m = torch.empty((Q, T), device=completed_refs_stacked.device, dtype=torch.float32)
    batch_l = torch.empty((Q, T), device=completed_refs_stacked.device, dtype=torch.float32)
    batch_acc = torch.empty(
        (Q, T, D), device=completed_refs_stacked.device, dtype=torch.float32
    )
    return logits, inv_rms, batch_m, batch_l, batch_acc


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
        stacked,
        qw,
        result,
        alpha,
        inv_rms,
        T,
        D,
        N,
        eps,
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
        grad_result,
        stacked,
        qw,
        alpha,
        inv_rms,
        grad_stacked,
        R_local,
        grad_alpha_buf,
        qw_dot_buf,
        T,
        D,
        N,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )
    R = R_local.sum(dim=0)
    return grad_stacked, R


@torch.library.impl(_lib, "fused_block_attnres_state", "CUDA")
def _fused_block_attnres_state_cuda(
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    completed_refs = _completed_refs_list_from_stacked(completed_refs_stacked)
    NC = completed_refs_stacked.shape[0]
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


@torch.library.impl(_lib, "fused_block_attnres_state_bwd", "CUDA")
def _fused_block_attnres_state_bwd_cuda(
    grad_result: torch.Tensor,
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    alpha: torch.Tensor,
    inv_rms: torch.Tensor,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    completed_refs = _completed_refs_list_from_stacked(completed_refs_stacked)
    NC = completed_refs_stacked.shape[0]
    T, D = partial.shape
    grad_result = grad_result.contiguous()
    grad_completed = torch.empty(
        (NC, T, D),
        device=partial.device,
        dtype=partial.dtype,
    )
    grad_partial = torch.empty_like(partial)

    scratch_f32 = torch.empty(
        (2, NC + 1, T), device=partial.device, dtype=torch.float32
    )
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


@torch.library.impl(_lib, "fused_block_attnres_state_precomputed", "CUDA")
def _fused_block_attnres_state_precomputed_cuda(
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    eps: float,
):
    """Forward state kernel using precomputed completed-ref logits/inv_rms."""
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    _wait_for_async_precompute(precomputed_inv_rms)

    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    completed_refs = _completed_refs_list_from_stacked(completed_refs_stacked)
    NC = completed_refs_stacked.shape[0]
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
    k._block_attnres_state_fwd_precomputed_kernel[(NUM_SMS,)](
        *padded_refs,
        partial,
        qw,
        precomputed_logits,
        precomputed_inv_rms,
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


@torch.library.impl(_lib, "fused_block_attnres_merge_partial", "CUDA")
def _fused_block_attnres_merge_partial_cuda(
    completed_refs_stacked: torch.Tensor,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    running_m: torch.Tensor,
    running_l: torch.Tensor,
    running_acc: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    eps: float,
):
    """Launch the merge-partial kernel (Phase 2).

    Merges the partial source into the completed running state, producing
    the final weighted-sum result and alpha/inv_rms for the backward pass.
    """
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    _wait_for_async_precompute(precomputed_inv_rms)

    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    NC = completed_refs_stacked.shape[0]
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

    k._block_attnres_merge_partial_kernel[(NUM_SMS,)](
        partial,
        qw,
        running_m,
        running_l,
        running_acc,
        precomputed_logits,
        precomputed_inv_rms,
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


@torch.library.impl(_lib, "batched_completed_dreduction", "CUDA")
def _batched_completed_dreduction_cuda(
    completed_refs_stacked: torch.Tensor,
    qw_list: list[torch.Tensor],
    NC: int,
    Q: int,
    eps: float,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    assert NC > 0 and NC <= _STATE_MAX_COMPLETED_REFS
    assert Q > 0 and Q <= _MAX_SUBLAYERS_PER_BLOCK

    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    completed_refs = _completed_refs_list_from_stacked(completed_refs_stacked)
    _nc, T, D = completed_refs_stacked.shape
    logits = torch.empty(
        (Q, NC, T), device=completed_refs_stacked.device, dtype=torch.float32
    )
    inv_rms = torch.empty((NC, T), device=completed_refs_stacked.device, dtype=torch.float32)

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, BLOCK_D, nw, ns = _pick_block_attnres_fwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    dummy = completed_refs[0]
    padded_refs = _pad_completed_refs(completed_refs, dummy)
    padded_qw = _pad_qw_list(qw_list)

    k._block_attnres_batched_completed_dreduction_kernel[(NUM_SMS,)](
        *padded_refs,
        *padded_qw,
        logits,
        inv_rms,
        T,
        D,
        NC,
        Q,
        eps,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )
    return logits, inv_rms


@torch.library.impl(_lib, "batched_completed_dreduction_async", "CUDA")
def _batched_completed_dreduction_async_cuda(
    completed_refs_stacked: torch.Tensor,
    qw_list: list[torch.Tensor],
    NC: int,
    Q: int,
    eps: float,
):
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    completed_refs = _completed_refs_list_from_stacked(completed_refs_stacked)
    device = completed_refs_stacked.device
    aux_stream = _get_async_precompute_stream(device)
    with torch.cuda.stream(aux_stream):
        for tensor in (*completed_refs, *qw_list):
            tensor.record_stream(aux_stream)
        logits, inv_rms = _batched_completed_dreduction_cuda(
            completed_refs_stacked,
            qw_list,
            NC,
            Q,
            eps,
        )
        logits.record_stream(aux_stream)
        inv_rms.record_stream(aux_stream)
        event = torch.cuda.Event()
        event.record(aux_stream)
    _register_async_precompute(inv_rms, event=event, tensors=(logits, inv_rms))
    return logits, inv_rms


@torch.library.impl(_lib, "batched_completed_wsum", "CUDA")
def _batched_completed_wsum_cuda(
    completed_refs_stacked: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    NC: int,
    Q: int,
    eps: float,
):
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    assert NC > 0 and NC <= _STATE_MAX_COMPLETED_REFS
    _wait_for_async_precompute(precomputed_inv_rms)
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    completed_refs = _completed_refs_list_from_stacked(completed_refs_stacked)
    _nc, T, D = completed_refs_stacked.shape
    device = completed_refs_stacked.device

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, _, nw, ns = _pick_block_attnres_fwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    dummy = completed_refs[0]
    padded_refs = _pad_completed_refs(completed_refs, dummy)

    # Pre-allocate full output tensors and write each batch directly into slices,
    # avoiding the torch.stack (cat) ops that appeared in the profiler trace.
    all_m = torch.empty((Q, T), device=device, dtype=torch.float32)
    all_l = torch.empty((Q, T), device=device, dtype=torch.float32)
    all_acc = torch.empty((Q, T, D), device=device, dtype=torch.float32)

    q_idx = 0
    while q_idx < Q:
        remaining = Q - q_idx
        q_batch = 2 if remaining >= 2 else 1

        # For Q_BATCH=2 use BLOCK_D=128 (half) to fit 2 accumulators.
        # For Q_BATCH=1 use full BLOCK_D=256.
        if q_batch == 2:
            BLOCK_D = min(128, _next_power_of_2(D))
        else:
            BLOCK_D = min(256, _next_power_of_2(D))

        # Write directly into the output slices — no torch.stack needed.
        batch_m = all_m[q_idx : q_idx + q_batch]
        batch_l = all_l[q_idx : q_idx + q_batch]
        batch_acc = all_acc[q_idx : q_idx + q_batch]

        # Slice logits for this batch: (q_batch, NC, T)
        # The slice along dim 0 is already contiguous in row-major layout.
        batch_logits = precomputed_logits[q_idx : q_idx + q_batch]

        k._block_attnres_batched_completed_wsum_kernel[(NUM_SMS,)](
            *padded_refs,
            batch_logits,
            precomputed_inv_rms,
            batch_m,
            batch_l,
            batch_acc,
            T,
            D,
            NC,
            q_batch,
            eps,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
            NUM_SMS=NUM_SMS,
            num_warps=nw,
            num_stages=ns,
        )

        q_idx += q_batch

    return all_m, all_l, all_acc


@torch.library.impl(_lib, "batched_completed_precompute", "CUDA")
def _batched_completed_precompute_cuda(
    completed_refs_stacked: torch.Tensor,
    qw_list: list[torch.Tensor],
    NC: int,
    Q: int,
    eps: float,
):
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs_stacked)
    completed_refs = _completed_refs_list_from_stacked(completed_refs_stacked)
    device = completed_refs_stacked.device
    aux_stream = _get_async_precompute_stream(device)
    with torch.cuda.stream(aux_stream):
        for tensor in (*completed_refs, *qw_list):
            tensor.record_stream(aux_stream)
        logits, inv_rms = _batched_completed_dreduction_cuda(
            completed_refs_stacked,
            qw_list,
            NC,
            Q,
            eps,
        )
        batch_m, batch_l, batch_acc = _batched_completed_wsum_cuda(
            completed_refs_stacked,
            logits,
            inv_rms,
            NC,
            Q,
            eps,
        )
        outputs = (logits, inv_rms, batch_m, batch_l, batch_acc)
        for tensor in outputs:
            tensor.record_stream(aux_stream)
        event = torch.cuda.Event()
        event.record(aux_stream)
    _register_async_precompute(inv_rms, event=event, tensors=outputs)
    return outputs


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Batched backward (Q_BATCH sublayers sharing completed refs)
# ═══════════════════════════════════════════════════════════════════════════


def batched_state_bwd(
    grad_results: list[torch.Tensor],
    completed_refs: tuple[torch.Tensor, ...],
    partials: list[torch.Tensor],
    queries: list[torch.Tensor],
    norm_weights: list[torch.Tensor],
    alphas: list[torch.Tensor],
    inv_rmss: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Batched backward for Q_BATCH sublayers sharing completed refs.

    Args:
        grad_results: Q_BATCH tensors of shape (T, D)
        completed_refs: NC tensors of shape (T, D) -- shared across sublayers
        partials: Q_BATCH tensors of shape (T, D) -- per-sublayer
        queries: Q_BATCH tensors of shape (D,)
        norm_weights: Q_BATCH tensors of shape (D,)
        alphas: Q_BATCH tensors of shape (NC+1, T)
        inv_rmss: Q_BATCH tensors of shape (NC+1, T)

    Returns:
        grad_completed: (NC, T, D) -- sum of per-sublayer contributions
        grad_partials: list of Q_BATCH (T, D) tensors
        Rs: list of Q_BATCH (D,) tensors
    """
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    Q_BATCH = len(grad_results)
    assert 1 <= Q_BATCH <= 2, f"Q_BATCH must be 1 or 2, got {Q_BATCH}"
    NC = len(completed_refs)
    T, D = partials[0].shape

    # Ensure contiguous grad_results
    grad_results = [g.contiguous() for g in grad_results]

    # Output: summed grad_completed
    grad_completed = torch.empty(
        (NC, T, D),
        device=partials[0].device,
        dtype=partials[0].dtype,
    )

    # Per-sublayer outputs
    grad_partials = [torch.empty_like(p) for p in partials]

    # Precompute qw for each sublayer
    qws = [
        (queries[q].float() * norm_weights[q].float()).to(partials[q].dtype)
        for q in range(Q_BATCH)
    ]

    # Per-sublayer scratch buffers
    scratch_f32 = [
        torch.empty((2, NC + 1, T), device=partials[0].device, dtype=torch.float32)
        for _ in range(Q_BATCH)
    ]
    grad_alpha_bufs = [s[0] for s in scratch_f32]
    qw_dot_bufs = [s[1] for s in scratch_f32]

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, BLOCK_D, nw, ns = _pick_block_attnres_bwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    # Per-sublayer R_local buffers
    R_locals = [
        torch.zeros((NUM_SMS, D), device=partials[0].device, dtype=torch.float32)
        for _ in range(Q_BATCH)
    ]

    padded_refs = _pad_completed_refs(completed_refs, partials[0])

    # For Q_BATCH=1, sublayer 1 pointers are unused -- pass sublayer 0's as dummies
    if Q_BATCH == 1:
        gr1 = grad_results[0]
        p1 = partials[0]
        qw1 = qws[0]
        a1 = alphas[0]
        ir1 = inv_rmss[0]
        gp1 = grad_partials[0]
        rl1 = R_locals[0]
        ga1 = grad_alpha_bufs[0]
        qd1 = qw_dot_bufs[0]
    else:
        gr1 = grad_results[1]
        p1 = partials[1]
        qw1 = qws[1]
        a1 = alphas[1]
        ir1 = inv_rmss[1]
        gp1 = grad_partials[1]
        rl1 = R_locals[1]
        ga1 = grad_alpha_bufs[1]
        qd1 = qw_dot_bufs[1]

    k._block_attnres_batched_state_bwd_kernel[(NUM_SMS,)](
        # Sublayer 0
        grad_results[0],
        partials[0],
        qws[0],
        alphas[0],
        inv_rmss[0],
        # Sublayer 1
        gr1,
        p1,
        qw1,
        a1,
        ir1,
        # Shared completed refs
        *padded_refs,
        # Outputs
        grad_completed,
        grad_partials[0],
        gp1,
        R_locals[0],
        rl1,
        # Scratch
        grad_alpha_bufs[0],
        qw_dot_bufs[0],
        ga1,
        qd1,
        T,
        D,
        NC,
        Q_BATCH=Q_BATCH,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )

    Rs = [rl.sum(dim=0) for rl in R_locals]
    return grad_completed, grad_partials, Rs


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Single-query completed weighted sum (kept for tests / fallback)
# ═══════════════════════════════════════════════════════════════════════════


def completed_wsum(
    completed_refs: tuple[torch.Tensor, ...],
    precomputed_logits: torch.Tensor,  # (NC, T) f32 -- one query's logits
    precomputed_inv_rms: torch.Tensor,  # (NC, T) f32 -- shared
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute online-softmax running state over completed refs (Phase 2).

    Uses precomputed logits from Phase 1 batched D-reduction.  Each completed
    ref is read exactly once (just the weighted-sum pass -- D-reduction is
    already done).

    Args:
        completed_refs: NC completed ref tensors, each (T, D) bf16.
        precomputed_logits: (NC, T) f32 -- this query's completed logits.
        precomputed_inv_rms: (NC, T) f32 -- shared completed inv_rms.
        eps: RMS norm epsilon.

    Returns:
        running_m: (T,) f32 -- running max logit over completed sources.
        running_l: (T,) f32 -- running sum-exp over completed sources.
        running_acc: (T, D) f32 -- weighted accumulator over completed sources.
    """
    from . import block_attnres_triton_kernels as k
    from .mhc_triton_kernels import _get_hw_config

    NC = len(completed_refs)
    assert NC > 0 and NC <= _STATE_MAX_COMPLETED_REFS
    T, D = completed_refs[0].shape

    device = completed_refs[0].device
    running_m = torch.empty((T,), device=device, dtype=torch.float32)
    running_l = torch.empty((T,), device=device, dtype=torch.float32)
    running_acc = torch.empty((T, D), device=device, dtype=torch.float32)

    NUM_SMS, nw, ns = _get_hw_config()
    BLOCK_T, BLOCK_D, nw, ns = _pick_block_attnres_fwd_meta(
        D,
        default_warps=nw,
        default_stages=ns,
    )

    dummy = completed_refs[0]
    padded_refs = _pad_completed_refs(completed_refs, dummy)

    k._block_attnres_completed_wsum_kernel[(NUM_SMS,)](
        *padded_refs,
        precomputed_logits,
        precomputed_inv_rms,
        running_m,
        running_l,
        running_acc,
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
    return running_m, running_l, running_acc


# ═══════════════════════════════════════════════════════════════════════════
# Autograd registration
# ═══════════════════════════════════════════════════════════════════════════

# --- fused_block_attnres (stacked, non-state path) -----------------------


def _setup_block_attnres_context(ctx, inputs, output):
    """Save only tensor inputs (skip eps float) plus all outputs."""
    stacked, query, norm_weight, _eps = inputs
    if isinstance(output, tuple):
        ctx.save_for_backward(stacked, query, norm_weight, *output)
    else:
        ctx.save_for_backward(stacked, query, norm_weight, output)


def _fused_block_attnres_backward(ctx, grad_result, _grad_alpha, _grad_inv_rms):
    stacked, query, norm_weight, _result, alpha, inv_rms = ctx.saved_tensors

    # Route through the registered op so AOTAutograd sees an opaque custom op
    # instead of tracing into the Triton launch helper with FakeTensors.
    grad_stacked, R = torch.ops.nanoplm_bar.fused_block_attnres_bwd(
        grad_result,
        stacked,
        query,
        norm_weight,
        alpha,
        inv_rms,
    )

    grad_query = norm_weight.float() * R
    grad_norm_weight = query.float() * R

    return (
        grad_stacked,
        grad_query.to(query.dtype),
        grad_norm_weight.to(norm_weight.dtype),
        None,
    )


torch.library.register_autograd(
    "nanoplm_bar::fused_block_attnres",
    _fused_block_attnres_backward,
    setup_context=_setup_block_attnres_context,
)

# --- fused_block_attnres_state (Phase 0 state path) ----------------------


def _setup_state_context(ctx, inputs, output):
    completed_refs_stacked, partial, query, norm_weight, _eps = inputs
    _result, alpha, inv_rms = output
    ctx.save_for_backward(
        completed_refs_stacked,
        partial,
        query,
        norm_weight,
        alpha,
        inv_rms,
    )


def _state_backward(ctx, grad_result, _grad_alpha, _grad_inv_rms):
    completed_refs_stacked, partial, query, norm_weight, alpha, inv_rms = (
        ctx.saved_tensors
    )

    # Route through the registered backward op so compiled backward stays opaque.
    grad_completed, grad_partial, R = torch.ops.nanoplm_bar.fused_block_attnres_state_bwd(
        grad_result,
        completed_refs_stacked,
        partial,
        query,
        norm_weight,
        alpha,
        inv_rms,
    )

    grad_query = norm_weight.float() * R
    grad_norm_weight = query.float() * R

    return (
        grad_completed,  # completed_refs_stacked
        grad_partial,  # partial
        grad_query.to(query.dtype),  # query
        grad_norm_weight.to(norm_weight.dtype),  # norm_weight
        None,  # eps
    )


torch.library.register_autograd(
    "nanoplm_bar::fused_block_attnres_state",
    _state_backward,
    setup_context=_setup_state_context,
)

# --- fused_block_attnres_state_precomputed (Phase 1) ----------------------


def _setup_state_precomputed_context(ctx, inputs, output):
    completed_refs_stacked, partial, query, norm_weight, _logits, _inv_rms, _eps = inputs
    _result, alpha, inv_rms = output
    ctx.save_for_backward(
        completed_refs_stacked,
        partial,
        query,
        norm_weight,
        alpha,
        inv_rms,
    )


def _state_precomputed_backward(ctx, grad_result, _grad_alpha, _grad_inv_rms):
    completed_refs_stacked, partial, query, norm_weight, alpha, inv_rms = (
        ctx.saved_tensors
    )

    grad_completed, grad_partial, R = torch.ops.nanoplm_bar.fused_block_attnres_state_bwd(
        grad_result,
        completed_refs_stacked,
        partial,
        query,
        norm_weight,
        alpha,
        inv_rms,
    )

    grad_query = norm_weight.float() * R
    grad_norm_weight = query.float() * R

    return (
        grad_completed,  # completed_refs_stacked
        grad_partial,  # partial
        grad_query.to(query.dtype),  # query
        grad_norm_weight.to(norm_weight.dtype),  # norm_weight
        None,  # precomputed_logits
        None,  # precomputed_inv_rms
        None,  # eps
    )


torch.library.register_autograd(
    "nanoplm_bar::fused_block_attnres_state_precomputed",
    _state_precomputed_backward,
    setup_context=_setup_state_precomputed_context,
)

# --- fused_block_attnres_merge_partial (Phase 2) --------------------------


def _setup_merge_partial_context(ctx, inputs, output):
    completed_refs_stacked, partial, query, norm_weight = inputs[:4]
    # inputs[4:9] = running_m, running_l, running_acc, logits, inv_rms
    # inputs[9] = eps
    _result, alpha, inv_rms = output
    ctx.save_for_backward(
        completed_refs_stacked,
        partial,
        query,
        norm_weight,
        alpha,
        inv_rms,
    )


def _merge_partial_backward(ctx, grad_result, _grad_alpha, _grad_inv_rms):
    completed_refs_stacked, partial, query, norm_weight, alpha, inv_rms = (
        ctx.saved_tensors
    )

    grad_completed, grad_partial, R = torch.ops.nanoplm_bar.fused_block_attnres_state_bwd(
        grad_result,
        completed_refs_stacked,
        partial,
        query,
        norm_weight,
        alpha,
        inv_rms,
    )

    grad_query = norm_weight.float() * R
    grad_norm_weight = query.float() * R

    return (
        grad_completed,  # completed_refs_stacked
        grad_partial,  # partial
        grad_query.to(query.dtype),  # query
        grad_norm_weight.to(norm_weight.dtype),  # norm_weight
        None,  # running_m
        None,  # running_l
        None,  # running_acc
        None,  # precomputed_logits
        None,  # precomputed_inv_rms
        None,  # eps
    )


torch.library.register_autograd(
    "nanoplm_bar::fused_block_attnres_merge_partial",
    _merge_partial_backward,
    setup_context=_setup_merge_partial_context,
)

# --- batched_completed_dreduction / batched_completed_wsum ----------------
# These ops are used at block-start to precompute constants from completed
# refs.  No gradient needs to flow through them (completed_refs are detached
# in the model forward).  We register autograd with None grads so that
# torch.compile can trace through without warnings or graph breaks.


def _setup_dreduction_context(ctx, inputs, output):
    _completed_refs_stacked, qw_list, _NC, _Q, _eps = inputs
    ctx.n_qw = len(qw_list)


def _dreduction_backward(ctx, _grad_logits, _grad_inv_rms):
    # No gradients flow through the block-start precomputation.
    return (
        None,  # completed_refs_stacked
        [None] * ctx.n_qw,  # qw_list (Tensor[])
        None,  # NC
        None,  # Q
        None,  # eps
    )


torch.library.register_autograd(
    "nanoplm_bar::batched_completed_dreduction",
    _dreduction_backward,
    setup_context=_setup_dreduction_context,
)


torch.library.register_autograd(
    "nanoplm_bar::batched_completed_dreduction_async",
    _dreduction_backward,
    setup_context=_setup_dreduction_context,
)


def _setup_wsum_context(ctx, inputs, output):
    pass


def _wsum_backward(ctx, _grad_m, _grad_l, _grad_acc):
    # No gradients flow through the block-start precomputation.
    return (
        None,  # completed_refs_stacked
        None,  # precomputed_logits
        None,  # precomputed_inv_rms
        None,  # NC
        None,  # Q
        None,  # eps
    )


torch.library.register_autograd(
    "nanoplm_bar::batched_completed_wsum",
    _wsum_backward,
    setup_context=_setup_wsum_context,
)


def _setup_precompute_context(ctx, inputs, output):
    _completed_refs_stacked, qw_list, _NC, _Q, _eps = inputs
    ctx.n_qw = len(qw_list)


def _precompute_backward(
    ctx,
    _grad_logits,
    _grad_inv_rms,
    _grad_m,
    _grad_l,
    _grad_acc,
):
    return (
        None,  # completed_refs_stacked
        [None] * ctx.n_qw,  # qw_list (Tensor[])
        None,  # NC
        None,  # Q
        None,  # eps
    )


torch.library.register_autograd(
    "nanoplm_bar::batched_completed_precompute",
    _precompute_backward,
    setup_context=_setup_precompute_context,
)


# ═══════════════════════════════════════════════════════════════════════════
# Public API — thin wrappers that dispatch through torch.ops
# ═══════════════════════════════════════════════════════════════════════════


def fused_block_attnres_from_state(
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Forward state path — dispatches through the custom op."""
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs)
    result, _alpha, _inv_rms = torch.ops.nanoplm_bar.fused_block_attnres_state(
        completed_refs_stacked, partial, query, norm_weight, eps
    )
    return result


def fused_block_attnres_from_state_precomputed(
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    eps: float,
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Forward state using precomputed completed-ref logits/inv_rms (Phase 1).

    The precomputed_logits and precomputed_inv_rms come from
    batched_completed_dreduction() and are treated as constants (no grad).
    """
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs)
    result, _alpha, _inv_rms = (
        torch.ops.nanoplm_bar.fused_block_attnres_state_precomputed(
            completed_refs_stacked,
            partial,
            query,
            norm_weight,
            precomputed_logits,
            precomputed_inv_rms,
            eps,
        )
    )
    return result


def fused_block_attnres_online_merge(
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    running_m: torch.Tensor,
    running_l: torch.Tensor,
    running_acc: torch.Tensor,
    precomputed_logits: torch.Tensor,
    precomputed_inv_rms: torch.Tensor,
    eps: float,
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Forward state using online merge of partial into completed running state (Phase 2).

    The running state (m, l, acc) comes from completed_wsum() and captures
    the partial softmax over completed refs.  This function merges the partial
    source contribution, producing the final weighted-sum result.

    The precomputed_logits, precomputed_inv_rms, running_m, running_l,
    running_acc are treated as constants (no grad flows through them).
    """
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs)
    result, _alpha, _inv_rms = torch.ops.nanoplm_bar.fused_block_attnres_merge_partial(
        completed_refs_stacked,
        partial,
        query,
        norm_weight,
        running_m,
        running_l,
        running_acc,
        precomputed_logits,
        precomputed_inv_rms,
        eps,
    )
    return result


def batched_completed_dreduction(
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...],
    qw_list: list[torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the batched completed-block D-reduction kernel.

    Args:
        completed_refs: NC completed ref tensors, each (T, D) bf16.
        qw_list: Q precomputed qw = (query * norm_weight) vectors, each (D,) bf16.
        eps: RMS norm epsilon.

    Returns:
        logits: (Q, NC, T) f32 -- per-query completed logits.
        inv_rms: (NC, T) f32 -- shared completed inv_rms.
    """
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs)
    NC = completed_refs_stacked.shape[0]
    Q = len(qw_list)
    return torch.ops.nanoplm_bar.batched_completed_dreduction(
        completed_refs_stacked, qw_list, NC, Q, eps
    )


def batched_completed_dreduction_async(
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...],
    qw_list: list[torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the batched completed-block D-reduction on an auxiliary stream."""
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs)
    NC = completed_refs_stacked.shape[0]
    Q = len(qw_list)
    return torch.ops.nanoplm_bar.batched_completed_dreduction_async(
        completed_refs_stacked, qw_list, NC, Q, eps
    )


def batched_completed_precompute(
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...],
    qw_list: list[torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Phase 1+3 precompute on an auxiliary stream and return stacked outputs."""
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs)
    NC = completed_refs_stacked.shape[0]
    Q = len(qw_list)
    return torch.ops.nanoplm_bar.batched_completed_precompute(
        completed_refs_stacked, qw_list, NC, Q, eps
    )


def batched_completed_wsum(
    completed_refs: torch.Tensor | tuple[torch.Tensor, ...],
    precomputed_logits: torch.Tensor,  # (Q, NC, T) f32 -- all queries' logits
    precomputed_inv_rms: torch.Tensor,  # (NC, T) f32 -- shared
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched completed weighted sum with query-pair batching (Phase 3).

    Processes Q queries in pairs (Q_BATCH=2) at BLOCK_D=128, so each
    pair of queries shares a single read of completed refs.  For odd Q,
    the last query is processed alone (Q_BATCH=1) at full BLOCK_D.

    Args:
        completed_refs: NC completed ref tensors, each (T, D) bf16.
        precomputed_logits: (Q, NC, T) f32 -- all queries' completed logits.
        precomputed_inv_rms: (NC, T) f32 -- shared completed inv_rms.
        eps: RMS norm epsilon.

    Returns:
        running_m: (Q, T) f32
        running_l: (Q, T) f32
        running_acc: (Q, T, D) f32
    """
    completed_refs_stacked = _ensure_completed_refs_stacked(completed_refs)
    NC = completed_refs_stacked.shape[0]
    Q = precomputed_logits.shape[0]
    return torch.ops.nanoplm_bar.batched_completed_wsum(
        completed_refs_stacked,
        precomputed_logits,
        precomputed_inv_rms,
        NC,
        Q,
        eps,
    )
