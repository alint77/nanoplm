"""Triton-accelerated varlen Canon depthwise convolution.

Registers three ``torch.library`` ops under the ``nanoplm_canon`` namespace:

  varlen_conv_fwd      — forward pass
  varlen_conv_bwd_dx   — backward: grad_x (transpose conv, flipped weights)
  varlen_conv_bwd_dw_db — backward: grad_weight + grad_bias (reduction over T)

The ops are opaque to Inductor — it cannot fuse them with neighbouring
LayerNorm, which is the root cause of the pathological backward kernel
explosion in the original roll-based Python implementation.  Activation
checkpointing (``USE_ACTIVATION_CHECKPOINTING_CANON``) continues to work
because ``torch.library.register_autograd`` hooks into the same
``saved_tensors_default_hooks`` mechanism that ``checkpoint()`` relies on.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any

import torch
import torch.library
import triton

# ── Library + op signatures ───────────────────────────────────────────────

_lib = torch.library.Library("nanoplm_canon", "DEF")

_lib.define(
    "varlen_conv_fwd(Tensor x, Tensor seq_id, Tensor weight, Tensor bias, int radius) -> Tensor"
)
_lib.define(
    "varlen_conv_bwd_dx(Tensor grad_out, Tensor seq_id, Tensor weight, int radius) -> Tensor"
)
_lib.define(
    "varlen_conv_bwd_dw_db(Tensor grad_out, Tensor x, Tensor seq_id, int radius) -> (Tensor, Tensor)"
)

# ── FakeTensor implementations (for torch.compile shape inference) ────────


@torch.library.register_fake("nanoplm_canon::varlen_conv_fwd")
def _fwd_fake(x, seq_id, weight, bias, radius):
    return torch.empty_like(x)


@torch.library.register_fake("nanoplm_canon::varlen_conv_bwd_dx")
def _bwd_dx_fake(grad_out, seq_id, weight, radius):
    return torch.empty_like(grad_out)


@torch.library.register_fake("nanoplm_canon::varlen_conv_bwd_dw_db")
def _bwd_dw_db_fake(grad_out, x, seq_id, radius):
    C = x.shape[1]
    K = 2 * radius + 1
    grad_w = torch.empty(C, K, dtype=torch.float32, device=x.device)
    grad_b = torch.empty(C, dtype=torch.float32, device=x.device)
    return grad_w, grad_b


# ── Autotune control ─────────────────────────────────────────────────────

_AUTOTUNE_CONTROL = threading.local()


def _autotune_disable_depth() -> int:
    return int(getattr(_AUTOTUNE_CONTROL, "disable_depth", 0))


@contextmanager
def disable_autotune_temporarily():
    """Temporarily force the fallback (hardcoded block-size) launch path."""
    _AUTOTUNE_CONTROL.disable_depth = _autotune_disable_depth() + 1
    try:
        yield
    finally:
        _AUTOTUNE_CONTROL.disable_depth = max(0, _autotune_disable_depth() - 1)


def _autotune_enabled() -> bool:
    v = os.getenv("NANOPLM_CANON_TRITON_AUTOTUNE", "1").strip().lower()
    return v not in {"0", "false", "off", "no"} and _autotune_disable_depth() == 0


def _autotune_status_enabled() -> bool:
    v = os.getenv("NANOPLM_CANON_TRITON_AUTOTUNE_STATUS", "1").strip().lower()
    return v not in {"0", "false", "off", "no"}


_AUTOTUNE_STATUS_SEEN: set[tuple[str, tuple[Any, ...]]] = set()


def _autotune_status_begin(
    *, kernel_name: str, autotuner: Any, args_by_name: dict[str, Any],
) -> tuple[tuple[Any, ...], bool, Any] | None:
    if not _autotune_status_enabled():
        return None
    key: list[Any] = []
    for name in getattr(autotuner, "keys", ()):
        if name in args_by_name:
            key.append(args_by_name[name])
    tkey = tuple(key)
    tag = (kernel_name, tkey)
    if tag in _AUTOTUNE_STATUS_SEEN:
        return None
    _AUTOTUNE_STATUS_SEEN.add(tag)

    in_mem_hit = tkey in getattr(autotuner, "cache", {})
    bench_before = getattr(autotuner, "bench_time", None)
    if in_mem_hit:
        cfg = autotuner.cache[tkey]
        print(f"[nanoplm][triton] autotune cache hit: {kernel_name} key={tkey} cfg={cfg}", flush=True)
    else:
        num_cfgs = len(getattr(autotuner, "configs", ()))
        print(
            f"[nanoplm][triton] resolving autotune config: {kernel_name} key={tkey} "
            f"(may benchmark up to {num_cfgs} configs on first run)",
            flush=True,
        )
    return tkey, in_mem_hit, bench_before


def _autotune_status_end(
    *, kernel_name: str, autotuner: Any, state: tuple[tuple[Any, ...], bool, Any] | None,
) -> None:
    if not _autotune_status_enabled() or state is None:
        return
    tkey, in_mem_hit, bench_before = state
    if in_mem_hit:
        return
    bench_after = getattr(autotuner, "bench_time", None)
    ran_benchmark = bench_after is not None and bench_after != bench_before
    cfg = getattr(autotuner, "cache", {}).get(tkey, getattr(autotuner, "best_config", None))
    if ran_benchmark:
        print(f"[nanoplm][triton] autotune finished: {kernel_name} selected={cfg}", flush=True)
    else:
        print(f"[nanoplm][triton] autotune loaded from disk cache: {kernel_name} selected={cfg}", flush=True)


# ── CUDA implementations ─────────────────────────────────────────────────

# Fallback per-kernel block sizes tuned on RTX 5090 (T=65536, C=768, K=7, bf16).
_FWD_BLOCK_T, _FWD_BLOCK_C = 64, 128        # 78% roofline
_BWD_DX_BLOCK_T, _BWD_DX_BLOCK_C = 32, 128  # 83% roofline
_BWD_DW_BLOCK_T, _BWD_DW_BLOCK_C = 256, 64  # 48% roofline (atomic_add bound)


def _needs_fp32_accum(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


@torch.library.impl(_lib, "varlen_conv_fwd", "CUDA")
def _fwd_cuda(x, seq_id, weight, bias, radius):
    from . import canon_triton_kernels as k

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    T, C = x.shape
    fp32_acc = _needs_fp32_accum(x.dtype)

    out = torch.empty_like(x)
    if _autotune_enabled():
        autotuner = k._canon_fwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_fwd", autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(C, META["BLOCK_C"]))
        autotuner[grid](
            x, seq_id, weight, bias, out,
            T, C, x.stride(0), x.stride(1),
            RADIUS=radius, FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(kernel_name="canon_fwd", autotuner=autotuner, state=status)
    else:
        grid = (triton.cdiv(T, _FWD_BLOCK_T), triton.cdiv(C, _FWD_BLOCK_C))
        k._canon_fwd_kernel[grid](
            x, seq_id, weight, bias, out,
            T, C, x.stride(0), x.stride(1),
            RADIUS=radius,
            BLOCK_T=_FWD_BLOCK_T,
            BLOCK_C=_FWD_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return out


@torch.library.impl(_lib, "varlen_conv_bwd_dx", "CUDA")
def _bwd_dx_cuda(grad_out, seq_id, weight, radius):
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    weight = weight.contiguous()
    T, C = grad_out.shape
    fp32_acc = _needs_fp32_accum(grad_out.dtype)

    grad_x = torch.empty_like(grad_out)
    if _autotune_enabled():
        autotuner = k._canon_bwd_dx_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_bwd_dx", autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(C, META["BLOCK_C"]))
        autotuner[grid](
            grad_out, seq_id, weight, grad_x,
            T, C, grad_out.stride(0), grad_out.stride(1),
            RADIUS=radius, FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(kernel_name="canon_bwd_dx", autotuner=autotuner, state=status)
    else:
        grid = (triton.cdiv(T, _BWD_DX_BLOCK_T), triton.cdiv(C, _BWD_DX_BLOCK_C))
        k._canon_bwd_dx_kernel[grid](
            grad_out, seq_id, weight, grad_x,
            T, C, grad_out.stride(0), grad_out.stride(1),
            RADIUS=radius,
            BLOCK_T=_BWD_DX_BLOCK_T,
            BLOCK_C=_BWD_DX_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return grad_x


@torch.library.impl(_lib, "varlen_conv_bwd_dw_db", "CUDA")
def _bwd_dw_db_cuda(grad_out, x, seq_id, radius):
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    x = x.contiguous()
    T, C = x.shape
    K = 2 * radius + 1
    fp32_acc = _needs_fp32_accum(x.dtype)

    # Accumulate weight/bias grads in fp32 for half-precision, native dtype otherwise.
    acc_dtype = torch.float32 if fp32_acc else x.dtype
    grad_w = torch.zeros(C, K, dtype=acc_dtype, device=x.device)
    grad_b = torch.zeros(C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._canon_bwd_dw_db_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_bwd_dw_db", autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(C, META["BLOCK_C"]))
        autotuner[grid](
            grad_out, x, seq_id, grad_w, grad_b,
            T, C, grad_out.stride(0), grad_out.stride(1),
            RADIUS=radius, FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(kernel_name="canon_bwd_dw_db", autotuner=autotuner, state=status)
    else:
        grid = (triton.cdiv(T, _BWD_DW_BLOCK_T), triton.cdiv(C, _BWD_DW_BLOCK_C))
        k._canon_bwd_dw_db_kernel[grid](
            grad_out, x, seq_id, grad_w, grad_b,
            T, C, grad_out.stride(0), grad_out.stride(1),
            RADIUS=radius,
            BLOCK_T=_BWD_DW_BLOCK_T,
            BLOCK_C=_BWD_DW_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return grad_w, grad_b


# ── Autograd glue ─────────────────────────────────────────────────────────


def _setup_context(ctx, inputs, output):
    x, seq_id, weight, _bias, radius = inputs
    ctx.save_for_backward(x, seq_id, weight)
    ctx.radius = int(radius)


def _backward(ctx, grad_out):
    x, seq_id, weight = ctx.saved_tensors
    radius = ctx.radius
    grad_out = grad_out.contiguous()

    grad_x = torch.ops.nanoplm_canon.varlen_conv_bwd_dx(
        grad_out, seq_id, weight, radius
    )
    grad_w, grad_b = torch.ops.nanoplm_canon.varlen_conv_bwd_dw_db(
        grad_out, x, seq_id, radius
    )
    return grad_x, None, grad_w, grad_b, None


torch.library.register_autograd(
    "nanoplm_canon::varlen_conv_fwd",
    _backward,
    setup_context=_setup_context,
)


# ── Public API ────────────────────────────────────────────────────────────


def varlen_canon_conv(
    x: torch.Tensor,
    seq_id: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Triton-accelerated varlen depthwise Canon convolution.

    Drop-in replacement for ``_varlen_canon_inner``.

    Args:
        x: ``(T, C)`` packed token features.
        seq_id: ``(T,)`` int tensor mapping each token to its sequence index.
        weight: ``(C, K)`` depthwise conv weights.
        bias: ``(C,)`` conv bias.
        radius: Conv half-width (``kernel_size // 2``).
    """
    return torch.ops.nanoplm_canon.varlen_conv_fwd(x, seq_id, weight, bias, radius)
