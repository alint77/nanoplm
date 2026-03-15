from __future__ import annotations

import torch
import torch.library
import triton
import triton.language as tl


_lib = torch.library.Library("nanoplm_moe", "DEF")
_lib.define("permute(Tensor x, Tensor indices) -> Tensor")
_lib.define("unpermute(Tensor x, Tensor indices) -> Tensor")


@triton.jit
def _moe_permute_kernel(
    x_ptr,
    idx_ptr,
    y_ptr,
    m,
    d,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < m) & (offs_d[None, :] < d)

    src_rows = tl.load(idx_ptr + offs_m, mask=offs_m < m, other=0).to(tl.int64)
    x_ptrs = x_ptr + src_rows[:, None] * stride_xm + offs_d[None, :] * stride_xd
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd
    vals = tl.load(x_ptrs, mask=mask, other=0)
    tl.store(y_ptrs, vals, mask=mask)


@triton.jit
def _moe_unpermute_kernel(
    x_ptr,
    idx_ptr,
    y_ptr,
    m,
    d,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < m) & (offs_d[None, :] < d)

    dst_rows = tl.load(idx_ptr + offs_m, mask=offs_m < m, other=0).to(tl.int64)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd
    y_ptrs = y_ptr + dst_rows[:, None] * stride_ym + offs_d[None, :] * stride_yd
    vals = tl.load(x_ptrs, mask=mask, other=0)
    tl.store(y_ptrs, vals, mask=mask)


def _validate_inputs(x: torch.Tensor, indices: torch.Tensor, opname: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{opname} expects a 2D tensor, got shape {tuple(x.shape)}")
    if indices.ndim != 1:
        raise ValueError(
            f"{opname} expects a 1D index tensor, got shape {tuple(indices.shape)}"
        )
    if indices.numel() != x.shape[0]:
        raise ValueError(
            f"{opname} expects a full row permutation: {indices.numel()} vs {x.shape[0]}"
        )


def _launch_permute_kernel(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _validate_inputs(x, indices, "nanoplm_moe::permute")
    x = x.contiguous()
    indices = indices.contiguous()
    y = torch.empty_like(x)
    m, d = x.shape
    grid = (triton.cdiv(m, 64), triton.cdiv(d, 128))
    _moe_permute_kernel[grid](
        x,
        indices,
        y,
        m,
        d,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=64,
        BLOCK_D=128,
        num_warps=4,
        num_stages=2,
    )
    return y


def _launch_unpermute_kernel(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _validate_inputs(x, indices, "nanoplm_moe::unpermute")
    x = x.contiguous()
    indices = indices.contiguous()
    y = torch.empty_like(x)
    m, d = x.shape
    grid = (triton.cdiv(m, 64), triton.cdiv(d, 128))
    _moe_unpermute_kernel[grid](
        x,
        indices,
        y,
        m,
        d,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=64,
        BLOCK_D=128,
        num_warps=4,
        num_stages=2,
    )
    return y


@torch.library.register_fake("nanoplm_moe::permute")
def _permute_fake(x: torch.Tensor, indices: torch.Tensor):
    _validate_inputs(x, indices, "nanoplm_moe::permute")
    return torch.empty_like(x)


@torch.library.register_fake("nanoplm_moe::unpermute")
def _unpermute_fake(x: torch.Tensor, indices: torch.Tensor):
    _validate_inputs(x, indices, "nanoplm_moe::unpermute")
    return torch.empty_like(x)


@torch.library.impl(_lib, "permute", "CUDA")
def _permute_cuda(x: torch.Tensor, indices: torch.Tensor):
    return _launch_permute_kernel(x, indices)


@torch.library.impl(_lib, "unpermute", "CUDA")
def _unpermute_cuda(x: torch.Tensor, indices: torch.Tensor):
    return _launch_unpermute_kernel(x, indices)


@torch.library.impl(_lib, "permute", "CPU")
def _permute_cpu(x: torch.Tensor, indices: torch.Tensor):
    _validate_inputs(x, indices, "nanoplm_moe::permute")
    return x[indices]


@torch.library.impl(_lib, "unpermute", "CPU")
def _unpermute_cpu(x: torch.Tensor, indices: torch.Tensor):
    _validate_inputs(x, indices, "nanoplm_moe::unpermute")
    out = torch.empty_like(x)
    out[indices] = x
    return out


def _setup_context(ctx, inputs, output):
    _x, indices = inputs
    ctx.save_for_backward(indices)


def _permute_backward(ctx, grad_out):
    (indices,) = ctx.saved_tensors
    return torch.ops.nanoplm_moe.unpermute(grad_out.contiguous(), indices), None


def _unpermute_backward(ctx, grad_out):
    (indices,) = ctx.saved_tensors
    return torch.ops.nanoplm_moe.permute(grad_out.contiguous(), indices), None


torch.library.register_autograd(
    "nanoplm_moe::permute",
    _permute_backward,
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "nanoplm_moe::unpermute",
    _unpermute_backward,
    setup_context=_setup_context,
)


def moe_permute(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.ops.nanoplm_moe.permute(x, indices)


def moe_unpermute(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.ops.nanoplm_moe.unpermute(x, indices)
