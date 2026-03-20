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


# ── Fused dispatch + combine ops ─────────────────────────────────────────────

_lib.define("scatter_dispatch(Tensor x, Tensor token_idx) -> Tensor")
_lib.define("scatter_add_bwd(Tensor grad_sorted, Tensor token_idx, int T) -> Tensor")
_lib.define(
    "gather_combine(Tensor expert_out_sorted, Tensor inv_map, "
    "Tensor weights, float scale) -> Tensor"
)
_lib.define(
    "gather_combine_bwd_eo(Tensor grad_out, Tensor inv_map, "
    "Tensor weights, float scale, int M, int eo_dtype) -> Tensor"
)
_lib.define(
    "gather_combine_bwd_w(Tensor grad_out, Tensor expert_out_sorted, "
    "Tensor inv_map, float scale) -> Tensor"
)


# ── build_inverse_map helper ─────────────────────────────────────────────────


def build_inverse_map(
    token_idx: torch.Tensor,
    slot_idx: torch.Tensor,
    T: int,
    top_k: int,
) -> torch.Tensor:
    """Build inv_map[token_idx[i], slot_idx[i]] = i for all i."""
    inv_map = torch.empty(T, top_k, dtype=torch.int32, device=token_idx.device)
    positions = torch.arange(
        token_idx.shape[0], dtype=torch.int32, device=token_idx.device,
    )
    inv_map[token_idx.long(), slot_idx.long()] = positions
    return inv_map


# ── scatter_dispatch: fuses repeat_interleave + permute ──────────────────────


@torch.library.register_fake("nanoplm_moe::scatter_dispatch")
def _scatter_dispatch_fake(x, token_idx):
    return torch.empty(
        token_idx.shape[0], x.shape[1], dtype=x.dtype, device=x.device,
    )


@torch.library.impl(_lib, "scatter_dispatch", "CUDA")
def _scatter_dispatch_cuda(x, token_idx):
    x = x.contiguous()
    token_idx = token_idx.contiguous()
    M = token_idx.shape[0]
    D = x.shape[1]
    y = torch.empty(M, D, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(D, 128))
    _moe_permute_kernel[grid](
        x, token_idx, y, M, D,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_D=128, num_warps=4, num_stages=2,
    )
    return y


@torch.library.impl(_lib, "scatter_dispatch", "CPU")
def _scatter_dispatch_cpu(x, token_idx):
    return x[token_idx.long()]


# ── scatter_add_bwd: Triton atomic scatter-add for scatter_dispatch backward ─


@triton.jit
def _moe_scatter_add_kernel(
    grad_sorted_ptr,
    token_idx_ptr,
    grad_x_ptr,
    M,
    D,
    stride_gs_m,
    stride_gs_d,
    stride_gx_m,
    stride_gx_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter-add: grad_x[token_idx[i]] += grad_sorted[i] via tl.atomic_add."""
    pid_m = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)

    dst_rows = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0).to(tl.int64)
    gs_ptrs = grad_sorted_ptr + offs_m[:, None] * stride_gs_m + offs_d[None, :] * stride_gs_d
    gx_ptrs = grad_x_ptr + dst_rows[:, None] * stride_gx_m + offs_d[None, :] * stride_gx_d

    vals = tl.load(gs_ptrs, mask=mask, other=0.0)
    tl.atomic_add(gx_ptrs, vals, mask=mask)


@torch.library.register_fake("nanoplm_moe::scatter_add_bwd")
def _scatter_add_bwd_fake(grad_sorted, token_idx, T):
    return torch.empty(T, grad_sorted.shape[1], dtype=grad_sorted.dtype, device=grad_sorted.device)


@torch.library.impl(_lib, "scatter_add_bwd", "CUDA")
def _scatter_add_bwd_cuda(grad_sorted, token_idx, T):
    grad_sorted = grad_sorted.contiguous()
    token_idx = token_idx.contiguous()
    M, D = grad_sorted.shape
    grad_x = torch.zeros(T, D, dtype=grad_sorted.dtype, device=grad_sorted.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(D, 128))
    _moe_scatter_add_kernel[grid](
        grad_sorted, token_idx, grad_x,
        M, D,
        grad_sorted.stride(0), grad_sorted.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        BLOCK_M=64, BLOCK_D=128,
        num_warps=4, num_stages=2,
    )
    return grad_x


@torch.library.impl(_lib, "scatter_add_bwd", "CPU")
def _scatter_add_bwd_cpu(grad_sorted, token_idx, T):
    D = grad_sorted.shape[1]
    grad_x = torch.zeros(T, D, dtype=grad_sorted.dtype, device=grad_sorted.device)
    grad_x.index_add_(0, token_idx.long(), grad_sorted)
    return grad_x


def _scatter_dispatch_setup_context(ctx, inputs, output):
    x, token_idx = inputs
    ctx.save_for_backward(token_idx)
    ctx.num_tokens = x.shape[0]


def _scatter_dispatch_backward(ctx, grad_sorted):
    (token_idx,) = ctx.saved_tensors
    return torch.ops.nanoplm_moe.scatter_add_bwd(
        grad_sorted, token_idx, ctx.num_tokens,
    ), None


torch.library.register_autograd(
    "nanoplm_moe::scatter_dispatch",
    _scatter_dispatch_backward,
    setup_context=_scatter_dispatch_setup_context,
)


def moe_scatter_dispatch(x: torch.Tensor, token_idx: torch.Tensor) -> torch.Tensor:
    return torch.ops.nanoplm_moe.scatter_dispatch(x, token_idx)


# ── gather_combine: fuses unpermute + mul(weights) + sum(top_k) ──────────────


@triton.jit
def _moe_gather_combine_fwd_kernel(
    eo_ptr, inv_ptr, w_ptr, out_ptr,
    scale,
    C,
    stride_eo_m, stride_eo_d,
    stride_inv_t, stride_inv_k,
    stride_w_t, stride_w_k,
    stride_out_t, stride_out_d,
    BLOCK_D: tl.constexpr,
    TOP_K: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    t = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < C

    acc = tl.zeros([BLOCK_D], dtype=tl.float32 if FP32_ACCUM else tl.float64)

    for k in range(TOP_K):
        pos = tl.load(inv_ptr + t * stride_inv_t + k * stride_inv_k).to(tl.int64)
        w = tl.load(w_ptr + t * stride_w_t + k * stride_w_k)
        row = tl.load(
            eo_ptr + pos * stride_eo_m + offs_d * stride_eo_d,
            mask=mask_d, other=0.0,
        )
        if FP32_ACCUM:
            row = row.to(tl.float32)
        acc += w * row

    acc = acc * scale
    tl.store(out_ptr + t * stride_out_t + offs_d * stride_out_d, acc, mask=mask_d)


@triton.jit
def _moe_gather_combine_bwd_eo_kernel(
    go_ptr, inv_ptr, w_ptr, ge_ptr,
    scale,
    C,
    stride_go_t, stride_go_d,
    stride_inv_t, stride_inv_k,
    stride_w_t, stride_w_k,
    stride_ge_m, stride_ge_d,
    BLOCK_D: tl.constexpr,
    TOP_K: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """Backward: grad_expert_sorted[inv_map[t,k]] = grad_out[t] * w[t,k] * scale."""
    t = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < C

    go = tl.load(go_ptr + t * stride_go_t + offs_d * stride_go_d, mask=mask_d, other=0.0)

    for k in range(TOP_K):
        pos = tl.load(inv_ptr + t * stride_inv_t + k * stride_inv_k).to(tl.int64)
        w = tl.load(w_ptr + t * stride_w_t + k * stride_w_k)
        val = go * w * scale
        # tl.store handles implicit fp32→bf16 cast when ge_ptr is bf16-typed
        tl.store(
            ge_ptr + pos * stride_ge_m + offs_d * stride_ge_d, val, mask=mask_d,
        )


@triton.jit
def _moe_gather_combine_bwd_w_kernel(
    go_ptr, eo_ptr, inv_ptr, gw_ptr,
    scale,
    C,
    stride_go_t, stride_go_d,
    stride_eo_m, stride_eo_d,
    stride_inv_t, stride_inv_k,
    stride_gw_t, stride_gw_k,
    BLOCK_D: tl.constexpr,
    TOP_K: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """Backward: grad_weights[t,k] = dot(grad_out[t], expert_out[inv_map[t,k]]) * scale."""
    t = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < C

    go = tl.load(go_ptr + t * stride_go_t + offs_d * stride_go_d, mask=mask_d, other=0.0)

    for k in range(TOP_K):
        pos = tl.load(inv_ptr + t * stride_inv_t + k * stride_inv_k).to(tl.int64)
        row = tl.load(
            eo_ptr + pos * stride_eo_m + offs_d * stride_eo_d,
            mask=mask_d, other=0.0,
        )
        if FP32_ACCUM:
            row = row.to(tl.float32)
        dot = tl.sum(go * row, axis=0)
        gw = dot * scale
        tl.store(gw_ptr + t * stride_gw_t + k * stride_gw_k, gw)


def _gather_combine_eager(expert_out_sorted, inv_map, weights, scale):
    """Composite eager implementation — used for CPU and C > 1024 fallback."""
    T, top_k = inv_map.shape
    C = expert_out_sorted.shape[1]
    flat_idx = inv_map.long().reshape(-1)
    rows = expert_out_sorted[flat_idx].view(T, top_k, C)
    if expert_out_sorted.dtype in (torch.bfloat16, torch.float16):
        rows = rows.float()
    return (rows * weights.unsqueeze(-1) * scale).sum(dim=1)


def _gather_combine_backward_eager(
    grad_out, expert_out_sorted, inv_map, weights, scale, eo_dtype,
):
    """Composite eager backward — used for CPU and C > 1024 fallback."""
    T, top_k = inv_map.shape
    C = expert_out_sorted.shape[1]
    M = expert_out_sorted.shape[0]
    fp32_accum = eo_dtype in (torch.bfloat16, torch.float16)

    grad_eo = torch.empty(M, C, dtype=eo_dtype, device=grad_out.device)
    for k in range(top_k):
        positions = inv_map[:, k].long()
        val = grad_out * weights[:, k : k + 1] * scale
        if fp32_accum:
            val = val.to(eo_dtype)
        grad_eo[positions] = val

    grad_w = torch.empty(T, top_k, dtype=weights.dtype, device=grad_out.device)
    for k in range(top_k):
        positions = inv_map[:, k].long()
        eo_k = expert_out_sorted[positions]
        if fp32_accum:
            eo_k = eo_k.float()
        grad_w[:, k] = (grad_out * eo_k).sum(dim=-1) * scale

    return grad_eo, None, grad_w, None


@torch.library.register_fake("nanoplm_moe::gather_combine")
def _gather_combine_fake(expert_out_sorted, inv_map, weights, scale):
    T = inv_map.shape[0]
    C = expert_out_sorted.shape[1]
    out_dtype = (
        torch.float32
        if expert_out_sorted.dtype != torch.float64
        else torch.float64
    )
    return torch.empty(T, C, dtype=out_dtype, device=expert_out_sorted.device)


def _launch_gather_combine_triton(expert_out_sorted, inv_map, weights, scale):
    T, top_k = inv_map.shape
    C = expert_out_sorted.shape[1]
    fp32_accum = expert_out_sorted.dtype != torch.float64
    out_dtype = torch.float32 if fp32_accum else torch.float64

    output = torch.empty(T, C, dtype=out_dtype, device=expert_out_sorted.device)
    BLOCK_D = triton.next_power_of_2(C)

    _moe_gather_combine_fwd_kernel[(T,)](
        expert_out_sorted, inv_map, weights, output,
        scale, C,
        expert_out_sorted.stride(0), expert_out_sorted.stride(1),
        inv_map.stride(0), inv_map.stride(1),
        weights.stride(0), weights.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_D=BLOCK_D, TOP_K=top_k, FP32_ACCUM=fp32_accum,
        num_warps=4, num_stages=1,
    )
    return output


@torch.library.impl(_lib, "gather_combine", "CUDA")
def _gather_combine_cuda(expert_out_sorted, inv_map, weights, scale):
    C = expert_out_sorted.shape[1]
    if C > 1024:
        return _gather_combine_eager(expert_out_sorted, inv_map, weights, scale)
    return _launch_gather_combine_triton(expert_out_sorted, inv_map, weights, scale)


@torch.library.impl(_lib, "gather_combine", "CPU")
def _gather_combine_cpu(expert_out_sorted, inv_map, weights, scale):
    return _gather_combine_eager(expert_out_sorted, inv_map, weights, scale)


# ── gather_combine backward ops (registered so torch.compile can trace) ──────

# Encode dtype as int so it can pass through the op schema.
_DTYPE_TO_INT = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2, torch.float64: 3}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_TO_INT.items()}


@torch.library.register_fake("nanoplm_moe::gather_combine_bwd_eo")
def _gather_combine_bwd_eo_fake(grad_out, inv_map, weights, scale, M, eo_dtype):
    C = grad_out.shape[1]
    return torch.empty(M, C, dtype=_INT_TO_DTYPE[eo_dtype], device=grad_out.device)


@torch.library.impl(_lib, "gather_combine_bwd_eo", "CUDA")
def _gather_combine_bwd_eo_cuda(grad_out, inv_map, weights, scale, M, eo_dtype):
    T, top_k = inv_map.shape
    C = grad_out.shape[1]
    out_dtype = _INT_TO_DTYPE[eo_dtype]
    fp32_accum = out_dtype != torch.float64

    if C > 1024:
        # Eager fallback
        grad_eo = torch.empty(M, C, dtype=out_dtype, device=grad_out.device)
        for k in range(top_k):
            positions = inv_map[:, k].long()
            val = grad_out * weights[:, k : k + 1] * scale
            if fp32_accum:
                val = val.to(out_dtype)
            grad_eo[positions] = val
        return grad_eo

    BLOCK_D = triton.next_power_of_2(C)
    grad_eo = torch.empty(M, C, dtype=out_dtype, device=grad_out.device)
    _moe_gather_combine_bwd_eo_kernel[(T,)](
        grad_out, inv_map, weights, grad_eo,
        scale, C,
        grad_out.stride(0), grad_out.stride(1),
        inv_map.stride(0), inv_map.stride(1),
        weights.stride(0), weights.stride(1),
        grad_eo.stride(0), grad_eo.stride(1),
        BLOCK_D=BLOCK_D, TOP_K=top_k, FP32_ACCUM=fp32_accum,
        num_warps=4, num_stages=1,
    )
    return grad_eo


@torch.library.impl(_lib, "gather_combine_bwd_eo", "CPU")
def _gather_combine_bwd_eo_cpu(grad_out, inv_map, weights, scale, M, eo_dtype):
    T, top_k = inv_map.shape
    C = grad_out.shape[1]
    out_dtype = _INT_TO_DTYPE[eo_dtype]
    fp32_accum = out_dtype in (torch.bfloat16, torch.float16)
    grad_eo = torch.empty(M, C, dtype=out_dtype, device=grad_out.device)
    for k in range(top_k):
        positions = inv_map[:, k].long()
        val = grad_out * weights[:, k : k + 1] * scale
        if fp32_accum:
            val = val.to(out_dtype)
        grad_eo[positions] = val
    return grad_eo


@torch.library.register_fake("nanoplm_moe::gather_combine_bwd_w")
def _gather_combine_bwd_w_fake(grad_out, expert_out_sorted, inv_map, scale):
    T, top_k = inv_map.shape
    return torch.empty(T, top_k, dtype=grad_out.dtype, device=grad_out.device)


@torch.library.impl(_lib, "gather_combine_bwd_w", "CUDA")
def _gather_combine_bwd_w_cuda(grad_out, expert_out_sorted, inv_map, scale):
    T, top_k = inv_map.shape
    C = expert_out_sorted.shape[1]
    fp32_accum = expert_out_sorted.dtype != torch.float64

    if C > 1024:
        # Eager fallback
        grad_w = torch.empty(T, top_k, dtype=grad_out.dtype, device=grad_out.device)
        for k in range(top_k):
            positions = inv_map[:, k].long()
            eo_k = expert_out_sorted[positions]
            if fp32_accum:
                eo_k = eo_k.float()
            grad_w[:, k] = (grad_out * eo_k).sum(dim=-1) * scale
        return grad_w

    BLOCK_D = triton.next_power_of_2(C)
    grad_w = torch.empty(T, top_k, dtype=grad_out.dtype, device=grad_out.device)
    _moe_gather_combine_bwd_w_kernel[(T,)](
        grad_out, expert_out_sorted, inv_map, grad_w,
        scale, C,
        grad_out.stride(0), grad_out.stride(1),
        expert_out_sorted.stride(0), expert_out_sorted.stride(1),
        inv_map.stride(0), inv_map.stride(1),
        grad_w.stride(0), grad_w.stride(1),
        BLOCK_D=BLOCK_D, TOP_K=top_k, FP32_ACCUM=fp32_accum,
        num_warps=4, num_stages=1,
    )
    return grad_w


@torch.library.impl(_lib, "gather_combine_bwd_w", "CPU")
def _gather_combine_bwd_w_cpu(grad_out, expert_out_sorted, inv_map, scale):
    T, top_k = inv_map.shape
    fp32_accum = expert_out_sorted.dtype in (torch.bfloat16, torch.float16)
    grad_w = torch.empty(T, top_k, dtype=grad_out.dtype, device=grad_out.device)
    for k in range(top_k):
        positions = inv_map[:, k].long()
        eo_k = expert_out_sorted[positions]
        if fp32_accum:
            eo_k = eo_k.float()
        grad_w[:, k] = (grad_out * eo_k).sum(dim=-1) * scale
    return grad_w


# ── gather_combine autograd ──────────────────────────────────────────────────


def _gather_combine_setup_context(ctx, inputs, output):
    expert_out_sorted, inv_map, weights, scale = inputs
    ctx.save_for_backward(expert_out_sorted, inv_map, weights)
    ctx.scale = float(scale)
    ctx.eo_dtype = expert_out_sorted.dtype


def _gather_combine_backward(ctx, grad_out):
    expert_out_sorted, inv_map, weights = ctx.saved_tensors
    scale = ctx.scale
    M = expert_out_sorted.shape[0]
    eo_dtype_int = _DTYPE_TO_INT[ctx.eo_dtype]
    grad_out = grad_out.contiguous()

    grad_eo = torch.ops.nanoplm_moe.gather_combine_bwd_eo(
        grad_out, inv_map, weights, scale, M, eo_dtype_int,
    )
    grad_w = torch.ops.nanoplm_moe.gather_combine_bwd_w(
        grad_out, expert_out_sorted, inv_map, scale,
    )
    return grad_eo, None, grad_w, None


torch.library.register_autograd(
    "nanoplm_moe::gather_combine",
    _gather_combine_backward,
    setup_context=_gather_combine_setup_context,
)


def moe_gather_combine(
    expert_out_sorted: torch.Tensor,
    inv_map: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.ops.nanoplm_moe.gather_combine(
        expert_out_sorted, inv_map, weights, scale,
    )
