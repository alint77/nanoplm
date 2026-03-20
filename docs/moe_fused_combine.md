# MoE Fused Dispatch + Combine Kernels

## Problem

The MoE forward path materializes 5 intermediate `(T*top_k, C)` tensors where only 2 are necessary.
This causes **~255ms of pointwise overhead** (11.3% of step) — more than the permute kernels and dispatch combined.

### Current data flow

```
Forward:
  L250: x_expanded = x.repeat_interleave(top_k, dim=0)             # WRITE (T*2, C) = 200MB
  L265: x_sorted   = moe_permute(x_expanded, sorted_idx)           # READ 200MB + WRITE 200MB
        ──── grouped_gemm (Wi → SwiGLU → Wo) ────
  L281: expert_out  = moe_unpermute(expert_out_sorted, sorted_idx)  # READ+WRITE 200MB each
  L283: temp        = expert_out * weights.unsqueeze(-1) * scale    # READ 200MB + WRITE 200MB
  L285: output      = temp.sum(dim=1)                               # READ 200MB + WRITE 100MB

Backward (autograd generates):
  grad_out → expand to (T, top_k, H)                                # WRITE 200MB
  grad_expert_out = grad_expanded * weights                          # READ+WRITE 200MB each
  grad_weights    = (grad_expanded * expert_out).sum(-1)             # READ 200+200MB
  grad_expert_sorted = moe_permute(grad_expert_out, ...)             # READ+WRITE 200MB each
                       ──── expert MLP backward (grouped_gemm + SwiGLU) ────
  grad_x_expanded = moe_unpermute(grad_x_sorted, ...)               # READ+WRITE 200MB each
  grad_x = fold grad_x_expanded back to (T, C)                      # READ 200MB + WRITE 100MB
```

Total IO for dispatch+combine: ~3.0 GB forward, ~4.0 GB backward = **7.0 GB**.
Actual time: **~255ms** due to 8-9 separate kernel launches per layer with cache-cold intermediates.

### Root causes

1. **`repeat_interleave` + `permute`**: two full `(T*top_k, C)` writes. The expanded tensor is immediately consumed by permute — never needs to exist.
2. **`unpermute` + `mul(weights)` + `sum(top_k)`**: three kernels on `(T*top_k, C)`. The unpermuted tensor is immediately consumed by weighted sum — also never needs to exist.
3. **Backward mirrors the same pattern** — autograd generates separate expand, mul, sum, permute, unpermute, fold kernels.

## Solution: two fused custom ops

### Shared metadata: precompute token_idx and slot_idx

Before either op, precompute once during dispatch:

```python
token_idx = (sorted_idx // top_k).to(torch.int32)   # (T*top_k,) — which original token
slot_idx  = (sorted_idx % top_k).to(torch.int32)     # (T*top_k,) — which top_k slot (0 or 1)
```

Use **int32** — T*top_k = 131072 fits comfortably, and halving metadata bandwidth reduces register pressure in the hot kernels. These tensors are reused by both dispatch and combine, avoiding repeated div/mod. Cost: ~0.5MB each.

**No buffer reuse** for these tensors. Unlike the existing dispatch workspaces (`_dispatch_sort_values` etc.), which are consumed and discarded within the same forward call, `token_idx`/`slot_idx`/`inv_map` are saved for backward via `ctx.save_for_backward`. In-place buffer reuse would corrupt them during gradient accumulation (multiple forwards before `.backward()`) or checkpointing re-entry. At ~1.5MB total, the CUDA caching allocator makes fresh `torch.empty` calls of the same size essentially free — the complexity of safe reuse isn't worth it.

### Op 1: `moe_scatter_dispatch` — fuses repeat_interleave + permute

```python
# Current (2 kernels, 2 writes of (T*topk, C)):
x_expanded = x.repeat_interleave(top_k, dim=0)
x_sorted = moe_permute(x_expanded, sorted_idx)

# Fused (1 kernel, 1 write of (T*topk, C)):
x_sorted = moe_scatter_dispatch(x, token_idx)
# Impl: x_sorted[i] = x[token_idx[i]]
```

- Forward IO: read `(T, C)` + `token_idx` + write `(T*topk, C)` = **300 MB** (was 600 MB)
- Backward: scatter-add `grad_sorted` back to `(T, C)` using `token_idx`

### Op 2: `moe_gather_combine` — fuses unpermute + mul(weights) + sum(top_k)

```python
# Current (3+ kernels, 3 reads of (T*topk, C)):
expert_out = moe_unpermute(expert_out_sorted, sorted_idx)
expert_out = expert_out.view(T, top_k, H)
expert_out = (expert_out * weights.unsqueeze(-1) * scale).sum(dim=1)

# Fused (1 kernel, 1 read of (T*topk, C)):
output = moe_gather_combine(expert_out_sorted, inv_map, weights, scale)
# Impl: out[t] = sum_k weights[t,k] * expert_out_sorted[inv_map[t,k]] * scale
```

- Forward IO: read `(T*topk, C)` + `(T, topk)` weights + `inv_map` + write `(T, C)` = **302 MB** (was 800+ MB)
- Backward: `grad_expert_sorted` + `grad_weights` (see Phase 2b below)

### Dtype contract

Router weights are **fp32** (moe.py L69: `raw_logits.float().sigmoid()`). The current combine expression `bf16_expert_out * fp32_weights` promotes to fp32 via PyTorch casting rules.

The fused combine kernel must:
1. Load expert_out in native dtype (bf16)
2. Cast to fp32 before multiply with weights
3. Accumulate the top_k sum in fp32
4. Store output as **fp32** (matching current semantics — the residual add with shared_out handles final dtype)

Same pattern as `FP32_ACCUM` in the Canon kernels: a `tl.constexpr` flag controlling conditional casts, with fp64 passthrough for gradcheck.

## IO savings

| Path | Current | Fused | Savings |
|---|---|---|---|
| Dispatch fwd | 600 MB | 300 MB | 300 MB |
| Combine fwd | 800 MB | 302 MB | 498 MB |
| Dispatch bwd | 600 MB | 302 MB | 298 MB |
| Combine bwd | 1000 MB | 504 MB | 496 MB |
| **Total** | **3000 MB** | **1408 MB** | **1592 MB** |

At 1792 GB/s theoretical savings = 0.89 ms/layer. Over 88 layer-calls: ~78 ms IO floor.
With kernel launch elimination and improved cache locality: **estimate 80-120 ms savings** (255ms → ~135-175ms).

## Implementation plan

### Phase 1a: `moe_scatter_dispatch` forward

Triton kernel + `torch.library` op. Safest first win.

- **Kernel**: 2D grid `(cdiv(T*topk, BLOCK_M), cdiv(C, BLOCK_D))`. Each program loads `token_idx[row]` (int32) to find the source token, copies the C-slice. Identical structure to existing `_moe_permute_kernel` but reads from `x` via `token_idx` indirection instead of `sorted_idx`.
- **CPU ref**: `return x[token_idx.long()]` (one-liner).
- **register_fake**: `return torch.empty(token_idx.shape[0], x.shape[1], ...)`.

### Phase 1b: `moe_scatter_dispatch` backward

Backward = scatter-add: accumulate `grad_sorted[i]` into `grad_x[token_idx[i]]`.

- **v1 (simple composite baseline)**: `grad_x = torch.zeros(T, C); grad_x.index_add_(0, token_idx.long(), grad_sorted)`. Note: CUDA `index_add_` is **not guaranteed deterministic** unless `torch.use_deterministic_algorithms(True)` is set. This is acceptable for training — we don't enforce deterministic mode in this codebase.
- **v2 (faster, later)**: Triton kernel with `tl.atomic_add`. With top_k=2, contention is exactly 2-way per token — acceptable on Blackwell. Can optimize further with a sort-based deterministic approach if needed.

Start with v1. It's correct, simple, and lets us validate the forward fusion immediately.

### Phase 2a: `moe_gather_combine` forward

Triton kernel. This is the bigger win.

- **Kernel design (specialized fast path for C <= 1024)**: 1D grid over T tokens, BLOCK_D covers full C (next_power_of_2). This is a specialization for the current hidden_size=768, not a generic design. The **CUDA impl** checks `C <= 1024` at dispatch time: if true, launch the Triton kernel; otherwise, fall back to the composite eager path (unpermute → mul → sum). This keeps the specialization contained — the op signature, CPU ref, `register_fake`, and tests are all shape-agnostic. Same guard applies to the backward kernels.
  - Requires an **inverse map** `inv_map: (T, top_k)` int32 tensor that maps each `(token, k)` to its position in the sorted array. Built once: `inv_map[token_idx[i], slot_idx[i]] = i`.
  - Each program loads `top_k` expert output rows from their sorted positions, multiplies by fp32 weights, accumulates in fp32, writes output.
  - Register pressure: `top_k * BLOCK_D` fp32 values + pointers + masks + temporaries. Nominal data is ~8KB but real occupancy depends on Triton's register allocation — **treat as measure-after-prototype, not proven**.
- **CPU ref**: the current 3-line eager version (unpermute → mul → sum).
- **Dtype**: load expert_out as bf16, cast to fp32, multiply by fp32 weights, accumulate fp32, store fp32. Use `FP32_ACCUM: tl.constexpr` for fp64 gradcheck passthrough.

### Phase 2b: `moe_gather_combine` backward

Two outputs: `grad_expert_sorted (T*topk, C)` and `grad_weights (T, topk)`.

**Backward dtype contract**: The forward takes bf16 `expert_out_sorted` and fp32 `weights`, returns fp32 output. The backward must respect each input's dtype:
- `grad_expert_sorted` → **bf16** (same dtype as `expert_out_sorted`). This feeds into grouped_gemm backward which expects bf16 inputs. Returning fp32 here would silently break the GEMM backward.
- `grad_weights` → **fp32** (same dtype as `weights`).
- Internally, compute in fp32 then cast `grad_expert_sorted` back to the input dtype before returning.

Split into **two kernels** rather than forcing one:

**Kernel 1: `grad_expert_sorted`** — scatter grad_out x weight to sorted expert positions.
- 1D grid over T tokens, BLOCK_D = next_power_of_2(C). For each token, load `grad_out[t]` (fp32) and `weights[t, k]` (fp32) for each k, compute `grad_out[t] * weights[t,k] * scale` in fp32, **cast to bf16**, write to `grad_expert_sorted` at the corresponding sorted position via `inv_map[t, k]`.
- No atomics needed — each sorted position is written by exactly one token.
- This is just the reverse of the forward gather.

**Kernel 2: `grad_weights`** — dot product of grad_out and expert_out per slot.
- 1D grid over T tokens. For each token and each k, load `expert_out_sorted[pos]` (bf16, cast to fp32), compute `dot(grad_out[t], expert_out_fp32) * scale` in fp32 where pos = `inv_map[t, k]`.
- Output: `(T, top_k)` **fp32** tensor.
- Full-row approach (C <= 1024 specialization): load `grad_out[t]` (BLOCK_D) once, load `expert_out_sorted[pos]` (BLOCK_D) for each k, do `tl.sum(a * b)`.
- This bakes in C <= 1024. For larger C, would need a tiled approach with partial sums.

**Optional fusion**: kernels 1 and 2 iterate over the same tokens and load the same data (`grad_out[t]`, `expert_out_sorted[pos]`). A fused version loads `grad_out[t]` once and for each k loads `expert_out_sorted[pos]`, then writes both `grad_expert_sorted[pos]` (bf16) and `grad_weights[t, k]` (fp32). Attempt this only after measuring occupancy on the separate kernels.

### Phase 3: Wiring + tests

- Register all ops via `torch.library` (DEF, register_fake, impl CUDA, impl CPU, register_autograd).
- Build `inv_map` once in `MoELayer.forward`, right after computing `token_idx` and `slot_idx`.
- Update `MoELayer.forward`:
  ```python
  # precompute metadata (reuse across dispatch + combine)
  token_idx = (sorted_idx // self.top_k).to(torch.int32)
  slot_idx = (sorted_idx % self.top_k).to(torch.int32)
  inv_map = build_inverse_map(token_idx, slot_idx, T, self.top_k)  # (T, top_k) int32

  # dispatch (was: repeat_interleave + permute)
  x_sorted = moe_scatter_dispatch(x, token_idx)

  # ... grouped_gemm + SwiGLU (unchanged) ...

  # combine (was: unpermute + view + mul + sum)
  expert_out = moe_gather_combine(
      expert_out_sorted, inv_map, weights,
      self.routed_scaling_factor,
  )
  ```
- **Tests** (mirror `test_canon_correctness.py` pattern):
  - Forward match vs eager reference (bf16, fp32)
  - Backward match vs eager reference (bf16, fp32)
  - fp64 gradcheck for both ops
  - Integration test: full MoE layer forward+backward matches eager
  - **CPU reference fp32 contract**: the CPU ref for `moe_gather_combine` must replicate the fp32 promotion — cast bf16 expert_out to fp32 before multiply+sum and return fp32 output. Without this, CPU-vs-CUDA comparison tests will fail on dtype mismatch or accumulation error.

## Staging order

1. `token_idx` / `slot_idx` plumbing + `moe_scatter_dispatch` forward + backward (composite baseline).
2. `inv_map` builder + `moe_gather_combine` forward.
3. `moe_gather_combine` backward as two separate kernels.
4. Only after a new profiler trace: consider fusing the two combine-backward kernels.

## What this does NOT change

- Grouped_gemm calls (already 78% roofline efficiency)
- SwiGLU activation (must materialize intermediate)
- Router computation (small)
- Sort/dispatch metadata (7ms, negligible)

## Profiler results

### Hardware & config

- RTX 5090: 1792 GB/s memory BW, 209.6 TFLOP/s bf16 tensor core
- T=65536, C=768, top_k=2, M=T*top_k=131072, E=11, inter=512, bf16
- 12 layers (11 MoE + 1 dense), FSDP, activation checkpointing (attn mode)
- Roofline crossover: 117 FLOP/byte for bf16 — all dispatch+combine kernels are BW-bound

### Three-run comparison

| Run | Description | GPU kernel time | Delta vs baseline |
|---|---|---|---|
| run-20031213 | Baseline (unfused) | 2748.7 ms | — |
| run-20031333 | V1: fused ops + `index_add_` bwd | 2727.7 ms | **-21.0 ms (-0.8%)** |
| run-20031346 | V2: fused ops + Triton `scatter_add` bwd | 2677.9 ms | **-70.8 ms (-2.6%)** |

### Dispatch+combine kernel budget

| Run | Dispatch+combine time | Delta |
|---|---|---|
| Baseline | 198.8 ms | — |
| V1 (fused + index_add_) | 151.4 ms | **-47.4 ms (-23.8%)** |
| V2 (fused + scatter_add) | 113.9 ms | **-84.9 ms (-42.7%)** |

### Baseline kernels removed (198.8ms total)

| Kernel | Time | Count | Avg | Replaced by |
|---|---|---|---|---|
| `_moe_permute_kernel` (bwd half) | 30.0 ms | 92 | 326 us | V1: `index_add_` → V2: `scatter_add` |
| `_moe_unpermute_kernel` | 46.1 ms | 180 | 256 us | `gather_combine_fwd` + `gather_combine_bwd_eo` |
| `fused_add_mul_sum_unsqueeze_view` | 27.1 ms | 88 | 308 us | `gather_combine_fwd` |
| `fused_expand_mul_sum_unsqueeze_view` | 24.2 ms | 92 | 263 us | `gather_combine_bwd_w` |
| `fused__to_copy_expand_mul_permute_unsqueeze_view` | 21.2 ms | 92 | 230 us | `gather_combine_bwd_eo` |
| `fused_clone_expand_unsqueeze` (repeat_interleave) | 16.3 ms | 88 | 185 us | `scatter_dispatch` fwd |
| `fused_sum_view` | 15.0 ms | 93 | 161 us | `gather_combine_fwd` |
| `fused_zeros` (bwd) | 4.6 ms | 92 | 50 us | eliminated (Triton scatter_add zeros internally) |

### V2 fused kernels (113.9ms total) — roofline analysis

All kernels are bandwidth-bound. IO calculated for T=65536, C=768, top_k=2, bf16/fp32.

| Kernel | Time | Count | Avg | IO (MB) | BW floor (us) | BW eff. |
|---|---|---|---|---|---|---|
| `_moe_scatter_add_kernel` | 24.3 ms | 92 | 265 us | 403 | 225 | **85%** |
| `_moe_gather_combine_bwd_w_kernel` | 24.1 ms | 92 | 262 us | 404 | 225 | **86%** |
| `_moe_gather_combine_bwd_eo_kernel` | 24.6 ms | 92 | 268 us | 404 | 225 | **84%** |
| `_moe_gather_combine_fwd_kernel` | 22.0 ms | 88 | 250 us | 404 | 225 | **90%** |
| `_moe_permute_kernel` (fwd) | 18.9 ms | 88 | 214 us | 303 | 169 | **79%** |

IO breakdown:
- `scatter_dispatch` fwd (`_moe_permute_kernel`): read x (T,C) bf16 100.7MB + token_idx 0.5MB + write x_sorted (M,C) bf16 201.3MB = 303MB
- `scatter_add` bwd: read grad_sorted (M,C) bf16 201.3MB + token_idx 0.5MB + RMW grad_x (T,C) bf16 201.4MB = 403MB
- `gather_combine` fwd: read expert_out (M,C) bf16 201.3MB + inv_map/weights 1.0MB + write output (T,C) fp32 201.3MB = 404MB
- `gather_combine_bwd_eo`: read grad_out (T,C) fp32 201.3MB + inv_map/weights 1.0MB + write grad_eo (M,C) bf16 201.3MB = 404MB
- `gather_combine_bwd_w`: read grad_out (T,C) fp32 201.3MB + expert_out (M,C) bf16 201.3MB + inv_map 0.5MB + write grad_w (T,top_k) fp32 0.5MB = 404MB

### V1 → V2: `index_add_` replacement

The CUDA `indexFuncLargeIndex` implementation of `index_add_` achieved only **28% bandwidth efficiency** (493 GB/s vs 1792 GB/s peak), at 609us avg per call (56.0ms total). It was the single largest bottleneck in the fused dispatch+combine path.

Replaced with `_moe_scatter_add_kernel` using `tl.atomic_add`. With top_k=2, contention is exactly 2-way per token — trivial for Blackwell. Result: 265us avg (**85% BW efficiency**), saving 31.7ms total.

The `fused_zeros` kernel (4.3ms in V1 for `torch.zeros` before `index_add_`) was also eliminated — the Triton kernel initializes its output buffer via a separate zero-fill launch that gets fused away by torch.compile.

### torch.compile lessons

1. **Backward ops must be registered custom ops.** The initial implementation broke `torch.compile` because the backward called Triton kernels directly. Dynamo can't trace through FakeTensors to raw kernel launches. Fixed by registering `gather_combine_bwd_eo`, `gather_combine_bwd_w`, and `scatter_add_bwd` as separate `torch.library` ops with DEF + `register_fake` + CUDA impl + CPU impl.

2. **Inductor cache invalidation.** After replacing `index_add_` with the Triton scatter_add kernel, the trace still showed `indexFuncLargeIndex`. The massive inductor cache in `/tmp/torchinductor_*` had compiled the old backward graph. Clearing the cache (`rm -rf /tmp/torchinductor_*`) resolved it — the next run correctly picked up the new kernel.

## Risk assessment

| Component | Confidence | Notes |
|---|---|---|
| `moe_scatter_dispatch` fwd | High | Trivial gather kernel, same pattern as existing permute |
| `moe_scatter_dispatch` bwd | High | `index_add_` composite for v1 (nondeterministic on CUDA, acceptable for training) |
| `moe_gather_combine` fwd | High | Full-row gather with explicit dtype contract. C <= 1024 specialization. |
| `moe_gather_combine` bwd (grad_expert) | High | Reverse of forward gather, no atomics, no reductions |
| `moe_gather_combine` bwd (grad_weights) | Medium | Full-row reduction over D. Clean for C <= 1024 but bakes in that assumption. Occupancy needs measurement. |
