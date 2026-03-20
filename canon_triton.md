# Triton Canon Varlen Depthwise Convolution

## Problem

Canon layers perform depthwise 1D convolution with sequence-boundary masking on
variable-length packed sequences. The original pure-PyTorch implementation
(`_varlen_canon_inner` in `modeling.py`) uses `torch.roll` + elementwise masking
in a loop over kernel taps.

Under `torch.compile`, TorchInductor fuses the Canon backward with neighbouring
LayerNorm ops, producing a pathological kernel explosion: 6+ Triton reduction
kernels that inflate memory traffic ~16x over the theoretical minimum. With 24
Canon calls per step (layers A/B/C/D × 6 blocks), this caused a ~15% throughput
regression.

### Roofline analysis

Canon is deeply memory-bound on RTX 5090:

| Parameter | Value |
|-----------|-------|
| T (tokens) | 65,536 |
| C (hidden) | 768 |
| K (kernel) | 7 (radius=3) |
| dtype | bf16 (2 bytes) |
| Arithmetic intensity | 3.50 FLOP/byte |
| Peak mem BW (RTX 5090) | 1,792 GB/s |
| Minimum IO (fwd) | 201.3 MB |
| Minimum IO (bwd) | 402.7 MB |
| Theoretical min (fwd) | 112 μs |
| Theoretical min (bwd) | 225 μs |

## Solution

Three hand-written Triton kernels registered as opaque `torch.library` ops.
Because they are opaque to Inductor, it cannot fuse them with LayerNorm —
eliminating the pathological backward kernel explosion entirely.

### Files

| File | Purpose |
|------|---------|
| `src/nanoplm/pretraining/models/modern_bert/canon_triton_kernels.py` | Triton kernel definitions (fwd, bwd_dx, bwd_dw_db) |
| `src/nanoplm/pretraining/models/modern_bert/canon_ops.py` | `torch.library` registration, CUDA launchers, autograd glue |
| `src/nanoplm/pretraining/models/modern_bert/modeling.py` | Integration point (line 575: `_varlen_canon_conv(...)`) |
| `tests/test_canon_correctness.py` | 44 correctness tests (fwd, bwd, gradcheck, integration) |
| `tests/canon_benchmark.py` | Compiled microbenchmark with roofline analysis |

### Architecture

```
varlen_canon_conv()                         # Public API
  → torch.ops.nanoplm_canon.varlen_conv_fwd # torch.library custom op
    → _canon_fwd_kernel                      # Triton JIT kernel

Backward (via register_autograd):
  → torch.ops.nanoplm_canon.varlen_conv_bwd_dx    # grad_x (transpose conv)
  → torch.ops.nanoplm_canon.varlen_conv_bwd_dw_db # grad_weight + grad_bias
```

**torch.library registration** provides:
- `DEF` — op schema (signature + return type)
- `register_fake` — FakeTensor shapes for torch.compile tracing
- `impl("CUDA")` — actual Triton kernel launchers
- `register_autograd` — backward decomposition into the two backward ops

This approach is compatible with `torch.utils.checkpoint.checkpoint(use_reentrant=False)`
because `register_autograd`'s `setup_context` / `save_for_backward` hooks into
the same `saved_tensors_default_hooks` mechanism that activation checkpointing
relies on.

### Kernel design

All three kernels tile over a 2D `(T, C)` grid. The inner K-tap loop uses
`tl.static_range(K)` so the compiler can schedule all neighbor loads
simultaneously.

**Forward (`_canon_fwd_kernel`)**:
`out[t, c] = bias[c] + Σ_k weight[c, k] * x[t + k - r, c] * valid`

- Loads `seq_id[t]` and `seq_id[t+offset]`, masks invalid cross-boundary taps
- `FP32_ACCUM` constexpr: accumulates in fp32 for bf16/fp16 inputs, native dtype for fp32/fp64

**Backward grad_x (`_canon_bwd_dx_kernel`)**:
Same structure as forward but with flipped weight indices (`K-1-k`) — the
transpose convolution.

**Backward grad_weight + grad_bias (`_canon_bwd_dw_db_kernel`)**:
Each block computes partial sums over its `BLOCK_T` tile, then uses
`tl.atomic_add` to accumulate into global `(C, K)` grad_weight and `(C,)`
grad_bias buffers (pre-zeroed). Weight/bias grads are always accumulated in
fp32 for half-precision inputs.

### Block size tuning (RTX 5090)

Per-kernel block sizes were profiled across multiple configurations. The key
constraint is register pressure — large accumulator tiles (e.g. 128×128 = 64KB
fp32) cause register spilling that destroys performance.

| Kernel | BLOCK_T | BLOCK_C | Raw efficiency | Notes |
|--------|---------|---------|----------------|-------|
| fwd | 64 | 128 | 78% | Sweet spot before spilling |
| bwd_dx | 32 | 128 | 83% | Smaller T tile = fewer registers |
| bwd_dw_db | 256 | 64 | 48% | Larger T tile reduces atomic_add contention |

The `bwd_dw_db` kernel is limited by `tl.atomic_add` contention from
`ceil(T/BLOCK_T)` blocks writing to the same `(C, K)` output. Larger `BLOCK_T`
reduces the number of atomic writers per output element.

## Results

### Isolated benchmark

Compiled microbenchmark (`torch.compile`, T=65536, C=768, K=7, 256 seqs, bf16, RTX 5090):

| Metric | Reference (compiled) | Triton (autotuned) | Speedup |
|--------|---------------------|-------------------|---------|
| Forward | 0.315 ms (36% eff) | 0.200 ms (56% eff) | 1.58× |
| Backward | 0.985 ms (23% eff) | 0.491 ms (46% eff) | 2.01× |
| Fwd+Bwd | 1.300 ms (26% eff) | 0.691 ms (49% eff) | 1.88× |

Autotune selected configs (T=65536, C=768):

| Kernel | BLOCK_T | BLOCK_C | warps | stages |
|--------|---------|---------|-------|--------|
| fwd | 32 | 128 | 8 | 2 |
| bwd_dx | 32 | 128 | 4 | 3 |
| bwd_dw_db | 256 | 64 | 4 | 3 |

### Training trace comparison (before/after)

Profiler traces from real 2×H100 FSDP training (T=65536, C=768, K=7, bf16, grad_accum=2,
activation checkpointing enabled, `torch.compile mode=max-autotune-no-cudagraphs`).

- **Before** — `run-20030051`: roll-based `_varlen_canon_inner`, step time ~567 ms, 10.7 MFU%
- **After** — `run-20030054`: Triton `varlen_canon_conv` with autotune, step time ~498 ms, 12.1 MFU%

#### GPU kernel time breakdown (8 profiled steps)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Total GPU kernel time | 2593.7 ms | 2502.9 ms | −90.8 ms |
| Canon kernel time | 544.0 ms (21.0%) | 134.4 ms (5.4%) | **−409.6 ms** |
| LayerNorm fused w/ roll | 381.1 ms (14.7%) | 0 ms | **−381.1 ms** |
| LayerNorm (clean) | 58.5 ms (2.3%) | 139.7 ms (5.6%) | +81.3 ms |
| **Canon + LN combined** | **983.6 ms (37.9%)** | **274.1 ms (11.0%)** | **−709.4 ms** |

The roll-based backward caused Inductor to fuse Canon with LayerNorm into pathological
combined kernels (`triton_per_fused_..._native_layer_norm_backward_roll_select_*`), each
running ~1.6 ms. These 381 ms of fused LN+roll kernels **vanish entirely** with the Triton
op approach. LayerNorm reverts to its clean ~140 ms baseline, and the three Canon kernels
run in a combined 134 ms — a **3.6× reduction** in Canon+LN GPU time.

#### Per-kernel timing in training

| Kernel | Count | Avg (μs) | Total (ms) | % GPU |
|--------|-------|----------|------------|-------|
| `_canon_fwd_kernel_autotuned` | 405 | 138 | 56.1 | 2.2% |
| `_canon_bwd_dx_kernel_autotuned` | 213 | 115 | 24.4 | 1.0% |
| `_canon_bwd_dw_db_kernel_autotuned` | 213 | 253 | 54.0 | 2.2% |

Training kernels run ~30% faster than isolated benchmarks (GPU pipeline already saturated,
overlapping compute/memory from surrounding ops).

#### Neighboring kernels in training

**Forward path** (most common pattern, 96 occurrences):
```
LayerNorm → canon_fwd → residual_add (triton_poi_fused_add)
```

**Backward path** (consistent across all 213 occurrences):
```
GEMM → canon_bwd_dx → 4× FillFunctor(zeros) → canon_bwd_dw_db → dtype_cast (fp32→bf16)
```

#### Wall-clock step time

| | Before | After | Delta |
|--|--------|-------|-------|
| Steady-state step time | ~567 ms | ~498 ms | **−69 ms (−12.2%)** |
| Throughput (tok/s) | ~462K | ~526K | **+64K (+13.9%)** |
| H100 MFU | 10.7% | 12.1% | +1.4 pp |

The 69 ms/step improvement exceeds the isolated Canon-only savings (~15 ms) because
eliminating the pathological backward fusion also unblocks Inductor from generating
better code for the surrounding LayerNorm and residual ops.

#### Top 5 kernels by GPU time

**Before (roll-based):**
| Rank | % GPU | Kernel |
|------|-------|--------|
| 1 | 6.8% | `triton_red_fused__to_copy_mul_sum_7` (reduction) |
| 2 | 6.7% | cutlass GEMM (64×128) |
| 3 | 6.6% | cutlass GEMM (128×64) |
| 4 | **6.1%** | `triton_per_fused_..._layer_norm_backward_roll_select_6` **(pathological)** |
| 5 | 5.7% | `triton_tem_fused_mm_t_4` (matmul) |

**After (Triton Canon):**
| Rank | % GPU | Kernel |
|------|-------|--------|
| 1 | 13.3% | NCCL ReduceScatter |
| 2 | 7.1% | cutlass GEMM (64×128) |
| 3 | 7.0% | cutlass GEMM (128×64) |
| 4 | 6.6% | `triton_tem_fused_mm_t_4` (matmul) |
| 5 | 5.1% | cutlass GEMM (128×128) |

Canon drops entirely out of the top 10. The profile is now dominated by GEMMs and NCCL
collectives — the expected profile for a well-optimized transformer.

### Remaining fusion opportunities

1. **LayerNorm → canon_fwd** (96 occurrences): fusing would eliminate one kernel launch +
   one intermediate (T,C) store/load = 100 MB round-trip, saving ~50-80 μs/call.
2. **bwd_dx + bwd_dw_db merge**: both read `grad_out`; merging saves one 100 MB pass
   (~56 μs). Risk: register pressure from combining both accumulation patterns.
3. **Absorb zero-fill into bwd_dw_db**: 4 tiny `FillFunctor` kernels (1 μs each) precede
   every `bwd_dw_db` call. Moving the zeroing into the Triton kernel eliminates 4 launch
   overheads per call.
4. **Fuse bwd_dw_db → dtype_cast**: the fp32→bf16 weight grad cast could be done in-kernel.

## Correctness

44 tests cover:

- **Forward**: parametrized over `radius={1,2,3}`, `C={64,768}`, multi-sequence layouts (2–12 seqs)
- **Backward**: grad_x, grad_weight, grad_bias compared against reference autograd (bf16 + fp32)
- **Gradcheck**: finite-difference verification at fp64 precision
- **Integration**: full `ModernBertCanonLayer` end-to-end (varlen + padded paths)

Run with:
```bash
pytest tests/test_canon_correctness.py -v
```

Benchmark with:
```bash
python tests/canon_benchmark.py [--T 65536] [--C 768] [--radius 3] [--n-seqs 256]
```

## Design decisions

**Why `torch.library` instead of `torch.autograd.Function`?**
`torch.autograd.Function` with `save_for_backward(x)` pins the full `(T, C)`
activation tensor for all 24 Canon calls, defeating activation checkpointing
and causing OOM. `torch.library.register_autograd` integrates with
`saved_tensors_default_hooks`, allowing checkpointing to swap tensors to/from
CPU or recompute them.

**Why not `torch.compiler.disable`?**
Wrapping Canon in `@torch.compiler.disable` forces a graph break before and
after every Canon call. With 24 Canon calls per step across the model, this
fragments compilation and prevents Inductor from optimizing the surrounding
ops. The torch.library approach keeps the graph intact — Inductor sees the ops
as opaque leaves but can still fuse everything around them.

**Why separate block sizes per kernel?**
The three kernels have different register pressure profiles and access patterns.
The forward and bwd_dx kernels are pure streaming ops that benefit from smaller
tiles to avoid spilling. The bwd_dw_db kernel performs a reduction via
`atomic_add`, where larger T tiles reduce contention but require more registers
for the inner loop. A single block size for all three leaves significant
performance on the table.

**Boundary handling vs reference**:
`torch.roll` wraps around modulo T. For a single sequence (all tokens share
seq_id=0), positions 0 and T-1 "see" each other as valid neighbors. The Triton
kernel uses explicit bounds checking (`0 <= t+offset < T`) and never wraps. This
difference only manifests for single-sequence inputs, where the model uses
`F.conv1d` instead — so in practice the implementations agree on all real inputs.
