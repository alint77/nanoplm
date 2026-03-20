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

## Benchmark results

Compiled benchmark (`torch.compile`, T=65536, C=768, K=7, 256 seqs, bf16, RTX 5090):

| Metric | Reference (compiled) | Triton (tuned) | Speedup |
|--------|---------------------|----------------|---------|
| Forward | 0.315 ms (36% eff) | 0.212 ms (53% eff) | 1.49× |
| Backward | 0.978 ms (23% eff) | 0.505 ms (45% eff) | 1.94× |
| Fwd+Bwd | 1.292 ms (26% eff) | 0.716 ms (47% eff) | 1.80× |

### Per-step projection (24 Canon calls)

| | Reference | Triton | Delta |
|--|-----------|--------|-------|
| Canon overhead | 31.0 ms | 17.2 ms | −13.8 ms |
| % of base step (573 ms) | 5.4% | 3.0% | −2.4 pp |

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
