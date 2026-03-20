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

Hand-written Triton kernels registered as opaque `torch.library` ops, with two
API levels:

1. **Unfused Canon conv** (`varlen_canon_conv`): three kernels (fwd, bwd_dx,
   bwd_dw_db) for standalone depthwise conv without LayerNorm.
2. **Fused LN+Conv** (`varlen_ln_canon_conv`): fuses LayerNorm normalization
   into the conv, eliminating intermediate (T,C) materialization. Uses
   FlashAttention-style 1D kernels for LN stats and backward.

Because they are opaque to Inductor, it cannot fuse them with LayerNorm —
eliminating the pathological backward kernel explosion entirely.

### Files

| File | Purpose |
|------|---------|
| `src/nanoplm/pretraining/models/modern_bert/canon_triton_kernels.py` | Triton kernel definitions (11 kernels + autotuned wrappers) |
| `src/nanoplm/pretraining/models/modern_bert/canon_ops.py` | `torch.library` registration, CUDA launchers, autograd glue |
| `src/nanoplm/pretraining/models/modern_bert/modeling.py` | Integration point (`varlen_ln_canon_conv(...)` / `varlen_canon_conv(...)`) |
| `tests/test_canon_correctness.py` | 82 correctness tests (fwd, bwd, gradcheck, integration) |
| `tests/canon_benchmark.py` | Compiled microbenchmark (unfused conv only) |
| `tests/canon_ln_benchmark.py` | Fused LN+Conv benchmark with roofline analysis |

### Architecture

```
varlen_ln_canon_conv()                                # Public API (fused)
  → torch.ops.nanoplm_canon.varlen_ln_conv_fwd       # fused forward
    → _ln_stats_kernel                                #   one-pass mean + rstd
    → _canon_ln_fwd_kernel                            #   inline LN + conv + skip

Backward (via register_autograd):
  → torch.ops.nanoplm_canon.varlen_ln_conv_bwd_dx_dgamma  # fused bwd steps 1-3
    → _fused_conv_bwd_dx_ln_bwd_kernel                     #   conv transpose + add + LN bwd
  → torch.ops.nanoplm_canon.varlen_ln_conv_bwd_dw_db      # conv weight/bias grads
    → _canon_ln_bwd_dw_db_partial_kernel                   #   partial-buffer reduction
```

```
varlen_canon_conv()                         # Public API (unfused)
  → torch.ops.nanoplm_canon.varlen_conv_fwd # standalone conv forward
    → _canon_fwd_kernel

Backward (via register_autograd):
  → torch.ops.nanoplm_canon.varlen_conv_bwd_dx    # grad_x (transpose conv)
  → torch.ops.nanoplm_canon.varlen_conv_bwd_dw_db # grad_weight + grad_bias
```

**torch.library registration** provides:
- `DEF` — op schema (signature + return type)
- `register_fake` — FakeTensor shapes for torch.compile tracing
- `impl("CUDA")` — actual Triton kernel launchers
- `register_autograd` — backward decomposition

This approach is compatible with `torch.utils.checkpoint.checkpoint(use_reentrant=False)`
because `register_autograd`'s `setup_context` / `save_for_backward` hooks into
the same `saved_tensors_default_hooks` mechanism that activation checkpointing
relies on.

### Kernel design

**Unfused Conv kernels** tile over a 2D `(T, C)` grid. The inner K-tap loop uses
`tl.static_range(K)` so the compiler can schedule all neighbor loads
simultaneously.

**Fused LN+Conv kernels** follow the FlashAttention Triton LayerNorm design:
1D grid with `BLOCK_N >= C` (full hidden dim in registers per row). This enables
single-pass stats, in-register LN normalize, and multi-row loops with partial
dgamma/dw/db accumulation — no atomics needed.

All kernels use `FP32_ACCUM: tl.constexpr` to accumulate in fp32 for bf16/fp16
inputs while preserving native precision for fp32/fp64 (required for gradcheck).

#### Kernel inventory

| Kernel | Grid | Purpose |
|--------|------|---------|
| `_canon_fwd_kernel` | 2D (T,C) | Standalone conv forward |
| `_canon_bwd_dx_kernel` | 2D (T,C) | Conv backward grad_x (transpose conv) |
| `_canon_bwd_dw_db_kernel` | 2D (T,C) | Conv backward grad_w/grad_b (atomic_add) |
| `_canon_ln_fwd_kernel` | 2D (T,C) | Fused LN+Conv forward (inline normalize + conv + skip) |
| `_ln_stats_kernel` | 1D (T) | One-pass mean + rstd per row |
| `_ln_bwd_kernel` | 1D (SMs) | LN backward dx + partial dgamma (standalone) |
| `_fused_conv_bwd_dx_ln_bwd_kernel` | 1D (SMs) | Fused conv transpose + add + LN backward |
| `_canon_ln_bwd_dw_db_partial_kernel` | 2D (T,C) | Conv bwd_dw_db with partial-buffer reduction |

#### Key design patterns

**One-pass LN stats** (`_ln_stats_kernel`): single kernel replaces
`x.to(fp32).mean(-1)` + `x.to(fp32).var(-1)` + `rsqrt`. One program per row,
`BLOCK_N` covers full hidden dim. Eliminates 2 separate reduction passes.

**Fused conv_bwd_dx + LN backward** (`_fused_conv_bwd_dx_ln_bwd_kernel`):
merges three operations into one kernel launch:
1. Conv transpose (K-tap loop over neighbor `grad_out` rows)
2. Residual add (`grad_ln_out = grad_out + grad_conv`)
3. LN backward (dx + partial dgamma)

This eliminates two intermediate (T,C) tensors (`grad_conv`, `grad_ln_out`) —
~400 MB of IO saved. Uses 1D grid with multi-row loop; L2 cache absorbs most
of the K-fold neighbor `grad_out` reuse.

**Partial-buffer reduction** (`_canon_ln_bwd_dw_db_partial_kernel`): replaces
the `tl.atomic_add` approach with per-block partial buffers. Each T-block writes
its partial sums to `partial_w[pid_t, :]` and `partial_b[pid_t, :]`. The host
reduces with `.sum(0)` afterward. Eliminates all atomic contention.

**Inline LN recompute in dw_db**: the conv weight grad kernel recomputes
`LN(x) = (x - mean) * rstd * gamma` inline for each K-tap neighbor, avoiding
materialization of the LN output. Mean/rstd (T×fp32 each, ~256KB) stay in L2.

### Autotune

All kernels use `triton.autotune` with `cache_results=True` for persistent
caching. Configs sweep `num_warps` for 1D kernels and `BLOCK_T × BLOCK_C ×
num_warps × num_stages` for 2D kernels. The `bwd_dw_db` partial kernel uses a
`pre_hook` to zero partial buffers before each autotune trial.

Autotune can be disabled via `NANOPLM_CANON_TRITON_AUTOTUNE=0` for deterministic
kernel selection (uses hardcoded fallback block sizes tuned on RTX 5090).

Selected configs (T=65536, C=768, RTX 5090):

| Kernel | Config | warps | stages |
|--------|--------|-------|--------|
| `_canon_ln_fwd` | BLOCK_T=32, BLOCK_C=128 | 4 | 2 |
| `_ln_stats` | (1D) | 4 | 3 |
| `_fused_conv_bwd_dx_ln_bwd` | (1D) | 1 | 3 |
| `_canon_ln_bwd_dw_db_partial` | BLOCK_T=128, BLOCK_C=64 | 4 | 2 |

## Results

### Isolated benchmark (fused LN+Conv)

Microbenchmark (T=65536, C=768, K=7, 256 seqs, bf16, RTX 5090):

| Metric | Unfused (LN + Triton Conv) | Fused (LN+Conv) | Speedup |
|--------|---------------------------|-----------------|---------|
| Forward | 0.444 ms (51% eff) | 0.206 ms (55% eff) | 2.16× |
| Backward | 1.396 ms (32% eff) | 0.876 ms (51% eff) | 1.59× |
| Fwd+Bwd | 1.840 ms (37% eff) | 1.082 ms (52% eff) | 1.70× |

Per-step projection (24 Canon layers): **18.2 ms recovered** vs unfused.

### Isolated benchmark (unfused Conv only)

Compiled microbenchmark (`torch.compile`, same config):

| Metric | Reference (compiled) | Triton (autotuned) | Speedup |
|--------|---------------------|-------------------|---------|
| Forward | 0.315 ms (36% eff) | 0.200 ms (56% eff) | 1.58× |
| Backward | 0.985 ms (23% eff) | 0.491 ms (46% eff) | 2.01× |
| Fwd+Bwd | 1.300 ms (26% eff) | 0.691 ms (49% eff) | 1.88× |

### Training trace comparison

Profiler traces from real training (T=65536, C=768, K=7, bf16,
activation checkpointing enabled, `torch.compile mode=max-autotune-no-cudagraphs`).

#### Evolution across iterations

| Trace | Description | Canon+LN time | % step |
|-------|-------------|---------------|--------|
| `run-20030051` | Roll-based (torch.compile) | 983.6 ms | 37.9% |
| `run-20030054` | Triton Conv (unfused) | 274.1 ms | 12.7% |
| `run-20030152` | First fused attempt (regression) | 305.0 ms | 13.4% |
| **`run-20030914`** | **Fused LN+Conv (current)** | **171.4 ms** | **8.3%** |

Total improvement: **812 ms saved** (5.7× reduction) from roll-based to current fused.

#### Per-kernel efficiency (run-20030914)

| Kernel | Count | Avg (μs) | Total (ms) | % step | BW (GB/s) | Eff |
|--------|-------|----------|------------|--------|-----------|-----|
| `_fused_conv_bwd_dx_ln_bwd` | 204 | 258 | 52.7 | 2.6% | 1172 | 65% |
| `_canon_ln_fwd` | 388 | 132 | 51.4 | 2.5% | 1525 | 85% |
| `_canon_ln_bwd_dw_db_partial` | 204 | 176 | 35.8 | 1.7% | 1149 | 64% |
| `_ln_stats` | 387 | 67 | 26.0 | 1.3% | 1509 | 84% |
| **Total Canon** | | | **171.4** | **8.3%** | | |

The forward kernels (`_canon_ln_fwd` at 85%, `_ln_stats` at 84%) are near the
memory bandwidth ceiling. The backward kernels (`_fused_conv_bwd_dx_ln_bwd` at
65%, `_canon_ln_bwd_dw_db_partial` at 64%) have lower efficiency due to K-fold
neighbor access patterns that exceed L2 cache reuse.

#### Step time context

Canon at 8.3% of step time is now comparable to FlashAttention (~8%) and well
below GEMMs (~42%) and NCCL collectives (~14%). The profile is dominated by
GEMMs and communication — the expected profile for a well-optimized transformer.

#### Wall-clock step time (roll-based → current)

| | Roll-based | Triton unfused | Fused LN+Conv |
|--|------------|---------------|---------------|
| Steady-state step time | ~567 ms | ~498 ms | ~498 ms* |
| Canon+LN GPU time | 983.6 ms | 274.1 ms | 171.4 ms |

*Step time reduction from the fused path is partially masked by communication
overlap. The 103 ms kernel time reduction frees GPU cycles for better compute/
communication overlap.

### Remaining optimization opportunities

1. **`_fused_conv_bwd_dx_ln_bwd` (65% eff)**: the per-row conv transpose loads
   K=7 neighbor `grad_out` rows per row. With `rows_per_program` ≈ 64, L2
   absorbs most reuse, but the ~35% gap from peak suggests some L2 misses on
   row-chunk boundaries. Increasing `rows_per_program` (fewer programs, more
   rows each) could improve locality.

2. **`_canon_ln_bwd_dw_db_partial` (64% eff)**: each K-tap loads neighbor x and
   recomputes LN inline. The K-fold x access is the bottleneck. Pre-materializing
   LN(x) would trade one (T,C) write for K-1 recomputes — at K=7 the recomputes
   are likely cheaper due to L2-resident mean/rstd, so the current approach is
   near-optimal.

3. **Fully fused backward**: merging dw_db accumulation into the per-row kernel
   would save one `grad_out` read pass (~100 MB), but requires K register-resident
   accumulators per channel (28KB at K=7, BLOCK_N=1024) — feasible but increases
   register pressure.

## Correctness

82 tests cover:

- **Unfused Conv**: forward (bf16/fp32), backward (bf16/fp32), gradcheck (fp64),
  multi-sequence layouts
- **Fused LN+Conv**: forward (bf16/fp32, C={64,768}), backward (bf16/fp32),
  gradcheck (fp64), multi-sequence layouts
- **Integration**: full `ModernBertCanonLayer` end-to-end (varlen + padded paths)

Run with:
```bash
pytest tests/test_canon_correctness.py -v
```

Benchmark with:
```bash
python tests/canon_ln_benchmark.py [--T 65536] [--C 768] [--radius 3] [--n-seqs 256]
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

**Why FlashAttention-style 1D kernels for LN?**
The LN stats, LN backward, and fused conv_bwd_dx + LN backward kernels use a
1D grid with `BLOCK_N >= C` (full hidden dim in registers). This pattern, from
FlashAttention's Triton LayerNorm, enables:
- Single-pass stats (no two-pass mean+var)
- Per-row backward with all reductions in registers (no shared memory)
- Multi-row loops with partial dgamma accumulation (no atomics)
- Natural fusion with the conv transpose (K-tap loop over neighbors in registers)

The 2D `(T, C)` tiled approach used for the unfused conv kernels cannot fuse
with LN backward because LN's reduction over C requires the full hidden dim.

**Why partial buffers instead of atomic_add for dw_db?**
The unfused `_canon_bwd_dw_db_kernel` uses `tl.atomic_add` with `ceil(T/BLOCK_T)`
blocks writing to the same `(C, K)` output — severe contention at 48% efficiency.
The fused `_canon_ln_bwd_dw_db_partial_kernel` writes to per-block partial buffers
and reduces on the host with `.sum(0)`. This eliminates all atomic contention
and improved efficiency to 64%.

**Why fuse conv_bwd_dx + LN backward?**
The unfused backward computed: (1) conv_bwd_dx → write `grad_conv`, (2) add →
write `grad_ln_out`, (3) LN backward → read `grad_ln_out`. This materialized
two (T,C) intermediates (~400 MB). The fused kernel computes the conv transpose
in registers, adds `grad_out`, and runs LN backward — all in one pass per row.
This improved backward efficiency from 39% to 51%.

**Boundary handling vs reference**:
`torch.roll` wraps around modulo T. For a single sequence (all tokens share
seq_id=0), positions 0 and T-1 "see" each other as valid neighbors. The Triton
kernel uses explicit bounds checking (`0 <= t+offset < T`) and never wraps. This
difference only manifests for single-sequence inputs, where the model uses
`F.conv1d` instead — so in practice the implementations agree on all real inputs.
