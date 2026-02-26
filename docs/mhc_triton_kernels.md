# mHC-lite Triton kernels: pipeline + trace mapping

This report explains:
- Where the **pure-torch mHC-lite** path lives (model + pipeline)
- Which **Triton kernels** it launches
- What the **last two** `chrome_trace.json` files contain on **GPU stream 7**
- A mapping from each **mHC-lite-related kernel name** back to the responsible code

## Where mHC-lite lives (pure-torch path)

- mHC-lite block implementation: `src/nanoplm/pretraining/models/modern_bert/modeling.py:923` (`class MHCLiteBlock`)
- Config flags:
  - `src/nanoplm/pretraining/models/modern_bert/modeling.py:146` (`ModernBertConfig.use_mhc_lite`, `mhc_n_streams`, `mhc_triton_fused`)
  - Layer wrapping: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1054`
- Pure-torch training loop uses `torch.compile`:
  - `src/nanoplm/pretraining/pure_pipeline.py:845`
  - Packed/varlen forward passes `cu_seqlens`/`max_seqlen` so the model runs the `(T, …)` token-major path (required for the Triton-fused mHC-lite path): `src/nanoplm/pretraining/pure_pipeline.py:1063`
- Triton-fused mHC-lite is only used when all of the following are true:
  - `triton_fused=True`, CUDA, bf16, and `x_streams.dim() == 3` (`(T, n, C)`)
  - Gate logic: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1037`

Note: there is **no Sinkhorn / full mHC** implementation in `src/`; only mHC-lite is present.

## Forward graph (math → ops/kernels)

Implemented in `MHCLiteBlock._forward_triton` at `src/nanoplm/pretraining/models/modern_bert/modeling.py:994`.

Per layer:
1) **K1 (Triton)** fused RMSNorm + projection  
   Call site: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1007`  
   Op: `torch.ops.nanoplm_mhc.fused_rmsnorm_project(x_flat, W_all.weight)`
2) **K2 (PyTorch)** coefficients + permutation mixing  
   - `sigmoid/softmax` on the small projected head (size `2n + n!`)  
   - `H_res = a_res @ perm_mat`  
   Code: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1020`
3) **K3 (Triton)** pre-map (weighted stream aggregation)  
   Call site: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1024`  
   Op: `torch.ops.nanoplm_mhc.fused_pre_map(x_streams, h_pre)`
4) **Transformer layer** (wrapped encoder layer, no residual inside the wrapper)
5) **K4 (Triton)** post-res (merged stream mixing + post scaling)  
   Call site: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1030`  
   Op: `torch.ops.nanoplm_mhc.fused_post_res(x_streams, layer_output, H_merged, h_post)`

## Triton custom-op stack (responsibility chain)

To keep `torch.compile` happy (no graph breaks), the fused kernels are exposed as dispatcher ops:
- Op definitions + FakeTensor support + autograd registration: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:1`
- Triton kernel implementations: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:1`
- `modeling.py` imports ops for registration: `src/nanoplm/pretraining/models/modern_bert/modeling.py:22`

## Chrome traces (main GPU compute stream = 7)

The repo contains exactly two profiler traces:
- `output/pretraining_checkpoints/run-26021148-2/profiler_traces/chrome_trace.json`
- `output/pretraining_checkpoints/run-26021149-2/profiler_traces/chrome_trace.json`

### `run-26021148-2` (stream 7): mHC-lite fused path present

On stream 7, the trace contains the mHC-lite Triton kernels below (total: **142 calls**, ~**218.4 ms** on stream 7):
- `_fused_rmsnorm_project_fwd_kernel` (17 calls, ~14.4 ms)
- `_fused_rmsnorm_project_bwd_dx_kernel` (18 calls, ~32.9 ms)
- `_fused_pre_map_fwd_kernel` (17 calls, ~17.6 ms)
- `_fused_pre_map_bwd_dx_kernel` (18 calls, ~20.9 ms)
- `_fused_pre_map_bwd_hpre_kernel` (18 calls, ~19.5 ms)
- `_fused_post_res_fwd_kernel_n4` (18 calls, ~36.2 ms)
- `_fused_post_res_bwd_xlo_kernel_n4` (18 calls, ~37.4 ms)
- `_fused_post_res_bwd_Hhp_kernel` (18 calls, ~39.6 ms)

The trace also contains stream-7 kernels caused by the remaining **PyTorch** parts of mHC-lite (identified by matching `aten::mm` shapes to mHC-lite math):
- `H_res = a_res @ perm_mat` (`[T,24] x [24,16]`): CUTLASS WMMA bf16 kernel (16 calls, ~0.048 ms total)
- Backward of that matmul (`[T,16] x [16,24]`): CUTLASS WMMA bf16 kernel (16 calls, ~0.048 ms total)
- `grad_W_all = (grad_out * inv_rms)^T @ x_flat` (from K1 backward): CUTLASS bf16 GEMM kernel (24 calls, ~15.0 ms total) + a small `cublasLt::splitKreduce_kernel` (~0.04 ms total)

### `run-26021149-2` (stream 7): mHC-lite fused path absent

No `_fused_{rmsnorm,pre_map,post_res}_*` kernel names appear on stream 7 in this trace, so it does **not** include the fused mHC-lite path.

## RTX 5090 theoretical minimum time + efficiency (T=65536, C=2560, n=4)

This section estimates a *roofline-style* lower bound and compares it to the measured kernel durations from:
- `output/pretraining_checkpoints/run-26021148-2/profiler_traces/chrome_trace.json` (GPU stream 7)

Assumptions:
- Shapes: `T=65536`, `C=2560`, `n=4` (`nC=10240`), `D_out=32` (because `2n + n! = 8 + 24`)
- Data types as used by the fused path: `x_streams/x_flat` bf16, `h_pre/h_post/H_merged` float32 (passed as `.float()`), outputs bf16
- Peak DRAM bandwidth for RTX 5090: **~1792 GB/s**
- “Theoretical mathematical minimum time” = `(mandatory bytes read+written) / peak_bandwidth`
  - Ignores launch overhead, instruction/latency limits, and cache effects (e.g. `W_all.weight` can be L2-resident), so treat this as an approximation.

| kernel | avg time (ms) | min time @1792 GB/s (ms) | efficiency (=min/avg) | achieved BW (GB/s) |
|---|---:|---:|---:|---:|
| `_fused_rmsnorm_project_fwd_kernel` | 0.847 | 0.752 | 88.8% | 1590.5 |
| `_fused_rmsnorm_project_bwd_dx_kernel` | 1.825 | 1.503 | 82.3% | 1475.6 |
| `_fused_pre_map_fwd_kernel` | 1.037 | 0.937 | 90.3% | 1619.0 |
| `_fused_pre_map_bwd_dx_kernel` | 1.163 | 0.937 | 80.5% | 1442.9 |
| `_fused_pre_map_bwd_hpre_kernel` | 1.080 | 0.937 | 86.7% | 1553.9 |
| `_fused_post_res_fwd_kernel_n4` | 2.009 | 1.688 | 84.0% | 1506.1 |
| `_fused_post_res_bwd_xlo_kernel_n4` | 2.075 | 1.688 | 81.3% | 1457.7 |
| `_fused_post_res_bwd_Hhp_kernel` | 2.199 | 1.688 | 76.8% | 1375.6 |

## SM90 (H100) optimizations + trace validation (T=65536, C=2560, n=4)

### Optimizations applied

The original kernel configs were tuned for SM120 (RTX 5090). On SM90 (H100 80GB HBM3, 132 SMs, 3350 GB/s peak BW), three kernels had poor efficiency due to register pressure and shared memory limits. All fixes are gated behind `cc_major == 9` so the SM120 path is unchanged.

**1. `_fused_post_res_bwd_Hhp_kernel` — restructured inner C-loop (18% → 83%)**

Root cause: the original inner loop loaded 9 2D tiles simultaneously (lo + x0..x3 + go0..go3), each `(BLOCK_T=64, BLOCK_C=256)` bf16→f32. With 8 warps (256 threads): 9 tiles × 64 × 256 / 256 threads = 576 f32 regs/thread. H100 limit is 255 → massive register spilling.

Fix: phased tile streaming. Load 4 `go` tiles first (stay live), then stream `lo` once (compute `ghp`, `lo` dies), then stream `x_j` one at a time (max live: 4 go + 1 x = 5 tiles). SM90 launch config: `BLOCK_C=128`, `num_stages=2`. The restructured kernel body applies to all architectures (it's a strict improvement in load ordering) but the reduced `BLOCK_C` is SM90-only.

**2. `_fused_rmsnorm_project_bwd_dx_kernel` — reduced BLOCK_T (57% → 77%)**

Root cause: `BLOCK_T=128` creates a `(128, D_out=32)` f32 dot output = 128×32/256 = 16 regs, but the full pipeline (x tile, W tile, grad_out, proj_out, accumulators) hits ~170 regs/thread.

Fix: `BLOCK_T=64` for SM90 only. Halves register pressure, enables 2-block occupancy per SM.

**3. `_fused_pre_map_fwd_kernel` — reduced BLOCK_C + pipeline tuning (61% → 69%)**

Root cause: `BLOCK_C=256` with `num_stages=4` needs 4 × (n=4 x tiles) × 64×256×2B = 512KB pipeline buffers. H100 has 228KB shared mem → pipelining silently disabled by Triton.

Fix: `BLOCK_C=128`, `num_stages=3` for SM90 only → 4 × 3 × 16KB = 192KB, fits in 228KB.

### Trace validation (`run-26021500-2`, H100 80GB)

Trace: `output/pretraining_checkpoints/run-26021500-2/profiler_traces/chrome_trace.json`
Config: `hidden_size=2560`, `mhc_n_streams=4`, `num_hidden_layers=4`, `micro_batch_size=128`, packing enabled → `T=65536` tokens/microbatch.
Profiler captured 9 training steps (steps 10-15 per config). 132 SMs, 8 warps, stream 7.

| kernel | bytes (MB) | LB (us) | trace avg (us) | eff % | clean* eff % | achieved BW (GB/s) |
|---|---:|---:|---:|---:|---:|---:|
| `rmsnorm_project_fwd` | 1347 | 402 | 459 | 82% | **88%** | 2937 |
| `rmsnorm_project_bwd_dx` | 2694 | 804 | 1040 | 77% | **77%** | 2591 |
| `pre_map_fwd` | 1679 | 501 | 723 | 69% | **69%** | 2322 |
| `pre_map_bwd_dx` | 1679 | 501 | 601 | 83% | **83%** | 2794 |
| `pre_map_bwd_hpre` | 1679 | 501 | 609 | 82% | **82%** | 2757 |
| `post_res_fwd_n4` | 3025 | 903 | 1063 | 85% | **85%** | 2846 |
| `post_res_bwd_xlo_n4` | 3361 | 1003 | 1065 | 94% | **94%** | 3156 |
| `post_res_bwd_Hhp` | 3025 | 903 | 1089† | 67%→ | **83%** | 2779 |
| **TOTAL** | **18489** | **5519** | **6648** | | **83%** | **2781** |

\* "Clean" = excluding invocations that overlap with NCCL ReduceScatter on stream 26.

† The Hhp kernel shows bimodal latency: 21 fast invocations at 1089 us (83% eff) and 15 slow invocations at ~1690 us (53% eff). Every slow invocation overlaps with `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` on a separate stream, confirming the slowdown is memory bandwidth contention from communication overlap — not a kernel issue.

**Per-layer total: 6.65 ms (clean) / 5.52 ms (lower bound) = 83% overall bandwidth efficiency.**

### Benchmark vs trace comparison

The microbenchmark (`tests/mhc_triton_kernels_benchmark.py`, T=2048, C=256) accurately predicts real-training efficiency:

| kernel | benchmark eff % | trace clean eff % | delta |
|---|---:|---:|---:|
| `rmsnorm_project_fwd` | 89% | 88% | -1% |
| `rmsnorm_project_bwd_dx` | 78% | 77% | -1% |
| `pre_map_fwd` | 69% | 69% | 0% |
| `pre_map_bwd_dx` | 81% | 83% | +2% |
| `pre_map_bwd_hpre` | 83% | 82% | -1% |
| `post_res_fwd_n4` | 84% | 85% | +1% |
| `post_res_bwd_xlo_n4` | 94% | 94% | 0% |
| `post_res_bwd_Hhp` | 84% | 83% | -1% |

All within ±2%. The benchmark is a reliable proxy for real training performance.

### Architecture-specific launch configs

All SM90 tuning is gated behind `cc_major == 9`. SM120 and other architectures use the original configs.

| kernel | param | SM90 (H100) | SM120 (RTX 5090) / other |
|---|---|---|---|
| K1 bwd (`rmsnorm_project_bwd_dx`) | BLOCK_T | 64 | 128 (original) |
| K3 fwd (`pre_map_fwd`) | BLOCK_C | 128 | min(256, C) |
| K3 fwd (`pre_map_fwd`) | num_stages | 3 | from `_get_hw_config()` |
| K4 bwd Hhp (`post_res_bwd_Hhp`) | BLOCK_C | 128 | min(256, C) |
| K4 bwd Hhp (`post_res_bwd_Hhp`) | num_stages | 2 | from `_get_hw_config()` |

The Hhp kernel body restructuring (phased 4+1 tile streaming) applies to all architectures — it reduces max live 2D tiles from 9 to 5 without changing the computation, which is neutral-to-beneficial everywhere.

## Kernel → responsible code

### mHC-lite Triton kernels (directly “ownable”)

| Trace kernel name | Purpose | Kernel definition | Launcher (torch op impl) | Model callsite |
|---|---|---:|---:|---:|
| `_fused_rmsnorm_project_fwd_kernel` | K1 forward: RMSNorm(x_flat) + dot with `W_all.weight` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:56` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:113` | `src/nanoplm/pretraining/models/modern_bert/modeling.py:1007` |
| `_fused_rmsnorm_project_bwd_dx_kernel` | K1 backward: dX for fused RMSNorm+proj | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:112` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:156` | via autograd registration: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:354` |
| `_fused_pre_map_fwd_kernel` | K3 forward: `layer_input[t,c] = Σ_j h_pre[t,j] * x[t,j,c]` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:673` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:186` | `src/nanoplm/pretraining/models/modern_bert/modeling.py:1024` |
| `_fused_pre_map_bwd_dx_kernel` | K3 backward: dX_streams | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:720` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:218` | via autograd registration: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:371` |
| `_fused_pre_map_bwd_hpre_kernel` | K3 backward: d(h_pre) (reduces over C) | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:767` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:230` | via autograd registration: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:371` |
| `_fused_post_res_fwd_kernel_n4` | K4 forward (n=4): `H_merged @ x + h_post * layer_output` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:258` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:268` | `src/nanoplm/pretraining/models/modern_bert/modeling.py:1030` |
| `_fused_post_res_bwd_xlo_kernel_n4` | K4 backward A (n=4): dX_streams + d(layer_output) | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:361` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:313` | via autograd registration: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:375` |
| `_fused_post_res_bwd_Hhp_kernel` | K4 backward B: d(H_merged) + d(h_post) (reduces over C) | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:465` | `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:332` | via autograd registration: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:375` |

### mHC-lite-specific (non-Triton) GPU kernels seen on stream 7

These kernels are not defined in this repo (they come from CUTLASS/cuBLAS), but they are still attributable to specific mHC-lite code paths:
- `H_res = a_res @ perm_mat` (and its backward matmul): `src/nanoplm/pretraining/models/modern_bert/modeling.py:1020`
- `grad_W_all = (grad_out * inv_rms)^T @ x_flat` (K1 backward weight grad): `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:365`
