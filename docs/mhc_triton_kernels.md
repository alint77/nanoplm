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
