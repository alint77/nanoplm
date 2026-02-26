# mHC-lite Triton kernels (current state)

This document tracks the **current** mHC-lite Triton setup in this repo:
- active kernels
- where each kernel is launched
- current launch-config logic (hardcoded vs heuristic path)
- one-shot gridsearch workflow for finding hardcoded params

## End-to-end mHC-lite path

Main model path:
- `src/nanoplm/pretraining/models/modern_bert/modeling.py` (`MHCLiteBlock._forward_triton`)

Per-layer flow:
1. **K1 (Triton):** fused RMSNorm + projection  
   `torch.ops.nanoplm_mhc.fused_rmsnorm_project(...)`
2. **K2 (PyTorch):** coefficient math (`sigmoid/softmax`, `a_res @ perm_mat`)
3. **K3 (Triton):** pre-map  
   `torch.ops.nanoplm_mhc.fused_pre_map(...)`
4. Wrapped transformer layer
5. **K4 (Triton):** post-res  
   `torch.ops.nanoplm_mhc.fused_post_res(...)`

Custom op registration + autograd lives in:
- `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py`

Kernel implementations live in:
- `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py`

## Active Triton kernels

The mHC-lite Triton stack currently uses **7 kernels**:

1. `_fused_rmsnorm_project_fwd_kernel`
2. `_fused_rmsnorm_project_bwd_dx_kernel`
3. `_fused_pre_map_fwd_kernel`
4. `_fused_pre_map_bwd_dx_kernel`
5. `_fused_pre_map_bwd_hpre_kernel`
6. `_fused_post_res_fwd_kernel_n4`
7. `_fused_post_res_bwd_fused_kernel_n4`

Important update:
- K4 backward is now a **single fused kernel** (`_fused_post_res_bwd_fused_kernel_n4`).
- The old split K4 backward kernels were removed:
  - `_fused_post_res_bwd_xlo_kernel_n4`
  - `_fused_post_res_bwd_Hhp_kernel`

## Launch config logic (what is hardcoded vs heuristic)

Source of truth: launcher code in `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py`.

### K1
- `fused_rmsnorm_project` fwd:
  - `BLOCK_T=128`
  - `BLOCK_K=min(128, next_power_of_2(nC))`
  - warps/stages from `_get_hw_config()`
- `fused_rmsnorm_project_bwd_dx`:
  - SM90: `BLOCK_T=64`
  - other arch: `BLOCK_T=128`
  - `BLOCK_K=min(128, next_power_of_2(nC))`

### K3
- `fused_pre_map` fwd:
  - **SM90 hardcoded path:** `BLOCK_T=128`, `BLOCK_C=128`, `num_warps=8`, `num_stages=4`
  - **heuristic path (non-SM90):** `BLOCK_T=64`, `BLOCK_C=min(256, next_power_of_2(C))`, `num_stages` from `_get_hw_config()`
- `fused_pre_map_backward`:
  - `BLOCK_T=64`
  - `BLOCK_C=min(256, next_power_of_2(C))`
  - warps/stages from `_get_hw_config()`

### K4
- `fused_post_res` fwd:
  - `BLOCK_T=64` on `cc_major >= 9`, else `32`
  - `BLOCK_C=128` when `C >= 128`, else `next_power_of_2(C)`
  - warps: `8` on `cc_major >= 9`, else default
- `fused_post_res_backward` (fused K4 bwd):
  - SM90: `BLOCK_T=32`, `BLOCK_C=128`, `num_stages=2`
  - other arch: `BLOCK_T=32`, `BLOCK_C=min(256, next_power_of_2(C))`, `num_stages` default

## One-shot gridsearch workflow

Script:
- `tests/mhc_triton_gridsearch.py`

What it searches:
- launch configs for the 7 active kernels above
- candidate dimensions include `BLOCK_T`, `BLOCK_K`/`BLOCK_C`, `num_warps`, `num_stages`

Example:
```bash
python3 tests/mhc_triton_gridsearch.py \
  --T 65536 --C 1024 --n 4 \
  --warmup 8 --iters 25 \
  --json-out output/logs/mhc_gridsearch_h100_65536_1024.json
```

Behavior:
- runs a one-time sweep
- prints best config per kernel + top-N trials
- writes JSON payload (device, shape, timing, per-kernel best/top/failed counts)

Intended usage:
1. Run gridsearch once on target hardware/shape.
2. Copy best configs into launcher logic in `mhc_triton_ops.py`.
3. Keep runtime path simple (no online autotune during training).

## Benchmarking before/after hardcoding

Microbenchmark script:
- `tests/mhc_triton_kernels_benchmark.py`

Use it to compare:
1. current hardcoded launcher values
2. new hardcoded values derived from gridsearch output

Example:
```bash
python3 tests/mhc_triton_kernels_benchmark.py --T 65536 --C 1024 --n 4 --iters 50 --warmup 10
```

The benchmark prints per-kernel:
- avg/median/min/max latency
- achieved bandwidth
- optional roofline efficiency (if `--peak-gbps` is provided)

## Constraints

- mHC Triton kernels are specialized for `n=4`.
- Custom ops enforce `n=4` and raise on other values.
