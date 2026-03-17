# Block AttnRes Optimization — Implementation Log

Reference plan: `BLOCK_ATTNRES_OPTIMIZATION_PLAN_FINAL.md`

## Baseline

Trace: `run-17031118-4`, thread 7, `ProfilerStep#11-#14` averages.

| Kernel | ms/step | calls/step |
|---|---|---|
| `_block_attnres_state_bwd_kernel` | 52.2 | 25 |
| `_block_attnres_state_fwd_kernel` | 19.5 | 25 |
| **Combined** | **71.7** | |
| Step time (thread 7) | 321.1 | |

---

## Phase 0: Backward Kernel Cleanup

### 0a. R_local — eliminate atomic contention ✅ Shipped

**Change**: Replaced `tl.atomic_add` to a shared `R_ptr (D,)` with per-SM writes to `R_local_ptr (NUM_SMS, D)`. Each SM writes to its own row; the ops layer sums rows after the kernel.

**Files changed**:
- `block_attnres_triton_kernels.py`: Both `_block_attnres_bwd_kernel` and `_block_attnres_state_bwd_kernel` — `R_ptr` → `R_local_ptr`, zero SM row at kernel start, `tl.atomic_add` → `tl.load` + `tl.store`.
- `block_attnres_triton_ops.py`: Allocate `R_local = torch.zeros((NUM_SMS, D))`, pass to kernel, `R = R_local.sum(dim=0)` after.

**Also shipped**: `_pick_block_attnres_bwd_meta` now returns `(BLOCK_T, BLOCK_D, num_warps, num_stages)` — a 4-tuple with BLOCK_D explicitly returned. Currently computes the same value as before (`min(256, next_power_of_2(D))`), but the infrastructure is in place for future BLOCK_D tuning.

**Result (trace `run-17031247`)**: No measurable improvement at current training scale. Backward: `55.0 ms/step` (+5.4% — but this trace included the 0b regression, see below). After reverting 0b, the isolated 0a effect is expected to be neutral-to-slightly-positive. The atomic contention benefit scales with T (more T-tiles = more SM contention on `R_ptr`); at the current packed-token size with 108 SMs and ~60 T-tiles, contention is moderate. The change is still correct and zero-risk.

**Tests**: All 23 `test_block_attnres.py` pass.

### 0b. Register-resident scratch accumulators ❌ Reverted

**What was tried**: Inverted the Pass 1 loop nesting from D-tile-outer/source-inner to source-outer/D-tile-inner. This kept `grad_alpha` and `qw_dot` accumulators in registers across D-tiles, eliminating per-D-tile scratch loads/stores.

**Why it regressed**: The loop inversion caused `go_tile` (grad_result) and `qw_tile` (query weights) to be re-loaded `N` times per D-tile instead of once. For `N=8` sources (NC=7 completed + 1 partial), that is `8×` more grad_result/qw reads per D-tile.

Quantified:
- Scratch traffic eliminated: `N × 2 buffers × BLOCK_T × 4 bytes × D/BLOCK_D iterations ≈ 24 KB/T-tile`
- Extra go_tile/qw_tile reads introduced: `(N-1) × D/BLOCK_D × BLOCK_T × BLOCK_D × 2 bytes ≈ 672 KB/T-tile`

The scratch traffic is `~3%` of source traffic. The original code's comment was correct: *"keeps grad_result/qw loads outside the source loop, which matters much more than the tiny scratch traffic."*

**Result**: `+5.4%` backward regression in `run-17031247`. Reverted.

**Lesson**: The D-tile-outer/source-inner nesting is load-optimal for this kernel structure. The scratch buffers are the unavoidable cost of keeping go_tile shared across sources in the inner loop. Do not invert this nesting.

### 0c. BLOCK_D sweep ❌ Reverted (for now)

**What was tried**: Set backward `BLOCK_D=128` (down from 256) on A100 to reduce register pressure and improve occupancy.

**Why it was reverted**: Confounded with the 0b regression. BLOCK_D=128 doubles D-tile iterations (6 vs 3 for D=768). While total HBM bytes are the same (smaller tiles, more iterations), the extra iterations increase go_tile/qw_tile reload count and loop overhead. Needs isolated benchmarking.

**Status**: Infrastructure for BLOCK_D tuning is in place (`_pick_block_attnres_bwd_meta` returns BLOCK_D). Isolated sweep on real training shapes is still worth doing as a future step, but it was not the right change to ship alongside 0b.

---

## Phase 0 — Summary

| Sub-phase | Status | Effect |
|---|---|---|
| 0a: R_local | ✅ Shipped | Neutral at current scale; correct for larger T |
| 0b: Scratch elimination | ❌ Reverted | -5.4% regression from loop inversion |
| 0c: BLOCK_D sweep | ❌ Reverted | Confounded; needs isolated test |

**Net result**: The backward kernel is structurally unchanged from baseline except for the R_local atomic elimination. No measurable throughput improvement from Phase 0.

**Key takeaway**: The existing backward kernel is already well-optimized for its current structure. The per-D-tile scratch traffic and atomic contention are not meaningful bottlenecks. The dominant cost is the repeated reading of completed-ref source data across sublayers — which is the structural problem that Phases 1-4 address.

---

## Current state of the code

Committed changes vs baseline (`9442282`):
- `block_attnres_triton_kernels.py`: `R_ptr` → `R_local_ptr` in both backward kernels, zero SM row at start, `tl.atomic_add` → `tl.load`/`tl.store`. Minor comment improvements.
- `block_attnres_triton_ops.py`: `R_local (NUM_SMS, D)` allocation, `.sum(dim=0)` after kernel. Both meta pickers now return explicit block shapes, with SM120-specific launch heuristics for forward and backward. Added `_next_power_of_2` helper.

All 23 tests in `test_block_attnres.py` pass.

---

## SM120 / RTX 5090 Retune

Previous tuning was done on 4xA100. On the current 2xRTX5090 setup, the AttnRes kernels were still using the generic SM120 fallback rather than A100-specific launch choices.

### What was changed ✅ Shipped

**Change**: Added explicit SM120 launch heuristics in `block_attnres_triton_ops.py` for the current packed-token training shape (`T=32768`, `D=768`):

- forward: `BLOCK_T=32`, `BLOCK_D=256`, `num_warps=4`, `num_stages=2`
- backward: `BLOCK_T=32`, `BLOCK_D=128`, `num_warps=4`, `num_stages=2`

The forward meta picker now returns `(BLOCK_T, BLOCK_D, num_warps, num_stages)` just like backward, so the SM120 path can tune `BLOCK_D` explicitly.

### Why these settings

Benchmarked directly on RTX 5090 with the real traced shape and weighted by the actual `NC+1` call mix from the latest AttnRes-on trace:

- forward state kernel best weighted config: `32x256`
- backward state kernel best weighted config: `32x128`

Microbench improvement vs the old SM120 fallback (`64x256`):

- forward: `0.3160 ms/call` → `0.2188 ms/call` (`-30.8%`)
- backward: `0.7195 ms/call` → `0.4659 ms/call` (`-35.2%`)

### Trace result

Compared the pre-retune AttnRes-on trace `run-17031601-2` against the latest retuned AttnRes-on trace `run-17031621`, thread 7, `ProfilerStep#11-#14` averages.

| Metric | Before (`run-17031601-2`) | After (`run-17031621`) | Delta |
|---|---:|---:|---:|
| Step time | `642.4 ms/step` | `630.1 ms/step` | `-12.3 ms` (`-1.9%`) |
| `_block_attnres_state_bwd_kernel` | `76.8 ms/step` | `56.0 ms/step` | `-20.8 ms` (`-27.1%`) |
| `_block_attnres_state_fwd_kernel` | `42.1 ms/step` | `36.1 ms/step` | `-5.9 ms` (`-14.0%`) |
| **Combined AttnRes kernels** | **`118.8 ms/step`** | **`92.2 ms/step`** | **`-26.7 ms` (`-22.5%`)** |
| AttnRes share of step | `18.5%` | `14.6%` | `-3.9 pts` |

Per-call kernel durations from the trace:

- `_block_attnres_state_bwd_kernel`: `767.7 us` → `559.5 us`
- `_block_attnres_state_fwd_kernel`: `420.8 us` → `362.0 us`

Register count dropped materially in the trace as well:

- backward: `255` → `181` regs/thread
- forward: `192.9` → `118.7` regs/thread

Occupancy reported in the trace stayed low (`~8%` achieved) on both runs, so the gain came primarily from the better tile shape and lower register pressure rather than an occupancy jump.

### Current overhead on 2xRTX5090

Using AttnRes-off `run-17031555` vs the latest retuned AttnRes-on `run-17031621`:

- step delta on thread 7: `+118.6 ms/step` (`+23.2%`)
- direct AttnRes kernels: `92.2 ms/step`

This is a clear improvement over the pre-retune tax:

- before retune: `+130.9 ms/step`, with `118.8 ms/step` in direct AttnRes kernels
- after retune: `+118.6 ms/step`, with `92.2 ms/step` in direct AttnRes kernels

### Takeaway

Unlike the earlier A100 Phase 0 work, SM120 retuning did have real headroom. The retune removed about `26.7 ms/step` from the direct AttnRes kernels and improved thread-7 step time by about `12.3 ms/step`, but the structure of the bottleneck is unchanged: the dominant remaining cost is still the repeated rereading of completed refs across sublayers.

---

## Next steps

Phase 0 confirmed that kernel-level micro-optimizations have minimal headroom on the existing backward structure. The path forward is the structural batching from the plan:

1. **Phase 1**: Batched completed-block D-reduction (forward only). Read completed refs once for all Q queries' logits/inv_rms. Low register pressure, straightforward kernel.

2. **Phase 2**: Online partial merge architecture. Establish the per-sublayer merge pattern so completed refs are never re-scanned after the batched pass.

3. **Phase 3**: Batched completed weighted sum. Fuse with Phase 1 into a single kernel. Query-pair batching at BLOCK_D=128 to manage register pressure.

4. **Phase 4**: Batched backward. Mirror the forward batching for the dominant 54 ms/step kernel.

Phase 1 is the right next step — it captures meaningful HBM savings with low risk and validates the batched kernel pattern before the more complex Phases 2-3.
