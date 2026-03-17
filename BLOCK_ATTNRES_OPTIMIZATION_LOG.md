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
- `block_attnres_triton_ops.py`: `R_local (NUM_SMS, D)` allocation, `.sum(dim=0)` after kernel. `_pick_block_attnres_bwd_meta` returns 4-tuple with BLOCK_D. Added `_next_power_of_2` helper.

All 23 tests in `test_block_attnres.py` pass.

---

## Next steps

Phase 0 confirmed that kernel-level micro-optimizations have minimal headroom on the existing backward structure. The path forward is the structural batching from the plan:

1. **Phase 1**: Batched completed-block D-reduction (forward only). Read completed refs once for all Q queries' logits/inv_rms. Low register pressure, straightforward kernel.

2. **Phase 2**: Online partial merge architecture. Establish the per-sublayer merge pattern so completed refs are never re-scanned after the batched pass.

3. **Phase 3**: Batched completed weighted sum. Fuse with Phase 1 into a single kernel. Query-pair batching at BLOCK_D=128 to manage register pressure.

4. **Phase 4**: Batched backward. Mirror the forward batching for the dominant 54 ms/step kernel.

Phase 1 is the right next step — it captures meaningful HBM savings with low risk and validates the batched kernel pattern before the more complex Phases 2-3.
