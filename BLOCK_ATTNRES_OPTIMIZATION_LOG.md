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

---

## Phase 1: Batched Completed-Block D-Reduction (Forward Only)

### What was implemented ✅ Complete

Phase 1 pre-computes the D-reduction (RMS norm + dot product → logits) for all completed refs in a single batched kernel launch, so each completed ref is read once across all Q sublayers in the block, instead of once per sublayer.

**New kernels** (`block_attnres_triton_kernels.py`):
- `_block_attnres_batched_completed_dreduction_kernel`: Reads each completed ref once, computes `inv_rms (NC, T)` (shared across all Q queries) and `logits (Q, NC, T)` (per-query). Uses `_load_qw_tile` helper with static if-chain dispatch for up to 8 query vectors. This kernel has no Pass 2, so it was never affected by the non-determinism bug.
- `_block_attnres_state_fwd_precomputed_kernel`: Same as the standard state forward kernel but skips Pass 1 for completed refs — loads precomputed logits/inv_rms instead. Only computes D-reduction for the partial source. Pass 2 (weighted sum) still reads completed sources per sublayer.
- `_load_qw_tile`: Static if-chain helper for loading one of up to 8 qw query vectors by constexpr index.

**New ops-layer functions** (`block_attnres_triton_ops.py`):
- `_pad_qw_list()`: Pads a list of qw vectors to length 8 for kernel pointer arguments.
- `batched_completed_dreduction()`: Allocates output tensors and launches the batched kernel.
- `_fused_block_attnres_state_precomputed_cuda()`: Launches the precomputed forward kernel.
- `_BlockAttnResStatePrecomputedFn`: Autograd Function (backward identical to standard path since saved alpha/inv_rms are the same).
- `fused_block_attnres_from_state_precomputed()`: Public API.

**Modeling layer additions** (`modeling.py`):
- `BlockAttnResOp.forward_state_precomputed()`: Dispatches to precomputed kernel.
- `BlockAttnResOp.compute_qw()`: Returns `(query * norm_weight).to(input_dtype)` for use by the batched kernel.
- `_BlockAttnResState`: Extended with `precomputed_logits`, `precomputed_inv_rms`, `precomputed_sublayer_idx` fields, plus `set_precomputed()`, `consume_precomputed()` methods. Precomputed data is cleared on `append_partial()`.
- `_block_attnres_block_qw_schedule()`: Pre-computes which `BlockAttnResOp` instances belong to each block. Raises `ValueError` if any block starts mid-layer (attn ends one block, MLP starts next in same layer).
- Both model forward loops (varlen and SDPA paths) wired to call `batched_completed_dreduction()` at block-starting layers when Triton is available, then each sublayer consumes precomputed data.

**Constraint**: Only layer-boundary block starts are supported. Mid-layer block starts raise `ValueError` with guidance to adjust `block_attnres_num_blocks`.

### SM120 non-determinism bug fix ✅ Fixed

During Phase 1 testing, the precomputed forward equivalence tests revealed non-deterministic results from all three forward kernels. Root-caused to a **Triton 3.6.0 compiler bug on SM120** where store-then-load to the same global address within a persistent-kernel loop can return stale values.

**Root cause evidence**:
- Alpha/inv_rms *outputs* (written once, never re-read) were always deterministic
- Only the *result* (Pass 2 weighted sum, which re-read alpha from `alpha_ptr`) was non-deterministic
- Bug occurred even with `NUM_SMS=1`, `num_stages=1`, no `flatten=True` — ruling out cross-SM races and pipelining
- Proven fix: keeping alpha in registers (never re-reading from `alpha_ptr`) made the kernel perfectly deterministic

**Fix applied to all three forward kernels**:
- `_block_attnres_fwd_kernel` (stacked, N sources)
- `_block_attnres_state_fwd_kernel` (NC completed + 1 partial)
- `_block_attnres_state_fwd_precomputed_kernel` (NC precomputed + 1 partial)

The fix uses 9 individual register variables (`logit_0` through `logit_8`, then `alpha_0` through `alpha_8`) with static if-chains to select by source index. Softmax (max, exp, sum, normalize) is computed entirely in registers. Alpha values are written to `alpha_ptr` as output for the backward pass but never re-read within the forward kernel. This is verbose but reliable, and the prototype (`_test_reg_alpha.py`) confirmed 20/20 deterministic runs.

**Backward kernels** have the same store→load pattern with `grad_alpha_ptr` and `qw_dot_ptr` but were not fixed here — they accumulate across D-tiles (not just sources) so the register approach is harder. This is noted for future work.

### Tests

All 38 tests pass (23 original + 15 new Phase 1 tests):
- `test_batched_dreduction_logits_inv_rms` — 9 parametrized cases (NC ∈ {1,2,4} × Q ∈ {1,3,8})
- `test_precomputed_forward_matches_standard` — 4 parametrized cases (NC ∈ {1,2} × Q ∈ {1,3})
- `test_precomputed_backward_matches_standard` — 2 parametrized cases (NC ∈ {1,3})

### Next steps

Phase 1 is complete. The batched D-reduction pattern is validated and wired end-to-end. The forward non-determinism bug is fixed for all forward kernels.

Remaining optimization phases from the plan:
1. **Phase 2**: Online partial merge — establish per-sublayer merge so completed refs are never re-scanned after the batched pass.
2. **Phase 3**: Batched completed weighted sum — fuse Pass 1 and Pass 2 for completed refs into a single kernel. Query-pair batching at BLOCK_D=128 to manage register pressure.
3. **Phase 4**: Batched backward — mirror the forward batching for the dominant backward kernel.

---

## Phase 2: Online Partial Merge Architecture

### What was implemented ✅ Complete

Phase 2 eliminates per-sublayer rescanning of completed refs in the forward pass by computing a running online-softmax state `(m, l, acc)` over completed refs once per block per query, then merging each sublayer's partial contribution with a lightweight merge kernel that only touches the partial source.

**Architecture**:
1. At block start: Phase 1's `batched_completed_dreduction()` computes logits/inv_rms (as before). Then `completed_wsum()` is called once per query to read all completed refs and produce the online-softmax running state `(running_m, running_l, running_acc)`.
2. Per sublayer: Instead of re-reading all NC completed refs for the weighted sum, `forward_state_online_merge()` only D-reduces the partial source and merges it into the precomputed running state. The merge is a lightweight elementwise kernel over 1 source.

**Online softmax update math**:
```
For each completed source j (in completed_wsum kernel):
  new_m = max(m, logit_j)
  correction = exp(m - new_m)
  exp_j = exp(logit_j - new_m)
  acc = acc * correction + exp_j * src_j    (per D-tile)
  l = l * correction + exp_j
  m = new_m

For partial merge (in merge_partial kernel):
  logit_p = inv_rms_p * dot(partial, qw)
  new_m = max(m_c, logit_p)
  correction = exp(m_c - new_m)
  exp_p = exp(logit_p - new_m)
  result = (acc * correction + exp_p * partial) / (l * correction + exp_p)
```

**Backward compatibility**: The merge-partial kernel reconstructs `alpha[j] = exp(logit_j - m_final) / l_final` for all NC+1 sources using precomputed logits. This produces identical alpha/inv_rms tensors to the standard forward, so the existing backward kernel `_fused_block_attnres_state_bwd_cuda` is reused as-is.

**New kernels** (`block_attnres_triton_kernels.py`):
- `_block_attnres_completed_wsum_kernel`: Persistent-loop kernel that reads NC completed refs with online softmax using precomputed logits from Phase 1. Outputs `(running_m (T,), running_l (T,), running_acc (T, D))` in f32. Uses global memory for the `(BLOCK_T, BLOCK_D)` accumulator between sources (same pattern as FlashAttention).
- `_block_attnres_merge_partial_kernel`: Persistent-loop kernel that D-reduces the partial source, merges into the running state, and outputs `result (T, D) bf16`, `alpha (NC+1, T) f32`, `inv_rms (NC+1, T) f32` for the backward pass.

**New ops-layer functions** (`block_attnres_triton_ops.py`):
- `completed_wsum()`: Allocates output tensors and launches the completed-wsum kernel.
- `_fused_block_attnres_merge_partial_cuda()`: Launches the merge-partial kernel.
- `_BlockAttnResOnlineMergeFn`: Autograd Function (backward identical to standard path since saved alpha/inv_rms match).
- `fused_block_attnres_online_merge()`: Public API.

**Modeling layer additions** (`modeling.py`):
- `BlockAttnResOp.forward_state_online_merge()`: Dispatches to the online-merge Triton kernel (NC ≤ 8) or falls back to the standard path.
- `_BlockAttnResState`: Extended with `online_running_states` field (list of per-query `(m, l, acc)` tuples), plus `set_online_running_states()` and `consume_online_merge()` methods. Running states are cleared on `append_partial()`.
- All three consumption sites (`_forward_block_attnres_full`, `_forward_block_attnres_local`, `_run_block_attnres_mlp`) now try Phase 2 online merge first, then fall back to Phase 1 precomputed, then to the base per-sublayer path.
- Both model forward loops (varlen and SDPA) compute `completed_wsum()` per-query at block start after `batched_completed_dreduction()`, storing the running states in the state object.

### Tests

All 47 tests pass (38 existing + 9 new Phase 2 tests):
- `test_completed_wsum_matches_reference` — 3 parametrized cases (NC ∈ {1, 3, 8})
- `test_online_merge_forward_matches_standard` — 4 parametrized cases (Q ∈ {1, 2} × NC ∈ {1, 3})
- `test_online_merge_backward_matches_standard` — 2 parametrized cases (NC ∈ {1, 3})

### HBM traffic analysis

Phase 2 changes the per-sublayer work from "D-reduce NC completed refs + weighted-sum NC+1 sources" to "D-reduce 1 partial + elementwise merge". The completed refs are now read Q times per block (once per query for the wsum pass) instead of Q times per block for both D-reduction and weighted-sum.

The dominant benefit is not total HBM reduction (Phase 1 already batched the D-reduction), but that the sequential per-sublayer work becomes trivially cheap — only 1 source instead of NC+1. Phase 3 will further reduce the completed-wsum reads from Q to ceil(Q/2) via query-pair batching.

### Next steps

Phase 2 is complete. The online-merge architecture is validated end-to-end with all kernels wired into the forward loops.

Remaining optimization phases from the plan:
1. **Phase 3**: Batched completed weighted sum — batch the `completed_wsum()` across query pairs to read completed refs ceil(Q/2) times instead of Q times per block.
2. **Phase 4**: Batched backward — mirror the forward batching for the dominant backward kernel.
3. **Phase 5**: `torch.compile` graph fragmentation fix — the Phase 1 batched D-reduction + Phase 2 completed_wsum calls at block start create graph breaks that add ~53 ms/step compile overhead. Deferred to Phase 5.

---

## Phase 3: Batched Completed Weighted Sum

### What was implemented ✅ Complete

Phase 3 batches the completed-ref weighted sum across query pairs (Q_BATCH=2 at BLOCK_D=128), so completed refs are read ceil(Q/2) times per block instead of Q times. This builds on Phase 2's online-merge architecture — the running state `(m, l, acc)` is produced by the new batched kernel instead of per-query `completed_wsum()` calls.

**Architecture decision — Option B (simpler, better)**: Instead of the plan's fused D-reduction+wsum kernel (which would read each source 2× per batch — once for D-reduction, once for wsum), we keep Phase 1's shared D-reduction (1 read total) and only batch the weighted sum (ceil(Q/2) reads). Total reads per source per block: `1 + ceil(Q/2)`. For Q=4: 3 reads vs the plan's 4. Simpler code and fewer HBM reads.

**Register pressure at Q_BATCH=2, BLOCK_D=128**: Each query holds a `(BLOCK_T=32, BLOCK_D=128) = 4096` f32 accumulator. Two queries = 8192 f32 values — same footprint as the single-query kernel at BLOCK_D=256. This is feasible and matches what was already proven to work.

**New kernel** (`block_attnres_triton_kernels.py`):
- `_block_attnres_batched_completed_wsum_kernel`: Processes Q_BATCH (1 or 2) queries simultaneously over NC completed sources. Each source tile is loaded once per (T-tile, D-tile) and shared across both queries in the batch. Uses `tl.constexpr Q_BATCH` with static `if Q_BATCH > 1` branches for the second query's online-softmax state. Accumulators are stored/loaded from global memory across D-tile iterations (same pattern as the single-query kernel and FlashAttention). For `j == 0`, accumulators are initialized directly (no load-then-correct). Persistent SM scheduling with `tl.range(pid, num_t_tiles, NUM_SMS, flatten=True)`.

**New ops-layer function** (`block_attnres_triton_ops.py`):
- `batched_completed_wsum()`: Processes Q queries in pairs, launching the batched kernel with BLOCK_D=128 for pairs and BLOCK_D=256 for singleton remainder. Returns a flat list of per-query `(running_m, running_l, running_acc)` tuples matching Phase 2's interface.

**Modeling layer changes** (`modeling.py`):
- Both forward loops (varlen ~line 2536, SDPA ~line 2706) now call `batched_completed_wsum()` instead of the per-query `completed_wsum()` loop. Comments updated to say "Phase 1+3". The import changed from `completed_wsum` to `batched_completed_wsum`.
- The backward path is unchanged — Phase 2's merge-partial kernel still reconstructs alpha/inv_rms for the existing backward kernel.

**SM120 non-determinism**: The batched kernel does NOT need the register-alpha workaround because it uses the online-softmax `(m, l, acc)` representation (same as Phase 2's completed_wsum kernel). Alpha values are never stored and re-loaded within the kernel.

### HBM traffic analysis

Per-block completed-ref reads across phases:

| Phase | D-reduction reads | Weighted-sum reads | Total per source |
|---|---|---|---|
| Baseline (pre-Phase 1) | Q reads | Q reads | 2Q |
| Phase 1 only | 1 read | Q reads | Q + 1 |
| Phase 1 + 2 (per-query wsum) | 1 read | Q reads | Q + 1 |
| **Phase 1 + 3 (batched wsum)** | **1 read** | **ceil(Q/2) reads** | **1 + ceil(Q/2)** |

For Q=4 (typical block with 4 sublayers): 3 reads vs 5 reads — a 40% reduction in completed-ref HBM traffic from the Phase 1+2 baseline.

### Tests

All 69 tests pass (47 existing + 22 new Phase 3 tests):
- `test_batched_completed_wsum_matches_per_query` — 12 parametrized cases (NC ∈ {1, 3, 8} × Q ∈ {1, 2, 3, 4}): verifies batched kernel output matches per-query PyTorch reference for all query counts and completed-ref counts.
- `test_batched_wsum_forward_matches_standard` — 8 parametrized cases (Q ∈ {1, 2, 3, 4} × NC ∈ {1, 3}): verifies full pipeline (dreduction → batched_wsum → merge) matches `forward_state()`.
- `test_batched_wsum_backward_matches_standard` — 2 parametrized cases (NC ∈ {1, 3}): verifies gradients through the batched wsum + merge path match the standard backward.

### Next steps

Phase 3 is complete. The forward path now reads each completed ref `1 + ceil(Q/2)` times per block instead of `Q + 1` times.

Remaining optimization phases from the plan:
1. **Phase 4**: Batched backward — mirror the forward batching for the dominant backward kernel.
2. **Phase 5**: `torch.compile` graph fragmentation fix — the Phase 1 batched D-reduction + Phase 3 batched completed_wsum calls at block start create graph breaks that add ~53 ms/step compile overhead. Deferred to Phase 5.

---

## Phase 4: Batched Backward (Kernel + Ops + Tests)

### Goal

Create a batched backward kernel that processes Q_BATCH=2 sublayers simultaneously, sharing completed-ref source tile reads in both Pass 1 (D-reduction) and Pass 2 (gradient writes). This mirrors Phases 1-3 which batched the forward path.

**Scope**: Kernel + ops wrapper + standalone tests only. Autograd integration is deferred — each sublayer's `torch.autograd.Function.backward` runs sequentially in reverse order and must return gradients immediately, making batching across backward calls architecturally hard.

### Implementation

#### Kernel: `_block_attnres_batched_state_bwd_kernel`

**Location**: `block_attnres_triton_kernels.py`, inserted between the Phase 2/3 forward kernels and the existing `_block_attnres_state_bwd_kernel`.

**Design**: Q_BATCH is a `tl.constexpr` (1 or 2). Each sublayer has its own `(grad_result, partial, qw, alpha, inv_rms)` inputs and `(grad_partial, R_local)` outputs. Completed refs are shared.

**Key optimizations**:
- **Pass 1 (D-reduction)**: For each D-tile, loads Q_BATCH `go_tile` and `qw_tile`, then for each completed source j, loads `src_tile` ONCE and accumulates Q_BATCH separate `grad_alpha[j]` and `qw_dot[j]`. Partial source handling is per-sublayer (not shared — each sublayer has a different partial).
- **Softmax backward**: Independent per sublayer (no sharing opportunity).
- **Pass 2 (grad write + R)**: For each D-tile, loads Q_BATCH `go_tile`, then for each completed source j, loads `src_tile` ONCE. Computes per-sublayer gradient contributions and **sums** them into a single `grad_completed` output tensor `(NC, T, D)`. Per-sublayer `grad_partial` and `R_local` are written separately.

**Register pressure**: At BLOCK_D=128, BLOCK_T=32, the additional live data for Q_BATCH=2 vs 1 is one extra `go_tile` (BLOCK_T × BLOCK_D = 4096 f32 values) plus one extra `qw_tile` (BLOCK_D = 128 f32 values). Total ~5 (BLOCK_T × BLOCK_D) tiles at Q_BATCH=2 — same footprint as one sublayer at BLOCK_D=256.

#### Ops wrapper: `batched_state_bwd()`

**Location**: `block_attnres_triton_ops.py`, inserted after `_fused_block_attnres_state_bwd_cuda` and before the Phase 1 section.

**Signature**:
```python
def batched_state_bwd(
    grad_results: list[torch.Tensor],      # Q_BATCH × (T, D)
    completed_refs: tuple[torch.Tensor, ...], # NC × (T, D) — shared
    partials: list[torch.Tensor],           # Q_BATCH × (T, D)
    queries: list[torch.Tensor],            # Q_BATCH × (D,)
    norm_weights: list[torch.Tensor],       # Q_BATCH × (D,)
    alphas: list[torch.Tensor],             # Q_BATCH × (NC+1, T)
    inv_rmss: list[torch.Tensor],           # Q_BATCH × (NC+1, T)
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    # Returns: grad_completed (NC, T, D), grad_partials list, Rs list
```

For Q_BATCH=1, sublayer 1 pointers are passed as dummies (sublayer 0's tensors). The kernel's `tl.constexpr` branching ensures sublayer 1 code is dead-eliminated at compile time.

### Tests

12 new tests in `TestBatchedStateBwd` class:

- `test_batched_bwd_matches_per_sublayer` — 6 parametrized cases (Q_BATCH ∈ {1, 2} × NC ∈ {1, 3, 8}): verifies batched kernel output matches sum of per-sublayer backward for grad_completed, and per-sublayer grad_partial/R match individually.
- `test_batched_bwd_grad_completed_sum` — 3 parametrized cases (NC ∈ {1, 3, 8}): verifies the summed grad_completed from the batched kernel (Q_BATCH=2) equals the sum of two individual `_fused_block_attnres_state_bwd_cuda` calls, checked per completed source.
- `test_batched_bwd_various_dims` — 3 parametrized cases (D ∈ {64, 256, 768}): verifies correctness across dimension sizes that exercise different BLOCK_D tile counts.

All 81 tests pass (69 existing + 12 new).

### Next steps

Phase 4 kernel and ops are complete. Remaining work:
1. **Autograd integration** (deferred): Requires either a block-level autograd Function or a shared-state backward batcher using hooks. Both have significant implementation risk due to sequential backward dependencies.
2. ~~**Phase 5**: `torch.compile` graph fragmentation fix.~~ → Done ✅

---

## Phase 5: Fix torch.compile Graph Fragmentation ✅

### Problem

Phase 1-3 ops (`batched_completed_dreduction`, `batched_completed_wsum`, and
`_BlockAttnResOnlineMergeFn`) were plain Python functions launching Triton
kernels directly — NOT registered as `torch.library` custom ops with FakeTensor
support. `torch.compile`/Dynamo could not trace through them, causing graph
breaks. Profiling showed SM120 baseline 630.1 ms/step → 699.3 ms/step (+69.2
ms), where AttnRes kernels improved by 2.7 ms but non-AttnRes overhead
increased by 71.9 ms due to graph fragmentation.

### Strategy

Registered ALL state-path operations as `torch.library` custom ops with
FakeTensor implementations and `register_autograd`, replacing the three
`torch.autograd.Function` subclasses (`_BlockAttnResStateFn`,
`_BlockAttnResStatePrecomputedFn`, `_BlockAttnResOnlineMergeFn`).

### Changes

**`block_attnres_triton_ops.py`** — Full rewrite of dispatch layer:

1. **7 `_lib.define()` calls** for all ops: `fused_block_attnres`,
   `fused_block_attnres_bwd`, `fused_block_attnres_state`,
   `fused_block_attnres_state_bwd`, `fused_block_attnres_state_precomputed`,
   `fused_block_attnres_merge_partial`, `batched_completed_dreduction`,
   `batched_completed_wsum`. Variable-arity `completed_refs` uses `Tensor[]`
   (TensorList) in schemas.

2. **8 `@register_fake` FakeTensor implementations** — correct output
   shapes/dtypes for torch.compile tracing.

3. **7 CUDA implementations via `@torch.library.impl`** — existing kernel
   launch code adapted to receive `list[torch.Tensor]` instead of tuples.

4. **5 `register_autograd()` calls** replacing the 3 `autograd.Function`
   subclasses:
   - `fused_block_attnres` — stacked path (existing, now registered)
   - `fused_block_attnres_state` — Phase 0 state path
   - `fused_block_attnres_state_precomputed` — Phase 1
   - `fused_block_attnres_merge_partial` — Phase 2
   - `batched_completed_dreduction` — no-op backward (constants, no grad)
   - `batched_completed_wsum` — no-op backward (constants, no grad)

   The batch ops needed `setup_context` to save `len(completed_refs)` and
   return `[None] * n` lists matching the `Tensor[]` input tree structure
   (PyTorch's `supports_tensorlist` wrapper requires matching tree specs).

5. **5 public API wrappers** dispatch through `torch.ops.nanoplm_bar.*` —
   signatures unchanged, internal routing changed from
   `AutogradFunction.apply()` to `torch.ops` calls.

**`test_block_attnres.py`** — Added `TestCompileCompat` class:

5 tests verifying `torch.compile(fullgraph=True)` traces each public API
function into a single graph with no breaks:
- `test_state_forward_no_graph_break`
- `test_state_precomputed_forward_no_graph_break`
- `test_online_merge_forward_no_graph_break`
- `test_batched_dreduction_no_graph_break`
- `test_batched_wsum_no_graph_break`

### Key discoveries

- `Tensor[]` backward must return `[None] * n` (list), not bare `None`, to
  match the pytree input spec. Returning `None` causes `RuntimeError: Expected
  the return from backward to be of the same structure as the inputs`.

- `_pad_completed_refs` signature updated from `tuple[torch.Tensor, ...]` to
  `tuple[torch.Tensor, ...] | list[torch.Tensor]` since custom ops receive
  lists.

- `batched_completed_wsum` Python loop (processing Q queries in pairs) is kept
  inside the CUDA impl — opaque to the compiler. The custom op returns stacked
  tensors `(Q, T)`, `(Q, T)`, `(Q, T, D)`, unpacked to `list[tuple]` in the
  public wrapper.

### Tests

86 tests pass (81 existing + 5 new compile-compat). Zero autograd UserWarnings
(previously 8 warnings about missing Autograd dispatch key).

### Files changed

- `src/nanoplm/pretraining/models/modern_bert/block_attnres_triton_ops.py`
- `tests/test_block_attnres.py`

---

## Phase 5 Follow-up: Fix Dynamo Recompile Limit Hit

### Problem

After Phase 5, profiling (`run-17032008-2`) revealed 498 recompile events during
warmup and — critically — `_run_block_attnres_mlp` **hit Dynamo's default
`cache_size_limit` of 8** (`[21/8] torch._dynamo hit config.recompile_limit`).
After hitting the limit, Dynamo falls back to eager execution for that function
for all subsequent calls, eliminating compiled-graph speedups.

Root cause: the block attnres state dataclass (`_BlockAttnResState`) carries
mutable Python integers (`num_completed`, `precomputed_sublayer_idx`) that Dynamo
specializes on. With 6 blocks of 4 sublayers each, functions like
`_run_block_attnres_mlp` encounter up to ~12 distinct guard combinations — well
above the default limit of 8.

Guard conditions triggering recompiles per function:

| Function | Guard variables | Recompiles needed |
|---|---|---|
| `_run_block_attnres_mlp` | `num_completed`, `precomputed_sublayer_idx`, `mlp_ends_block` | 8+ (hit limit) |
| `_forward_block_attnres_local` | `num_completed`, `partial_block.requires_grad` | 7 |
| `batched_completed_dreduction` | `len(completed_refs)` | 5 |
| `batched_completed_wsum` | `len(completed_refs)` | 5 |
| `resume_in__run_block_attnres_mlp_at_1716` | `mlp_ends_block`, `num_completed` | 4 |

### Fix

Replaced `_maybe_expand_dynamo_cache_for_batch_warmup` in `pure_pipeline.py`
with a generalised `_maybe_expand_dynamo_cache` that raises
`torch._dynamo.config.cache_size_limit` for both batch-size warmup AND block
attnres. For 6 blocks: `max(16, 4 × 6) = 24`. The previous limit of 8 was
unchanged when `batch_size_warmup_steps=0` (the typical config).

Formula: `max(16, 4 × num_blocks)` — empirically safe for the combinatorial
product of `num_completed × sublayer_idx × ends_block` guard dimensions.

### Files changed

- `src/nanoplm/pretraining/pure_pipeline.py` — renamed+generalised function,
  updated call site to pass `use_block_attnres` and `block_attnres_num_blocks`
  from the model config.

### Tests

All 86 tests still pass.
