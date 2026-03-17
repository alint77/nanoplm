# Block AttnRes Optimization Plan (Final)

## Context

Baseline trace: `run-17031118-4`, thread `7` (main GPU compute), 4x A100-80GB PCIe, FSDP sublayer sharding, `torch.compile`.

Steady-state numbers from `ProfilerStep#11-#14`:

- Step time: about `342.8 ms/step`
- Real GPU work on thread `7`: about `321.6 ms/step`
- Dominant Block AttnRes kernels:
  - `_block_attnres_state_fwd_kernel`: about `19.5 ms/step`
  - `_block_attnres_state_bwd_kernel`: about `54.0 ms/step`
- Combined state-kernel cost: about `73.5 ms/step`
- If surrounding AttnRes-related bf16 add/update work is included, the practical AttnRes tax is closer to `~80 ms/step`

Model/config:

- `hidden_size=768`
- `num_hidden_layers=12`
- `block_attnres_num_blocks=6`
- block sizes: `(4, 4, 4, 4, 4, 4)` sublayers per block
- therefore `Q=4` sublayers per block
- max completed refs during forward: `NC=7` (`embedding + 6 completed blocks`)

## Core diagnosis

The remaining AttnRes cost is now mostly real memory traffic, not wrapper overhead.

The dominant structural waste is:

- **repeated rereads of completed-block tensors across sublayers inside a block** — completed refs are static within a block, but the current schedule scans them independently per sublayer

Secondary:

- the existing backward kernel has concrete inefficiencies (atomic contention, unnecessary global scratch traffic) that can ship immediately without any schedule redesign

## HBM traffic model

Per block with `Q` sublayers and `NC` completed refs, for bf16 source tensors of shape `(T, D)`:

| Approach | Completed reads | Partial reads | Total | vs current (NC=5) |
|---|---|---|---|---|
| Current | `2Q × NC × TD` | `2Q × TD` | `(8NC + 8)TD` | `1.0×` — `48TD` |
| Phase 1: batched completed D-reduction | `(Q+1) × NC × TD` | `2Q × TD` | `(5NC + 8)TD` | `1.45×` — `33TD` |
| Phase 2+3: + online merge + batched completed wsum | `2 × NC × TD` | `2 × TD` | `(2NC + 2)TD` | `4.0×` — `12TD` |

Key observations:

- completed-ref traffic dominates — at `NC=5`, it is `83%` of block traffic
- partial handling is only `17%` of block traffic, but online merge eliminates it as a bonus
- the `4×` end-state requires all of Phases 1-3 landed and fused

## Phase 0: Immediate Backward Kernel Cleanup

### Objective

Improve `_block_attnres_state_bwd_kernel` without changing the schedule or model API. This is the largest single AttnRes kernel cost (`54 ms/step`, 17% of step time) and the lowest-risk optimization surface.

### 0a. Remove `R_ptr` atomic contention

Current issue:

- every SM atomically adds into the same `R_ptr (D,)` buffer
- this creates avoidable contention and L2 line bouncing across all 108 SMs
- with 3 D-tiles per T-tile, that is 324 overlapping atomic operations per step

Implementation:

1. In `block_attnres_triton_ops.py`, allocate `R_local` with shape `(NUM_SMS, D)` instead of one shared `(D,)` tensor.
2. In the backward kernel, write each program's `R_chunk` into `R_local[pid]` — a plain `tl.store`, no atomics.
3. Launch a small reduction kernel afterward to sum `R_local` rows into final `R`.
4. Cost: `108 × 768 × 4 = 324 KB` extra memory (negligible).

Expected benefit: `5-15%` of backward kernel time (`3-8 ms/step`).

### 0b. Reduce scratch traffic carefully

Current issue:

- pass 1 of the backward kernel uses global scratch buffers for `grad_alpha` and `qw_dot` — shape `(NC+1, T)` each
- within each T-tile, the D-tile loop loads, accumulates, and stores these per source per D-tile iteration
- this traffic is purely temporary within a single T-tile's lifetime

Important constraints:

- the current backward kernel already runs at high register pressure
  - `(BLOCK_T, BLOCK_D)` = `(64, 256)` source/grad tiles: `16384` values each → `64` registers per thread at 256 threads
  - two such tiles loaded simultaneously → `128` registers per thread before accumulators and scalars
- a fully register-resident rewrite may spill or reduce occupancy

However, even if register count stays the same, eliminating the per-D-tile scratch round-trips frees HBM bandwidth in the inner loop. The win may come from traffic reduction rather than register savings.

Recommended implementation:

1. Prototype a local-accumulator version at smaller `BLOCK_T` (e.g., 32) where register pressure is lower.
2. Measure both register count and runtime before switching the default.
3. Be willing to combine with a `BLOCK_T` reduction or split-pass design.

Expected benefit: potentially useful, but must be measured. Do not assume this is free.

### 0c. Re-sweep backward tuning

Current issue:

- forward and backward have different register-pressure profiles
- backward likely wants a different `BLOCK_T` / `BLOCK_D` operating point

Benchmark on real training shapes:

- `BLOCK_T in {16, 32, 64, 128}`
- `BLOCK_D in {64, 128, 256}` where register budget allows

Validate against: kernel time, register count, achieved occupancy.

### 0d. Validation

- `tests/test_block_attnres.py` green
- no schedule or model API change
- fresh trace against `run-17031118-4`

### Expected outcome

Combined 0a + 0b + 0c: about `9-15 ms/step` reduction. Target: backward kernel from `~54 ms` to `~40-45 ms`.

---

## Phase 1: Batched Completed-Block D-Reduction (Forward Only)

### Objective

Read each completed ref once for all `Q` queries' D-reduction work, instead of once per sublayer.

### Why this is the right first structural change

For completed refs:

- `inv_rms` depends only on the source, not the query — it is shared across all `Q` sublayers
- `dot(src, qw)` differs by query, but all queries share the same source tile load

The D-reduction (pass 1) batches naturally: one source tile load, `Q` cheap dot-product accumulations on top. No register-pressure concern because the per-query accumulators are `(BLOCK_T,)` scalars, not `(BLOCK_T, BLOCK_D)` tiles.

### Register pressure analysis

Per source tile load, we add:

- `sum_sq` contribution: shared, `1 × (BLOCK_T,)` accumulator
- `dot_qw_q` contribution: per query, `Q × (BLOCK_T,)` accumulators

For `Q=4`, `BLOCK_T=32`: `4 × 32 + 32 = 160` floats for scalar accumulators. Trivial. The source tile `(BLOCK_T, BLOCK_D)` dominates register usage regardless. No `BLOCK_D` reduction needed.

### Kernel design

```text
_block_attnres_batched_completed_dreduction_kernel(
    completed_refs...,      # NC separate (T, D) tensors
    qw_ptrs...,             # Q separate (D,) vectors (precomputed query × norm_weight)
    logits_ptr,             # (Q, NC, T) output
    inv_rms_ptr,            # (NC, T) output — shared across queries
    T, D, NC, Q,
    BLOCK_T, BLOCK_D, NUM_SMS,
)
```

Per T-tile, for each completed source `j`:

1. Loop over D-tiles: load `src_tile` once, accumulate shared `sum_sq` and `Q` dot products.
2. Compute `inv_rms_j` (shared), `logits[q][j] = inv_rms_j × dot_q` for each query `q`.
3. Store to global output.

### Integration

At the start of each block in `modeling.py`:

1. Gather the block's `Q` query/norm_weight pairs for all sublayers.
2. Launch the batched completed-ref D-reduction once.
3. Each sublayer's `forward_state` receives precomputed logits and inv_rms for the completed portion, and only needs to:
   - compute the partial-block D-reduction (1 source, unchanged)
   - softmax over `NC + 1` logits (`NC` precomputed + 1 partial)
   - weighted sum (pass 2, unchanged — still reads completed sources per sublayer)

### What this does not change

- completed weighted sum is still per-sublayer (pass 2 still reads completed sources `Q` times)
- partial handling is unchanged
- backward is unchanged

### HBM savings

- Pass 1 completed reads: from `Q × NC × TD` to `1 × NC × TD`
- Pass 2 completed reads: unchanged at `Q × NC × TD`
- Total: from `2Q × NC × TD` to `(Q+1) × NC × TD`
- For `Q=4`, `NC=5`: from `40TD` to `25TD` — `1.6×` reduction on completed traffic

### Validation

- Add forward-equivalence tests: batched D-reduction logits/inv_rms must match per-sublayer computation to bf16 tolerance.
- Full `test_block_attnres.py` green.
- Reprofile.

---

## Phase 2: Online Partial Merge Architecture

### Objective

Introduce the scheduling change that eliminates per-sublayer rescanning of completed refs, by merging each sublayer's partial contribution into precomputed completed-block running state via online softmax.

### Why this must come before batched weighted sum

There is a hard sequential dependency: `partial_block[q]` depends on the output of sublayer `q-1`, which itself depends on `forward_state` at sublayer `q-1`. You cannot batch the completed weighted sum across queries until online merge exists, because the softmax weights over `NC+1` sources depend on per-sublayer partial contributions that do not exist at batch time.

The correct ordering is:

1. Batch the static completed-block work (Phase 1 D-reduction, Phase 3 weighted sum).
2. Keep the sequential partial evolution explicit.
3. Merge partial contributions online as they become available.

This matches the paper's two-phase inference strategy (Section 4).

### Running-state design

The completed-block pass (Phase 3, or temporarily per-sublayer) produces per-query running state:

- `m_q`: running max logit `(T,)` — max over completed-source logits
- `l_q`: running sum-exp `(T,)` — sum of `exp(logit_j - m_q)` over completed sources
- `acc_q`: weighted accumulator `(T, D)` — `Σ_j exp(logit_j - m_q) × src_j`

After each sublayer produces a new `partial_block`, compute the partial contribution and merge:

```text
logit_p = inv_rms_p × dot(partial_block, qw_q)
new_max = max(m_q, logit_p)
correction = exp(m_q - new_max)
exp_p = exp(logit_p - new_max)
acc_q = acc_q × correction + exp_p × partial_block
l_q = l_q × correction + exp_p
m_q = new_max
result_q = acc_q / l_q
```

### Per-sublayer API change

Current per-sublayer call:

```python
h = forward_state(completed_refs, partial_block)  # full scan of all sources
```

After Phase 2:

```python
# Once per block (before sublayer loop):
running_state = completed_block_pass(completed_refs, qw_list)

# Per sublayer:
h, running_state = merge_partial(running_state[q], partial_block, qw_q)
out = sublayer(h)
partial_block = partial_block + out
```

This changes the `autograd.Function` boundary from per-sublayer to per-block for the completed portion. The merge is a lightweight per-sublayer kernel over 1 source.

### Complexity concern

The `acc_q × correction` rescaling touches all `(T, D)` elements when the max changes. This is a cheap elementwise operation but adds a kernel launch per sublayer per block. In practice, since partial logits are typically smaller than the max completed logit, the correction factor is often `1.0` and the rescale is a no-op (the branch can be skipped).

### Validation

- Forward equivalence against current per-sublayer state path.
- Gradient agreement (backward still uses old per-sublayer path at this stage).
- Reprofile.

---

## Phase 3: Batched Completed Weighted Sum (Forward)

### Objective

Batch the completed-ref weighted sum (pass 2) across queries, so completed refs are read only twice total per block: once for D-reduction (Phase 1), once for weighted sum (this phase). The output feeds directly into the per-query online-merge running state from Phase 2.

### Why this depends on Phase 2

The weighted sum uses softmax weights that depend on all source logits including partial. With online softmax (Phase 2), the completed-block pass computes a partial softmax over just the completed sources, producing running `(m, l, acc)` state. The per-sublayer merge then incorporates the partial contribution. Without Phase 2's online-merge design, you would need final softmax weights — which require the partial logit — making cross-query batching impossible.

### Register pressure constraint

Batching weighted sum across `Q` queries needs `Q` independent `(BLOCK_T, BLOCK_D)` accumulators.

- At `BLOCK_D=256`: `Q × 32 × 256 = 32768` f32 values — exceeds register capacity
- At `BLOCK_D=128` with `Q=2` pairs: `2 × 32 × 128 = 8192` values — feasible
- At `BLOCK_D=64` with `Q=4`: `4 × 32 × 64 = 8192` values — feasible but 12 D-tile iterations

### Recommended approach: query-pair batching

Process 2 queries at a time with `BLOCK_D=128`:

- `ceil(Q/2) = 2` passes over completed sources
- reads completed sources `2 × 2 = 4` times total instead of `Q × 2 = 8`
- much safer register footprint than full `Q=4` batching at `BLOCK_D=64`
- if benchmarks show headroom, try full `Q=4` at `BLOCK_D=64` afterward

### Kernel design

```text
_block_attnres_batched_completed_fwd_kernel(
    completed_refs...,          # NC separate (T, D)
    qw_ptrs...,                 # Q_BATCH separate (D,), where Q_BATCH ≤ Q
    running_max_ptr,            # (Q_BATCH, T) output
    running_sum_exp_ptr,        # (Q_BATCH, T) output
    running_acc_ptr,            # (Q_BATCH, T, D) output
    T, D, NC, Q_BATCH,
    BLOCK_T, BLOCK_D=128, NUM_SMS,
)
```

Single kernel, two passes per source with online softmax:

- **Pass 1** (D-reduction): For each completed source, load source tiles once, compute shared `inv_rms` and `Q_BATCH` logits. Update per-query `(m, l)` running state.
- **Pass 2** (weighted accumulation): For each completed source, load source tiles once, accumulate into `Q_BATCH` weighted-sum accumulators with online-softmax scaling.

Output: per-query `(m_q, l_q, acc_q)` running state ready for Phase 2's partial merge.

### Fusion end-state

In the final architecture, this kernel subsumes Phase 1's D-reduction kernel. Phase 1 exists as a stepping stone — once Phase 3 lands, the separate D-reduction kernel is retired and the fused kernel handles both passes. Completed refs are read exactly twice per block total: once in pass 1 (D-reduction), once in pass 2 (weighted accumulation).

### HBM savings (query-pair batching)

- Completed reads: from `2Q × NC × TD` to `2 × ceil(Q/2) × NC × TD = 4 × NC × TD`
- For `Q=4`, `NC=5`: from `40TD` to `20TD` — `2×` reduction on completed traffic
- Full `Q=4` batching at `BLOCK_D=64` would reach `2 × NC × TD = 10TD` — `4×`

### Validation

- Forward equivalence against current per-sublayer state path.
- Gradient agreement (backward still uses old per-sublayer path at this stage).
- Reprofile.

---

## Phase 4: Batched Backward

### Objective

Build the backward counterpart to the blockwise forward (Phases 1-3). This is the highest-impact phase because backward is `3.1×` the forward cost, but also the hardest.

### Why this is last

The backward must handle:

- batched completed-block gradients (mirror of Phase 3)
- partial-block gradients through the online-merge chain (reverse of Phase 2)
- parameter-gradient accumulation (already improved by Phase 0a)

The forward must be stable before the backward can target it.

### Recommended split

**4a. Batched completed-block backward**

Mirrors Phase 3's forward. Read each completed source twice total for all query gradients:

- Pass 1: compute per-query `grad_logit`, `grad_alpha`, `grad_inv_rms` from saved `(alpha, inv_rms)`.
- Pass 2: compute `grad_src` for each completed source (accumulated across queries), and `grad_qw` per query.

Same register-pressure constraints as forward Phase 3 — use query-pair batching at `BLOCK_D=128`.

**4b. Partial-block backward (online merge reverse)**

Reverse the merge updates from Phase 2. For each sublayer `q` in reverse order:

- Given saved `(m_q, l_q)` at each merge point and the partial logit.
- Compute `grad_partial_block` and `grad_running_state` through the correction factor `exp(m_old - m_new)` and the division by `l_q`.

The gradient through the correction factor is:

```text
d_correction/d_m_old = correction  (since correction = exp(m_old - m_new))
d_acc_new/d_acc_old = correction
d_acc_new/d_partial = exp_p
```

Mathematically tractable but requires saving per-merge-point `(m_q, l_q)` — `Q × 2 × T` extra scalars per block.

### Forward intermediates to save

| Tensor | Shape | Purpose |
|---|---|---|
| `alpha` | `(Q, NC+1, T)` | softmax weights per query per source |
| `inv_rms` | `(NC, T)` | shared RMS normalization (completed only) |
| `merge_max` | `(Q, T)` | running max before each partial merge |
| `merge_sum_exp` | `(Q, T)` | running sum-exp before each partial merge |
| `partial_logits` | `(Q, T)` | partial source logit at each merge |

### Validation

- Gradient agreement against current per-sublayer state backward path across all block sizes and `NC` values.
- Extended `tests/test_block_attnres.py` covering uneven blocks and final aggregation.
- Fresh profiler trace.

---

## Phase 5: Compile Stability

### Objective

Stabilize the final blockwise schedule under `torch.compile`.

### Why this is last

Most compile cleanup should happen after the blockwise scheduler shape is settled. Otherwise, that work gets invalidated by later architectural changes.

### Known remaining issues

- boundary booleans causing graph breaks
- changing `num_completed` as Python integer state
- residual scheduler polymorphism

### Approach

1. Make the block executor (from Phase 3 forward) the primary compiled unit.
2. Move boundary decisions outside the compiled region.
3. Avoid compiling over changing Python integer state like `num_completed` — either make it a tensor or use separate compiled paths per block shape.

This is important for startup and graph stability, but is not the main steady-state throughput lever.

---

## Delivery Summary

| Phase | Description | Effort | Target |
|---|---|---|---|
| **0** | Backward kernel cleanup (atomic_add, scratch, BLOCK_T) | 2-3 days | backward `54 → ~40-45 ms` |
| **1** | Batched completed D-reduction (forward only) | ~1 week | forward completed pass-1 reads `÷ Q` |
| **2** | Online partial merge architecture | ~1 week | eliminate per-sublayer completed rescanning |
| **3** | Batched completed weighted sum (forward, fuses with Phase 1) | ~1 week | completed reads `2× (pair)` to `4× (full)` reduction |
| **4** | Batched backward | 2-3 weeks | backward completed reads `2-4×` reduction |
| **5** | Compile stability | as needed | startup / recompilation cleanup |

## Validation Plan

At every phase:

1. `tests/test_block_attnres.py` must stay green.
2. Add forward-equivalence tests against the current state path for all source counts, uneven block shapes, and final aggregation.
3. Add gradient agreement tests before changing defaults.
4. Capture a fresh profiler trace and compare thread `7` against `run-17031118-4`.
5. Recompute inferred HBM traffic from the actual kernel launch pattern and compare against estimates in this document.

## Success Metrics

Primary metric: thread-7 time spent in AttnRes kernels (forward + backward combined).

| Milestone | State-kernel target | Total AttnRes target |
|---|---|---|
| Baseline | `73.5 ms/step` | `~80 ms/step` |
| After Phase 0 | `~59-64 ms/step` | `~65-71 ms/step` |
| After Phase 3 | `~50-55 ms/step` | `~57-62 ms/step` |
| After Phase 4 | `~25-35 ms/step` | `~30-40 ms/step` |

End-state target: `2-2.7×` total AttnRes reduction, bringing combined cost to `~30-40 ms/step`.

## Recommended Immediate Next Task

Start with Phase 0.

Reason:

- it is low risk
- it directly targets the single largest kernel (`54 ms/step`)
- it improves the current implementation even if the larger schedule redesign takes time

After that, move to Phase 1 (batched completed-block D-reduction forward), then Phase 2 (online merge architecture) which unlocks Phase 3.
