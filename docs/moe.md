# Mixture-of-Experts: Design and Implementation

## Purpose

This document is the single source of truth for nanoPLM's MoE work.

It does three jobs:

1. Summarizes the research path across public MoE models.
2. Explains why Ling-mini-2.0 / Ling-V2 became the main inspiration point.
3. Documents the current nanoPLM implementation and the reasons behind its design choices.

## Scope

nanoPLM's MoE exists only in the pure-torch path.

Requirements:

- `use_moe: true`
- `use_packing: true`
- `use_static_inp_size: true`

Not supported:

- HF ModernBERT path
- Transformer Engine path
- capacity-limited / token-dropping MoE
- expert parallel / all-to-all dispatch

The implementation is intentionally "single-program, dropless, grouped-GEMM MoE", not a distributed expert-parallel MoE system.

## Research Path

The original goal was not just "add top-k experts". The goal was to understand what modern strong MoEs are actually doing, especially in routing, load balancing, and small-scale stability.

The main public references examined were:

- DeepSeek-V3
- Qwen3-MoE
- Qwen3-Next
- Qwen3.5
- GLM-4.5
- GLM-5
- Ling-mini-2.0 / Ling-V2

### What the research consistently showed

Across the strongest recent MoEs, the design center has shifted away from old Switch-style recipes.

The recurring frontier pattern is:

- dropless routing
- top-k expert selection
- one shared expert or a shared dense path
- bias-corrected routing or another explicit load-balancing mechanism
- strong router stabilization
- communication-aware routing restrictions at large expert count

The most important architectural lesson was that modern good MoEs are not just "dense FFN, but sparse". They are routing systems first and MLP systems second.

### DeepSeek / GLM family

DeepSeek-V3 and the GLM-4.5 / GLM-5 family are especially important because they expose a coherent routing style:

- `sigmoid` routing rather than softmax
- top-k selection
- one shared expert
- expert-selection bias correction
- no token dropping
- z-loss or equivalent router stabilization

This family treats load balancing as a routing-control problem rather than as a pure auxiliary-loss problem.

That was the biggest conceptual shift that influenced nanoPLM.

### Qwen family

Qwen3-MoE, Qwen3-Next, and Qwen3.5 were useful mainly as a contrast class.

Their public implementations emphasize:

- `softmax` top-k routing
- more explicit auxiliary balancing
- in Qwen3-Next / Qwen3.5, a gated shared-expert branch

They are strong references, but less attractive as a direct blueprint for nanoPLM because the public code centers less on bias-corrected sigmoid routing and more on the Qwen-specific hybrid backbone / softmax router family.

### Why Ling mattered more than the others

Ling-mini-2.0 ended up being the most relevant reference for nanoPLM, not because it is the biggest or most famous MoE, but because it is the best public example of a **serious small-activation MoE**.

That matters for nanoPLM because we care much more about the small-to-mid model regime than about 200B-plus frontier cluster designs.

Ling is unusually valuable because it combines:

- a scaling-law paper focused on MoE architecture choice
- a released checkpoint/config in a relatively modest size regime
- public model code
- public training patches and scripts

That combination is rare.

## Why Ling Became the Anchor Reference

Ling-mini-2.0 is compelling for nanoPLM for four reasons.

### 1. It lives in a transferable size regime

Most public MoE systems are so large that many of their choices only make sense because the cluster budget is enormous.

Ling is different. It demonstrates a sparse MoE recipe at a scale where:

- routing instability still matters
- hidden size is not massive
- expert count still changes behavior materially
- systems constraints still look somewhat like an advanced single-node project

That makes Ling far more transferable to nanoPLM than a giant model whose stability comes partly from sheer scale.

### 2. The paper is about architecture choice, not only a checkpoint

The Ling scaling-law paper's main message is that:

- expert activation ratio is the dominant MoE efficiency lever
- expert granularity has an optimum rather than "bigger is always better"
- shared-expert ratio and leading dense layers matter, but as secondary knobs

That framing is exactly what a project like nanoPLM needs.

It helps answer:

- how many experts should a small model try?
- how small should each expert be?
- how much dense path should remain?

### 3. The code reveals real training handling

Ling's open training stack shows details many releases hide:

- router kept in fp32
- expert-bias updates
- zero-mean bias update variant
- group-limited routing
- small z-loss

Those are not cosmetic details. They are exactly the kind of decisions that determine whether a small MoE trains cleanly or collapses.

### 4. It is close to the DeepSeek-style routing family

Ling is not a copy of DeepSeek-V3, but it is clearly in the same design neighborhood:

- sigmoid router
- selection-only bias correction
- shared expert
- group-limited routing
- router stabilization

That gave nanoPLM a much clearer path than the Qwen-style softmax route.

## What nanoPLM took from the research

The current nanoPLM MoE is best understood as:

- structurally inspired by DeepSeek / GLM / Ling
- scoped down to a single-node pure-torch implementation
- adapted for small-model experimentation

The key design decisions that came directly out of the research are:

- **Sigmoid top-k routing**
  Ling, DeepSeek, and GLM all reinforced that sigmoid routing is a good fit for sparse expert selection.

- **One always-on shared expert**
  This preserves a dense path for token-generic processing and reduces the brittleness of fully routed sparse MLPs.

- **Bias-corrected routing**
  Expert usage is nudged through a selection-only bias rather than relying only on a classic aux loss.

- **Router kept numerically conservative**
  Router logits are handled in fp32 and regularized with z-loss.

- **Leading dense layers**
  The first few layers can remain dense because early token processing is less specialization-heavy.

- **Dropless grouped-GEMM dispatch**
  grouped GEMM allows jagged token counts per expert without token dropping or padding to capacity.

- **Optional group-limited routing**
  This gives a path toward higher expert counts without turning routing into a fully unconstrained search.

## Where nanoPLM intentionally differs from Ling

Ling is the main inspiration, not the blueprint.

The current implementation differs in a few deliberate ways:

- **No expert-parallel / DeepEP stack**
  nanoPLM stays single-program and FSDP-sharded. There is no all-to-all expert dispatch.

- **No separate `moe_intermediate_size` yet**
  The routed experts currently reuse `intermediate_size`. Ling uses a dedicated expert width.

- **No Ling-style giant expert counts by default**
  nanoPLM keeps the config scalable, but current runs often use far fewer experts than Ling's `256`.

- **A real differentiable aux loss is still present**
  Ling belongs to the bias-corrected near-noaux family. nanoPLM currently keeps a differentiable router-mass balance term as a second stabilizer.

- **Routed scaling is conservative**
  Ling uses a larger routed scaling factor. nanoPLM currently defaults to `1.0` for stability in the small-model regime.

So the current design is "Ling-style routing logic with a simpler local systems stack and more conservative stabilization defaults."

## Architecture

Each MoE layer replaces a dense SwiGLU MLP with:

- `N` routed experts stored as stacked tensors
- `1` fixed shared expert implemented as a normal `ModernBertSwiGLUMLP`
- `1` sigmoid top-k router

Per-token flow:

```text
MoELayer.forward(x):
  shared_out = shared_expert(x)
  weights, indices, z_loss, router_scores = router(x)

  # dispatch metadata
  sort expanded expert ids by expert
  token_idx = sorted_idx // top_k   # which original token
  slot_idx  = sorted_idx % top_k    # which top_k slot
  inv_map = build_inverse_map(token_idx, slot_idx, T, top_k)

  # dispatch: fused repeat_interleave + permute
  x_sorted = scatter_dispatch(x, token_idx)

  # expert compute
  grouped_gemm through expert Wi
  SwiGLU activation
  grouped_gemm through expert Wo

  # combine: fused unpermute + weighted sum
  routed_out = gather_combine(expert_out_sorted, inv_map, weights, scale)

  aux = balance_loss(router_scores) + z_loss
  return routed_out + shared_out
```

### Expert computation and weight layout

The routed experts are stored as stacked tensors:

```python
Wi = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * intermediate_size))
Wo = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
```

This layout is:

- `(num_experts, in, out)`
- intentionally chosen for grouped GEMM
- not the same as `nn.Linear`'s `(out, in)`

The shared expert is a normal dense MLP module.

### Why grouped GEMM is the core primitive

Earlier MoE designs often rely on fixed capacity per expert:

- choose a capacity factor
- pad each expert to capacity
- drop overflow tokens
- track masks carefully

That is not attractive for nanoPLM.

The current nanoPLM implementation uses a custom
`torch.ops.nanoplm_moe.grouped_gemm` operator backed by an in-tree CUTLASS CUDA
extension.

That is a good match because it:

- natively handles jagged per-expert token counts
- avoids capacity hyperparameters
- avoids overflow token dropping
- avoids wasted compute on padded expert slots

That is why routed experts are stored as stacked expert tensors instead of as
separate `nn.Linear` modules.

The earlier implementation used the external `grouped_gemm` package. The
current code no longer depends on that library. Instead, nanoPLM vendors
CUTLASS under `third_party/cutlass` and JIT-builds a local extension on first
use through `torch.utils.cpp_extension.load`.

### Why nanoPLM does not use PyTorch `grouped_mm`

PyTorch's grouped MM path was considered, but it is not the current backend.

The practical reason is performance behavior in the actual MoE training path:

- it introduced host-side synchronization around grouped GEMM launch/setup
- that created CPU waits in the critical path
- those waits stalled the GPU work queue instead of keeping dispatch fully
  device-driven
- in practice this made the routed expert path materially slower

For nanoPLM's MoE usage, that matters because grouped GEMM is not an
occasional side op. It is the core expert compute primitive and runs twice per
MoE layer (`Wi` and `Wo`) in forward, then again through the same primitive in
backward.

So the current backend choice is not just "CUTLASS is lower level." It is:

- PyTorch `grouped_mm` matched the math
- but its runtime behavior caused CPU sync and GPU-thread stalls in this workload
- the in-house CUTLASS path lets nanoPLM keep grouped metadata and scheduling in
  a more GPU-native execution path

### Grouped GEMM operator stack

The current operator has three layers:

1. A Python-side `torch.library` definition with fake tensor and autograd
   registrations in `moe_grouped_gemm_ops.py`.
2. A small Python loader in `cutlass_grouped_gemm.py`
   that compiles and caches the extension under `.cache/torch_extensions`.
3. A CUTLASS-backed CUDA implementation in
   `csrc/moe_cutlass_grouped_gemm.cu`
   that builds grouped metadata on device and launches a CUTLASS
   `GemmGrouped` kernel in `GroupScheduleMode::kDeviceOnly`.

The current CUDA backend supports:

- `float16`
- `bfloat16`
- `trans_a=False, trans_b=False`
- `trans_a=False, trans_b=True`
- `trans_a=True, trans_b=False`

For CPU execution, and as a safety/reference path for unsupported CUDA dtypes,
the Python op falls back to a simple per-expert PyTorch implementation.

The op contract:

```python
torch.ops.nanoplm_moe.grouped_gemm(a, b, batch_sizes, trans_a=False, trans_b=False)
```

- `a`: packed 2D token matrix
- `b`: stacked 3D expert weights for the forward `NN` and `NT` cases, or a
  packed 2D matrix for the `TN` reduction-style case used in backward
- `batch_sizes`: 1D `int64` tensor of per-expert token counts

Backward is not a separate fused kernel family; it reuses grouped GEMM with different transpose settings.

## Fused Dispatch and Combine Ops

The dispatch and combine paths between the router and the grouped GEMMs are
implemented as fused Triton kernels in `moe_triton_ops.py`. These replace the
naive multi-kernel PyTorch sequences that were the original bottleneck.

### Motivation

The unfused path materialized many intermediate `(T*top_k, C)` tensors:

```text
Forward (unfused):
  x_expanded = x.repeat_interleave(top_k)       # WRITE (M, C)
  x_sorted = permute(x_expanded, sorted_idx)     # READ+WRITE (M, C)
  ... grouped_gemm ...
  expert_out = unpermute(expert_out_sorted)       # READ+WRITE (M, C)
  temp = expert_out * weights * scale             # READ+WRITE (M, C)
  output = temp.sum(dim=1)                        # READ+WRITE
```

Total dispatch+combine IO was ~7 GB per layer (forward + backward), spread
across 8-9 separate kernel launches with cache-cold intermediates. On the
original unfused code this cost 198.8 ms across all layers.

### Op 1: `scatter_dispatch`

Fuses `repeat_interleave` + `permute` into a single gather:

```python
x_sorted = moe_scatter_dispatch(x, token_idx)
# Impl: x_sorted[i] = x[token_idx[i]]
```

Forward IO: read `(T, C)` + `token_idx` + write `(M, C)` = 303 MB (was 600 MB).

Backward uses a Triton `scatter_add` kernel with `tl.atomic_add`. With
`top_k=2`, contention is exactly 2-way per token — trivial on modern GPUs.

### Op 2: `gather_combine`

Fuses `unpermute` + `mul(weights)` + `sum(top_k)` into a single kernel:

```python
output = moe_gather_combine(expert_out_sorted, inv_map, weights, scale)
# Impl: out[t] = sum_k weights[t,k] * expert_out_sorted[inv_map[t,k]] * scale
```

Forward IO: read `(M, C)` + weights + `inv_map` + write `(T, C)` = 404 MB (was 800+ MB).

The backward is a single fused kernel that computes both `grad_expert_sorted`
and `grad_weights` in one pass, loading `grad_out[t]` once:

```python
# For each token t and each slot k:
#   grad_eo[inv_map[t,k]] = grad_out[t] * weights[t,k] * scale
#   grad_w[t,k] = dot(grad_out[t], expert_out[inv_map[t,k]]) * scale
```

Fusing these saved a full redundant `grad_out` DRAM read per token.

### Dispatch metadata

Before either fused op, precompute once during dispatch:

```python
token_idx = (sorted_idx // top_k).to(torch.int32)   # which original token
slot_idx  = (sorted_idx % top_k).to(torch.int32)    # which top_k slot
inv_map = build_inverse_map(token_idx, slot_idx, T, top_k)  # (T, top_k) int32
```

`int32` is used because `T*top_k` = 131072 fits comfortably, and halving
metadata bandwidth reduces register pressure. These tensors are saved for
backward via `ctx.save_for_backward` — no buffer reuse, since checkpoint
re-entry or gradient accumulation would corrupt shared buffers.

### Dtype contract

Router weights are fp32. Expert outputs are bf16. The fused kernels:

1. Load `expert_out` in native dtype (bf16)
2. Cast to fp32 before multiply with weights
3. Accumulate the top_k sum in fp32
4. Store output as fp32 (matching PyTorch's implicit promotion rules)

A `FP32_ACCUM: tl.constexpr` flag controls the conditional cast, with fp64
passthrough for `gradcheck`.

### Eager fallbacks

All Triton kernels are specialized for `C <= 1024` (current hidden_size=768).
For larger C, the CUDA impls fall back to eager PyTorch paths. CPU always uses
eager. The op signatures, `register_fake`, and autograd wiring are
shape-agnostic — only the CUDA dispatch checks `C`.

### `torch.compile` integration

All ops (forward and backward) are registered via `torch.library`:

- `DEF` + `register_fake` for shape/dtype inference
- `impl("CUDA")` + `impl("CPU")` for execution
- `register_autograd` for backward wiring

Two lessons from the implementation:

1. **Backward ops must be registered custom ops.** Direct Triton kernel
   launches in the backward break `torch.compile` because Dynamo can't trace
   FakeTensors through raw kernel calls.

2. **Inductor cache invalidation.** After changing backward implementations,
   the inductor cache in `/tmp/torchinductor_*` can serve stale compiled
   graphs. Clear it when changing op implementations.

## Router Design

### Sigmoid top-k

The router in `moe.py` projects hidden states to expert logits, casts them to fp32, applies sigmoid, and selects `top_k` experts.

Why sigmoid:

- expert scores remain independent before top-k
- multiple experts can all look strongly relevant
- this matches the DeepSeek / GLM / Ling family more closely than softmax routing

### FP32 router

The router gate output is cast to fp32 before routing math:

- small score differences determine winner selection
- bf16 noise can corrupt rankings
- Ling explicitly keeps router state in fp32

This was adopted directly because small-model MoEs are extremely sensitive to router instability.

### Group-limited routing

When `moe_n_group > 1`, routing is constrained in two stages:

1. experts are partitioned into equal groups
2. groups are scored by the sum of their top-2 expert scores
3. only the best `moe_topk_group` groups survive
4. final top-k routing happens inside the surviving groups

This is lifted directly from the Ling / DeepSeek-style communication-aware routing idea, even though nanoPLM itself does not use expert parallelism yet.

Set `moe_n_group: 1` to disable it.

## Load Balancing and Router Stabilization

### Selection-only routing bias

If `moe_use_bias_correction` is enabled, the router keeps a persistent `correction_bias` buffer.

This bias:

- is added only for expert selection
- is **not** used in the final combine weights

That distinction is important. It lets nanoPLM steer expert usage without corrupting the semantic weighting of the selected experts.

### Global bias update

The bias update lives in `pure_pipeline.py`.

The update flow is:

1. each MoE forward stores per-expert token counts
2. counts are accumulated across gradient accumulation microsteps
3. counts are `all_reduce`d across ranks at the optimizer boundary
4. overused experts get negative bias, underused experts get positive bias
5. the delta is zero-mean centered before applying

That zero-mean centering is directly inspired by Ling's `bias-zero-mean-update`.

### Z-loss

The router also returns a z-loss:

- mean squared pre-sigmoid logits
- discourages logit explosion
- keeps sigmoid in its useful gradient regime

This matches the general router-stabilization strategy seen in Ling and the DeepSeek family.

### Auxiliary loss

nanoPLM currently still keeps a differentiable auxiliary balance loss:

- router scores are normalized per token
- mean expert mass is measured across the batch
- deviation from uniform usage is penalized

This is not the main balancing mechanism. The main mechanism is still bias correction.

The aux loss stays because in the small-model regime it is a cheap extra stabilizer, especially before expert specialization settles down.

## Dense-first Layers and Active Compute Matching

The config computes:

- `moe_layer_flags`
- `moe_dense_intermediate_size = (moe_top_k + 1) * intermediate_size`

This means:

- the first `moe_leading_dense_layers` can remain dense
- those dense layers are widened so their active compute matches the sparse MoE layers

That is a Ling-aligned decision:

- early layers often benefit less from routing
- but they should still be compute-matched for fair ablations

## FSDP and Optimizer Handling

### FSDP

FSDP sees the routed expert weights as ordinary parameters. There is:

- no expert parallelism
- no router-specific sharding trick
- no expert all-to-all

The correction bias is a buffer, so it stays replicated and is updated directly after the global count reduction.

### Optimizer grouping

The optimizer logic in `optim.py` is MoE-specific.

Originally, the routed expert tensors were falling through to AdamW simply because they are stored as 3D stacks.

That was not desirable.

The current behavior is:

- router gate stays on AdamW
- 1D / embedding / unembedding params stay on AdamW
- routed expert stacks `mlp.Wi` and `mlp.Wo` are explicitly kept on Muon / NorMuon

This matters because:

- the routed expert stacks are most of the model's parameter count
- they are conceptually batches of 2D expert matrices
- Dion's NorMuon explicitly supports non-flattened 3D tensors as batches of 2D matrices

So the optimizer behavior now matches the MoE storage design instead of accidentally fighting it.

## Operational Constraints

### Packed static-shape training

MoE currently assumes the packed static-token path:

- `use_packing: true`
- `use_static_inp_size: true`

This keeps:

- `T` fixed per batch
- `T * top_k` fixed
- expert-offset tensor shapes fixed

That makes the MoE path friendly to `torch.compile(dynamic=False)`.

### Grouped GEMM build/runtime behavior

The in-tree CUTLASS backend is built lazily on first CUDA use.

Operational implications:

- the machine needs a CUDA toolkit compatible with `torch.version.cuda`
- the first CUDA MoE call pays the extension build cost
- later calls reuse the cached extension
- supported GPU targets depend on the extension build arch list, typically
  controlled through `TORCH_CUDA_ARCH_LIST`

The current kernel template is instantiated with a CUTLASS `Sm80` architecture
tag, so the intended support range is Ampere-and-newer GPUs.

### Activation checkpointing

MoE works with:

- `activation_checkpointing_mode: "attn"`
- `activation_checkpointing_mode: "layer"`

MoE does **not** work with:

- `activation_checkpointing_mode: "attn+mlp"`

That is enforced in config validation.

When `layer` mode is active, the MoE forward is recomputed during backward.
The dispatch workspace buffers (`_dispatch_sort_values`, etc.) disable reuse
in this mode to avoid clobbering metadata still needed by the backward pass.

### Eval path

Training uses the packed 2D token path. Eval can still accept padded 3D inputs:

- reshape to flat tokens
- run identical MoE logic
- reshape back

## Current Config Surface

The active MoE knobs are:

```yaml
model:
  use_moe: true
  moe_num_experts: 8
  moe_top_k: 2
  moe_use_bias_correction: true
  moe_aux_loss_coef: 0.01
  moe_z_loss_coef: 5e-5
  moe_routed_scaling_factor: 1.0
  moe_n_group: 1
  moe_topk_group: 1
  moe_bias_update_rate: 1e-3
  moe_leading_dense_layers: 1
```

There is no longer a separate config for:

- number of shared experts
- router scoring function
- a standalone "enable group routing" flag

Those are fixed by design:

- one shared expert
- sigmoid routing
- group routing inferred from `moe_n_group > 1`

## Testing

Tests are in two files:

- `test_moe_grouped_gemm_ops.py` — grouped GEMM correctness:
  - CPU reference behavior
  - CUDA CUTLASS behavior
  - all three transpose modes used by forward/backward
  - zero-token experts
  - backward parity against the PyTorch reference path

- `test_moe_fused_ops.py` — fused dispatch/combine correctness:
  - `scatter_dispatch` forward match vs eager reference (bf16, fp32, CPU, CUDA)
  - `scatter_dispatch` backward match vs eager reference
  - `scatter_dispatch` fp64 gradcheck
  - `gather_combine` forward match vs eager reference (bf16, fp32, CPU, CUDA)
  - `gather_combine` backward match vs eager reference (including grad_weights)
  - `gather_combine` fp64 gradcheck
  - `gather_combine` output dtype contract (bf16 input + fp32 weights = fp32 output)
  - `build_inverse_map` correctness
  - full MoE layer integration test (forward + backward vs pure PyTorch reference)

## Performance

### Hardware and config

- RTX 5090: 1792 GB/s memory BW, 209.6 TFLOP/s bf16 tensor core
- T=65536, C=768, top_k=2, M=T*top_k=131072, E=11, inter=512, bf16
- 12 layers (11 MoE + 1 dense), FSDP, activation checkpointing (attn mode)
- Roofline crossover: 117 FLOP/byte for bf16 — all dispatch+combine kernels are BW-bound

### Profiler comparison

| Run | Description | GPU kernel time | Delta vs baseline |
|---|---|---|---|
| run-20031213 | Baseline (unfused) | 2748.7 ms | -- |
| run-20031346 | Fused ops + Triton scatter_add bwd | 2677.9 ms | -70.8 ms (-2.6%) |
| run-20031659 | Fused backward (bwd_eo + bwd_w merged) | 2572.3 ms | -176.4 ms (-6.4%) |

### Dispatch+combine kernel budget

| Run | Time | Delta |
|---|---|---|
| Baseline | 198.8 ms | -- |
| Fused ops + scatter_add | 113.9 ms | -84.9 ms (-42.7%) |
| Fused backward | 103.6 ms | -95.2 ms (-47.9%) |

### Baseline kernels removed (198.8 ms total)

| Kernel | Time | Count | Avg | Replaced by |
|---|---|---|---|---|
| `_moe_unpermute_kernel` | 46.1 ms | 180 | 256 us | `gather_combine` fwd + bwd |
| `_moe_permute_kernel` (bwd half) | 30.0 ms | 92 | 326 us | Triton `scatter_add` |
| `fused_add_mul_sum_unsqueeze_view` | 27.1 ms | 88 | 308 us | `gather_combine` fwd |
| `fused_expand_mul_sum_unsqueeze_view` | 24.2 ms | 92 | 263 us | `gather_combine` bwd |
| `fused__to_copy_expand_mul_permute_unsqueeze_view` | 21.2 ms | 92 | 230 us | `gather_combine` bwd |
| `fused_clone_expand_unsqueeze` (repeat_interleave) | 16.3 ms | 88 | 185 us | `scatter_dispatch` fwd |
| `fused_sum_view` | 15.0 ms | 93 | 161 us | `gather_combine` fwd |
| `fused_zeros` | 4.6 ms | 92 | 50 us | eliminated |

### Current fused kernels (103.6 ms total) -- roofline analysis

All kernels are bandwidth-bound. IO calculated for T=65536, C=768, top_k=2, bf16/fp32.

| Kernel | Time | Count | Avg | IO (MB) | BW floor (us) | BW eff. |
|---|---|---|---|---|---|---|
| `_moe_gather_combine_bwd_fused_kernel` | 36.3 ms | 92 | 394 us | 607 | 339 | **86%** |
| `_moe_scatter_add_kernel` | 24.3 ms | 92 | 265 us | 403 | 225 | **85%** |
| `_moe_gather_combine_fwd_kernel` | 22.0 ms | 88 | 250 us | 404 | 225 | **90%** |
| `_moe_gather_kernel` (scatter_dispatch fwd) | 18.9 ms | 88 | 214 us | 303 | 169 | **79%** |

IO breakdown:

- `scatter_dispatch` fwd: read x (T,C) bf16 100.7MB + token_idx 0.5MB + write x_sorted (M,C) bf16 201.3MB = 303MB
- `scatter_add` bwd: read grad_sorted (M,C) bf16 201.3MB + token_idx 0.5MB + RMW grad_x (T,C) bf16 201.4MB = 403MB
- `gather_combine` fwd: read expert_out (M,C) bf16 201.3MB + inv_map/weights 1.0MB + write output (T,C) fp32 201.3MB = 404MB
- `gather_combine` bwd fused: read grad_out (T,C) fp32 201.3MB + expert_out (M,C) bf16 201.3MB + inv_map/weights 1.5MB + write grad_eo (M,C) bf16 201.3MB + grad_w (T,top_k) fp32 0.5MB = 607MB

### Key optimization: `index_add_` replacement

The CUDA `indexFuncLargeIndex` implementation of `index_add_` achieved only
28% bandwidth efficiency (493 GB/s vs 1792 GB/s peak), at 609 us avg per call.
Replaced with a Triton `scatter_add` kernel using `tl.atomic_add`. With
`top_k=2`, contention is exactly 2-way per token — result: 265 us avg (85% BW
efficiency).

### Key optimization: fused backward

The original backward used two separate kernels (`bwd_eo` and `bwd_w`), each
loading `grad_out[t]` from DRAM independently. Merging them into a single
kernel that loads `grad_out` once saved 127 us per call (11.8 ms total across
92 invocations).

## Sources

- Qwen global-batch load balancing: [https://qwenlm.github.io/blog/global-load-balance/](https://qwenlm.github.io/blog/global-load-balance/)
- DeepSeek-V3 report: [https://arxiv.org/html/2412.19437](https://arxiv.org/html/2412.19437)
- GLM-5 config: [https://huggingface.co/zai-org/GLM-5-FP8/blob/main/config.json](https://huggingface.co/zai-org/GLM-5-FP8/blob/main/config.json)
- MiniMax-M2 config: [https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json)
- Ling scaling-law paper: [https://arxiv.org/abs/2507.17702](https://arxiv.org/abs/2507.17702)
- Ling HTML paper: [https://arxiv.org/html/2507.17702](https://arxiv.org/html/2507.17702)
- Ling-V2 repo: [https://github.com/inclusionAI/Ling-V2](https://github.com/inclusionAI/Ling-V2)
- Ling-mini-2.0 model card: [https://huggingface.co/inclusionAI/Ling-mini-2.0](https://huggingface.co/inclusionAI/Ling-mini-2.0)
- Ling-mini-base-2.0 config: [https://huggingface.co/inclusionAI/Ling-mini-base-2.0/blob/main/config.json](https://huggingface.co/inclusionAI/Ling-mini-base-2.0/blob/main/config.json)
- CUTLASS overview: [https://docs.nvidia.com/cutlass/latest/overview.html](https://docs.nvidia.com/cutlass/latest/overview.html)
- CUTLASS grouped GEMM example: [https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped)
- PyTorch custom C++/CUDA ops tutorial: [https://pytorch.org/tutorials/advanced/cpp_custom_ops.html](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)
