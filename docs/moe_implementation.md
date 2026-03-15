# Mixture-of-Experts: Research, Design, and Implementation

## Purpose

This document is the single source of truth for nanoPLM's MoE work.

It does three jobs:

1. Summarizes the research path across public MoE models.
2. Explains why Ling-mini-2.0 / Ling-V2 became the main inspiration point.
3. Documents the current nanoPLM implementation and the reasons behind its design choices.

This replaces the older split between "implementation notes" and the separate Ling research report.

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
  `grouped_mm` allows jagged token counts per expert without token dropping or padding to capacity.

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

## Current nanoPLM Architecture

Each MoE layer replaces a dense SwiGLU MLP with:

- `N` routed experts stored as stacked tensors
- `1` fixed shared expert implemented as a normal `ModernBertSwiGLUMLP`
- `1` sigmoid top-k router

Per-token flow:

```text
shared_out = shared_expert(x)
weights, indices = router(x)
dispatch tokens to experts
run routed expert MLPs with grouped_mm
combine routed outputs with normalized weights
add shared_out
```

More explicitly:

```text
MoELayer.forward(x):
  shared_out = shared_expert(x)
  weights, indices, z_loss, router_scores = router(x)
  x_expanded = repeat_interleave(x, top_k)
  sort expanded tokens by expert id
  grouped_mm through expert Wi
  SwiGLU
  grouped_mm through expert Wo
  unsort back to token order
  routed_out = weighted_sum(expert_out, weights) * routed_scaling_factor
  aux = balance_loss(router_scores) + z_loss
  return routed_out + shared_out
```

## Why `grouped_mm` is the core primitive

Earlier MoE designs often rely on fixed capacity per expert:

- choose a capacity factor
- pad each expert to capacity
- drop overflow tokens
- track masks carefully

That is not attractive for nanoPLM.

`torch.nn.functional.grouped_mm` is a better match because it:

- natively handles jagged per-expert token counts
- avoids capacity hyperparameters
- avoids overflow token dropping
- avoids wasted compute on padded expert slots

That is why routed experts are stored as stacked expert tensors instead of as separate `nn.Linear` modules.

## Router Design

### Sigmoid top-k

The router in [moe.py](/workspace/nanoplm/src/nanoplm/pretraining/models/modern_bert/moe.py) projects hidden states to expert logits, casts them to fp32, applies sigmoid, and selects `top_k` experts.

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

The bias update lives in [pure_pipeline.py](/workspace/nanoplm/src/nanoplm/pretraining/pure_pipeline.py).

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

## Expert Computation and Weight Layout

The routed experts are stored as stacked tensors:

```python
Wi = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * intermediate_size))
Wo = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
```

This layout is:

- `(num_experts, in, out)`
- intentionally chosen for `grouped_mm`
- not the same as `nn.Linear`'s `(out, in)`

The shared expert is a normal dense MLP module.

Dispatch flow:

1. expand each token `top_k` times
2. flatten selected expert ids
3. stable-sort by expert id
4. compute expert boundaries with `bincount` + `cumsum`
5. run grouped expert `Wi`
6. apply SwiGLU
7. run grouped expert `Wo`
8. unsort back to token order
9. combine selected expert outputs with normalized weights

This is the simplest dropless local MoE design that still uses the right primitive.

## Dense-first Layers and Active Compute Matching

The config in [modeling.py](/workspace/nanoplm/src/nanoplm/pretraining/models/modern_bert/modeling.py) computes:

- `moe_layer_flags`
- `moe_dense_intermediate_size = (moe_top_k + 1) * intermediate_size`

This means:

- the first `moe_leading_dense_layers` can remain dense
- those dense layers are widened so their active compute matches the sparse MoE layers

That is a very Ling-aligned decision:

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

The optimizer logic in [optim.py](/workspace/nanoplm/src/nanoplm/pretraining/optim.py) is worth calling out because it is MoE-specific.

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

### Activation checkpointing

MoE works with:

- `activation_checkpointing_mode: "attn"`
- `activation_checkpointing_mode: "layer"`

MoE does **not** work with:

- `activation_checkpointing_mode: "attn+mlp"`

That is enforced in config validation.

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

## Why this is the current landing point

The final design choice was not "copy Ling".

It was:

- take the **routing philosophy** from DeepSeek / GLM / Ling
- take the **small-scale relevance** and **training visibility** from Ling
- keep the **systems scope** modest enough for nanoPLM

That is why the current implementation looks the way it does:

- not a toy top-2 MoE
- not a full expert-parallel megacluster stack
- but a serious sparse MLP design with modern routing and stabilization choices

In short:

- DeepSeek and GLM clarified the routing family
- Qwen clarified the alternative softmax family
- Ling made the small-scale case convincing
- nanoPLM implemented the subset that is most defensible and most transferable

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
- PyTorch `grouped_mm`: [https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html)
