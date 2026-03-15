# Mixture-of-Experts Implementation

## Overview

nanoPLM supports Mixture-of-Experts (MoE) as a drop-in replacement for the dense MLP in any encoder layer. The implementation uses `torch.nn.functional.grouped_mm` — a CUTLASS-backed grouped GEMM — as the core compute primitive. No token dropping, no capacity factor, no padding waste.

Pure-torch pipeline only. No Transformer Engine, no FP8.

## Architecture

Each MoE layer replaces a single dense MLP with:

- **N routed experts** — stacked weight tensors processed via `grouped_mm`
- **1 shared expert** — a standard `ModernBertSwiGLUMLP` that processes all tokens
- **1 sigmoid top-k router** — selects which routed experts each token uses

Per-token forward: the router picks `top_k` experts. Those experts process the token via SwiGLU. Their outputs are combined using renormalized sigmoid weights. The shared expert output is added on top.

```
MoELayer.forward(x):
  shared_out = shared_expert(x)              # all tokens, one MLP
  weights, indices = router(x)               # sigmoid top-k selection
  dispatched = sort_by_expert(x, indices)    # sorted permutation
  expert_out = grouped_swiglu(dispatched)    # grouped_mm through SwiGLU
  combined = unsort_and_combine(expert_out, weights)
  return combined + shared_out
```

### Why grouped_mm over bmm

The previous design considered `torch.bmm` with fixed-capacity padding. That approach requires a `capacity_factor` hyperparameter, drops tokens that overflow capacity, needs overflow mask bookkeeping to avoid gradient corruption, and wastes compute on padding.

`grouped_mm` handles jagged per-expert token counts natively. No capacity factor, no dropping, no padding. It is the correct primitive for dropless MoE.

Verified properties (A100 SM80, PyTorch 2.10, BF16):

| Property | Result |
|---|---|
| Natively autograd differentiable | yes — no custom `autograd.Function` needed |
| `torch.compile(dynamic=False)` with varying `offs` | works, zero recompilations |
| Weight layout | `(num_experts, in_features, out_features)` |

Note: `grouped_mm` backward has a stride bug with `tensor.sum()` gradients (expanded stride `(0, 0)`). This does not affect real training — `cross_entropy`, `mean()`, and all practical loss functions produce proper gradients.

### What doesn't exist / wasn't used

- `torch._grouped_linear` — does not exist in PyTorch 2.10
- `torch.nn.functional.scaled_grouped_mm` — requires SM89+ (H100/Ada), unavailable on A100
- `torchtune.modules.moe.GroupedExperts` — gates `grouped_mm` behind SM>=90, falls back to a Python for-loop on SM80

## Routing and Load Balancing

### Sigmoid top-k router

The router projects hidden states to `(num_experts,)` logits, applies sigmoid, then selects `top_k` experts per token. Unlike softmax routing, sigmoid allows each expert's score to be independent — the router can express "these two experts are both highly relevant" without forcing one down to make the other go up.

### Routing bias correction (DeepSeek-V3 style)

The primary load balancing mechanism. A `(num_experts,)` bias vector is added to router logits **before** top-k selection but is **not** included in the combine weights. This means the bias steers which experts get selected without distorting how their outputs are weighted.

The bias is updated once per optimizer step based on global expert load statistics:

1. Each forward pass records per-expert token counts
2. Counts are accumulated across gradient accumulation micro-steps
3. At the optimizer step boundary, counts are `all_reduce`d across ranks
4. The bias is nudged: overused experts get negative bias, underused get positive
5. Accumulators are reset

This is one `all_reduce` on a tiny `(num_experts,)` tensor per optimizer step — essentially free. The bias is stored as a registered buffer (not a parameter) so FSDP doesn't shard it.

**Why global/batch-level and not micro-batch:** Qwen's research shows that micro-batches tend to be domain-homogeneous. Forcing uniform expert usage per micro-batch fights natural specialization. Balancing over the full optimization step (across all micro-steps and all ranks) allows natural within-batch imbalance while preventing long-term expert collapse.

### Auxiliary loss (optional safety valve)

A sequence-level balance regularizer, off by default (`moe_aux_loss_coef: 0.0`). Measures squared deviation from uniform expert usage per forward pass. Only turn it on if expert collapse is observed despite bias correction.

This matches the frontier pattern: DeepSeek-V3, GLM-5, and MiniMax-M2 all use routing bias correction as the primary mechanism, with aux loss as a secondary/optional complement.

## Research Context

The design follows the consensus of modern MoE architectures:

| Model | Experts | Top-k | Shared | Routing | Bias correction | Aux loss |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | 256 | 8 | 1 | sigmoid | yes | tiny, secondary |
| GLM-5 | 256 | 8 | 1 | sigmoid | yes | noaux_tc |
| MiniMax-M2 | 256 | 8 | 0 | sigmoid | yes | 0.001 |
| Qwen3-Next | 512 | 10 | 1 | — | — | 0.001 |
| **nanoPLM** | configurable | configurable | 1 | sigmoid | yes | 0.0 (off) |

The practical frontier is: shared expert + sigmoid routing + global bias correction + dropless dispatch. Not: micro-batch auxiliary balancing with capacity-limited dropping.

## Implementation Details

### File structure

All MoE logic lives in one file:

```
src/nanoplm/pretraining/models/modern_bert/moe.py    (~175 lines)
  - Router        — sigmoid top-k with bias correction
  - MoELayer      — dispatch, grouped_mm SwiGLU, combine
```

Modified files:
- `modeling.py` — config fields, MLP selection, weight init, aux loss collection
- `pure_pipeline.py` — guard rails, routing bias update hook, FLOP calculation
- `pretrain.py` — active vs total parameter logging

### Weight storage

Routed experts use stacked weight tensors for `grouped_mm`:

```python
Wi = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * intermediate_size))
Wo = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
```

This is `(G, in, out)` layout — transposed relative to `nn.Linear`'s `(out, in)`. Cannot reuse `nn.Linear` modules for routed experts.

The shared expert is a normal `ModernBertSwiGLUMLP` instance.

### Dispatch flow

```python
# 1. Expand tokens by top_k
x_expanded = x.repeat_interleave(top_k, dim=0)     # (T*top_k, H)
expert_flat = indices.flatten()                      # (T*top_k,)

# 2. Sort by expert assignment → contiguous slabs per expert
sorted_idx = expert_flat.argsort(stable=True)
x_sorted = x_expanded[sorted_idx]

# 3. Compute offs for grouped_mm
counts = expert_flat.bincount(minlength=num_experts)
offs = counts.cumsum(0).to(torch.int32)

# 4. Grouped SwiGLU
wi = F.grouped_mm(x_sorted, self.Wi, offs=offs)
x_proj, gate = wi.chunk(2, dim=-1)
activated = F.silu(gate) * x_proj
expert_out_sorted = F.grouped_mm(activated, self.Wo, offs=offs)

# 5. Unsort and weighted combine
expert_out = torch.empty_like(expert_out_sorted)
expert_out[sorted_idx] = expert_out_sorted
expert_out = expert_out.view(T, top_k, H)
expert_out = (expert_out * weights.unsqueeze(-1)).sum(dim=1)
```

With `use_static_inp_size=True`, `T` is fixed per batch, so `T * top_k` is a compile-time constant. `offs` shape `(num_experts,)` is fixed — only values change per batch. `torch.compile(dynamic=False)` does not recompile when `offs` values change.

### Eval path

The MoE forward handles both packed 2D `(T, H)` training input and padded 3D `(B, S, H)` eval input. For 3D input, it reshapes to `(B*S, H)`, runs the identical routing/dispatch/combine flow, then reshapes back. Eval runs under `@torch.compiler.disable` so no compile concerns.

### FSDP integration

No special handling. The stacked `Wi`/`Wo` parameters are standard `nn.Parameter` tensors — FSDP shards them normally across devices. The routing bias is a registered buffer, so FSDP leaves it unsharded. No expert parallelism, no all-to-all.

The bias update writes directly to the buffer with `add_()`. Since every rank computes the same delta (counts are `all_reduce`d first), all ranks stay in sync without additional communication.

### FLOP calculation

Active FLOPs per token per MoE layer:
- Router gate: `2 * H * num_experts`
- Routed experts: `top_k * 6 * H * intermediate_size` (SwiGLU)
- Shared expert: `6 * H * intermediate_size` (SwiGLU)

The FLOP estimator respects `moe_layer_pattern` (mixed dense/MoE layers) and `no_mlp_on_first_layer`.

### Checkpointing

All MoE state is captured by standard `state_dict()` / `load_state_dict()`:
- `mlp.Wi`, `mlp.Wo` — routed expert stacked weights
- `mlp.router.gate.weight` — router projection
- `mlp.router.correction_bias` — routing bias (persistent buffer)
- `mlp.shared_expert.*` — shared expert MLP weights

Ephemeral per-forward state (`last_expert_counts`, `last_aux_loss`) is correctly excluded — it's recomputed each forward pass. The pipeline-level `_moe_count_accum` resets to zero on resume, starting a fresh accumulation window.

### Weight initialization

Routed expert weights use the same scheme as the dense MLP:
- `Wi`: uniform `(-bound, bound)`
- `Wo`: zeros (GPT-2 style output projection init)
- Router gate: uniform `(-bound, bound)`
- Shared expert: standard MLP init path

### Interaction with other features

| Feature | Status |
|---|---|
| Activation checkpointing (attn mode) | works — MLP is not checkpointed in attn mode |
| Activation checkpointing (attn+mlp) | raises error — not compatible with MoE |
| Canon layers A/C | orthogonal — operate on attention/residual |
| Canon layer D | not applied inside routed experts |
| DiffAttn v2 | orthogonal — attention only |
| ProRes | works — MoELayer returns same shape as dense MLP |
| MHC-lite | works — MoELayer is just a module inside the encoder layer |
| Batch-size warmup | works — MoE has static shapes regardless |

## Configuration

```yaml
model:
  use_moe: true
  moe_num_experts: 8               # number of routed experts
  moe_top_k: 2                     # experts activated per token
  moe_num_shared_experts: 1        # shared expert (always 1 for now)
  moe_scoring_func: "sigmoid"      # router scoring function
  moe_use_bias_correction: true    # DeepSeek-V3 style routing bias
  moe_aux_loss_coef: 0.0           # sequence-level balance loss (0 = off)
  moe_layer_pattern: null          # "DM" for every-other-layer MoE, null = all MoE
```

Requirements: `use_packing: true` and `use_static_inp_size: true`. The pipeline raises if these are not set.

### Layer patterns

`moe_layer_pattern` controls which layers are MoE vs dense:
- `null` or `"M"` — all layers are MoE
- `"DM"` — alternating dense/MoE (layers 0, 2, 4... are dense; 1, 3, 5... are MoE)
- `"DDM"` — every third layer is MoE
- Any combination of `D` (dense) and `M` (MoE), tiled across layers

### Active parameter matching for ablations

To compare MoE vs dense fairly, match **active** parameters per token. Active params = total params minus inactive routed expert weights.

Each token activates `top_k` routed experts + 1 shared expert = `(top_k + 1)` MLPs. To match a dense model with `intermediate_size = I`:

```
moe_intermediate_size = I / (top_k + 1)
```

Example with `hidden_size=1024, intermediate_size=2048` dense baseline (43.0M params):

| Config | `intermediate_size` | Active params | Total params | Ratio |
|---|---|---|---|---|
| Dense | 2048 | 43.0M | 43.0M | 1.0x |
| top_k=2, E=8 | 683 | 43.1M | 93.4M | 2.2x |
| top_k=1, E=16 | 1024 | 43.1M | 231.8M | 5.4x |

Both MoE configs match the dense baseline's active compute per token within 0.1%.

## Sources

- [Qwen blog: Global Batch Load Balance](https://qwenlm.github.io/blog/global-load-balance/)
- [DeepSeek-V3 technical report](https://arxiv.org/html/2412.19437)
- [GLM-5-FP8 config](https://huggingface.co/zai-org/GLM-5-FP8/blob/main/config.json)
- [MiniMax-M2 config](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json)
- [PyTorch grouped_mm docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html)
