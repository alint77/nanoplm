# MoE Implementation Spec for nanoPLM

## Scope

Pure-torch pipeline only. No TE, no FP8. BF16 `torch.nn.functional.grouped_mm` as the core primitive.

Hard requirements:

- `use_packing=true`
- `use_static_inp_size=true`
- the packed flat-token static-shape forward path
- raise on any other path (non-packed, dynamic-shape, padded fallback, TE)

No expert parallelism — the full model fits on each GPU.

## Integration points

- `pretrain.yaml` — new `moe_*` config keys
- `src/nanoplm/pretraining/models/modern_bert/modeling.py` — config fields, MLP selection, FSDP wrapping, weight init
- `src/nanoplm/pretraining/models/modern_bert/moe.py` — new file, all MoE logic
- `src/nanoplm/pretraining/pure_pipeline.py` — auxiliary loss plumbing

## Research summary

### Cross-model pattern

Across Qwen, DeepSeek-V3, GLM-5, and MiniMax-M2, the public evidence points to:

1. **global/batch-level balancing** beats micro-batch balancing
2. **routing correction bias** is mainstream (DeepSeek-V3, GLM-5, MiniMax-M2)
3. **shared experts** are common (Qwen3.5, Qwen3-Next, GLM-5)
4. **sigmoid-scored top-k routing** is standard in the newest configs (GLM-5: `scoring_func="sigmoid"`)
5. **no token dropping** (DeepSeek-V3 explicitly)
6. old-style aux loss still appears in public configs but looks more like a compatibility or secondary mechanism

The practical frontier is: shared expert + batch/global-aware balancing + routing-bias correction + dropless dispatch.

Not: pure micro-batch auxiliary balancing with capacity-limited dropping.

Full model-by-model research is in the appendix at the end of this file.

## Core primitive: `torch.nn.functional.grouped_mm`

### What it is

CUTLASS-backed grouped GEMM designed for MoE. Multiplies a flat 2D token tensor by a 3D stacked-expert weight tensor, handling jagged per-expert token counts via an `offs` index tensor.

```python
torch.nn.functional.grouped_mm(mat_a, mat_b, *, offs=None, bias=None, out_dtype=None)
```

- `mat_a`: `(total_dispatched, K)` — flat sorted tokens, sliced into groups by `offs`
- `mat_b`: `(num_experts, K, N)` — stacked expert weights
- `offs`: `(num_experts,)` int32 — cumulative endpoint of each expert's token slice
- Returns: `(total_dispatched, N)`

### Verified properties (tested on A100 SM80, PyTorch 2.10)

| Property | Result |
|---|---|
| Works on SM80 (A100) with BF16 | yes |
| Natively autograd differentiable | yes — no custom `autograd.Function` needed |
| `torch.compile(dynamic=False)` with varying `offs` values | works, zero recompilations |
| `torch.compile(dynamic=True)` | also works |
| Weight layout | `(G, in_features, out_features)` — standard matmul, NOT transposed like nn.Linear |

### Why not `torch.bmm`

The previous discussion recommended `torch.bmm` with fixed-capacity padding. That approach requires:

- padding every expert slab to uniform capacity
- a `capacity_factor` hyperparameter (e.g. 1.25)
- token dropping when experts overflow capacity
- overflow mask bookkeeping to avoid gradient corruption

`grouped_mm` eliminates all of this. It handles jagged token counts natively. No capacity factor, no dropping, no padding waste.

Performance comparison on A100 (SwiGLU forward, 16384 dispatched tokens, 8 experts, 768/1536):

| Method | Time | Waste | Drops tokens? |
|---|---|---|---|
| `grouped_mm` | 1.35 ms | 0% | no |
| `bmm` cf=1.25 | 1.08 ms | 25% | yes (overflow) |
| `bmm` no-drop | 1.52 ms | 71% (pad to max) | no |
| `for-loop` | 1.40 ms | 0% | no |

`bmm` with cf=1.25 is ~20% faster in raw kernel time on A100, but it drops tokens (silent gradient corruption) and wastes 25% of compute on padding. `grouped_mm` is the right choice: simpler, correct, no dropping, and PyTorch is actively optimizing the CUTLASS kernels for newer hardware.

### What does NOT exist

- `torch._grouped_linear` — does not exist in PyTorch 2.10
- `torch.nn.functional.scaled_grouped_mm` — exists but requires SM89+ (H100/Ada), not available on A100
- `torchao.prototype.moe_training` — not installed, and its FP8 paths require SM89+/SM100+
- `torchtune.modules.moe.GroupedExperts` — exists but gates `grouped_mm` behind SM>=90, falling back to a Python for-loop on SM80. We call `grouped_mm` directly.

## Architecture decisions

### Routing

- `sigmoid` scoring (not softmax)
- `top_k=2`
- 8 routed experts
- 1 shared expert (reuses existing `ModernBertSwiGLUMLP`)
- renormalize top-k weights: `weights = weights / weights.sum(-1, keepdim=True)`

### Balancing

- **routing-bias-based global/batch balancing** as the primary mechanism
- a learnable or EMA-updated `(num_experts,)` bias vector added to router logits before top-k selection, but NOT used in the combine weights
- bias updated based on expert load statistics accumulated across gradient accumulation steps and across ranks (one `all_reduce` on a `(num_experts,)` counter per optimizer step)
- optional tiny sequence-level aux loss as a safety valve, default coefficient 0.0 (off)

### Dispatch

- dropless — all tokens always processed
- sorted permutation: expand tokens by top_k, sort by expert index, compute offs from bincount
- `grouped_mm` through SwiGLU on the flat sorted tensor
- inverse permutation + weighted combine
- no capacity factor, no capacity management, no overflow masks

### Weight storage

Routed experts use stacked weight tensors (not `nn.Linear` modules):

```python
Wi = Parameter(torch.empty(num_experts, hidden_size, 2 * intermediate_size))   # (8, 768, 3072)
Wo = Parameter(torch.empty(num_experts, intermediate_size, hidden_size))        # (8, 1536, 768)
```

This is the `(G, in, out)` layout that `grouped_mm` requires. It is transposed relative to `nn.Linear`'s `(out, in)` layout.

Shared expert is a normal `ModernBertSwiGLUMLP` instance — it processes all tokens through a single MLP with no dispatch.

### What is explicitly excluded

- no activation checkpointing support for MoE layers (user never uses MLP AC anyway)
- no Canon-D support inside routed experts (user only uses Canon AC)
- no FP8 / TE path
- no expert parallelism

## Implementation spec

### New file: `src/nanoplm/pretraining/models/modern_bert/moe.py`

#### `Router` (nn.Module)

```python
class Router(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k, bias_correction=True):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        if bias_correction:
            self.correction_bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.correction_bias = None
```

Forward:
1. `logits = sigmoid(self.gate(x))` — `(total_tokens, num_experts)`
2. If correction_bias: add bias to logits before top-k, but do NOT include bias in combine weights
3. `weights, indices = topk(logits, k=self.top_k)` — each `(total_tokens, top_k)`
4. `weights = weights / weights.sum(-1, keepdim=True)` — renormalize
5. Return `weights, indices`

#### `MoELayer` (nn.Module)

```python
class MoELayer(nn.Module):
    def __init__(self, config):
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k

        # Router
        self.router = Router(config.hidden_size, self.num_experts, self.top_k)

        # Routed expert weights (stacked for grouped_mm)
        self.Wi = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, 2 * config.intermediate_size))
        self.Wo = nn.Parameter(torch.empty(self.num_experts, config.intermediate_size, config.hidden_size))
        self.drop = nn.Dropout(config.mlp_dropout)

        # Shared expert (reuses existing MLP class)
        self.shared_expert = ModernBertSwiGLUMLP(config)
```

Forward signature matches existing MLP interface:

```python
def forward(self, x, position_ids=None, cu_seqlens=None, attention_mask=None):
```

Forward flow:

```
 1. shared_out = self.shared_expert(x, position_ids, cu_seqlens, attention_mask)
 2. weights, indices = self.router(x)             # (T, top_k), (T, top_k)
 3. x_expanded = x.repeat_interleave(top_k, 0)    # (T*top_k, H)
 4. expert_flat = indices.flatten()                 # (T*top_k,)
 5. sorted_idx = expert_flat.argsort(stable=True)
 6. x_sorted = x_expanded[sorted_idx]              # (T*top_k, H)
 7. counts = expert_flat.bincount(minlength=num_experts)
 8. offs = counts.cumsum(0).to(torch.int32)         # (num_experts,)
 9. wi = F.grouped_mm(x_sorted, self.Wi, offs=offs)  # (T*top_k, 2*inter)
10. x_proj, gate = wi.chunk(2, dim=-1)
11. activated = silu(gate) * x_proj                  # (T*top_k, inter)
12. expert_out_sorted = F.grouped_mm(self.drop(activated), self.Wo, offs=offs)  # (T*top_k, H)
13. expert_out = torch.zeros_like(expert_out_sorted)
14. expert_out[sorted_idx] = expert_out_sorted       # unsort
15. expert_out = expert_out.view(T, top_k, H)        # (T, top_k, H)
16. expert_out = (expert_out * weights.unsqueeze(-1)).sum(dim=1)  # (T, H)
17. return expert_out + shared_out
```

With `use_static_inp_size=True`:
- `T` (total_tokens) is fixed per batch → `T * top_k` is a compile-time constant
- `offs` shape `(num_experts,)` is fixed
- only `offs` values change between batches (routing is stochastic)
- `torch.compile(dynamic=False)` does NOT recompile when `offs` values change

#### Auxiliary loss (optional)

Sequence-level balance regularizer, returned separately:

```python
# fraction of tokens routed to each expert
frac = counts.float() / counts.sum()
# uniform target
target = torch.ones_like(frac) / self.num_experts
# balance loss: squared deviation from uniform
aux_loss = ((frac - target) ** 2).sum() * self.num_experts
```

Default coefficient: 0.0 (off). Only turn on if expert collapse is observed.

### Config additions to `ModernBertConfig`

New fields in the dataclass:

```python
use_moe: bool = False
moe_num_experts: int = 8
moe_top_k: int = 2
moe_num_shared_experts: int = 1
moe_scoring_func: str = "sigmoid"
moe_use_bias_correction: bool = True
moe_aux_loss_coef: float = 0.0
moe_layer_pattern: Optional[str] = None   # e.g. "DM" for dense/moe alternating. None = all MoE.
```

`moe_layer_pattern` works like `attn_layer_pattern`: a tiled string where `M` = MoE layer, `D` = dense (normal MLP). Example: `"DM"` makes every other layer MoE. `None` or `"M"` makes all layers MoE.

Add to `__post_init__`:
```python
if self.use_moe:
    if self.moe_layer_pattern is not None:
        pattern = self.moe_layer_pattern.upper().strip()
        _map = {"M": True, "D": False}
        self.moe_layer_flags = [_map[pattern[i % len(pattern)]] for i in range(self.num_hidden_layers)]
    else:
        self.moe_layer_flags = [True] * self.num_hidden_layers
```

### Config additions to `pretrain.yaml`

```yaml
model:
  # ... existing keys ...
  use_moe: false
  moe_num_experts: 8
  moe_top_k: 2
  moe_num_shared_experts: 1
  moe_scoring_func: "sigmoid"
  moe_use_bias_correction: true
  moe_aux_loss_coef: 0.0
  moe_layer_pattern: null     # "DM" for every-other-layer MoE, null for all MoE
```

### Changes to `modeling.py`

#### MLP selection (line ~1017)

Currently:
```python
if config.mlp_activation == "srelu":
    self.mlp = ModernBertSReluMLP(config)
elif config.mlp_activation == "swiglu":
    self.mlp = ModernBertSwiGLUMLP(config)
else:
    self.mlp = ModernBertMLP(config)
```

Add MoE branch:
```python
if config.use_moe and config.moe_layer_flags[layer_idx]:
    from .moe import MoELayer
    self.mlp = MoELayer(config)
elif config.mlp_activation == "srelu":
    self.mlp = ModernBertSReluMLP(config)
elif config.mlp_activation == "swiglu":
    self.mlp = ModernBertSwiGLUMLP(config)
else:
    self.mlp = ModernBertMLP(config)
```

No changes to the MLP call site — `MoELayer.forward` matches the existing MLP forward signature `(x, position_ids, cu_seqlens, attention_mask)`.

#### Weight initialization (line ~1868)

Add MoE init alongside existing MLP init:
```python
if isinstance(enc.mlp, MoELayer):
    # Routed experts: same scheme as dense MLP
    nn.init.uniform_(enc.mlp.Wi, -bound, bound)
    nn.init.zeros_(enc.mlp.Wo)
    # Router gate
    nn.init.uniform_(enc.mlp.router.gate.weight, -bound, bound)
    # Shared expert: handled by existing init (it has .Wi and .Wo attributes)
    nn.init.uniform_(enc.mlp.shared_expert.Wi.weight, -bound, bound)
    nn.init.zeros_(enc.mlp.shared_expert.Wo.weight)
    if enc.mlp.shared_expert.Wi.bias is not None:
        nn.init.zeros_(enc.mlp.shared_expert.Wi.bias)
    if enc.mlp.shared_expert.Wo.bias is not None:
        nn.init.zeros_(enc.mlp.shared_expert.Wo.bias)
```

#### FSDP wrapping (line ~507)

No special handling needed. `fully_shard(enc.mlp, ...)` already wraps whatever module `enc.mlp` is. The stacked `Wi`/`Wo` parameters inside `MoELayer` are just `nn.Parameter` tensors — FSDP shards them normally across devices. No EP, no all-to-all.

### Changes to `pure_pipeline.py`

#### Auxiliary loss plumbing

Collect MoE aux losses from all layers and add to the main loss. Two options:

**Option A (recommended): Accumulate during forward via a context list.**

In `ModernBertModel.forward`, before the layer loop, create a list. Each `MoELayer.forward` appends its aux loss. After the loop, sum them.

```python
# In ModernBertForMaskedLM.forward, after getting loss:
if self.config.use_moe and self.config.moe_aux_loss_coef > 0:
    aux_losses = [layer.mlp.last_aux_loss for layer in self.model.layers
                  if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'last_aux_loss')]
    aux_loss = sum(aux_losses) / max(len(aux_losses), 1)
    loss = loss + self.config.moe_aux_loss_coef * aux_loss
```

Each `MoELayer` stores `self.last_aux_loss` during forward. This avoids changing the return signature of any existing forward method.

**Option B: Return in the dict.**

Add `"aux_loss"` to the return dict. Modify `pure_pipeline.py` to read it:
```python
loss = out["loss"]
if "aux_loss" in out:
    loss = loss + config.moe_aux_loss_coef * out["aux_loss"]
```

Option A is cleaner because it doesn't change any existing function signatures or return types.

#### Guard rails

Add at the start of the training function:
```python
if config.use_moe:
    assert config.use_packing, "MoE requires use_packing=True"
    assert config.use_static_inp_size, "MoE requires use_static_inp_size=True"
```

### Eval path

Reshape `(B, S, H)` → `(B*S, H)`, run the identical MoE forward, reshape back. Since eval uses `@torch.compiler.disable`, no compile concerns. The MoE forward path is shape-agnostic — it only needs a 2D `(tokens, hidden)` input.

### Interaction with existing features

| Feature | Interaction | Action |
|---|---|---|
| Activation checkpointing (attn mode) | MLP is not checkpointed in `attn` mode | no issue |
| Activation checkpointing (attn+mlp mode) | would checkpoint MoE layer including router | raise if `use_moe` and AC mode is `attn+mlp` |
| Canon layers A/C | operate on attention/residual, not MLP | orthogonal, no issue |
| Canon layer D | inside MLP Wi output | skip for MoE layers (shared expert has no Canon D either when `use_moe=True`) |
| DiffAttn v2 | attention-only | orthogonal |
| ProRes | scales residual: `x = x + alpha * mlp_out` | works — MoELayer returns same shape as MLP |
| MHC-lite | wraps encoder layers | works — MoELayer is just a module inside the encoder layer |
| Batch-size warmup | triggers `torch.compile.reset()` | works — MoE forward has static shapes regardless |
| FSDP sublayer wrapping | `fully_shard(enc.mlp, ...)` | works — stacked Parameters shard normally |

### Routing bias update procedure

The routing correction bias is updated per optimizer step, not per forward pass:

1. During forward: each `MoELayer` records `self.last_expert_counts` — a `(num_experts,)` tensor of how many tokens each expert received
2. After backward, before optimizer step: accumulate counts across gradient accumulation steps and across ranks via `dist.all_reduce(counts, op=ReduceOp.SUM)`
3. Compute target: `target_count = total_counts / num_experts`
4. Update bias: `correction_bias += lr_bias * (target_count - actual_count) / target_count`
5. This is cheap — one all_reduce on an `(8,)` tensor per optimizer step

The bias steers expert selection without distorting the actual gating weights. Tokens are selected using `logits + bias` but combined using the original `logits` values.

## File structure summary

```
New:
  src/nanoplm/pretraining/models/modern_bert/moe.py    (~150-200 lines)
    - Router (nn.Module)
    - MoELayer (nn.Module)

Modified:
  src/nanoplm/pretraining/models/modern_bert/modeling.py
    - ModernBertConfig: add moe_* fields + __post_init__ validation
    - ModernBertEncoderLayer.__init__: MoE branch in MLP selection
    - init_weights: MoE weight initialization
    - ModernBertForMaskedLM.forward: aux loss accumulation

  src/nanoplm/pretraining/pure_pipeline.py
    - Guard rails (assert packing + static inp size)
    - Routing bias update hook (after backward, before optimizer step)

  pretrain.yaml
    - New moe_* config keys
```

---

## Appendix: Model-by-model research

### Qwen global-batch balancing guidance

Source: [Qwen blog: Global Batch Load Balance](https://qwenlm.github.io/blog/global-load-balance/)

- Qwen argues that micro-batch load balancing is not ideal
- domain-homogeneous micro-batches force artificial balancing and suppress specialization
- they recommend balancing over a larger/global batch
- this is the clearest official statement from a major modern MoE family about balancing philosophy

### Qwen3.5

Source: [Qwen3.5-35B-A3B README](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/README.md)

- 35B total, 3B activated, 256 experts, 8 routed + 1 shared activated
- keeps the modern shared expert pattern
- the literal activation count does not transfer to a small 8-expert model (top_k=8 on 8 experts destroys sparsity)

### Qwen3-Next

Sources: [Qwen3-Next docs](https://huggingface.co/docs/transformers/model_doc/qwen3_next), [config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking/blob/main/config.json)

- num_experts=512, num_experts_per_tok=10, shared_expert_intermediate_size=512, router_aux_loss_coef=0.001
- clearly in the shared-expert high-sparsity family
- the transferable lesson is the routing philosophy, not the exact giant-model numbers

### DeepSeek-V3

Source: [DeepSeek-V3 technical report](https://arxiv.org/html/2412.19437)

- auxiliary-loss-free load balancing via routing correction bias
- extremely small sequence-wise auxiliary loss as complement only
- no token dropping
- strongest public evidence for the frontier direction in MoE routing/balancing

### DeepSeek-V3.2

Sources: [paper page](https://huggingface.co/papers/2512.02556), [paper PDF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)

- introduces DeepSeek Sparse Attention (DSA) as main change
- MoE routing/balancing remains aligned with V3
- reinforces that frontier teams optimize the whole training system, not just router formulas

### GLM-5

Sources: [model card](https://huggingface.co/zai-org/GLM-5-FP8), [config](https://huggingface.co/zai-org/GLM-5-FP8/blob/main/config.json)

- n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8
- scoring_func="sigmoid", topk_method="noaux_tc", norm_topk_prob=true, ep_size=1
- config contains `e_score_correction_bias`
- very clearly in the no-aux / routing-correction / shared-expert family
- strongest structural match to our desired direction

### MiniMax-M2

Sources: [docs](https://huggingface.co/docs/transformers/model_doc/minimax_m2), [config commit](https://huggingface.co/MiniMaxAI/MiniMax-M2/commit/44cefa66f81ec7ffebcb580c02985726ef4d829e), [M2.1 config](https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/main/config.json)

- num_experts_per_tok=8, num_local_experts=256, router_aux_loss_coef=0.001
- use_routing_bias=true, e_score_correction_bias exposed in M2.1 config
- also in the routing-bias-corrected modern family

## Source list

- [Qwen blog: Global Batch Load Balance](https://qwenlm.github.io/blog/global-load-balance/)
- [Qwen3.5-35B-A3B README](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/README.md)
- [Qwen3-Next docs](https://huggingface.co/docs/transformers/model_doc/qwen3_next)
- [Qwen3-Next-80B-A3B config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking/blob/main/config.json)
- [DeepSeek-V3 technical report](https://arxiv.org/html/2412.19437)
- [DeepSeek-V3.2 paper page](https://huggingface.co/papers/2512.02556)
- [DeepSeek-V3.2 official paper PDF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)
- [DeepSeek-V3.2-Exp README](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/README.md)
- [GLM-5-FP8 model card](https://huggingface.co/zai-org/GLM-5-FP8)
- [GLM-5-FP8 config](https://huggingface.co/zai-org/GLM-5-FP8/blob/main/config.json)
- [MiniMax-M2 docs](https://huggingface.co/docs/transformers/model_doc/minimax_m2)
- [MiniMax-M2 config commit with `use_routing_bias`](https://huggingface.co/MiniMaxAI/MiniMax-M2/commit/44cefa66f81ec7ffebcb580c02985726ef4d829e)
- [MiniMax-M2.1 config](https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/main/config.json)
- [PyTorch grouped_mm docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html)
