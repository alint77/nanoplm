# Interleaved Head Attention (IHA)

## Overview

Interleaved Head Attention is a cross-head mixing mechanism that constructs P
pseudo-head tokens per head as learned linear combinations of all H heads' Q, K,
and V projections, interleaves them into the sequence dimension, runs standard
attention on the expanded sequence, then collapses back. This yields up to HP²
attention patterns without custom kernels.

Reference: *Interleaved-Head Attention* paper (source in
`interleaved_head_attn_paper_latex/`).

## Algorithm

Given input X ∈ R^{N×D}, H attention heads, P pseudo-heads per head, head
dimension d = D/H:

**Parameters:** α^Q, α^K, α^V ∈ R^{H×P×H} (mixing weights); α^O ∈ R^{H×P×H}
(output collapse weights). The paper calls the output matrix R ∈ R^{H×HP} — our
α^O is the same thing reshaped as (H, P, H).

**Steps:**

1. **Standard QKV projection** — project X through W_Q, W_K, W_V and reshape to
   (N, H, d).

2. **Pseudo-head mixing** — for each head h and pseudo-index p, construct a
   pseudo-Q/K/V as a learned weighted sum over all H original heads:

   ```
   Q̃[h,p] = Σ_g α^Q[h,p,g] · Q[g]      # shape: (N, d)
   ```

   Implemented as `einsum('tgd,hpg->thpd', Q, α^Q)`.

3. **Interleave into sequence** — merge the P pseudo-tokens at each position
   into the sequence dimension. Token at position n expands to P consecutive
   virtual tokens at positions [n·P, n·P+1, ..., n·P+(P-1)]. The sequence
   grows from length N to N·P per head. Implemented via
   `permute(0,2,1,3).reshape(T*P, H, d)` (varlen) or
   `permute(0,1,3,2,4).reshape(B, H, S*P, d)` (SDPA).

4. **Standard attention** on the N·P-length sequence, per head. Compatible with
   FlashAttention (varlen) and PyTorch SDPA.

5. **Un-merge and collapse** — reshape output from (T·P, H, d) back to
   (T, P, H, d), then collapse pseudo-heads via learned output mixing:

   ```
   O[h,t] = Σ_p Σ_g α^O[h,p,g] · P[g,p,t]    # shape: (d,)
   ```

   Implemented as `einsum('tpgd,hpg->thd', output, α^O)`.

6. **Concat heads and Wo projection** — standard output projection, unchanged.

## Configuration

### YAML (`pretrain.yaml`)

```yaml
model:
  use_iha: true               # enable IHA (pure_torch only)
  iha_num_pseudo_heads: 2     # P pseudo-heads per head (default: num_attention_heads)
```

### CLI

```
nanoplm pretrain run --pure-torch --use-iha --iha-num-pseudo-heads 2
```

### Config fields

| Field | Type | Default | Location |
|-------|------|---------|----------|
| `use_iha` | bool | `false` | `ProtModernBertMLMConfig`, `ModernBertConfig` |
| `iha_num_pseudo_heads` | int \| null | `null` (= num_attention_heads) | `ProtModernBertMLMConfig`, `ModernBertConfig` |

When `iha_num_pseudo_heads` is null, it defaults to `num_attention_heads` in
`ModernBertConfig.__post_init__`.

## Mutual exclusivity gates

IHA is gated against three other features. All gates produce `ValueError` at
config validation time (`ModernBertConfig.__post_init__`), before any model
construction.

| Combination | Reason |
|-------------|--------|
| IHA + DiffAttnV2 | Both modify head structure (IHA expands pseudo-heads, DiffV2 doubles Q heads for differential subtraction). Cannot compose. |
| IHA + RePO | IHA assigns distinct virtual RoPE positions to each pseudo-token; RePO replaces fixed positions with content-dependent learned positions. The two schemes conflict. |
| IHA + GQA (num_kv_heads ≠ num_attention_heads) | IHA mixes across all H heads symmetrically. Asymmetric Q/KV head counts break the cross-head mixing. |

Additional path-level gates:

| Path | Gate |
|------|------|
| HF (`ProtModernBertMLM`) | `ValueError`: IHA is pure-torch only |
| TE (`TEProtModernBertMLM`) | `ValueError`: IHA is pure-torch only |
| CLI `run()` | `click.ClickException` if `use_iha` without `--pure-torch` |
| CLI `from_yaml()` | `click.ClickException` if `model.use_iha=true` without `pure_torch: true` |

## Sliding-window-only application

Per the paper's hybrid schedule (4 sliding-window IHA layers : 1 global MHA
layer), IHA is only applied to sliding-window attention layers. Global attention
layers use standard MHA regardless of the `use_iha` flag.

This is implemented in `ModernBertAttention.__init__`:

```python
self._is_sliding = config.layer_types[layer_idx] == "sliding_attention"
self.use_iha = config.use_iha and self._is_sliding
```

Consequence: alpha parameters are only allocated on sliding layers. Global layers
have `use_iha=False` and `iha_P=0`, with zero parameter overhead.

## RoPE handling

The paper explicitly states that each pseudo-token gets a **distinct** RoPE
position:

> "giving each (n,p) its own virtual position assigns each pseudo-head token a
> distinct RoPE phase, and variable-length inference is handled by generating
> RoPE for length NP"

Token at original position n expands to P virtual positions:
`[n·P, n·P+1, ..., n·P+(P-1)]`.

### Implementation

The model forward pass pre-computes a **full unindexed RoPE table** (cos/sin for
positions 0 through max_position_embeddings-1) and passes it to IHA layers via
the `iha_rope_full` parameter. Each IHA attention module computes virtual
position indices and indexes into this table.

**Varlen path** (`_forward_varlen`):

```python
cos_full, sin_full = iha_rope_full          # (1, max_pos, head_dim)
virt_pos = (position_ids * P).unsqueeze(1) + torch.arange(P, ...)  # (T, P)
virt_pos = virt_pos.reshape(-1)             # (T*P,)
cos_sin = (cos_full[0, virt_pos], sin_full[0, virt_pos])
```

Here `position_ids` contains per-sequence positions from `cu_seqlens` (e.g.
`[0,1,2,0,1]` for two sequences of length 3 and 2), so virtual positions reset
per sequence.

**SDPA path** (`forward`):

```python
cos_full, sin_full = iha_rope_full          # (1, max_pos, head_dim)
virt_pos = torch.arange(S, ...) * P         # (S,)
virt_pos = virt_pos.unsqueeze(1) + torch.arange(P, ...)  # (S, P)
virt_pos = virt_pos.reshape(-1)             # (S*P,)
cos_sin = (cos_full[:, virt_pos], sin_full[:, virt_pos])
```

### Data flow for `iha_rope_full`

The full rope table is built once in `ModernBertModel.forward` and threaded
through to attention:

```
ModernBertModel.forward
  ├── builds _iha_rope_full dict {layer_type: (cos, sin)}
  │   (varlen: uses existing cos_f/sin_f full tables)
  │   (SDPA: calls rotary_emb(seq_len * P) for expanded length)
  └── passes iha_rope_full=... to each layer call
        └── ModernBertEncoderLayer.forward(iha_rope_full=...)
              └── ModernBertAttention.forward(iha_rope_full=...)
                    └── _forward_varlen(iha_rope_full=...) or uses it in SDPA path
```

Non-IHA layers receive `iha_rope_full=None` and ignore it. The mHC-lite sublayer
wrapper (`ModernBertAttnResidual`) passes it through via `**_kwargs`.

### Window size scaling

When IHA expands the sequence from N to N·P, the sliding window is also scaled:

```python
cu_seqlens = cu_seqlens * P
max_seqlen = max_seqlen * P
if window_size != (-1, -1):
    window_size = (window_size[0] * P, window_size[1] * P)
```

For FLOP-matching per the paper, set the base sliding window to
`W = N / (2·P²)` so that the effective window after scaling is
`W·P = N / (2·P)`, giving per-layer cost O(H · N² · d / 2).

### Attention mask scaling (SDPA only)

The SDPA path expands the attention mask to cover the N·P sequence:

```python
attn_mask = attn_mask.repeat_interleave(P, dim=-1).repeat_interleave(P, dim=-2)
```

## Initialization

Alpha parameters use **identity-like initialization** so the model starts as
exact standard MHA (in `ModernBertForMaskedLM.init_weights`):

```python
eye_H = torch.eye(H)
alpha_init = eye_H.unsqueeze(1).expand(H, P, H).clone()
# α^Q, α^K, α^V: each pseudo-head p selects exactly head h (identity)
enc.attn.alpha_q.data.copy_(alpha_init)
enc.attn.alpha_k.data.copy_(alpha_init)
enc.attn.alpha_v.data.copy_(alpha_init)
# α^O: identity / P (averages P identical pseudo-heads → original output)
enc.attn.alpha_o.data.copy_(alpha_init / P)
```

At initialization: pseudo-head (h, p) copies head h's Q/K/V exactly (identity
mixing). Output collapse averages P identical copies → recovers original head
output. Combined with zero-init on Wo, the attention branch contributes zero to
the residual stream at step 0 — standard residual stream initialization.

## Optimizer routing

IHA alpha parameters have `ndim == 3`. In the optimizer routing logic
(`src/nanoplm/pretraining/optim.py`), the classification is:

- `ndim == 1` or embedding → AdamW
- `ndim == 2` → Muon (or AdamW if zero-init-fragile)
- **else (ndim ≥ 3) → AdamW**

So alpha params are trained with AdamW, which is appropriate — they are small
(4 × H × P × H floats per layer) and don't benefit from Muon's
orthogonalization.

## Parameter count

Per IHA-enabled layer (sliding layers only):

```
4 alpha matrices × H × P × H = 4·H²·P parameters
```

Example with H=8, P=2: 4 × 64 × 2 = 512 params per layer. Negligible compared
to Wqkv (3 × hidden² = 1.77M for hidden=768).

## Cost analysis

### FLOPs

Attention FLOPs per IHA layer scale as O(H · (N·P)² · d) = O(P² · H · N² · d).
This is a **P² multiplier** on attention compute for IHA layers.

The paper FLOP-matches with a hybrid schedule:
- 4 sliding-window IHA layers with window W = N/(2P²)
- 1 global MHA layer (no IHA)
- Average cost per 5 layers ≈ O(H·N²·d), matching global MHA

### Memory

Activation memory for attention also scales with N·P:
- Q, K, V tensors: P× larger on IHA layers
- Flash attention workspace: proportional to expanded sequence length
- Activation checkpointing on attention helps (recomputes forward during
  backward) but the recomputed forward still creates P×-expanded tensors

### Practical guidance

| P | Attention cost multiplier | Recommended window (FLOP-match) |
|---|--------------------------|--------------------------------|
| 2 | 4× | N/8 |
| 3 | 9× | N/18 |
| 4 | 16× | N/32 |
| 8 | 64× | N/128 |

For small models (hidden=768, H=8), P=2 or P=3 is practical. The paper uses
P=20 on a 2.4B model with 128 H200 GPUs — that scale tolerates the overhead.

## Files

| File | What changed |
|------|-------------|
| `src/nanoplm/pretraining/models/modern_bert/modeling.py` | Core implementation: config fields, validation gates, `ModernBertAttention` (alpha params, varlen + SDPA forward paths, RoPE virtual positions), `ModernBertEncoderLayer` (iha_rope_full threading), `ModernBertModel` (full rope table construction + plumbing), `init_weights` (identity init) |
| `src/nanoplm/pretraining/models/modern_bert/model.py` | `ProtModernBertMLMConfig`: added `use_iha`, `iha_num_pseudo_heads` fields. `ProtModernBertMLM`: gate (IHA is pure-torch only) |
| `src/nanoplm/pretraining/models/modern_bert/pure_model.py` | `PureProtModernBertMLM`: passes `use_iha`, `iha_num_pseudo_heads` to `ModernBertConfig`. `TEProtModernBertMLM`: gate (IHA is pure-torch only) |
| `src/nanoplm/cli/pretrain.py` | CLI options `--use-iha`, `--iha-num-pseudo-heads`. Gates in `run()` and `from_yaml()`. YAML template updated. |
| `src/nanoplm/pretraining/optim.py` | No changes needed — alpha params (ndim=3) route to AdamW via existing else branch |

## Differences from paper

| Aspect | Paper | Our implementation |
|--------|-------|--------------------|
| Model | 2.4B decoder-only, H=20, P=20 | Encoder MLM, H/P configurable |
| Layer application | Sliding-window IHA layers only, global layers are standard MHA | Same — `use_iha and _is_sliding` |
| α^O shape | R ∈ R^{H×HP} (flat) | α^O ∈ R^{H×P×H} (reshaped, mathematically equivalent) |
| RoPE positions | Distinct virtual positions per pseudo-token | Same — virtual position n·P+p |
| FLOP-matching | Window W = N/(2P²), 4:1 sliding:global ratio | User must set `sliding_window` and `attn_layer_pattern` manually |
| Optimizer | AdamW for everything | AdamW for alpha params, Muon/NormUon for 2D weight matrices |
