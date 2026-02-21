"""Pure PyTorch ModernBERT for masked language modeling.

The model is intentionally small and readable:
- pre-norm transformer blocks
- RoPE attention (full + sliding-window layers)
- GLU MLP (or SwiGLU replacement)
- explicit, centralized initialization
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Manifold-Constrained Hyper-Connections (mHC) — POC flag
# Set USE_MHC=True to enable multi-stream residual connections.
# Only the residual-stream expansion is implemented; resid_lambdas and
# x0_lambdas are assumed False.  See arXiv:2512.24880.
# ---------------------------------------------------------------------------
USE_MHC: bool = True
MHC_N: int = 4  # number of residual streams (n in the paper)

_HAS_FLASH_VARLEN = False
_flash_varlen_fn = None
_FLASH_HAS_DROPOUT = False

if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
    try:
        # FA3 (H100 / sm90)
        from kernels import get_kernel

        _fa3 = get_kernel("varunneal/flash-attention-3")
        _fa3 = getattr(_fa3, "flash_attn_interface", _fa3)
        _flash_varlen_fn = _fa3.flash_attn_varlen_func
        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = False  # FA3 removed dropout_p
    except ImportError:
        pass

if not _HAS_FLASH_VARLEN:
    try:
        # FA2 (Ampere+, RTX 30xx/40xx/50xx)
        from flash_attn import flash_attn_varlen_func as _flash_varlen_fn

        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = True  # FA2 supports dropout_p
    except ImportError:
        pass


@dataclass
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    mlp_activation: str = "swiglu"
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    norm_eps: float = 1e-5
    norm_bias: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: Optional[int] = None
    unk_token_id: int = 2
    mask_token_id: int = 3
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    embedding_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    decoder_bias: bool = True
    classifier_bias: bool = False
    classifier_activation: str = "gelu"
    sparse_prediction: bool = False
    sparse_pred_ignore_index: int = -100
    tie_word_embeddings: bool = True
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    use_resid_lambdas: bool = False
    use_x0_lambdas: bool = False
    use_qk_norm: bool = False
    resid_lambda_init: float = 1.0
    x0_lambda_init: float = 0.1
    # mHC settings (used only when USE_MHC=True at module level)
    mhc_n: int = MHC_N  # number of residual streams

    head_dim: int = field(init=False)
    sliding_window: int = field(init=False)
    layer_types: list[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{self.hidden_size} vs {self.num_attention_heads}"
            )

        attn_stride = max(1, int(self.global_attn_every_n_layers))
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.sliding_window = self.local_attention // 2
        self.mlp_activation = self.mlp_activation.lower()
        if self.mlp_activation not in {"swiglu", "glu"}:
            raise ValueError(
                f"Unsupported mlp_activation: {self.mlp_activation}. Supported: ['swiglu', 'glu']"
            )
        self.layer_types = [
            "full_attention" if i % attn_stride == 0 else "sliding_attention"
            for i in range(self.num_hidden_layers)
        ]


def _get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: (B, H, S, D), cos/sin: (1, S, D)
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _full_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.bool().all():
        return None
    return attention_mask[:, None, None, :].bool()


def _sliding_attention_mask(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    sliding_window: int,
    device: torch.device,
) -> torch.Tensor:
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    mask = (q - kv).abs() <= sliding_window
    mask = mask[None, None, :, :]

    if attention_mask is not None and not attention_mask.bool().all():
        mask = mask & attention_mask[:, None, None, :].bool()

    return mask


# ---------------------------------------------------------------------------
# Unpadding helpers for flash_attn_varlen_func
# ---------------------------------------------------------------------------


def _unpad_input(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive unpadding metadata from a (batch, seq_len) attention mask.

    Returns:
        indices:    (total_tokens,) – flat indices of non-padding positions.
        cu_seqlens: (batch + 1,)    – cumulative sequence lengths (int32).
        max_seqlen: 0-D int32 tensor – longest sequence in the batch.
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max()  # keep as tensor to avoid torch.compile graph break
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _pad_output(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Scatter a flat (total_tokens, …) tensor back to (batch, seqlen, …)."""
    out = torch.zeros(
        (batch * seqlen, *hidden.shape[1:]),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    out[indices] = hidden
    return out.view(batch, seqlen, *hidden.shape[1:])


def _position_ids_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    total: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert cu_seqlens to per-token position IDs (reset to 0 per sequence).

    Example: cu_seqlens=[0,3,5,9] → [0,1,2, 0,1, 0,1,2,3]
    """
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    offsets = cu_seqlens[:-1].repeat_interleave(seq_lens)
    return torch.arange(total, device=device, dtype=torch.int32) - offsets


# ---------------------------------------------------------------------------
# mHC building blocks (pure PyTorch reference implementation)
# ---------------------------------------------------------------------------


def _mhc_sinkhorn_unrolled(logits: torch.Tensor, tmax: int = 20, eps: float = 1e-6) -> torch.Tensor:
    # Extract streams
    r0 = logits[..., 0:4]
    r1 = logits[..., 4:8]
    r2 = logits[..., 8:12]
    r3 = logits[..., 12:16]

    r0 = torch.exp(r0 - torch.amax(r0, dim=-1, keepdim=True))
    r1 = torch.exp(r1 - torch.amax(r1, dim=-1, keepdim=True))
    r2 = torch.exp(r2 - torch.amax(r2, dim=-1, keepdim=True))
    r3 = torch.exp(r3 - torch.amax(r3, dim=-1, keepdim=True))

    for _ in range(tmax):
        # Unrolled sum for denominator
        s0 = r0[..., 0:1] + r0[..., 1:2] + r0[..., 2:3] + r0[..., 3:4]
        s1 = r1[..., 0:1] + r1[..., 1:2] + r1[..., 2:3] + r1[..., 3:4]
        s2 = r2[..., 0:1] + r2[..., 1:2] + r2[..., 2:3] + r2[..., 3:4]
        s3 = r3[..., 0:1] + r3[..., 1:2] + r3[..., 2:3] + r3[..., 3:4]

        r0 = r0 / (s0 + eps)
        r1 = r1 / (s1 + eps)
        r2 = r2 / (s2 + eps)
        r3 = r3 / (s3 + eps)

        c = r0 + r1 + r2 + r3 + eps
        r0 = r0 / c
        r1 = r1 / c
        r2 = r2 / c
        r3 = r3 / c

    # Stack or concat back optimally
    return torch.cat([r0, r1, r2, r3], dim=-1).view(*logits.shape[:-1], 4, 4)


class MHCLayer(nn.Module):
    """Manifold-Constrained Hyper-Connections wrapper (arXiv:2512.24880).

    Wraps a sublayer (attention or MLP) that expects a single-stream tensor
    of shape ``(..., C)`` and lifts it to operate on an *n*-stream residual
    state of shape ``(..., n, C)``.

    Following the paper exactly:
    - ``C`` is the FULL hidden_size (same as the baseline model width).
    - The state ``x ∈ R^{n×C}`` has n streams each of size hidden_size.
    - Pre-aggregation ``h_pre @ x → (1, C)`` feeds the sublayer at full width.
    - The sublayer is identical to the baseline (no parameter reduction).
    - Only the routing projections (phi) are added: phi ∈ R^{nC × (n²+2n)},
      which is negligible overhead (e.g. 3072×24 for n=4, C=768).

    Forward:
        1. Compute routing coefficients from the flattened n×C state via
           fused RMS-norm + linear projection.
        2. Pre-aggregate n streams to one via h_pre: (n,C) → (C,).
        3. Run the wrapped sublayer on the full-width single-stream result.
        4. Scatter back to n streams: h_res @ x + h_post^T * f_out.

    Parameters:
        layer:      The sublayer to wrap (ModernBertAttention or MLP).
        n:          Number of residual streams (``MHC_N``).
        c:          Full per-stream channel dimension (= ``hidden_size``).
        tmax:       Sinkhorn-Knopp iterations (default 20).
        rms_eps:    Epsilon for RMS normalisation of the flattened state.
        sinkhorn_eps: Epsilon inside Sinkhorn iterations.
        post_mult:  Scale applied to ``h_post`` after sigmoid (default 2).
    """

    def __init__(
        self,
        layer: nn.Module,
        n: int,
        c: int,
        tmax: int = 20,
        rms_eps: float = 1e-6,
        sinkhorn_eps: float = 1e-6,
        post_mult: float = 2.0,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.n = n
        self.c = c
        self.tmax = tmax
        self.rms_eps = rms_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.post_mult = post_mult

        k = n * c            # flattened stream dimension
        m = n * n + 2 * n    # total routing outputs: h_res (n²) + h_pre (n) + h_post (n)

        # Learnable projection: flattened stream → routing logits
        self.phi = nn.Parameter(torch.empty(k, m))
        # Learnable bias (float32 for stability)
        self.b = nn.Parameter(torch.zeros(m))
        # Per-part learnable scale factors
        self.alpha_pre  = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res  = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.phi, std=0.02)

    # ------------------------------------------------------------------

    def _coeffs(self, x: torch.Tensor):
        """Compute (h_pre, h_post, h_res) from x: (..., n, c).

        Uses self.n / self.c (fixed at construction) so torch.compile(dynamic=False)
        sees fully-static shapes in the coefficient computation graph.
        """
        n, c = self.n, self.c
        lead = x.shape[:-2]   # (T,) for varlen; (B,S) for SDPA
        x_mat = x.reshape(-1, n * c)   # (T, k) keep in native precision (bfloat16)

        # RMS-norm over the flattened dimension, calculate in float32 for stability
        invr = torch.rsqrt(x_mat.pow(2).float().mean(dim=-1, keepdim=True) + self.rms_eps)
        # Project using native precision, then scale.
        mix  = (x_mat @ self.phi.to(x_mat.dtype)).float() * invr   # (T, m)

        # Fused Splitting + Sigmoids + Sinkhorn iterations entirely in PyTorch for pure Dynamo integration
        pre_logits  = mix[..., :n]     * self.alpha_pre.to(mix.dtype)  + self.b[:n].to(mix.dtype)
        post_logits = mix[..., n:2*n]  * self.alpha_post.to(mix.dtype) + self.b[n:2*n].to(mix.dtype)
        res_logits  = mix[..., 2*n:] * self.alpha_res.to(mix.dtype) + self.b[2*n:].to(mix.dtype)

        h_pre  = torch.sigmoid(pre_logits.float()).to(x.dtype)
        h_post = (torch.sigmoid(post_logits.float()) * self.post_mult).to(x.dtype)
        h_res  = _mhc_sinkhorn_unrolled(res_logits.float(), self.tmax, self.sinkhorn_eps).to(x.dtype)

        # Restore leading dims (no-op for varlen where lead=(T,) already)
        h_pre  = h_pre.reshape(*lead, n)
        h_post = h_post.reshape(*lead, n)
        h_res  = h_res.reshape(*lead, n, n)
        return h_pre, h_post, h_res

    def forward(self, x: torch.Tensor, **layer_kwargs) -> torch.Tensor:
        """
        Args:
            x:            Multi-stream state ``(..., n, C)`` where C = hidden_size.
            **layer_kwargs: Passed through verbatim to the wrapped sublayer.

        Returns:
            Updated multi-stream state ``(..., n, C)``.
        """
        h_pre, h_post, h_res = self._coeffs(x)

        # Pre-aggregate: h_pre-weighted sum over stream dim → (..., C) at full width
        x_in = (x * h_pre.unsqueeze(-1)).sum(dim=-2)

        # Run sublayer on full-width single-stream input
        f_out = self.layer(x_in, **layer_kwargs)  # (..., C)

        f_out_flat = f_out.unsqueeze(-2)  # (..., 1, C)
        hp = h_post.unsqueeze(-1)         # (..., n, 1)

        # Replaced custom Triton bmm with explicit loops. Inductor correctly parses
        # and unrolls nested loops to form a single optimal pointwise fusion block 
        # without exceeding internal XBLOCK sizing parameters.
        res_gather = torch.zeros_like(x)
        for i in range(self.n):
            acc = 0.0
            for j in range(self.n):
                acc = acc + h_res[..., i:i+1, j:j+1] * x[..., j:j+1, :]
            res_gather[..., i:i+1, :] = acc
            
        return res_gather + hp * f_out_flat


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            "inv_freq_full",
            self._build_inv_freq(config.global_rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_sliding",
            self._build_inv_freq(config.local_rope_theta),
            persistent=False,
        )

    def _build_inv_freq(self, theta: float) -> torch.Tensor:
        channel = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
        return 1.0 / (theta ** (channel / self.config.head_dim))

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_type == "full_attention":
            inv_freq = self.inv_freq_full
        else:
            inv_freq = self.inv_freq_sliding

        inv_freq = inv_freq.to(device=device)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None].to(dtype=dtype), emb.sin()[None].to(dtype=dtype)


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.Wo = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = _get_activation(config.hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x_proj) * gate))


class ModernBertSwiGLUMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.Wo = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )
        self.drop = nn.Dropout(config.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.drop(F.silu(gate) * x_proj))


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.use_qk_norm = config.use_qk_norm
        self.dropout = config.attention_dropout
        self.scale = self.head_dim ** -0.5

        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.attention_bias,
        )
        self.Wo = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )

    # -- varlen (flash-attention) path -----------------------------------------

    def _forward_varlen(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        total = x.shape[0]  # (total_tokens, hidden)
        qkv = self.Wqkv(x).view(total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each: (total, H, D)

        cos, sin = cos_sin
        q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        # When max_seqlen is already a plain int (static-shape mode) skip .item()
        # to avoid a graph break.  For tensor values (dynamic mode) .item() is
        # fine — flash_attn is an opaque C extension at a graph boundary.
        max_s = max_seqlen if isinstance(max_seqlen, int) else max_seqlen.item()
        kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            softmax_scale=self.scale,
            window_size=window_size,
        )
        if _FLASH_HAS_DROPOUT:
            kwargs["dropout_p"] = self.dropout if self.training else 0.0

        y = _flash_varlen_fn(q, k, v, **kwargs)
        if isinstance(y, tuple):
            y = y[0]

        y = y.contiguous().view(total, -1)  # (total, hidden)
        return self.out_drop(self.Wo(y))

    # -- SDPA (fallback) path --------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            return self._forward_varlen(x, cos_sin, cu_seqlens, max_seqlen, window_size)

        bsz, seq_len, _ = x.shape
        qkv = self.Wqkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = cos_sin
        q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            scale=self.scale,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_drop(self.Wo(y))


class _NormedAttn(nn.Module):
    """Thin wrapper combining pre-norm + attention for use inside MHCLayer."""

    def __init__(self, norm: nn.Module, attn: ModernBertAttention) -> None:
        super().__init__()
        self.norm = norm
        self.attn = attn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.attn(self.norm(x), **kwargs)


class _NormedMLP(nn.Module):
    """Thin wrapper combining pre-norm + MLP for use inside MHCLayer."""

    def __init__(self, norm: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.mlp = mlp

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.mlp(self.norm(x))


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.use_mhc = USE_MHC

        if USE_MHC:
            # Paper §3: the sublayer F operates at the FULL hidden_size C.
            # h_pre aggregates n streams → one C-dim vector fed into the
            # unmodified attention/MLP.  No inner_cfg needed.
            n = config.mhc_n
            c = config.hidden_size  # full width — matches paper exactly
            attn_norm = (
                nn.Identity()
                if layer_idx == 0
                else nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            )
            mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            attn = ModernBertAttention(config)
            mlp = (
                ModernBertSwiGLUMLP(config)
                if config.mlp_activation == "swiglu"
                else ModernBertMLP(config)
            )
            self.mhc_attn = MHCLayer(_NormedAttn(attn_norm, attn), n=n, c=c)
            self.mhc_mlp  = MHCLayer(_NormedMLP(mlp_norm, mlp),   n=n, c=c)
        else:
            self.attn_norm = (
                nn.Identity()
                if layer_idx == 0
                else nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            )
            self.attn = ModernBertAttention(config)
            self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.mlp = (
                ModernBertSwiGLUMLP(config)
                if config.mlp_activation == "swiglu"
                else ModernBertMLP(config)
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.use_mhc:
            # x: (T, n, c) — multi-stream state
            # MHCLayer returns the updated multi-stream state in-place of the residual.
            # The residual connection is embedded inside MHCLayer (h_res mixing).
            x = self.mhc_attn(
                x,
                cos_sin=cos_sin,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                window_size=window_size,
            )
            x = self.mhc_mlp(x)
            return x

        x = x + self.attn(
            self.attn_norm(x),
            cos_sin=cos_sin,
            attn_mask=attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            window_size=window_size,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.use_mhc = USE_MHC
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        if config.use_resid_lambdas:
            self.resid_lambdas = nn.Parameter(torch.ones(config.num_hidden_layers))
        else:
            self.register_parameter("resid_lambdas", None)
        if config.use_x0_lambdas:
            self.x0_lambdas = nn.Parameter(torch.zeros(config.num_hidden_layers))
        else:
            self.register_parameter("x0_lambdas", None)
        # final_norm and rotary_emb are identical to baseline in mHC:
        # the sublayer and the collapsed output both operate at full hidden_size.
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_emb = ModernBertRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        _cu_seqlens: Optional[torch.Tensor] = None,
        _max_seqlen: Optional[int] = None,
        _position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ---- varlen (flash-attention) path --------------------------------
        if _cu_seqlens is not None:
            device = input_ids.device
            x = self.embeddings(input_ids)  # (total_tokens, hidden_size)
            x0 = x if self.x0_lambdas is not None else None

            if self.use_mhc:
                # Paper §3: expand (T, C) → (T, n, C) by repeating the embedding
                # across n streams.  All streams start identical and specialise
                # during training via the learned h_res mixing.
                x = x.unsqueeze(1).expand(-1, self.config.mhc_n, -1).contiguous()

            # Pre-compute RoPE tables up to max_position_embeddings (fixed size
            # avoids graph breaks / recompilation) and index by _position_ids.
            rope_len = self.config.max_position_embeddings
            cos_f, sin_f = self.rotary_emb(
                rope_len, device, x.dtype, "full_attention"
            )
            cos_s, sin_s = self.rotary_emb(
                rope_len, device, x.dtype, "sliding_attention"
            )
            rope = {
                "full_attention": (
                    cos_f[0, _position_ids],
                    sin_f[0, _position_ids],
                ),
                "sliding_attention": (
                    cos_s[0, _position_ids],
                    sin_s[0, _position_ids],
                ),
            }
            windows = {
                "full_attention": (-1, -1),
                "sliding_attention": (
                    self.config.sliding_window,
                    self.config.sliding_window,
                ),
            }

            for i, layer in enumerate(self.layers):
                if self.resid_lambdas is not None:
                    x = self.resid_lambdas[i] * x
                if self.x0_lambdas is not None:
                    x = x + self.x0_lambdas[i] * x0
                lt = layer.attention_type
                x = layer(
                    x,
                    cos_sin=rope[lt],
                    cu_seqlens=_cu_seqlens,
                    max_seqlen=_max_seqlen,
                    window_size=windows[lt],
                )

            if self.use_mhc:
                # Collapse (T, n, C) → (T, C): mean over streams, then norm.
                x = self.final_norm(x.mean(dim=1))
            else:
                x = self.final_norm(x)
            return x

        # ---- SDPA (fallback) path -----------------------------------------
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}"
            )
        device = input_ids.device

        x = self.embeddings(input_ids)
        x0 = x if self.x0_lambdas is not None else None

        if self.use_mhc:
            # (B, S, C) → (B, S, n, C)
            x = x.unsqueeze(2).expand(-1, -1, self.config.mhc_n, -1).contiguous()

        attn_masks = {
            "full_attention": _full_attention_mask(attention_mask),
            "sliding_attention": _sliding_attention_mask(
                attention_mask,
                seq_len=seq_len,
                sliding_window=self.config.sliding_window,
                device=device,
            ),
        }

        rope = {
            "full_attention": self.rotary_emb(
                seq_len=seq_len,
                device=device,
                dtype=x.dtype,
                layer_type="full_attention",
            ),
            "sliding_attention": self.rotary_emb(
                seq_len=seq_len,
                device=device,
                dtype=x.dtype,
                layer_type="sliding_attention",
            ),
        }

        for i, layer in enumerate(self.layers):
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[i] * x
            if self.x0_lambdas is not None:
                x = x + self.x0_lambdas[i] * x0
            layer_type = layer.attention_type
            x = layer(x, attn_mask=attn_masks[layer_type], cos_sin=rope[layer_type])

        if self.use_mhc:
            # Collapse (B, S, n, C) → (B, S, C): mean over streams, then norm.
            x = self.final_norm(x.mean(dim=2))
        else:
            x = self.final_norm(x)
        return x


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.classifier_bias,
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.act = _get_activation(config.classifier_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(x)))


class ModernBertForMaskedLM(nn.Module):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        self.init_weights()

        if config.tie_word_embeddings:
            self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    @torch.no_grad()
    def init_weights(self) -> None:
        
        width = self.config.hidden_size
        bound = math.sqrt(3.0 / width)
        embedding_std = 0.02 if self.config.tie_word_embeddings else 1.0

        nn.init.normal_(
            self.model.embeddings.tok_embeddings.weight,
            mean=0.0,
            std=embedding_std,
        )

        for module in self.modules():
            if module.__class__.__name__ == "RMSNorm":
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

        for layer in self.model.layers:
            if USE_MHC:
                # Initialise the inner attn/MLP accessed via MHCLayer wrappers.
                attn = layer.mhc_attn.layer.attn
                mlp  = layer.mhc_mlp.layer.mlp
            else:
                attn = layer.attn
                mlp  = layer.mlp

            nn.init.uniform_(attn.Wqkv.weight, -bound, bound)
            nn.init.zeros_(attn.Wo.weight)
            nn.init.uniform_(mlp.Wi.weight, -bound, bound)
            nn.init.zeros_(mlp.Wo.weight)

            if attn.Wqkv.bias is not None:
                nn.init.zeros_(attn.Wqkv.bias)
            if attn.Wo.bias is not None:
                nn.init.zeros_(attn.Wo.bias)
            if mlp.Wi.bias is not None:
                nn.init.zeros_(mlp.Wi.bias)
            if mlp.Wo.bias is not None:
                nn.init.zeros_(mlp.Wo.bias)

            if USE_MHC:
                # phi already initialised in MHCLayer.__init__ with normal_(0.02)
                # b, alpha_* already set to zeros/ones; nothing extra needed here.
                pass

        nn.init.uniform_(self.head.dense.weight, -bound, bound)
        if self.head.dense.bias is not None:
            nn.init.zeros_(self.head.dense.bias)

        decoder_std = embedding_std if self.config.tie_word_embeddings else 0.001
        nn.init.normal_(self.decoder.weight, mean=0.0, std=decoder_std)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        if self.model.resid_lambdas is not None:
            self.model.resid_lambdas.fill_(self.config.resid_lambda_init)
        if self.model.x0_lambdas is not None:
            self.model.x0_lambdas.fill_(self.config.x0_lambda_init)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings.tok_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        # ---- static packed path (pre-flattened by collator) ---------------
        # When both cu_seqlens and position_ids are provided and input_ids is
        # already 1-D (flat), the collator has done all unpadding / position-id
        # computation.  No data-dependent ops here → dynamic=False safe.
        if cu_seqlens is not None and position_ids is not None and input_ids.dim() == 1:
            x = self.model(
                input_ids,
                _cu_seqlens=cu_seqlens,
                _max_seqlen=max_seqlen,  # pass int directly — no tensor
                _position_ids=position_ids,
            )
            # x: (F, hidden) where F is fixed flat length

            if labels is not None:
                logits = self.decoder(self.head(x))
                loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    ignore_index=self.sparse_pred_ignore_index,
                )
            else:
                logits = self.decoder(self.head(x))
                loss = None

            return {"loss": loss, "logits": logits}

        use_varlen = (
            _HAS_FLASH_VARLEN
            and input_ids.is_cuda
            and attention_mask is not None
        )

        # ---- varlen (flash-attention) path --------------------------------
        if use_varlen:
            batch, seq_len = input_ids.shape
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

            if cu_seqlens is not None:
                # Packed path: cu_seqlens provided by the packing collator.
                # max_seqlen may be an int or a tensor; keep consistent.
                if isinstance(max_seqlen, int):
                    max_seqlen_t = torch.tensor(max_seqlen, dtype=torch.int32)
                else:
                    max_seqlen_t = max_seqlen
            else:
                # Unpacked path: derive cu_seqlens from attention_mask.
                _indices, cu_seqlens, max_seqlen_t = _unpad_input(attention_mask)
                indices = _indices

            flat_ids = input_ids.view(-1)[indices]  # (total_tokens,)
            position_ids = _position_ids_from_cu_seqlens(
                cu_seqlens, flat_ids.shape[0], flat_ids.device
            )

            x = self.model(
                flat_ids,
                _cu_seqlens=cu_seqlens,
                _max_seqlen=max_seqlen_t,
                _position_ids=position_ids,
            )
            # x: (total_tokens, hidden) — flat, no padding

            if self.sparse_prediction and labels is not None:
                flat_labels = labels.view(-1)[indices]
                keep = flat_labels != self.sparse_pred_ignore_index
                logits = self.decoder(self.head(x[keep]))
                loss = F.cross_entropy(logits.float(), flat_labels[keep])
            elif labels is not None:
                flat_labels = labels.view(-1)[indices]
                logits = self.decoder(self.head(x))
                loss = F.cross_entropy(
                    logits.float(),
                    flat_labels,
                    ignore_index=self.sparse_pred_ignore_index,
                )
            else:
                logits = self.decoder(self.head(x))
                logits = _pad_output(logits, indices, batch, seq_len)
                loss = None

            return {"loss": loss, "logits": logits}

        # ---- SDPA (fallback) path -----------------------------------------
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.sparse_prediction and labels is not None:
            flat_labels = labels.view(-1)
            x = x.view(flat_labels.shape[0], -1)
            keep = flat_labels != self.sparse_pred_ignore_index
            x = x[keep]
            flat_labels = flat_labels[keep]
        else:
            flat_labels = labels

        logits = self.decoder(self.head(x))

        loss = None
        if labels is not None:
            if self.sparse_prediction:
                loss = F.cross_entropy(logits.float(), flat_labels)
            else:
                loss = F.cross_entropy(
                    logits.float().view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=self.sparse_pred_ignore_index,
                )

        return {"loss": loss, "logits": logits}

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable)
        )


def map_hf_state_dict_to_pure(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in hf_state_dict.items() if not k.startswith("_")}


def map_pure_state_dict_to_hf(pure_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return dict(pure_state_dict)
