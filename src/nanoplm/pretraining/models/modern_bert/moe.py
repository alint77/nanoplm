"""Mixture-of-Experts layer for nanoPLM.

Uses ``torch.nn.functional.grouped_mm`` for expert compute — a CUTLASS-backed
grouped GEMM that handles jagged per-expert token counts natively.  No capacity
factor, no token dropping, no padding to uniform capacity.

Only supports the packed flat-token static-shape forward path
(``use_packing=True``, ``use_static_inp_size=True``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Sigmoid top-k router with optional routing-bias correction."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        bias_correction: bool = True,
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        if bias_correction:
            # Correction bias: added to logits for expert *selection* only,
            # NOT included in the combine weights.  Updated externally based
            # on global expert-load statistics (DeepSeek-V3 style).
            self.register_buffer(
                "correction_bias", torch.zeros(num_experts), persistent=True,
            )
        else:
            self.correction_bias = None

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: ``(total_tokens, hidden_size)``

        Returns:
            weights: ``(total_tokens, top_k)`` — combine weights (from raw
                     sigmoid scores, bias-free, renormalized).
            indices: ``(total_tokens, top_k)`` — selected expert indices.
        """
        logits = self.gate(x).sigmoid()  # (T, E)

        if self.correction_bias is not None:
            # Use biased logits for selection, raw logits for combine weights.
            selection_logits = logits + self.correction_bias
        else:
            selection_logits = logits

        weights, indices = torch.topk(selection_logits, self.top_k, dim=-1)

        # Combine weights come from the *unbiased* sigmoid scores.
        if self.correction_bias is not None:
            weights = logits.gather(dim=-1, index=indices)

        # Renormalize so combine weights sum to 1 per token.
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices


class MoELayer(nn.Module):
    """Drop-in MLP replacement with Mixture-of-Experts.

    Forward signature matches ``ModernBertSwiGLUMLP`` so the encoder layer
    can swap it in without any call-site changes.
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .modeling import ModernBertSwiGLUMLP

        self.num_experts: int = config.moe_num_experts
        self.top_k: int = config.moe_top_k
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = config.intermediate_size

        # Router
        self.router = Router(
            config.hidden_size,
            self.num_experts,
            self.top_k,
            bias_correction=config.moe_use_bias_correction,
        )

        # Routed expert weights — stacked for grouped_mm.
        # Layout: (num_experts, in_features, out_features).
        # This is transposed relative to nn.Linear's (out, in).
        self.Wi = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, 2 * config.intermediate_size),
        )
        self.Wo = nn.Parameter(
            torch.empty(self.num_experts, config.intermediate_size, config.hidden_size),
        )
        self.drop = nn.Dropout(config.mlp_dropout)

        # Shared expert — a normal MLP that processes all tokens.
        self.shared_expert = ModernBertSwiGLUMLP(config)

        # Bookkeeping set during forward for external consumption.
        self.last_expert_counts: Optional[torch.Tensor] = None
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Handle both packed 2D (T, H) and padded 3D (B, S, H) input.
        orig_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        T, H = x.shape

        # --- shared expert (processes all tokens) ---
        shared_out = self.shared_expert(
            x,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )

        # --- routing ---
        weights, indices = self.router(x)  # (T, top_k), (T, top_k)

        # --- dispatch: expand, sort by expert, compute offs ---
        x_expanded = x.repeat_interleave(self.top_k, dim=0)  # (T*top_k, H)
        expert_flat = indices.flatten()  # (T*top_k,)
        sorted_idx = expert_flat.argsort(stable=True)
        x_sorted = x_expanded[sorted_idx]  # (T*top_k, H)

        counts = expert_flat.bincount(minlength=self.num_experts)
        offs = counts.cumsum(0).to(torch.int32)  # (num_experts,)

        # Store for external bias-update logic.
        self.last_expert_counts = counts.detach()

        # --- expert SwiGLU via grouped_mm ---
        wi = F.grouped_mm(x_sorted, self.Wi, offs=offs)  # (T*top_k, 2*inter)
        x_proj, gate = wi.chunk(2, dim=-1)
        activated = F.silu(gate) * x_proj  # (T*top_k, inter)
        expert_out_sorted = F.grouped_mm(
            self.drop(activated), self.Wo, offs=offs,
        )  # (T*top_k, H)

        # --- combine: unsort + weighted sum ---
        expert_out = torch.empty_like(expert_out_sorted)
        expert_out[sorted_idx] = expert_out_sorted
        expert_out = expert_out.view(T, self.top_k, H)
        expert_out = (expert_out * weights.unsqueeze(-1)).sum(dim=1)  # (T, H)

        # --- optional aux loss (sequence-level balance regularizer) ---
        frac = counts.float() / counts.sum()
        target = torch.ones_like(frac) / self.num_experts
        self.last_aux_loss = ((frac - target) ** 2).sum() * self.num_experts

        out = expert_out + shared_out
        if len(orig_shape) == 3:
            out = out.view(orig_shape)
        return out
