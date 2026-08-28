from typing import List, Protocol

import torch
from torch import nn

from ltx_core.model.transformer.rope import apply_rotary_emb
from ltx_core.utils import rms_norm


class PreAttentionCallable(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: nn.Module,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class PytorchPreAttention(PreAttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: nn.Module,
        mask: torch.Tensor | None,  # noqa: ARG002
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
        if pe is not None:
            # A structurally pruned branch keeps arbitrary original heads. RoPE
            # frequencies are head-major, so retaining only the first H slices
            # would change positional encoding.  Select the recorded original
            # slices before applying RoPE; stock checkpoints leave this None.
            indices = getattr(attn_module, "rope_head_indices", None)
            def select_heads(freqs):
                if indices is None:
                    return freqs
                # RoPE is a (cos, sin) pair in the split implementation.
                # Keep this tolerant of a future tensor-only representation.
                if isinstance(freqs, tuple):
                    return tuple(value[:, indices] for value in freqs)
                return freqs[:, indices]

            q_pe = select_heads(pe)
            k_pe = select_heads(pe if k_pe is None else k_pe)
            q = apply_rotary_emb(q, q_pe, attn_module.rope_type)
            k = apply_rotary_emb(k, k_pe, attn_module.rope_type)
        return q, k


class AdaZeroCallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor: ...


class PytorchAdaZeroFunction(AdaZeroCallable):
    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return rms_norm(x, eps=eps) * (1 + scale) + shift


class PostSACallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> List[torch.Tensor]: ...


class PytorchPostSAFunction(PostSACallable):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> List[torch.Tensor]:
        x_fma = x + y * gate
        return x_fma, rms_norm(x_fma, norm_weights, eps=eps)


class GatedAttentionCallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        attn_module: nn.Module,
    ) -> torch.Tensor: ...


class PytorchGatedAttention(GatedAttentionCallable):
    def __call__(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        attn_module: nn.Module,
    ) -> torch.Tensor:
        gate_logits = attn_module.to_gate_logits(x)  # (B, T, H)
        b, t, _ = attn_out.shape
        out = attn_out.view(b, t, attn_module.heads, attn_module.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
        out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
        return out.view(b, t, attn_module.heads * attn_module.dim_head)
