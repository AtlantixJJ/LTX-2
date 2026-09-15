"""Per-layer key/value cache for causal (autoregressive) video self-attention.

Strictly additive: every entry point below is reached only when a caller passes a cache
explicitly. With ``kv_caches=None`` -- the default on :class:`~ltx_core.model.transformer.modality.Modality`
-- not one tensor changes, so every existing bit-exactness gate is untouched by construction.

**Why a cache needs causality.** Under full bidirectional attention a context token's K/V
depend on the noisy tokens sitting beside it in the same window, so they are different in
every window and cannot be reused. Under block-causal attention a token attends only to
its own block and earlier ones, so once a block's content is final its K/V are final too --
which is exactly the condition that makes them cacheable. The two features are one feature.

**The contract, in the order the caller must use it.**

1. ``read``-only forward of the block being denoised: queries attend over
   ``[cached history | this block's own K/V]``. Nothing is written, so gradient
   checkpointing and recomputation are safe.
2. ``backward`` on that block's loss.
3. ``write`` forward of the block's *clean* latent under ``torch.no_grad()``: the same
   tokens at timestep 0, whose K/V land in the cache for every later block.

Steps 1 and 3 are separate forwards because the K/V a later block wants are those of the
*denoised* content, and the denoising forward only ever sees the noisy version of it.

``read`` hands out plain views, never copies -- a copy would be ~17 MB per layer per latent
frame at the 22B geometry, i.e. hundreds of MB per forward. That is safe because the view is
immediately consumed by ``torch.cat``, whose output is a fresh tensor, and that output (not
the view) is what autograd saves -- so a later in-place ``write`` cannot corrupt a graph built
from an earlier ``read``, whatever order the two happen in. The step ordering above is about
*semantics* (a block's cached K/V must come from its denoised content), not about autograd
safety; ``scripts/onestep_avatar/causal_core.py`` is the one place that sequences it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    """One transformer layer's cached self-attention keys and values.

    ``k``/``v`` are ``(B, capacity, heads * dim_head)`` -- the layout
    :class:`~ltx_core.model.transformer.attention.Attention` produces before it splits heads,
    so nothing is reshaped on the way in or out. Valid entries are always the prefix
    ``[0, length)``: this is a sliding *context*, not a ring buffer, because the retained
    span has to stay contiguous for a single ``cat`` and for eviction to be one copy.
    """

    k: torch.Tensor
    v: torch.Tensor
    length: int = 0

    @property
    def capacity(self) -> int:
        return self.k.shape[1]

    def read(self, end: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Views of the first ``end`` cached tokens, or ``(None, None)`` when there are none."""
        end = min(end, self.length)
        if end <= 0:
            return None, None
        return self.k[:, :end], self.v[:, :end]

    def write(self, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        """Store ``k``/``v`` at ``[start, start + tokens)``, detached.

        Detached because a cached key is an *input* to every later block, never something a
        later block's loss optimises through -- the same rule the AR carryover follows.
        """
        end = start + k.shape[1]
        if end > self.capacity:
            raise ValueError(
                f"writing tokens [{start}, {end}) overflows a {self.capacity}-token cache; "
                f"raise the capacity or evict first"
            )
        self.k[:, start:end] = k.detach().to(self.k.dtype)
        self.v[:, start:end] = v.detach().to(self.v.dtype)
        self.length = max(self.length, end)

    def keep(self, spans: list[tuple[int, int]]) -> None:
        """Compact the cache down to ``spans``, in order, packed at the front.

        Eviction is a copy rather than a ring so that the survivors stay one contiguous
        prefix. At the 22B geometry a latent frame is ~17 MB per layer, so this moves a few
        tens of MB against a forward pass that moves orders of magnitude more -- the
        simplicity is worth more than the bytes.
        """
        pieces_k = [self.k[:, start:end] for start, end in spans]
        pieces_v = [self.v[:, start:end] for start, end in spans]
        kept = sum(end - start for start, end in spans)
        if kept > self.capacity:
            raise ValueError(f"cannot keep {kept} tokens in a {self.capacity}-token cache")
        # Through a temporary: the destination overlaps the sources whenever a span moves
        # left onto tokens another span still needs.
        self.k[:, :kept] = torch.cat(pieces_k, dim=1).clone()
        self.v[:, :kept] = torch.cat(pieces_v, dim=1).clone()
        self.length = kept

    def reset(self) -> None:
        self.length = 0


def allocate_kv_caches(
    num_layers: int,
    *,
    batch_size: int,
    capacity: int,
    inner_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[LayerKVCache]:
    """One :class:`LayerKVCache` per transformer layer, allocated up front.

    Up front rather than grown on demand: the peak is the whole point of the budget
    (``2 * num_layers * capacity * inner_dim`` elements), so it should fail at allocation
    rather than mid-rollout.
    """
    return [
        LayerKVCache(
            k=torch.zeros(batch_size, capacity, inner_dim, device=device, dtype=dtype),
            v=torch.zeros(batch_size, capacity, inner_dim, device=device, dtype=dtype),
        )
        for _ in range(num_layers)
    ]
