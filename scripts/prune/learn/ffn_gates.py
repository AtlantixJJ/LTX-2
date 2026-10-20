"""Learn individual FFN choices with binary forwards and aligned retained counts.

The update owner selects once before each optimizer update. The microbatch
owner makes a new sigmoid surrogate graph. Providers reuse that graph through
checkpoint recomputation. Fixed masks retain a zero-gradient score path for
DDP. Shape, count, finiteness and channel-order failures are explicit.
"""

from __future__ import annotations

import torch
from torch import nn

from scripts.prune.core.records import tensor_hash


def aligned_width(update, width, kept, *, alignment=128, warmup=50, ramp_end=300):
    if any(type(x) is not int for x in (update, width, kept, alignment, warmup, ramp_end)):
        raise ValueError("FFN width schedule requires integers")
    if update < 0 or alignment < 1 or not 0 < kept <= width or width % alignment or kept % alignment:
        raise ValueError("invalid aligned FFN width")
    if not 0 <= warmup < ramp_end:
        raise ValueError("invalid FFN removal ramp")
    packets = (width - kept) // alignment
    elapsed = min(max(update - warmup, 0), ramp_end - warmup)
    return width - alignment * (packets * elapsed // (ramp_end - warmup))


def select_channels(scores, kept):
    if scores.ndim != 2 or not torch.isfinite(scores).all() or type(kept) is not int or not 0 < kept <= scores.shape[1]:
        raise ValueError("invalid FFN scores or retained count")
    # Stable sorting resolves exact ties by lower original channel ID on CPU and CUDA.
    order = torch.argsort(scores.detach(), dim=1, descending=True, stable=True)
    return torch.zeros_like(scores).scatter_(1, order[:, :kept], 1)


class FFNGates(nn.Module):
    def __init__(self, blocks=48, width=16384, kept=14848, *, alignment=128, warmup=50, ramp_end=300):
        super().__init__()
        if type(blocks) is not int or blocks < 1:
            raise ValueError("FFN block count must be positive")
        aligned_width(0, width, kept, alignment=alignment, warmup=warmup, ramp_end=ramp_end)
        self.alpha = nn.Parameter(torch.zeros(blocks, width, dtype=torch.float32))
        self.width, self.kept = width, kept
        self.schedule = dict(alignment=alignment, warmup=warmup, ramp_end=ramp_end)
        self.decision = torch.ones_like(self.alpha)
        self.current = None
        self.fixed = False

    def begin_update(self, update, *, all_one=False):
        if self.fixed and not all_one:
            return
        kept = self.width if all_one else aligned_width(update, self.width, self.kept, **self.schedule)
        self.decision = select_channels(self.alpha, kept)

    def freeze(self):
        self.decision = select_channels(self.alpha, self.kept)
        self.fixed = True

    def microbatch(self):
        if self.decision.device != self.alpha.device:
            self.decision = self.decision.to(self.alpha.device)
        p = self.alpha.sigmoid()
        self.current = self.decision + (p * 0 if self.fixed else p - p.detach())
        return self.current

    def provider(self, name):
        if not name.endswith(".ff") or self.current is None:
            raise ValueError("FFN provider requires a current forward graph")
        return self.current[int(name.split(".")[0])]

    def snapshot(self):
        return dict(
            retained_per_block=self.decision.detach().sum(1).int().cpu().tolist(),
            total_channels=self.alpha.numel(),
            removed_channels=int((self.decision == 0).sum()),
            mask_sha256=tensor_hash(self.decision),
            fixed=self.fixed,
        )
