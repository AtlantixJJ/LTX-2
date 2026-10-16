"""Learn scalar head decisions without changing native attention connectivity."""

from __future__ import annotations

import math

import torch
from torch import nn


def feasible_counts(allowed):
    totals = {0}
    for counts in allowed:
        if not counts or any(type(k) is not int or k < 1 for k in counts):
            raise ValueError("allowed counts must be positive integers")
        totals = {a + b for a in totals for b in counts}
    return sorted(totals)


def nearest_removed(requested, allowed, heads=32):
    if not math.isfinite(requested) or not 0 <= requested <= len(allowed) * heads:
        raise ValueError("invalid removal request")
    choices = [len(allowed) * heads - k for k in feasible_counts(allowed)]
    return min(choices, key=lambda r: (abs(r - requested), r))


def select_legal(scores, kept, allowed):
    """Exact block/count DP; equal scores keep lower original head identities."""
    scores = scores.detach().double().cpu()
    if scores.ndim != 2 or not torch.isfinite(scores).all() or len(allowed) != len(scores):
        raise ValueError("invalid selector scores or block coverage")
    if type(kept) is not int or kept not in feasible_counts(allowed):
        raise ValueError("infeasible retained count")
    states = {0: (0.0, ())}
    orders = []
    for block, counts in zip(scores, allowed, strict=True):
        if any(k > len(block) for k in counts):
            raise ValueError("allowed width exceeds source heads")
        order = sorted(range(len(block)), key=lambda h: (-float(block[h]), h))
        orders.append(order)
        values = {k: math.fsum(float(block[h]) for h in order[:k]) for k in sorted(set(counts))}
        next_states = {}
        for total, (value, vector) in states.items():
            for count, addition in values.items():
                n = total + count
                if n > kept:
                    continue
                item = (value + addition, (*vector, count))
                previous = next_states.get(n)

                # Original block/head IDs dominate the count-vector tie breaker.
                def tie_key(v):
                    return tuple(
                        (b, h) for b, (o, k) in enumerate(zip(orders, v, strict=True)) for h in sorted(o[:k])
                    ), v

                if (
                    previous is None
                    or item[0] > previous[0]
                    or (item[0] == previous[0] and tie_key(item[1]) < tie_key(previous[1]))
                ):
                    next_states[n] = item
        states = next_states
    mask = torch.zeros_like(scores, dtype=torch.float32)
    for b, count in enumerate(states[kept][1]):
        mask[b, orders[b][:count]] = 1
    return mask


class HeadGates(nn.Module):
    """One decision per update; a fresh autograd graph per microbatch."""

    def __init__(self, blocks=48, heads=32, allowed=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((blocks, heads), 3.0, dtype=torch.float32))
        self.allowed = allowed or [[16, 20, 24, 28, 32] for _ in range(blocks)]
        self.phase = "soft"
        self.decision = None
        self.current = None
        self.progressive = None

    def begin_update(self, phase, kept=None, scores=None, ramp=1.0):
        if phase not in ("soft", "binary", "fixed", "compact", "all_one"):
            raise ValueError("unknown gate phase")
        self.phase = phase
        if phase in ("binary", "compact", "fixed"):
            self.decision = select_legal(self.alpha if scores is None else scores, kept, self.allowed).to(
                self.alpha.device
            )
        self.ramp = ramp
        if not 0 <= ramp <= 1:
            raise ValueError("invalid gate ramp")

    def microbatch(self):
        p = self.alpha.sigmoid()
        if self.progressive is not None and self.phase == "soft":
            self.current = self.progressive.compose(self.alpha)
        elif self.phase == "soft":
            self.current = p
        elif self.phase == "binary":
            self.current = self.decision + (p - p.detach())
        elif self.phase == "fixed":
            self.current = (1 - self.ramp) * torch.sigmoid(
                torch.tensor(3.0, device=p.device)
            ) + self.ramp * self.decision
        elif self.phase == "compact":
            self.current = self.decision
        else:
            self.current = torch.ones_like(p)
        return self.current

    def provider(self, name):
        if not name.endswith(".attn1") or self.current is None:
            raise ValueError("provider requires a current self-attention microbatch")
        return self.current[int(name.split(".")[0])]

    def snapshot(self):
        if self.progressive is not None and hasattr(self.progressive, "one"):
            free = self.progressive.free
            p = self.alpha.detach().float().cpu().sigmoid()[free]
            per_head = self.alpha.detach().float().cpu().sigmoid().masked_fill(~free, float("nan"))
            return dict(
                counts=self.progressive.counts(),
                population="free_heads_only",
                mean=float(p.mean()) if p.numel() else None,
                quantiles=torch.quantile(p, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])).tolist() if p.numel() else None,
                block_means=[float(v) if torch.isfinite(v) else None for v in per_head.nanmean(1)],
                le_0_1=float((p <= 0.1).float().mean()) if p.numel() else None,
                lt_0_5=float((p < 0.5).float().mean()) if p.numel() else None,
                ge_0_9=float((p >= 0.9).float().mean()) if p.numel() else None,
            )
        p = (self.progressive.compose(self.alpha) if self.progressive else self.alpha.sigmoid()).detach().cpu()
        return {
            **({"counts": self.progressive.counts()} if self.progressive else {}),
            "mean": float(p.mean()),
            "quantiles": torch.quantile(p.flatten(), torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])).tolist(),
            "block_means": p.mean(1).tolist(),
            "le_0_1": float((p <= 0.1).float().mean()),
            "lt_0_5": float((p < 0.5).float().mean()),
            "ge_0_9": float((p >= 0.9).float().mean()),
        }
