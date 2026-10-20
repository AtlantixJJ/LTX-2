"""V3 family-local clipped gates and bounded joint selection.

Summaries use O(units), not O(window * units), memory. All decisions use the
pre-event state. Original IDs are row-major.
See doc/learn/readmission.md for the state and transaction contract.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction

import torch

FORMAT = "joint_clip_reset_v3"


def clipped_gate(alpha, alive):
    """Clamp forward to [0,1]; use PyTorch's inclusive [-3,3] derivative."""
    surrogate = alpha.clamp(-3.0, 3.0)
    gate = alpha.clamp(0.0, 1.0).detach() + (surrogate - surrogate.detach())
    return alive.to(alpha.device) * gate


def cap(fraction, count):
    return int(Fraction(str(fraction)) * count)


class Family:
    def __init__(self, shape, *, strong, persistent, minimum, block_cap=1.0, enabled=True):
        if len(shape) != 2 or shape[1] < minimum or not 0 < persistent <= strong:
            raise ValueError("invalid family geometry or thresholds")
        self.config = dict(strong=strong, persistent=persistent, minimum=minimum, block_cap=block_cap, enabled=enabled)
        self.kept = torch.ones(shape, dtype=torch.bool)
        self.permanent = torch.zeros(shape, dtype=torch.bool)
        self.returned = torch.zeros(shape, dtype=torch.bool)
        self.persist = torch.zeros(shape, dtype=torch.bool)
        self.removal_drop = torch.zeros(shape)
        self.removed_at = torch.zeros(shape, dtype=torch.int64)
        self.last_rise = torch.zeros(shape)
        self.reset_summary(self.kept.float())

    def reset_summary(self, scores):
        self.final = scores.clone()
        self.minimum = scores.clone()
        self.maximum = scores.clone()
        self.late_start = scores.clone()
        self.samples = 0

    def compose(self, alpha, *, fixed=False):
        if fixed or not self.config["enabled"]:
            return self.kept.to(alpha.device).float() + alpha * 0
        return clipped_gate(alpha, ~self.permanent)

    def penalty(self, alpha):
        active = (~self.permanent).to(alpha.device) if self.config["enabled"] else torch.zeros_like(alpha)
        return (torch.relu(alpha + 3) * active).sum()

    def observe(self, alpha, interval=50, late=10):
        value = alpha.detach().float().cpu().clone()
        if value.shape != self.kept.shape or not torch.isfinite(value).all() or alpha.dtype != torch.float32:
            raise ValueError("invalid family scores")
        if value[self.permanent].ne(0).any():
            raise ValueError("permanent scores changed")
        self.samples += 1
        self.final = value
        self.minimum = torch.minimum(self.minimum, value)
        self.maximum = torch.maximum(self.maximum, value)
        if self.samples == interval - late:
            self.late_start = value.clone()

    def candidates(self, *, removals=True):
        drop, rise = 1 - self.final.double(), self.final.double()
        depth, peak = 1 - self.minimum.double(), self.maximum.double()
        down_back = (self.final.double() - self.minimum.double()) / depth.clamp_min(1e-30)
        up_back = (peak - self.final.double()) / peak.clamp_min(1e-30)
        late = self.final.double() - self.late_start.double()
        clean = self.kept & (down_back <= 0.1) & (late < 0)
        strong = clean & (drop >= self.config["strong"])
        qualified = clean & (drop >= self.config["persistent"]) & ~self.returned
        persistent = qualified & self.persist
        down = (strong | persistent) if removals else torch.zeros_like(strong)
        up = ~self.kept & ~self.permanent & (rise >= self.config["strong"]) & (up_back <= 0.1) & (late > 0)
        if not self.config["enabled"]:
            down.zero_()
            up.zero_()
            qualified.zero_()
        return dict(
            down=down,
            up=up,
            qualified=qualified,
            strong=strong,
            drop=drop,
            rise=rise,
            down_giveback=down_back,
            up_giveback=up_back,
            late_change=late,
        )

    @staticmethod
    def ranked(mask, movement, giveback):
        ids = mask.flatten().nonzero().flatten()
        # Stable sorts preserve original IDs as the final tie breaker.
        ids = ids[torch.argsort(giveback.flatten()[ids], stable=True)]
        return ids[torch.argsort(movement.flatten()[ids], descending=True, stable=True)]

    def event(self, update, *, removals=True):
        c = self.candidates(removals=removals)
        before = self.kept.clone()
        alive = int(before.sum())
        limit = cap(0.04, alive)
        block_counts = before.sum(1)
        block_limits = torch.minimum(
            (block_counts - self.config["minimum"]).clamp_min(0),
            torch.tensor([cap(self.config["block_cap"], int(n)) for n in block_counts]),
        )
        down_order = self.ranked(c["down"], c["drop"], c["down_giveback"])
        # Rank within each block, then apply the global cap. At most units loops
        # are avoided: sorting and cumulative ranks operate on tensors.
        blocks = down_order // before.shape[1]
        by_block = torch.argsort(blocks, stable=True)
        sorted_blocks = blocks[by_block]
        positions = torch.arange(len(blocks))
        starts = torch.zeros_like(positions)
        if len(blocks):
            starts[1:] = torch.where(sorted_blocks[1:] != sorted_blocks[:-1], positions[1:], 0)
            starts = torch.cummax(starts, 0).values
        within = torch.empty_like(positions)
        within[by_block] = positions - starts
        eligible = down_order[within < block_limits[blocks]]
        selected = eligible[:limit]
        up_order = self.ranked(c["up"], c["rise"], c["up_giveback"])
        returned = up_order[:limit]
        flat_kept = self.kept.flatten()
        flat_kept[selected] = False
        self.permanent.flatten()[selected] = self.returned.flatten()[selected]
        flat_kept[returned] = True
        self.returned.flatten()[returned] = True
        self.removal_drop.flatten()[selected] = c["drop"].flatten()[selected].float()
        self.removed_at.flatten()[selected] = update
        self.last_rise = torch.where(~before, self.final, torch.zeros_like(self.final))
        self.persist = c["qualified"] & self.kept
        self.persist.flatten()[returned] = False
        result = dict(
            update=update,
            candidates=c,
            selected=selected,
            readmitted=returned,
            pre_kept=before,
            pre_scores=self.final.clone(),
            cap=limit,
            block_limits=block_limits,
            global_cap_bound=len(eligible) > limit,
            block_cap_suppressed=len(down_order) - len(eligible),
            eligible_removals=len(down_order),
            eligible_returns=len(up_order),
            summary={k: getattr(self, k).clone() for k in ("minimum", "maximum", "late_start", "final")},
        )
        self.reset_summary(self.kept.float())
        return result

    def align(self, multiple):
        restored = []
        for b, row in enumerate(self.kept):
            need = (-int(row.sum())) % multiple
            if not need:
                continue
            ids = (~row).nonzero().flatten()
            ids = ids[torch.argsort(self.removal_drop[b, ids], stable=True)]
            ids = ids[torch.argsort(self.last_rise[b, ids], descending=True, stable=True)]
            ids = ids[torch.argsort(self.permanent[b, ids].int(), stable=True)][:need]
            if len(ids) != need:
                raise ValueError("alignment exceeds original width")
            row[ids] = True
            self.permanent[b, ids] = False
            restored.extend((b, int(i)) for i in ids)
        self.persist.zero_()
        self.reset_summary(self.kept.float())
        return restored

    def counts(self):
        return dict(
            kept=int(self.kept.sum()),
            removed=int((~self.kept).sum()),
            permanent=int(self.permanent.sum()),
            returned=int(self.returned.sum()),
            returned_kept=int((self.returned & self.kept).sum()),
            per_block=self.kept.sum(1).tolist(),
        )

    def state_dict(self):
        return copy.deepcopy(vars(self))

    @classmethod
    def from_state(cls, state):
        obj = cls(state["kept"].shape, **state["config"])
        for key in ("kept", "permanent", "returned", "persist"):
            if state[key].dtype != torch.bool or state[key].shape != obj.kept.shape:
                raise ValueError("invalid family mask state")
        if (state["kept"] & state["permanent"]).any() or (state["persist"] & ~state["kept"]).any():
            raise ValueError("inconsistent family state")
        if (state["kept"].sum(1) < state["config"]["minimum"]).any():
            raise ValueError("family minimum violated")
        for key in ("final", "minimum", "maximum", "late_start", "removal_drop", "last_rise"):
            if state[key].shape != obj.kept.shape or not torch.isfinite(state[key]).all():
                raise ValueError("invalid family summary")
        obj.__dict__.update(copy.deepcopy(state))
        return obj


class JointPolicy:
    def __init__(
        self,
        heads=(48, 32),
        channels=(48, 16384),
        *,
        enabled=True,
        interval=50,
        late=10,
        earliest=200,
        limit=1500,
        recovery=200,
        alignment=128,
    ):
        if limit % interval or earliest % interval or not 0 < late < interval or earliest < 2 * interval:
            raise ValueError("invalid joint event schedule")
        self.config = dict(
            interval=interval, late=late, earliest=earliest, limit=limit, recovery=recovery, alignment=alignment
        )
        self.heads = Family(heads, strong=0.25, persistent=0.15, minimum=1)
        self.channels = Family(
            channels, strong=0.35, persistent=0.25, minimum=alignment, block_cap=0.25, enabled=enabled
        )
        self.update = self.quiet = 0
        self.selection_end = None
        self.stop_reason = None
        self.aligned = False
        self.alignment_ids = []
        self.projection_mode = False

    @property
    def recovering(self):
        return self.selection_end is not None

    @property
    def final_update(self):
        return None if self.selection_end is None else self.selection_end + self.config["recovery"]

    def record(self, update, heads, channels, optimizers):
        if update != self.update + 1 or (self.final_update is not None and update > self.final_update):
            raise ValueError("joint cursor mismatch")
        self.update = update
        if self.recovering:
            return None
        for family, alpha in ((self.heads, heads), (self.channels, channels)):
            family.observe(alpha, self.config["interval"], self.config["late"])
        if update % self.config["interval"]:
            return None
        removals = update <= self.config["limit"] - 2 * self.config["interval"]
        unresolved = any(f.candidates()["down"].any() for f in (self.heads, self.channels)) if not removals else False
        events = {name: getattr(self, name).event(update, removals=removals) for name in ("heads", "channels")}
        busy = any(e["eligible_removals"] or e["eligible_returns"] for e in events.values())
        self.quiet = 0 if busy else self.quiet + 1
        # A terminal return-only phase must not pretend ignored drop candidates converged.
        if update == self.config["limit"]:
            self.stop_reason = "hard_limit"
        elif update >= self.config["earliest"] and self.quiet >= 2 and removals:
            self.stop_reason = "two_quiet_events"
        if self.stop_reason:
            self.selection_end = update
        self.reset(heads, channels, optimizers)
        return dict(
            update=update,
            families=events,
            quiet=self.quiet,
            return_only=not removals,
            remaining_return_windows=(self.config["limit"] - update) // self.config["interval"],
            selection_end=self.selection_end,
            reason=self.stop_reason,
            unresolved=bool(unresolved),
        )

    @torch.no_grad()
    def reset(self, heads, channels, optimizers):
        for f, alpha in ((self.heads, heads), (self.channels, channels)):
            alpha.copy_(f.kept.to(alpha.device))
            alpha.grad = None
        for optimizer in optimizers:
            optimizer.state.clear()

    def align(self, heads, channels, optimizers):
        if not self.recovering or self.aligned:
            raise ValueError("alignment requires unaligned selection end")
        self.alignment_ids = self.channels.align(self.config["alignment"])
        self.aligned = True
        self.reset(heads, channels, optimizers)
        return self.alignment_ids

    def state_dict(self):
        return dict(
            format=FORMAT,
            config=self.config.copy(),
            heads=self.heads.state_dict(),
            channels=self.channels.state_dict(),
            update=self.update,
            quiet=self.quiet,
            selection_end=self.selection_end,
            stop_reason=self.stop_reason,
            aligned=self.aligned,
            alignment_ids=self.alignment_ids,
            projection_mode=self.projection_mode,
        )

    @classmethod
    def from_state(cls, state):
        if state.get("format") != FORMAT:
            raise ValueError("unsupported joint policy format")
        obj = cls(state["heads"]["kept"].shape, state["channels"]["kept"].shape, **state["config"])
        obj.heads, obj.channels = Family.from_state(state["heads"]), Family.from_state(state["channels"])
        for key in ("update", "quiet", "selection_end", "stop_reason", "aligned", "alignment_ids"):
            setattr(obj, key, copy.deepcopy(state[key]))
        obj.projection_mode = state.get("projection_mode", False)
        expected_samples = 0 if obj.recovering or obj.projection_mode else obj.update % obj.config["interval"]
        if obj.heads.samples != expected_samples or obj.channels.samples != expected_samples:
            raise ValueError("joint partial window differs from cursor")
        if obj.update < 0 or obj.update > (obj.final_update or obj.config["limit"]):
            raise ValueError("invalid joint cursor")
        return obj


def budget_projection(head_scores, channel_scores, budget, *, cost=704, minimum=128):
    """GRASPrune prefix, including charged block guards; scores are p, not p/c."""
    if not math.isfinite(budget) or cost <= 0:
        raise ValueError("invalid compute budget")
    h, c = head_scores.detach().cpu(), channel_scores.detach().cpu()
    if h.ndim != 2 or c.ndim != 2 or h.shape[0] != c.shape[0] or minimum > c.shape[1]:
        raise ValueError("invalid projection geometry")
    values = torch.cat((h.flatten(), c.flatten()))
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite projection scores")
    costs = torch.ones(len(values), dtype=torch.int64)
    costs[: h.numel()] = cost
    mask = torch.zeros(len(values), dtype=torch.bool)
    for b in range(len(h)):
        mask[b * h.shape[1] + torch.argsort(h[b], descending=True, stable=True)[0]] = True
        ids = torch.argsort(c[b], descending=True, stable=True)[:minimum]
        mask[h.numel() + b * c.shape[1] + ids] = True
    reserved = int(costs[mask].sum())
    if not reserved <= budget <= int(costs.sum()):
        raise ValueError("infeasible projection budget")
    order = torch.argsort(values, descending=True, stable=True)
    order = order[~mask[order]]
    # Prefix selection stops at the FIRST unaffordable unit, without skipping it.
    prefix = costs[order].cumsum(0) <= budget - reserved
    mask[order[prefix]] = True
    spent = int(costs[mask].sum())
    return (
        mask[: h.numel()].reshape(h.shape).float(),
        mask[h.numel() :].reshape(c.shape).float(),
        dict(spent=spent, slack=budget - spent),
    )
