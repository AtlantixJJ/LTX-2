"""Eligible-population decisions followed by a checkpointed, exact fixed-gate tail."""

from __future__ import annotations

import copy
import math

import torch

from scripts.prune.learn.progressive import ProgressivePolicy, selection, statistics, validate_config
from scripts.prune.learn.zero_one import ZeroOnePolicy, select_one


def select_gate_only(free, alpha, config):
    """Every remaining free head is eligible; regret and travel have no role."""
    gates = alpha.float().cpu().sigmoid()
    candidates = [dict(block=b, head=h, gate=float(gates[b, h])) for b, h in free.nonzero().tolist()]
    candidates.sort(key=lambda c: (-c["gate"], c["block"], c["head"]))
    requested = max(config["minimum_count"], math.ceil(config["fixed_one_fraction"] * len(candidates)))
    for rank, item in enumerate(candidates, 1):
        item.update(rank=rank, selected=rank <= requested)
    return dict(
        one_candidates=candidates,
        one_selected_ids=[[c["block"], c["head"]] for c in candidates[:requested]],
        one_requested_count=requested,
        one_eligible_count=len(candidates),
        one_selected_count=min(requested, len(candidates)),
        one_shortfall=max(0, requested - len(candidates)),
    )


class EligiblePolicy(ZeroOnePolicy):
    def __init__(self, alpha, config, base=None):
        ProgressivePolicy.__init__(self, alpha, config)
        if self.config.get("selection_population") != "eligible":
            raise ValueError("eligible policy requires eligible-population settings")
        self.base = copy.deepcopy(base)
        self.one = torch.zeros_like(self.free)
        self.start_update = 0
        if base is not None:
            parent = ProgressivePolicy.from_state(base)
            if parent.config["hardening_interval"] != self.interval:
                raise ValueError("migration interval changed")
            if not torch.equal(alpha.detach().float().cpu(), parent.window[-1]):
                raise ValueError("migration logits differ from parent")
            self.free, self.frozen = parent.free.clone(), parent.frozen.clone()
            self.one = getattr(parent, "one", torch.zeros_like(parent.free)).clone()
            self.window = [v.clone() for v in parent.window]
            self.update = self.start_update = parent.update
            self.history = copy.deepcopy(parent.history)
        self.new_history = []
        self.finalization_start = self.final_update = None

    def _commit(self, update, alpha, travel, decrease, regret):
        if self.final_update is not None:
            raise ValueError("a decision cannot occur during finalization")
        choice = selection(self.free, travel, decrease, regret, self.config)
        if self.config.get("fixed_one_selection") == "gate_only":
            remaining = self.free.clone()
            for b, h in choice["selected_ids"]:
                remaining[b, h] = False
            ones = select_gate_only(remaining, alpha, self.config)
        else:
            ones = select_one(self.free, alpha, travel, regret, self.config)
        zero_ids, one_ids = choice["selected_ids"], ones["one_selected_ids"]
        if set(map(tuple, zero_ids)) & set(map(tuple, one_ids)):
            raise ValueError("zero and one selections overlap")
        for b, h in zero_ids + one_ids:
            self.free[b, h] = False
            self.frozen[b, h] = alpha[b, h]
        for b, h in one_ids:
            self.one[b, h] = True
        forced = self.free.nonzero().tolist() if choice["eligible_count"] == 0 else []
        for b, h in forced:
            self.free[b, h], self.one[b, h] = False, True
            self.frozen[b, h] = alpha[b, h]
        reason = None
        if not self.free.any():
            reason = "no_zero_eligible" if choice["eligible_count"] == 0 else "no_free_heads"
            self.finalization_start = update
            self.final_update = update + self.config["finalize_updates"]
        event = dict(
            update=update,
            config=dict(self.config),
            **choice,
            **ones,
            forced_one_selected_ids=forced,
            forced_one_selected_count=len(forced),
            new_fixed_one_count=len(one_ids) + len(forced),
            terminal_reason=reason,
            finalization_start=self.finalization_start,
            final_update=self.final_update,
            travel=travel.tolist(),
            decrease=decrease.tolist(),
            regret=[[float(v) if torch.isfinite(v) else None for v in row] for row in regret],
            final_logits=alpha.tolist(),
            counts=self.counts(),
        )
        self.history.append(event)
        self.new_history.append(event)
        return event

    def record(self, update, alpha, optimizer=None):
        if (
            update != self.update + 1
            or alpha.shape != self.free.shape
            or not torch.isfinite(alpha).all()
            or (self.final_update is not None and update > self.final_update)
        ):
            raise ValueError("eligible cursor, shape, finite logits or final bound changed")
        value = alpha.detach().float().cpu().clone()
        if not torch.equal(value[~self.free], self.frozen[~self.free]):
            raise ValueError("fixed logits changed")
        self.update = update
        if self.final_update is not None:
            self.window = [value]
            return None
        self.window.append(value)
        if update % self.interval:
            return None
        if len(self.window) != self.interval + 1:
            raise ValueError("eligible decision requires interval+1 snapshots")
        event = self._commit(update, value, *statistics(self.window))
        self.window = [value]
        if optimizer is not None:
            self.mask_gradient(alpha)
            self.protect(alpha, optimizer)
        return event

    def state_dict(self):
        state = super().state_dict()
        state.update(
            format="progressive_eligible_v1", finalization_start=self.finalization_start, final_update=self.final_update
        )
        return state

    @classmethod
    def from_state(cls, state, expected_config=None):
        if state.get("format") != "progressive_eligible_v1":
            raise ValueError("unsupported eligible policy state")
        config = validate_config(state["config"])
        if expected_config is not None and config != validate_config(expected_config):
            raise ValueError("eligible resume hyperparameters changed")
        frozen = torch.tensor(state["frozen"], dtype=torch.float32)
        initial = torch.tensor(state["base"]["window"][-1], dtype=torch.float32) if state["base"] else frozen
        policy = cls(initial, config, state["base"])
        cursor = state["update"]
        if type(cursor) is not int or cursor < policy.start_update:
            raise ValueError("invalid eligible cursor")
        next_event = (policy.start_update // policy.interval + 1) * policy.interval
        for event in state["history"]:
            if event["update"] != next_event or event["update"] > cursor or event["config"] != config:
                raise ValueError("eligible event cursor or settings changed")
            travel = torch.tensor(event["travel"], dtype=torch.float64)
            decrease = torch.tensor(event["decrease"], dtype=torch.float64)
            alpha = torch.tensor(event["final_logits"], dtype=torch.float32)
            if (
                travel.shape != policy.free.shape
                or decrease.shape != travel.shape
                or alpha.shape != travel.shape
                or not torch.isfinite(travel).all()
                or not torch.isfinite(decrease).all()
                or not torch.isfinite(alpha).all()
                or (travel < decrease.abs()).any()
                or not torch.equal(alpha[~policy.free], policy.frozen[~policy.free])
            ):
                raise ValueError("invalid eligible statistics or frozen coordinates")
            regret = torch.where(travel > 0, (1 - decrease.abs() / travel).clamp(0, 1), float("nan"))
            expected = policy._commit(event["update"], alpha, travel, decrease, regret)
            if event != expected:
                raise ValueError("eligible decision receipt differs")
            next_event += policy.interval
        if policy.final_update is None and next_event <= cursor:
            raise ValueError("eligible event missing")
        if (
            state["finalization_start"] != policy.finalization_start
            or state["final_update"] != policy.final_update
            or (policy.final_update is not None and cursor > policy.final_update)
        ):
            raise ValueError("eligible finalization cursor differs")
        window = [torch.tensor(v, dtype=torch.float32) for v in state["window"]]
        expected_length = 1 if policy.final_update is not None else cursor % policy.interval + 1
        if (
            len(window) != expected_length
            or any(v.shape != policy.free.shape or not torch.isfinite(v).all() for v in window)
            or not torch.equal(policy.free, torch.tensor(state["free"], dtype=torch.bool))
            or not torch.equal(policy.one, torch.tensor(state["one"], dtype=torch.bool))
            or frozen.shape != policy.free.shape
            or not torch.isfinite(frozen).all()
            or not torch.equal(policy.frozen[~policy.free], frozen[~policy.free])
            or any(not torch.equal(v[~policy.free], frozen[~policy.free]) for v in window)
        ):
            raise ValueError("eligible masks, frozen logits or window differ")
        policy.frozen, policy.window, policy.update = frozen, window, cursor
        return policy
