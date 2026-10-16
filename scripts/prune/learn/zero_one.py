"""Versioned zero/one continuation; fixed coordinates never return to the free set."""

from __future__ import annotations

import copy
import math

import torch

from scripts.prune.learn.progressive import ProgressivePolicy, selection, statistics, validate_config


def select_one(free, alpha, travel, regret, config):
    gates = alpha.float().sigmoid()
    candidates = [
        dict(block=b, head=h, gate=float(gates[b, h]), regret=float(regret[b, h]))
        for b, h in free.nonzero().tolist()
        if travel[b, h] > 0 and regret[b, h] >= config["fixed_one_regret_threshold"]
    ]
    candidates.sort(key=lambda c: (-c["gate"], -c["regret"], c["block"], c["head"]))
    requested = math.ceil(config["fixed_one_fraction"] * free.numel())
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


class ZeroOnePolicy(ProgressivePolicy):
    def __init__(self, alpha, config, base=None):
        super().__init__(alpha, config)
        if "fixed_one_fraction" not in self.config:
            raise ValueError("zero/one policy requires fixed-one settings")
        self.base = copy.deepcopy(base)
        self.one = torch.zeros_like(self.free)
        self.start_update = 0
        if base is not None:
            parent = ProgressivePolicy.from_state(base)
            if parent.update % self.interval or len(parent.window) != 1:
                raise ValueError("policy migration requires a committed event checkpoint")
            if not torch.equal(alpha.detach().float().cpu(), parent.window[-1]):
                raise ValueError("migration logits differ from parent")
            self.free, self.frozen = parent.free.clone(), parent.frozen.clone()
            self.window = [parent.window[-1].clone()]
            self.update = self.start_update = parent.update
            self.history = copy.deepcopy(parent.history)
        self.new_history = []

    def compose(self, alpha):
        return super().compose(alpha) + self.one.to(alpha.device, dtype=alpha.dtype)

    def counts(self):
        return dict(
            free=int(self.free.sum()), removed=int((~self.free & ~self.one).sum()), fixed_one=int(self.one.sum())
        )

    def record(self, update, alpha, optimizer=None):
        if update != self.update + 1 or alpha.shape != self.free.shape or not torch.isfinite(alpha).all():
            raise ValueError("zero/one cursor, shape or finite logits changed")
        self.update = update
        self.window.append(alpha.detach().float().cpu().clone())
        if update % self.interval:
            return None
        if len(self.window) != self.interval + 1:
            raise ValueError("zero/one decision requires interval+1 snapshots")
        travel, decrease, regret = statistics(self.window)
        choice = selection(self.free, travel, decrease, regret, self.config)
        ones = select_one(self.free, self.window[-1], travel, regret, self.config)
        zero_ids, one_ids = choice["selected_ids"], ones["one_selected_ids"]
        if set(map(tuple, zero_ids)) & set(map(tuple, one_ids)):
            raise ValueError("zero/one decisions overlap")
        for b, h in zero_ids + one_ids:
            self.free[b, h] = False
            self.frozen[b, h] = self.window[-1][b, h]
        for b, h in one_ids:
            self.one[b, h] = True
        event = dict(
            update=update,
            config=dict(self.config),
            **choice,
            **ones,
            travel=travel.tolist(),
            decrease=decrease.tolist(),
            regret=[[float(v) if torch.isfinite(v) else None for v in row] for row in regret],
            final_logits=self.window[-1].tolist(),
            counts=self.counts(),
        )
        self.history.append(event)
        self.new_history.append(event)
        self.window = [self.window[-1]]
        if optimizer is not None:
            self.mask_gradient(alpha)
            self.protect(alpha, optimizer)
        return event

    def state_dict(self):
        return copy.deepcopy(
            dict(
                format="progressive_zero_one_v1",
                config=self.config,
                base=self.base,
                free=self.free.tolist(),
                one=self.one.tolist(),
                frozen=self.frozen.tolist(),
                window=[v.tolist() for v in self.window],
                update=self.update,
                history=self.new_history,
            )
        )

    @classmethod
    def from_state(cls, state, expected_config=None):
        if state.get("format") != "progressive_zero_one_v1":
            raise ValueError("unsupported zero/one policy format")
        config = validate_config(state["config"])
        if expected_config is not None and config != validate_config(expected_config):
            raise ValueError("zero/one resume hyperparameters changed")
        frozen = torch.tensor(state["frozen"], dtype=torch.float32)
        initial = (
            torch.tensor(state["base"]["window"][-1], dtype=torch.float32) if state["base"] is not None else frozen
        )
        policy = cls(initial, config, state["base"])
        if type(state["update"]) is not int or state["update"] < policy.start_update:
            raise ValueError("invalid zero/one cursor")
        expected_events = list(range(policy.start_update + policy.interval, state["update"] + 1, policy.interval))
        if [e["update"] for e in state["history"]] != expected_events:
            raise ValueError("zero/one event history differs")
        for event in state["history"]:
            if event["config"] != config:
                raise ValueError("zero/one historical hyperparameters changed")
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
            ):
                raise ValueError("invalid zero/one event statistics")
            regret = torch.where(travel > 0, (1 - decrease.abs() / travel).clamp(0, 1), float("nan"))
            expected = {
                **selection(policy.free, travel, decrease, regret, config),
                **select_one(policy.free, alpha, travel, regret, config),
            }
            if any(event[k] != v for k, v in expected.items()):
                raise ValueError("zero/one event ranking or selection differs")
            regrets = [[float(v) if torch.isfinite(v) else None for v in row] for row in regret]
            if event["regret"] != regrets or not torch.equal(alpha[~policy.free], policy.frozen[~policy.free]):
                raise ValueError("zero/one event regret or frozen logits differ")
            for b, h in event["selected_ids"] + event["one_selected_ids"]:
                policy.free[b, h] = False
                policy.frozen[b, h] = alpha[b, h]
            for b, h in event["one_selected_ids"]:
                policy.one[b, h] = True
            if event["counts"] != policy.counts():
                raise ValueError("zero/one event counts differ")
            policy.history.append(copy.deepcopy(event))
            policy.new_history.append(copy.deepcopy(event))
        window = [torch.tensor(v, dtype=torch.float32) for v in state["window"]]
        if (
            len(window) != state["update"] % policy.interval + 1
            or any(v.shape != policy.free.shape or not torch.isfinite(v).all() for v in window)
            or not torch.equal(policy.free, torch.tensor(state["free"], dtype=torch.bool))
            or not torch.equal(policy.one, torch.tensor(state["one"], dtype=torch.bool))
            or not torch.equal(policy.frozen[~policy.free], frozen[~policy.free])
            or any(not torch.equal(v[~policy.free], frozen[~policy.free]) for v in window)
        ):
            raise ValueError("zero/one masks, frozen coordinates or window differ")
        policy.frozen, policy.window, policy.update = frozen, window, state["update"]
        return policy
