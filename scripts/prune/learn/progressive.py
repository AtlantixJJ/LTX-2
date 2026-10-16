"""Configurable global removals from post-update logit trajectories; no width selector."""

from __future__ import annotations

import copy
import math

import torch

from scripts.prune.learn import checkpoints

DEFAULTS = dict(hardening_interval=50, removal_fraction=0.01, regret_threshold=0.10)


def validate_config(config):
    extra = {"fixed_one_fraction", "fixed_one_regret_threshold"}
    if set(config) not in (set(DEFAULTS), set(DEFAULTS) | extra):
        raise ValueError("progressive policy requires interval, fraction and regret threshold only")
    interval, fraction, threshold = (config[k] for k in DEFAULTS)
    if type(interval) is not int or interval < 1:
        raise ValueError("hardening_interval must be a positive integer")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("removal_fraction must be in (0,1]")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("regret_threshold must be in [0,1]")
    if extra <= set(config):
        fraction, high = config["fixed_one_fraction"], config["fixed_one_regret_threshold"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 < fraction <= 1
            or isinstance(high, bool)
            or not isinstance(high, (int, float))
            or not math.isfinite(high)
            or not config["regret_threshold"] < high <= 1
        ):
            raise ValueError("fixed-one fraction or disjoint high-regret threshold differs")
    return dict(config)


def statistics(window):
    values = torch.stack(window).double()
    if len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("invalid progressive logit window")
    travel = values.diff(dim=0).abs().sum(0)
    decrease = values[0] - values[-1]
    regret = torch.where(travel > 0, (1 - decrease.abs() / travel).clamp(0, 1), float("nan"))
    return travel, decrease, regret


def selection(free, travel, decrease, regret, config):
    """Rank only eligible original IDs. Constant heads have undefined regret."""
    candidates = [
        dict(block=b, head=h, travel=float(travel[b, h]), decrease=float(decrease[b, h]), regret=float(regret[b, h]))
        for b, h in free.nonzero().tolist()
        if travel[b, h] > 0 and decrease[b, h] > 0 and regret[b, h] <= config["regret_threshold"]
    ]
    candidates.sort(key=lambda c: (c["regret"], -c["decrease"], c["block"], c["head"]))
    requested = math.ceil(config["removal_fraction"] * free.numel())
    for rank, candidate in enumerate(candidates, 1):
        candidate.update(rank=rank, selected=rank <= requested)
    selected = [[c["block"], c["head"]] for c in candidates[:requested]]
    return dict(
        candidates=candidates,
        selected_ids=selected,
        eligible_count=len(candidates),
        requested_count=requested,
        selected_count=len(selected),
        shortfall=max(0, requested - len(selected)),
    )


class ProgressivePolicy:
    def __init__(self, alpha, config=None):
        self.config = validate_config(DEFAULTS if config is None else config)
        if alpha.ndim != 2 or not torch.isfinite(alpha).all():
            raise ValueError("progressive logits must be a finite block/head matrix")
        self.free = torch.ones_like(alpha, dtype=torch.bool, device="cpu")
        self.frozen = alpha.detach().float().cpu().clone()
        self.window = [self.frozen.clone()]
        self.update = 0
        self.history = []

    @property
    def interval(self):
        return self.config["hardening_interval"]

    def compose(self, alpha):
        return alpha.sigmoid() * self.free.to(alpha.device)

    def mask_gradient(self, alpha):
        if alpha.grad is not None:
            alpha.grad.mul_(self.free.to(alpha.device))

    @torch.no_grad()
    def protect(self, alpha, optimizer):
        fixed = ~self.free.to(alpha.device)
        alpha[fixed] = self.frozen.to(alpha.device)[fixed]
        for value in optimizer.state.get(alpha, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == alpha.shape:
                value.masked_fill_(fixed, 0)

    def record(self, update, alpha, optimizer=None):
        if update != self.update + 1 or alpha.shape != self.free.shape or not torch.isfinite(alpha).all():
            raise ValueError("progressive cursor, shape or finite logits changed")
        self.update = update
        self.window.append(alpha.detach().float().cpu().clone())
        if update % self.interval:
            return None
        if len(self.window) != self.interval + 1:
            raise ValueError("progressive decision requires interval+1 snapshots")
        travel, decrease, regret = statistics(self.window)
        choice = selection(self.free, travel, decrease, regret, self.config)
        for b, h in choice["selected_ids"]:
            self.free[b, h] = False
            self.frozen[b, h] = self.window[-1][b, h]
        event = dict(
            update=update,
            config=dict(self.config),
            **choice,
            travel=travel.tolist(),
            decrease=decrease.tolist(),
            regret=[[float(v) if torch.isfinite(v) else None for v in row] for row in regret],
            final_logits=self.window[-1].tolist(),
            counts=self.counts(),
        )
        self.history.append(event)
        self.window = [self.window[-1]]
        if optimizer is not None:
            self.mask_gradient(alpha)
            self.protect(alpha, optimizer)
        return event

    def counts(self):
        return dict(free=int(self.free.sum()), removed=int((~self.free).sum()))

    def state_dict(self):
        return copy.deepcopy(
            dict(
                format="progressive_removal_v2",
                config=self.config,
                free=self.free.tolist(),
                frozen=self.frozen.tolist(),
                window=[v.tolist() for v in self.window],
                update=self.update,
                history=self.history,
            )
        )

    @classmethod
    def from_state(cls, state, expected_config=None):
        if state.get("format") == "progressive_zero_one_v1":
            from scripts.prune.learn.zero_one import ZeroOnePolicy

            return ZeroOnePolicy.from_state(state, expected_config)
        if state.get("format") != "progressive_removal_v2":
            raise ValueError("unsupported progressive policy state")
        if expected_config is not None and state["config"] != validate_config(expected_config):
            raise ValueError("progressive resume hyperparameters changed")
        policy = cls(torch.tensor(state["frozen"], dtype=torch.float32), state["config"])
        policy.free = torch.tensor(state["free"], dtype=torch.bool)
        policy.window = [torch.tensor(v, dtype=torch.float32) for v in state["window"]]
        policy.update, policy.history = state["update"], copy.deepcopy(state["history"])
        if (
            type(policy.update) is not int
            or policy.update < 0
            or policy.free.shape != policy.frozen.shape
            or len(policy.window) != policy.update % policy.interval + 1
            or any(v.shape != policy.free.shape or not torch.isfinite(v).all() for v in policy.window)
        ):
            raise ValueError("invalid progressive masks or window cursor")
        if [e["update"] for e in policy.history] != list(range(policy.interval, policy.update + 1, policy.interval)):
            raise ValueError("progressive event history differs")
        free = torch.ones_like(policy.free)
        for event in policy.history:
            if event["config"] != policy.config:
                raise ValueError("progressive historical hyperparameters changed")
            travel = torch.tensor(event["travel"], dtype=torch.float64)
            decrease = torch.tensor(event["decrease"], dtype=torch.float64)
            regret = torch.where(travel > 0, (1 - decrease.abs() / travel).clamp(0, 1), float("nan"))
            if (
                travel.shape != free.shape
                or decrease.shape != free.shape
                or not torch.isfinite(travel).all()
                or not torch.isfinite(decrease).all()
                or (travel < decrease.abs()).any()
            ):
                raise ValueError("invalid progressive event statistics")
            expected = selection(free, travel, decrease, regret, policy.config)
            if any(event[k] != v for k, v in expected.items()):
                raise ValueError("progressive event ranking, threshold or selection differs")
            regrets = [[float(v) if torch.isfinite(v) else None for v in row] for row in regret]
            if event["regret"] != regrets:
                raise ValueError("progressive regret receipt differs")
            for b, h in event["selected_ids"]:
                free[b, h] = False
            if event["counts"] != dict(free=int(free.sum()), removed=int((~free).sum())):
                raise ValueError("progressive event counts differ")
        if not torch.equal(free, policy.free):
            raise ValueError("progressive history and masks differ")
        return policy

    def verify_ranks(self, device):
        """Compare the complete policy using a tensor collective on every rank."""
        import torch.distributed as dist

        if dist.is_initialized():
            raw = bytes.fromhex(checkpoints.digest(self.state_dict()))
            value = torch.tensor(list(raw), dtype=torch.uint8, device=device)
            gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, value)
            if any(not torch.equal(value, other) for other in gathered):
                raise ValueError("distributed progressive policy differs")


def new_policy(alpha, config):
    """Select the configured policy without changing legacy removal-only construction."""
    if "fixed_one_fraction" in config:
        from scripts.prune.learn.zero_one import ZeroOnePolicy

        return ZeroOnePolicy(alpha, config)
    return ProgressivePolicy(alpha, config)
