"""Versioned clipped-score removal, survivor resets, and an exact adapter-only tail.

Raw scores stay FP32 and are never projected between events. Each event consumes
one complete window, then starts a new window from the binary reset snapshot.
The reset jump belongs to neither trajectory. A survivor's drop from one is the
removal evidence: the regret-free drop selector removes heads that kept their
drop and still fell at the window end. Historical sigmoid states cannot be
migrated into this policy. See doc/learn/clipped_reset.md.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction

import torch

from scripts.prune.learn.progressive import ProgressivePolicy, statistics

METHOD = "clip01_ste3_reset_v2"
DEFAULTS = dict(
    method=METHOD,
    activation="clip01_v1",
    surrogate="clamp_minus3_plus3_inclusive_v1",
    reset="survivors_one_removed_zero_v1",
    gate_optimizer_reset="all_adam_state_v1",
    selector="regret_free_drop_v1",
    hardening_interval=50,
    full_speed_drop=0.5,
    strong_fraction=0.5,
    persistent_fraction=0.3,
    giveback_limit=0.10,
    late_updates=10,
    cap_fraction=0.04,
    empty_events_to_stop=3,
    selection_updates=500,
    finalize_updates=300,
)


# Optional stop rules. The default keeps selecting until empty events or the cap.
STOP_RULES = ("empty_events", "no_strong_or_no_candidates")


def is_clipped(config):
    return (config or {}).get("method") == METHOD


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_config(config):
    if set(config) - {"stop_rule"} != set(DEFAULTS):
        raise ValueError("clipped-reset settings must have the complete versioned schema")
    if config.get("stop_rule", "empty_events") not in STOP_RULES:
        raise ValueError("unsupported clipped-reset stop_rule")
    for key in ("method", "activation", "surrogate", "reset", "gate_optimizer_reset", "selector"):
        if config[key] != DEFAULTS[key]:
            raise ValueError(f"unsupported clipped-reset {key}")
    for key in ("hardening_interval", "late_updates", "empty_events_to_stop", "selection_updates", "finalize_updates"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"clipped-reset {key} must be a positive integer")
    if config["selection_updates"] % config["hardening_interval"]:
        raise ValueError("selection cap must end at an event")
    if config["late_updates"] > config["hardening_interval"]:
        raise ValueError("late updates must fit inside one window")
    for key in ("full_speed_drop", "strong_fraction", "persistent_fraction"):
        if not _number(config[key]) or config[key] <= 0:
            raise ValueError(f"clipped-reset {key} must be positive")
    if not _number(config["giveback_limit"]) or not 0 <= config["giveback_limit"] <= 1:
        raise ValueError("invalid clipped-reset giveback_limit")
    if not _number(config["cap_fraction"]) or not 0 < config["cap_fraction"] <= 1:
        raise ValueError("invalid clipped-reset cap_fraction")
    return dict(config)


def drop_statistics(window, late_updates):
    """FP64 drop, give-back, late change, travel and recorded path regret for one window."""
    values = torch.stack(window).double()
    travel, drop, path_regret = statistics(window)
    low = values.min(0).values
    span = values[0] - low
    giveback = torch.where(span > 0, (values[-1] - low) / span.clamp_min(1e-300), float("nan"))
    late = values[-1] - values[-1 - late_updates]
    return drop, giveback, late, travel, path_regret


def drop_selection(free, persist, window, config):
    """Strong or persistent regret-free drops, lowest score first, capped per event; no minimum."""
    drop, giveback, late, travel, path_regret = drop_statistics(window, config["late_updates"])
    clean = free & (giveback <= config["giveback_limit"]) & (late < 0)
    strong = clean & (drop >= config["strong_fraction"] * config["full_speed_drop"])
    qualified = clean & (drop >= config["persistent_fraction"] * config["full_speed_drop"])
    persistent = qualified & persist
    chosen = strong | persistent
    candidates = [
        dict(
            block=b, head=h, tier="strong" if strong[b, h] else "persistent",
            drop=float(drop[b, h]), giveback=float(giveback[b, h]), late_change=float(late[b, h]),
            path_regret=float(path_regret[b, h]),
        )
        for b, h in chosen.nonzero().tolist()
    ]
    candidates.sort(
        key=lambda c: (-c["drop"], c["giveback"], c["path_regret"] if math.isfinite(c["path_regret"]) else math.inf,
                       c["block"], c["head"])
    )
    alive = int(free.sum())
    cap = int(Fraction(str(config["cap_fraction"])) * alive)
    for rank, candidate in enumerate(candidates, 1):
        candidate.update(rank=rank, selected=rank <= cap)
    selected = [[c["block"], c["head"]] for c in candidates[:cap]]
    nulls = lambda t: [[float(v) if math.isfinite(v) else None for v in row] for row in t.tolist()]  # noqa: E731
    return dict(
        candidates=candidates,
        selected_ids=selected,
        eligible_count=len(candidates),
        requested_count=len(candidates),
        selected_count=len(selected),
        shortfall=len(candidates) - len(selected),
        alive_before=alive,
        cap=cap,
        cap_bound=len(candidates) > cap,
        strong_count=int(strong.sum()),
        persistent_count=int((persistent & ~strong).sum()),
        qualified=qualified,
        drop=drop.tolist(),
        giveback=nulls(giveback),
        late_change=late.tolist(),
        travel=travel.tolist(),
        path_regret=nulls(path_regret),
    )


def clipped_gate(alpha, alive):
    """Clamp forward to [0,1]; use PyTorch's inclusive [-3,3] derivative."""
    surrogate = alpha.clamp(-3.0, 3.0)
    gate = alpha.clamp(0.0, 1.0).detach() + (surrogate - surrogate.detach())
    return alive.to(alpha.device) * gate


class ClippedResetPolicy(ProgressivePolicy):
    def __init__(self, alpha, config=None):
        self.config = validate_config(DEFAULTS if config is None else config)
        if alpha.dtype != torch.float32 or alpha.ndim != 2 or not alpha.numel() or not torch.all(alpha == 1):
            raise ValueError("fresh clipped-reset scores must be an FP32 all-one matrix")
        self.free = torch.ones_like(alpha, dtype=torch.bool, device="cpu")
        self.frozen = torch.zeros_like(alpha, device="cpu")
        self.window = [alpha.detach().cpu().clone()]
        self.update, self.history = 0, []
        self.finalization_start = self.final_update = None
        # Persistence flags: regret-free drop >= persistent threshold in the previous window.
        self.persist = torch.zeros_like(self.free)
        self.empty_events = 0

    @property
    def recovering(self):
        return self.finalization_start is not None

    def compose(self, alpha):
        return clipped_gate(alpha, self.free)

    def mask_gradient(self, alpha):
        if alpha.grad is not None:
            alpha.grad.mul_(0 if self.recovering else self.free.to(alpha.device))

    @torch.no_grad()
    def protect(self, alpha, optimizer=None):
        fixed = torch.ones_like(self.free) if self.recovering else ~self.free
        mask = fixed.to(alpha.device)
        expected = (
            self.free.to(device=alpha.device, dtype=alpha.dtype) if self.recovering else self.frozen.to(alpha.device)
        )
        alpha[mask] = expected[mask]
        if optimizer is not None:
            for value in optimizer.state.get(alpha, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == alpha.shape:
                    value.masked_fill_(mask, 0)

    @staticmethod
    def _validate_optimizer(alpha, optimizer):
        if optimizer is not None:
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            if (
                not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
                or len(parameters) != 1
                or parameters[0] is not alpha
            ):
                raise ValueError("score reset requires a separate gate-only Adam optimizer")

    @torch.no_grad()
    def record(self, update, alpha, optimizer=None):
        self._validate_optimizer(alpha, optimizer)
        value = alpha.detach().float().cpu().clone()
        if (
            type(update) is not int
            or update != self.update + 1
            or alpha.dtype != torch.float32
            or value.shape != self.free.shape
            or not torch.isfinite(value).all()
            or (self.final_update is not None and update > self.final_update)
            or not torch.equal(value[~self.free], self.frozen[~self.free])
        ):
            raise ValueError("clipped-reset score, mask, or global cursor differs")
        if self.recovering:
            if not torch.equal(value, self.free.float()):
                raise ValueError("recovery must preserve the terminal binary scores")
            self.update, self.window = update, [value]
            return None
        self.update = update
        self.window.append(value)
        if update % self.interval:
            return None
        if len(self.window) != self.interval + 1:
            raise ValueError("clipped-reset event requires interval+1 snapshots")
        choice = drop_selection(self.free, self.persist, self.window, self.config)
        qualified = choice.pop("qualified")
        incoming = self.persist.clone()
        before_alive = self.free.clone()
        for b, h in choice["selected_ids"]:
            self.free[b, h] = False
        self.persist = qualified & self.free
        # An event is empty only without candidates and without a new persistence flag.
        evidence = choice["eligible_count"] or self.persist.any()
        self.empty_events = 0 if evidence else self.empty_events + 1
        reason = None
        if not self.free.any():
            reason = "all_heads_removed"
        elif self.config.get("stop_rule") == "no_strong_or_no_candidates" and (
            choice["eligible_count"] == 0 or choice["strong_count"] == 0
        ):
            # This event still commits its persistent candidates before the tail.
            reason = "no_strong_or_no_candidates"
        elif self.empty_events >= self.config["empty_events_to_stop"]:
            reason = "evidence_exhausted"
        elif update == self.config["selection_updates"]:
            reason = "pilot_cap"
        if reason is not None:
            self.finalization_start = update
            self.final_update = update + self.config["finalize_updates"]
        reset = self.free.float()
        event = dict(
            update=update,
            config=dict(self.config),
            stage="post_reset",
            **choice,
            incoming_persistence=incoming.tolist(),
            outgoing_persistence=self.persist.tolist(),
            empty_events=self.empty_events,
            terminal_reason=reason,
            finalization_start=self.finalization_start,
            final_update=self.final_update,
            trajectory=[v.tolist() for v in self.window],
            pre_event_scores=value.tolist(),
            pre_event_gates=clipped_gate(value, before_alive).tolist(),
            post_reset_scores=reset.tolist(),
            reset_delta=(reset - value).tolist(),
            gate_cycle=update // self.interval,
            gate_adam_step_after=0,
            counts=self.counts(),
        )
        alpha.copy_(reset.to(alpha.device))
        if optimizer is not None:
            # Removing the state resets both moments AND Adam's local step.
            # Parameter groups retain their LR, betas, epsilon and weight decay.
            optimizer.state.clear()
        alpha.grad = None
        self.history.append(event)
        self.window = [reset.clone()]
        return event

    def counts(self):
        return dict(alive=int(self.free.sum()), removed=int((~self.free).sum()))

    def state_dict(self):
        return copy.deepcopy(
            dict(
                format="progressive_clip_reset_v2",
                config=self.config,
                free=self.free.tolist(),
                frozen=self.frozen.tolist(),
                window=[v.tolist() for v in self.window],
                update=self.update,
                history=self.history,
                finalization_start=self.finalization_start,
                final_update=self.final_update,
                persist=self.persist.tolist(),
                empty_events=self.empty_events,
                event_stage="post_reset"
                if self.history and self.history[-1]["update"] == self.update
                else "between_events",
            )
        )

    @classmethod
    def from_state(cls, state, expected_config=None):
        if state.get("format") != "progressive_clip_reset_v2":
            raise ValueError("unsupported clipped-reset state")
        config = validate_config(state["config"])
        if expected_config is not None and config != validate_config(expected_config):
            raise ValueError("clipped-reset resume settings changed")
        free = torch.tensor(state["free"], dtype=torch.bool)
        policy = cls(torch.ones(free.shape, dtype=torch.float32), config)
        cursor = state["update"]
        if type(cursor) is not int or not 0 <= cursor <= config["selection_updates"] + config["finalize_updates"]:
            raise ValueError("invalid clipped-reset global cursor")
        # Replay saved windows through the actual decision/reset owner. Comparing
        # the complete receipt validates ranking, reset values and terminal rules.
        for event in state["history"]:
            trajectory = [torch.tensor(v, dtype=torch.float32) for v in event["trajectory"]]
            if len(trajectory) != policy.interval + 1 or not torch.equal(trajectory[0], policy.window[-1]):
                raise ValueError("clipped-reset window start or length changed")
            for value in trajectory[1:]:
                expected = policy.record(policy.update + 1, value)
            if event != expected or policy.update > cursor:
                raise ValueError("clipped-reset event receipt changed")
        window = [torch.tensor(v, dtype=torch.float32) for v in state["window"]]
        if not window:
            raise ValueError("missing clipped-reset window")
        if policy.recovering:
            if len(window) != 1 or not torch.equal(window[0], policy.free.float()):
                raise ValueError("clipped-reset recovery scores changed")
            if not policy.update <= cursor <= policy.final_update:
                raise ValueError("clipped-reset recovery cursor changed")
            policy.update, policy.window = cursor, window
        else:
            if not torch.equal(window[0], policy.window[-1]) or len(window) != cursor - policy.update + 1:
                raise ValueError("clipped-reset partial window changed")
            if len(window) > policy.interval:
                raise ValueError("missing clipped-reset event")
            for value in window[1:]:
                policy.record(policy.update + 1, value)
        if state != policy.state_dict():
            raise ValueError("clipped-reset masks, cursor, or transaction stage changed")
        return policy
