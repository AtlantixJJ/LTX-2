"""Versioned clipped-score removal, survivor resets, and an exact adapter-only tail.

Raw scores stay FP32 and are never projected between events. Each event consumes
one complete window, then starts a new window from the binary reset snapshot.
The reset jump belongs to neither trajectory. Historical sigmoid states cannot
be migrated into this policy. See doc/learn/clipped_reset.md.
"""

from __future__ import annotations

import copy

import torch

from scripts.prune.learn.progressive import ProgressivePolicy, selection, statistics

DEFAULTS = dict(
    method="clip01_ste3_reset_v1",
    activation="clip01_v1",
    surrogate="clamp_minus3_plus3_inclusive_v1",
    reset="survivors_one_removed_zero_v1",
    gate_optimizer_reset="all_adam_state_v1",
    hardening_interval=50,
    removal_fraction=0.02,
    regret_threshold=0.10,
    selection_population="eligible",
    minimum_count=16,
    selection_updates=500,
    finalize_updates=300,
)


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError("clipped-reset settings must have the complete versioned schema")
    for key in ("method", "activation", "surrogate", "reset", "gate_optimizer_reset", "selection_population"):
        if config[key] != DEFAULTS[key]:
            raise ValueError(f"unsupported clipped-reset {key}")
    for key in ("hardening_interval", "minimum_count", "selection_updates", "finalize_updates"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"clipped-reset {key} must be a positive integer")
    if config["selection_updates"] % config["hardening_interval"]:
        raise ValueError("selection cap must end at an event")
    for key, lower, inclusive in (("removal_fraction", 0, False), ("regret_threshold", 0, True)):
        value = config[key]
        if type(value) not in (int, float) or not (lower <= value <= 1 if inclusive else lower < value <= 1):
            raise ValueError(f"invalid clipped-reset {key}")
    return dict(config)


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
        travel, decrease, regret = statistics(self.window)
        choice = selection(self.free, travel, decrease, regret, self.config)
        before_alive = self.free.clone()
        for b, h in choice["selected_ids"]:
            self.free[b, h] = False
        reason = None
        if not self.free.any():
            reason = "all_heads_removed"
        elif choice["eligible_count"] == 0:
            reason = "no_eligible_heads"
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
            terminal_reason=reason,
            finalization_start=self.finalization_start,
            final_update=self.final_update,
            trajectory=[v.tolist() for v in self.window],
            travel=travel.tolist(),
            decrease=decrease.tolist(),
            rate=(decrease / self.interval).tolist(),
            regret=[[float(v) if torch.isfinite(v) else None for v in row] for row in regret],
            pre_event_scores=value.tolist(),
            pre_event_gates=clipped_gate(value, before_alive).tolist(),
            post_reset_scores=reset.tolist(),
            reset_delta=(reset - value).tolist(),
            gate_cycle=update // self.interval,
            gate_adam_step_after=0,
            selected_fraction=choice["selected_count"] / choice["eligible_count"] if choice["eligible_count"] else 0.0,
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
                format="progressive_clip_reset_v1",
                config=self.config,
                free=self.free.tolist(),
                frozen=self.frozen.tolist(),
                window=[v.tolist() for v in self.window],
                update=self.update,
                history=self.history,
                finalization_start=self.finalization_start,
                final_update=self.final_update,
                event_stage="post_reset"
                if self.history and self.history[-1]["update"] == self.update
                else "between_events",
            )
        )

    @classmethod
    def from_state(cls, state, expected_config=None):
        if state.get("format") != "progressive_clip_reset_v1":
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
