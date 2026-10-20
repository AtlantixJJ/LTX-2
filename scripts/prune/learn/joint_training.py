"""V3 joint clipped selection, task-gradient receipts and exact checkpoint state.

The historical FFN learner remains separate. The DDP reducer averages the total
loss once. The known analytical penalty is subtracted to recover task gradients
in raw-score units; executed-gate gradients are captured before its surrogate.
See doc/learn/joint_training.md.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from scripts.prune.core import ltx_adapter, records
from scripts.prune.learn import adapters, checkpoints, losses
from scripts.prune.learn.readmission import JointPolicy, budget_projection
from scripts.prune.score import hooks


class ClipGates(nn.Module):
    def __init__(self, shape, family):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        self.family = family
        self.phase = "soft"
        self.current = None
        self.decision = torch.ones(shape)
        self.fixed = False
        self.gate_gradient = None
        self.projected = False

    def microbatch(self):
        if self.phase == "all_one":
            self.current = torch.ones_like(self.alpha) + self.alpha * 0
        else:
            self.current = self.family.compose(self.alpha, fixed=self.fixed)
        if self.current.requires_grad:
            self.current.register_hook(self.capture_gradient)
        return self.current

    def capture_gradient(self, gradient):
        self.gate_gradient = gradient.detach().float().clone()

    def provider(self, name):
        if self.current is None:
            raise ValueError("missing joint gate graph")
        return self.current[int(name.split(".")[0])]

    def snapshot(self):
        if self.projected and not self.fixed:
            mask = self.decision.bool().cpu()
            return dict(
                retained_per_block=mask.sum(1).tolist(),
                counts=dict(kept=int(mask.sum()), removed=int((~mask).sum()), permanent=0, returned=0),
                mask_sha256=records.tensor_hash(mask),
                fixed=False,
            )
        return dict(
            retained_per_block=self.family.kept.sum(1).tolist(),
            counts=self.family.counts(),
            mask_sha256=records.tensor_hash(self.family.kept),
            fixed=self.fixed,
        )


class JointStep(nn.Module):
    def __init__(self, model, config, *, blocks=48, heads=32):
        super().__init__()
        self.model = model.requires_grad_(False)
        self.config = config
        self.policy = JointPolicy(
            (blocks, heads),
            (blocks, config["ffn_width"]),
            enabled=config["arm"] != "heads_clip_reset_v3",
            **config["policy"],
        )
        self.heads = ClipGates((blocks, heads), self.policy.heads)
        self.ffn = ClipGates((blocks, config["ffn_width"]), self.policy.channels)
        device = next(model.parameters()).device
        self.heads.to(device)
        self.ffn.to(device)
        self.attached = adapters.attach(model, config["seed"], config["head_rank"])
        if config["arm"] != "heads_clip_reset_v3":
            self.attached.update(adapters.attach_ffn(model, config["seed"], config["ffn_rank"]))
        self.live_heads = hooks.attach_head_provider(model, self.heads.provider)
        self.live_ffn = hooks.attach_ffn_provider(model, self.ffn.provider)
        generator = torch.Generator().manual_seed(42)
        self.sample_ids = torch.randperm(self.ffn.alpha.numel(), generator=generator)[:4096]
        self.event = None
        self.task_gradients = None
        self.trace = None
        self.projection = None
        self.optimizer_damping = None
        if config["arm"] == "budget_joint_v3":
            self.policy.projection_mode = True
            self.heads.projected = self.ffn.projected = True
            with torch.no_grad():
                self.heads.alpha.zero_()
                self.ffn.alpha.zero_()

    @property
    def frozen(self):
        return self.policy.recovering

    def configure(self, update, *, all_one=False):
        self.heads.phase = self.ffn.phase = "all_one" if all_one else "soft"
        self.heads.fixed = self.ffn.fixed = self.frozen
        if self.config["arm"] == "budget_joint_v3" and not self.frozen and not all_one:
            total = self.heads.alpha.numel() * 704 + self.ffn.alpha.numel()
            ramp_end = self.config.get("budget_ramp_end", min(300, self.config["selection_updates"] - 100))
            ramp_start = self.config.get("budget_ramp_start", 50)
            portion = min(max((update - ramp_start) / (ramp_end - ramp_start), 0), 1)
            budget = total + portion * (self.config["budget"] - total)
            h, c, self.projection = budget_projection(
                (self.heads.alpha / 1.5).sigmoid(),
                (self.ffn.alpha / 1.5).sigmoid(),
                budget,
                minimum=self.config["ffn_alignment"],
            )
            self.heads.decision, self.ffn.decision = h, c

    def masks(self):
        for gates in (self.heads, self.ffn):
            if self.config["arm"] == "budget_joint_v3" and not self.frozen and gates.phase != "all_one":
                p = (gates.alpha / 1.5).sigmoid()
                gates.current = gates.decision.to(p.device) + (p - p.detach())
                gates.current.register_hook(gates.capture_gradient)
            else:
                gates.microbatch()

    def forward(self, video, epsilon, capture, prefix_tokens, update):
        self.masks()
        velocity, _ = ltx_adapter.velocity_forward(self.model, video)
        terms = losses.velocity_mse(velocity, epsilon, capture, prefix_tokens)
        penalty = self.heads.alpha.sum() * 0 + self.ffn.alpha.sum() * 0
        if not self.frozen and self.config["arm"] != "budget_joint_v3":
            weight = self.config["lambda_max"] * (0.1 + 0.9 * min(update / 100, 1)) / self.heads.alpha.numel()
            penalty = weight * (
                self.policy.heads.penalty(self.heads.alpha)
                + self.config["rho"] * self.policy.channels.penalty(self.ffn.alpha)
            )
        loss = terms["official"] + penalty
        if not torch.isfinite(loss):
            raise ValueError("nonfinite joint loss")
        return loss, terms["official"].detach(), penalty.detach()

    def close(self):
        self.live_heads.detach_all()
        self.live_ffn.detach_all()


class GateOptimizers:
    def __init__(self, step):
        optimizer = torch.optim.AdamW if step.config["arm"] == "budget_joint_v3" else torch.optim.Adam
        self.items = [
            optimizer([g.alpha], lr=step.config["gate_lr"], betas=(0.9, 0.999), eps=eps, weight_decay=0)
            for g, eps in ((step.heads, 1e-8), (step.ffn, 1e-12))
        ]
        if step.config["arm"] == "budget_joint_v3":
            self.items[0].param_groups[0]["eps"] = 1e-12

    def zero_grad(self, **kwargs):
        for item in self.items:
            item.zero_grad(**kwargs)

    def state_dict(self):
        return [o.state_dict() for o in self.items]

    def load_state_dict(self, states):
        for item, state in zip(self.items, states, strict=True):
            item.load_state_dict(state)


def optimizers(step):
    return (
        torch.optim.AdamW(
            adapters.parameters(step.attached),
            lr=step.config["adapter_lr"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        ),
        GateOptimizers(step),
    )


def update(step, wrapped, case, number, adapter_optimizer, gate_optimizer):
    grid, video, _, capture, epsilon, _, _ = case
    step.configure(number)
    adapter_optimizer.zero_grad(set_to_none=True)
    gate_optimizer.zero_grad(set_to_none=True)
    tail = number - step.policy.selection_end if step.frozen else 0
    lr = step.config["adapter_lr"] * (
        max(1 - tail / step.config["recovery_updates"], 0)
        if step.frozen
        else min(number / step.config["warmup_updates"], 1)
    )
    for group in adapter_optimizer.param_groups:
        group["lr"] = lr
    loss, official, penalty = wrapped(video, epsilon, capture, grid.tokens_per_latent_frame, number)
    loss.backward()
    gradients = {}
    for name, g in (("heads", step.heads), ("channels", step.ffn)):
        task = g.gate_gradient.clone()
        # Gate-output hooks see local task gradients before the surrogate. This
        # separate average is diagnostic; never apply it to parameter gradients.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(task)
            task.div_(torch.distributed.get_world_size())
        gradients[name] = task.cpu()
        if not torch.isfinite(g.alpha.grad).all():
            raise ValueError("nonfinite gate gradient")
        permanent = g.family.permanent.to(g.alpha.device)
        if g.alpha.grad[permanent].ne(0).any():
            raise ValueError("permanent total gradient is nonzero")
    step.task_gradients = gradients
    anorm = torch.nn.utils.clip_grad_norm_(adapters.parameters(step.attached), 1, error_if_nonfinite=True)
    hnorm, fnorm = float(step.heads.alpha.grad.norm()), float(step.ffn.alpha.grad.norm())
    adapter_optimizer.step()
    step.event = None
    if not step.frozen:
        damping = {}
        for family_name, g, o in zip(("heads", "channels"), (step.heads, step.ffn), gate_optimizer.items, strict=True):
            o.step()
            moments = o.state[g.alpha]
            beta2 = o.param_groups[0]["betas"][1]
            denominator = (moments["exp_avg_sq"] / (1 - beta2 ** float(moments["step"]))).sqrt()
            factor = denominator / (denominator + o.param_groups[0]["eps"])
            damping[family_name] = dict(
                epsilon=o.param_groups[0]["eps"],
                median_factor_per_block=factor.detach().cpu().median(1).values.tolist(),
                total_gradient_rms_per_block=g.alpha.grad.square().mean(1).sqrt().detach().cpu().tolist(),
            )
            with torch.no_grad():
                if step.config["arm"] != "budget_joint_v3":
                    g.alpha[g.family.permanent.to(g.alpha.device)] = 0
        step.optimizer_damping = damping
        step.trace = dict(
            update=number,
            heads=step.heads.alpha.detach().cpu().clone(),
            channels=step.ffn.alpha.detach().cpu().flatten()[step.sample_ids].clone(),
        )
        if step.config["arm"] == "budget_joint_v3":
            step.policy.update = number
            if number == step.config["selection_updates"]:
                step.policy.selection_end = number
                step.policy.stop_reason = "matched_A_endpoint"
                step.policy.heads.kept = step.heads.decision.bool().cpu()
                step.policy.channels.kept = step.ffn.decision.bool().cpu()
                for f in (step.policy.heads, step.policy.channels):
                    f.reset_summary(f.kept.float())
                step.policy.reset(step.heads.alpha, step.ffn.alpha, gate_optimizer.items)
                step.event = dict(update=number, reason="matched_A_endpoint", selection_end=number)
        else:
            step.event = step.policy.record(number, step.heads.alpha, step.ffn.alpha, gate_optimizer.items)
    else:
        step.policy.record(number, step.heads.alpha, step.ffn.alpha, gate_optimizer.items)
    return dict(
        total_loss=float(loss.detach()),
        official=float(official),
        penalty=float(penalty),
        adapter_gradient_norm=float(anorm),
        head_gradient_norm=hnorm,
        ffn_gradient_norm=fnorm,
        adapter_lr=lr,
    )


def state(step, ao, go, *, study, update, world_size, rng, input_chain):
    return dict(
        format="joint_learning_state_v3",
        study=study,
        update=update,
        world_size=world_size,
        draw_cursor=4 * update,
        input_chain=input_chain,
        frozen=step.frozen,
        heads_alpha=step.heads.alpha.detach().cpu().clone(),
        ffn_alpha=step.ffn.alpha.detach().cpu().clone(),
        heads_mask=step.policy.heads.kept.float(),
        ffn_mask=step.policy.channels.kept.float(),
        policy=step.policy.state_dict(),
        adapters=adapters.state_dict(step.attached),
        adapter_optimizer=copy.deepcopy(ao.state_dict()),
        gate_optimizer=copy.deepcopy(go.state_dict()),
        rng=rng,
        sample_ids=step.sample_ids.clone(),
        trace=copy.deepcopy(step.trace),
        projected_heads=step.heads.decision.detach().cpu().clone(),
        projected_channels=step.ffn.decision.detach().cpu().clone(),
    )


def read(path, *, study=None, world_size=None):
    return checkpoints.read_state(
        path, format="joint_learning_state_v3", label="joint", study=study, world_size=world_size, draw_stride=4
    )


def restore(step, ao, go, payload, *, rank=0, restore_rng=True):
    policy = JointPolicy.from_state(payload["policy"])
    if policy.config != step.policy.config or policy.update != payload["update"]:
        raise ValueError("joint checkpoint policy differs")
    for gates, key, family in ((step.heads, "heads_alpha", policy.heads), (step.ffn, "ffn_alpha", policy.channels)):
        value = payload[key]
        if value.shape != gates.alpha.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("invalid joint checkpoint scores")
        with torch.no_grad():
            gates.alpha.copy_(value)
        gates.family = family
        gates.fixed = policy.recovering
    step.policy = policy
    step.heads.decision = payload["projected_heads"].to(step.heads.alpha.device)
    step.ffn.decision = payload["projected_channels"].to(step.ffn.alpha.device)
    for gates, family, key in ((step.heads, policy.heads, "heads_mask"), (step.ffn, policy.channels, "ffn_mask")):
        if not torch.equal(payload[key], family.kept.float()):
            raise ValueError("joint checkpoint mask differs from policy")
        if policy.recovering and not torch.equal(gates.alpha.detach().cpu(), family.kept.float()):
            raise ValueError("joint recovery score differs from binary mask")
        if not policy.recovering and step.config["arm"] != "budget_joint_v3":
            if not torch.equal(gates.alpha.detach().cpu(), family.final):
                raise ValueError("joint checkpoint score differs from window summary")
    step.sample_ids, step.trace = payload["sample_ids"].clone(), copy.deepcopy(payload["trace"])
    adapters.load_state(step.attached, payload["adapters"])
    ao.load_state_dict(payload["adapter_optimizer"])
    go.load_state_dict(payload["gate_optimizer"])
    if restore_rng:
        checkpoints.restore_rng(payload["rng"][rank])


def publish_record(path, payload):
    records.publish_payload(path, payload)


def save(path, payload):
    return checkpoints.publish_state(path, payload)
