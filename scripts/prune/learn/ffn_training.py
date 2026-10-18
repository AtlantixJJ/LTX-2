"""Numerical joint head/FFN steps and independently versioned learning state."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from scripts.prune.core import ltx_adapter
from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import adapters, checkpoints, losses
from scripts.prune.learn.ffn_gates import FFNGates
from scripts.prune.learn.gates import HeadGates, select_legal
from scripts.prune.score import hooks


class JointStep(nn.Module):
    def __init__(self, model, config, *, blocks=48, heads=32, allowed=None):
        super().__init__()
        self.model = model.requires_grad_(False)
        self.config = config
        self.heads = HeadGates(blocks, heads, allowed)
        self.ffn = FFNGates(
            blocks,
            config["ffn_width"],
            config["ffn_kept"],
            alignment=config["ffn_alignment"],
            warmup=config["ffn_warmup"],
            ramp_end=config["ffn_ramp_end"],
        )
        self.heads.to(next(model.parameters()).device)
        self.ffn.to(next(model.parameters()).device)
        if config["arm"] == "joint_heads_ffn":
            self.attached = adapters.attach(model, config["seed"], config["head_rank"])
        else:
            self.attached = {}
        self.attached.update(adapters.attach_ffn(model, config["seed"], config["ffn_rank"]))
        self.live_heads = hooks.attach_head_provider(model, self.heads.provider)
        self.live_ffn = hooks.attach_ffn_provider(model, self.ffn.provider)
        self.head_kept = blocks * heads - config["removed_heads"]
        self.frozen = False

    def configure(self, update, *, all_one=False):
        if all_one:
            self.heads.phase = "all_one"
            self.ffn.begin_update(0, all_one=True)
        elif self.frozen:
            self.heads.phase = "compact" if self.config["arm"] == "joint_heads_ffn" else "all_one"
        else:
            if self.config["arm"] == "ffn_only":
                self.heads.phase = "all_one"
            elif update <= self.config["head_soft_updates"]:
                self.heads.phase = "soft"
            else:
                self.heads.begin_update("binary", self.head_kept)
            self.ffn.begin_update(update)

    def freeze(self):
        self.ffn.freeze()
        self.heads.decision = (
            select_legal(self.heads.alpha, self.head_kept, self.heads.allowed).to(self.heads.alpha.device)
            if self.config["arm"] == "joint_heads_ffn"
            else torch.ones_like(self.heads.alpha)
        )
        self.frozen = True
        self.configure(self.config["selection_updates"] + 1)

    def masks(self):
        self.heads.microbatch()
        self.ffn.microbatch()

    def forward(self, video, epsilon, capture, prefix_tokens, update):
        self.masks()
        velocity, _ = ltx_adapter.velocity_forward(self.model, video)
        terms = losses.velocity_mse(velocity, epsilon, capture, prefix_tokens)
        loss, penalty = losses.phase_loss(
            terms,
            self.heads.current,
            update=update,
            phase=self.heads.phase,
            accumulation=1,
            lambda_max=self.config["lambda_max"] if self.config["arm"] == "joint_heads_ffn" else 0,
            lambda_warmup=self.config["lambda_warmup"],
            lambda_start_multiplier=self.config["lambda_start_multiplier"],
            logits=self.heads.alpha,
            sparsity_kind=self.config["sparsity_penalty"],
            logit_offset=self.config["sparsity_logit_offset"],
        )
        # Fixed and all-one head paths still participate in the same DDP reducer.
        loss = loss + self.heads.alpha.sum() * 0 + self.ffn.alpha.sum() * 0
        if not torch.isfinite(loss):
            raise ValueError("nonfinite joint head/FFN loss")
        return loss, terms["official"].detach(), penalty.detach()

    def close(self):
        self.live_ffn.detach_all()
        self.live_heads.detach_all()


def optimizers(step):
    cfg = step.config
    kwargs = dict(betas=tuple(cfg["betas"]), eps=cfg["adam_epsilon"], weight_decay=0)
    return (
        torch.optim.AdamW(adapters.parameters(step.attached), lr=cfg["adapter_lr"], **kwargs),
        torch.optim.AdamW(
            [step.heads.alpha, step.ffn.alpha] if cfg["arm"] == "joint_heads_ffn" else [step.ffn.alpha],
            lr=cfg["gate_lr"],
            **kwargs,
        ),
    )


def update(step, wrapped, case, number, adapter_optimizer, gate_optimizer):
    grid, video, _, capture, epsilon, _, _ = case
    step.configure(number)
    step.heads.alpha.grad = None
    adapter_optimizer.zero_grad(set_to_none=True)
    gate_optimizer.zero_grad(set_to_none=True)
    for group in adapter_optimizer.param_groups:
        group["lr"] = step.config["adapter_lr"] * min(number / step.config["warmup_updates"], 1)
    loss, official, penalty = wrapped(video, epsilon, capture, grid.tokens_per_latent_frame, number)
    loss.backward()
    params = adapters.parameters(step.attached)
    anorm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
    hnorm = torch.nn.utils.clip_grad_norm_([step.heads.alpha], 1, error_if_nonfinite=True)
    fnorm = torch.nn.utils.clip_grad_norm_([step.ffn.alpha], 1, error_if_nonfinite=True)
    if any(p.grad is None for p in params):
        raise ValueError("missing adapter gradient in joint training")
    adapter_optimizer.step()
    if not step.frozen:
        gate_optimizer.step()
    if number == step.config["selection_updates"]:
        step.freeze()
    return dict(
        total_loss=float(loss.detach()),
        official=float(official),
        penalty=float(penalty),
        adapter_gradient_norm=float(anorm),
        head_gradient_norm=float(hnorm),
        ffn_gradient_norm=float(fnorm),
    )


def state(step, adapter_optimizer, gate_optimizer, *, study, update, world_size, rng, input_chain):
    return dict(
        format="head_ffn_learning_state_v1",
        study=study,
        update=update,
        world_size=world_size,
        draw_cursor=world_size * update,
        input_chain=input_chain,
        frozen=step.frozen,
        heads_alpha=step.heads.alpha.detach().cpu().clone(),
        heads_mask=step.heads.decision.detach().cpu().clone() if step.heads.decision is not None else None,
        ffn_alpha=step.ffn.alpha.detach().cpu().clone(),
        ffn_mask=step.ffn.decision.detach().cpu().clone(),
        head_phase=step.heads.phase,
        adapters=adapters.state_dict(step.attached),
        adapter_optimizer=adapter_optimizer.state_dict(),
        gate_optimizer=gate_optimizer.state_dict(),
        rng=rng,
    )


def save(path, payload):
    path = Path(path)
    state_digest = unchanged_state_hash(payload)
    if path.exists():
        if unchanged_state_hash(read(path)) != state_digest:
            raise ValueError("replayed FFN checkpoint differs")
    else:

        def write(p):
            with p.open("wb") as stream:
                torch.save(payload, stream)

        checkpoints.publish(path, write)
    checkpoints.publish_json(
        path.with_suffix(".json"),
        dict(
            format=payload["format"],
            study=payload["study"],
            update=payload["update"],
            sha256=file_sha256(path),
            state_sha256=state_digest,
            draw_cursor=payload["draw_cursor"],
        ),
    )


def read(path, *, study=None, world_size=None):
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError("FFN checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != "head_ffn_learning_state_v1"
        or unchanged_state_hash(payload) != receipt["state_sha256"]
    ):
        raise ValueError("FFN checkpoint state changed")
    if study is not None and payload["study"] != study:
        raise ValueError("FFN checkpoint belongs to another study")
    if world_size is not None and payload["world_size"] != world_size:
        raise ValueError("FFN checkpoint world size differs")
    if payload["draw_cursor"] != payload["world_size"] * payload["update"]:
        raise ValueError("FFN checkpoint draw cursor differs")
    if len(payload["rng"]) != payload["world_size"]:
        raise ValueError("FFN checkpoint rank RNG coverage differs")
    return payload


def restore(step, adapter_optimizer, gate_optimizer, payload, *, rank=0, restore_rng=True):
    if payload["heads_alpha"].shape != step.heads.alpha.shape or payload["ffn_alpha"].shape != step.ffn.alpha.shape:
        raise ValueError("FFN checkpoint score geometry differs")
    for key in ("heads_alpha", "ffn_alpha", "ffn_mask"):
        if payload[key].dtype != torch.float32 or not torch.isfinite(payload[key]).all():
            raise ValueError("FFN checkpoint scores or masks are invalid")
    mask = payload["ffn_mask"]
    expected = step.ffn.kept if payload["frozen"] else int(mask[0].sum())
    if (
        mask.shape != step.ffn.alpha.shape
        or not ((mask == 0) | (mask == 1)).all()
        or not (mask.sum(1) == expected).all()
    ):
        raise ValueError("FFN checkpoint binary width differs")
    if not 0 < expected <= step.ffn.width or expected % step.ffn.schedule["alignment"]:
        raise ValueError("FFN checkpoint retained width is not legal")
    head_mask = payload["heads_mask"]
    if payload["frozen"] and head_mask is None:
        raise ValueError("frozen FFN checkpoint lacks its head mask")
    if head_mask is not None:
        if head_mask.shape != step.heads.alpha.shape or not ((head_mask == 0) | (head_mask == 1)).all():
            raise ValueError("invalid original head mask in FFN checkpoint")
        kept = step.heads.alpha.numel() if step.config["arm"] == "ffn_only" else step.head_kept
        if int(head_mask.sum()) != kept or any(
            int(row.sum()) not in allowed for row, allowed in zip(head_mask, step.heads.allowed, strict=True)
        ):
            raise ValueError("FFN checkpoint head count is not legal")
    with torch.no_grad():
        step.heads.alpha.copy_(payload["heads_alpha"])
        step.ffn.alpha.copy_(payload["ffn_alpha"])
    step.ffn.decision = mask.to(step.ffn.alpha.device)
    step.ffn.fixed = step.frozen = payload["frozen"]
    step.heads.decision = (
        payload["heads_mask"].to(step.heads.alpha.device) if payload["heads_mask"] is not None else None
    )
    step.heads.phase = payload["head_phase"]
    adapters.load_state(step.attached, payload["adapters"])
    adapter_optimizer.load_state_dict(payload["adapter_optimizer"])
    gate_optimizer.load_state_dict(payload["gate_optimizer"])
    if restore_rng:
        checkpoints.restore_rng(payload["rng"][rank])
