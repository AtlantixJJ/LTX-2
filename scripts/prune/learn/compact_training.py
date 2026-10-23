"""Train fresh FP32 adapters on a fixed, physically compact BF16 backbone.

Original joint adapters are already fused into the prepared base. Attach new
zero-output adapters, freeze the base and optimize generated-token velocity MSE
with the configured warmup/cosine learning rate. No selection or sparsity loss
runs here. NoGateOptimizer satisfies the shared checkpoint interface without
owning gate parameters. tests/test_compact_recovery.py checks exact Adam replay."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import torch
from torch import nn

from scripts.prune.core import ltx_adapter
from scripts.prune.learn import adapters, checkpoints, losses


class JointStep(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model.requires_grad_(False)
        self.config = config
        self.attached = adapters.attach(model, config["seed"], config["head_rank"])
        self.attached.update(adapters.attach_ffn(model, config["seed"], config["ffn_rank"]))
        self.heads = SimpleNamespace(phase="fixed_compact", current=None)
        self.ffn = SimpleNamespace(current=None)
        self.policy = SimpleNamespace(selection_end=0, final_update=config["recovery_updates"])

    def configure(self, update, *, all_one=False):
        pass

    def masks(self):
        pass

    def close(self):
        pass

    def forward(self, video, epsilon, capture, prefix_tokens, update):
        velocity, _ = ltx_adapter.velocity_forward(self.model, video)
        loss = losses.velocity_mse(velocity, epsilon, capture, prefix_tokens)["official"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite compact velocity loss")
        return loss, loss.detach(), loss.detach() * 0


class NoGateOptimizer:
    def state_dict(self):
        return {}


def optimizers(step):
    return torch.optim.AdamW(adapters.parameters(step.attached), lr=step.config["adapter_lr"],
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0), NoGateOptimizer()


def learning_rate(config, number):
    warmup, total = config["warmup_updates"], config["recovery_updates"]
    if number <= warmup:
        return config["adapter_lr"] * number / warmup
    portion = min((number - warmup) / max(total - warmup, 1), 1)
    return config["minimum_lr"] + (config["adapter_lr"] - config["minimum_lr"]) * (1 + math.cos(math.pi * portion)) / 2


def update(step, wrapped, case, number, adapter_optimizer, gate_optimizer):
    grid, video, _, capture, epsilon, _, _ = case
    adapter_optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step.config, number)
    for group in adapter_optimizer.param_groups:
        group["lr"] = lr
    loss, official, penalty = wrapped(video, epsilon, capture, grid.tokens_per_latent_frame, number)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(adapters.parameters(step.attached), 1, error_if_nonfinite=True)
    adapter_optimizer.step()
    return dict(total_loss=float(loss.detach()), official=float(official), penalty=float(penalty),
                adapter_gradient_norm=float(norm), head_gradient_norm=0.0, ffn_gradient_norm=0.0, adapter_lr=lr)


def state(step, ao, go, *, study, update, world_size, rng, input_chain):
    return dict(format="compact_learning_state_v1", study=study, update=update, world_size=world_size,
                draw_cursor=4 * update, input_chain=input_chain, adapters=adapters.state_dict(step.attached),
                adapter_optimizer=copy.deepcopy(ao.state_dict()), rng=rng)


def read(path, *, study=None, world_size=None):
    return checkpoints.read_state(path, format="compact_learning_state_v1", label="compact", study=study,
                                   world_size=world_size, draw_stride=4)


def restore(step, ao, go, payload, *, rank=0, restore_rng=True):
    adapters.load_state(step.attached, payload["adapters"])
    ao.load_state_dict(payload["adapter_optimizer"])
    if restore_rng:
        checkpoints.restore_rng(payload["rng"][rank])


save = checkpoints.publish_state
