"""Four-rank native FSDP check of two-clip gradients and exact checkpoint resume.

Run with torchrun --nproc_per_node=4 in ltx. This small model has frozen BF16
weights, FP32 V/output adapters, live head gates and native checkpoint calls.
Compare a distributed update with serial accumulation; then replay the next
update from a checkpoint and require exact adapter/logit equality.
"""

import argparse
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.transformer_args import TransformerArgs
from scripts.prune.learn import adapters, checkpoints
from scripts.prune.learn.distributed import Runtime, assignment
from scripts.prune.learn.gates import HeadGates
from scripts.prune.score import hooks


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads, self.dim_head = 4, 1
        self.to_v = torch.nn.Linear(4, 4)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    def forward(self, x):
        return self.to_out[0](self.to_v(x))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1, self.attn2 = Attention(), Attention()
        self.scale_shift_table = torch.nn.Parameter(torch.tensor(0.1234567), requires_grad=False)

    def forward(self, video, audio=None):
        x = (self.attn1(video.x).float() + self.scale_shift_table).tanh().to(video.x.dtype)
        return replace(video, x=x), audio


class Velocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Block(), Block()])

    def forward(self, video, audio=None, perturbations=None):
        x = video.latent
        args = TransformerArgs(
            x=x,
            context=torch.zeros_like(x),
            context_mask=None,
            timesteps=torch.ones_like(x),
            embedded_timestep=torch.zeros_like(x),
            positional_embeddings=None,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=True,
        )
        for block in self.transformer_blocks:
            args, _ = checkpoint(block, args, None, use_reentrant=False)
        return args.x, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = Runtime()
    torch.manual_seed(42)
    model = X0Model(Velocity().to(runtime.device, torch.bfloat16))
    for block in model.velocity_model.transformer_blocks:
        block.scale_shift_table.data = torch.tensor(0.1234567, device=runtime.device, dtype=torch.float32)
    reference = copy.deepcopy(model)
    gates = HeadGates(2, 4, [[2, 4], [2, 4]]).to(runtime.device)
    reference_gates = copy.deepcopy(gates)
    attached = adapters.attach(model, 42, 2)
    reference_adapters = adapters.attach(reference, 42, 2)
    parameters = adapters.parameters(attached)
    ao = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    go = torch.optim.AdamW([gates.alpha], lr=0.01, weight_decay=0)
    reference_ao = torch.optim.AdamW(adapters.parameters(reference_adapters), lr=0.001, weight_decay=0)
    reference_go = torch.optim.AdamW([reference_gates.alpha], lr=0.01, weight_decay=0)
    model = runtime.wrap(model, parameters, Block)
    assert runtime.inventory["replicated_frozen_bytes"] == 8
    frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
    h = hooks.attach_head_provider(model, gates.provider)
    rh = hooks.attach_head_provider(reference, reference_gates.provider)

    def step(candidate, decisions, matrices, optimizer, gate_optimizer, distributed):
        optimizer.zero_grad(set_to_none=True)
        gate_optimizer.zero_grad(set_to_none=True)
        decisions.phase = "soft"
        draws = [assignment(runtime.rank)] if distributed else [0, 1]
        for slot in draws:
            g = decisions.microbatch()
            x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(100 + slot)).to(
                runtime.device, torch.bfloat16
            )
            value, _ = candidate.velocity_model(SimpleNamespace(latent=x))
            loss = value.float().square().mean() + 0.001 * g.mean()
            (loss / (1 if distributed else 2)).backward()
        if distributed:
            runtime.reduce_gradients(adapters.parameters(matrices) + [decisions.alpha])
        torch.nn.utils.clip_grad_norm_(adapters.parameters(matrices), 1.0)
        torch.nn.utils.clip_grad_norm_([decisions.alpha], 1.0)
        optimizer.step()
        gate_optimizer.step()

    step(model, gates, attached, ao, go, True)
    step(reference, reference_gates, reference_adapters, reference_ao, reference_go, False)
    for key, value in adapters.state_dict(attached).items():
        torch.testing.assert_close(value, adapters.state_dict(reference_adapters)[key], atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(gates.alpha, reference_gates.alpha, atol=2e-6, rtol=0)
    if any(p._version != version for p, version in frozen):
        raise ValueError("FSDP changed a frozen tensor version")
    path = args.output / "state.pt"
    runtime.save(
        path,
        identity={"study": "small_fsdp"},
        update=1,
        phase="soft",
        gates=gates,
        adapters=attached,
        adapter_optimizer=ao,
        gate_optimizer=go,
    )
    step(model, gates, attached, ao, go, True)
    expected = adapters.state_dict(attached)
    expected_alpha = gates.alpha.detach().clone()
    state = checkpoints.load(path, {"study": "small_fsdp"})
    adapters.load_state(attached, state["adapters"])
    gates.alpha.data.copy_(state["alpha"])
    ao.load_state_dict(state["adapter_optimizer"])
    go.load_state_dict(state["gate_optimizer"])
    checkpoints.restore_rng(state["rng"])
    step(model, gates, attached, ao, go, True)
    if any(not torch.equal(value, expected[key]) for key, value in adapters.state_dict(attached).items()):
        raise ValueError("resumed FSDP adapter update differs")
    if not torch.equal(gates.alpha, expected_alpha):
        raise ValueError("resumed FSDP logit update differs")
    runtime.agree(checkpoints.tensor_hash(gates.alpha))
    if runtime.primary:
        checkpoints.publish_json(
            args.output / "passed.json",
            {"passed": True, "world_size": 4, "serial_update": True, "exact_resume": True, "frozen_versions": True},
        )
        print("four-rank FSDP serial-gradient and exact-resume checks passed", flush=True)
    h.detach_all()
    rh.detach_all()
    runtime.close()


if __name__ == "__main__":
    main()
