"""Real four-rank CPU DDP versus serial accumulation, with gates and checkpointing."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from ltx_core.model.transformer.model import X0Model
from scripts.prune.core import records
from scripts.prune.data.learned_head_study import draw_list
from scripts.prune.learn import adapters, checkpoints, protocol
from scripts.prune.learn.ddp import DDPRuntime
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.step import TrainingStep
from scripts.prune.score import hooks


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads, self.dim_head = 4, 1
        self.to_v = torch.nn.Linear(4, 4)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    def forward(self, x):
        return self.to_out[0](self.to_v(x))


class Velocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.attn1, block.attn2 = Attention(), Attention()
        self.transformer_blocks = torch.nn.ModuleList([block])

    def forward(self, video, audio, perturbations):
        return checkpoint(self.transformer_blocks[0].attn1, video.latent, use_reentrant=False), None


def cfg():
    return {
        "distributed_training": protocol.DDP_METHOD,
        "uniform_f18": True,
        "gpu_ids": [0, 1, 2, 3],
        "global_unique_clips": 4,
        "rank_draw_assignment": [0, 1, 2, 3],
        "accumulation": 1,
        "lambda_max": 0.1,
        "lambda_warmup": 100,
        "lambda_start_multiplier": 0.1,
        "sparsity_penalty": "relu_logits",
    }


def inputs(rank, update):
    generator = torch.Generator().manual_seed(100 * update + rank)
    y = torch.randn(1, 18, 4, generator=generator)
    eps = torch.randn(1, 18, 4, generator=generator)
    sigma = 0.5 + rank * 0.1
    latent = (1 - sigma) * y + sigma * eps
    latent[:, :1] = y[:, :1]
    timesteps = torch.full((1, 18, 1), sigma)
    timesteps[:, :1] = 0
    return SimpleNamespace(latent=latent, timesteps=timesteps), eps, y


def worker(rank, store, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=4)
    runtime = DDPRuntime.__new__(DDPRuntime)
    runtime.rank = runtime.local_rank = rank
    runtime.world = 4
    runtime.device = torch.device("cpu")
    torch.manual_seed(42)
    model = X0Model(Velocity()).requires_grad_(False)
    modules = adapters.attach(model, 42, 2)
    gates = HeadGates(1, 4, [[2, 4]])
    live = hooks.attach_head_provider(model, gates.provider)
    serial_model = copy.deepcopy(model)
    serial_modules = {
        name: serial_model.velocity_model.transformer_blocks[0].attn1.to_v
        if name.endswith(".v")
        else serial_model.velocity_model.transformer_blocks[0].attn1.to_out[0]
        for name in modules
    }
    serial_gates = HeadGates(1, 4, [[2, 4]])
    # deepcopy retains hooks; replace them with the serial gate provider.
    for module in serial_model.modules():
        module._forward_pre_hooks.clear()
    serial_live = hooks.attach_head_provider(serial_model, serial_gates.provider)
    step = runtime.wrap_step(TrainingStep(model, gates, cfg(), "learned_joint"))
    serial = TrainingStep(serial_model, serial_gates, cfg(), "learned_joint")
    params = adapters.parameters(modules)
    ao = torch.optim.AdamW(params, lr=0.001, weight_decay=0)
    go = torch.optim.AdamW([gates.alpha], lr=0.1, weight_decay=0)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}

    def update(number, compare=False):
        ao.zero_grad(set_to_none=True)
        go.zero_grad(set_to_none=True)
        loss, _, _ = step(*inputs(rank, number), 1, number, "soft", None, 1)
        loss.backward()
        runtime.reduce_gradients(params + [gates.alpha])
        if compare:
            for other in range(4):
                serial_loss, _, _ = serial(*inputs(other, number), 1, number, "soft", None, 4)
                serial_loss.backward()
            for actual, expected in zip(
                params + [gates.alpha], adapters.parameters(serial_modules) + [serial_gates.alpha], strict=True
            ):
                torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=1e-5)
        ao.step()
        go.step()
        digest = records.digest(
            {
                "alpha": records.tensor_hash(gates.alpha),
                "adapters": {k: records.tensor_hash(v) for k, v in adapters.state_dict(modules).items()},
            }
        )
        runtime.agree(digest)
        return digest

    update(1, compare=True)
    saved = root + "/state.pt"
    runtime.save(
        saved,
        identity={"study": "tiny"},
        update=1,
        phase="soft",
        gates=gates,
        adapters=modules,
        adapter_optimizer=ao,
        gate_optimizer=go,
    )
    expected = update(2)
    state = checkpoints.load(saved, {"study": "tiny"})
    assert state["draw_cursor"] == 4 and state["rng"]["world_size"] == 4
    adapters.load_state(modules, state["adapters"])
    gates.alpha.data.copy_(state["alpha"])
    ao.load_state_dict(state["adapter_optimizer"])
    go.load_state_dict(state["gate_optimizer"])
    checkpoints.restore_rng(state["rng"])
    assert update(2) == expected
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)
    rows = runtime.draws([{"rank": rank}])
    assert [r["rank"] for r in rows] == [0, 1, 2, 3]
    live.detach_all()
    serial_live.detach_all()
    runtime.close()


def test_four_rank_ddp_gradient_mean_and_exact_resume(tmp_path):
    mp.start_processes(worker, args=(str(tmp_path / "gloo"), str(tmp_path)), nprocs=4, join=True, start_method="spawn")


def test_four_draw_schedule_and_protocol_refuse_legacy_batch():
    config = cfg()
    protocol.validate(config)
    cycle = [0, 6, 1, 5, 2, 4, 3]
    draws = draw_list([{"relative_dir": "one"}], 42, updates=7, sigma_indices=cycle, draws_per_update=4)
    assert len(draws) == 28
    assert {i: sum(d["sigma_index"] == i for d in draws) for i in range(7)} == dict.fromkeys(range(7), 4)
    selected = [protocol.update_draws(draws, 1, config, rank)[0] for rank in range(4)]
    assert [d["microbatch"] for d in selected] == [0, 1, 2, 3]
    for wrong in (
        {"uniform_f18": False},
        {"accumulation": 2},
        {"rank_draw_assignment": [0, 1, 0, 1]},
        {"global_unique_clips": 2},
    ):
        with pytest.raises(ValueError, match="DDP requires"):
            protocol.validate({**config, **wrong})
