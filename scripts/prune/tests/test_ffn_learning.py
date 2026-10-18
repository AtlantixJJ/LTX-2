"""Independent FFN algebra, real checkpointed graphs, resume and DDP gradients."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.model import X0Model
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import adapters, checkpoints, ffn_training
from scripts.prune.learn.ffn_gates import FFNGates, aligned_width, select_channels
from scripts.prune.score import export_learned_ffn, hooks


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
        block.attn1, block.attn2, block.ff = Attention(), Attention(), FeedForward(4, 4, inner_dim=16)
        self.transformer_blocks = torch.nn.ModuleList([block])

    def forward(self, video, audio, perturbations):
        def block(x):
            b = self.transformer_blocks[0]
            return b.ff(b.attn1(x)) + x

        return checkpoint(block, video.latent, use_reentrant=False), None


def config(arm="joint_heads_ffn"):
    return dict(
        arm=arm,
        seed=42,
        head_rank=2,
        ffn_rank=2,
        ffn_width=16,
        ffn_kept=8,
        ffn_alignment=4,
        ffn_warmup=0,
        ffn_ramp_end=2,
        head_soft_updates=2,
        selection_updates=4,
        recovery_updates=2,
        removed_heads=2,
        gate_lr=0.01,
        adapter_lr=0.001,
        warmup_updates=2,
        betas=[0.9, 0.999],
        adam_epsilon=1e-8,
        lambda_max=0.01,
        lambda_warmup=2,
        lambda_start_multiplier=0.1,
        sparsity_penalty="relu_logits",
        sparsity_logit_offset=3.0,
    )


def make_step(arm="joint_heads_ffn"):
    torch.manual_seed(42)
    return ffn_training.JointStep(X0Model(Velocity()), config(arm), blocks=1, heads=4, allowed=[[2, 4]])


def case(rank=0, number=1):
    generator = torch.Generator().manual_seed(100 * number + rank)
    capture = torch.randn(1, 6, 4, generator=generator)
    epsilon = torch.randn(1, 6, 4, generator=generator)
    latent = 0.5 * capture + 0.5 * epsilon
    latent[:, :1] = capture[:, :1]
    time = torch.full((1, 6, 1), 0.5)
    time[:, :1] = 0
    return (
        SimpleNamespace(tokens_per_latent_frame=1),
        SimpleNamespace(latent=latent, timesteps=time),
        capture[:, :1],
        capture,
        epsilon,
        None,
        {},
    )


def test_aligned_schedule_and_exact_ties():
    assert [aligned_width(t, 16384, 14848) for t in (0, 50, 70, 71, 299, 300, 1000)] == [
        16384,
        16384,
        16384,
        16256,
        14976,
        14848,
        14848,
    ]
    scores = torch.tensor([[2.0, 2.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    assert select_channels(scores, 2).tolist() == [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        aligned_width(1, 16384, 14746)
    with pytest.raises(ValueError):
        select_channels(torch.full((1, 4), float("nan")), 2)


def test_binary_surrogate_and_independent_graphs():
    gates = FFNGates(1, 16, 8, alignment=4, warmup=0, ramp_end=2)
    gates.begin_update(2)
    for _ in range(2):
        value = gates.microbatch()
        assert torch.equal(value.detach(), torch.tensor([[1.0] * 8 + [0.0] * 8]))
        value.sum().backward()
    assert torch.equal(gates.alpha.grad, torch.full_like(gates.alpha, 0.5))
    gates.freeze()
    gates.alpha.grad = None
    gates.microbatch().sum().backward()
    assert torch.equal(gates.alpha.grad, torch.zeros_like(gates.alpha))


def test_native_ffn_provider_matches_manual_post_gelu_mask():
    torch.manual_seed(17)
    ff = FeedForward(4, 4, inner_dim=16)
    block = SimpleNamespace(ff=ff)
    model = SimpleNamespace(transformer_blocks=[block])
    gates = FFNGates(1, 16, 8, alignment=4, warmup=0, ramp_end=1)
    with torch.no_grad():
        gates.alpha.copy_(torch.arange(16).flip(0).reshape(1, -1))
    gates.begin_update(1)
    x = torch.randn(1, 3, 4)
    activation = ff.net[0](x)
    expected = torch.nn.functional.linear(
        activation * gates.decision.reshape(1, 1, -1), ff.net[2].weight, ff.net[2].bias
    )
    with hooks.attach_ffn_provider(model, gates.provider):
        gates.microbatch()
        actual = checkpoint(ff, x, use_reentrant=False)
        assert torch.equal(actual, expected)
        actual.square().sum().backward()
    assert gates.alpha.grad is not None and gates.alpha.grad.norm() > 0


@pytest.mark.parametrize("arm", ["ffn_only", "joint_heads_ffn"])
def test_joint_frozen_backbone_and_exact_resume_across_freeze(tmp_path, arm):
    step = make_step(arm)
    ao, go = ffn_training.optimizers(step)
    frozen = {n: p.detach().clone() for n, p in step.model.named_parameters() if not p.requires_grad}
    for number in (1, 2, 3):
        metrics = ffn_training.update(step, step, case(number=number), number, ao, go)
        assert metrics["ffn_gradient_norm"] > 0
    payload = ffn_training.state(
        step, ao, go, study="tiny", update=3, world_size=1, rng=[checkpoints.rng_state()], input_chain="test"
    )
    path = tmp_path / "state.pt"
    ffn_training.save(path, payload)
    for number in (4, 5):
        ffn_training.update(step, step, case(number=number), number, ao, go)
    expected = ffn_training.state(
        step, ao, go, study="tiny", update=5, world_size=1, rng=[checkpoints.rng_state()], input_chain="test"
    )
    expected_hash = unchanged_state_hash(expected)
    ffn_training.restore(step, ao, go, ffn_training.read(path, study="tiny", world_size=1))
    for number in (4, 5):
        metrics = ffn_training.update(step, step, case(number=number), number, ao, go)
    repeated = ffn_training.state(
        step, ao, go, study="tiny", update=5, world_size=1, rng=[checkpoints.rng_state()], input_chain="test"
    )
    assert unchanged_state_hash(repeated) == expected_hash
    assert metrics["ffn_gradient_norm"] == metrics["head_gradient_norm"] == 0
    assert step.ffn.fixed and step.ffn.decision.sum() == 8
    assert all(torch.equal(frozen[n], p) for n, p in step.model.named_parameters() if n in frozen)
    with pytest.raises(ValueError, match="world size"):
        ffn_training.read(path, world_size=4)
    step.close()


def test_ffn_adapter_and_adam_transfer_use_original_channel_ids():
    state = {
        "0.ff.up.A": torch.randn(2, 4),
        "0.ff.up.B": torch.randn(16, 2),
        "0.ff.down.A": torch.randn(2, 16),
        "0.ff.down.B": torch.randn(4, 2),
    }
    mask = torch.zeros(1, 16)
    mask[0, [1, 3, 5, 7]] = 1
    out = adapters.transfer_ffn(state, mask)
    assert torch.equal(out["0.ff.up.B"], state["0.ff.up.B"][[1, 3, 5, 7]])
    assert torch.equal(out["0.ff.down.A"], state["0.ff.down.A"][:, [1, 3, 5, 7]])
    names = list(state)
    optimizer = dict(
        param_groups=[dict(params=list(range(4)))],
        state={i: dict(step=torch.tensor(3.0), exp_avg=v, exp_avg_sq=v.square()) for i, v in enumerate(state.values())},
    )
    transferred = adapters.transfer_ffn_optimizer(optimizer, names, mask)
    assert torch.equal(transferred["state"][1]["exp_avg"], out["0.ff.up.B"])
    assert transferred["state"][1]["step"] == 3


def test_ffn_merge_rounds_effective_weights_once_and_keeps_bias(tmp_path):
    from scripts.prune.score.export_pruned import PREFIX

    source, output = tmp_path / "base.safetensors", tmp_path / "merged.safetensors"
    tensors = {
        f"{PREFIX}.0.ff.net.0.proj.weight": torch.randn(8, 4).bfloat16(),
        f"{PREFIX}.0.ff.net.2.weight": torch.randn(4, 8).bfloat16(),
        f"{PREFIX}.0.ff.net.2.bias": torch.randn(4).bfloat16(),
    }
    state = {
        "0.ff.up.A": torch.randn(2, 4),
        "0.ff.up.B": torch.randn(8, 2),
        "0.ff.down.A": torch.randn(2, 8),
        "0.ff.down.B": torch.randn(4, 2),
    }
    save_file(tensors, str(source), metadata={"config": json.dumps({"transformer": {}})})
    export_learned_ffn.merge(source, state, output)
    with safe_open(output, framework="pt", device="cpu") as handle:
        for projection, suffix in (("up", "net.0.proj.weight"), ("down", "net.2.weight")):
            name = f"{PREFIX}.0.ff.{suffix}"
            expected = (
                tensors[name].float() + state[f"0.ff.{projection}.B"] @ state[f"0.ff.{projection}.A"]
            ).bfloat16()
            assert torch.equal(handle.get_tensor(name), expected)
        assert torch.equal(handle.get_tensor(f"{PREFIX}.0.ff.net.2.bias"), tensors[f"{PREFIX}.0.ff.net.2.bias"])


def ddp_worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=4)
    step = make_step()
    serial = make_step()
    wrapped = DistributedDataParallel(step, broadcast_buffers=False)
    step.configure(2)
    serial.configure(2)
    c = case(rank)
    wrapped(c[1], c[4], c[3], 1, 2)[0].backward()
    for other in range(4):
        c = case(other)
        (serial(c[1], c[4], c[3], 1, 2)[0] / 4).backward()
    for actual, expected in zip(step.parameters(), serial.parameters(), strict=True):
        if actual.requires_grad:
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-5)
    step.close()
    serial.close()
    dist.destroy_process_group()


def test_joint_four_rank_ddp_matches_serial_average(tmp_path):
    mp.start_processes(ddp_worker, args=(str(tmp_path / "store"),), nprocs=4, start_method="fork", join=True)
