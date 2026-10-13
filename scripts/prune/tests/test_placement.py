import copy
from dataclasses import replace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from ltx_core.model.transformer.transformer_args import TransformerArgs
from scripts.prune.learn.placement import move_args, partition


def arguments(x):
    return TransformerArgs(
        x=x,
        context=torch.zeros_like(x),
        context_mask=torch.ones(x.shape[:2], dtype=torch.bool, device=x.device),
        timesteps=torch.ones_like(x),
        embedded_timestep=torch.zeros_like(x),
        positional_embeddings=(torch.zeros_like(x), torch.ones_like(x)),
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=None,
        cross_gate_timestep=None,
        enabled=True,
    )


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)

    def forward(self, video, audio=None):
        return replace(video, x=self.projection(video.x).tanh()), audio


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(4)])

    def forward(self, video):
        for block in self.transformer_blocks:
            video, _ = checkpoint(block, video, None, use_reentrant=False)
        return video.x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_model = Core()


def test_cpu_transport_preserves_fields_and_gradient():
    x = torch.randn(1, 2, 4, requires_grad=True)
    source = arguments(x)
    transported = move_args(source, "cpu")
    assert transported.x is x
    assert transported.enabled
    assert transported.context_mask.dtype == torch.bool
    transported.x.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x)
    with pytest.raises(ValueError, match="causal"):
        move_args(replace(source, kv_write=True), "cpu")
    with pytest.raises(TypeError, match="TransformerArgs"):
        move_args(x, "cpu")


def test_partition_hooks_recompute_and_cleanup_on_failure():
    model = Model()
    with pytest.raises(ValueError, match="distinct"):
        with partition(model, ["cpu", "cpu"]):
            pass
    with pytest.raises(RuntimeError, match="test failure"):
        with partition(model, ["cpu"]):
            model.velocity_model(arguments(torch.randn(1, 2, 4))).sum().backward()
            assert all(p.grad is not None for p in model.parameters())
            raise RuntimeError("test failure")
    assert all(
        not block._forward_pre_hooks and not block._forward_hooks for block in model.velocity_model.transformer_blocks
    )


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four visible GPUs")
def test_four_gpu_checkpoint_matches_single_gpu_gradients():
    torch.manual_seed(42)
    reference = Model().cuda(0)
    student = copy.deepcopy(reference)
    x = torch.randn(1, 3, 4, device="cuda:0", requires_grad=True)
    baseline = reference.velocity_model(arguments(x))
    baseline.square().mean().backward()
    expected = [p.grad.detach().cpu().clone() for p in reference.parameters()]
    expected_input = x.grad.detach().clone()
    y = x.detach().clone().requires_grad_(True)
    with partition(student, [f"cuda:{i}" for i in range(4)]):
        output = student.velocity_model(arguments(y))
        assert output.device == torch.device("cuda:0")
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        output.square().mean().backward()
        torch.testing.assert_close(y.grad, expected_input, rtol=0, atol=0)
        for parameter, expected_gradient in zip(student.parameters(), expected, strict=True):
            torch.testing.assert_close(parameter.grad.cpu(), expected_gradient, rtol=0, atol=0)
