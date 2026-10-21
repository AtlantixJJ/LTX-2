"""Projected evaluation masks must not prevent later training gradients."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.prune.learn.joint_training import ClipGates, JointStep


@pytest.mark.parametrize("evaluation", [torch.no_grad, torch.inference_mode])
def test_projected_monitor_then_training(evaluation):
    step = JointStep.__new__(JointStep)
    nn.Module.__init__(step)
    step.config = {"arm": "budget_joint_v3"}
    step.policy = SimpleNamespace(recovering=False)
    for name in ("heads", "ffn"):
        gates = ClipGates((1, 4), None)
        gates.alpha.data.copy_(torch.tensor([[-2., 0., 1., 3.]]))
        gates.decision = torch.tensor([[1., 0., 1., 0.]])
        setattr(step, name, gates)
    with evaluation():
        step.masks()
        for gates in (step.heads, step.ffn):
            assert torch.equal(gates.current, gates.decision)
            assert not gates.current.requires_grad
    step.masks()
    for gates in (step.heads, step.ffn):
        assert torch.equal(gates.current.detach(), gates.decision)
        weights = torch.tensor([[1., 2., 3., 4.]])
        (gates.current * weights).sum().backward()
        p = (gates.alpha.detach() / 1.5).sigmoid()
        torch.testing.assert_close(gates.alpha.grad, weights * p * (1-p) / 1.5)
        assert torch.equal(gates.gate_gradient, weights)
