from __future__ import annotations

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from scripts.prune.learn.gates import HeadGates
from scripts.prune.score import hooks


def test_head_mask_zeroes_exactly_that_heads_slice(block: torch.nn.Module) -> None:
    attention = block.transformer_blocks[0].attn1
    inputs = torch.randn(1, 5, 32)
    manual = inputs.clone().reshape(1, 5, 4, 8)
    manual[:, :, 2, :] = 0
    expected = attention.to_out[0](manual.reshape(1, 5, 32))
    masks = {"0.attn1": torch.tensor([1.0, 1.0, 0.0, 1.0]), "0.attn2": torch.ones(4)}
    with hooks.attach_head_masks(block, masks, requires_grad=False):
        actual = attention.to_out[0](inputs)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, attention.to_out[0](inputs))


def test_hooks_are_fully_removed_on_context_exit(block: torch.nn.Module) -> None:
    attention = block.transformer_blocks[0].attn1
    before = len(attention.to_out[0]._forward_pre_hooks)
    with hooks.attach_head_masks(block):
        assert len(attention.to_out[0]._forward_pre_hooks) == before + 1
    assert len(attention.to_out[0]._forward_pre_hooks) == before


def test_module_names_match_remove_head_cli_grammar(block: torch.nn.Module) -> None:
    assert list(dict(hooks.iter_video_attention(block))) == ["0.attn1", "0.attn2"]
    assert list(dict(hooks.iter_video_ffn(block))) == ["0.ff"]


def test_wrong_width_mask_is_rejected(block: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match="!= heads"):
        hooks.attach_head_masks(block, {"0.attn1": torch.ones(7)})


def test_ffn_mask_zeroes_intermediate_channels(block: torch.nn.Module) -> None:
    ffn = block.transformer_blocks[0].ff
    inputs = torch.randn(1, 4, 128)
    keep = torch.ones(128)
    keep[:32] = 0
    manual = inputs.clone()
    manual[..., :32] = 0
    with hooks.attach_ffn_masks(block, {"0.ff": keep}, requires_grad=False):
        assert torch.equal(ffn.net[2](inputs), ffn.net[2](manual))


def test_live_provider_retains_gate_graph_and_ignores_text(block: torch.nn.Module) -> None:

    gates = HeadGates(1, 4, [[2, 4]])
    gates.begin_update("soft")
    attention = block.transformer_blocks[0].attn1
    source = torch.randn(1, 5, 32)
    with hooks.attach_head_provider(block, gates.provider):
        for _ in range(2):
            gates.microbatch()
            attention.to_out[0](source).square().mean().backward()
        assert not block.transformer_blocks[0].attn2.to_out[0]._forward_pre_hooks
    assert gates.alpha.grad is not None
    assert torch.isfinite(gates.alpha.grad).all()
    assert not attention.to_out[0]._forward_pre_hooks


def test_live_gate_checkpoint_recomputation_is_stable(block: torch.nn.Module) -> None:

    gate = HeadGates(1, 4, [[2, 4]])
    attention = block.transformer_blocks[0].attn1
    observed = []

    def provider(name: str) -> torch.Tensor:
        observed.append(gate.current.detach().clone())
        return gate.provider(name)

    gate.begin_update("binary", 2)
    gate.microbatch()
    x = torch.randn(1, 3, 32, requires_grad=True)
    with hooks.attach_head_provider(block, provider):
        value = checkpoint(attention.to_out[0], x, use_reentrant=False)
        value.square().sum().backward()
    assert len(observed) >= 2
    assert all(torch.equal(observed[0], v) for v in observed)
    assert gate.alpha.grad is not None
    assert torch.isfinite(gate.alpha.grad).all()
