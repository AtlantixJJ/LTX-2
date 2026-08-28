from __future__ import annotations

import pytest
import torch

from scripts.prune.score import hooks


def test_head_mask_zeroes_exactly_that_heads_slice(block):
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


def test_hooks_are_fully_removed_on_context_exit(block):
    attention = block.transformer_blocks[0].attn1
    before = len(attention.to_out[0]._forward_pre_hooks)
    with hooks.attach_head_masks(block):
        assert len(attention.to_out[0]._forward_pre_hooks) == before + 1
    assert len(attention.to_out[0]._forward_pre_hooks) == before


def test_module_names_match_remove_head_cli_grammar(block):
    assert list(dict(hooks.iter_video_attention(block))) == ["0.attn1", "0.attn2"]
    assert list(dict(hooks.iter_video_ffn(block))) == ["0.ff"]


def test_wrong_width_mask_is_rejected(block):
    with pytest.raises(ValueError, match="!= heads"):
        hooks.attach_head_masks(block, {"0.attn1": torch.ones(7)})


def test_ffn_mask_zeroes_intermediate_channels(block):
    ffn = block.transformer_blocks[0].ff
    inputs = torch.randn(1, 4, 128)
    keep = torch.ones(128)
    keep[:32] = 0
    manual = inputs.clone()
    manual[..., :32] = 0
    with hooks.attach_ffn_masks(block, {"0.ff": keep}, requires_grad=False):
        assert torch.equal(ffn.net[2](inputs), ffn.net[2](manual))
