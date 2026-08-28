from __future__ import annotations

import pytest
import torch

from scripts.prune.score import lstsq


def test_ridge_recovers_a_known_linear_map():
    torch.manual_seed(1)
    weight = torch.randn(3, 6)
    accumulator = lstsq.RidgeAccumulator(6, 3, "cpu")
    for _ in range(20):
        inputs = torch.randn(6, 64)
        accumulator.add(inputs, weight @ inputs)
    assert accumulator.samples == 1280
    assert (accumulator.solve(ridge=1e-6) - weight).abs().max() < 1e-4


def test_empty_accumulator_refuses_to_solve():
    with pytest.raises(ValueError, match="empty calibration accumulator"):
        lstsq.RidgeAccumulator(4, 2, "cpu").solve()


def test_head_index_expands_to_contiguous_feature_slices():
    assert lstsq.head_index(torch.tensor([0, 2]), dim_head=4).tolist() == list(range(4)) + list(range(8, 12))


def test_attention_accumulator_masks_to_task_tokens(block):
    attention = block.transformer_blocks[0].attn1
    accumulator, add = lstsq.attention_accumulator(attention, torch.tensor([0, 1]))
    add(torch.randn(1, 3, 32), torch.tensor([[[1.0], [0.0], [1.0]]]))
    assert accumulator.samples == 2
