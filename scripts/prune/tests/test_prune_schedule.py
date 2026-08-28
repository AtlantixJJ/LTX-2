from __future__ import annotations

import pytest
import torch

from scripts.prune.score import ffn_scores, hooks, prune_schedule


def test_iterative_head_masks_hit_target_exactly(block):
    torch.manual_seed(0)
    scores = {name: torch.rand(attention.heads) for name, attention in hooks.iter_video_attention(block)}
    masks, history = prune_schedule.iterative_head_masks(block, target_sparsity=0.25, rounds=2, rescore=lambda _: scores)
    assert sum(int((value == 0).sum()) for value in masks.values()) == 2
    assert len(history) == 2 and history[-1]["total"] == 8


def test_iterative_head_masks_remove_lowest_scoring_heads(block):
    scores = {"0.attn1": torch.tensor([0.9, 0.1, 0.8, 0.7]), "0.attn2": torch.tensor([0.6, 0.5, 0.4, 0.2])}
    masks, _ = prune_schedule.iterative_head_masks(block, target_sparsity=0.25, rounds=1, rescore=lambda _: scores)
    assert masks["0.attn1"].tolist() == [1, 0, 1, 1]
    assert masks["0.attn2"].tolist() == [1, 1, 1, 0]


def test_rescore_sees_current_mask_each_round(block):
    seen = []

    def rescore(masks):
        seen.append({name: value.clone() for name, value in masks.items()})
        return {name: torch.rand(attention.heads) for name, attention in hooks.iter_video_attention(block)}

    prune_schedule.iterative_head_masks(block, target_sparsity=0.5, rounds=2, rescore=rescore)
    assert int(sum((value == 0).sum() for value in seen[0].values())) == 0
    assert int(sum((value == 0).sum() for value in seen[1].values())) > 0


@pytest.mark.parametrize(("sparsity", "rounds"), [(1.0, 1), (-0.1, 1), (0.5, 0)])
def test_invalid_arguments_raise(block, sparsity, rounds):
    with pytest.raises(ValueError):
        prune_schedule.iterative_head_masks(block, target_sparsity=sparsity, rounds=rounds, rescore=lambda _: {})


def test_ffn_masks_keep_every_layer_executable():
    masks = ffn_scores.masks_from_scores({"0.ff": torch.arange(128.0)}, target_sparsity=0.99, min_keep=1)
    assert int(masks["0.ff"].sum()) == 1 and masks["0.ff"][127] == 1
