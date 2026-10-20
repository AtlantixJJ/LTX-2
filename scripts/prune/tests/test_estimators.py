"""Small exact checks for native structural-unit score definitions."""

import pytest
import torch

from scripts.prune.score import estimators, token_sampling


def test_rms_proxy_and_stable_fractional_allocation() -> None:
    scores = estimators.rms_projection_scores(torch.tensor([4.0, 1.0, 1.0]), torch.tensor([2.0, 1.0, 1.0]), 1)
    assert scores.tolist() == [4.0, 1.0, 1.0]
    assert estimators.fractional_masks({"0.attn1": scores}, 0.34)["0.attn1"] == [1, 0, 1]
    with pytest.raises(ValueError, match="finite"):
        estimators.fractional_masks({"0.ff": torch.tensor([float("nan")])}, 0.1)


def test_sample_indices_use_same_spatial_positions_per_generated_frame() -> None:
    indices = token_sampling.sample_indices(15, 1, 5, 2, torch.device("cpu"))
    assert indices.tolist() == [5, 7, 9, 10, 12, 14]
