"""Small exact checks for native structural-unit score definitions."""

import pytest
import torch

from scripts.prune.score import estimators
from scripts.prune.score.whole_clip_d0_scores import _sample_indices


def test_exact_local_energy_accounts_for_within_head_covariance() -> None:
    activation = torch.tensor([[[1.0, 1.0, 2.0, 0.0], [3.0, -3.0, 0.0, 2.0]]])
    weight = torch.tensor([[1.0, -1.0, 1.0, 0.0]])
    score = estimators.exact_local_head_energy(activation, weight, heads=2,
                                                sample_indices=torch.tensor([0, 1]))
    assert score.tolist() == pytest.approx([float(18**0.5), float(2**0.5)])


def test_rms_proxy_and_stable_fractional_allocation() -> None:
    scores = estimators.rms_projection_scores(torch.tensor([4.0, 1.0, 1.0]),
                                               torch.tensor([2.0, 1.0, 1.0]), 1)
    assert scores.tolist() == [4.0, 1.0, 1.0]
    assert estimators.fractional_masks({"0.attn1": scores}, 0.34)["0.attn1"] == [1, 0, 1]
    with pytest.raises(ValueError, match="finite"):
        estimators.fractional_masks({"0.ff": torch.tensor([float("nan")])}, 0.1)


def test_sample_indices_use_same_spatial_positions_per_generated_frame() -> None:
    indices = _sample_indices(15, 5, 2, torch.device("cpu"))
    assert indices.tolist() == [5, 7, 9, 10, 12, 14]
