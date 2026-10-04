"""Sampling budget and native spatial coordinates are independent of scoring quality."""

import hashlib
import struct

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import causal_core
from scripts.prune.score import token_sampling
from scripts.prune.score.whole_clip_d0_scores import _sample_indices


@pytest.mark.parametrize(("height", "width", "stride"), [(32, 32, 16), (3, 5, 4), (1, 5, 2), (7, 1, 3)])
def test_default_sampler_exactly_preserves_old_stride(height: int, width: int, stride: int) -> None:
    device = torch.device("cpu")
    tokens = 5 * height * width
    assert torch.equal(token_sampling.sample_indices(tokens, height, width, stride, device),
                       _sample_indices(tokens, height * width, stride, device))


@pytest.mark.parametrize(("height", "width", "stride"), [(32, 32, 16), (5, 11, 7), (3, 17, 4),
                                               (1, 13, 3), (13, 1, 2), (4, 7, 1), (4, 7, 100)])
def test_balanced_grid_matches_budget_uniqueness_and_excludes_c0(height: int, width: int, stride: int) -> None:
    per_frame = height * width
    indices = token_sampling.sample_indices(5 * per_frame, height, width, stride, torch.device("cpu"),
                                            sampler="balanced_2d_midpoint_v1")
    budget = len(range(0, per_frame, stride))
    assert indices.numel() == 4 * budget
    assert torch.unique(indices).numel() == indices.numel()
    assert indices.min() >= per_frame
    assert indices.max() < 5 * per_frame
    spatial = indices.reshape(4, budget) % per_frame
    assert torch.equal(spatial, spatial[0].expand_as(spatial))
    row_counts = torch.unique(spatial[0] // width, return_counts=True)[1]
    assert row_counts.max() - row_counts.min() <= 1


def test_32_grid_expands_columns_without_claiming_full_spatial_coverage() -> None:
    record = _record(32, 32, 16, "balanced_2d_midpoint_v1")
    assert record["tokens_per_generated_frame"] == 64
    assert record["sampled_rows"] == [2, 6, 10, 14, 18, 22, 26, 30]
    assert record["sampled_columns"] == [2, 6, 10, 14, 18, 22, 26, 30]
    assert _record(32, 32, 16, "stride")["sampled_columns"] == [0, 16]


def _record(height: int, width: int, stride: int, sampler: str) -> dict:
    indices = token_sampling.sample_indices(3 * height * width, height, width, stride,
                                            torch.device("cpu"), sampler=sampler)
    return token_sampling.sampling_record(indices, tokens=3 * height * width, height=height, width=width,
                                          stride=stride, sampler=sampler)


def test_index_hash_has_explicit_integer_encoding_and_geometry_pin() -> None:
    indices = torch.tensor([5, 9, 17], dtype=torch.int64)
    expected = hashlib.sha256(b"".join(struct.pack("<Q", index) for index in [5, 9, 17])).hexdigest()
    assert token_sampling.index_sha256(indices) == expected
    assert _record(3, 5, 2, "balanced_2d_midpoint_v1") == _record(3, 5, 2, "balanced_2d_midpoint_v1")
    assert _record(3, 5, 2, "balanced_2d_midpoint_v1") != _record(5, 3, 2, "balanced_2d_midpoint_v1")


def test_sampler_indices_match_real_patchifier_row_column_coordinates() -> None:
    height, width, frames = 3, 5, 4
    geometry = causal_core.CausalGeometry(scale_factors=SpatioTemporalScaleFactors(time=8, height=32, width=32))
    grid = causal_core.ClipGrid.build(frames, height * 32, width * 32, 30, geometry,
                                    device=torch.device("cpu"), dtype=torch.float32, latent_channels=1)
    latent = torch.empty(1, 1, frames, height, width)
    for frame in range(frames):
        for row in range(height):
            for column in range(width):
                latent[0, 0, frame, row, column] = 100 * frame + 10 * row + column
    indices = token_sampling.sample_indices(frames * height * width, height, width, 3,
                                            torch.device("cpu"), sampler="balanced_2d_midpoint_v1")
    selected = grid.patchify(latent)[0, indices, 0]
    expected = [100 * (index // (height * width)) + 10 * ((index % (height * width)) // width) + index % width
                for index in indices.tolist()]
    assert selected.tolist() == expected


@pytest.mark.parametrize(("tokens", "height", "width", "stride", "sampler"), [
    (15, 3, 5, 2, "stride"), (31, 3, 5, 2, "stride"),
    (30, 0, 5, 2, "stride"), (30, 3, 5, 0, "stride"), (30, 3, 5, True, "stride"), (30, 3, 5, 2, "random")])
def test_invalid_grid_or_sampler_fails(tokens: int, height: int, width: int, stride: int, sampler: str) -> None:
    with pytest.raises(ValueError, match=r"token|stride|sampler"):
        token_sampling.sample_indices(tokens, height, width, stride, torch.device("cpu"), sampler=sampler)


def test_mutated_indices_cannot_reuse_a_sampling_record() -> None:
    indices = token_sampling.sample_indices(30, 3, 5, 2, torch.device("cpu"))
    indices[0] = 0
    with pytest.raises(ValueError, match="declared deterministic"):
        token_sampling.sampling_record(indices, tokens=30, height=3, width=5, stride=2, sampler="stride")
