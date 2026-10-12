"""Matched direct-mode checks against the causal mode's one-block training call."""

from __future__ import annotations

import copy
import random
from typing import Never

import pytest
import torch

from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.model import bidirectional, common, sampling
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.tests.support.model import _model
from scripts.onestep_avatar.training import config


@pytest.mark.parametrize("guide_mode", ["d0", "d1"])
@pytest.mark.parametrize("sigma", [0.725, 1.0])
@pytest.mark.parametrize("start", [0, 2])
def test_same_inputs_match_original_prediction_loss_and_gradient(
    guide_mode: str, sigma: float, start: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    scale = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    geometry = causal_core.CausalGeometry(scale, block_latent_frames=2, context_latent_frames=1)
    grid = common.ClipGrid.build(
        3, 64, 64, 30, geometry, device=torch.device("cpu"), dtype=torch.bfloat16, latent_channels=8
    )
    rng = torch.Generator().manual_seed(7)
    capture_master = torch.randn(8, 3 + start, 2, 2, generator=rng).bfloat16()
    guide_master = torch.randn(8, 3 + start, 2, 2, generator=rng).bfloat16()
    context = torch.randn(1, 3, 16, generator=rng).bfloat16()
    original = _model().bfloat16()
    current = copy.deepcopy(original)
    # A segment is a clip of its own: its frame 0 (the future c0) is master frame ``start``.
    capture = capture_master[:, start : start + 3].contiguous()
    guide = guide_master[:, start : start + 3].contiguous()
    assert torch.equal(capture[:, 0], capture_master[:, start])
    cache = causal_core.BlockCache.allocate(
        grid, geometry, num_layers=2, inner_dim=8, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    old = causal_core.train_sample(
        original,
        context,
        grid,
        grid.patchify(capture.unsqueeze(0)),
        grid.patchify(guide.unsqueeze(0)),
        geometry,
        [0],
        lambda loss: loss.backward(),
        sigma=sigma,
        seed=19,
        cache=cache,
        guide_mode=guide_mode,
        accumulation=3,
    )

    def forbidden(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise AssertionError("bidirectional mode called a causal helper")

    monkeypatch.setattr(causal_core.BlockCache, "allocate", forbidden)
    monkeypatch.setattr(causal_core, "prime_cache", forbidden)
    monkeypatch.setattr(causal_core, "denoise_block", forbidden)
    monkeypatch.setattr(causal_core, "refresh_block", forbidden)
    target = grid.patchify(capture.unsqueeze(0))
    source_guide = grid.patchify(guide.unsqueeze(0))
    result = bidirectional.train_sample(
        current,
        context,
        grid,
        target,
        source_guide,
        lambda loss: loss.backward(),
        sigma=sigma,
        seed=19,
        guide_mode=guide_mode,
        accumulation=3,
    )
    assert result["loss"] == old["loss"]
    for (name, old_parameter), (_, parameter) in zip(
        original.named_parameters(), current.named_parameters(), strict=True
    ):
        if old_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            assert torch.equal(old_parameter.grad, parameter.grad), name
    modalities = []

    def predict(modality: Modality) -> torch.Tensor:
        modalities.append(modality)
        return common.denoised_from_velocity_model(current)(modality)

    output, counts = bidirectional.sample(
        predict,
        context,
        grid,
        common.source_for(target, source_guide, guide_mode),
        target[:, : grid.tokens_per_latent_frame],
        schedule=[sigma, 0],
        seed=19,
    )
    assert common.full_frame_mse(output, target).item() == result["loss"]
    assert counts == {"denoise_calls": 1, "prime_calls": 0, "refresh_calls": 0}
    assert len(modalities) == 1
    modality = modalities[0]
    assert modality.sigma.dtype == torch.float32
    assert modality.timesteps.dtype == torch.float32
    assert modality.sigma.item() == torch.tensor(sigma, dtype=torch.float32).item()
    assert modality.kv_caches is None
    assert not modality.kv_write
    assert modality.attention_mask is None
    assert torch.equal(modality.latent[:, :4], target[:, :4])
    assert torch.count_nonzero(modality.timesteps[:, :4]) == 0
    assert torch.equal(output[:, :4], target[:, :4])


def test_random_frame_selection_preserves_the_original_seed_keys() -> None:
    key = config.window_start_draw(42)["key"]
    for step in range(5):
        for rank in range(2):
            expected = random.Random(key.format(seed=42, step=step, rank=rank, slot=1)).randrange(28 - 17 + 1)
            assert bidirectional.plan_samples(
                28, span_latent_frames=17, start_policy="random", seed=42, step=step, rank=rank, slot=1
            ) == (expected, expected + 17)
    assert common.pixel_frames_for(17, 8) == 129


@pytest.mark.parametrize("levels", [[float("nan"), 0], [float("inf"), 0], [1.1, 0], [0.5, 0.6, 0], [0.5, -0.1, 0]])
def test_invalid_schedule_fails_before_prediction(levels: list[float]) -> None:
    with pytest.raises(ValueError, match="schedule"):
        sampling.validate_schedule(levels)
