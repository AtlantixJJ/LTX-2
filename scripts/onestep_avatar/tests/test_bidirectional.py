"""Matched direct-mode checks against the original one-block training path."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.model import bidirectional, common, sampling
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.tests.test_causal_core import _model
from scripts.onestep_avatar.training import engine as train


@pytest.mark.parametrize("guide_mode", ["d0", "d1"])
@pytest.mark.parametrize("sigma", [0.725, 1.0])
@pytest.mark.parametrize("start", [0, 2])
def test_same_inputs_match_original_prediction_loss_and_gradient(guide_mode, sigma, start, monkeypatch):
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
    chain = train.window_chain(
        train.Chain("fixture", "train", "actor", True, [0], guide_master, capture_master, 30), start, 3
    )
    capture = capture_master[:, start:start+3].contiguous()
    guide = guide_master[:, start:start+3].contiguous()
    assert torch.equal(chain.z_y, capture) and torch.equal(chain.z_g, guide)
    assert torch.equal(capture[:, 0], capture_master[:, start])
    cache = causal_core.BlockCache.allocate(
        grid, geometry, num_layers=2, inner_dim=8, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    old = train.train_chain(
        original,
        context,
        chain,
        geometry,
        cache,
        SimpleNamespace(device=torch.device("cpu"), backward=lambda loss: loss.backward()),
        sigma0=sigma,
        seed=19,
        latent_channels=8,
        guide_mode=guide_mode,
        accumulation=3,
    )

    def forbidden(*args, **kwargs):
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

    def predict(modality):
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
    assert modality.kv_caches is None and not modality.kv_write and modality.attention_mask is None
    assert torch.equal(modality.latent[:, :4], target[:, :4])
    assert torch.count_nonzero(modality.timesteps[:, :4]) == 0
    assert torch.equal(output[:, :4], target[:, :4])


def test_random_frame_selection_preserves_the_original_seed_keys():
    for step in range(5):
        for rank in range(2):
            expected = train.window_start_for(42, step=step, rank=rank, slot=1, latent_frames=28, window=17)
            assert bidirectional.plan_samples(
                28, span_latent_frames=17, start_policy="random", seed=42, step=step, rank=rank, slot=1
            ) == (expected, expected + 17)
    assert common.pixel_frames_for(17, 8) == 129


@pytest.mark.parametrize("levels", [[float("nan"), 0], [float("inf"), 0], [1.1, 0], [0.5, 0.6, 0], [0.5, -0.1, 0]])
def test_invalid_schedule_fails_before_prediction(levels):
    with pytest.raises(ValueError):
        sampling.validate_schedule(levels)
