"""Saved small-model tensors pin the extraction across calls, gradients and cache state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import dataset
from scripts.onestep_avatar.model import bidirectional, causal, common
from scripts.onestep_avatar.tests.test_causal_core import _model

FIXTURES = dataset.WORKSPACE_ROOT / "expr/onestep_avatar/two_mode_restructure_20261005/fixtures"


@pytest.mark.parametrize(
    "case", ["bidirectional_reference", "causal_blocks_zero_one", "causal_later_start", "causal_cache_removal"]
)
@pytest.mark.parametrize("precision", ["historical_bfloat16", "ordinary_float32"])
def test_mode_matches_saved_prediction_loss_gradient_and_cache(case: str, precision: str, monkeypatch) -> None:
    path = FIXTURES / f"{case}.pt"
    if not path.exists():
        pytest.skip(f"saved baseline fixture is not installed: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if precision == "historical_bfloat16":
        original_modality = common.block_modality
        def historical_modality(*args, **kwargs):
            modality = original_modality(*args, **kwargs)
            # Exact old producer: only global sigma inherited the latent dtype.
            # Token timestep precision remains float32, as in the saved calls.
            return replace(modality, sigma=modality.sigma.to(modality.latent.dtype))
        monkeypatch.setattr(common, "block_modality", historical_modality)
        monkeypatch.setattr(causal, "block_modality", historical_modality)

    model = _model(prompt_adaln=True).bfloat16()
    model.load_state_dict(saved["model_state"])
    settings = saved["geometry"]
    geometry = causal.CausalGeometry(
        SpatioTemporalScaleFactors(*settings["scale_factors"]),
        settings["block_latent_frames"],
        settings["context_latent_frames"],
        settings["sink_latent_frames"],
    )
    capture = saved["capture"]
    grid = common.ClipGrid.build(
        capture.shape[1],
        capture.shape[2] * 32,
        capture.shape[3] * 32,
        30,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        latent_channels=capture.shape[0],
    )
    target = grid.patchify(capture.unsqueeze(0))
    guide = grid.patchify(saved["guide"].unsqueeze(0))
    calls = []

    def record_call(module, args, kwargs, output):
        modality = kwargs["video"]
        calls.append(
            {
                "latent": modality.latent.detach().clone(),
                "sigma": modality.sigma.detach().clone(),
                "timesteps": modality.timesteps.detach().clone(),
                "positions": modality.positions.detach().clone(),
                "velocity": output[0].detach().clone(),
            }
        )

    handle = model.register_forward_hook(record_call, with_kwargs=True)
    cache = None
    if case == "bidirectional_reference":
        result = bidirectional.train_sample(
            model,
            saved["context"],
            grid,
            target,
            guide,
            lambda loss: loss.backward(),
            sigma=saved["sigma"],
            seed=saved["noise_seed"],
            accumulation=saved["accumulation"],
        )
        expected_calls = [call for call in saved["calls"] if call["sigma"].item() > 0]
        assert result["prime_calls"] == 0
    else:
        cache = causal.BlockCache.allocate(
            grid, geometry, num_layers=2, inner_dim=8, device=torch.device("cpu"), dtype=torch.bfloat16
        )
        result = causal.train_sample(
            model,
            saved["context"],
            grid,
            target,
            guide,
            geometry,
            saved["blocks"],
            lambda loss: loss.backward(),
            sigma=saved["sigma"],
            seed=saved["noise_seed"],
            cache=cache,
            accumulation=saved["accumulation"],
        )
        expected_calls = saved["calls"]
        assert result["prime_calls"] == 1
        assert result["refresh_calls"] == len(saved["blocks"]) - 1
    handle.remove()
    if precision == "ordinary_float32":
        assert all(call["sigma"].dtype == call["timesteps"].dtype == torch.float32 for call in calls)
        assert torch.isfinite(torch.tensor(result["loss"]))
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        first = next(call for call in calls if call["sigma"].item() > 0)
        old_first = next(call for call in expected_calls if call["sigma"].item() > 0)
        for field in ("latent", "timesteps", "positions"):
            assert torch.equal(first[field], old_first[field]), (case, field)
        assert not torch.equal(first["sigma"].float(), old_first["sigma"].float())
        return
    assert result["loss"] == saved["metrics"]["loss"]
    assert result["mse"] == saved["metrics"]["mse"]
    assert len(calls) == len(expected_calls)
    for actual, expected in zip(calls, expected_calls, strict=True):
        for field in actual:
            assert torch.equal(actual[field], expected[field]), (case, field)
    for name, parameter in model.named_parameters():
        if name in saved["gradients"]:
            assert torch.equal(parameter.grad, saved["gradients"][name]), (case, name)
        else:
            assert parameter.grad is None, (case, name)
    if cache is not None:
        for current, expected in zip(cache.caches, saved["cache"], strict=True):
            assert current.length == expected["length"]
            assert torch.equal(current.k[:, : current.length], expected["k"])
            assert torch.equal(current.v[:, : current.length], expected["v"])
