"""Eight-block diagnostics keep full-master noise and the shared cached rollout."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.prune.core.session as sessions
from ltx_core.model.transformer.model import X0Model
from scripts.onestep_avatar.experiments import causality as evaluate
from scripts.onestep_avatar.model import causal, common
from scripts.onestep_avatar.tests.support.fixtures_experiment_test_causality_diagnostic import (
    fixture_inputs,
)
from scripts.onestep_avatar.tests.support.model import CHANNELS, SCALE, _context, _model
from scripts.prune.core.session import DTYPE


def _global_epsilons(
    source: torch.Tensor, grid: common.ClipGrid, plan: list[tuple[int, int]], seed: int
) -> list[torch.Tensor]:
    """One epsilon over every token, sliced by block: the saved full-master noise rule."""
    epsilon = common.epsilon_block(source, seed)
    return [epsilon[:, slice(*grid.token_span(*span))] for span in plan]


def _original_rollout(
    model: torch.nn.Module,
    context: torch.Tensor,
    grid: common.ClipGrid,
    geometry: causal.CausalGeometry,
    capture: torch.Tensor,
    guide: torch.Tensor,
    sigma: float,
    *,
    seed: int,
    epsilons: list[torch.Tensor],
    blocks: list[tuple[int, int]],
) -> torch.Tensor:
    """The cached D1 rollout through the model owner: one ``causal.rollout`` call with a
    fresh deployed cache, capture ``c0`` and the guide as noising source.
    """
    base = common.base_model(model)
    cache = causal.BlockCache.allocate(
        grid,
        geometry,
        num_layers=len(base.transformer_blocks),
        inner_dim=base.inner_dim,
        device=torch.device("cpu"),
        dtype=DTYPE,
    )
    capture_tokens = grid.patchify(capture.unsqueeze(0).to(dtype=DTYPE))
    source = grid.patchify(guide.unsqueeze(0).to(dtype=DTYPE))
    c0 = capture_tokens[:, : grid.tokens_per_latent_frame]
    tokens, _ = causal.rollout(
        common.denoised_from_x0_model(model),
        grid,
        geometry,
        cache,
        source,
        context,
        sigma,
        seed=seed,
        blocks=blocks,
        teacher_forcing=False,
        teacher_tokens=None,
        schedule=None,
        kv_source="refresh",
        first_frame_condition=c0,
        block_epsilons=epsilons,
        history_mode="cache",
    )
    covered = blocks[-1][1]
    return grid.unpatchify_block(tokens[:, : covered * grid.tokens_per_latent_frame], covered)


@pytest.mark.parametrize("frames", [17, 21])
def test_matches_original_two_rollouts_and_actual_call_counts(frames: int, monkeypatch: pytest.MonkeyPatch) -> None:

    model = X0Model(_model().to(dtype=DTYPE))
    capture = torch.randn(CHANNELS, frames, 2, 2, generator=torch.Generator().manual_seed(8))
    guide = torch.randn(CHANNELS, frames, 2, 2, generator=torch.Generator().manual_seed(9))
    session = SimpleNamespace(
        device=torch.device("cpu"),
        context=_context().to(dtype=DTYPE),
        model=SimpleNamespace(scale_factors=SCALE, caps=SimpleNamespace(latent_channels=CHANNELS)),
    )
    geometry = causal.deployed_geometry(SCALE)
    grid = common.ClipGrid.build(
        frames, 64, 64, 30, geometry, device=session.device, dtype=DTYPE, latent_channels=CHANNELS
    )
    plan = geometry.plan(frames)[:8]
    target = grid.patchify(capture.unsqueeze(0).to(dtype=DTYPE))
    eps_a = _global_epsilons(target, grid, plan, 42)
    eps_b = _global_epsilons(target, grid, plan, 99)
    expected = []
    with torch.no_grad():
        for epsilon in (eps_a, eps_a[:4] + eps_b[4:]):
            output = _original_rollout(
                model, session.context, grid, geometry, capture, guide, 0.421875, seed=42, epsilons=epsilon, blocks=plan
            )
            expected.append(output.float())
    original_sample = causal.sample
    actual = []

    def sample(*args: object, **kwargs: object) -> tuple[torch.Tensor, dict[str, int]]:
        assert not torch.is_grad_enabled()
        tokens, counts = original_sample(*args, **kwargs)
        actual.append(grid.unpatchify_block(tokens[:, : 17 * grid.tokens_per_latent_frame], 17).float())
        return tokens, counts

    monkeypatch.setattr(causal, "sample", sample)
    result = evaluate.causality_probe(model, session, capture, guide, 30, 0.421875)
    assert len(actual) == 2
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert result["latent_frames_compared_equal"] == [0, 9]
    assert result["earlier_blocks_bit_identical"]
    assert result["later_blocks_changed"]
    assert result["later_blocks_max_abs_diff"] > 0
    assert all(record["call_counts"]["model_calls"] == 16 for record in result["records"])
    assert all(record["call_counts"]["denoise_calls"] == 8 for record in result["records"])
    assert all(record["call_counts"]["refresh_calls"] == 8 for record in result["records"])
    assert result["records"][0]["noise_sha256"] != result["records"][1]["noise_sha256"]


def test_checked_orchestration_and_atomic_saved_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    _dev, adapter, view, capture, guide = fixture_inputs(tmp_path, monkeypatch)
    events = []

    def check(path: Path, conditions: dict) -> dict:
        assert path == adapter
        assert conditions["schedule"] == [0.421875, 0.0]
        assert conditions["mode_settings"]["blocks_per_sample"] == 8
        assert conditions["task"]["objective"] == "white"
        assert conditions["task"]["guide_mode"] == "d1"
        assert conditions["history_mode"] == "cache"
        assert conditions["kv_source"] == "refresh"
        assert conditions["global_sigma_dtype"] == "float32"
        assert conditions["application_method"] == "peft_unmerged_fp32"
        events.append("check")
        return {"contract": {"fixture": "adapter"}, "adapter_sha256": evaluate.sha256(adapter)}

    def open_session(*_args: object, **_kwargs: object) -> SimpleNamespace:
        assert events == ["check"]
        events.append("open")
        # Keep the session's keyword spelling while replacing only model work.
        return SimpleNamespace(transformer=lambda: nullcontext(object()))

    def probe(
        _transformer: object, _session: object, target: torch.Tensor, source: torch.Tensor, fps: float, sigma: float
    ) -> dict[str, object]:
        assert target is capture
        assert source is guide
        assert fps == 30
        assert sigma == 0.421875
        return {
            "shared_noise_blocks": [0, 1, 2, 3],
            "changed_noise_blocks": [4, 5, 6, 7],
            "latent_frames_compared_equal": [0, 9],
            "earlier_blocks_bit_identical": True,
            "later_blocks_max_abs_diff": 0.5,
        }

    monkeypatch.setattr(evaluate.evaluate, "check_adapter", check)
    monkeypatch.setattr(sessions, "open_session", open_session)
    monkeypatch.setattr(evaluate, "causality_probe", probe)
    output = tmp_path / "diagnostic.json"
    result = evaluate.evaluate_causality(adapter, view, output, gpu_id=0, sigma=0.421875)
    assert result["checkpoint"] == str(adapter)
    assert result["view"] == str(view)
    assert len(result["input_sha256"]) == 4
    assert result["earlier_blocks_bit_identical"]
    assert json.loads(output.read_text()) == result


def test_short_master_fails_before_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    _, adapter, view, _, _ = fixture_inputs(tmp_path, monkeypatch, frames=15)
    monkeypatch.setattr(sessions, "open_session", lambda *_args, **_kwargs: pytest.fail("opened weights"))
    with pytest.raises(ValueError, match="eight complete blocks"):
        evaluate.evaluate_causality(adapter, view, tmp_path / "result.json", gpu_id=0, sigma=0.421875)


def test_existing_result_preserved(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(b"original evidence")
    with pytest.raises(ValueError, match="fresh output"):
        evaluate.evaluate_causality(tmp_path, tmp_path, output, gpu_id=0, sigma=0.421875)
    assert output.read_bytes() == b"original evidence"
