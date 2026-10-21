"""Matched training tensors, unchanged gradients, real media and RNG isolation."""

import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ltx_core.model.transformer.model import X0Model
from ltx_core.tiling import TileSizeConfig
from scripts.prune.core import records
from scripts.prune.learn import adapters, checkpoints, joint_training, visualization
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.step import TrainingStep
from scripts.prune.score import hooks
from scripts.prune.tests.test_ddp_training import cfg
from scripts.prune.tests.test_ffn_learning import Velocity, case, config, make_step
from scripts.prune.tests.test_training_windows import case as window_case


def test_default_cadence_disable_override_and_primary_rank():
    step = torch.nn.Linear(1, 1)
    assert [n for n in range(301) if visualization.arm(step, {}, n)] == [100, 200, 300]
    assert not visualization.arm(step, {"visualization_interval": 0}, 100)
    assert visualization.arm(step, {"visualization_interval": 3}, 6)
    assert visualization.arm(SimpleNamespace(module=step), {}, 100, primary=False)
    visualization.capture_prediction(step, torch.ones(1))
    assert step.visualization_prediction is None
    for value in (-1, 1.5, True, "100"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            visualization.arm(step, {"visualization_interval": value}, 100)


def fixture_step(family):
    torch.manual_seed(42)
    if family == "ffn":
        step = make_step()
        step.configure(1)
    elif family == "joint":
        settings = config()
        settings.update(
            arm="joint_clip_reset_v3",
            rho=1 / 704,
            policy=dict(interval=2, late=1, earliest=4, limit=6, recovery=2, alignment=4),
        )
        step = joint_training.JointStep(X0Model(Velocity()), settings, blocks=1, heads=4)
        step.configure(1)
    else:
        model = X0Model(Velocity()).requires_grad_(False)
        adapters.attach(model, 42, 2)
        gates = HeadGates(1, 4, [[2, 4]])
        step = TrainingStep(model, gates, dict(cfg(), person_coefficient=0), "learned_joint")
        step.live = hooks.attach_head_provider(model, gates.provider)
    return step


@pytest.mark.parametrize("family", ["heads", "ffn", "joint"])
def test_preview_captures_actual_forward_without_changing_gradients(family):
    results = []
    for enabled in (False, True):
        step = fixture_step(family)
        visualization.arm(step, {"visualization_interval": int(enabled)}, 1)
        grid, video, _, capture, epsilon, _, _ = case()
        args = (video, epsilon, capture, grid.tokens_per_latent_frame, 1)
        output = step(*args, "soft", accumulation=1) if family == "heads" else step(*args)
        output[0].backward()
        results.append((output, {name: p.grad.clone() for name, p in step.named_parameters() if p.grad is not None}))
        prediction = step.visualization_prediction
        if enabled:
            assert prediction.shape == capture.shape and prediction.device.type == "cpu"
            assert not prediction.requires_grad and prediction.grad_fn is None
            torch.testing.assert_close(prediction[:, :1], capture[:, :1], rtol=0, atol=0)
            visualization.capture_prediction(step, prediction + 100)
            assert step.visualization_prediction is prediction  # The first draw owns the preview.
        else:
            assert prediction is None
        step.live.detach_all() if family == "heads" else step.close()
    for a, b in zip(results[0][0], results[1][0], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=0, atol=0)


def preview_case(tmp_path):
    value, current, _ = window_case(3)
    records.publish_json(tmp_path / "manifest.json", {})
    value = (*value[:-2], tmp_path, {"training_window": {"latent_start": 4}})
    step = torch.nn.Linear(1, 1)
    visualization.arm(step, {}, 100)
    prediction = value[3].clone() + 0.25
    visualization.capture_prediction(step, prediction.requires_grad_())
    preview = visualization.snapshot(step, value, update=100, sigma=0.9, view="fixture")
    assert step.visualization_prediction is None
    assert visualization.snapshot(step, value) is None
    torch.testing.assert_close(preview["latents"]["noisy"], value[0].unpatchify_block(value[1].latent, 3))
    torch.testing.assert_close(preview["latents"]["gt"], value[0].unpatchify_block(value[3], 3))
    assert torch.equal(preview["latents"]["predicted_x0"][:, :, :1], preview["latents"]["gt"][:, :, :1])
    assert not torch.equal(preview["latents"]["predicted_x0"][:, :, 1:], preview["latents"]["gt"][:, :, 1:])
    current.device = torch.device("cpu")
    return preview, current


def test_real_videos_and_labeled_synchronized_preview(tmp_path, monkeypatch):
    preview, current = preview_case(tmp_path)
    current.model.paths = SimpleNamespace(video_vae=lambda: "fixture.safetensors")
    decoded = []
    tiling = TileSizeConfig.default()

    def select_tiling(path, **kwargs):
        assert path == "fixture.safetensors"
        assert kwargs == dict(height=64, width=64, num_frames=17, device=current.device)
        return tiling

    monkeypatch.setattr(visualization, "tiling_config_for_vae", select_tiling)

    class Decoder:
        def decode_video(self, latent, tiling, generator):
            assert torch.is_inference_mode_enabled()
            assert tiling.height.tile_size == 768
            decoded.append(latent.clone())
            torch.rand(1)  # Loading or decoding may consume the global RNG.
            pixels = torch.nn.functional.interpolate(latent[:, :1].float(), size=(17, 64, 64), mode="trilinear")
            pixels = pixels[0, 0, :, :, :, None].expand(-1, -1, -1, 3).sigmoid()
            yield pixels[:8]
            yield pixels[8:]

    @contextmanager
    def decoder():
        yield Decoder()

    current.decoder = decoder
    destination = tmp_path / "visualizations/update_0100"
    before = records.unchanged_state_hash(checkpoints.rng_state([]))
    visualization.save(current, preview, destination)
    assert records.unchanged_state_hash(checkpoints.rng_state([])) == before
    assert len(decoded) == 3
    for role in ("noisy", "gt", "predicted_x0"):
        assert (destination / f"{role}.mp4").stat().st_size > 0
        torch.testing.assert_close(torch.load(destination / f"{role}.pt", weights_only=True), preview["latents"][role])
    rendering = records.read_json(destination / "rendering.json")
    assert [p["title"] for p in rendering["panels"]] == ["Noisy latent", "GT latent", "Predicted x0"]
    assert all(p["source_frames"] == list(range(17)) and p["row"] == 0 for p in rendering["panels"])
    assert rendering["common_settings"]["training_window"] == {"latent_start": 4}
    assert rendering["common_settings"]["prediction_weights"] == "training forward before the optimizer update"
    from scripts.onestep_avatar import media

    media.verify_saved_video(destination / "comparison.mp4", 17, 30)
    media.verify_saved_png(destination / "poster.png", *rendering["display_size"])
    # A replay retains the same latent evidence and does not consume RNG.
    visualization.save(current, preview, destination)
    assert len(decoded) == 3
    assert rendering["font_size"] * 480 / rendering["display_size"][0] >= 16
    preview["latents"]["noisy"].add_(1)
    with pytest.raises(ValueError, match="different latent tensors"):
        visualization.save(current, preview, destination)


@pytest.mark.parametrize("fail", [False, True])
def test_rng_and_rank_failure_vote(tmp_path, monkeypatch, fail):
    def write(*args):
        random.random()
        np.random.rand()
        torch.rand(3)
        if fail:
            raise OSError("decoder failure")

    votes = []
    runtime = SimpleNamespace(any=lambda value: votes.append(value) or value)
    monkeypatch.setattr(visualization, "_write", write)
    before = records.unchanged_state_hash(checkpoints.rng_state([]))
    current = SimpleNamespace(device=torch.device("cpu"))
    if fail:
        with pytest.raises(OSError, match="decoder failure"):
            visualization.save(current, {}, tmp_path, runtime=runtime)
    else:
        visualization.save(current, {}, tmp_path, runtime=runtime)
    assert records.unchanged_state_hash(checkpoints.rng_state([])) == before
    assert votes == [fail]
    runtime.any = lambda value: True
    with pytest.raises(RuntimeError, match="rank zero"):
        visualization.save(current, None, Path(tmp_path), runtime=runtime)
