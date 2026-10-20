"""Check temporal crop alignment, conditioning, saved identities and replay."""

import json
import random
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.model import common
from scripts.prune.core.provenance import file_sha256
from scripts.prune.data import training_windows as windows
from scripts.prune.learn import heads, losses


def case(frames, seed=42):
    model = SimpleNamespace(scale_factors=SpatioTemporalScaleFactors(8, 32, 32))
    current = SimpleNamespace(model=model)
    grid = common.ClipGrid.build(frames, 64, 64, 30, model, device="cpu", dtype=torch.bfloat16, latent_channels=2)
    dense = (
        torch.arange(frames, dtype=torch.bfloat16)[None, None, :, None, None].expand(1, 2, frames, 2, 2).contiguous()
    )
    capture = grid.patchify(dense)
    epsilon = common.epsilon_block(capture, seed)
    c0 = capture[:, :4]
    noisy = common.with_clean_prefix(common.mix_block_noise(capture, epsilon, 0.9), c0)
    video = common.block_modality(
        grid,
        noisy,
        torch.zeros(1, 2, 2),
        0.9,
        token_slices=[grid.token_span(0, frames)],
        attention_mask=None,
        clean_prefix_tokens=4,
        sigma_dtype=torch.float32,
    )
    return (grid, video, c0, capture, epsilon, "input", {"seed": seed}), current, dense


@pytest.mark.parametrize("frames", [18, 28, 36])
def test_contiguous_window_and_local_clean_first_frame(frames):
    seed = next(s for s in range(100) if frames == 18 or windows.bounds(frames, s)[0] > 0)
    full, current, _ = case(frames, seed)
    cropped = windows.crop_case(full, current, seed, 0.9)
    grid, video, c0, capture, epsilon, _, manifest = cropped
    start, stop = windows.bounds(frames, seed)
    assert grid.latent_frames == 18 and capture.shape == (1, 72, 2)
    assert torch.equal(capture, full[3][:, start * 4 : stop * 4])
    assert torch.equal(epsilon, full[4][:, start * 4 : stop * 4])
    assert torch.equal(c0, capture[:, :4]) and torch.equal(video.latent[:, :4], c0)
    assert torch.equal(video.latent[:, 4:], common.mix_block_noise(capture, epsilon, 0.9)[:, 4:])
    assert (video.timesteps[:, :4] == 0).all() and (video.timesteps[:, 4:] == 0.9).all()
    assert video.sigma.dtype == torch.float32 and video.attention_mask is None and video.kv_caches is None
    assert torch.equal(video.positions, case(18, seed)[0][1].positions)
    assert video.keyframes_mask[:, :4].any() and not video.keyframes_mask[:, 4:].any()
    assert manifest["training_window"]["latent_start"] == start
    assert "training_window" not in full[6]
    velocity = losses.velocity_target(epsilon, capture)
    velocity[:, :4] += 100
    assert losses.velocity_mse(velocity, epsilon, capture, 4)["official"] == 0


def test_f18_preserves_native_input_bit_for_bit():
    full, current, _ = case(18)
    cropped = windows.crop_case(full, current, 42, 0.9)
    for field in fields(full[1]):
        before, after = getattr(full[1], field.name), getattr(cropped[1], field.name)
        assert torch.equal(before, after) if isinstance(before, torch.Tensor) else before == after
    assert torch.equal(full[3], cropped[3]) and torch.equal(full[4], cropped[4])


def test_bounds_cover_all_starts_and_do_not_consume_global_rng():
    before = random.getstate(), torch.get_rng_state().clone()
    draws = [windows.bounds(28, seed) for seed in range(300)]
    assert {start for start, _ in draws} == set(range(11))
    assert draws == [windows.bounds(28, seed) for seed in range(300)]
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[1])
    assert windows.bounds(18, 1) == (0, 18)
    for frames, seed in ((17, 1), (28, True), (28.0, 1)):
        with pytest.raises(ValueError, match="at least 18"):
            windows.bounds(frames, seed)


def test_option_is_opt_in_and_region_weights_follow_the_window(monkeypatch):
    full, current, _ = case(28)
    monkeypatch.setattr(heads, "input_case", lambda *args: full)
    assert heads.training_input_case({"config": {}}, current, None, 42, 0.9) is full
    cropped = heads.training_input_case({"config": {"uniform_f18": True}}, current, None, 42, 0.9)
    record = cropped[-1]["training_window"]
    weights = {"person": torch.arange(112)[None], "face": torch.ones(1, 112)}
    actual = windows.slice_regions(weights, record)
    lo, hi = record["latent_start"] * 4, record["latent_stop"] * 4
    assert torch.equal(actual["person"], weights["person"][:, lo:hi])
    assert actual["face"].shape == (1, 72)
    with pytest.raises(ValueError, match="full source"):
        windows.slice_regions({"person": torch.ones(1, 72)}, record)
    with pytest.raises(ValueError, match="boolean"):
        windows.enabled({"uniform_f18": "yes"})


def saved_case(tmp_path):
    seed = 42
    full, current, dense = case(28, seed)
    capture_path = tmp_path / "capture.pt"
    torch.save({"schema_version": 2, "master": dense[0], "fps": 30}, capture_path)
    torch.save(full[4], tmp_path / "epsilon.pt")
    row = {
        "view": "case",
        "sigma": 0.9,
        "schedule": [0.9, 0.0],
        "artifacts": {
            "capture": str(capture_path),
            "capture_sha256": file_sha256(capture_path),
            "fps": 30,
            "epsilon": "epsilon.pt",
            "epsilon_sha256": file_sha256(tmp_path / "epsilon.pt"),
        },
    }
    manifest = {
        "whole_clip": True,
        "attention": "full_bidirectional",
        "objective": "white",
        "seed": seed,
        "latent_dtype": "torch.bfloat16",
        "sigmas": [0.9],
        "videos": [row],
        "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return windows.crop_case(full, current, seed, 0.9)[-1]["training_window"]


def test_resume_and_export_validate_actual_source_slices(tmp_path):
    record = saved_case(tmp_path)
    windows.validate_saved_window(tmp_path, record)
    for changed in (
        {**record, "latent_start": 0},
        {**record, "capture_tensor_sha256": "wrong"},
        {**record, "epsilon_tensor_sha256": "wrong"},
        {**record, "noise_seed": 43},
    ):
        with pytest.raises(ValueError, match="window changed"):
            windows.validate_saved_window(tmp_path, changed)
    torch.save({"master": torch.zeros(2, 28, 2, 2), "fps": 30}, tmp_path / "capture.pt")
    with pytest.raises(ValueError, match="source changed"):
        windows.validate_saved_window(tmp_path, record)
