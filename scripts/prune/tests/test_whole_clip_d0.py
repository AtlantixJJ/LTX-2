"""Small exact checks for the whole-clip one-step direction comparison."""

from __future__ import annotations

import pytest
import torch

from scripts.prune.data import whole_clip
from scripts.prune.evaluate.whole_clip_d0 import _direction_metrics, _verify_pair


def test_direction_uses_noisy_input_and_excludes_clean_keyframe() -> None:
    capture = torch.zeros(1, 1, 2, 1, 2)
    epsilon = torch.ones(1, 4, 1)
    base = torch.zeros_like(capture)
    candidate = base.clone()
    candidate[:, :, 0] = 100  # the conditioned frame must not affect any metric
    candidate[:, :, 1] = 0.25
    result = _direction_metrics(base, candidate, capture, epsilon, 0.5)
    # On frame 1, x_sigma=.5, so baseline direction=1 and candidate direction=.5.
    assert result["direction_relative_l2"] == pytest.approx(0.5)
    assert result["direction_cosine"] == pytest.approx(1.0)
    assert result["baseline_capture_mse"] == 0
    assert result["candidate_capture_mse"] == pytest.approx(0.0625)


def test_rejects_non_bidirectional_or_unmatched_noise_setup() -> None:
    manifest = {
        "whole_clip": True, "objective": "white", "sigmas": [0.5], "seed": 42,
        "trajectory_only": False, "geometry": {"block_latent_frames": 1},
        "attention": "full_bidirectional", "latent_dtype": "torch.bfloat16",
        "text_context": {"sha256": "x"},
        "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1},
        "model": {"model_key": "2.5", "video_vae_fingerprint": "vae",
                  "transformer_fingerprint": "a"}, "videos": [],
    }
    candidate = {**manifest, "model": {**manifest["model"], "transformer_fingerprint": "b"}}
    _verify_pair(manifest, candidate)
    with pytest.raises(ValueError, match="unmatched manifest field: seed"):
        _verify_pair(manifest, {**candidate, "seed": 43})
    with pytest.raises(ValueError, match="whole-clip"):
        _verify_pair(manifest, {**candidate, "whole_clip": False})
    with pytest.raises(ValueError, match="video_vae_fingerprint"):
        _verify_pair(manifest, {**candidate, "model": {**candidate["model"], "video_vae_fingerprint": "other"}})


def test_saved_d0_manifest_rejects_wrong_schedule_and_duplicate_rows() -> None:
    row = {"view": "capture/view00", "sigma": 0.5, "schedule": [0.5, 0.0]}
    manifest = {"whole_clip": True, "trajectory_only": False, "attention": "full_bidirectional",
                "objective": "white", "sigmas": [0.5], "videos": [row], "seed": 42,
                "latent_dtype": "torch.bfloat16",
                "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1}}
    whole_clip.validate_manifest(manifest)
    with pytest.raises(ValueError, match="one-step"):
        whole_clip.validate_manifest({**manifest, "videos": [{**row, "schedule": [0.5, 0.25, 0.0]}]})
    with pytest.raises(ValueError, match="duplicate"):
        whole_clip.validate_manifest({**manifest, "videos": [row, row]})


@pytest.mark.parametrize("change", ["cfg", "stg", "rescale", "passes", "lora", "dtype", "sigma"])
def test_unsupported_reconstruction_rejected(change) -> None:
    manifest = {"whole_clip": True, "trajectory_only": False, "attention": "full_bidirectional",
                "objective": "white", "sigmas": [0.5], "videos": [], "seed": 42,
                "latent_dtype": "torch.bfloat16",
                "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1}}
    if change in ("cfg", "stg", "rescale"):
        manifest["guidance"][change] = 3
    elif change == "passes":
        manifest["guidance"]["passes_per_step"] = 2
    elif change == "lora":
        manifest["checkpoint"] = "lora.safetensors"
    elif change == "dtype":
        manifest["latent_dtype"] = "torch.float32"
    else:
        manifest["sigmas"] = [-0.5]
    with pytest.raises(ValueError):
        whole_clip.validate_manifest(manifest)


def test_saved_noise_content_mutation_rejected(tmp_path) -> None:
    from scripts.prune.core import provenance

    path = tmp_path / "epsilon.pt"
    torch.save(torch.ones(1, 4, 2, dtype=torch.bfloat16), path)
    row = {"artifacts": {"epsilon": path.name, "epsilon_sha256": provenance.file_sha256(path)}}
    assert whole_clip.load_epsilon(tmp_path, row).shape == (1, 4, 2)
    torch.save(torch.zeros(1, 4, 2, dtype=torch.bfloat16), path)
    with pytest.raises(ValueError, match="epsilon content changed"):
        whole_clip.load_epsilon(tmp_path, row)
