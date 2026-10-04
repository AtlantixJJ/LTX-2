"""Synthetic reconstruction math and real native-checkpoint cache provenance, without a GPU."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.prune.core import provenance
from scripts.prune.data import whole_clip
from scripts.prune.score import ffn_reconstruction as ridge
from scripts.prune.score import token_sampling
from scripts.prune.tests.test_export_depth import _config, _model


def _matrices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(10)
    features = torch.randn(6, 9, generator=generator, dtype=torch.float64)
    targets = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    source = torch.randn(3, 9, generator=generator, dtype=torch.float64)
    return features, targets, source


def test_dual_correction_agrees_with_independent_primal_ridge_and_chunks() -> None:
    features, target, source = _matrices()
    strength = 0.03
    lhs = features.T @ features / len(features) + strength * torch.eye(features.shape[1], dtype=torch.float64)
    rhs = features.T @ (target - features @ source.T) / len(features)
    expected = source + torch.linalg.solve(lhs, rhs).T
    measured = ridge.fit_output_projection(features, target, source, ridge_lambda=strength, output_chunk=1)
    torch.testing.assert_close(measured["weight"], expected, atol=1e-12, rtol=1e-12)
    large_chunk = ridge.fit_output_projection(features, target, source, ridge_lambda=strength, output_chunk=20)
    torch.testing.assert_close(measured["weight"], large_chunk["weight"], atol=1e-12, rtol=1e-12)
    assert measured["diagnostics"]["solve_dtype"] == "torch.float64"
    assert not measured["weight"].requires_grad


def test_correlated_deleted_features_allow_local_synthetic_reconstruction() -> None:
    generator = torch.Generator().manual_seed(12)
    retained = torch.randn(40, 3, generator=generator, dtype=torch.float64)
    removed = (retained[:, 0] - 2 * retained[:, 2])[:, None]
    full = torch.cat((retained, removed), dim=1)
    weight = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    target = full @ weight.T
    result = ridge.fit_output_projection(retained, target, weight[:, :3], ridge_lambda=1e-8)
    assert result["diagnostics"]["local_mse_after"] < result["diagnostics"]["local_mse_before"] * 1e-10
    assert result["diagnostics"]["qualification"] == "local_calibration_diagnostic_only"


def test_rank_deficient_and_zero_features_remain_finite() -> None:
    features, target, source = _matrices()
    features[:, 1] = features[:, 0]
    result = ridge.fit_output_projection(features, target, source, ridge_lambda=0.1)
    assert torch.isfinite(result["weight"]).all()
    zero = ridge.fit_output_projection(torch.zeros_like(features), target, source, ridge_lambda=0.1)
    assert torch.equal(zero["weight"], source)
    assert zero["diagnostics"]["local_mse_after"] == zero["diagnostics"]["local_mse_before"]


def test_no_removal_exact_source_target_is_an_identity_control() -> None:
    features, _, source = _matrices()
    target = features @ source.T
    result = ridge.fit_output_projection(features, target, source, ridge_lambda=0.1)
    torch.testing.assert_close(result["weight"], source, atol=1e-15, rtol=1e-15)
    assert result["diagnostics"]["local_mse_before"] < 1e-27


@pytest.mark.parametrize("failure", ["nan", "shape", "ridge_zero", "ridge_nan", "samples", "memory", "chunk"])
def test_invalid_fit_inputs_fail_before_solve(failure: str) -> None:
    features, target, source = _matrices()
    options = {"ridge_lambda": 0.01}
    if failure == "nan":
        features[0, 0] = float("nan")
    elif failure == "shape":
        target = target[:-1]
    elif failure == "ridge_zero":
        options["ridge_lambda"] = 0
    elif failure == "ridge_nan":
        options["ridge_lambda"] = float("nan")
    elif failure == "samples":
        options["max_samples"] = 5
    elif failure == "memory":
        options["max_memory_bytes"] = 1
    else:
        options["output_chunk"] = 0
    with pytest.raises(ValueError, match=r"finite|dimensions|ridge_lambda|cap"):
        ridge.fit_output_projection(features, target, source, **options)


def _cache(tmp_path: Path) -> tuple[Path, dict]:
    config = _config()
    config["transformer"]["per_layer_ff_inner_dim"] = [256] * 4
    source = tmp_path / "source.safetensors"
    model = _model(config)
    save_file({"model.diffusion_model." + key: value for key, value in model.state_dict().items()}, str(source),
              metadata={"config": json.dumps(config)})
    capture = tmp_path / "capture.pt"
    torch.save({"schema_version": 2, "master": torch.zeros(8, 7, 2, 2, dtype=torch.bfloat16), "fps": 30}, capture)
    noise = tmp_path / "epsilon.pt"
    torch.save(torch.zeros(1, 28, 8, dtype=torch.bfloat16), noise)
    manifest = {
        "whole_clip": True, "trajectory_only": False, "attention": "full_bidirectional", "objective": "white",
        "text_context": {}, "geometry": {}, "seed": 42, "latent_dtype": "torch.bfloat16", "sigmas": [0.725, 1.0],
        "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1},
        "model": {"model_key": "2.5", "transformer_path": str(source),
                  "transformer_fingerprint": provenance.checkpoint_fingerprint(source), "video_vae_fingerprint": "vae"},
        "videos": [{"view": "calibration/views/view00", "sigma": sigma, "schedule": [sigma, 0],
                    "artifacts": {"capture": str(capture), "capture_sha256": provenance.file_sha256(capture),
                                  "epsilon": noise.name, "epsilon_sha256": provenance.file_sha256(noise),
                                  "fps": 30, "blocks": [[0, 7]]}} for sigma in (0.725, 1.0)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    stamp = whole_clip.native_provenance(tmp_path, manifest, ["calibration/views/view00"], [0.725, 1.0])
    mask = tmp_path / "mask.json"
    mask.write_text(json.dumps({"candidate_format": "whole_clip_d0_mask_v1", "provenance": stamp,
                               "masks": {f"{layer}.ff": [1] * 128 + [0] * 128 for layer in range(4)}}))
    payload = tmp_path / "samples.pt"
    torch.save({"retained_features": torch.zeros(8, 128), "teacher_output": torch.zeros(8, 8)}, payload)
    indices = token_sampling.sample_indices(28, 2, 2, 2, torch.device("cpu"), sampler="balanced_2d_midpoint_v1")
    sampling = token_sampling.sampling_record(indices, tokens=28, height=2, width=2, stride=2,
                                              sampler="balanced_2d_midpoint_v1")
    selected = ridge.midpoint_subsample(indices, 4)
    cache = {
        "cache_format": ridge.CACHE_FORMAT, "target": ridge.TARGET, "selection": ridge.SELECTION,
        "provenance": stamp, "source_checkpoint": str(source), "mask_path": str(mask),
        "mask_sha256": provenance.file_sha256(mask), "branch": "0.ff", "retained_indices": list(range(128)),
        "payload": str(payload), "payload_sha256": provenance.file_sha256(payload),
        "cases": [{"view": "calibration/views/view00", "sigma": sigma, "sampling": sampling,
                   "token_indices": selected.tolist(), "token_indices_sha256": token_sampling.index_sha256(selected),
                   "row_slice": [order * 4, (order + 1) * 4]} for order, sigma in enumerate((0.725, 1.0))],
    }
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(cache))
    return path, manifest


def test_calibration_cache_pins_real_checkpoint_geometry_mask_noise_and_sampler(tmp_path: Path) -> None:
    path, manifest = _cache(tmp_path)
    tensors, report = ridge.load_calibration_cache(path, baseline=manifest)
    assert tensors["retained_features"].shape == (8, 128)
    assert report["dimensions"]["samples"] == 8
    assert all(case["token_indices"][0] >= 4 for case in report["cache"]["cases"])


@pytest.mark.parametrize("change", ["heldout", "duplicate", "c0", "geometry", "sampler_hash", "keep",
                                   "unequal_quota", "mask_hash", "target", "scope"])
def test_mutated_cache_manifest_is_rejected(tmp_path: Path, change: str) -> None:
    path, _ = _cache(tmp_path)
    cache = json.loads(path.read_text())
    first = cache["cases"][0]
    if change == "heldout":
        first["view"] = "heldout/views/view00"
    elif change == "duplicate":
        cache["cases"][1] = copy.deepcopy(first)
    elif change == "c0":
        first["token_indices"][0] = 0
    elif change == "geometry":
        first["sampling"]["latent_height"], first["sampling"]["latent_width"] = 1, 4
    elif change == "sampler_hash":
        first["sampling"]["token_indices_sha256"] = "changed"
    elif change == "keep":
        cache["retained_indices"] = list(reversed(cache["retained_indices"]))
    elif change == "unequal_quota":
        full = token_sampling.sample_indices(28, 2, 2, 2, torch.device("cpu"), sampler="balanced_2d_midpoint_v1")
        subset = ridge.midpoint_subsample(full, 3)
        first.update(token_indices=subset.tolist(), token_indices_sha256=token_sampling.index_sha256(subset),
                     row_slice=[0, 3])
        cache["cases"][1]["row_slice"] = [3, 7]
    elif change == "mask_hash":
        cache["mask_sha256"] = "changed"
    elif change == "target":
        cache["target"] = "native_bf16_output_including_bias"
    else:
        cache["provenance"]["calibration_views"] = ["different/views/view00"]
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match=r"cache|calibration|native|width-mask"):
        ridge.load_calibration_cache(path)


@pytest.mark.parametrize("change", ["epsilon_missing", "epsilon_mutated", "capture", "payload", "manifest",
                                   "source"])
def test_mutated_calibration_artifacts_are_rejected(tmp_path: Path, change: str) -> None:
    path, _ = _cache(tmp_path)
    if change == "epsilon_missing":
        (tmp_path / "epsilon.pt").unlink()
    elif change == "epsilon_mutated":
        torch.save(torch.ones(1, 28, 8, dtype=torch.bfloat16), tmp_path / "epsilon.pt")
    elif change == "capture":
        (tmp_path / "capture.pt").touch()
        with (tmp_path / "capture.pt").open("ab") as stream:
            stream.write(b"changed")
    elif change == "payload":
        with (tmp_path / "samples.pt").open("ab") as stream:
            stream.write(b"changed")
    elif change == "manifest":
        with (tmp_path / "manifest.json").open("a") as stream:
            stream.write(" ")
    else:
        with (tmp_path / "source.safetensors").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match=r"changed|content|checkpoint"):
        ridge.load_calibration_cache(path)


def test_cache_sample_memory_and_alignment_caps_are_enforced(tmp_path: Path) -> None:
    path, _ = _cache(tmp_path)
    with pytest.raises(ValueError, match="sample cap"):
        ridge.load_calibration_cache(path, max_samples=7)
    with pytest.raises(ValueError, match="byte cap"):
        ridge.load_calibration_cache(path, max_payload_bytes=1)
    with pytest.raises(ValueError, match="aligned width"):
        ridge.load_calibration_cache(path, retained_alignment=256)


@pytest.mark.parametrize("failure", ["nan", "shape", "dtype"])
def test_hash_matching_payload_still_requires_valid_finite_fp32_shapes(tmp_path: Path, failure: str) -> None:
    path, _ = _cache(tmp_path)
    cache = json.loads(path.read_text())
    tensors = torch.load(cache["payload"], weights_only=True)
    if failure == "nan":
        tensors["retained_features"][0, 0] = float("nan")
    elif failure == "shape":
        tensors["retained_features"] = tensors["retained_features"][:-1]
    else:
        tensors["retained_features"] = tensors["retained_features"].to(torch.bfloat16)
    torch.save(tensors, cache["payload"])
    cache["payload_sha256"] = provenance.file_sha256(cache["payload"])
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="finite FP32"):
        ridge.load_calibration_cache(path)
