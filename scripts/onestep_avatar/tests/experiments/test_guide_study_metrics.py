"""Exact formulas, hash refusal and canonical planner checks."""

from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import hashing
from scripts.onestep_avatar.experiments import guide_study_metrics as metrics
from scripts.onestep_avatar.tests.support.model import _geometry


def tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1, 1)


def test_metrics_and_controls() -> None:
    output, capture, guide = tensor([0, 2, 4]), tensor([0, 1, 2]), tensor([0, 3, 5])
    result = metrics.measures(output, capture, guide, [(0, 2), (2, 3)])
    assert result["capture_mse"] == 2.5
    assert result["guide_mse"] == 1
    assert result["guide_capture_mse"] == 6.5
    assert result["per_frame_capture_mse"] == [0, 1, 4]
    assert result["boundary_temporal_residual_mae"] == 1
    assert metrics.equal_control(output, output.clone())
    assert not metrics.equal_control(output, capture)
    assert metrics.equal_control(output, capture, (0, 1))


def test_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "output.pt"
    value = tensor([0, 1, 2])
    torch.save(value, path)
    record = {"output": {"path": str(path), "sha256": hashing.sha256(path), "shape": list(value.shape)}}
    assert torch.equal(metrics.read_output(record), value)
    torch.save(value + 1, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        metrics.read_output(record)


def test_ranges_from_public_planner() -> None:
    record = {
        "frames": 7,
        "mode_settings": {"block_latent_frames": 2, "context_latent_frames": 8},
        "schedule": [0.725, 0],
        "call_counts": {"denoise_calls": 3, "refresh_calls": 3, "prime_calls": 0},
    }
    ranges = metrics.reconstructed_ranges(record, _geometry().scale_factors)
    assert ranges == [(0, 3), (3, 5), (5, 7)]
    record["call_counts"]["denoise_calls"] = 4
    with pytest.raises(ValueError, match="disagree"):
        metrics.reconstructed_ranges(record, _geometry().scale_factors)


def test_control_pairs_allow_declared_prompt_and_arm_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = []
    for arm, prompt, delta in (("d0", "generic", 0), ("d1", "generic", 0), ("d1", "different", 1)):
        case = {
            "id": arm + prompt,
            "actor": "0008_01",
            "variant": "dev",
            "schedule": [1, 0],
            "prompt": prompt,
            "history": None,
            "sigma": 1,
            "arm": arm,
            "roles": [{"prompt": "P0" if prompt == "generic" else "P4"}],
        }
        record = {
            "schedule": [1, 0],
            "call_counts": {"model_calls": 3},
            "noise_sha256": "n",
            "capture_sha256": "c",
            "c0_sha256": "c0",
            "source": "s",
            "guide_sha256": None if arm == "d0" else "g",
            "input_file_hashes": {"capture": "c", **({"guide": "g"} if arm == "d1" else {})},
            "text_sha256": prompt,
            "conditions": {"model": {"base_sha256": "w"}},
        }
        rows.append((case, record, tensor([0, 1 + delta, 2]), [], {"path": case["id"], "sha256": "hash"}))
    monkeypatch.setattr(metrics, "load_results", lambda *_: ({}, rows, []))
    result = metrics.controls(tmp_path, False)
    assert result["passed"]
    assert result["complete"]
    assert len(result["checks"]) == 1
    rows[1] = (*rows[1][:2], tensor([0, 2, 2]), *rows[1][3:])
    assert not metrics.controls(tmp_path, False)["passed"]
