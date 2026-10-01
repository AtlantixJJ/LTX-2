from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prune.data import whole_clip
from scripts.prune.score import hooks


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "native_mask.json"
    manifest = {"whole_clip": True, "trajectory_only": False, "attention": "full_bidirectional",
                "objective": "white", "text_context": {}, "geometry": {}, "seed": 42,
                "latent_dtype": "torch.bfloat16", "sigmas": [0.5],
                "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1},
                "model": {"model_key": "2.5", "transformer_fingerprint": "abc", "video_vae_fingerprint": "vae"},
                "videos": [{"view": "subject/views/view00", "sigma": 0.5, "schedule": [0.5, 0],
                            "artifacts": {"capture_sha256": "capture", "epsilon_sha256": "epsilon",
                                          "fps": 30, "blocks": [[0, 2]]}}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    path.write_text(json.dumps({"candidate_format": "whole_clip_d0_mask_v1",
        "provenance": whole_clip.native_provenance(tmp_path, manifest, ["subject/views/view00"], [0.5]),
        "masks": {"0.attn1": [1, 0], "0.attn2": [0, 1]}}))
    return path


def test_valid_report_has_content_identity(report):
    masks, digest = hooks.read_mask_artifact(
        report, model_key="2.5", fingerprint="abc",
        widths={"0.attn1": 2, "0.attn2": 2, "0.ff": 4},
    )
    assert masks["0.attn1"] == [1, 0] and len(digest) == 64


@pytest.mark.parametrize("change", ["fingerprint", "partial", "nonbinary", "empty", "wrong_width"])
def test_invalid_report_fails_before_mask_install(report, change):
    payload = json.loads(report.read_text())
    masks = payload["masks"]
    if change == "fingerprint":
        payload["provenance"]["transformer_fingerprint"] = "other"
    elif change == "partial":
        del masks["0.attn2"]
    elif change == "nonbinary":
        masks["0.attn1"] = [1, 0.5]
    elif change == "empty":
        masks["0.attn1"] = [0, 0]
    else:
        masks["0.attn1"] = [1]
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc",
                                 widths={"0.attn1": 2, "0.attn2": 2, "0.ff": 4})


def test_native_consumer_requires_task_and_input_provenance(report: Path, tmp_path: Path) -> None:
    widths = {"0.attn1": 2, "0.attn2": 2, "0.ff": 4}
    missing_task = json.loads(report.read_text())
    missing_task["provenance"].pop("task")
    report.write_text(json.dumps(missing_task))
    with pytest.raises(ValueError, match="mask task"):
        hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc", widths=widths,
                                 expected_task="whole_clip_d0")
    payload = json.loads(report.read_text())
    payload["candidate_format"] = "whole_clip_d0_mask_v1"
    manifest = {"whole_clip": True, "trajectory_only": False, "attention": "full_bidirectional",
                "objective": "white", "text_context": {}, "geometry": {}, "seed": 42,
                "latent_dtype": "torch.bfloat16", "sigmas": [0.5],
                "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1},
                "model": {"model_key": "2.5", "transformer_fingerprint": "abc", "video_vae_fingerprint": "vae"},
                "videos": [{"view": "subject/views/view00", "sigma": 0.5, "schedule": [0.5, 0],
                            "artifacts": {"capture_sha256": "capture", "epsilon_sha256": "epsilon",
                                          "fps": 30, "blocks": [[0, 2]]}}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    payload["provenance"] = whole_clip.native_provenance(tmp_path, manifest, ["subject/views/view00"], [0.5])
    report.write_text(json.dumps(payload))
    hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc", widths=widths,
                             expected_task="whole_clip_d0")
    hooks.require_native_heldout_scope(report, view="heldout", sigmas=[0.5])
    with pytest.raises(ValueError, match="used to calibrate"):
        hooks.require_native_heldout_scope(report, view="subject/views/view01", sigmas=[0.5])
    with pytest.raises(ValueError, match="not included"):
        hooks.require_native_heldout_scope(report, view="heldout", sigmas=[0.75])
    duplicate = {**manifest, "videos": [*manifest["videos"], {**manifest["videos"][0], "view": "renamed/views/view00"}]}
    with pytest.raises(ValueError, match="capture source"):
        hooks.require_native_heldout_scope(report, view="renamed/views/view00", sigmas=[.5], baseline=duplicate)
    for field in ("seed", "video_vae_fingerprint", "baseline_manifest_sha256"):
        broken = json.loads(json.dumps(payload))
        broken["provenance"].pop(field)
        report.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="native"):
            hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc", widths=widths,
                                     expected_task="whole_clip_d0")
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="seed"):
        hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc", widths=widths,
                                 expected_task="whole_clip_d0", baseline={**manifest, "seed": 43})
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "seed": 43}))
    with pytest.raises(ValueError, match="manifest content changed"):
        hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc", widths=widths,
                                 expected_task="whole_clip_d0")
    payload["provenance"].pop("sigmas")
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete native"):
        hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc", widths=widths,
                                 expected_task="whole_clip_d0")
