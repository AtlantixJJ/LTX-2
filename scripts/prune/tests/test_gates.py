from __future__ import annotations

import json
from copy import deepcopy

from scripts.prune.evaluate import gates


def _evaluation(tmp_path, *, windows=200):
    grid, video = tmp_path / "grid.png", tmp_path / "review.mp4"
    grid.touch()
    video.touch()
    return {
        "provenance": {"model_key": "2.5", "video_vae_path": "vae", "transformer_fingerprint": "abc"},
        "geometry": {"window_frames": 25, "overlap_frames": 9},
        "seed": 42, "student_sigmas": [0.725, 0.421875, 0.0], "target": "vae_encoded_source_latent",
        "T0": {"held_out": {"rel_l2_chunk": {"mean": 0.3}}},
        "T1": {"psnr_vs_teacher": 30.0},
        "T2": {"windows": windows, "clip": "source", "source_sha256": "video-id",
               "source_frame_windows": [[i * 16, i * 16 + 25] for i in range(windows)],
               "fps": 30.0, "stride_frames": 16,
               "psnr_slope_db_per_100_chunks": -5.0},
        "T3": {"grid": str(grid), "video": str(video)},
    }


def _profile(evaluation, seconds):
    return {"provenance": evaluation["provenance"], "geometry": evaluation["geometry"],
            "seed": evaluation["seed"], "clip": evaluation["T2"]["clip"],
            "gpu_name": "A6000", "gpu_index": 1,
            "rows": [{"window_index": i, "refine_s": seconds} for i in range(evaluation["T2"]["windows"])]}


def test_matched_quality_and_speed_pass(tmp_path):
    baseline = _evaluation(tmp_path)
    candidate = deepcopy(baseline)
    candidate["T0"]["held_out"]["rel_l2_chunk"]["mean"] = 0.34
    candidate["T1"]["psnr_vs_teacher"] = 29.6
    candidate["T2"]["psnr_slope_db_per_100_chunks"] = -9.0
    result = gates.verdict(baseline=baseline, candidate=candidate, baseline_profile=_profile(baseline, 2.0),
                           profile=_profile(candidate, 1.0))
    assert result["pass"] and result["measurements"]["speedup"] == 2.0


def test_missing_or_unmatched_evidence_fails(tmp_path):
    baseline = _evaluation(tmp_path)
    candidate = deepcopy(baseline)
    result = gates.verdict(baseline=baseline, candidate=candidate, baseline_profile=None, profile=None)
    assert not result["pass"] and not result["checks"]["speed"]
    candidate["seed"] = 43
    result = gates.verdict(baseline=baseline, candidate=candidate, baseline_profile=_profile(baseline, 2),
                           profile=_profile(candidate, 1))
    assert not result["checks"]["matched_inputs"]
    slow = gates.verdict(baseline=baseline, candidate=baseline, baseline_profile=_profile(baseline, 1),
                         profile=_profile(baseline, 2))
    assert slow["measurements"]["speedup"] == 0.5 and not slow["checks"]["speed"]


def test_short_coverage_and_each_quality_limit_fail(tmp_path):
    baseline = _evaluation(tmp_path, windows=7)
    candidate = deepcopy(baseline)
    cases = [
        ("T0", ("T0", "held_out", "rel_l2_chunk", "mean"), 0.4),
        ("T1", ("T1", "psnr_vs_teacher"), 28.0),
        ("T2_drift", ("T2", "psnr_slope_db_per_100_chunks"), -20.0),
    ]
    for check, keys, value in cases:
        bad = deepcopy(candidate)
        slot = bad
        for key in keys[:-1]:
            slot = slot[key]
        slot[keys[-1]] = value
        result = gates.verdict(baseline=baseline, candidate=bad, baseline_profile=_profile(baseline, 2),
                               profile=_profile(bad, 1), mode="short", rollout_chunks=7)
        assert not result["checks"][check]
    long = gates.verdict(baseline=baseline, candidate=candidate, baseline_profile=_profile(baseline, 2),
                         profile=_profile(candidate, 1), rollout_chunks=200)
    assert not long["checks"]["T2_coverage"]


def test_cli_writes_failed_verdict_and_exits_nonzero(tmp_path, monkeypatch):
    baseline = _evaluation(tmp_path, windows=7)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    output = tmp_path / "verdict.json"
    monkeypatch.setattr("sys.argv", ["gates", "--baseline", str(path), "--candidate", str(path),
                                     "--output", str(output)])
    assert gates.main() == 1
    assert json.loads(output.read_text())["pass"] is False
