from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prune import artifacts


def test_every_gate_name_resolves_under_model_root():
    for name in artifacts.GATES:
        path = artifacts.gate("2.5", name)
        assert path.parent == artifacts.root("2.5") and path.name == f"{name}.json"


def test_unknown_gate_name_is_hard_error():
    with pytest.raises(KeyError):
        artifacts.gate("2.5", "teacher_manifest")


def test_manifest_points_at_file_teacher_actually_writes(calibration_index):
    assert artifacts.manifest("2.5").exists()
    manifest = json.loads(artifacts.manifest("2.5").read_text())
    assert len(manifest["corpus"]) == 44
    assert len(manifest["split"]["calibration"]) == 29 and len(manifest["split"]["held_out"]) == 15


def test_calibration_index_is_where_sweep_script_looks():
    assert artifacts.calibration_index("2.5").relative_to(artifacts.root("2.5")) == Path("calibration/index.json")


def test_run_dir_appends_to_run_index(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUT_ROOT", tmp_path)
    first = artifacts.run_dir("2.5", "head-scores", script="head_scores", argv=["--x"])
    second = artifacts.run_dir("2.5", "head-scores", script="head_scores", argv=["--y"])
    assert first != second
    lines = (tmp_path / "2.5" / "runs" / "index.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2 and json.loads(lines[0])["script"] == "head_scores"


def test_summarize_reads_only_paths_artifacts_can_produce():
    source = Path("scripts/prune/summarize_phase0.py").read_text()
    assert "teacher_manifest" not in source and '"refiner_prune"' not in source
