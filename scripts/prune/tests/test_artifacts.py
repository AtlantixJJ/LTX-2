from __future__ import annotations

import json

import pytest

from scripts.prune.core import artifacts


def test_every_gate_name_resolves_under_model_root():
    for name in artifacts.GATES:
        path = artifacts.gate("2.5", name)
        assert path.parent == artifacts.root("2.5") and path.name == f"{name}.json"


def test_unknown_gate_name_is_hard_error():
    with pytest.raises(KeyError):
        artifacts.gate("2.5", "teacher_manifest")


def test_run_dir_appends_to_run_index(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUT_ROOT", tmp_path)
    first = artifacts.run_dir("2.5", "head-scores", script="head_scores", argv=["--x"])
    second = artifacts.run_dir("2.5", "head-scores", script="head_scores", argv=["--y"])
    assert first != second
    lines = (tmp_path / "2.5" / "runs" / "index.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2 and json.loads(lines[0])["script"] == "head_scores"
