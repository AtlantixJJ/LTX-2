from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prune.core import artifacts


def test_every_gate_name_resolves_under_model_root() -> None:
    for name in artifacts.GATES:
        path = artifacts.gate("2.5", name)
        assert path.parent == artifacts.root("2.5")
        assert path.name == f"{name}.json"


def test_unknown_gate_name_is_hard_error() -> None:
    with pytest.raises(KeyError):
        artifacts.gate("2.5", "teacher_manifest")


def test_run_dir_appends_to_run_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifacts, "OUT_ROOT", tmp_path)
    first = artifacts.run_dir("2.5", "head-scores", script="head_scores", argv=["--x"])
    second = artifacts.run_dir("2.5", "head-scores", script="head_scores", argv=["--y"])
    assert first != second
    lines = (tmp_path / "2.5" / "runs" / "index.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["script"] == "head_scores"


def test_verified_hash_memo_never_hides_a_fresh_rewrite(tmp_path):
    import os

    from scripts.prune.core import provenance

    path = tmp_path / "input.bin"
    path.write_bytes(b"a" * 64)
    old = 1_000_000_000
    os.utime(path, ns=(old, old))
    first = provenance.verified_sha256(path)
    assert provenance.verified_sha256(path) == first
    path.write_bytes(b"b" * 64)
    assert provenance.verified_sha256(path) != first
