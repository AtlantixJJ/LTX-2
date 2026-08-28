from __future__ import annotations

import json
from pathlib import Path

from scripts.prune.core import artifacts
from scripts.prune.data import source_target


def test_no_module_refers_to_a_teacher_schedule_any_more():
    src = Path("scripts/prune/data/source_target.py").read_text()
    assert "--validate" not in src and "16 step" not in src.lower()


def test_freeze_writes_where_artifacts_says_it_does(model, tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUT_ROOT", tmp_path)
    path = source_target.freeze(model)
    assert path == artifacts.manifest("2.5")
    assert json.loads(path.read_text())["target"]["kind"] == "vae_encoded_source_latent"
