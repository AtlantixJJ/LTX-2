from __future__ import annotations

import collections
import json

import pytest

from scripts.prune.core import artifacts
from scripts.prune.data import chunk_states, records


def test_select_spans_many_clips_not_one(model):
    picked = records.select(artifacts.calibration("2.5"), split="calibration", limit=24)
    assert len(picked) == 24
    assert len({path.name.split("__n")[0] for path in picked}) >= 8


def test_select_balances_two_families(model):
    picked = records.select(artifacts.calibration("2.5"), split="calibration", limit=24)
    families = collections.Counter("renoised" if "__renoised" in path.name else "on_policy" for path in picked)
    assert families["on_policy"] == 12 and families["renoised"] == 12


def test_split_filter_matches_frozen_manifest(model):
    held = records.select(artifacts.calibration("2.5"), split="held_out")
    assert len(held) == 180
    names = {chunk_states.load_record(path)[2].clip for path in held[:20]}
    assert names <= set(json.loads(artifacts.manifest("2.5").read_text())["split"]["held_out"])


def test_limit_above_population_returns_everything(model):
    assert len(records.select(artifacts.calibration("2.5"), split="held_out", limit=10_000)) == 180


def test_zero_limit_is_rejected(model):
    with pytest.raises(ValueError):
        records.select(artifacts.calibration("2.5"), split="held_out", limit=0)


def test_empty_selection_names_fix(tmp_path):
    with pytest.raises(SystemExit, match="build-calibration"):
        records.select(tmp_path, split="calibration")
