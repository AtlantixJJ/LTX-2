"""Moved acceptance CLIs keep their normal entry behavior and source binding."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.support.children import normal_environment, run_help_rows


@pytest.mark.subprocess
def test_moved_checker_help_in_normal_environment(subtests: object) -> None:
    rows = run_help_rows(
        [
            "scripts.onestep_avatar.experiments.training_update_check",
            "scripts.onestep_avatar.experiments.training_slice_check",
            "scripts.onestep_avatar.experiments.adapter_effect_check",
            "scripts.onestep_avatar.experiments.continuation_check",
            "scripts.onestep_avatar.experiments.stock_parity",
        ],
        normal_environment(),
    )
    for row in rows:
        with subtests.test(module=row["id"]):
            assert row["returncode"] == 0, row["stderr"]
            assert "usage:" in row["stdout"]


def test_disabled_child_blocker_allows_empty_experiment_marker() -> None:
    environment = dict(os.environ)
    environment.pop("ONESTEP_AVATAR_BLOCK_EXPERIMENTS", None)
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.onestep_avatar.experiments"],
        cwd=LTX_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name",
    ["training_update_check", "training_slice_check", "adapter_effect_check", "continuation_check", "stock_parity"],
)
def test_moved_checker_source_bound_as_extra_owner(name: str) -> None:
    checker = importlib.import_module("scripts.onestep_avatar.experiments." + name)
    entry = "scripts/onestep_avatar/experiments/" + name + ".py"
    if name == "continuation_check":
        assert checker.EXTRA_SOURCES[0] == entry
    else:
        assert entry == checker.ENTRY
    marker = "scripts/onestep_avatar/experiments/__init__.py"
    assert marker in checker.EXTRA_SOURCES
    experimental = software.capture("evaluation", "bidirectional", extra_sources=checker.EXTRA_SOURCES)
    assert experimental["sources"][entry] == sha256(LTX_ROOT / entry)
    assert experimental["sources"][marker] == sha256(LTX_ROOT / marker)
    software.check_current(experimental)


@pytest.mark.parametrize(
    ("name", "retired"), [("training_update_check", "supervised_reference"), ("continuation_check", "supervised_run")]
)
def test_checker_has_no_retired_self_launch(name: str, retired: str) -> None:
    del retired
    checker = importlib.import_module("scripts.onestep_avatar.experiments." + name)
    with pytest.raises(SystemExit):
        checker.parse_args(["--supervise", "--process-ledger", "/tmp/unused"])
