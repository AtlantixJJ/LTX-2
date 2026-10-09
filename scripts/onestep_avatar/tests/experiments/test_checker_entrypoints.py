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


@pytest.mark.parametrize(
    "module",
    ["training_update_check", "training_slice_check", "adapter_effect_check", "continuation_check", "stock_parity"],
)
def test_moved_checker_help_in_normal_environment(module: str) -> None:
    environment = dict(os.environ)
    environment.pop("ONESTEP_AVATAR_BLOCK_EXPERIMENTS", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.onestep_avatar.experiments." + module, "--help"],
        cwd=LTX_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


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
    ordinary = software.capture("evaluation", "bidirectional")
    assert not any("/experiments/" in path for path in ordinary["sources"])
    marker = "scripts/onestep_avatar/experiments/__init__.py"
    assert marker in checker.EXTRA_SOURCES
    experimental = software.capture("evaluation", "bidirectional", extra_sources=checker.EXTRA_SOURCES)
    assert experimental["sources"][entry] == sha256(LTX_ROOT / entry)
    assert experimental["sources"][marker] == sha256(LTX_ROOT / marker)
    software.check_current(experimental)



@pytest.mark.parametrize(("name", "retired"), [
    ("training_update_check", "supervised_reference"), ("continuation_check", "supervised_run")])
def test_checker_has_no_retired_self_launch(name: str, retired: str) -> None:
    checker = importlib.import_module("scripts.onestep_avatar.experiments." + name)
    assert not hasattr(checker, retired)
    with pytest.raises(SystemExit):
        checker.parse_args(["--supervise", "--process-ledger", "/tmp/unused"])
