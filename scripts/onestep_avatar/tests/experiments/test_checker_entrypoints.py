"""Moved acceptance CLIs keep their normal entry behavior and source binding."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.execution import process_registry, queue, software, supervision
from scripts.onestep_avatar.experiments import continuation_check as continuation_checker
from scripts.onestep_avatar.experiments import training_update_check as update_checker
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


class ControlledRegistry:
    """No OS process or GPU reservation; observe the unchanged helper's calls."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = "b" * 32
        self.started_ticks = 42
        self.released = False

    def choose(self, _inventory: dict[int, int], *, training: bool) -> tuple[int, ...]:
        assert training is False
        return (0,)

    def acquire(self, gpus: tuple[int, ...], *, job: str) -> bool:
        assert gpus == (0,)
        assert job
        return True

    def refresh(self, *, child_pid: int) -> None:
        assert child_pid == 20

    def observe_workers(self) -> dict[str, bool]:
        return {"complete": True, "workers_live": False}

    def release(self) -> None:
        self.released = True


def controlled_launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[ControlledRegistry, SimpleNamespace]:
    registry = ControlledRegistry(tmp_path / "ledger.json")
    monkeypatch.setattr(process_registry, "ProcessRegistry", lambda _path: registry)
    monkeypatch.setattr(process_registry, "gpu_memory", lambda: {0: 0})
    monkeypatch.setattr(queue, "process_identity", lambda pid: {"pid": pid, "start_ticks": 1})
    monkeypatch.setattr(supervision, "prepare_notifications", lambda *_args, **_kwargs: {})
    child = SimpleNamespace(pid=20, poll=lambda: 0)
    return registry, child


def test_moved_update_checker_self_child_preserves_command_and_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = update_checker
    registry, child = controlled_launcher(monkeypatch, tmp_path)
    output = tmp_path / "serial"
    job = tmp_path / "original_job.json"
    saved = {
        "queue_launch": {"numerical_environment": {}},
        "resource_budget": {"wall_seconds_per_phase": 1800, "sha256": "c" * 64},
    }
    monkeypatch.setattr(checker, "check_launch", lambda *_args: ({"id": "original", "sha256": "a" * 64}, saved, "no"))
    observed = {}

    def popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed.update(command=command, **kwargs)
        return child

    def supervise(*_args: object, **kwargs: object) -> dict[str, str]:
        assert kwargs["phase_seconds"] == 1800
        output.mkdir()
        (output / "result.json").write_text(json.dumps({"state": "passed", "fixture": "command-only"}))
        return {"state": "passed"}

    monkeypatch.setattr(checker.subprocess, "Popen", popen)
    monkeypatch.setattr(supervision, "supervise", supervise)
    result = checker.supervised_reference(job, output, 4, registry.path, trace=True)
    assert result["fixture"] == "command-only"
    assert observed["command"] == [
        sys.executable,
        "-m",
        "scripts.onestep_avatar.experiments.training_update_check",
        "--job",
        str(job.resolve()),
        "--output",
        str(output.resolve()),
        "--world-size",
        "4",
        "--consumer-trace",
    ]
    assert observed["cwd"] == LTX_ROOT
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert observed["start_new_session"] is True
    assert registry.released


@pytest.mark.parametrize("phase", ["capture", "reference", "control"])
def test_moved_continuation_checker_self_child_preserves_phase_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    checker = continuation_checker
    registry, child = controlled_launcher(monkeypatch, tmp_path)
    output = tmp_path / "observation"
    args = SimpleNamespace(
        output=output,
        phase=phase,
        sigma=0.725,
        history="gt",
        process_ledger=registry.path,
        snapshot=tmp_path / "snapshot.json",
        protocol=tmp_path / "protocol.json",
        control_plan=tmp_path / "controls.json",
        control_id="matched",
    )
    prepared = {
        name: {}
        for name in (
            "software",
            "fixed_inputs",
            "protocol",
            "original_result",
            "source_snapshot",
            "budget",
            "estimate",
            "control",
            "changed_noise",
        )
    }
    monkeypatch.setattr(checker, "_check_prepared", lambda _prepared: None)
    observed = {}

    def popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed.update(command=command, **kwargs)
        return child

    def supervise(*_args: object, **kwargs: object) -> dict[str, str]:
        assert kwargs["overall_seconds"] == 1800
        output.mkdir()
        record = {
            "state": "complete",
            "software": prepared["software"],
            "fixed_inputs": prepared["fixed_inputs"],
            "launch_binding": {
                "attempt_token": registry.token,
                "job_sha256": observed["env"]["ONESTEP_AVATAR_QUEUE_JOB_SHA256"],
            },
        }
        (output / "continuation.json").write_text(json.dumps(record))
        return {"state": "passed"}

    monkeypatch.setattr(checker.subprocess, "Popen", popen)
    monkeypatch.setattr(supervision, "supervise", supervise)
    result = checker.supervised_run(args, prepared)
    assert result["state"] == "complete"
    launch = output.parent / (output.name + ".supervision") / "launch.json"
    expected = [
        sys.executable,
        "-m",
        "scripts.onestep_avatar.experiments.continuation_check",
        "--phase",
        phase,
        "--output",
        str(output),
        "--gpu-id",
        "0",
        "--launch-record",
        str(launch),
    ]
    if phase == "reference":
        expected.extend(["--snapshot", str(args.snapshot.resolve())])
    else:
        expected.extend(["--protocol", str(args.protocol.resolve())])
        if phase == "control":
            expected.extend(["--control-plan", str(args.control_plan.resolve()), "--control-id", args.control_id])
        else:
            expected.extend(["--sigma", "0.725", "--history", "gt"])
    assert observed["command"] == expected
    assert observed["cwd"] == LTX_ROOT
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert observed["start_new_session"] is True
    assert registry.released
