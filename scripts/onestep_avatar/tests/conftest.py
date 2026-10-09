"""Controlled lifecycle fixtures; real launch-gate checks never use this fixture."""
import importlib.util
import os
from pathlib import Path

import pytest

from scripts.onestep_avatar import PACKAGE_ROOT
from scripts.onestep_avatar.execution import queue_launch

if os.environ.get("ONESTEP_AVATAR_BLOCK_EXPERIMENTS") == "1":
    hook_directory = PACKAGE_ROOT / "tests/blocked_imports"
    hook_spec = importlib.util.spec_from_file_location(
        "onestep_avatar_experiment_import_blocker", hook_directory / "sitecustomize.py")
    if hook_spec is None or hook_spec.loader is None:
        raise ImportError("experiment import blocker is absent")
    hook_spec.loader.exec_module(importlib.util.module_from_spec(hook_spec))
    previous_path = os.environ.get("PYTHONPATH")
    if previous_path is None or str(hook_directory) not in previous_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = str(hook_directory) + (
            "" if previous_path is None else os.pathsep + previous_path)
    collect_ignore = ["experiments"]


@pytest.fixture
def controlled_training_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the synthetic marker test on hash/mode checks; science has its own tests."""
    from scripts.onestep_avatar.training import engine

    monkeypatch.setattr(engine, 'verify_training_conditions', lambda _job, _checkpoint, _contract: None)


@pytest.fixture
def controlled_evaluation_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope synthetic lifecycle records to lifecycle and generic tensor checks.

    These tests do not contain checked masters, weights or actual sampler output.
    Scientific result binding is tested separately against the production verifier.
    No production path bypasses the verifier.
    """
    from scripts.onestep_avatar import evaluate

    monkeypatch.setattr(evaluate, 'verify_evaluation_conditions', lambda _arguments, _paths: None)
    monkeypatch.setattr(evaluate, 'evaluation_evidence_paths', lambda _arguments, paths: paths)


@pytest.fixture
def controlled_queue_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep old fake-Popen tests scoped to receipts/retries, not OS registration.

    Each caller already replaces Popen with an immediate synthetic completion.
    These substitutions do not establish bootstrap/grant or process correctness.
    Real-process integration tests retain every production protocol function.
    """
    monkeypatch.setattr(queue_launch, 'guard_command', lambda path: queue_launch.load_request(path)['command'])

    def registration(path: Path, child, _command, _refresh) -> dict:  # noqa: ANN001 -- controlled child interfaces
        request = queue_launch.load_request(path)
        return {'identity': {'pid': child.pid, 'start_ticks': 1,
                             'command': request['command'], 'terminal': False}}

    monkeypatch.setattr(queue_launch, 'wait_registration', registration)
    monkeypatch.setattr(queue_launch, 'publish_grant', lambda _path: {})
