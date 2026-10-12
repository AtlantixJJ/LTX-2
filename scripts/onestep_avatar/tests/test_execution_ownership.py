"""Infrastructure stays independent of scheduling and provenance pins extracted owners."""

import os
import subprocess
import sys

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.execution import (
    attempt,
    clock,
    completion,
    devices,
    jobs,
    journal,
    processes,
    queue,
    recovery,
    software,
)


def test_infrastructure_imports_no_scheduler_or_model() -> None:
    source = """
import importlib, sys
for name in ('process_registry', 'supervision', 'queue_launch'):
    importlib.import_module('scripts.onestep_avatar.execution.' + name)
assert 'scripts.onestep_avatar.execution.queue' not in sys.modules
assert 'torch' not in sys.modules
assert not any(name.startswith('scripts.onestep_avatar.experiments') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=LTX_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_public_queue_imports_are_the_defining_functions() -> None:
    for owner, name in (
        (jobs, "prepare_job"),
        (jobs, "job_command"),
        (journal, "queue_state"),
        (recovery, "missing_supervision_recovery"),
        (completion, "completion_receipt"),
        (attempt, "run_child"),
        (processes, "process_identity"),
        (devices, "parse_gpu_memory"),
        (clock, "OverallDeadlineError"),
    ):
        assert getattr(queue, name) is getattr(owner, name)


def test_training_software_pins_each_extracted_execution_owner() -> None:
    sources = software.source_files("training", "causal")
    for name in ("jobs", "journal", "recovery", "completion", "attempt", "clock", "processes"):
        assert f"scripts/onestep_avatar/execution/{name}.py" in sources


def test_ordinary_profiles_pin_shared_process_observations() -> None:
    for profile in software.PROFILES:
        mode = None if profile == "decoding" else "causal"
        assert "scripts/onestep_avatar/execution/processes.py" in software.source_files(profile, mode)
