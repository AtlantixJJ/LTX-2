"""Actual Torch policy, original observations and import-light launch binding."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.tests.test_applied_runtime import inventory
from scripts.onestep_avatar.tests.test_training_launch_binding import original_job
from scripts.onestep_avatar.training import numerics, runtime


@pytest.fixture(autouse=True)
def restore_flags() -> Iterator[None]:
    saved = numerics.capture()
    yield
    torch.use_deterministic_algorithms(saved["deterministic_algorithms"], warn_only=saved["deterministic_warn_only"])
    torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = saved["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]


def test_apply_uses_actual_flags_and_preserves_separate_cudnn_tf32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    separate_tf32 = torch.backends.cudnn.allow_tf32
    observed = numerics.apply()
    assert all(observed[key] == value for key, value in numerics.POLICY.items())
    assert observed["cudnn_allow_tf32"] == separate_tf32
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("required", [False, True])
def test_wrong_inherited_workspace_is_never_replaced(monkeypatch: pytest.MonkeyPatch, required: bool) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="inherited workspace"):
        numerics.apply(environment_required=required)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_queued_missing_workspace_refuses_without_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="inherited workspace"):
        numerics.apply(environment_required=True)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_missing_workspace_after_cuda_initialization_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="after CUDA initialization"):
        numerics.apply()
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_correct_inherited_workspace_allows_backend_detection_before_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    torch.use_deterministic_algorithms(False)
    observed = numerics.apply(environment_required=True)
    numerics.validate(observed, required=True)
    assert observed["deterministic_algorithms"] is True


def test_original_observed_cudnn_tf32_must_match_without_changing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    original = {**numerics.POLICY, "cudnn_allow_tf32": not torch.backends.cudnn.allow_tf32}
    preserved = torch.backends.cudnn.allow_tf32
    with pytest.raises(ValueError, match="original native reference"):
        numerics.apply(expected=original, environment_required=True)
    assert torch.backends.cudnn.allow_tf32 == preserved


@pytest.mark.parametrize("field", list(numerics.POLICY))
def test_runtime_rejects_each_wrong_original_numerical_setting(field: str) -> None:
    record = inventory(numerical=True)
    for rank in record["ranks"]:
        value = rank["numerics"][field]
        rank["numerics"][field] = not value if isinstance(value, bool) else ":16:8"
    with pytest.raises(ValueError, match="numerical policy"):
        runtime.validate(record, 4, "bf16", native=True, numerical_policy=True)


def test_historical_runtime_reader_preserves_scope_and_refuses_current_requirement() -> None:
    old = inventory()
    runtime.validate(old, 4, "bf16", native=True)
    with pytest.raises(ValueError, match="schema-two"):
        runtime.validate(old, 4, "bf16", native=True, numerical_policy=True)
    with pytest.raises(ValueError, match="schema-two"):
        runtime.numerical_policy(old)
    assert all("numerics" not in rank for rank in old["ranks"])


def test_observed_schema_two_runtime_requires_complete_rank_agreement() -> None:
    record = inventory(numerical=True)
    expected = runtime.numerical_policy(record)
    runtime.validate(record, 4, "bf16", native=True, numerical_policy=expected)
    record["ranks"][2]["numerics"]["cudnn_allow_tf32"] = not expected["cudnn_allow_tf32"]
    with pytest.raises(ValueError, match=r"original native reference|ranks disagree"):
        runtime.validate(record, 4, "bf16", native=True, numerical_policy=expected)


@pytest.mark.parametrize("change", ["missing", "integer_flag"])
def test_schema_two_never_default_fills_missing_or_malformed_flags(change: str) -> None:
    record = inventory(numerical=True)
    if change == "missing":
        del record["ranks"][0]["numerics"]
    else:
        record["ranks"][0]["numerics"]["deterministic_algorithms"] = 1
    with pytest.raises(ValueError, match=r"runtime|numerical policy"):
        runtime.validate(record, 4, "bf16", native=True, numerical_policy=True)


def test_current_queue_environment_and_historical_reader_are_both_explicit(tmp_path: Path) -> None:
    _raw, prepared, current, *_ = original_job(tmp_path)
    _, environment = queue.job_command(prepared, (0, 1, 2, 3))
    assert current["schema_version"] == 2
    assert current["numerical_environment"] == numerics.ENVIRONMENT
    assert all(environment[key] == value for key, value in numerics.ENVIRONMENT.items())
    old = queue.training_launch_record(prepared, schema_version=1)
    assert queue.verify_training_launch(old, prepared) == prepared
    assert "numerical_environment" not in old
    old["numerical_environment"] = dict(numerics.ENVIRONMENT)
    with pytest.raises(ValueError, match="launch identity"):
        queue.verify_training_launch(old, prepared)


@pytest.mark.parametrize("missing", [False, True])
def test_changed_schema_two_environment_refuses_launch(tmp_path: Path, missing: bool) -> None:
    _raw, prepared, launch, *_ = original_job(tmp_path)
    if missing:
        del launch["numerical_environment"]
    else:
        launch["numerical_environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    with pytest.raises(ValueError, match="launch identity"):
        queue.verify_training_launch(launch, prepared)




def test_typed_cli_is_import_light_then_bootstraps_before_native_imports() -> None:
    source = """
import os, sys
from scripts.onestep_avatar import train
assert 'torch' not in sys.modules
assert 'scripts.onestep_avatar.training.engine' not in sys.modules
try:
    train.main(['--mode', 'bidirectional', '--help'])
except SystemExit as result:
    assert result.code == 0
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
import torch
assert not torch.cuda.is_initialized()
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run([sys.executable, "-c", source], cwd=LTX_ROOT,
                            env=environment, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
