"""The declared allocated-memory quantity, exact budget and complete rank inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar.training import resources


@pytest.fixture
def budget(tmp_path: Path) -> dict:
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48000000000,
                                "tolerance": {"relative_l2": 0.02}}))
    return resources.read_budget(path)


def measured(monkeypatch: pytest.MonkeyPatch, budget: dict, *, allocated: int = 47000000000,
             elapsed: float = 100) -> tuple[resources.Phase, list]:
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("sync", str(device))))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: events.append(("reset", str(device))))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 49000000000)
    ticks = iter([10.0, 10.0 + elapsed])
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(ticks))
    phase = resources.Phase(torch.device("cuda:2"), "load", 0, budget).start()
    return phase, events


def test_phase_uses_synchronized_allocator_counters(monkeypatch: pytest.MonkeyPatch, budget: dict) -> None:
    phase, events = measured(monkeypatch, budget)
    record = phase.finish()
    assert events == [("sync", "cuda:2"), ("reset", "cuda:2"), ("sync", "cuda:2")]
    assert record["peak_allocated_bytes"] == 47000000000
    assert record["peak_reserved_bytes"] == 49000000000
    assert record["elapsed_s"] == 100
    resources.validate_records([record], 1, ["load"], budget)


@pytest.mark.parametrize(("allocated", "elapsed", "error"), [
    (48000000001, 100, "allocated-memory"), (47000000000, 1800.001, "wall-time")])
def test_breached_phase_retains_failed_measurements(
    monkeypatch: pytest.MonkeyPatch, budget: dict, allocated: int, elapsed: float, error: str
) -> None:
    phase, _events = measured(monkeypatch, budget, allocated=allocated, elapsed=elapsed)
    record = phase.finish()
    assert record["state"] == "failed"
    assert error in record["error"]
    assert record["peak_allocated_bytes"] == allocated
    with pytest.raises(ValueError, match="evidence"):
        resources.validate_records([record], 1, ["load"], budget)


def test_exact_limit_is_allowed_and_reserved_is_separate(monkeypatch: pytest.MonkeyPatch, budget: dict) -> None:
    phase, _ = measured(monkeypatch, budget, allocated=48000000000, elapsed=1800)
    resources.validate_records([phase.finish()], 1, ["load"], budget)


def test_changed_budget_preserves_failure_and_original_digest(monkeypatch: pytest.MonkeyPatch, budget: dict) -> None:
    phase, _ = measured(monkeypatch, budget)
    Path(budget["path"]).write_text(json.dumps({"wall_seconds_per_phase": 1900,
                                              "memory_limit_allocated_bytes": 48000000000}))
    record = phase.finish()
    assert record["state"] == "failed"
    assert "changed" in record["error"]
    assert record["budget_sha256"] == budget["sha256"]
    with pytest.raises(ValueError, match="changed"):
        resources.validate_records([record], 1, ["load"], budget)


def test_exception_is_retained_with_measured_peak(monkeypatch: pytest.MonkeyPatch, budget: dict) -> None:
    phase, _ = measured(monkeypatch, budget)
    record = phase.finish(error="RuntimeError: fixed diagnostic failure")
    assert record["error"] == "RuntimeError: fixed diagnostic failure"
    assert record["state"] == "failed"
    assert record["peak_allocated_bytes"] == 47000000000


@pytest.mark.parametrize("change", ["missing_rank", "missing_phase", "duplicate", "no_budget", "cpu", "nan", "bool",
                                    "bad_device", "bool_schema"])
def test_native_validator_refuses_incomplete_or_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch, budget: dict, change: str
) -> None:
    phase, _ = measured(monkeypatch, budget)
    record = phase.finish()
    world, phases = 1, ["load"]
    records = [record]
    if change == "missing_rank":
        world = 2
    elif change == "missing_phase":
        phases.append("update")
    elif change == "duplicate":
        records.append(record.copy())
    elif change == "no_budget":
        budget = None
    elif change == "cpu":
        record.update(device="cpu", peak_allocated_bytes=None, peak_reserved_bytes=None)
    elif change == "nan":
        record["elapsed_s"] = float("nan")
    elif change == "bool":
        record["peak_allocated_bytes"] = True
    elif change == "bad_device":
        record["device"] = "cuda:invalid"
    elif change == "bool_schema":
        record["schema_version"] = True
    with pytest.raises(ValueError, match=r"resource|budget"):
        resources.validate_records(records, world, phases, budget)


def test_cpu_measurement_is_explicit_and_cannot_accept_native_budget(budget: dict) -> None:
    record = resources.Phase(torch.device("cpu"), "load", 0, None).start().finish()
    assert record["peak_allocated_bytes"] is None
    assert record["peak_reserved_bytes"] is None
    with pytest.raises(ValueError, match="CUDA"):
        resources.Phase(torch.device("cpu"), "load", 0, budget).start()


@pytest.mark.parametrize("payload", [[], {}, {"wall_seconds_per_phase": True, "memory_limit_allocated_bytes": 1},
                                     {"wall_seconds_per_phase": 1, "memory_limit_allocated_bytes": 0}])
def test_missing_or_invalid_budget_fails_before_phase(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="budget"):
        resources.read_budget(path)


def test_failed_synchronization_preserves_original_error(monkeypatch: pytest.MonkeyPatch, budget: dict) -> None:
    phase, _ = measured(monkeypatch, budget)
    def denied(_device: torch.device) -> None:
        raise RuntimeError("CUDA observer failed")
    monkeypatch.setattr(torch.cuda, "synchronize", denied)
    record = phase.finish(error="original model failure")
    assert record["state"] == "failed"
    assert record["error"] == "original model failure"
    assert record["peak_allocated_bytes"] is None
