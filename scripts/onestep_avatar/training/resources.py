"""Synchronized process-local resource evidence; see doc/training/resources.md."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import torch

from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV
from scripts.onestep_avatar.hashing import sha256

BUDGET_ENV = "ONESTEP_AVATAR_EFFECTIVE_BUDGET"
BUDGET_SHA_ENV = "ONESTEP_AVATAR_EFFECTIVE_BUDGET_SHA256"


def read_budget(path: Path | None) -> dict | None:
    """Bind exact existing budget bytes; choose no new limits or tolerances."""
    if path is None:
        return None
    path = path.resolve()
    data = path.read_bytes()
    record = json.loads(data)
    if not isinstance(record, dict):
        raise ValueError("resource budget must be a JSON record")
    wall, memory = record.get("wall_seconds_per_phase"), record.get("memory_limit_allocated_bytes")
    if (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0
            or type(memory) is not int or memory <= 0):
        raise ValueError("resource budget requires positive wall and allocated-byte limits")
    overall = record.get("overall_seconds")
    if "overall_seconds" in record and (
        type(overall) not in (int, float) or not math.isfinite(overall) or overall <= 0
    ):
        raise ValueError("resource budget overall_seconds must be finite and positive")
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "wall_seconds_per_phase": wall, "memory_limit_allocated_bytes": memory,
            "tolerance": record.get("tolerance"), **({"overall_seconds": overall}
            if "overall_seconds" in record else {})}


def default_budget() -> dict:
    """Bind the declared queue defaults without creating a budget file or overall deadline."""
    limits = {"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48_000_000_000, "tolerance": None}
    data = json.dumps(limits, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"path": None, "sha256": hashlib.sha256(data).hexdigest(), **limits}


def effective_budget(path: Path | None = None) -> dict:
    """Use exact explicit scientific bytes or the canonical queue default."""
    return default_budget() if path is None else read_budget(path)


def _check_budget_identity(budget: dict) -> None:
    if not isinstance(budget, dict) or (budget.get("path") is not None and not isinstance(budget["path"], str)):
        raise ValueError("effective resource budget must be a checked record")
    expected = default_budget() if budget.get("path") is None else read_budget(Path(budget["path"]))
    if expected != budget:
        raise ValueError("resource budget bytes changed since preflight")


def budget_environment(budget: dict) -> dict[str, str]:
    """Carry one checked effective budget to the child; no file is invented for defaults."""
    _check_budget_identity(budget)
    return {
        BUDGET_ENV: json.dumps(budget, sort_keys=True, separators=(",", ":"), allow_nan=False),
        BUDGET_SHA_ENV: budget["sha256"],
    }


def budget_from_environment(environment: dict[str, str] | None = None) -> dict | None:
    """Require complete queue-bound effective limits; absent binding preserves ordinary behavior."""
    environment = os.environ if environment is None else environment
    data, digest = environment.get(BUDGET_ENV), environment.get(BUDGET_SHA_ENV)
    if data is None and digest is None:
        return None
    if (
        not isinstance(data, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or re.fullmatch(r"[0-9a-f]{32}", environment.get(TOKEN_ENV, "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", environment.get(JOB_ENV, "")) is None
    ):
        raise ValueError("effective resource budget requires paired queue-bound JSON and hash")
    budget = json.loads(data)
    if not isinstance(budget, dict) or budget.get("sha256") != digest:
        raise ValueError("effective resource budget hash differs from its record")
    _check_budget_identity(budget)
    return budget


def check_budget(budget: dict | None) -> None:
    if budget is not None:
        _check_budget_identity(budget)
        inherited = budget_from_environment()
        if inherited is not None and inherited != budget:
            raise ValueError("phase resource budget differs from its bound queue budget")


def training_phases(settings, *, step: int | None = None) -> list[str]:  # noqa: ANN001 -- typed settings duck interface
    """Keep notification and completion inventories in the actual save order."""
    end = settings.steps if step is None else step
    if type(end) is not int or not 0 <= end <= settings.steps:
        raise ValueError("resource inventory has an invalid completed step")
    phases = ["load"]
    if settings.save_initial:
        phases.append("export:0")
    for update in range(1, end + 1):
        phases.append(f"update:{update}")
        if (update % settings.save_every == 0 or update == settings.steps
                or (settings.save_initial and update == 1)):
            phases.append(f"export:{update}")
    return phases


class Phase:
    """Reset/measure one local phase; callers preserve the record before raising."""

    def __init__(self, device: torch.device, phase: str, rank: int, budget: dict | None):
        inherited = budget_from_environment()
        if inherited is not None and budget is not None and budget != inherited:
            raise ValueError("phase resource budget differs from its bound queue budget")
        self.device, self.phase, self.rank = torch.device(device), phase, rank
        self.budget = inherited if budget is None else budget
        self.started: float | None = None

    def start(self) -> Phase:
        check_budget(self.budget)
        if self.device.type != "cuda" and self.budget is not None:
            raise ValueError("native resource budget requires CUDA allocated-memory measurements")
        self.started = time.monotonic()
        from scripts.onestep_avatar.execution.supervision import (  # noqa: PLC0415 -- optional bound observer
            notify_phase,
        )
        notify_phase(self.phase, "begin", self.rank,
                     budget_sha256=None if self.budget is None else self.budget["sha256"])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def finish(self, error: str | None = None) -> dict:
        if self.started is None:
            raise ValueError("resource phase was not started")
        allocated = reserved = None
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
                allocated = torch.cuda.max_memory_allocated(self.device)
                reserved = torch.cuda.max_memory_reserved(self.device)
            except Exception as observation_error:
                error = error or f"resource measurement failed: {type(observation_error).__name__}: {observation_error}"
        elapsed = time.monotonic() - self.started
        self.started = None
        try:
            check_budget(self.budget)
        except (OSError, ValueError) as changed:
            error = error or str(changed)
        if self.budget is not None:
            if elapsed > self.budget["wall_seconds_per_phase"]:
                error = error or "resource phase exceeded wall-time limit"
            if allocated is not None and allocated > self.budget["memory_limit_allocated_bytes"]:
                error = error or "resource phase exceeded allocated-memory limit"
        try:
            from scripts.onestep_avatar.execution.supervision import (  # noqa: PLC0415 -- optional bound observer
                notify_phase,
            )
            notify_phase(self.phase, "end", self.rank,
                         budget_sha256=None if self.budget is None else self.budget["sha256"])
        except Exception as notification_error:
            error = error or f"phase notification failed: {type(notification_error).__name__}: {notification_error}"
        return {"schema_version": 1, "phase": self.phase, "rank": self.rank, "device": str(self.device),
                "elapsed_s": elapsed, "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                "budget_sha256": None if self.budget is None else self.budget["sha256"],
                "state": "failed" if error else "passed", "error": error}


def validate_records(records: list[dict], world: int, phases: list[str], budget: dict) -> None:
    """Require complete actual CUDA evidence under the unchanged declared budget."""
    if budget is None:
        raise ValueError("native acceptance requires a bound resource budget")
    check_budget(budget)
    if (type(world) is not int or world < 1 or not isinstance(phases, list) or not phases
            or any(not isinstance(phase, str) or not phase for phase in phases) or len(set(phases)) != len(phases)):
        raise ValueError("native resource rank/phase declaration is invalid")
    expected = {(rank, phase) for rank in range(world) for phase in phases}
    found = set()
    for record in records:
        pair = (record.get("rank"), record.get("phase"))
        allocated, reserved, elapsed = (record.get(name) for name in
                                       ("peak_allocated_bytes", "peak_reserved_bytes", "elapsed_s"))
        if (type(record.get("rank")) is not int or pair not in expected or pair in found
                or type(record.get("schema_version")) is not int or record.get("schema_version") != 1
                or record.get("state") != "passed"
                or record.get("error") is not None or record.get("budget_sha256") != budget["sha256"]
                or not isinstance(record.get("device"), str) or re.fullmatch(r"cuda:[0-9]+", record["device"]) is None
                or type(allocated) is not int or allocated < 0 or type(reserved) is not int or reserved < allocated
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                or elapsed > budget["wall_seconds_per_phase"] or allocated > budget["memory_limit_allocated_bytes"]):
            raise ValueError("native resource evidence is missing, invalid or exceeds the budget")
        found.add(pair)
    if found != expected:
        raise ValueError("native resource evidence is missing rank/phase measurements")


def read_records(output: Path, world: int, *, step: int | None = None) -> tuple[list[dict], dict[str, str]]:
    """Read every rank's saved records and bind the actual file bytes."""
    if step is not None:
        output = output / "resource_snapshots" / f"step_{step:05d}"
    records, identities = [], {}
    for rank in range(world):
        path = output / f"resources_rank{rank}.jsonl"
        data = path.read_bytes()
        identities[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        records.extend(json.loads(line) for line in data.splitlines())
        if sha256(path) != identities[str(path.resolve())]:
            raise ValueError("resource evidence changed while reading")
    return records, identities


def save_snapshot(output: Path, rank: int, step: int) -> None:
    """Keep checkpoint resource evidence immutable as later phases append to the journal."""
    from scripts.onestep_avatar.corpus.dataset import atomic_write  # noqa: PLC0415 -- existing artifact publisher
    source = output / f"resources_rank{rank}.jsonl"
    target = output / "resource_snapshots" / f"step_{step:05d}" / source.name
    if target.exists():
        raise ValueError("checkpoint resource snapshot already exists")
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(target, lambda temporary: temporary.write_bytes(data))
