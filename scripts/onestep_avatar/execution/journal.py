"""Journal owner for managed execution; see doc/execution/journal.md."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from scripts.onestep_avatar.execution import clock as execution_clock
from scripts.onestep_avatar.execution import processes as process_handles
from scripts.onestep_avatar.execution import recovery


def record_attempt(row: dict, attempt: dict) -> None:
    """Start one attempt without aliasing its historical record to the current row."""
    row.update(state="running", error=None, returncode=None, child_identity=None, child_session=None, **attempt)
    row["attempts"].append(dict(attempt))


def update_attempt(row: dict, fields: dict, *, attempt_fields: dict | None = None) -> None:
    """Publish shared bindings once; allow terminal history to retain its existing field inventory."""
    row.update(fields)
    row["attempts"][-1].update(fields if attempt_fields is None else attempt_fields)


def read_queue_state(path: Path, jobs: list[dict]) -> dict:  # noqa: PLR0912 -- validate persisted lifecycle fields together
    """Validate append-only job identities without mutating state or claiming ownership."""
    state = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "job_order": [], "jobs": {}}
    if not isinstance(state, dict) or state.get("schema_version") != 1 or not isinstance(state.get("jobs"), dict):
        raise ValueError("invalid queue state schema")
    owner = state.get("owner_pid")
    if owner is not None and (type(owner) is not int or owner <= 0):
        raise ValueError("invalid queue owner PID")
    order = state.get("job_order")
    identifiers = [job["id"] for job in jobs]
    if not isinstance(order, list) or order != identifiers[: len(order)]:
        raise ValueError("queue revisions may only append jobs")
    if set(state["jobs"]) != set(order):
        raise ValueError("queue state job inventory differs from its order")
    for job in jobs:
        prior = state["jobs"].get(job["id"])
        if job["id"] in state["jobs"]:
            if not isinstance(prior, dict):
                raise ValueError("invalid saved queue job record")
            attempts = prior.get("attempts")
            if not isinstance(attempts, list) or any(not isinstance(attempt, dict) for attempt in attempts):
                raise ValueError("invalid saved queue attempts")
            child = prior.get("child_pid")
            if child is not None and (type(child) is not int or child <= 0):
                raise ValueError("invalid queue child PID")
            session = prior.get("child_session")
            if session is not None and (type(session) is not int or session <= 0 or session != child):
                raise ValueError("invalid queue child session")
        if prior is not None and prior.get("sha256") != job["sha256"]:
            raise ValueError("saved queue job settings changed")
        if prior is None:
            state["jobs"][job["id"]] = {"sha256": job["sha256"], "state": "pending", "attempts": []}
        elif prior.get("state") not in ("pending", "running", "complete", "failed"):
            raise ValueError("invalid saved queue job state")
        if prior is not None:
            execution_clock.validate_execution_clock(job, prior)
    state["job_order"] = identifiers
    return state


def _publish_state(path: Path, state: dict) -> None:
    """Make a successful locked transaction durable before returning ownership."""
    from scripts.onestep_avatar.corpus.dataset import atomic_write  # noqa: PLC0415 -- shared publication primitive

    def write_state(temporary: Path) -> None:
        with temporary.open("w") as output:
            json.dump(state, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

    atomic_write(path, write_state)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def queue_state(path: Path, jobs: list[dict], *, recover: bool = False):  # noqa: ANN201 -- locked recovery gates
    """Write state atomically only after a successful, exclusively owned transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            state = read_queue_state(path, jobs)
            owner = state.get("owner_pid")
            if owner != os.getpid() and process_handles.process_alive(owner):
                raise ValueError("queue state belongs to a live owner")
            if owner != os.getpid() or recover:
                recovery.recover_interrupted_jobs(path, jobs, state, recover=recover)
            state["owner_pid"] = os.getpid()
            yield state

            _publish_state(path, state)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
