"""Recovery owner for managed execution; see doc/execution/recovery.md."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.onestep_avatar.execution import clock as execution_clock
from scripts.onestep_avatar.execution import completion as evidence
from scripts.onestep_avatar.execution import processes as process_handles
from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_PROTOCOL


def missing_supervision_recovery(path: Path, row: dict) -> dict:
    """Classify a bounded, already closed owned attempt; never infer success."""
    import hashlib  # noqa: PLC0415 -- immutable interruption evidence

    from scripts.onestep_avatar.execution import (  # noqa: PLC0415 -- explicit operational evidence only
        process_registry,
        queue_launch,
    )
    from scripts.onestep_avatar.execution.queue_protocol import TOKEN_ENV  # noqa: PLC0415 -- original token

    deadline = time.monotonic() + 10
    journal_bytes = queue_launch._bounded_bytes(path, deadline, limit=8 * 1024 * 1024)
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        raise ValueError("missing-supervision recovery requires an original approved attempt")
    original = attempts[-1]
    if any(key in item for item in (row, original) for key in ("supervision_result", "supervision_result_sha256")):
        raise ValueError("present or partial supervision evidence must pass ordinary verification")
    fields = (
        "command",
        "gpus",
        "child_pid",
        "child_identity",
        "owner_pid",
        "process_ledger",
        "environment_changes",
        "launch_protocol",
        "launch_request",
        "launch_request_sha256",
        "attempt_started_ticks",
        "supervision_contract",
        "runtime_contract",
        "runtime_contract_sha256",
    )
    if row.get("launch_protocol") != LAUNCH_PROTOCOL or any(row.get(key) != original.get(key) for key in fields):
        raise ValueError("missing-supervision recovery original launch bindings differ")
    request_path = Path(row["launch_request"])
    request_bytes = queue_launch._bounded_bytes(request_path, deadline)
    request = json.loads(request_bytes)
    queue_launch.validate_request(request)
    if (
        hashlib.sha256(request_bytes).hexdigest() != row["launch_request_sha256"]
        or request["journal"] != str(path.resolve())
        or request["job_sha256"] != row["sha256"]
        or request["command"] != row["command"]
        or request["owner_identity"]["pid"] != row["owner_pid"]
    ):
        raise ValueError("missing-supervision recovery original request differs")
    bootstrap_bytes = queue_launch._bounded_bytes(request_path.parent / "bootstrap.json", deadline)
    grant_bytes = queue_launch._bounded_bytes(request_path.parent / "grant.json", deadline)
    bootstrap, grant = json.loads(bootstrap_bytes), json.loads(grant_bytes)
    if bootstrap != grant:
        raise ValueError("missing-supervision recovery original approval differs")
    queue_launch.verify_journal(request_path, request, bootstrap)
    token = row["environment_changes"].get(TOKEN_ENV)
    registry = object.__new__(process_registry.ProcessRegistry)
    registry.path = Path(row["process_ledger"]).resolve()
    with registry._lock(timeout=max(deadline - time.monotonic(), 1e-12)):
        ledger = registry._read()
        owned = ledger["attempts"].get(token)
        recovery = owned.get("recovery") if isinstance(owned, dict) else None
        if (
            not isinstance(recovery, dict)
            or owned.get("state") != "closed"
            or owned.get("owner") != request["owner_identity"]
            or owned.get("child_pid") != row["child_pid"]
            or owned.get("gpus") != row["gpus"]
            or owned.get("job") != request["job_id"]
            or owned.get("attempt_started_ticks") != row["attempt_started_ticks"]
            or recovery.get("basis") != "bounded_dead_owner_recovery"
            or recovery.get("token") != token
            or recovery.get("job_sha256") != row["sha256"]
            or recovery.get("request") != str(request_path.resolve())
            or recovery.get("request_sha256") != row["launch_request_sha256"]
            or recovery.get("journal") != str(path.resolve())
            or recovery.get("journal_sha256") != hashlib.sha256(journal_bytes).hexdigest()
            or recovery.get("registered_workers_absent") is not True
            or recovery.get("continuous_supervision") is not False
            or recovery.get("containment_complete") is not False
        ):
            raise ValueError("missing-supervision recovery lacks the exact bounded owned-attempt closure")
        previous = {key: value for key, value in owned.items() if key != "recovery"}
        previous["state"] = "active"
        if queue_launch._row_digest(previous) != recovery.get("original_row_sha256"):
            raise ValueError("missing-supervision recovery original owned observations differ")
        processes = owned.get("processes")
        if not isinstance(processes, dict) or not processes or len(processes) > 4096:
            raise ValueError("missing-supervision recovery has no bounded registered handle inventory")
        identities = [owned["owner"], *[item["identity"] for item in processes.values()]]
        for identity in identities:
            queue_launch._recovery_identity(identity)
        observations = [queue_launch._ended_handles(identities, deadline)]
        observations.append(queue_launch._ended_handles(identities, deadline))
        if (
            registry._read() != ledger
            or journal_bytes != queue_launch._bounded_bytes(path, deadline, limit=8 * 1024 * 1024)
            or request_bytes != queue_launch._bounded_bytes(request_path, deadline)
            or bootstrap_bytes != queue_launch._bounded_bytes(request_path.parent / "bootstrap.json", deadline)
            or grant_bytes != queue_launch._bounded_bytes(request_path.parent / "grant.json", deadline)
        ):
            raise ValueError("missing-supervision recovery evidence changed during inspection")
    return {
        "basis": "missing_original_supervision",
        "operational_closure": recovery,
        "registered_handle_observations": observations,
        "continuous_supervision": False,
        "unknown_descendants_unproven": True,
        "scientific_success": False,
        "prior_attempt": json.loads(json.dumps(original, allow_nan=False)),
    }


def recover_interrupted_jobs(path: Path, jobs: list[dict], state: dict, *, recover: bool) -> None:  # noqa: PLR0912 -- ordered recovery gates
    """Check interrupted handles; adopt outputs only on explicit recovery under the journal lock."""
    running = [job for job in jobs if state["jobs"][job["id"]]["state"] == "running"]
    unapproved = {}
    for job in running:
        row = state["jobs"][job["id"]]
        if process_handles.process_alive(row.get("child_pid")):
            raise ValueError("queue has a surviving child; recovery is required")
        if row.get("child_pid") is None or (row.get("child_identity") is None and row.get("launch_protocol")):
            from scripts.onestep_avatar.execution.queue_launch import (  # noqa: PLC0415
                inspect_unapproved_launch,
            )

            unapproved[job["id"]] = inspect_unapproved_launch(row)
        elif process_handles.inspect_child(row) == "live":
            raise ValueError("queue has a surviving child; recovery is required")
    if running and not recover:
        raise ValueError("queue interrupted attempts require explicit recovery")
    for job in running:
        row = state["jobs"][job["id"]]
        if job["id"] in unapproved:
            result = {
                "state": "failed",
                "error": "queue owner interrupted before launch approval",
                "recovered_at": time.time(),
                "launch_recovery": unapproved[job["id"]],
            }
            row.update(result)
            row["attempts"][-1].update(result)
        elif row.get("overall_deadline_at") is not None and time.time() >= row["overall_deadline_at"]:
            row.update(
                state="failed",
                error="overall_deadline: interrupted job deadline expired",
                failure_reason="overall_deadline",
                recovered_at=time.time(),
            )
            if row["attempts"]:
                row["attempts"][-1].update(state="failed", error=row["error"], failure_reason="overall_deadline")
        elif evidence.verify_completion(job):
            if (
                "effective_budget" in job
                and row.get("launch_protocol") == LAUNCH_PROTOCOL
                and not any(key in row for key in ("supervision_result", "supervision_result_sha256"))
            ):
                recovery = missing_supervision_recovery(path, row)
                result = {
                    "state": "failed",
                    "error": "missing_supervision: original owner ended without its supervisor result",
                    "failure_reason": "missing_supervision",
                    "recovered_at": time.time(),
                    "interruption_recovery": recovery,
                }
                row.update(result)
                row["attempts"][-1].update(result)
                continue
            if "effective_budget" in job:
                evidence.verify_attempt_evidence(job, row)
            receipt = evidence.completion_receipt(job, attempt=row if "effective_budget" in job else None)
            observed_at = time.time()
            try:
                execution_clock.remaining_deadline(row, now=observed_at)
            except execution_clock.OverallDeadlineError:
                row.update(
                    state="failed",
                    error="overall_deadline: expired during recovery verification",
                    failure_reason="overall_deadline",
                    recovered_at=observed_at,
                )
            else:
                if "execution" in receipt:
                    receipt["execution"]["binding"]["completed_at"] = observed_at
                row.update(state="complete", receipt=receipt, recovered_at=observed_at)
        else:
            row.update(
                state="failed",
                error="queue owner interrupted before verified completion",
                recovered_at=time.time(),
            )
