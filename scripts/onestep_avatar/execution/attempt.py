"""Attempt owner for managed execution; see doc/execution/attempt.md."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_PROTOCOL

if TYPE_CHECKING:
    from scripts.onestep_avatar.execution.process_registry import ProcessRegistry
from scripts.onestep_avatar.execution import clock as execution_clock
from scripts.onestep_avatar.execution import completion as evidence
from scripts.onestep_avatar.execution import jobs as job_contracts
from scripts.onestep_avatar.execution import journal as state_store
from scripts.onestep_avatar.execution import processes as process_handles


@dataclass(frozen=True)
class LaunchPlan:
    """The exact contracts and environment prepared before any child exists."""

    command: list[str]
    changes: dict[str, str]
    budget: dict
    profile: dict
    notifications: Path
    runtime_contract: Path | None
    supervision_path: Path
    launch_binding: Path | None


def _prepare_launch(job: dict, claims: ProcessRegistry, log_path: Path, *, foreground: bool) -> LaunchPlan:
    """Freeze launch, phase and runtime contracts before publishing the attempt."""
    from scripts.onestep_avatar.execution import queue_launch, runtime, supervision  # noqa: PLC0415
    from scripts.onestep_avatar.training import resources  # noqa: PLC0415

    command, changes = job_contracts.job_command(job, tuple(sorted(claims.owned)))
    if foreground:
        changes["PYTHONUNBUFFERED"] = "1"
    if job.get("sha256") is not None:
        from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV  # noqa: PLC0415

        changes.update({TOKEN_ENV: claims.token, JOB_ENV: job["sha256"]})
    claims.refresh()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launch_binding = None
    if job["kind"] == "train":
        from scripts.onestep_avatar.execution.queue_protocol import (  # noqa: PLC0415 -- exact launch evidence
            LAUNCH_ENV,
        )

        launch_record = job_contracts.training_launch_record(job, tuple(sorted(claims.owned)))
        if command != launch_record["command"]:
            raise ValueError("queue command changed before launch binding")
        launch_binding = log_path.with_name(f"{log_path.stem}.{claims.token}.launch.json").resolve()
        queue_launch.publish_exclusive(launch_binding, launch_record)
        changes[LAUNCH_ENV] = str(launch_binding)
    budget = job.get("effective_budget", resources.default_budget())
    changes.update(resources.budget_environment(budget))
    profile = job_contracts.execution_profile(job)
    notifications = log_path.with_name(f"{log_path.stem}.{claims.token}.phases.json").resolve()
    changes.update(
        supervision.prepare_notifications(
            notifications,
            token=claims.token,
            job_sha256=job["sha256"],
            world=profile["world"],
            phases=profile["phases"],
            budget_sha256=budget["sha256"],
        )
    )
    runtime_contract = None
    if job["kind"] != "train":
        runtime_contract = log_path.with_name(f"{log_path.stem}.{claims.token}.runtime.json").resolve()
        changes.update(
            runtime.prepare_contract(
                runtime_contract,
                token=claims.token,
                job_sha256=job["sha256"],
                budget=budget,
                generic_phase=profile["generic_phase"],
                numerical_policy=job.get("numerical_policy"),
            )
        )
    supervision_path = log_path.with_name(f"{log_path.stem}.{claims.token}.supervision.json").resolve()
    return LaunchPlan(
        command, changes, budget, profile, notifications, runtime_contract, supervision_path, launch_binding
    )


def _start_attempt(
    job: dict,
    claims: ProcessRegistry,
    plan: LaunchPlan,
    state_path: Path,
    jobs: list[dict] | None,
    log_path: Path,
    dependency_receipts: dict[str, dict] | None,
) -> tuple[Path, dict]:
    """Bind dependencies and the original clock in one running journal publication."""
    from scripts.onestep_avatar.execution import queue_launch  # noqa: PLC0415

    registry_path = getattr(claims, "path", None)
    if jobs is None:
        raise ValueError("persistent child execution requires the full prepared job list")
    if sum(candidate == job for candidate in jobs) != 1:
        raise ValueError("queue child differs from its prepared job list")
    with state_store.queue_state(state_path, jobs) as state:
        row = state["jobs"][job["id"]]
        if row["state"] != "pending":
            raise ValueError("queue child requires a pending saved job")
        if dependency_receipts is None:
            ready = job in evidence.ready_jobs(jobs, state)
        else:
            dependencies = job.get("dependencies", [])
            ready = set(dependency_receipts) == set(dependencies) and all(
                state["jobs"][dependency]["state"] == "complete"
                and state["jobs"][dependency]["receipt"] == dependency_receipts[dependency]
                and state["jobs"][dependency]["sha256"] == dependency_receipts[dependency]["job_sha256"]
                for dependency in dependencies
            )
        if not ready:
            raise ValueError("queue child dependencies are not verified complete")
        request_path = queue_launch.prepare_request(
            state_path.parent / "launches" / claims.token,
            token=claims.token,
            job_sha256=job["sha256"],
            owner_identity=process_handles.process_identity(os.getpid()),
            job_id=job["id"],
            journal=state_path,
            command=plan.command,
        )
        started_at = time.time()
        clock = execution_clock.begin_execution_clock(job, row, now=started_at)
        attempt = {
            "command": plan.command,
            "log": str(log_path.resolve()),
            "started_at": started_at,
            "gpus": sorted(claims.owned),
            "owner_pid": os.getpid(),
            "child_pid": None,
            "environment_changes": plan.changes,
            "attempt_started_ticks": claims.started_ticks,
            "launch_protocol": LAUNCH_PROTOCOL,
            "launch_request": str(request_path.resolve()),
            "launch_request_sha256": queue_launch.digest(request_path),
            **clock,
        }
        if registry_path is not None:
            attempt["process_ledger"] = str(registry_path.resolve())
        attempt["supervision_contract"] = str(plan.notifications)
        if plan.runtime_contract is not None:
            attempt.update(
                runtime_contract=str(plan.runtime_contract),
                runtime_contract_sha256=queue_launch.digest(plan.runtime_contract),
            )
        if plan.launch_binding is not None:
            attempt.update(
                training_launch=str(plan.launch_binding),
                training_launch_sha256=queue_launch.digest(plan.launch_binding),
            )
        state_store.record_attempt(row, attempt)
    return request_path, clock


def run_child(  # noqa: PLR0912, PLR0915 -- child lifecycle and persistent journal
    job: dict,
    claims: ProcessRegistry,
    log_path: Path,
    *,
    poll_seconds: float = 1,
    state_path: Path | None = None,
    jobs: list[dict] | None = None,
    dependency_receipts: dict[str, dict] | None = None,
    foreground: bool = False,
) -> dict:
    """Run one claimed package child with optional persistent attempt records."""
    import subprocess  # noqa: PLC0415 -- package-only subprocess execution

    from scripts.onestep_avatar.execution import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
        queue_launch,
        runtime,
        supervision,
        terminal,
    )

    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("child polling interval must be finite and positive")
    if not claims.owned:
        raise ValueError("child execution requires owned GPU reservations")
    child = None
    returncode = None
    journal_started = False
    request_path = None
    changes = {}
    registry_path = getattr(claims, "path", None)
    clock = {}
    supervision_path = None
    result = None
    try:
        output = Path(job["output"])
        if log_path.resolve().is_relative_to(output.resolve()):
            raise ValueError("queue child log must stay outside its output directory")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("queue child requires a fresh output directory")
        plan = _prepare_launch(job, claims, log_path, foreground=foreground)
        command, changes = plan.command, plan.changes
        budget, profile = plan.budget, plan.profile
        notifications, runtime_contract = plan.notifications, plan.runtime_contract
        supervision_path = plan.supervision_path
        if state_path is not None:
            request_path, clock = _start_attempt(job, claims, plan, state_path, jobs, log_path, dependency_receipts)
            journal_started = True
        else:
            clock = execution_clock.begin_execution_clock(job, {"attempts": []}, now=time.time())
        with log_path.open("x") as log, terminal.follow_log(log_path, enabled=foreground):
            launch_command = command if request_path is None else queue_launch.guard_command(request_path)
            child = subprocess.Popen(
                launch_command,
                cwd=LTX_ROOT,
                env={**os.environ, **changes},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            claims.refresh(child_pid=child.pid)
            if state_path is not None:
                remaining = execution_clock.remaining_deadline(clock)
                bootstrap = queue_launch.wait_registration(
                    request_path,
                    child,
                    launch_command,
                    lambda: claims.refresh(child_pid=child.pid),
                    timeout=45 if remaining is None else min(45, remaining),
                )
                identity = bootstrap["identity"]
                with state_store.queue_state(state_path, jobs) as state:
                    state_store.update_attempt(
                        state["jobs"][job["id"]],
                        {
                            "child_pid": child.pid,
                            "child_identity": identity,
                            "child_session": child.pid,
                        },
                    )
                execution_clock.remaining_deadline(clock)
                queue_launch.publish_grant(request_path)
            worker_record = {
                "child_session": child.pid,
                "environment_changes": changes,
                "attempt_started_ticks": claims.started_ticks,
            }
            if registry_path is not None:
                worker_record["process_ledger"] = str(registry_path.resolve())
            if state_path is not None:
                worker_record = state_store.read_queue_state(state_path, jobs)["jobs"][job["id"]]
            else:
                identity = process_handles.process_identity(child.pid)
            result = supervision.supervise(
                child,
                identity=identity,
                command=command,
                worker_record=worker_record,
                claims=claims,
                gpus=tuple(sorted(claims.owned)),
                evidence_path=supervision_path,
                notifications_path=notifications,
                startup_seconds=budget["wall_seconds_per_phase"],
                phase_seconds=budget["wall_seconds_per_phase"],
                shutdown_seconds=30,
                overall_seconds=execution_clock.remaining_deadline(clock),
                poll_seconds=min(poll_seconds, 1),
                sampled_total_device_limit_bytes=profile.get("sampled_total_device_limit_bytes"),
            )
            returncode = result["returncode"]
            if state_path is not None:
                with state_store.queue_state(state_path, jobs) as state:
                    row = state["jobs"][job["id"]]
                    binding = {
                        "supervision_result": str(supervision_path),
                        "supervision_result_sha256": queue_launch.digest(supervision_path),
                    }
                    state_store.update_attempt(row, binding)
            if result["state"] != "passed":
                if (result.get("error") or "").startswith("KeyboardInterrupt:"):
                    raise KeyboardInterrupt(result["error"])
                if clock.get("overall_deadline_at") is not None and time.time() >= clock["overall_deadline_at"]:
                    raise execution_clock.OverallDeadlineError(f"overall_deadline: {result['error']}; log: {log_path}")
                raise execution_clock.QueueChildError(
                    f"bounded package child failed: {result['error']}; log: {log_path}"
                )
            if returncode != 0:
                raise execution_clock.QueueChildError(f"queue child failed with exit {returncode}; log: {log_path}")
            if process_handles.owned_workers_live(worker_record):
                raise ValueError("queue child still has live workers after leader exit")
            execution_clock.remaining_deadline(clock)
            if runtime_contract is not None:
                runtime.verify_result(runtime_contract, queue_launch.digest(runtime_contract))
            if result["workers_absent"] is not True:
                raise ValueError("queue success requires complete owned-worker absence")
            receipt = evidence.completion_receipt(
                job,
                attempt=None
                if state_path is None
                else state_store.read_queue_state(state_path, jobs)["jobs"][job["id"]],
            )
            execution_clock.remaining_deadline(clock)
            result = {
                "child_pid": child.pid,
                "returncode": returncode,
                "command": command,
                "environment_changes": changes,
                "log": str(log_path.resolve()),
                "receipt": receipt,
            }
            if state_path is not None:
                with state_store.queue_state(state_path, jobs) as state:
                    finished_at = time.time()
                    execution_clock.remaining_deadline(clock, now=finished_at)
                    if "execution" in receipt:
                        receipt["execution"]["binding"]["completed_at"] = finished_at
                    state_store.update_attempt(
                        state["jobs"][job["id"]],
                        {"state": "complete", "finished_at": finished_at, **result},
                        attempt_fields={"state": "complete", **result},
                    )
            return result
    except BaseException as error:
        # Registration and grant failure must stop the same owned guard/ranks too.
        # The existing supervisor remains the sole bounded stop owner.
        if child is not None and result is None:
            identity = process_handles.process_identity(child.pid)
            if identity is not None:
                worker_record = {
                    "child_session": child.pid,
                    "environment_changes": changes,
                    "attempt_started_ticks": claims.started_ticks,
                    **({"process_ledger": str(registry_path.resolve())} if registry_path is not None else {}),
                }
                if journal_started:
                    worker_record = state_store.read_queue_state(state_path, jobs)["jobs"][job["id"]]
                result = supervision.supervise(
                    child,
                    identity=identity,
                    command=command,
                    worker_record=worker_record,
                    claims=claims,
                    gpus=tuple(sorted(claims.owned)),
                    evidence_path=supervision_path,
                    notifications_path=None,
                    startup_seconds=30,
                    phase_seconds=30,
                    shutdown_seconds=30,
                    overall_seconds=1e-12,
                    poll_seconds=min(poll_seconds, 1),
                )
                returncode = result["returncode"]
        if journal_started:
            with state_store.queue_state(state_path, jobs) as state:
                row = state["jobs"][job["id"]]
                if row["state"] == "running":
                    live = child is not None and (
                        child.poll() is None
                        or process_handles.owned_workers_live(
                            {
                                "child_session": child.pid,
                                "environment_changes": changes,
                                "attempt_started_ticks": claims.started_ticks,
                                **(
                                    {"process_ledger": str(registry_path.resolve())}
                                    if registry_path is not None
                                    else {}
                                ),
                            }
                        )
                    )
                    row.update(
                        state="running" if live else "failed",
                        error=f"{type(error).__name__}: {error}",
                        child_pid=None if child is None else child.pid,
                        returncode=returncode,
                        finished_at=time.time(),
                        failure_reason=(
                            "overall_deadline"
                            if isinstance(error, execution_clock.OverallDeadlineError)
                            or (
                                clock.get("overall_deadline_at") is not None
                                and time.time() >= clock["overall_deadline_at"]
                            )
                            else "cancelled"
                            if isinstance(error, KeyboardInterrupt)
                            else "child_failure"
                        ),
                    )
                    if supervision_path is not None and supervision_path.is_file():
                        row.update(
                            supervision_result=str(supervision_path),
                            supervision_result_sha256=queue_launch.digest(supervision_path),
                        )
                    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- terminal log evidence

                    row["attempts"][-1].update(
                        state=row["state"],
                        error=row["error"],
                        returncode=returncode,
                        finished_at=row["finished_at"],
                        log_sha256=sha256(log_path) if log_path.is_file() else None,
                        failure_reason=row["failure_reason"],
                        **(
                            {key: row[key] for key in ("supervision_result", "supervision_result_sha256")}
                            if "supervision_result" in row
                            else {}
                        ),
                    )
        raise
    finally:
        if child is None or (
            child.poll() is not None
            and not process_handles.owned_workers_live(
                {
                    "child_session": child.pid,
                    "environment_changes": changes,
                    "attempt_started_ticks": claims.started_ticks,
                    **({"process_ledger": str(registry_path.resolve())} if registry_path is not None else {}),
                }
            )
        ):
            claims.release()


def startup_retry_evidence(job: dict, row: dict) -> dict | None:  # noqa: PLR0911, PLR0912 -- conservative ordered retry gates
    """Authorize proven startup contention with no surviving workers or update evidence."""
    from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, PREFIX, TOKEN_ENV  # noqa: PLC0415
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415

    attempts = row.get("attempts", [])
    if row.get("failure_reason") == "overall_deadline":
        return None
    try:
        execution_clock.remaining_deadline(row)
    except execution_clock.OverallDeadlineError:
        return None
    if job["kind"] != "train" or row.get("state") != "failed" or not attempts or len(attempts) >= 4:
        return None
    attempt = attempts[-1]
    if (
        attempt.get("state") != "failed"
        or type(attempt.get("returncode")) is not int
        or attempt["returncode"] == 0
        or not attempt.get("child_session")
        or attempt["returncode"] != row.get("returncode")
        or attempt.get("child_pid") != row.get("child_pid")
        or attempt["child_session"] != row.get("child_session")
    ):
        return None
    if process_handles.inspect_child(row) == "live":
        return None
    changes = attempt.get("environment_changes", {})
    token = changes.get(TOKEN_ENV)
    if not token or changes.get(JOB_ENV) != job["sha256"]:
        return None
    log = Path(attempt.get("log", ""))
    if not log.is_file() or sha256(log) != attempt.get("log_sha256"):
        return None
    contended = []
    for line in log.read_text().splitlines():
        if PREFIX not in line:
            continue
        if not line.startswith(PREFIX):
            return None
        try:
            event = json.loads(line[len(PREFIX) :])
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(event, dict)
            or set(event) != {"schema_version", "event", "token", "job_sha256", "rank", "reason"}
            or type(event.get("schema_version")) is not int
            or event["schema_version"] != 1
            or event.get("token") != token
            or event.get("job_sha256") != job["sha256"]
            or type(event.get("rank")) is not int
            or not 0 <= event["rank"] < 4
            or event.get("event") != "startup_contended"
            or event.get("reason") not in ("cuda_oom", "port_in_use")
        ):
            return None  # Includes any rank's updates_begin boundary.
        contended.append(event["rank"])
    output = Path(job["output"])
    if not contended or any(path.stat().st_size for path in output.rglob("metrics_rank*.jsonl")):
        return None
    return {
        "token": token,
        "log": str(log.resolve()),
        "log_sha256": sha256(log),
        "contended_ranks": sorted(set(contended)),
        "attempt": len(attempts),
    }


def retry_startup_failure(job: dict, jobs: list[dict], state_path: Path) -> bool:
    """Preserve a proven failed attempt before making that same job pending again."""
    import shutil  # noqa: PLC0415 -- preserve terminal logs without changing originals

    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415

    output, moved = Path(job["output"]), None
    if startup_retry_evidence(job, state_store.read_queue_state(state_path, jobs)["jobs"][job["id"]]) is None:
        return False
    try:
        with state_store.queue_state(state_path, jobs) as state:
            row = state["jobs"][job["id"]]
            evidence = startup_retry_evidence(job, row)
            if evidence is None:
                return False
            archive = output.parent / "superseded_startup_contention" / f"{output.name}_{evidence['token']}"
            files = {}
            if output.exists():
                if output.is_symlink() or not output.is_dir() or any(path.is_symlink() for path in output.rglob("*")):
                    raise ValueError("startup archive requires ordinary output files")
                files = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()}
            archive.mkdir(parents=True, exist_ok=False)
            if output.exists():
                moved = archive / "output"
                output.rename(moved)
            shutil.copyfile(evidence["log"], archive / "child.log")
            if sha256(archive / "child.log") != evidence["log_sha256"]:
                raise ValueError("startup log changed during preservation")
            if moved is not None and files != {
                str(path.relative_to(moved)): sha256(path) for path in moved.rglob("*") if path.is_file()
            }:
                raise ValueError("startup output changed during preservation")
            record = {
                "job_id": job["id"],
                "job_sha256": job["sha256"],
                **evidence,
                "original_output": str(output.resolve()),
                "files": files,
                "error": row["error"],
                "returncode": row["returncode"],
            }
            (archive / "preservation.json").write_text(json.dumps(record, indent=2) + "\n")
            (archive / "README.md").write_text(
                "Startup contention before training began. The replacement is the same saved\n"
                "queue job; it remains pending until another attempt completes. Original\n"
                "output bytes and the failed log are preserved here. Do not cite this attempt\n"
                "as a scientific result. The queue journal records whether preservation\n"
                "and retry publication succeeded.\n"
            )
            row["attempts"][-1].update(archive=str(archive.resolve()), retry_evidence=evidence)
            row.update(
                state="pending",
                child_pid=None,
                child_identity=None,
                child_session=None,
                startup_retries=len(row["attempts"]),
            )
        return True
    except BaseException:
        if moved is not None and moved.exists():
            if output.exists():
                raise ValueError("startup archive rollback refused an occupied output path") from None
            moved.rename(output)
        raise
