"""Queue owner for managed execution; see doc/execution/queue.md."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.onestep_avatar.execution import attempt as attempt_runner
from scripts.onestep_avatar.execution import clock as execution_clock
from scripts.onestep_avatar.execution import completion as evidence
from scripts.onestep_avatar.execution import jobs as job_contracts
from scripts.onestep_avatar.execution import journal as state_store
from scripts.onestep_avatar.execution.attempt import (
    retry_startup_failure,
    run_child,
    startup_retry_evidence,
)
from scripts.onestep_avatar.execution.clock import (
    OverallDeadlineError,
    QueueChildError,
    begin_execution_clock,
    remaining_deadline,
    validate_execution_clock,
)
from scripts.onestep_avatar.execution.completion import (
    completion_receipt,
    read_training_marker,
    ready_jobs,
    verify_attempt_evidence,
    verify_completion,
    verify_decoder_completion,
    verify_receipt,
)
from scripts.onestep_avatar.execution.devices import (
    GPU_POOL,
    TRAIN_PROCESSES,
    parse_gpu_memory,
    validate_pool,
)

# Public import compatibility; internal execution uses each defining owner.
from scripts.onestep_avatar.execution.jobs import (
    EXPERIMENTS,
    ORDINARY,
    execution_profile,
    experiment_entry,
    job_command,
    prepare_job,
    prepare_jobs,
    read_training_launch,
    training_launch_record,
    validate_completion_descriptor,
    validate_experiment_job,
    validate_job_list,
    verify_training_launch,
)
from scripts.onestep_avatar.execution.journal import (
    queue_state,
    read_queue_state,
)
from scripts.onestep_avatar.execution.processes import (
    inspect_child,
    live_attempt_processes,
    live_session_processes,
    owned_workers_live,
    process_alive,
    process_identity,
)
from scripts.onestep_avatar.execution.recovery import (
    missing_supervision_recovery,
    recover_interrupted_jobs,
)

if TYPE_CHECKING:
    import argparse

__all__ = [
    "EXPERIMENTS",
    "GPU_POOL",
    "ORDINARY",
    "TRAIN_PROCESSES",
    "OverallDeadlineError",
    "QueueChildError",
    "begin_execution_clock",
    "completion_receipt",
    "dispatch_ready",
    "execute_jobs",
    "execution_profile",
    "experiment_entry",
    "inspect_child",
    "job_command",
    "live_attempt_processes",
    "live_session_processes",
    "main",
    "missing_supervision_recovery",
    "owned_workers_live",
    "parse_gpu_memory",
    "prepare_job",
    "prepare_jobs",
    "process_alive",
    "process_identity",
    "queue_state",
    "read_queue_state",
    "read_training_launch",
    "read_training_marker",
    "ready_jobs",
    "recover_interrupted_jobs",
    "recover_jobs",
    "remaining_deadline",
    "retry_startup_failure",
    "run_child",
    "startup_retry_evidence",
    "training_launch_record",
    "validate_completion_descriptor",
    "validate_execution_clock",
    "validate_experiment_job",
    "validate_job_list",
    "verify_attempt_evidence",
    "verify_completion",
    "verify_decoder_completion",
    "verify_receipt",
    "verify_training_launch",
]


def dispatch_ready(
    jobs: list[dict],
    state: dict,
    state_path: Path,
    claims_dir: Path,
    *,
    gpu_pool: tuple[int, ...] = GPU_POOL,
    foreground: bool = False,
    startup_retries: bool = True,
) -> bool:
    """Refresh device availability and run at most one eligible reserved child."""
    ready = evidence.ready_jobs(jobs, state)
    if not ready:
        return False
    from scripts.onestep_avatar.execution.process_registry import gpu_memory  # noqa: PLC0415

    def inventory() -> dict[int, int]:
        return gpu_memory(timeout=10)

    memory = inventory()
    for job in ready:
        row = state["jobs"][job["id"]]
        try:
            execution_clock.remaining_deadline(row)
        except execution_clock.OverallDeadlineError:
            with state_store.queue_state(state_path, jobs) as journal:
                journal["jobs"][job["id"]].update(
                    state="failed",
                    error="overall_deadline: original job deadline expired before retry",
                    failure_reason="overall_deadline",
                    finished_at=time.time(),
                )
            return True
        from scripts.onestep_avatar.execution.process_registry import ProcessRegistry  # noqa: PLC0415 -- current policy

        claims = ProcessRegistry(claims_dir)
        selection_options = {} if gpu_pool == GPU_POOL else {"pool": gpu_pool}
        gpus = claims.choose(memory, training=job["kind"] == "train", **selection_options)
        if gpus is None or not claims.acquire(gpus, job=job["id"], training=job["kind"] == "train"):
            continue
        # Claims cannot prevent an external process from starting between
        # the first observation and our reservation.
        try:
            memory = inventory()
        except BaseException:
            claims.release()
            raise
        if any(memory[gpu] >= 1024 for gpu in gpus):
            claims.release()
            continue
        log = state_path.parent / "logs" / f"{claims.token}.log"
        if foreground:
            print(  # noqa: T201 -- requested CLI output
                f"Starting {job['id']} on GPUs {','.join(map(str, gpus))}; log: {log.resolve()}", flush=True
            )
        try:
            receipts = {
                dependency: json.loads(json.dumps(state["jobs"][dependency]["receipt"]))
                for dependency in job.get("dependencies", [])
            }
            options = {"foreground": True} if foreground else {}
            attempt_runner.run_child(
                job, claims, log, state_path=state_path, jobs=jobs, dependency_receipts=receipts, **options
            )
        except execution_clock.QueueChildError:
            if not startup_retries or not attempt_runner.retry_startup_failure(job, jobs, state_path):
                raise
            return True
        return True
    return False


def execute_jobs(  # noqa: PLR0912 -- append-only dispatch and failure gates
    jobs_path: Path,
    state_path: Path,
    claims_dir: Path,
    *,
    once: bool,
    poll_seconds: float = 30,
    gpu_pool: tuple[int, ...] = GPU_POOL,
    foreground: bool = False,
    startup_retries: bool = True,
) -> int:
    """Reread append-only work, verify dependencies, and wait without reserving devices."""
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("queue polling interval must be finite, positive and at most 60 seconds")
    validate_pool(gpu_pool)
    observed = []
    while True:
        jobs = job_contracts.prepare_jobs(jobs_path)
        if any(job["kind"] == "train" for job in jobs) and len(gpu_pool) < TRAIN_PROCESSES:
            raise ValueError("training GPU pool must contain at least four devices")
        identities = [(job["id"], job["sha256"]) for job in jobs]
        if identities[: len(observed)] != observed:
            raise ValueError("queue revisions may only append unchanged jobs")
        observed = identities
        state = state_store.read_queue_state(state_path, jobs)
        if not once:
            rows = [state["jobs"][job["id"]] for job in jobs]
            if any(row["state"] == "running" for row in rows):
                raise ValueError("queue running attempts require explicit recovery before loop execution")
            if any(row["state"] == "failed" for row in rows):
                if foreground:
                    print(  # noqa: T201 -- requested CLI output
                        "Queue stopped: a job failed; inspect the state file and child log.", flush=True
                    )
                return 1
            if all(row["state"] == "complete" for row in rows):
                for job in jobs:
                    receipt = state["jobs"][job["id"]].get("receipt")
                    if not isinstance(receipt, dict) or not evidence.verify_receipt(job, receipt):
                        raise ValueError("completed queue job has missing verified evidence")
                if foreground:
                    print(  # noqa: T201 -- requested CLI output
                        "Queue complete: all jobs verified.", flush=True
                    )
                return 0
        dispatch_options = {} if gpu_pool == GPU_POOL else {"gpu_pool": gpu_pool}
        if foreground:
            dispatch_options["foreground"] = True
        if not startup_retries:
            dispatch_options["startup_retries"] = False
        if dispatch_ready(jobs, state, state_path, claims_dir, **dispatch_options):
            if once:
                updated = state_store.read_queue_state(state_path, jobs)
                if any(row["state"] == "failed" for row in updated["jobs"].values()):
                    return 1
                return (
                    2
                    if any(
                        row["state"] == "pending" and row["attempts"] and row["attempts"][-1].get("retry_evidence")
                        for row in updated["jobs"].values()
                    )
                    else 0
                )
            continue
        if once:
            return 2 if evidence.ready_jobs(jobs, state) else 0
        if foreground:
            print(  # noqa: T201 -- requested CLI output
                f"Waiting for free GPUs or dependencies; checking again in {poll_seconds:g}s. Ctrl+C stops.", flush=True
            )
        time.sleep(poll_seconds)


def recover_jobs(jobs_path: Path, state_path: Path) -> dict:
    """Recover recorded terminated children without launching work or changing claims."""
    if not state_path.is_file():
        raise ValueError("queue recovery requires an existing saved journal")
    jobs = job_contracts.prepare_jobs(jobs_path)
    state = state_store.read_queue_state(state_path, jobs)
    recovery_needed = any(state["jobs"][job["id"]]["state"] == "running" for job in jobs)
    if recovery_needed:
        with state_store.queue_state(state_path, jobs, recover=True) as recovered:
            state = recovered
    return {
        "recovered": recovery_needed,
        "jobs": [{"id": job["id"], "state": state["jobs"][job["id"]]["state"]} for job in jobs],
    }


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Validate execution controls and choose the journal before reading any jobs."""
    import argparse  # noqa: PLC0415 -- lightweight queue CLI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--state", type=Path, help="journal path; default: state.json in the jobs directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="claim available GPUs and run ready jobs")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="execute with live child output and Ctrl+C cleanup; defaults to --loop",
    )
    parser.add_argument("--no-startup-retries", action="store_true", help="stop on the first child failure, including startup OOM")
    parser.add_argument("--recover", action="store_true", help="inspect terminated children and verify saved outputs")
    parser.add_argument("--process-ledger", type=Path, help="one shared JSON record of package-owned processes")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--once", action="store_true")
    execution.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30, help="loop wait interval, at most 60 seconds")
    parser.add_argument(
        "--gpu-pool",
        default=",".join(map(str, GPU_POOL)),
        help="comma-separated allowed physical GPU IDs for this owner",
    )
    args = parser.parse_args(argv)
    if args.state is None:
        args.state = args.jobs.parent / "state.json"
    try:
        gpu_pool = tuple(int(value) for value in args.gpu_pool.split(","))
        validate_pool(gpu_pool)
    except ValueError as error:
        parser.error(str(error))
    if sum((args.dry_run, args.execute, args.foreground, args.recover)) != 1:
        parser.error("choose exactly one of --dry-run, --execute, --foreground or --recover")
    if args.foreground:
        args.execute = True
        args.loop = not args.once
    if args.execute and (not (args.once or args.loop) or args.process_ledger is None):
        parser.error("--execute requires --once or --loop and --process-ledger")
    if (args.loop or args.once) and not args.execute:
        parser.error("--once and --loop require --execute")
    if not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 60:
        parser.error("--poll-seconds must be finite, positive and at most 60")
    args.gpu_pool = gpu_pool
    return args, parser


def main(argv: list[str] | None = None) -> int:
    """Review, dispatch once, or continuously execute normalized package-owned jobs."""
    args, parser = _parse_args(argv)
    gpu_pool, ledger = args.gpu_pool, args.process_ledger
    if args.recover:
        print(  # noqa: T201 -- requested CLI output
            json.dumps(recover_jobs(args.jobs, args.state), indent=2)
        )
        return 0
    if args.execute:
        from scripts.onestep_avatar.execution.terminal import stop_signals  # noqa: PLC0415 -- execution only

        with stop_signals() as received:
            try:
                if args.foreground:
                    print(  # noqa: T201 -- requested CLI output
                        f"Queue owner PID {os.getpid()}; state: {args.state.resolve()}. Ctrl+C stops owned work.",
                        flush=True,
                    )
                options = {"foreground": True} if args.foreground else {}
                if args.no_startup_retries:
                    options["startup_retries"] = False
                return execute_jobs(
                    args.jobs,
                    args.state,
                    ledger,
                    once=args.once,
                    poll_seconds=args.poll_seconds,
                    gpu_pool=gpu_pool,
                    **options,
                )
            except KeyboardInterrupt:
                print(  # noqa: T201 -- requested CLI output
                    "Queue stopped; inspect the saved state and logs for the interrupted attempt.", flush=True
                )
                return 128 + (received[0] if received else 2)
    jobs = job_contracts.prepare_jobs(args.jobs)
    if any(job["kind"] == "train" for job in jobs) and len(gpu_pool) < TRAIN_PROCESSES:
        parser.error("training GPU pool must contain at least four devices")
    state = state_store.read_queue_state(args.state, jobs)
    planned = []
    for job in jobs:
        gpus = gpu_pool[:TRAIN_PROCESSES] if job["kind"] == "train" else gpu_pool[:1]
        command, changes = job_contracts.job_command(job, gpus)
        planned.append(
            {
                "id": job["id"],
                "sha256": job["sha256"],
                "state": state["jobs"][job["id"]]["state"],
                "command": command,
                "environment_changes": changes,
                "planned_physical_gpus": list(gpus),
            }
        )
    print(  # noqa: T201 -- requested CLI output
        json.dumps(
            {"validation": "arguments_and_identity_only", "gpu_availability_checked": False, "jobs": planned}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
