"""Shared saved-input builders and fixtures; never a collected test module."""

import json
import sys
from pathlib import Path

import pytest

from scripts.onestep_avatar.execution import process_registry, supervision
from scripts.onestep_avatar.execution import processes as queue_processes
from scripts.onestep_avatar.execution import queue_launch as launch
from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, LAUNCH_PROTOCOL, TOKEN_ENV
from scripts.onestep_avatar.tests.support.tokens import token


def register(path: Path, bootstrap: dict) -> dict:
    request = launch.load_request(path)
    row = {
        "state": "running",
        "sha256": request["job_sha256"],
        "command": request["command"],
        "launch_protocol": LAUNCH_PROTOCOL,
        "launch_request": str(path.resolve()),
        "launch_request_sha256": bootstrap["request_sha256"],
        "child_pid": bootstrap["identity"]["pid"],
        "child_identity": bootstrap["identity"],
        "environment_changes": {TOKEN_ENV: request["token"], JOB_ENV: request["job_sha256"]},
    }
    state = {"schema_version": 1, "owner_pid": request["owner_identity"]["pid"], "jobs": {"case": row}}
    Path(request["journal"]).write_text(json.dumps(state))
    return state


def ended_owned_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """A completed two-rank workload whose exact original subreaper has ended."""
    owner = {"pid": 2147483601, "start_ticks": 10, "command": ["original-owner"], "terminal": False}
    child = {
        "pid": 2147483602,
        "start_ticks": 11,
        "command": [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "2",
            "-m",
            "scripts.onestep_avatar.train",
        ],
        "terminal": False,
    }
    ranks = [
        {"pid": 2147483603 + rank, "start_ticks": 12, "command": ["original-rank", str(rank)], "terminal": False}
        for rank in range(2)
    ]
    request = launch.prepare_request(
        tmp_path / "launch",
        token=token(),
        job_sha256="a" * 64,
        owner_identity=owner,
        job_id="case",
        journal=tmp_path / "journal.json",
        command=child["command"],
    )
    bootstrap = {
        "schema_version": 1,
        "token": token(),
        "job_sha256": "a" * 64,
        "request_sha256": launch.digest(request),
        "identity": {**child, "command": launch.guard_command(request)},
    }
    for name in ("bootstrap.json", "grant.json"):
        (request.parent / name).write_text(json.dumps(bootstrap))
    notifications = tmp_path / "phases.json"
    changes = supervision.prepare_notifications(
        notifications, token=token(), job_sha256="a" * 64, world=2, phases=["load", "export:0"], budget_sha256="c" * 64
    )
    changes.update({TOKEN_ENV: token(), JOB_ENV: "a" * 64})
    for rank, identity in enumerate(ranks):
        monkeypatch.setattr(queue_processes, "process_identity", lambda _pid, identity=identity: identity)
        for phase in ("load", "export:0"):
            for event in ("begin", "end"):
                supervision.notify_phase(
                    phase, event, rank, budget_sha256="c" * 64, environment={**changes, "RANK": str(rank)}
                )
    state = register(request, bootstrap)
    ledger = tmp_path / "processes.json"
    row = state["jobs"]["case"]
    row.update(
        owner_pid=owner["pid"],
        gpus=[0, 1],
        process_ledger=str(ledger),
        environment_changes=changes,
        supervision_contract=str(notifications),
        attempt_started_ticks=9,
    )
    row["attempts"] = [json.loads(json.dumps(row))]
    (tmp_path / "journal.json").write_text(json.dumps(state))
    owned = {
        "state": "active",
        "owner": owner,
        "child_pid": child["pid"],
        "gpus": [0, 1],
        "job": "case",
        "attempt_started_ticks": 9,
        "baseline": [],
        "containment": "linux_subreaper_v1",
        "processes": {
            str(identity["pid"]): {"identity": identity, "roles": [role], "commands": [identity["command"]]}
            for identity, role in [(child, "child"), *[(rank, "descendant") for rank in ranks]]
        },
        "reported_rank_identities": [],
        "observations": [
            {"complete": True, "error": None, "identities": [child, *ranks], "schema_version": 1, "workers_live": True}
        ],
    }
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempts": {
                    token(): owned,
                    "unrelated-history": {"state": "closed", "preserved": "original unrelated bytes"},
                },
            }
        )
    )
    monkeypatch.setattr(queue_processes, "process_identity", lambda _pid: None)
    monkeypatch.setattr(process_registry, "gpu_memory", lambda **_kwargs: dict.fromkeys(range(8), 0))
    return ledger, request, owned, ranks, notifications
