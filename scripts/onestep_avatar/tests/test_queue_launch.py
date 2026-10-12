"""The real child remains inert until durable registration and launch approval."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

import pytest

from scripts.onestep_avatar.execution import attempt as queue_attempt
from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.execution import journal as queue_journal
from scripts.onestep_avatar.execution import process_registry, queue, runtime, supervision
from scripts.onestep_avatar.execution import processes as queue_processes
from scripts.onestep_avatar.execution import queue_launch as launch
from scripts.onestep_avatar.execution.process_registry import ProcessRegistry
from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, LAUNCH_PROTOCOL, TOKEN_ENV
from scripts.onestep_avatar.tests.support.fixtures_queue_launch import (
    ended_owned_attempt,
    register,
)
from scripts.onestep_avatar.tests.support.tokens import token
from scripts.onestep_avatar.training import resources


def request_for(tmp_path: Path, *, owner: dict | None = None) -> tuple[Path, Path]:
    executed = tmp_path / "executed.json"
    command = [
        sys.executable,
        "-c",
        "import os,json; from pathlib import Path; "
        "from scripts.onestep_avatar.execution.queue import process_identity; "
        f"Path({str(executed)!r}).write_text(json.dumps(process_identity(os.getpid())))",
    ]
    path = launch.prepare_request(
        tmp_path / "launch",
        token=token(),
        job_sha256="a" * 64,
        owner_identity=queue_processes.process_identity(os.getpid()) if owner is None else owner,
        job_id="case",
        journal=tmp_path / "journal.json",
        command=command,
    )
    return path, executed


def start_guard(
    path: Path, log: IO[bytes], *, timeout: int = 5, environment: dict[str, str] | None = None
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.onestep_avatar.execution.queue_launch",
            "--request",
            str(path),
            "--timeout",
            str(timeout),
            "--poll-seconds",
            "0.01",
        ],
        env={**os.environ, TOKEN_ENV: token(), JOB_ENV: "a" * 64, **(environment or {})},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_bootstrap(path: Path, child: subprocess.Popen) -> dict:
    deadline = time.monotonic() + 5
    while not (path.parent / "bootstrap.json").is_file():
        if child.poll() is not None:
            pytest.fail("guard exited before registration")
        if time.monotonic() >= deadline:
            pytest.fail("guard did not register")
        time.sleep(0.01)
    return json.loads((path.parent / "bootstrap.json").read_text())


def stop(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.terminate()
    child.wait(timeout=5)


def test_ended_owned_recovery_preserves_original_evidence_and_states_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, request, owned, ranks, _notifications = ended_owned_attempt(tmp_path, monkeypatch)
    original_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and path != ledger}
    result = launch.recover_owned_attempt(ledger, request, timeout=2)
    assert result["basis"] == "bounded_dead_owner_recovery"
    assert result["registered_workers_absent"] is True
    assert result["continuous_supervision"] is False
    assert result["containment_complete"] is False
    assert len(result["notifications"]["event_sha256"]) == 8
    assert [item["identity"] for item in result["notifications"]["rank_identities"]] == ranks
    record = json.loads(ledger.read_bytes())
    changed = record["attempts"][token()]
    assert {key: value for key, value in changed.items() if key not in ("state", "recovery")} == {
        key: value for key, value in owned.items() if key != "state"
    }
    assert changed["state"] == "closed"
    assert changed["recovery"] == result
    assert record["attempts"]["unrelated-history"] == {"state": "closed", "preserved": "original unrelated bytes"}
    assert all(path.read_bytes() == data for path, data in original_files.items())


def test_interrupted_recovery_releases_devices_without_claiming_the_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, request, owned, _ranks, notifications = ended_owned_attempt(tmp_path, monkeypatch)
    # The workload stopped during startup: no rank reported its phases.
    notifications.write_bytes(b"")
    with pytest.raises(ValueError, match="Expecting value"):
        launch.recover_owned_attempt(ledger, request, timeout=2)
    assert json.loads(ledger.read_bytes())["attempts"][token()]["state"] == "active"
    result = launch.recover_owned_attempt(ledger, request, timeout=2, interrupted=True)
    assert result["basis"] == "bounded_interrupted_owner_recovery"
    assert result["workload_complete"] is False
    assert result["notifications"] is None
    assert result["registered_workers_absent"] is True
    assert result["containment_complete"] is False
    changed = json.loads(ledger.read_bytes())["attempts"][token()]
    assert changed["state"] == "closed"
    assert {key: value for key, value in changed.items() if key not in ("state", "recovery")} == {
        key: value for key, value in owned.items() if key != "state"
    }
    observed = process_registry.inspect_record(ledger, token())
    assert observed["basis"] == "recovered_interrupted_handles_ended"
    assert observed["workload_complete"] is False
    assert observed["workers_live"] is False


def test_interrupted_recovery_keeps_live_handle_busy_gpu_and_running_journal_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, request, _owned, ranks, _notifications = ended_owned_attempt(tmp_path, monkeypatch)
    original = ledger.read_bytes()
    monkeypatch.setattr(queue_processes, "process_identity", lambda pid: ranks[0] if pid == ranks[0]["pid"] else None)
    with pytest.raises(ValueError, match="registered handle remains live"):
        launch.recover_owned_attempt(ledger, request, timeout=2, interrupted=True)
    monkeypatch.setattr(queue_processes, "process_identity", lambda _pid: None)
    monkeypatch.setattr(process_registry, "gpu_memory", lambda **_kwargs: {**dict.fromkeys(range(8), 0), 1: 4096})
    with pytest.raises(ValueError, match="idle selected GPU"):
        launch.recover_owned_attempt(ledger, request, timeout=2, interrupted=True)
    assert ledger.read_bytes() == original
    journal = Path(json.loads(request.read_bytes())["journal"])
    state = json.loads(journal.read_bytes())
    state["jobs"]["case"]["state"] = "failed"
    journal.write_text(json.dumps(state))
    monkeypatch.setattr(process_registry, "gpu_memory", lambda **_kwargs: dict.fromkeys(range(8), 0))
    with pytest.raises(ValueError, match="journal does not bind"):
        launch.recover_owned_attempt(ledger, request, timeout=2, interrupted=True)
    assert ledger.read_bytes() == original


@pytest.mark.parametrize(
    "failure",
    [
        "missing_rank_end",
        "changed_rank",
        "wrong_contract_hash",
        "missing_contract_hash",
        "extra_event",
        "wrong_world",
        "live_owner",
        "live_child",
        "live_rank",
        "reused_rank",
        "denied_rank",
        "uncontained_rank",
        "incomplete_containment",
        "busy_gpu",
        "incomplete_gpu",
        "invalid_gpu",
        "wrong_ledger",
        "changed_attempt",
        "unapproved_grant",
        "expired",
    ],
)
def test_ended_owned_recovery_refuses_uncertainty_without_ledger_mutation(  # noqa: PLR0912, PLR0915 -- one complete fixture lifecycle with all assertions and cleanup
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    ledger, request, owned, ranks, notifications = ended_owned_attempt(tmp_path, monkeypatch)
    state_path = tmp_path / "journal.json"
    state = json.loads(state_path.read_bytes())
    record = json.loads(ledger.read_bytes())
    changed = record["attempts"][token()]
    events = notifications.with_name(notifications.name + ".events")
    if failure == "missing_rank_end":
        (events / "rank0001.phase000001.end.json").unlink()
    elif failure == "changed_rank":
        target = events / "rank0001.phase000001.end.json"
        event = json.loads(target.read_bytes())
        event["identity"]["start_ticks"] += 1
        target.write_text(json.dumps(event))
    elif failure in ("wrong_contract_hash", "missing_contract_hash"):
        for item in (state["jobs"]["case"], state["jobs"]["case"]["attempts"][-1]):
            item["environment_changes"][supervision.SUPERVISION_SHA_ENV] = (
                "d" * 64 if failure == "wrong_contract_hash" else None
            )
    elif failure == "extra_event":
        (events / "unknown.json").write_text("{}")
    elif failure == "wrong_world":
        original = json.loads(request.read_bytes())
        original["command"][4] = "3"
        request.write_text(json.dumps(original))
    elif failure in ("live_owner", "live_child", "live_rank", "reused_rank", "denied_rank"):
        identity = (
            owned["owner"]
            if failure == "live_owner"
            else owned["processes"][str(owned["child_pid"])]["identity"]
            if failure == "live_child"
            else ranks[0]
        )

        def observe(pid: int) -> dict | None:
            if pid != identity["pid"]:
                return None
            if failure == "denied_rank":
                raise PermissionError("controlled exact-handle denial")
            return {**identity, "start_ticks": identity["start_ticks"] + int(failure == "reused_rank")}

        monkeypatch.setattr(queue_processes, "process_identity", observe)
    elif failure == "uncontained_rank":
        del changed["processes"][str(ranks[1]["pid"])]
        changed["reported_rank_identities"].append({"identity": ranks[1], "role": "rank:1"})
    elif failure == "incomplete_containment":
        changed["observations"][0]["complete"] = False
    elif failure in ("busy_gpu", "incomplete_gpu", "invalid_gpu"):
        memory = dict.fromkeys(range(8), 0)
        if failure == "busy_gpu":
            memory[1] = 1024
        elif failure == "incomplete_gpu":
            del memory[1]
        else:
            memory[1] = True
        monkeypatch.setattr(process_registry, "gpu_memory", lambda **_kwargs: memory)
    elif failure == "wrong_ledger":
        for item in (state["jobs"]["case"], state["jobs"]["case"]["attempts"][-1]):
            item["process_ledger"] = str(tmp_path / "another.json")
    elif failure == "changed_attempt":
        state["jobs"]["case"]["attempts"][-1]["attempt_started_ticks"] += 1
    elif failure == "unapproved_grant":
        (request.parent / "grant.json").unlink()
    state_path.write_text(json.dumps(state))
    ledger.write_text(json.dumps(record))
    original_bytes = ledger.read_bytes()
    with pytest.raises((ValueError, OSError, TimeoutError)):
        launch.recover_owned_attempt(ledger, request, timeout=1e-12 if failure == "expired" else 2)
    assert ledger.read_bytes() == original_bytes


def test_ended_owned_recovery_rechecks_handles_after_direct_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, request, _owned, ranks, _notifications = ended_owned_attempt(tmp_path, monkeypatch)
    original_bytes = ledger.read_bytes()

    def inventory(**kwargs: object) -> dict[int, int]:
        del kwargs
        monkeypatch.setattr(
            queue_processes, "process_identity", lambda pid: ranks[0] if pid == ranks[0]["pid"] else None
        )
        return dict.fromkeys(range(8), 0)

    monkeypatch.setattr(process_registry, "gpu_memory", inventory)
    with pytest.raises(ValueError, match="remains live"):
        launch.recover_owned_attempt(ledger, request, timeout=2)
    assert ledger.read_bytes() == original_bytes


@pytest.mark.parametrize("special", ["fifo", "oversized"])
def test_owned_recovery_rejects_nonregular_or_oversized_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, special: str
) -> None:
    ledger, request, _owned, _ranks, notifications = ended_owned_attempt(tmp_path, monkeypatch)
    target = notifications.with_name(notifications.name + ".events") / "rank0001.phase000001.end.json"
    target.unlink()
    if special == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(b" " * 65537)
    original_bytes = ledger.read_bytes()
    started = time.monotonic()
    with pytest.raises(ValueError, match="bounded regular file"):
        launch.recover_owned_attempt(ledger, request, timeout=2)
    assert time.monotonic() - started < 1
    assert ledger.read_bytes() == original_bytes


def approved_direct_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """The actual one-rank package route with independent saved runtime bindings."""
    ledger, request, owned, _ranks, _notifications = ended_owned_attempt(tmp_path, monkeypatch)
    original = json.loads(request.read_bytes())
    command = [
        sys.executable,
        "-m",
        "scripts.onestep_avatar.experiments.prune_baseline",
        "--spec",
        str(tmp_path / "spec.json"),
        "--output",
        str(tmp_path / "output"),
        "--gpu-id",
        "0",
    ]
    original["command"] = command
    request.write_text(json.dumps(original))
    bootstrap = json.loads((request.parent / "bootstrap.json").read_bytes())
    bootstrap["request_sha256"] = launch.digest(request)
    for name in ("bootstrap.json", "grant.json"):
        (request.parent / name).write_text(json.dumps(bootstrap))
    child = {**owned["processes"][str(owned["child_pid"])]["identity"], "command": command}
    notifications, contract = tmp_path / "direct-phases.json", tmp_path / "runtime.json"
    budget = resources.default_budget()
    changes = supervision.prepare_notifications(
        notifications,
        token=original["token"],
        job_sha256=original["job_sha256"],
        world=1,
        phases=["runtime"],
        budget_sha256=budget["sha256"],
    )
    changes.update(
        runtime.prepare_contract(
            contract,
            token=original["token"],
            job_sha256=original["job_sha256"],
            budget=budget,
            generic_phase=True,
            numerical_policy=None,
        )
    )
    changes.update(
        {
            **resources.budget_environment(budget),
            TOKEN_ENV: original["token"],
            JOB_ENV: original["job_sha256"],
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    runtime.result_path(contract).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_sha256": changes[runtime.CONTRACT_SHA_ENV],
                "token": original["token"],
                "job_sha256": original["job_sha256"],
                "state": "passed",
                "error": None,
                "numerical_policy_requested": None,
                "numerical_policy_observed": None,
                "resource_records": [
                    {
                        "schema_version": 1,
                        "phase": "runtime",
                        "rank": 0,
                        "device": "cuda:0",
                        "elapsed_s": 0.01,
                        "peak_allocated_bytes": 100,
                        "peak_reserved_bytes": 200,
                        "budget_sha256": budget["sha256"],
                        "state": "passed",
                        "error": None,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(queue_processes, "process_identity", lambda _pid: child)
    for event in ("begin", "end"):
        supervision.notify_phase("runtime", event, 0, budget_sha256=budget["sha256"], environment=changes)
    state = register(request, bootstrap)
    row = state["jobs"]["case"]
    row.update(
        owner_pid=owned["owner"]["pid"],
        gpus=[0],
        process_ledger=str(ledger),
        environment_changes=changes,
        supervision_contract=str(notifications),
        attempt_started_ticks=9,
        runtime_contract=str(contract),
        runtime_contract_sha256=changes[runtime.CONTRACT_SHA_ENV],
    )
    row["attempts"] = [json.loads(json.dumps(row))]
    (tmp_path / "journal.json").write_text(json.dumps(state))
    owned.update(
        gpus=[0],
        processes={
            str(child["pid"]): {
                "identity": child,
                "roles": ["child"],
                "commands": [command],
            }
        },
        observations=[
            {"schema_version": 1, "complete": True, "error": None, "identities": [child], "workers_live": True}
        ],
    )
    record = json.loads(ledger.read_bytes())
    record["attempts"][original["token"]] = owned
    ledger.write_text(json.dumps(record))
    monkeypatch.setattr(queue_processes, "process_identity", lambda _pid: None)
    return ledger, request, owned, notifications, contract


def test_approved_direct_recovery_preserves_science_and_lost_supervision_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, request, owned, _notifications, contract = approved_direct_attempt(tmp_path, monkeypatch)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and path != ledger}
    result = launch.recover_owned_attempt(ledger, request, timeout=2)
    assert result["notifications"]["world"] == 1
    assert result["notifications"]["phases"] == ["runtime"]
    assert result["direct_runtime"]["runtime_result_sha256"] == launch.digest(runtime.result_path(contract))
    assert result["direct_runtime"]["child_measurements_checked"] is True
    assert result["direct_runtime"]["supervision_acceptance"] is False
    assert result["continuous_supervision"] is False
    assert result["unknown_descendants_unproven"] is True
    assert result["original_exitcode"] is None
    closed = json.loads(ledger.read_bytes())["attempts"][token()]
    assert closed["state"] == "closed"
    assert {key: value for key, value in closed.items() if key not in ("state", "recovery")} == {
        key: value for key, value in owned.items() if key != "state"
    }
    assert all(path.read_bytes() == data for path, data in original.items())
    with pytest.raises(ValueError, match="ledger differs"):
        launch.recover_owned_attempt(ledger, request, timeout=2)


@pytest.mark.parametrize(
    "failure",
    [
        "unsupported_module",
        "training_module",
        "nested_module",
        "wrong_gpu",
        "duplicate_gpu",
        "nested_execute",
        "direct_rank_count",
        "relative_python",
        "wrong_physical_gpu",
        "runtime_environment",
        "runtime_attempt",
        "runtime_hash",
        "runtime_failed",
        "runtime_missing",
        "result_missing",
        "runtime_budget",
        "notification_budget",
        "notification_phase",
        "world",
        "rank_child",
        "live_child",
        "busy_gpu",
        "runtime_changed_during_inventory",
    ],
)
def test_approved_direct_recovery_refuses_uncertain_contracts_without_publication(  # noqa: PLR0912, PLR0915 -- complete tamper lifecycle
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    ledger, request, owned, notifications, contract = approved_direct_attempt(tmp_path, monkeypatch)
    state_path = tmp_path / "journal.json"
    state, record = json.loads(state_path.read_bytes()), json.loads(ledger.read_bytes())
    row, attempt = state["jobs"]["case"], state["jobs"]["case"]["attempts"][-1]
    if failure in (
        "unsupported_module",
        "training_module",
        "nested_module",
        "wrong_gpu",
        "duplicate_gpu",
        "nested_execute",
        "direct_rank_count",
        "relative_python",
    ):
        requested = json.loads(request.read_bytes())
        command = requested["command"]
        if failure == "unsupported_module":
            command[2] = "unreviewed.owner"
        elif failure == "training_module":
            command[2] = "scripts.onestep_avatar.train"
        elif failure == "nested_module":
            command[3:3] = ["-m", "nested.owner"]
        elif failure == "wrong_gpu":
            command[-1] = "1"
        elif failure == "duplicate_gpu":
            command[3:3] = ["--gpu-id=0"]
        elif failure == "nested_execute":
            command[3:3] = ["--execute=yes"]
        elif failure == "direct_rank_count":
            command[3:3] = ["--num_processes", "1"]
        else:
            command[0] = "python"
        request.write_text(json.dumps(requested))
        for name in ("bootstrap.json", "grant.json"):
            saved = json.loads((request.parent / name).read_bytes())
            saved["request_sha256"] = launch.digest(request)
            (request.parent / name).write_text(json.dumps(saved))
        for item in (row, attempt):
            item.update(command=command, launch_request_sha256=launch.digest(request))
    elif failure == "wrong_physical_gpu":
        for item in (row, attempt):
            item["environment_changes"]["CUDA_VISIBLE_DEVICES"] = "1"
    elif failure == "runtime_environment":
        for item in (row, attempt):
            item["environment_changes"][runtime.CONTRACT_ENV] = str(tmp_path / "elsewhere.json")
    elif failure == "runtime_attempt":
        attempt["runtime_contract_sha256"] = "d" * 64
    elif failure == "runtime_hash":
        contract.write_bytes(contract.read_bytes() + b" ")
    elif failure == "runtime_failed":
        saved = json.loads(runtime.result_path(contract).read_bytes())
        saved.update(state="failed", error="controlled failure")
        runtime.result_path(contract).write_text(json.dumps(saved))
    elif failure in ("runtime_missing", "result_missing"):
        (contract if failure == "runtime_missing" else runtime.result_path(contract)).unlink()
    elif failure == "runtime_budget":
        for item in (row, attempt):
            item["environment_changes"].pop(resources.BUDGET_ENV)
    elif failure in ("notification_budget", "notification_phase", "world"):
        saved = json.loads(notifications.read_bytes())
        saved[{"notification_budget": "budget_sha256", "notification_phase": "phases", "world": "world"}[failure]] = (
            "d" * 64 if failure == "notification_budget" else ["another"] if failure == "notification_phase" else 2
        )
        notifications.write_text(json.dumps(saved))
        for item in (row, attempt):
            item["environment_changes"][supervision.SUPERVISION_SHA_ENV] = launch.digest(notifications)
    elif failure == "rank_child":
        child = owned["processes"][str(owned["child_pid"])]["identity"]
        rank = {**child, "pid": child["pid"] + 1}
        record["attempts"][token()]["processes"][str(rank["pid"])] = {
            "identity": rank,
            "roles": ["descendant"],
            "commands": [rank["command"]],
        }
        record["attempts"][token()]["observations"][0]["identities"].append(rank)
        for target in notifications.with_name(notifications.name + ".events").glob("*.json"):
            saved = json.loads(target.read_bytes())
            saved["identity"] = rank
            target.write_text(json.dumps(saved))
    elif failure == "live_child":
        child = owned["processes"][str(owned["child_pid"])]["identity"]
        monkeypatch.setattr(queue_processes, "process_identity", lambda pid: child if pid == child["pid"] else None)
    elif failure == "busy_gpu":
        monkeypatch.setattr(process_registry, "gpu_memory", lambda **_kwargs: {0: 1024})
    else:

        def inventory(**_kwargs: object) -> dict:
            contract.write_bytes(contract.read_bytes() + b" ")
            return dict.fromkeys(range(8), 0)

        monkeypatch.setattr(process_registry, "gpu_memory", inventory)
    state_path.write_text(json.dumps(state))
    ledger.write_text(json.dumps(record))
    original = ledger.read_bytes()
    with pytest.raises((ValueError, OSError)):
        launch.recover_owned_attempt(ledger, request, timeout=2)
    assert ledger.read_bytes() == original


def direct_registered_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    ledger, _request, owned, _ranks, _notifications = ended_owned_attempt(tmp_path, monkeypatch)
    child = owned["processes"][str(owned["child_pid"])]
    command = [
        sys.executable,
        "-m",
        "scripts.onestep_avatar.experiments.adapter_effect_check",
        "--output",
        str(tmp_path / "scientific-result"),
        "--gpu-id",
        "0",
        "--decode",
    ]
    child["identity"]["command"] = command
    child["commands"] = [command]
    owned["processes"] = {str(owned["child_pid"]): child}
    owned["observations"][0]["identities"] = [child["identity"]]
    record = json.loads(ledger.read_bytes())
    record["attempts"][token()] = owned
    ledger.write_text(json.dumps(record))
    launch_path = tmp_path / "direct-launch.json"
    launch_path.write_text(
        json.dumps(
            {
                "command": command,
                "physical_gpu": 0,
                "visible_gpu": 0,
                "environment_changes": {TOKEN_ENV: token(), JOB_ENV: "a" * 64, "CUDA_VISIBLE_DEVICES": "0"},
            }
        )
    )
    # Existing own attempt is one selected physical device.
    owned["gpus"] = [0]
    record["attempts"][token()] = owned
    ledger.write_text(json.dumps(record))
    return ledger, launch_path, owned


def registered_recovery(ledger: Path, launch_path: Path, owned: dict, **changes: object) -> dict:
    row_hash = changes.pop("expected_row_sha256") if "expected_row_sha256" in changes else launch._row_digest(owned)
    launch_hash = (
        changes.pop("expected_launch_sha256") if "expected_launch_sha256" in changes else launch.digest(launch_path)
    )
    return launch.recover_registered_attempt(
        ledger,
        token(),
        expected_row_sha256=row_hash,
        launch_evidence=launch_path,
        expected_launch_sha256=launch_hash,
        timeout=changes.pop("timeout", 2),
        **changes,
    )


def test_registered_nontraining_recovery_preserves_original_row_and_limited_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, launch_path, owned = direct_registered_attempt(tmp_path, monkeypatch)
    original_launch = launch_path.read_bytes()
    result = registered_recovery(ledger, launch_path, owned)
    assert result["basis"] == "registered_handle_bookkeeping_recovery"
    assert result["registered_processes_absent"] is True
    assert result["continuous_supervision"] is False
    assert result["containment_complete"] is False
    assert result["unknown_descendants_unproven"] is True
    assert result["original_exitcode"] is None
    assert "workers_absent" not in result
    assert "scientific_acceptance" not in result
    record = json.loads(ledger.read_bytes())
    recovered = record["attempts"][token()]
    assert recovered == {**owned, "state": "closed", "recovery": result}
    assert record["attempts"]["unrelated-history"] == {"state": "closed", "preserved": "original unrelated bytes"}
    assert launch_path.read_bytes() == original_launch


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_row_hash",
        "wrong_launch_hash",
        "missing_row_hash",
        "missing_launch_hash",
        "wrong_token",
        "train",
        "accelerate",
        "unsupported_module",
        "supervise",
        "multiple_gpus",
        "wrong_gpu_mapping",
        "boolean_gpu",
        "rank_role",
        "reported_rank",
        "uncontained_descendant",
        "extra_child",
        "incomplete_containment",
        "bad_identity",
        "live_owner",
        "live_child",
        "reused_child",
        "denied_child",
        "busy_gpu",
        "missing_gpu",
        "invalid_gpu",
        "launch_symlink",
        "ledger_symlink",
        "nonregular_launch",
        "expired",
    ],
)
def test_registered_nontraining_recovery_refuses_ambiguous_evidence(  # noqa: PLR0912, PLR0915 -- one complete fixture lifecycle with all assertions and cleanup
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    ledger, launch_path, owned = direct_registered_attempt(tmp_path, monkeypatch)
    saved_launch = json.loads(launch_path.read_bytes())
    changes = {}
    child = owned["processes"][str(owned["child_pid"])]
    if failure in ("wrong_row_hash", "missing_row_hash"):
        changes["expected_row_sha256"] = "d" * 64 if failure == "wrong_row_hash" else None
    elif failure in ("wrong_launch_hash", "missing_launch_hash"):
        changes["expected_launch_sha256"] = "d" * 64 if failure == "wrong_launch_hash" else None
    elif failure == "wrong_token":
        saved_launch["environment_changes"][TOKEN_ENV] = "d" * 32
    elif failure in ("train", "accelerate", "unsupported_module", "supervise"):
        command = child["identity"]["command"]
        if failure == "train":
            command[2] = "scripts.onestep_avatar.train"
        elif failure == "accelerate":
            command[2] = "accelerate.commands.launch"
        elif failure == "unsupported_module":
            command[2] = "scripts.onestep_avatar.execution.queue"
        else:
            command.append("--supervise")
        saved_launch["command"] = command
    elif failure == "multiple_gpus":
        owned["gpus"] = [0, 1]
    elif failure == "wrong_gpu_mapping":
        saved_launch["environment_changes"]["CUDA_VISIBLE_DEVICES"] = "1"
    elif failure == "boolean_gpu":
        saved_launch["physical_gpu"] = False
    elif failure == "rank_role":
        child["roles"].append("rank:0")
    elif failure == "reported_rank":
        owned["reported_rank_identities"] = [{"identity": child["identity"], "role": "rank:0"}]
    elif failure in ("uncontained_descendant", "extra_child"):
        identity = {**child["identity"], "pid": child["identity"]["pid"] + 1, "command": ["contained-worker"]}
        owned["processes"][str(identity["pid"])] = {
            "identity": identity,
            "roles": ["child" if failure == "extra_child" else "descendant"],
            "commands": [identity["command"]],
        }
    elif failure == "incomplete_containment":
        owned["observations"][0]["complete"] = False
    elif failure == "bad_identity":
        child["identity"]["start_ticks"] = True
    elif failure in ("live_owner", "live_child", "reused_child", "denied_child"):
        identity = owned["owner"] if failure == "live_owner" else child["identity"]

        def observe(pid: int) -> dict | None:
            if pid != identity["pid"]:
                return None
            if failure == "denied_child":
                raise PermissionError("controlled registered-handle denial")
            return {**identity, "start_ticks": identity["start_ticks"] + int(failure == "reused_child")}

        monkeypatch.setattr(queue_processes, "process_identity", observe)
    elif failure in ("busy_gpu", "missing_gpu", "invalid_gpu"):
        memory = dict.fromkeys(range(8), 0)
        if failure == "busy_gpu":
            memory[0] = 1024
        elif failure == "missing_gpu":
            del memory[0]
        else:
            memory[0] = True
        monkeypatch.setattr(process_registry, "gpu_memory", lambda **_kwargs: memory)
    elif failure == "expired":
        changes["timeout"] = 1e-12
    record = json.loads(ledger.read_bytes())
    record["attempts"][token()] = owned
    ledger.write_text(json.dumps(record))
    launch_path.write_text(json.dumps(saved_launch))
    if failure in ("launch_symlink", "ledger_symlink"):
        target = launch_path if failure == "launch_symlink" else ledger
        other = target.with_name(target.name + ".original")
        target.rename(other)
        target.symlink_to(other)
    elif failure == "nonregular_launch":
        original_hash = launch.digest(launch_path)
        launch_path.unlink()
        os.mkfifo(launch_path)
        changes["expected_launch_sha256"] = original_hash
    original_ledger = ledger.read_bytes()
    with pytest.raises((ValueError, OSError, TimeoutError)):
        registered_recovery(ledger, launch_path, owned, **changes)
    assert ledger.read_bytes() == original_ledger


@pytest.mark.parametrize("mutation", ["launch", "row", "live_child"])
def test_registered_recovery_rechecks_original_handles_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    ledger, launch_path, owned = direct_registered_attempt(tmp_path, monkeypatch)
    row_hash, launch_hash = launch._row_digest(owned), launch.digest(launch_path)

    def inventory(**kwargs: object) -> dict[int, int]:
        del kwargs
        if mutation == "launch":
            launch_path.write_text("{}")
        elif mutation == "row":
            record = json.loads(ledger.read_bytes())
            record["attempts"][token()]["job"] = "changed"
            ledger.write_text(json.dumps(record))
        else:
            identity = owned["processes"][str(owned["child_pid"])]["identity"]
            monkeypatch.setattr(
                queue_processes, "process_identity", lambda pid: identity if pid == identity["pid"] else None
            )
        return dict.fromkeys(range(8), 0)

    monkeypatch.setattr(process_registry, "gpu_memory", inventory)
    with pytest.raises(ValueError):  # noqa: PT011 -- mutation controls check refusal type and preserved bytes across different validators
        registered_recovery(
            ledger, launch_path, owned, expected_row_sha256=row_hash, expected_launch_sha256=launch_hash
        )
    row = json.loads(ledger.read_bytes())["attempts"][token()]
    assert row["state"] == "active"
    assert "recovery" not in row


def test_registered_recovery_covers_ended_real_package_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = tmp_path / "real-processes.json"
    identity_file = tmp_path / "token.json"
    owner_source = (
        "import json, subprocess, sys\nfrom pathlib import Path\n"
        "from scripts.onestep_avatar.execution.process_registry import ProcessRegistry\n"
        f"owner=ProcessRegistry(Path({str(ledger)!r}),inventory=lambda:{{gpu:0 for gpu in range(8)}})\n"
        'assert owner.acquire((0,),job="real-direct-media")\n'
        'child=subprocess.Popen([sys.executable,"-m","scripts.onestep_avatar.media","--help"],'
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "owner.refresh(child_pid=child.pid)\n"
        f'Path({str(identity_file)!r}).write_text(json.dumps({{"token":owner.token}}))\n'
        "child.wait(timeout=20)\n"
        "# Deliberately preserve interrupted owner bookkeeping.\n"
    )
    owner = subprocess.Popen([sys.executable, "-c", owner_source])
    try:
        assert owner.wait(timeout=30) == 0
    finally:
        stop(owner)
    token = json.loads(identity_file.read_bytes())["token"]
    row = json.loads(ledger.read_bytes())["attempts"][token]
    assert queue_processes.process_identity(row["owner"]["pid"]) is None
    assert queue_processes.process_identity(row["child_pid"]) is None
    launch_path = tmp_path / "real-launch.json"
    command = row["processes"][str(row["child_pid"])]["identity"]["command"]
    launch_path.write_text(
        json.dumps(
            {
                "command": command,
                "physical_gpu": 0,
                "visible_gpu": 0,
                "environment_changes": {TOKEN_ENV: token, JOB_ENV: "a" * 64, "CUDA_VISIBLE_DEVICES": "0"},
            }
        )
    )
    monkeypatch.setattr(process_registry, "gpu_memory", lambda **_kwargs: dict.fromkeys(range(8), 0))
    result = launch.recover_registered_attempt(
        ledger,
        token,
        expected_row_sha256=launch._row_digest(row),
        launch_evidence=launch_path,
        expected_launch_sha256=launch.digest(launch_path),
        timeout=2,
    )
    assert result["registered_processes_absent"] is True
    assert result["original_exitcode"] is None


@pytest.mark.parametrize("approval", [None, "valid", "bad_grant", "bad_journal", "bad_request"])
def test_grant_published_during_owner_inspection_keeps_all_approval_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approval: str | None
) -> None:
    path, executed = request_for(tmp_path)
    original_identity = queue_processes.process_identity
    injected, commands = [], []
    monkeypatch.setenv(TOKEN_ENV, token())
    monkeypatch.setenv(JOB_ENV, "a" * 64)

    def inspect(pid: int) -> dict | None:
        bootstrap_path = path.parent / "bootstrap.json"
        if not injected and bootstrap_path.exists():
            injected.append(True)
            if approval:
                bootstrap = json.loads(bootstrap_path.read_text())
                register(path, bootstrap)
                # The publisher still proves a live original owner and child.
                launch.publish_grant(path)
                changed = {
                    "bad_grant": path.parent / "grant.json",
                    "bad_journal": tmp_path / "journal.json",
                    "bad_request": path,
                }.get(approval)
                if changed:
                    value = json.loads(changed.read_text())
                    if approval == "bad_grant":
                        value["token"] = "c" * 32
                    elif approval == "bad_journal":
                        value["jobs"]["case"]["command"] = ["other"]
                    else:
                        value["command"] = ["other"]
                    changed.write_text(json.dumps(value))
            return None  # Observe owner exit after the loop's initial absent-grant check.
        return original_identity(pid)

    monkeypatch.setattr(queue_processes, "process_identity", inspect)
    monkeypatch.setattr(
        launch.os,
        "execvpe",
        lambda executable, command, environment: commands.append((executable, command, environment)),
    )
    if approval == "valid":
        launch.run_guard(path, timeout=1)
        request = launch.load_request(path)
        assert len(commands) == 1
        assert commands[0][:2] == (request["command"][0], request["command"])
        assert commands[0][2][JOB_ENV] == request["job_sha256"]
    else:
        with pytest.raises(ValueError, match=r"before approval|grant or request changed|journal does not bind"):
            launch.run_guard(path, timeout=1)
        assert not commands
    assert injected == [True]
    assert not executed.exists()


def test_exec_requires_approval_and_keeps_registered_pid_and_start_ticks(tmp_path: Path) -> None:
    path, executed = request_for(tmp_path)
    with (tmp_path / "child.log").open("w") as log:
        child = start_guard(path, log)
        try:
            bootstrap = wait_bootstrap(path, child)
            assert bootstrap["identity"] == queue_processes.process_identity(child.pid)
            assert not executed.exists()
            assert child.poll() is None
            register(path, bootstrap)
            assert not executed.exists()
            assert launch.publish_grant(path) == bootstrap
            assert child.wait(timeout=5) == 0
            actual = json.loads(executed.read_text())
            assert actual["pid"] == child.pid
            assert actual["start_ticks"] == bootstrap["identity"]["start_ticks"]
            assert actual["command"] == launch.load_request(path)["command"]
        finally:
            stop(child)


@pytest.mark.parametrize(
    "field",
    [
        "state",
        "sha256",
        "command",
        "child_pid",
        "child_identity",
        "launch_request",
        "launch_request_sha256",
        "environment_changes",
        "launch_protocol",
    ],
)
def test_unbound_journal_cannot_publish_grant_or_execute(tmp_path: Path, field: str) -> None:
    path, executed = request_for(tmp_path)
    with (tmp_path / "child.log").open("w") as log:
        child = start_guard(path, log)
        try:
            bootstrap = wait_bootstrap(path, child)
            state = register(path, bootstrap)
            state["jobs"]["case"][field] = None
            Path(launch.load_request(path)["journal"]).write_text(json.dumps(state))
            with pytest.raises((ValueError, AttributeError), match=r"launch journal|has no attribute"):
                launch.publish_grant(path)
            assert not (path.parent / "grant.json").exists()
            assert not executed.exists()
            assert child.poll() is None
        finally:
            stop(child)


def test_missing_pid_registration_and_missing_journal_never_authorize_execution(tmp_path: Path) -> None:
    path, executed = request_for(tmp_path)
    with (tmp_path / "child.log").open("w") as log:
        child = start_guard(path, log)
        try:
            bootstrap = wait_bootstrap(path, child)
            with pytest.raises(FileNotFoundError):
                launch.publish_grant(path)
            state = register(path, bootstrap)
            state["jobs"]["case"]["child_pid"] = None
            Path(launch.load_request(path)["journal"]).write_text(json.dumps(state))
            with pytest.raises(ValueError):  # noqa: PT011 -- mutation controls check refusal type and preserved bytes across different validators
                launch.publish_grant(path)
            assert not executed.exists()
        finally:
            stop(child)


def test_owner_exit_before_approval_starts_no_consumer(tmp_path: Path) -> None:
    owner = subprocess.Popen(
        [sys.executable, "-c", 'import time; print("ready",flush=True); time.sleep(30)'],
        stdout=subprocess.PIPE,
        text=True,
    )
    child = None
    try:
        assert owner.stdout.readline().strip() == "ready"
        path, executed = request_for(tmp_path, owner=queue_processes.process_identity(owner.pid))
        with (tmp_path / "child.log").open("w") as log:
            child = start_guard(path, log)
            wait_bootstrap(path, child)
            stop(owner)
            assert child.wait(timeout=5) != 0
            assert not executed.exists()
            assert not (path.parent / "grant.json").exists()
    finally:
        stop(owner)
        if child is not None:
            stop(child)


def test_timeout_is_a_failure_not_permission_to_execute(tmp_path: Path) -> None:
    path, executed = request_for(tmp_path)
    with (tmp_path / "child.log").open("w") as log:
        child = start_guard(path, log, timeout=0.1)
        try:
            wait_bootstrap(path, child)
            assert child.wait(timeout=5) != 0
            assert not executed.exists()
        finally:
            stop(child)
    assert "approval deadline expired" in (tmp_path / "child.log").read_text()


def test_changed_request_while_waiting_refuses_execution(tmp_path: Path) -> None:
    path, executed = request_for(tmp_path)
    with (tmp_path / "child.log").open("w") as log:
        child = start_guard(path, log)
        try:
            wait_bootstrap(path, child)
            path.write_text(path.read_text() + "\n")
            assert child.wait(timeout=5) != 0
            assert not executed.exists()
        finally:
            stop(child)


def test_wrong_environment_and_preexisting_grant_refuse_before_bootstrap(tmp_path: Path) -> None:
    path, executed = request_for(tmp_path)
    with (tmp_path / "child.log").open("w") as log:
        child = start_guard(path, log, environment={TOKEN_ENV: "c" * 32})
        assert child.wait(timeout=5) != 0
        assert not (path.parent / "bootstrap.json").exists()
        (path.parent / "grant.json").write_text("{}")
        child = start_guard(path, log)
        assert child.wait(timeout=5) != 0
    assert not executed.exists()
    assert not (path.parent / "bootstrap.json").exists()


def test_record_publication_is_exclusive_and_invalid_input_creates_no_directory(tmp_path: Path) -> None:
    path, _ = request_for(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        launch.publish_exclusive(path, {"replacement": True})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="invalid launch request identity"):
        launch.prepare_request(
            tmp_path / "invalid",
            token="bad",
            job_sha256="a" * 64,
            owner_identity=queue_processes.process_identity(os.getpid()),
            job_id="case",
            journal=tmp_path / "j",
            command=["python"],
        )
    assert not (tmp_path / "invalid").exists()


def test_owner_may_exit_after_durable_approval_without_revoking_registered_child(tmp_path: Path) -> None:
    executed = tmp_path / "executed.json"
    script = """
import json,os,subprocess,sys,time
from pathlib import Path
from scripts.onestep_avatar.execution import queue, queue_launch as launch
from scripts.onestep_avatar.execution.queue_protocol import TOKEN_ENV,JOB_ENV,LAUNCH_PROTOCOL
root=Path(sys.argv[1])
from uuid import uuid4
attempt_token=uuid4().hex
command=[
    sys.executable,'-c',
    "import os,json; from pathlib import Path; "
    "from scripts.onestep_avatar.execution.queue import process_identity; Path("
    +repr(str(root/'executed.json'))+
    ").write_text(json.dumps(process_identity(os.getpid())))"
]
path=launch.prepare_request(root/'launch',token=attempt_token,job_sha256='a'*64,owner_identity=queue.process_identity(os.getpid()),job_id='case',journal=root/'journal.json',command=command)
child=subprocess.Popen([sys.executable,'-m','scripts.onestep_avatar.execution.queue_launch','--request',str(path),'--timeout','5'],env={**os.environ,TOKEN_ENV:attempt_token,JOB_ENV:'a'*64},start_new_session=True)
deadline=time.monotonic()+5
while not (path.parent/'bootstrap.json').exists():
 if time.monotonic()>deadline: raise RuntimeError('missing bootstrap')
 time.sleep(.01)
bootstrap=json.loads((path.parent/'bootstrap.json').read_text())
request=launch.load_request(path)
row={'state':'running','sha256':request['job_sha256'],'command':command,'launch_protocol':LAUNCH_PROTOCOL,'launch_request':str(path.resolve()),'launch_request_sha256':bootstrap['request_sha256'],'child_pid':child.pid,'child_identity':bootstrap['identity'],'environment_changes':{TOKEN_ENV:attempt_token,JOB_ENV:'a'*64}}
(root/'journal.json').write_text(json.dumps({'schema_version':1,'owner_pid':os.getpid(),'jobs':{'case':row}}))
launch.publish_grant(path)
os._exit(0)
"""
    with (tmp_path / "owner.log").open("w") as log:
        owner = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=log, stderr=subprocess.STDOUT)
        try:
            assert owner.wait(timeout=5) == 0
            bootstrap = json.loads((tmp_path / "launch/bootstrap.json").read_text())
            deadline = time.monotonic() + 5
            while True:
                handle = queue_processes.process_identity(bootstrap["identity"]["pid"])
                if handle is None or handle["terminal"]:
                    break
                assert handle["start_ticks"] == bootstrap["identity"]["start_ticks"]
                if time.monotonic() >= deadline:
                    pytest.fail("approved child never finished")
                time.sleep(0.01)
            actual = json.loads(executed.read_text())
            assert actual["pid"] == bootstrap["identity"]["pid"]
            assert actual["start_ticks"] == bootstrap["identity"]["start_ticks"]
        finally:
            stop(owner)
            bootstrap_path = tmp_path / "launch/bootstrap.json"
            if bootstrap_path.exists():
                saved = json.loads(bootstrap_path.read_text())["identity"]
                current = queue_processes.process_identity(saved["pid"])
                if current and not current["terminal"] and current["start_ticks"] == saved["start_ticks"]:
                    os.kill(saved["pid"], 15)


@pytest.mark.parametrize("corruption", ["schema_bool", "unbound_journal"])
def test_child_rechecks_grant_and_journal_instead_of_trusting_publication(tmp_path: Path, corruption: str) -> None:
    path, executed = request_for(tmp_path)
    with (tmp_path / "child.log").open("w") as log:
        child = start_guard(path, log)
        try:
            bootstrap = wait_bootstrap(path, child)
            state = register(path, bootstrap)
            approved = dict(bootstrap)
            if corruption == "schema_bool":
                approved["schema_version"] = True
            else:
                state["jobs"]["case"]["child_pid"] = None
                Path(launch.load_request(path)["journal"]).write_text(json.dumps(state))
            # Bypass the parent validator deliberately; the real child must refuse.
            launch.publish_exclusive(path.parent / "grant.json", approved)
            assert child.wait(timeout=5) != 0
            assert not executed.exists()
        finally:
            stop(child)


@pytest.mark.usefixtures("controlled_queue_observation")
@pytest.mark.parametrize("changed", [None, "grant", "bootstrap", "request", "command", "ticks", "protocol"])
def test_real_dispatch_registers_before_grant_and_observes_approved_transition(  # noqa: PLR0915 -- one complete fixture lifecycle with all assertions and cleanup
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str | None
) -> None:
    executed, release = tmp_path / "executed.json", tmp_path / "release"
    command = [
        sys.executable,
        "-c",
        "import os,json,time; from pathlib import Path; "
        "from scripts.onestep_avatar.execution.queue import process_identity; "
        f"Path({str(executed)!r}).write_text(json.dumps(process_identity(os.getpid()))); "
        f"release=Path({str(release)!r}); "
        "\nwhile not release.exists(): time.sleep(.01)",
    ]
    job = {
        "id": "case",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "dependencies": [],
        "arguments": ["--mode", "causal"],
        "output": str(tmp_path / "output"),
    }
    monkeypatch.setattr(queue_jobs, "job_command", lambda *_: (command, {}))
    monkeypatch.setattr(
        queue_completion, "completion_receipt", lambda *_, **__: {"job_sha256": job["sha256"], "evidence": []}
    )
    claims = ProcessRegistry(tmp_path / "ledger.json", inventory=lambda: dict.fromkeys(range(8), 0))
    assert claims.acquire((4,), job="case")
    state_path = tmp_path / "state.json"
    original = launch.publish_grant
    observed = []

    def approve(path: Path) -> dict:
        state = json.loads(state_path.read_text())
        row = state["jobs"]["case"]
        assert row["child_pid"] == row["child_identity"]["pid"]
        assert row["child_identity"]["command"] == launch.guard_command(path)
        assert row["attempts"][-1]["child_identity"] == row["child_identity"]
        assert not executed.exists()
        result = original(path)
        deadline = time.monotonic() + 5
        while not executed.exists():
            if time.monotonic() >= deadline:
                pytest.fail("approved consumer did not execute")
            time.sleep(0.01)
        assert queue_processes.inspect_child(row) == "live"
        actual = json.loads(executed.read_text())
        assert actual["command"] == command
        assert actual["pid"] == row["child_pid"]
        assert actual["start_ticks"] == row["child_identity"]["start_ticks"]
        if changed in ("grant", "bootstrap", "request"):
            changed_path = path if changed == "request" else path.parent / (changed + ".json")
            original_bytes = changed_path.read_bytes()
            changed_record = json.loads(original_bytes)
            changed_record["token"] = "c" * 32
            try:
                changed_path.write_text(json.dumps(changed_record))
                with pytest.raises(ValueError):  # noqa: PT011 -- mutation controls check refusal type and preserved bytes across different validators
                    queue_processes.inspect_child(row)
            finally:
                changed_path.write_bytes(original_bytes)
            assert queue_processes.inspect_child(row) == "live"
        elif changed:
            invalid = json.loads(json.dumps(row))
            if changed == "command":
                invalid["command"] = ["other"]
            if changed == "ticks":
                invalid["child_identity"]["start_ticks"] += 1
            if changed == "protocol":
                invalid["launch_protocol"] = "unrecognized"
            with pytest.raises(ValueError):  # noqa: PT011 -- mutation controls check refusal type and preserved bytes across different validators
                queue_processes.inspect_child(invalid)
        observed.append(row)
        release.write_text("exit")
        return result

    monkeypatch.setattr(launch, "publish_grant", approve)
    try:
        result = queue_attempt.run_child(
            job, claims, tmp_path / "child.log", state_path=state_path, jobs=[job], poll_seconds=0.01
        )
        row = json.loads(state_path.read_text())["jobs"]["case"]
        assert row["state"] == "complete"
        assert result["returncode"] == 0
        assert row["child_identity"] == observed[0]["child_identity"]
        assert row["launch_protocol"] == LAUNCH_PROTOCOL
        assert not claims.owned
        assert row["command"] == command
    finally:
        release.write_text("exit")


def test_real_dispatch_registration_failure_never_grants_and_bounded_stops_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, executed = request_for(tmp_path)
    request = launch.load_request(path)
    job = {
        "id": "case",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "dependencies": [],
        "arguments": ["--mode", "causal"],
        "output": str(tmp_path / "output"),
    }
    monkeypatch.setattr(queue_jobs, "job_command", lambda *_: (request["command"], {}))
    claims = process_registry.ProcessRegistry(tmp_path / "ledger.json", inventory=lambda: dict.fromkeys(range(8), 0))
    assert claims.acquire((4,), job="case")
    original = launch.wait_registration
    child_handle = []

    def fail(path: Path, child: subprocess.Popen, command: list[str], refresh: object, **kwargs: object) -> None:
        child_handle.append(child)
        original(path, child, command, refresh, **kwargs)
        raise ValueError("controlled registration interruption")

    monkeypatch.setattr(launch, "wait_registration", fail)
    state_path = tmp_path / "state.json"
    try:
        with pytest.raises(ValueError, match="controlled registration interruption"):
            queue_attempt.run_child(
                job, claims, tmp_path / "child.log", state_path=state_path, jobs=[job], poll_seconds=0.01
            )
        row = json.loads(state_path.read_text())["jobs"]["case"]
        assert row["state"] == "failed"
        assert row["child_pid"] == child_handle[0].pid
        assert not (Path(row["launch_request"]).parent / "grant.json").exists()
        assert not executed.exists()
        assert not claims.owned
        assert child_handle[0].poll() is not None
        observed = json.loads(Path(row["supervision_result"]).read_text())
        assert observed["workers_absent"] is True
        assert observed["state"] == "failed"
    finally:
        for child in child_handle:
            stop(child)
        if claims.owned:
            claims.release()


@pytest.mark.parametrize("fail_sync", [False, True])
def test_journal_sync_precedes_publication_and_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_sync: bool
) -> None:
    job = {"id": "case", "sha256": "a" * 64}
    path = tmp_path / "state.json"
    with queue_journal.queue_state(path, [job]):
        pass
    original = path.read_bytes()
    actual = os.fsync
    observations = []

    def sync(fd: int) -> None:
        target = str(Path(f"/proc/self/fd/{fd}").readlink())
        observations.append(target)
        if fail_sync and target.endswith(".json"):
            raise OSError("controlled sync failure")
        actual(fd)

    monkeypatch.setattr(queue.os, "fsync", sync)
    if fail_sync:
        with pytest.raises(OSError, match="controlled sync failure"), queue_journal.queue_state(path, [job]) as state:
            state["jobs"]["case"]["error"] = "changed"
        assert path.read_bytes() == original
        assert len(observations) == 1
    else:
        with queue_journal.queue_state(path, [job]) as state:
            state["jobs"]["case"]["error"] = "changed"
        assert observations[0] == str(path.with_name(f".state.tmp.{os.getpid()}.json"))
        assert observations[1] == str(tmp_path)
        assert json.loads(path.read_text())["jobs"]["case"]["error"] == "changed"
