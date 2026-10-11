"""Missing original supervision fails honestly after exact owned-handle closure."""

import json
from pathlib import Path

import pytest

from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.execution import journal as queue_journal
from scripts.onestep_avatar.execution import processes as queue_processes
from scripts.onestep_avatar.execution import queue, queue_launch
from scripts.onestep_avatar.tests.test_queue_launch import ended_owned_attempt
from scripts.onestep_avatar.training import resources


def closed_interrupted_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    ledger, request, _owned, _ranks, _notifications = ended_owned_attempt(tmp_path, monkeypatch)
    journal = tmp_path / "journal.json"
    state = json.loads(journal.read_bytes())
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48_000_000_000, "overall_seconds": 3600}
        )
    )
    job = {"id": "case", "kind": "train", "sha256": "a" * 64, "effective_budget": resources.read_budget(budget_path)}
    for item in (state["jobs"]["case"], state["jobs"]["case"]["attempts"][-1]):
        item.update(started_at=100, first_attempt_started_at=100, overall_deadline_at=3700)
    state["job_order"] = ["case"]
    journal.write_text(json.dumps(state))
    result = queue_launch.recover_owned_attempt(ledger, request, timeout=2)
    (tmp_path / "scientific-output.json").write_text('{"saved_science":"preserved"}\n')
    monkeypatch.setattr(queue_processes, "process_alive", lambda _pid: False)
    monkeypatch.setattr(queue_processes, "process_identity", lambda _pid: None)
    monkeypatch.setattr(queue.time, "time", lambda: 105)
    return journal, ledger, request, job, result


def test_missing_supervisor_fails_before_deadline_without_scientific_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal, ledger, _request, job, closure = closed_interrupted_attempt(tmp_path, monkeypatch)
    before = json.loads(journal.read_bytes())
    prior = before["jobs"]["case"]["attempts"][-1]
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and path != journal}

    def forbidden(*_args, **_kwargs) -> None:
        pytest.fail("missing original supervision must not enter scientific success verification")

    monkeypatch.setattr(queue_completion, "verify_completion", lambda _job: True)
    monkeypatch.setattr(queue_completion, "verify_attempt_evidence", forbidden)
    monkeypatch.setattr(queue_completion, "completion_receipt", forbidden)
    with queue_journal.queue_state(journal, [job], recover=True) as recovered:
        row = recovered["jobs"]["case"]
        assert row["state"] == "failed"
        assert row["failure_reason"] == "missing_supervision"
        assert row["first_attempt_started_at"] == 100
        assert row["overall_deadline_at"] == 3700
        assert len(row["attempts"]) == 1
        assert row["interruption_recovery"]["prior_attempt"] == prior
        assert row["interruption_recovery"]["operational_closure"] == closure
        assert row["interruption_recovery"]["scientific_success"] is False
        assert row["interruption_recovery"]["continuous_supervision"] is False
        assert row["interruption_recovery"]["unknown_descendants_unproven"] is True
        assert "receipt" not in row
        assert "returncode" not in row or row["returncode"] is None
        assert row["attempts"][-1]["interruption_recovery"] == row["interruption_recovery"]
    assert json.loads(journal.read_bytes())["jobs"]["case"]["state"] == "failed"
    assert json.loads(ledger.read_bytes())["attempts"]["b" * 32]["recovery"] == closure
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_public_recovery_classifies_missing_supervisor_without_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal, _ledger, _request, job, _closure = closed_interrupted_attempt(tmp_path, monkeypatch)
    monkeypatch.setattr(queue_jobs, "prepare_jobs", lambda _path: [job])
    monkeypatch.setattr(queue_completion, "verify_completion", lambda _job: True)
    result = queue.recover_jobs(tmp_path / "jobs.json", journal)
    assert result["jobs"][0]["state"] == "failed"
    row = json.loads(journal.read_bytes())["jobs"]["case"]
    assert row["failure_reason"] == "missing_supervision"
    assert "receipt" not in row


@pytest.mark.parametrize(
    "failure",
    [
        "missing_closure",
        "active_ledger",
        "unproven_closure",
        "closure_token",
        "closure_journal",
        "original_row_changed",
        "journal_changed",
        "request_changed",
        "approval_changed",
        "attempt_changed",
        "live_child",
        "reused_child",
        "live_owner",
        "missing_supervisor_hash",
        "missing_supervisor_path",
        "present_corrupt_supervisor",
        "present_null_supervisor",
        "unregistered_launch",
        "corrupt_science",
    ],
)
def test_missing_supervisor_recovery_refuses_ambiguity_without_mutation(  # noqa: PLR0912 -- explicit controlled corruption
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    journal, ledger, request, job, _closure = closed_interrupted_attempt(tmp_path, monkeypatch)
    state, record = json.loads(journal.read_bytes()), json.loads(ledger.read_bytes())
    monkeypatch.setattr(queue_completion, "verify_completion", lambda _job: True)
    row = state["jobs"]["case"]
    owned = record["attempts"]["b" * 32]
    if failure == "missing_closure":
        owned.pop("recovery")
    elif failure == "active_ledger":
        owned["state"] = "active"
    elif failure == "unproven_closure":
        owned["recovery"]["registered_workers_absent"] = False
    elif failure == "closure_token":
        owned["recovery"]["token"] = "d" * 32
    elif failure == "closure_journal":
        owned["recovery"]["journal_sha256"] = "d" * 64
    elif failure == "original_row_changed":
        owned["observations"][0]["error"] = "controlled changed history"
    elif failure == "journal_changed":
        state["unexpected_change"] = True
    elif failure == "request_changed":
        request.write_bytes(request.read_bytes() + b" ")
    elif failure == "approval_changed":
        (request.parent / "grant.json").write_text("{}")
    elif failure == "attempt_changed":
        row["attempts"][-1]["command"] = ["changed"]
    elif failure in ("live_child", "reused_child", "live_owner"):
        identity = (
            owned["owner"] if failure == "live_owner" else owned["processes"][str(owned["child_pid"])]["identity"]
        )
        monkeypatch.setattr(
            queue_processes,
            "process_identity",
            lambda pid: (
                {**identity, "start_ticks": identity["start_ticks"] + int(failure == "reused_child")}
                if pid == identity["pid"]
                else None
            ),
        )
    elif failure == "unregistered_launch":
        for item in (row, row["attempts"][-1]):
            item["launch_protocol"] = "old_protocol"
        monkeypatch.setattr(
            queue_completion,
            "verify_attempt_evidence",
            lambda *_args: (_ for _ in ()).throw(ValueError("old supervision")),
        )
    elif failure == "corrupt_science":
        monkeypatch.setattr(
            queue_completion, "verify_completion", lambda *_args: (_ for _ in ()).throw(ValueError("corrupt science"))
        )
    else:
        if failure != "missing_supervisor_path":
            row["supervision_result"] = None if failure == "present_null_supervisor" else str(tmp_path / "absent.json")
        if failure not in ("missing_supervisor_hash", "present_null_supervisor"):
            row["supervision_result_sha256"] = "d" * 64
        monkeypatch.setattr(queue_completion, "verify_completion", lambda _job: True)

        def corrupt(*_args, **_kwargs) -> None:
            raise ValueError("present supervisor evidence remains invalid")

        monkeypatch.setattr(queue_completion, "verify_attempt_evidence", corrupt)
    journal.write_text(json.dumps(state))
    ledger.write_text(json.dumps(record))
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, OSError)), queue_journal.queue_state(journal, [job], recover=True):
        pytest.fail("ambiguous owned attempt adopted")
    assert all(path.read_bytes() == data for path, data in originals.items())
