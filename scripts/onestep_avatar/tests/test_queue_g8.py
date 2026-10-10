"""CPU controls for the expanded pool, exact budgets and persistent clocks."""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import LTX_ROOT, evaluate
from scripts.onestep_avatar.execution import devices, queue, runtime
from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV
from scripts.onestep_avatar.tests.claims_fixture import GPUClaims
from scripts.onestep_avatar.training import resources


def test_arbitrary_four_free_devices_and_single_shared_pool() -> None:
    memory = dict.fromkeys(range(8), 2048)
    for gpu in (1, 3, 6, 7):
        memory[gpu] = 0
    assert devices.select_devices(memory, training=True) == (1, 3, 6, 7)
    assert devices.select_devices(memory, training=False) == (1,)
    assert devices.select_devices(memory, training=True, busy=(7,)) is None
    for invalid in ((0, 0, 1, 2), (0, 1, 2, 8), (True, 1, 2, 3), (0, 1, 2)):
        with pytest.raises(ValueError, match="unique allowed GPUs"):
            devices.validate_selection(invalid, training=True)


@pytest.mark.parametrize("overall", [True, False, 0, -1, None, float("inf"), float("nan"), "10"])
def test_overall_limit_requires_finite_positive_number(tmp_path: Path, overall: object) -> None:
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48_000_000_000, "overall_seconds": overall}
        )
    )
    with pytest.raises(ValueError, match="overall_seconds"):
        resources.read_budget(path)


def test_default_budget_has_no_planning_writes_and_no_inferred_overall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = resources.default_budget()
    assert expected["wall_seconds_per_phase"] == 1800
    assert expected["memory_limit_allocated_bytes"] == 48_000_000_000
    assert "overall_seconds" not in expected
    assert expected["path"] is None
    assert resources.effective_budget() == expected
    assert not list(tmp_path.iterdir())
    env = {**resources.budget_environment(expected), TOKEN_ENV: "a" * 32, JOB_ENV: "b" * 64}
    assert resources.budget_from_environment(env) == expected
    for key in (TOKEN_ENV, JOB_ENV, resources.BUDGET_ENV, resources.BUDGET_SHA_ENV):
        with pytest.raises(ValueError, match="paired queue-bound"):
            resources.budget_from_environment({name: value for name, value in env.items() if name != key})


def test_original_deadline_survives_retry_and_restart_and_detects_reset(tmp_path: Path) -> None:
    job = {"id": "train", "sha256": "a" * 64, "effective_budget": {"overall_seconds": 10}}
    row = {"attempts": []}
    binding = queue.begin_execution_clock(job, row, now=100)
    row["attempts"].append({"started_at": 100, **binding})
    assert queue.begin_execution_clock(job, row, now=106) == binding
    row["attempts"].append({"started_at": 106, **binding})
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_order": ["train"],
                "jobs": {"train": {"sha256": job["sha256"], "state": "pending", **row}},
            }
        )
    )
    restarted = queue.read_queue_state(path, [job])["jobs"]["train"]
    assert queue.remaining_deadline(restarted, now=109) == 1
    with pytest.raises(queue.OverallDeadlineError, match="overall_deadline"):
        queue.remaining_deadline(restarted, now=110)
    restarted.update(first_attempt_started_at=106, overall_deadline_at=116)
    with pytest.raises(ValueError, match="first attempt"):
        queue.validate_execution_clock(job, restarted)


def test_runtime_outside_queue_calls_once_and_partial_binding_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.CONTRACT_ENV, raising=False)
    monkeypatch.delenv(runtime.CONTRACT_SHA_ENV, raising=False)
    calls = []
    assert runtime.run(lambda: calls.append("main")) is None
    assert calls == ["main"]
    monkeypatch.setenv(runtime.CONTRACT_ENV, "/missing")
    with pytest.raises(ValueError, match="incomplete"):
        runtime.run(lambda: pytest.fail("partial binding executed a model"))


def test_runtime_bound_failure_preserves_evidence_without_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "runtime.json"
    budget = resources.default_budget()
    changes = runtime.prepare_contract(
        contract, token="a" * 32, job_sha256="b" * 64, budget=budget, generic_phase=False, numerical_policy=None
    )
    for name, value in {
        **changes,
        **resources.budget_environment(budget),
        TOKEN_ENV: "a" * 32,
        JOB_ENV: "b" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    calls = []

    def operation() -> None:
        calls.append(1)
        raise RuntimeError("scientific failure")

    with pytest.raises(RuntimeError, match="scientific failure"):
        runtime.run(operation)
    assert calls == [1]
    record = json.loads(runtime.result_path(contract).read_text())
    assert record["state"] == "failed"
    assert "scientific failure" in record["error"]
    with pytest.raises(ValueError, match="successful contract"):
        runtime.verify_result(contract, changes[runtime.CONTRACT_SHA_ENV])


def test_current_scientific_receipt_cannot_unblock_without_supervision() -> None:
    job = {"sha256": "a" * 64, "effective_budget": resources.default_budget()}
    with pytest.raises(ValueError, match="supervised execution proof"):
        queue.verify_receipt(job, {"job_sha256": "a" * 64, "evidence": []})


def test_saved_execution_refuses_changed_physical_device_mapping(tmp_path: Path) -> None:
    job = queue.prepare_job(
        {
            "id": "case",
            "kind": "evaluate",
            "output": "output",
            "arguments": [
                "--mode",
                "causal",
                "--subset",
                "membership.json",
                "--output",
                "output",
                "--schedule",
                "1",
                "0",
            ],
            "completion": {"records": ["output/result.json"]},
        },
        tmp_path,
    )
    command, environment = queue.job_command(job, (0,))
    row = {
        "command": command,
        "gpus": [0],
        "child_identity": {},
        "environment_changes": environment,
        "supervision_result": str(tmp_path / "missing.json"),
        "supervision_result_sha256": "a" * 64,
        "supervision_contract": str(tmp_path / "phases.json"),
    }
    with pytest.raises(FileNotFoundError):
        queue.verify_attempt_evidence(job, row)  # Correct mapping proceeds to saved evidence.
    row["environment_changes"] = {**environment, "CUDA_VISIBLE_DEVICES": "1"}
    with pytest.raises(ValueError, match="environment differs from requested devices"):
        queue.verify_attempt_evidence(job, row)


@pytest.mark.usefixtures("controlled_queue_launch", "controlled_evaluation_conditions")
def test_final_transaction_cannot_publish_zero_exit_after_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {"wall_seconds_per_phase": 20, "memory_limit_allocated_bytes": 48_000_000_000, "overall_seconds": 10}
        )
    )
    output = tmp_path / "output"
    job = {
        "id": "case",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(output),
        "arguments": ["--mode", "causal", "--output", str(output)],
        "effective_budget": resources.read_budget(budget_path),
        "completion": {"records": [str(output / "result.json")]},
    }
    clock = [100.0]
    monkeypatch.setattr(queue.time, "time", lambda: clock[0])
    claims = GPUClaims(tmp_path / "claims")
    assert claims.acquire((0,), job="case")

    def launch(*_args, **_kwargs) -> object:
        evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {"mode": "causal"}, output)
        return SimpleNamespace(pid=99999, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(queue, "process_alive", lambda _: False)
    original = queue.queue_state
    transactions = []

    @contextmanager
    def locked(*args, **kwargs) -> object:
        with original(*args, **kwargs) as state:
            transactions.append(1)
            if len(transactions) == 4:
                clock[0] = 111.0  # The final lock/read consumes the remaining original time.
            yield state

    monkeypatch.setattr(queue, "queue_state", locked)
    state_path = tmp_path / "state.json"
    with pytest.raises(queue.OverallDeadlineError):
        queue.run_child(job, claims, tmp_path / "child.log", state_path=state_path, jobs=[job])
    row = json.loads(state_path.read_text())["jobs"]["case"]
    assert row["state"] == "failed"
    assert row["failure_reason"] == "overall_deadline"
    assert "receipt" not in row
    assert row["overall_deadline_at"] == 110
    assert not claims.owned


def test_two_cpu_queue_owners_share_disjoint_four_device_jobs(tmp_path: Path) -> None:
    """Real distinct owners exercise the queue/ledger topology without CUDA kernels."""
    source = tmp_path / "owner.py"
    source.write_text("""import json,sys,time
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.execution import queue,process_registry
name,root=sys.argv[1],Path(sys.argv[2])
memory=dict.fromkeys(range(8),0)
queue.ready_jobs=lambda jobs,state: jobs
original=queue.subprocess if hasattr(queue,"subprocess") else None
import subprocess
subprocess.run=lambda *a,**k:type("Observed",(),{"stdout":"\\n".join(f"{g}, 0" for g in range(8))})()
def child(job,claims,log,**kwargs):
 command,env=queue.job_command(job,tuple(sorted(claims.owned)))
 (root/(name+".ready")).write_text(json.dumps({"gpus":sorted(claims.owned),"command":command,"env":env}))
 deadline=time.monotonic()+10
 while not (root/(name+".release")).exists():
  if time.monotonic()>deadline: raise TimeoutError("CPU fixture timed out")
  time.sleep(.01)
 claims.release()
queue.run_child=child
config=root/"fsdp.yaml"
from scripts.onestep_avatar.hashing import sha256
job={"id":name,"kind":"train","arguments":["--mode","causal","--subset",str(root/"membership.json"),"--output",str(root/name)],
 "processes":4,"accelerate_config":str(config),"accelerate_config_sha256":sha256(config),"port":29530+(name=="second")}
assert queue.dispatch_ready([job],{"jobs":{name:{"state":"pending"}}},root/(name+".state.json"),root/"ledger.json")
""")
    (tmp_path / "fsdp.yaml").write_text("num_processes: 4\n")
    env = {**os.environ, "PYTHONPATH": str(LTX_ROOT), "CUDA_VISIBLE_DEVICES": ""}
    owners = []

    def wait_file(name: str) -> object:
        target = tmp_path / name
        deadline = time.monotonic() + 10
        while not target.exists():
            if time.monotonic() >= deadline:
                pytest.fail(f"CPU owner did not publish {name}")
            time.sleep(0.01)
        return json.loads(target.read_text())

    try:
        for name in ("first", "second"):
            owners.append(subprocess.Popen([sys.executable, str(source), name, str(tmp_path)], cwd=LTX_ROOT, env=env))
            row = wait_file(name + ".ready")
            assert len(row["gpus"]) == 4
            assert row["command"][row["command"].index("--num_processes") + 1] == "4"
        first = wait_file("first.ready")
        second = wait_file("second.ready")
        assert set(first["gpus"]).isdisjoint(second["gpus"])
        assert set(first["gpus"]) | set(second["gpus"]) == set(range(8))
        (tmp_path / "first.release").touch()
        assert owners[0].wait(timeout=5) == 0
        ledger = json.loads((tmp_path / "ledger.json").read_text())["attempts"]
        assert sorted(row["state"] for row in ledger.values()) == ["active", "closed"]
        (tmp_path / "second.release").touch()
        assert owners[1].wait(timeout=5) == 0
        assert all(
            row["state"] == "closed" for row in json.loads((tmp_path / "ledger.json").read_text())["attempts"].values()
        )
    finally:
        for name in ("first", "second"):
            (tmp_path / (name + ".release")).touch()
        for owner in owners:
            if owner.poll() is None:
                owner.terminate()
            owner.wait(timeout=5)
