"""CPU controls for bounded exact-child supervision; no GPU/native acceptance."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar, Never

import pytest

from scripts.onestep_avatar.execution import process_registry, queue, supervision
from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV

TOKEN = "1" * 32
JOB = "2" * 64
BUDGET = "3" * 64


class Claims:
    token = TOKEN
    owned: ClassVar[set[int]] = {0}

    def __init__(self):
        self.refreshes = []

    def refresh(self, *, child_pid: int) -> None:
        self.refreshes.append(child_pid)

    def register_identity(self, identity: dict, *, role: str) -> None:
        pass

    def observe_workers(self, *, timeout: float) -> dict:
        del timeout
        return {"schema_version": 1, "workers_live": False, "complete": True, "identities": []}

    def registered_identities(self, *, timeout: float) -> list:
        del timeout
        return []


def contract(tmp_path: Path, phases: list[str] | None = None, world: int = 1) -> tuple[Path, dict[str, str]]:
    path = tmp_path / "notifications.json"
    env = supervision.prepare_notifications(
        path,
        token=TOKEN,
        job_sha256=JOB,
        world=world,
        phases=phases or ["load"],
        budget_sha256=BUDGET,
    )
    return path, {**env, TOKEN_ENV: TOKEN, JOB_ENV: JOB}


def spawn(tmp_path: Path, code: str, env: dict[str, str]) -> tuple:
    ready = tmp_path / "ready"
    source = "import os, signal, time\nfrom pathlib import Path\n" + code.replace("READY", repr(str(ready)))
    command = [sys.executable, "-c", source]
    child = subprocess.Popen(command, env={**os.environ, **env}, start_new_session=True)
    deadline = time.monotonic() + 4
    while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), "CPU control child did not become ready"
    return child, command, queue.process_identity(child.pid)


def run(
    tmp_path: Path, child: subprocess.Popen, command: list[str], identity: dict, notifications: Path, **changes
) -> dict:
    claims = changes.pop("registry", None) or Claims()
    environment = {TOKEN_ENV: TOKEN, JOB_ENV: JOB}
    if notifications is not None:
        environment.update(
            {
                supervision.SUPERVISION_ENV: str(notifications.resolve()),
                supervision.SUPERVISION_SHA_ENV: supervision._contract(notifications)[1],
            }
        )
    row = {"environment_changes": environment, "child_session": child.pid}
    kwargs = {
        "identity": identity,
        "command": command,
        "worker_record": row,
        "claims": claims,
        "gpus": (0,),
        "evidence_path": tmp_path / "evidence.json",
        "notifications_path": notifications,
        "startup_seconds": 0.15,
        "phase_seconds": 0.15,
        "shutdown_seconds": 0.1,
        "inventory_timeout_seconds": 0.15,
        "poll_seconds": 0.01,
    }
    kwargs.update(changes)
    result = supervision.supervise(child, **kwargs)
    assert json.loads((tmp_path / "evidence.json").read_text()) == result
    assert claims.owned == {0}
    assert result["process_records_retained"] is True
    return result


def test_terminal_observation_between_poll_and_reap_is_normal_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(0.1)\n", {})
    original = supervision._observe
    calls = []

    def observe(*args) -> dict | None:
        calls.append(1)
        if len(calls) == 3:
            child.wait(timeout=1)
            return {**identity, "terminal": True}
        return original(*args)

    monkeypatch.setattr(supervision, "_observe", observe)
    try:
        result = run(tmp_path, child, command, identity, None, overall_seconds=2)
        assert result["state"] == "passed"
        assert result["workers_absent"]
        assert result["returncode"] == 0
        assert result["events"] == []
    finally:
        cleanup(child)


def test_empty_command_before_terminal_event_reaps_only_exact_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finish = tmp_path / "allow-exit"
    child, command, identity = spawn(
        tmp_path, f"Path(READY).write_text('ready')\nwhile not Path({str(finish)!r}).exists(): time.sleep(.005)\n", {}
    )
    original_observe = supervision._observe
    original_identity = queue.process_identity
    calls = []
    empty_reads = []

    def observe(*args) -> dict | None:
        calls.append(1)
        if len(calls) == 3:
            finish.touch()
            raise ValueError("transient empty command")
        return original_observe(*args)

    monkeypatch.setattr(supervision, "_observe", observe)
    monkeypatch.setattr(supervision, "_pidfd_terminal", lambda _descriptor: False)

    def read_identity(pid: int) -> dict | None:
        if len(calls) == 3 and pid == child.pid and not empty_reads:
            empty_reads.append(pid)
            return {**identity, "command": [], "terminal": False}
        return original_identity(pid)

    monkeypatch.setattr(queue, "process_identity", read_identity)
    try:
        result = run(tmp_path, child, command, identity, None, overall_seconds=2)
        assert result["state"] == "passed"
        assert result["workers_absent"]
        assert result["returncode"] == 0
        assert empty_reads == [child.pid]
    finally:
        cleanup(child)


def cleanup(child: subprocess.Popen) -> None:
    if child.poll() is None:
        descriptor = supervision._pidfd_open(child.pid)
        try:
            supervision._pidfd_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
    child.wait(timeout=2)


@pytest.fixture(autouse=True)
def isolated_worker_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    # Most controls isolate the supervisor from unrelated live host processes.
    monkeypatch.setattr(
        supervision,
        "_worker_inventory",
        lambda _record, _timeout: {
            "schema_version": 1,
            "workers_live": False,
            "complete": True,
        },
    )


def test_real_ignored_term_has_finite_kill_and_retains_claims(tmp_path: Path) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(
        tmp_path,
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\nPath(READY).write_text('ready')\ntime.sleep(20)\n",
        env,
    )
    started = time.monotonic()
    try:
        result = run(tmp_path, child, command, identity, path)
        assert time.monotonic() - started < 2
        assert "startup deadline" in result["error"]
        assert [event["signal"] for event in result["events"] if event["event"] == "signal"] == [
            "SIGTERM",
            "SIGKILL",
        ]
        assert result["returncode"] == -signal.SIGKILL
        assert result["workers_absent"] is True
    finally:
        cleanup(child)


def test_complete_rank_phase_notifications_pass(tmp_path: Path) -> None:
    path, env = contract(tmp_path, ["load", "export"])
    child, command, identity = spawn(
        tmp_path,
        "from scripts.onestep_avatar.execution.supervision import notify_phase\n"
        "Path(READY).write_text('ready')\ntime.sleep(.05)\n"
        f"notify_phase('load', 'begin', 0, budget_sha256={BUDGET!r})\n"
        f"notify_phase('load', 'end', 0, budget_sha256={BUDGET!r})\n"
        f"notify_phase('export', 'begin', 0, budget_sha256={BUDGET!r})\n"
        "time.sleep(.05)\n"
        f"notify_phase('export', 'end', 0, budget_sha256={BUDGET!r})\n",
        env,
    )
    try:
        result = run(tmp_path, child, command, identity, path)
        assert result["state"] == "passed"
        assert result["workers_absent"] is True
        assert len(result["notifications"]) == 4
    finally:
        cleanup(child)


@pytest.mark.parametrize(
    ("code", "error"),
    [
        (f"notify_phase('load', 'begin', 0, budget_sha256={BUDGET!r})\n", "phase load deadline"),
        (
            f"notify_phase('load', 'begin', 0, budget_sha256={BUDGET!r})\n"
            f"notify_phase('load', 'end', 0, budget_sha256={BUDGET!r})\n",
            "notification gap deadline",
        ),
    ],
)
def test_phase_and_gap_deadlines_are_finite(tmp_path: Path, code: str, error: str) -> None:
    path, env = contract(tmp_path, ["load", "export"])
    child, command, identity = spawn(
        tmp_path,
        "from scripts.onestep_avatar.execution.supervision import notify_phase\n"
        "Path(READY).write_text('ready')\ntime.sleep(.05)\n" + code + "time.sleep(20)\n",
        env,
    )
    try:
        result = run(tmp_path, child, command, identity, path)
        assert result["state"] == "failed"
        assert error in result["error"]
        assert result["elapsed_s"] < 2
    finally:
        cleanup(child)


def test_escaped_real_new_session_worker_receives_own_bounded_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, env = contract(tmp_path)
    registry = process_registry.ProcessRegistry(tmp_path / "own.json", inventory=lambda: dict.fromkeys(range(8), 0))
    registry.token = TOKEN
    assert registry.acquire((0,), job="escaped-worker-control")
    worker_file = tmp_path / "worker.json"
    code = (
        "import subprocess, sys, json\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], "
        "start_new_session=True)\n"
        f"Path({str(worker_file)!r}).write_text(json.dumps({{'pid': worker.pid}}))\n"
        "Path(READY).write_text('ready')\ntime.sleep(20)\n"
    )
    child, command, identity = spawn(tmp_path, code, env)
    worker_pid = json.loads(worker_file.read_text())["pid"]
    worker_identity = queue.process_identity(worker_pid)
    assert os.getsid(worker_pid) == worker_pid
    assert os.getsid(child.pid) == child.pid
    # This control uses real targeted subreaper inventory, with no host-wide scan.
    monkeypatch.setattr(supervision, "_worker_inventory", lambda owner, timeout: owner.observe_workers(timeout=timeout))
    try:
        result = run(tmp_path, child, command, identity, path, registry=registry)
        assert result["state"] == "failed"
        assert result["workers_absent"] is True
        signals = [event for event in result["events"] if event["event"] == "signal"]
        assert any(
            event["pid"] == worker_pid and event["start_ticks"] == worker_identity["start_ticks"] for event in signals
        )
        assert any(event["pid"] == child.pid for event in signals)
        worker = queue.process_identity(worker_pid)
        assert worker is None or worker["terminal"] is True
        assert result["elapsed_s"] < 2
        registry.release()
    finally:
        cleanup(child)
        worker = queue.process_identity(worker_pid)
        if worker is not None and not worker["terminal"]:
            descriptor = supervision._pidfd_open(worker_pid)
            try:
                supervision._pidfd_signal(descriptor, signal.SIGKILL)
            finally:
                os.close(descriptor)
        if worker is not None:
            os.waitpid(worker_pid, os.WNOHANG)


@pytest.mark.parametrize("observation", ["reuse", "missing", "denied"])
def test_unproven_handle_never_signals_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observation: str
) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(20)\n", env)
    original = queue.process_identity
    sent = []

    def observe(pid: int) -> dict | None:
        if pid != child.pid:
            return original(pid)
        if observation == "denied":
            raise PermissionError("controlled observation denial")
        return None if observation == "missing" else {**identity, "start_ticks": identity["start_ticks"] + 1}

    monkeypatch.setattr(queue, "process_identity", observe)
    monkeypatch.setattr(supervision, "_pidfd_signal", lambda *args: sent.append(args))
    try:
        result = run(tmp_path, child, command, identity, path)
        assert result["workers_absent"] is False
        assert result["state"] == "failed"
        assert sent == []
        assert child.poll() is None
    finally:
        monkeypatch.undo()
        cleanup(child)


def test_worker_inventory_timeout_preserves_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(20)\n", env)

    def inventory(record: object, timeout: float) -> Never:
        del record
        assert timeout == 0.15
        raise subprocess.TimeoutExpired("controlled-worker-inventory", timeout)

    monkeypatch.setattr(supervision, "_worker_inventory", inventory)
    try:
        result = run(tmp_path, child, command, identity, path)
        assert result["workers_absent"] is False
        assert result["events"][-1]["event"] == "worker_inventory_failed"
        assert result["elapsed_s"] < 2
    finally:
        cleanup(child)


def test_gpu_inventory_uses_separate_bound_and_finite_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(20)\n", env)

    def inventory(gpus: tuple[int, ...], timeout: float) -> dict[int, int]:
        assert gpus == (0,)
        assert timeout == 0.15
        return {0: 123}

    monkeypatch.setattr(supervision, "_gpu_inventory", inventory)
    try:
        result = run(tmp_path, child, command, identity, path, sampled_total_device_limit_bytes=122)
        assert "sampled total-device stop guard" in result["error"]
        assert result["sampled_total_device_bytes"][0]["gpus"] == {"0": 123}
        assert "memory_limit_allocated_bytes" not in result["limits"]
    finally:
        cleanup(child)


@pytest.mark.parametrize(("field", "value"), [(TOKEN_ENV, "4" * 32), (JOB_ENV, "5" * 64)])
def test_notification_attempt_mismatch_refuses(tmp_path: Path, field: str, value: str) -> None:
    path, env = contract(tmp_path)
    with pytest.raises(ValueError, match="bound attempt"):
        supervision.notify_phase("load", "begin", 0, budget_sha256=BUDGET, environment={**env, field: value})
    assert list(path.with_name(path.name + ".events").iterdir()) == []


def test_notification_budget_mismatch_and_duplicate_refuse(tmp_path: Path) -> None:
    _path, env = contract(tmp_path)
    with pytest.raises(ValueError, match="bound attempt"):
        supervision.notify_phase("load", "begin", 0, budget_sha256="6" * 64, environment=env)
    supervision.notify_phase("load", "begin", 0, budget_sha256=BUDGET, environment=env)
    with pytest.raises(FileExistsError):
        supervision.notify_phase("load", "begin", 0, budget_sha256=BUDGET, environment=env)


def test_consumed_notification_mutation_refuses(tmp_path: Path) -> None:
    path, env = contract(tmp_path)
    supervision.notify_phase("load", "begin", 0, budget_sha256=BUDGET, environment=env)
    value, digest = supervision._contract(path)
    progress = {0: {"index": 0, "active": None, "awaiting_since": 0, "identity": None}}
    seen = {}
    supervision._consume(path, value, digest, progress, seen, 0.1)
    event = next(path.with_name(path.name + ".events").iterdir())
    event.write_text("{}")
    with pytest.raises(ValueError, match="changed or disappeared"):
        supervision._consume(path, value, digest, progress, seen, 0.2)


def test_inventory_calls_only_own_registry_with_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    seen = []

    class OwnRegistry:
        def observe_workers(self, *, timeout: float) -> dict:
            seen.append(timeout)
            return {"schema_version": 1, "workers_live": False, "complete": True}

    assert supervision._worker_inventory(OwnRegistry(), 0.2)["workers_live"] is False
    assert seen == [0.2]


def test_notification_declared_rank_mismatch_refuses(tmp_path: Path) -> None:
    _, env = contract(tmp_path, world=2)
    with pytest.raises(ValueError, match="bound attempt"):
        supervision.notify_phase("load", "begin", 0, budget_sha256=BUDGET, environment={**env, "RANK": "1"})


def test_out_of_order_end_is_not_a_phase_begin(tmp_path: Path) -> None:
    path, env = contract(tmp_path)
    supervision.notify_phase("load", "end", 0, budget_sha256=BUDGET, environment=env)
    value, digest = supervision._contract(path)
    progress = {0: {"index": 0, "active": None, "awaiting_since": 0, "identity": None}}
    with pytest.raises(ValueError, match="out-of-order"):
        supervision._consume(path, value, digest, progress, {}, 0.1)


def test_notification_wrong_saved_rank_refuses(tmp_path: Path) -> None:
    path, env = contract(tmp_path)
    supervision.notify_phase("load", "begin", 0, budget_sha256=BUDGET, environment=env)
    event = next(path.with_name(path.name + ".events").iterdir())
    record = json.loads(event.read_bytes())
    event.write_text(json.dumps({**record, "rank": 1}))
    value, digest = supervision._contract(path)
    progress = {0: {"index": 0, "active": None, "awaiting_since": 0, "identity": None}}
    with pytest.raises(ValueError, match="rank/order/bindings"):
        supervision._consume(path, value, digest, progress, {}, 0.1)


def test_one_process_cannot_supply_two_native_ranks(tmp_path: Path) -> None:
    path, env = contract(tmp_path, world=2)
    for rank in (0, 1):
        supervision.notify_phase("load", "begin", rank, budget_sha256=BUDGET, environment=env)
    value, digest = supervision._contract(path)
    progress = {rank: {"index": 0, "active": None, "awaiting_since": 0, "identity": None} for rank in (0, 1)}
    with pytest.raises(ValueError, match="identity/rank"):
        supervision._consume(path, value, digest, progress, {}, 0.1)


def test_changed_contract_is_not_adopted_during_child_run(tmp_path: Path) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(
        tmp_path,
        f"Path(READY).write_text('ready')\ntime.sleep(.05)\nPath({str(path)!r}).write_text('{{}}')\ntime.sleep(20)\n",
        env,
    )
    try:
        result = run(tmp_path, child, command, identity, path)
        assert "contract is invalid or changed" in result["error"]
        assert result["state"] == "failed"
        assert result["elapsed_s"] < 2
    finally:
        cleanup(child)


def test_explicit_overall_guard_is_independent(tmp_path: Path) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(20)\n", env)
    try:
        result = run(tmp_path, child, command, identity, path, startup_seconds=1, overall_seconds=0.05)
        assert "overall_deadline" in result["error"]
        assert result["limits"]["overall_seconds"] == 0.05
        assert result["limits"]["startup_seconds"] == 1
    finally:
        cleanup(child)


def test_missing_pidfd_support_refuses_signal_and_keeps_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(20)\n", env)
    try:
        monkeypatch.setattr(supervision, "_pidfd_open", lambda _pid: (_ for _ in ()).throw(OSError("no pidfd")))
        result = run(tmp_path, child, command, identity, path)
        assert "no pidfd" in result["error"]
        assert result["workers_absent"] is False
        assert child.poll() is None
    finally:
        monkeypatch.undo()
        cleanup(child)


def test_gpu_inventory_timeout_fails_current_supervision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, env = contract(tmp_path)
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(20)\n", env)
    try:
        monkeypatch.setattr(
            supervision,
            "_gpu_inventory",
            lambda _gpus, timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", timeout)),
        )
        result = run(tmp_path, child, command, identity, path, sampled_total_device_limit_bytes=123)
        assert "TimeoutExpired" in result["error"]
        assert result["state"] == "failed"
        assert result["elapsed_s"] < 2
    finally:
        cleanup(child)


def test_plain_command_requires_explicit_overall_guard(tmp_path: Path) -> None:
    _, env = contract(tmp_path)
    child, command, identity = spawn(tmp_path, "Path(READY).write_text('ready')\ntime.sleep(20)\n", env)
    try:
        with pytest.raises(ValueError, match="explicit overall"):
            run(tmp_path, child, command, identity, None)
        result = run(tmp_path, child, command, identity, None, overall_seconds=0.05)
        assert result["state"] == "failed"
        assert "overall_deadline" in result["error"]
        assert result["limits"]["phase_seconds"] is None
        assert result["limits"]["startup_seconds"] is None
        assert result["notification_contract_sha256"] is None
        assert result["phase_observations"] == []
    finally:
        cleanup(child)
