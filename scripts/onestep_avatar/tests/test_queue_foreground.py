"""Human queue controls use live logs and the existing exact-worker shutdown."""

import io
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from scripts.onestep_avatar.execution import attempt as queue_attempt
from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.execution import process_registry, queue, supervision, terminal
from scripts.onestep_avatar.execution import processes as queue_processes


def test_log_streams_before_child_exit_and_drains_split_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "child.log"
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    with path.open("wb") as log, terminal.follow_log(path, enabled=True):
        log.write(b"first line\n\xe2")
        log.flush()
        deadline = time.monotonic() + 2
        while "first line" not in output.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert output.getvalue() == "first line\n"
        log.write(b"\x82\xac final bytes")
        log.flush()
    assert output.getvalue() == "first line\n€ final bytes"
    assert path.read_bytes() == "first line\n€ final bytes".encode()


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_cli_stop_saves_failure_and_ends_real_child_and_escaped_worker(  # noqa: PLR0915 -- end-to-end cancellation evidence
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stop_signal: int,
) -> None:
    """Real CPU handles, guard, ledger, supervisor and journal; no native model/GPU."""
    ready = tmp_path / "ready.json"
    output = tmp_path / "output"
    state_path = tmp_path / "state.json"
    ledger = tmp_path / "processes.json"
    log = tmp_path / "child.log"
    job = {
        "id": "human",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(output),
        "arguments": [],
        "dependencies": [],
        "completion": {"records": []},
    }
    code = (
        "import json, os, subprocess, sys, time\nfrom pathlib import Path\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        f"Path({str(output)!r}).mkdir()\n"
        f"Path({str(output / 'partial.txt')!r}).write_text('preserve')\n"
        "print('human child output', flush=True)\n"
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(), worker.pid]))\n"
        "time.sleep(60)\n"
    )
    command = [sys.executable, "-c", code]
    monkeypatch.setattr(queue_jobs, "job_command", lambda _job, _gpus: (command, {}))
    monkeypatch.setattr(
        queue_jobs, "execution_profile", lambda _job: {"world": 1, "phases": ["runtime"], "generic_phase": False}
    )
    monkeypatch.setattr(queue_completion, "completion_receipt", lambda *_a, **_k: pytest.fail("cancelled receipt"))
    original_observe = supervision._observe
    sent = []

    def observe(*args: object) -> dict | None:
        value = original_observe(*args)
        if ready.exists() and not sent:
            # Popen returns before the worker finishes exec. Interrupt only after
            # the ledger has observed its stable command; ambiguous startup has
            # separate fail-closed supervision tests.
            owned = next(iter(json.loads(ledger.read_text())["attempts"].values()))
            worker_pid = json.loads(ready.read_text())[1]
            worker = owned["processes"].get(str(worker_pid), {})
            if worker.get("identity", {}).get("command") == [sys.executable, "-c", "import time; time.sleep(60)"]:
                sent.append(stop_signal)
                os.kill(os.getpid(), stop_signal)
        return value

    monkeypatch.setattr(supervision, "_observe", observe)

    def execute(_jobs: Path, _state: Path, _ledger: Path, **options: object) -> dict:
        assert options["foreground"] is True
        assert options["once"] is False
        claims = process_registry.ProcessRegistry(ledger, inventory=lambda: dict.fromkeys(range(8), 0))
        assert claims.acquire((0,), job=job["id"])
        return queue_attempt.run_child(
            job, claims, log, state_path=state_path, jobs=[job], poll_seconds=0.01, foreground=True
        )

    monkeypatch.setattr(queue, "execute_jobs", execute)
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    try:
        assert (
            queue.main(
                [
                    "--jobs",
                    str(tmp_path / "jobs.json"),
                    "--state",
                    str(state_path),
                    "--foreground",
                    "--process-ledger",
                    str(ledger),
                ]
            )
            == 128 + stop_signal
        )
        assert time.monotonic() - started < 10
        assert sent == [stop_signal]
        assert "human child output" in capsys.readouterr().out
        assert "human child output" in log.read_text()
        row = json.loads(state_path.read_text())["jobs"]["human"]
        assert row["state"] == "failed"
        assert row["failure_reason"] == "cancelled"
        assert "receipt" not in row
        assert row["environment_changes"]["PYTHONUNBUFFERED"] == "1"
        evidence = json.loads(Path(row["supervision_result"]).read_text())
        assert evidence["state"] == "failed"
        assert evidence["workers_absent"] is True
        assert evidence["error"].startswith("KeyboardInterrupt:")
        assert (output / "partial.txt").read_text() == "preserve"
        for pid in json.loads(ready.read_text()):
            identity = queue_processes.process_identity(pid)
            assert identity is None or identity["terminal"]
        assert all(row["state"] == "closed" for row in json.loads(ledger.read_text())["attempts"].values())
        assert {number: signal.getsignal(number) for number in previous} == previous
    finally:
        if ready.exists():
            for pid in json.loads(ready.read_text()):
                identity = queue_processes.process_identity(pid)
                if identity is not None and not identity["terminal"]:
                    os.kill(pid, signal.SIGKILL)


def test_repeated_stop_signal_does_not_interrupt_cleanup() -> None:
    with terminal.stop_signals() as received:
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        assert received == [signal.SIGINT]


@pytest.mark.parametrize("extra", [["--execute"], ["--dry-run"], ["--recover"], []])
def test_foreground_rejects_conflicting_modes_or_missing_ledger(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        queue.main(
            ["--jobs", str(tmp_path / "missing.json"), "--state", str(tmp_path / "state.json"), "--foreground", *extra]
        )
    assert not list(tmp_path.iterdir())
