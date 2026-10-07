"""Observe real unreaped children, without mistaking surviving workers for death."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from scripts.onestep_avatar import queue
from scripts.onestep_avatar.queue_protocol import TOKEN_ENV

T = TypeVar("T")


def until(predicate: Callable[[], T]) -> T:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    pytest.fail("controlled child observation timed out")


@pytest.mark.parametrize("worker_survives", [False, True])
def test_unreaped_real_child_recovers_only_after_all_owned_workers_exit(  # noqa: PLR0915 -- real process lifecycle
    tmp_path: Path, worker_survives: bool
) -> None:
    ready, release, worker_file = tmp_path / "ready", tmp_path / "release", tmp_path / "worker.json"
    code = """import os,sys,time,json,subprocess
from pathlib import Path
from scripts.onestep_avatar.queue import process_identity
root=Path(sys.argv[1])
if sys.argv[2]=='yes':
    worker=subprocess.Popen(
        [sys.executable,'-c','import time; print("ready",flush=True); time.sleep(60)'],
        start_new_session=True,stdout=subprocess.PIPE,text=True)
    assert worker.stdout.readline().strip()=='ready'
    (root/'worker.json').write_text(json.dumps(process_identity(worker.pid)))
(root/'ready').write_text('ready')
while not (root/'release').exists(): time.sleep(.01)
"""
    claims = queue.GPUClaims(tmp_path / "claims")
    assert claims.acquire((4,), job="zombie")
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path), "yes" if worker_survives else "no"],
        env={**os.environ, TOKEN_ENV: claims.token},
        start_new_session=True,
    )
    worker = None
    row = None
    try:
        until(ready.exists)
        saved = queue.process_identity(child.pid)
        assert saved is not None
        assert not saved["terminal"]
        assert saved["command"]
        if worker_survives:
            worker = json.loads(worker_file.read_text())
        claims.refresh(child_pid=child.pid)
        row = {
            "sha256": "a" * 64,
            "state": "running",
            "attempts": [],
            "child_pid": child.pid,
            "child_identity": saved,
            "child_session": child.pid,
            "environment_changes": {TOKEN_ENV: claims.token},
            "attempt_started_ticks": claims.started_ticks,
        }
        job = {
            "id": "zombie",
            "sha256": "a" * 64,
            "kind": "evaluate",
            "dependencies": [],
            "output": str(tmp_path / "output"),
            "arguments": ["--mode", "causal"],
            "completion": {"records": [str(tmp_path / "output/result.json")]},
        }
        state_path = tmp_path / "state.json"
        with queue.queue_state(state_path, [job]) as state:
            state["jobs"]["zombie"] = row
        original = state_path.read_bytes()
        release.write_text("exit")
        # Do not poll/wait: Popen would reap the child and erase the regression case.
        terminal = until(
            lambda: identity if (identity := queue.process_identity(child.pid)) and identity["terminal"] else None
        )
        assert terminal["command"] == []
        assert terminal["start_ticks"] == saved["start_ticks"]
        assert queue.inspect_child(row) == ("live" if worker_survives else "terminal")
        if worker_survives:
            with pytest.raises(ValueError, match="surviving child"), queue.queue_state(state_path, [job], recover=True):
                pytest.fail("recovered a surviving worker")
            assert state_path.read_bytes() == original
            with pytest.raises(ValueError, match="live child"):
                claims.release()
            assert claims.owned == {4}
            assert (tmp_path / "claims/4").is_file()
            current = queue.process_identity(worker["pid"])
            assert current["start_ticks"] == worker["start_ticks"]
            os.kill(worker["pid"], signal.SIGTERM)
            until(lambda: not queue.owned_workers_live(row))
        with queue.queue_state(state_path, [job], recover=True) as state:
            assert state["jobs"]["zombie"]["state"] == "failed"
            assert "before verified completion" in state["jobs"]["zombie"]["error"]
        assert queue.inspect_child(row) == "terminal"
        claims.release()
        assert not claims.owned
        assert not (tmp_path / "claims/4").exists()
        assert child.wait(timeout=5) == 0
    finally:
        if worker is None and worker_file.exists():
            worker = json.loads(worker_file.read_text())
        if worker is not None:
            current = queue.process_identity(worker["pid"])
            if current and not current["terminal"] and current["start_ticks"] == worker["start_ticks"]:
                os.kill(worker["pid"], signal.SIGTERM)
        if child.returncode is None:
            child.terminate()
            child.wait(timeout=5)
        if claims.owned:
            until(lambda: row is None or not queue.owned_workers_live(row))
            claims.release()


@pytest.mark.parametrize("change", ["ticks", "missing_command", "live_empty", "terminal_changed"])
def test_empty_terminal_command_cannot_authorize_other_handle_changes(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    saved = {"pid": 42, "start_ticks": 100, "command": ["python", "child.py"], "terminal": False}
    current = {**saved, "command": [], "terminal": True}
    if change == "ticks":
        current["start_ticks"] += 1
    if change == "missing_command":
        del saved["command"]
    if change == "live_empty":
        current["terminal"] = False
    if change == "terminal_changed":
        current["command"] = ["other"]
    monkeypatch.setattr(queue, "process_identity", lambda _: current)
    with pytest.raises(ValueError, match="identity differs"):
        queue.inspect_child({"child_pid": 42, "child_identity": saved})
