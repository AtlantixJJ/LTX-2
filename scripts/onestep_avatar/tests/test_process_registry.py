"""CPU ownership controls: shared own ledger, direct device queries, no GPU work."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from scripts.onestep_avatar.execution import process_registry, supervision


def free():
    return {gpu: 0 for gpu in range(6)}


def registry(tmp_path):
    return process_registry.ProcessRegistry(tmp_path / "own_processes.json", inventory=free)


def stop(child):
    if child.poll() is None:
        descriptor = supervision._pidfd_open(child.pid)
        try:
            supervision._pidfd_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
    child.wait(timeout=2)


def test_direct_inventory_rechecked_and_shared_starts_coordinate(tmp_path):
    owner = registry(tmp_path)
    assert owner.choose(free(), training=True) == (0, 1, 2, 3)
    owner.inventory = lambda: {**free(), 0: 2048}
    assert owner.acquire((0, 1, 2, 3), job="first") is False
    assert not owner.path.exists()
    owner.inventory = free
    assert owner.acquire((0, 1, 2, 3), job="first") is True
    second = registry(tmp_path)
    assert second.acquire((0, 1, 2, 3), job="second") is False
    assert second.choose(free(), training=True) is None
    owner.release()
    assert owner.owned == set()
    record = json.loads(owner.path.read_bytes())
    assert record["attempts"][owner.token]["state"] == "closed"
    assert second.acquire((0, 1, 2, 3), job="second") is True
    second.release()


def test_real_new_session_orphan_is_adopted_and_blocks_close(tmp_path):
    owner = registry(tmp_path)
    assert owner.acquire((0,), job="orphan-control")
    ready = tmp_path / "worker.json"
    command = [sys.executable, "-c", "import json, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], "
        "start_new_session=True)\n"
        f"Path({str(ready)!r}).write_text(json.dumps({{'pid': worker.pid}}))\n"
        "time.sleep(.2)\n"]
    child = subprocess.Popen(command, start_new_session=True)
    worker_pid = None
    try:
        owner.refresh(child_pid=child.pid)
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        worker_pid = json.loads(ready.read_bytes())["pid"]
        assert os.getsid(worker_pid) == worker_pid
        child.wait(timeout=2)
        observed = owner.observe_workers(timeout=1)
        assert observed["complete"] is True and observed["workers_live"] is True
        assert worker_pid in [identity["pid"] for identity in observed["identities"]]
        with pytest.raises(ValueError, match="remain live"):
            owner.release()
        descriptor = supervision._pidfd_open(worker_pid)
        try:
            supervision._pidfd_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            found, _ = os.waitpid(worker_pid, os.WNOHANG)
            if found == worker_pid:
                worker_pid = None
                break
            time.sleep(.01)
        assert worker_pid is None
        owner.release()
        assert process_registry.inspect_record(owner.path, owner.token)["complete"] is True
        assert process_registry.workers_live(owner.path, owner.token) is False
        assert json.loads(owner.path.read_bytes())["attempts"][owner.token]["processes"]
    finally:
        stop(child)
        if worker_pid is not None:
            descriptor = supervision._pidfd_open(worker_pid)
            try:
                supervision._pidfd_signal(descriptor, signal.SIGKILL)
            finally:
                os.close(descriptor)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if os.waitpid(worker_pid, os.WNOHANG)[0] == worker_pid:
                    break
                time.sleep(.01)


def test_other_thread_children_are_in_targeted_scope(tmp_path):
    owner = registry(tmp_path)
    assert owner.acquire((0,), job="thread-fork-control")
    ready, end = threading.Event(), threading.Event()
    children = []

    def launch():
        children.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"]))
        ready.set()
        end.wait(timeout=3)

    thread = threading.Thread(target=launch)
    thread.start()
    try:
        assert ready.wait(timeout=2)
        observed = owner.observe_workers(timeout=1)
        assert observed["complete"] is True and observed["workers_live"] is True
        assert children[0].pid in [identity["pid"] for identity in observed["identities"]]
        with pytest.raises(ValueError, match="remain live"):
            owner.release()
    finally:
        end.set()
        thread.join(timeout=2)
        stop(children[0])
        owner.release()


@pytest.mark.parametrize("change", ["reuse", "denied", "owner_gone"])
def test_handle_uncertainty_preserves_record(tmp_path, monkeypatch, change):
    owner = registry(tmp_path)
    assert owner.acquire((0,), job="identity-control")
    original = process_registry._identity

    def observe(pid):
        if pid != owner.owner["pid"]:
            return original(pid)
        if change == "owner_gone":
            return None
        if change == "denied":
            raise PermissionError("controlled own-handle denial")
        return {**owner.owner, "start_ticks": owner.owner["start_ticks"] + 1}

    monkeypatch.setattr(process_registry, "_identity", observe)
    result = owner.observe_workers(timeout=1)
    assert result["complete"] is False and result["workers_live"] is True
    with pytest.raises(ValueError, match="containment is unproven"):
        owner.release()
    assert json.loads(owner.path.read_bytes())["attempts"][owner.token]["state"] == "active"
    assert process_registry.workers_live(owner.path, owner.token) is True


def test_live_notification_producer_must_be_own_descendant(tmp_path):
    owner = registry(tmp_path)
    assert owner.acquire((0,), job="rank-control")
    with pytest.raises(ValueError, match="owned descendant"):
        owner.register_identity(owner.owner, role="rank:0")
    owner.release()


def test_late_dead_unknown_notification_is_only_reported(tmp_path):
    owner = registry(tmp_path)
    assert owner.acquire((0,), job="late-rank-control")
    identity = {"pid": 2147483647, "start_ticks": 1, "command": ["reported-rank"], "terminal": False}
    owner.register_identity(identity, role="rank:0")
    row = json.loads(owner.path.read_bytes())["attempts"][owner.token]
    assert str(identity["pid"]) not in row["processes"]
    assert row["reported_rank_identities"] == [{"identity": identity, "role": "rank:0"}]
    owner.release()


def test_target_tree_observation_has_finite_deadline(tmp_path):
    owner = registry(tmp_path)
    assert owner.acquire((0,), job="deadline-control")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        owner.refresh(child_pid=child.pid)
        observed = owner.observe_workers(timeout=1e-12)
        assert observed["complete"] is False and "TimeoutError" in observed["error"]
    finally:
        stop(child)
        owner.release()


def test_shared_ledger_lock_wait_has_finite_deadline(tmp_path, monkeypatch):
    owner = registry(tmp_path)
    real_clock = time.monotonic
    moments = iter((0, 6))
    monkeypatch.setattr(process_registry.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(fcntl, "flock", lambda *args: (_ for _ in ()).throw(BlockingIOError()))
    started = real_clock()
    with pytest.raises(TimeoutError, match="lock deadline"):
        owner.choose(free(), training=True)
    assert real_clock() - started < 1


def test_oversized_and_nonregular_ledger_refuse(tmp_path):
    owner = registry(tmp_path)
    owner.path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="8 MiB"):
        owner.choose(free(), training=True)
    owner.path.unlink()
    os.mkfifo(owner.path)
    started = time.monotonic()
    with pytest.raises(ValueError, match="regular JSON file"):
        owner.choose(free(), training=True)
    assert time.monotonic() - started < 1
