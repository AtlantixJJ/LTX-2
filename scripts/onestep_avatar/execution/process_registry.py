"""Track our exact child processes in one ledger; see doc/execution/process_registry.md."""

from __future__ import annotations

import ctypes
import fcntl
import json
import math
import os
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path


def _identity(pid: int) -> dict | None:
    from scripts.onestep_avatar.execution.queue import process_identity  # noqa: PLC0415 -- single stable identity owner

    return process_identity(pid)


def _same_handle(left: dict | None, right: dict) -> bool:
    return left is not None and all(left.get(key) == right.get(key) for key in ("pid", "start_ticks"))


def _subreaper(*, enable: bool = False) -> bool:
    """Enable/query containment in this owner, without process privileges."""
    libc = ctypes.CDLL(None, use_errno=True)
    if enable and libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    current = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return current.value == 1


def gpu_memory(*, timeout: float = 5) -> dict[int, int]:
    """Read direct device occupancy in MiB, with a finite command timeout."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("GPU inventory timeout must be finite and positive")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=timeout,
    )
    from scripts.onestep_avatar.execution.queue import parse_gpu_memory  # noqa: PLC0415 -- existing inventory contract

    return parse_gpu_memory(result.stdout)


class ProcessRegistry:
    """Coordinate own starts and retain exact handles; never read reservation files."""

    def __init__(self, path: Path, *, inventory: Callable[[], dict[int, int]] | None = None):
        self.path = path.resolve()
        self.inventory = gpu_memory if inventory is None else inventory
        self.token = uuid.uuid4().hex
        self.started_ticks = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))
        self.owned: set[int] = set()
        self.owner = _identity(os.getpid())
        if self.owner is None or self.owner["terminal"] or not _subreaper(enable=True):
            raise ValueError("process registry requires its live original subreaper owner")

    @contextmanager
    def _lock(self, *, timeout: float = 5):  # noqa: ANN202 -- one stable inode locks atomic ledger replacements
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_name(self.path.name + ".lock").open("a") as handle:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("shared process ledger lock deadline exceeded") from None
                    time.sleep(0.01)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "attempts": {}}
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
                raise ValueError("shared process ledger must be a regular JSON file within 8 MiB")
            data = handle.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("shared process ledger exceeds 8 MiB")
        record = json.loads(data)
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record["schema_version"] != 1 or not isinstance(record.get("attempts"), dict)):
            raise ValueError("invalid shared process ledger")
        return record

    def _write(self, record: dict) -> None:
        data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("shared process ledger exceeds 8 MiB")
        with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".process-ledger-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _children(pid: int) -> list[int]:
        try:
            tasks = list(Path(f"/proc/{pid}/task").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            return []
        children = set()
        for task in tasks:
            try:
                raw = (task / "children").read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            children.update(int(value) for value in raw.split())
        if any(value <= 0 for value in children):
            raise ValueError("invalid targeted child inventory")
        return sorted(children)

    def _baseline(self) -> list[dict]:
        identities = [_identity(pid) for pid in self._children(self.owner["pid"])]
        return [identity for identity in identities if identity is not None]

    @staticmethod
    def _remember(row: dict, identity: dict, role: str) -> None:
        key = str(identity["pid"])
        previous = row["processes"].get(key)
        if previous is not None and not _same_handle(identity, previous["identity"]):
            raise ValueError("registered process PID was reused")
        if previous is None:
            previous = {"identity": identity, "roles": [], "commands": []}
            row["processes"][key] = previous
        previous["identity"] = identity
        if role not in previous["roles"]:
            previous["roles"].append(role)
        if identity["command"] and identity["command"] not in previous["commands"]:
            previous["commands"].append(identity["command"])

    def choose(self, memory: dict[int, int], *, training: bool) -> tuple[int, ...] | None:
        from scripts.onestep_avatar.execution.queue import (  # noqa: PLC0415 -- preserve existing GPU selection
            ALLOWED_GPUS,
            EVALUATION_PREFERENCE,
            TRAIN_GPUS,
        )

        if not ALLOWED_GPUS.issubset(memory) or any(type(value) is not int or value < 0 for value in memory.values()):
            raise ValueError("incomplete direct GPU memory inventory")
        with self._lock():
            records = self._read()["attempts"].values()
            busy = {gpu for row in records if row["state"] != "closed" for gpu in row["gpus"]}
        free = {gpu for gpu in ALLOWED_GPUS if memory[gpu] < 1024 and gpu not in busy}
        if training:
            return TRAIN_GPUS if set(TRAIN_GPUS).issubset(free) else None
        return next(((gpu,) for gpu in EVALUATION_PREFERENCE if gpu in free), None)

    def acquire(self, gpus: tuple[int, ...], *, job: str) -> bool:
        from scripts.onestep_avatar.execution.queue import ALLOWED_GPUS  # noqa: PLC0415 -- existing device restriction

        if self.owned or not gpus or len(set(gpus)) != len(gpus) or not set(gpus).issubset(ALLOWED_GPUS):
            raise ValueError("own process start requires unique allowed GPUs and a fresh attempt")
        with self._lock():
            record = self._read()
            if self.token in record["attempts"]:
                raise ValueError("a completed attempt requires a fresh process registry instance")
            if any(row["state"] != "closed" and (set(gpus).intersection(row["gpus"])
                   or _same_handle(row["owner"], self.owner)) for row in record["attempts"].values()):
                return False
            memory = self.inventory()
            if not ALLOWED_GPUS.issubset(memory) or any(
                    type(value) is not int or value < 0 for value in memory.values()):
                raise ValueError("incomplete direct GPU memory inventory")
            if any(memory[gpu] >= 1024 for gpu in gpus):
                return False
            record["attempts"][self.token] = {
                "state": "active", "owner": self.owner, "gpus": sorted(gpus), "job": job,
                "attempt_started_ticks": self.started_ticks, "baseline": self._baseline(),
                "containment": "linux_subreaper_v1", "child_pid": None, "processes": {},
                "observations": [], "reported_rank_identities": [],
            }
            self._write(record)
            self.owned = set(gpus)
        return True

    def _observe(self, row: dict, timeout: float) -> dict:  # noqa: PLR0912 -- complete targeted containment gates
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("owned worker observation timeout must be finite and positive")
        result = {"schema_version": 1, "workers_live": True, "complete": False,
                  "identities": [], "error": None}
        deadline = time.monotonic() + timeout
        try:
            owner = _identity(row["owner"]["pid"])
            if (not _same_handle(owner, row["owner"]) or owner["terminal"]
                    or owner["command"] != row["owner"]["command"] or os.getpid() != owner["pid"]
                    or row.get("containment") != "linux_subreaper_v1" or not _subreaper()):
                raise ValueError("original live owner containment is unavailable")
            observed = {}
            for _ in range(2):
                pending = self._children(owner["pid"])
                visited = set()
                while pending:
                    if time.monotonic() >= deadline or len(visited) >= 4096:
                        raise TimeoutError("targeted own-worker tree observation exceeded its bound")
                    pid = pending.pop()
                    if pid in visited:
                        continue
                    visited.add(pid)
                    identity = _identity(pid)
                    if identity is None or any(_same_handle(identity, item) for item in row["baseline"]):
                        continue
                    self._remember(row, identity, "descendant")
                    observed[pid] = identity
                    if not identity["terminal"]:
                        pending.extend(self._children(pid))
            # A registered handle may disappear from the tree after it exits.
            for process in row["processes"].values():
                identity = _identity(process["identity"]["pid"])
                if identity is not None:
                    if not _same_handle(identity, process["identity"]):
                        raise ValueError("registered worker PID was reused")
                    self._remember(row, identity, "observed")
                    observed[identity["pid"]] = identity
            # Orphans adopted after the last traversal still cannot be omitted
            # from an empty-tree conclusion.
            for pid in self._children(owner["pid"]):
                identity = _identity(pid)
                if identity is not None and not any(_same_handle(identity, item) for item in row["baseline"]):
                    self._remember(row, identity, "descendant")
                    observed[pid] = identity
            final_owner = _identity(owner["pid"])
            if (not _same_handle(final_owner, owner) or final_owner["terminal"]
                    or final_owner["command"] != owner["command"] or not _subreaper()):
                raise ValueError("original owner containment changed during observation")
            result.update(complete=True, identities=list(observed.values()),
                          workers_live=any(not identity["terminal"] for identity in observed.values()))
        except (OSError, ValueError, TimeoutError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        row["observations"].append(result)
        # Keep a bounded recent observer log; exact processes/commands are permanent.
        row["observations"] = row["observations"][-32:]
        return result

    def refresh(self, *, child_pid: int | None = None) -> None:
        if not self.owned:
            raise ValueError("own process refresh requires an active registered attempt")
        with self._lock():
            record = self._read()
            row = record["attempts"][self.token]
            if child_pid is not None:
                if row["child_pid"] not in (None, child_pid):
                    raise ValueError("registered launch child PID changed")
                identity = _identity(child_pid)
                if identity is None and row["child_pid"] is None:
                    raise ValueError("new launch child identity is unavailable")
                row["child_pid"] = child_pid
                if identity is not None:
                    self._remember(row, identity, "child")
            self._observe(row, 5)
            self._write(record)

    def register_identity(self, identity: dict, *, role: str) -> None:
        """Attach a bound notification's rank role to an existing own handle."""
        with self._lock():
            record = self._read()
            row = record["attempts"][self.token]
            observation = self._observe(row, 5)
            current = _identity(identity["pid"])
            if current is None and str(identity["pid"]) not in row["processes"]:
                # A late complete event can report an already-gone producer.
                # Preserve its role without adopting unknown live ownership.
                row["reported_rank_identities"].append({"identity": identity, "role": role})
                self._write(record)
                return
            if current is not None and (not _same_handle(current, identity)
                    or str(identity["pid"]) not in row["processes"]):
                raise ValueError("notification producer is not an exact owned descendant")
            if not observation["complete"]:
                raise ValueError("notification producer containment is incomplete")
            self._remember(row, identity, role)
            self._write(record)

    def observe_workers(self, *, timeout: float = 5) -> dict:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("owned worker observation timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        with self._lock(timeout=timeout):
            record = self._read()
            row = record["attempts"][self.token]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = {"schema_version": 1, "workers_live": True, "complete": False,
                          "identities": [], "error": "TimeoutError: targeted own-worker lock/read deadline exceeded"}
                row["observations"].append(result)
            else:
                result = self._observe(row, remaining)
            self._write(record)
            return result

    def registered_identities(self, *, timeout: float = 5) -> list[dict]:
        """Return this attempt's exact recorded descendants as stop candidates."""
        with self._lock(timeout=timeout):
            row = self._read()["attempts"][self.token]
            return [process["identity"] for process in row["processes"].values()]

    def release(self) -> None:
        """Close only proved-empty live-owner containment; preserve all history."""
        if not self.owned:
            return
        with self._lock():
            record = self._read()
            row = record["attempts"][self.token]
            observation = self._observe(row, 5)
            if not observation["complete"] or observation["workers_live"]:
                self._write(record)
                raise ValueError("own processes remain live or containment is unproven; ledger retained")
            row["state"] = "closed"
            self._write(record)
            self.owned.clear()


def inspect_record(path: Path, token: str, *, timeout: float = 5) -> dict:
    """Read one saved own record; another observer cannot inherit containment."""
    registry = object.__new__(ProcessRegistry)
    registry.path, registry.token = path.resolve(), token
    record = registry._read()
    if token not in record["attempts"]:
        raise ValueError("saved own process token is absent from the shared ledger")
    row = record["attempts"][token]
    if row["state"] == "closed":
        return {"schema_version": 1, "workers_live": False, "complete": True,
                "identities": [], "error": None, "basis": "saved_complete_empty_containment"}
    return registry._observe(row, timeout)


def workers_live(path: Path, token: str, *, timeout: float = 5) -> bool:
    """Incomplete own evidence remains live/unknown; never scan other environments."""
    observation = inspect_record(path, token, timeout=timeout)
    return not observation["complete"] or observation["workers_live"]


def observe_attempt(path: Path, token: str, *, timeout: float = 5) -> dict:
    """Read-only saved-row interface used by the existing queue owner."""
    return inspect_record(path, token, timeout=timeout)
