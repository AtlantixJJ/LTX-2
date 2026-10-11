"""File-based GPU claims for queue lifecycle tests.

Production dispatch claims devices through ``execution.process_registry.ProcessRegistry``.
This helper offers the same interface (``token``, ``owned``, ``started_ticks``, ``choose``,
``acquire``, ``refresh``, ``release``) so tests can drive ``queue.run_child`` without a
shared ledger.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from scripts.onestep_avatar.execution.devices import GPU_POOL, select_devices
from scripts.onestep_avatar.execution.processes import live_attempt_processes, live_session_processes, process_alive


class GPUClaims:
    """Serialize reservations and remove only claims carrying this attempt's token."""

    def __init__(self, directory: Path, *, ttl: float = 600):
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("claim TTL must be positive")
        self.directory = directory
        self.ttl = ttl
        self.token = uuid.uuid4().hex
        self.started_ticks = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))
        self.owned: set[int] = set()

    @contextmanager
    def _lock(self):  # noqa: ANN202 -- private context manager
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".claims.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self, gpu: int) -> tuple[dict | None, float] | None:
        path = self.directory / str(gpu)
        try:
            # Use one open inode for both content and age.
            with path.open() as handle:
                stamp = os.fstat(handle.fileno()).st_mtime
                raw = handle.read()
        except FileNotFoundError:
            return None
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            record = None  # Non-JSON claims use the TTL alone.
        return (record if isinstance(record, dict) else None), stamp

    def _active(self, claim: tuple[dict | None, float]) -> bool:
        record, stamp = claim
        if time.time() - stamp < self.ttl:
            return True
        return (
            record is not None and (
                process_alive(record.get("owner_pid")) or process_alive(record.get("child_pid"))
                or (record.get("child_session") is not None and bool(live_session_processes(record["child_session"])))
                or (record.get("token") is not None and bool(live_attempt_processes(
                    record["token"], started_ticks=record.get("attempt_started_ticks")
                )))
            )
        )

    def choose(self, memory: dict[int, int], *, training: bool) -> tuple[int, ...] | None:
        """Choose only eligible unclaimed cards; acquisition still rechecks claims."""
        if not set(GPU_POOL).issubset(memory) or any(value < 0 for value in memory.values()):
            raise ValueError("incomplete or invalid GPU memory inventory")
        free = set()
        with self._lock():
            for gpu in GPU_POOL:
                claim = self._read(gpu)
                if memory[gpu] < 1024 and (claim is None or not self._active(claim)):
                    free.add(gpu)
        return select_devices(memory, training=training, busy=set(GPU_POOL) - free)

    def acquire(self, gpus: tuple[int, ...], *, job: str) -> bool:
        """Reserve every requested GPU exclusively, or roll back this attempt."""
        import tempfile  # noqa: PLC0415 -- complete exclusive claim publication

        if not gpus or len(set(gpus)) != len(gpus) or not set(gpus).issubset(GPU_POOL) or self.owned:
            raise ValueError("claim request must name unique allowed GPUs on a fresh attempt")
        record = {"token": self.token, "owner_pid": os.getpid(), "child_pid": None, "job": job,
                  "attempt_started_ticks": self.started_ticks}
        with self._lock():
            try:
                for gpu in gpus:
                    path = self.directory / str(gpu)
                    claim = self._read(gpu)
                    if claim is not None and self._active(claim):
                        self._release_owned()
                        return False
                    if claim is not None:
                        path.unlink(missing_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, prefix=".claim-") as handle:
                        json.dump(record, handle)
                        handle.flush()
                        self.owned.add(gpu)
                        try:
                            os.link(handle.name, path)
                        except FileExistsError:
                            self._release_owned()
                            return False
            except BaseException:
                self._release_owned()
                raise
        return True

    def refresh(self, *, child_pid: int | None = None) -> None:
        """Refresh only token-matched claims; loss of ownership fails visibly."""
        with self._lock():
            for gpu in self.owned:
                claim = self._read(gpu)
                if claim is None or claim[0] is None or claim[0].get("token") != self.token:
                    raise ValueError("GPU reservation ownership changed")
                record = {**claim[0], "child_pid": claim[0].get("child_pid") if child_pid is None else child_pid}
                if child_pid is not None:
                    record["child_session"] = child_pid
                path = self.directory / str(gpu)
                # Update through the open inode; never replace a foreign path.
                with path.open("r+") as handle:
                    current = json.load(handle)
                    if current.get("token") != self.token:
                        raise ValueError("GPU reservation ownership changed")
                    handle.seek(0)
                    json.dump(record, handle)
                    handle.truncate()
                    os.utime(handle.fileno(), None)

    def _release_owned(self) -> None:
        for gpu in self.owned:
            claim = self._read(gpu)
            if claim is not None and claim[0] is not None and claim[0].get("token") == self.token:
                (self.directory / str(gpu)).unlink(missing_ok=True)
        self.owned.clear()

    def release(self) -> None:
        """Release this attempt; preserve reservations replaced by another owner."""
        with self._lock():
            for gpu in self.owned:
                claim = self._read(gpu)
                if (
                    claim is not None
                    and claim[0] is not None
                    and claim[0].get("token") == self.token
                    and (process_alive(claim[0].get("child_pid")) or (
                        claim[0].get("child_session") is not None
                        and live_session_processes(claim[0]["child_session"])
                    ) or (claim[0].get("token") is not None and live_attempt_processes(
                        claim[0]["token"], started_ticks=claim[0].get("attempt_started_ticks")
                    )))
                ):
                    raise ValueError("cannot release a GPU used by a live child")
            self._release_owned()
