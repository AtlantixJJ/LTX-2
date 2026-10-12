"""Adapt the real ledger to the historical, ledger-free launch interface.

Guarded-recovery tests exercise saved rows from before process-ledger binding.
Expose only that old interface; hiding `path` prevents the launcher from adding
current ledger evidence to those historical rows. Acquisition, refresh and
release still use ProcessRegistry, with no file-claim or TTL implementation.
"""

from pathlib import Path

from scripts.onestep_avatar.execution.process_registry import ProcessRegistry


class LegacyClaims:
    def __init__(self, path: Path) -> None:
        self._registry = ProcessRegistry(path, inventory=lambda: dict.fromkeys(range(8), 0))

    @property
    def token(self) -> str:
        return self._registry.token

    @property
    def started_ticks(self) -> int:
        return self._registry.started_ticks

    @property
    def owned(self) -> set[int]:
        return self._registry.owned

    def acquire(self, gpus: tuple[int, ...], *, job: str) -> bool:
        return self._registry.acquire(gpus, job=job)

    def refresh(self, *, child_pid: int | None = None) -> None:
        return self._registry.refresh(child_pid=child_pid)

    def release(self) -> None:
        return self._registry.release()
