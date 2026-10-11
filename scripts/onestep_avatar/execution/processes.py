"""Processes owner for managed execution; see doc/execution/processes.md."""

from __future__ import annotations

import os
import time
from pathlib import Path

from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_PROTOCOL


def _read_process_environment(entry: Path, start_ticks: str) -> bytes | None:
    """Resolve brief exit-time denial only by readable or confirmed terminal handles."""
    for attempt in range(10):
        try:
            return (entry / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            return None
        except PermissionError:
            try:
                raw = (entry / "stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                return None
            fields = raw[raw.rfind(")") + 1 :].split()
            if len(fields) < 20 or fields[19] != start_ticks:
                raise
            if fields[0] in ("Z", "X", "x"):
                return None
            if attempt == 9:
                raise
            time.sleep(0.01)
    raise AssertionError("bounded environment observation exhausted without a decision")


def live_attempt_processes(  # noqa: PLR0912 -- stable handle and token observation gates
    token: str, *, started_ticks: int | None = None, proc_root: Path = Path("/proc")
) -> list[int]:
    """Find token-bound workers even when Torch Elastic gives ranks new sessions."""
    from scripts.onestep_avatar.execution.queue_protocol import TOKEN_ENV  # noqa: PLC0415

    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("worker inspection requires an exact attempt token")
    if started_ticks is not None and (type(started_ticks) is not int or started_ticks < 0):
        raise ValueError("worker inspection requires valid attempt start ticks")
    marker = os.fsencode(TOKEN_ENV + "=" + token)
    live = set()
    for _ in range(2):
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.geteuid():
                    continue
                raw = (entry / "stat").read_text()
                fields = raw[raw.rfind(")") + 1 :].split()
                if len(fields) < 20:
                    raise ValueError("malformed process stat during worker inspection")
                if started_ticks is not None and int(fields[19]) < started_ticks:
                    continue  # Older processes cannot inherit this newly created token.
                if fields[0] in ("Z", "X", "x"):
                    continue
                raw_environment = _read_process_environment(entry, fields[19])
                if raw_environment is None:
                    continue
                environment = raw_environment.split(b"\0")
                if marker not in environment:
                    continue
                after = (entry / "stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            final = after[after.rfind(")") + 1 :].split()
            if len(final) < 20 or final[19] != fields[19]:
                raise ValueError("worker handle changed during inspection")
            if final[0] not in ("Z", "X", "x"):
                live.add(int(entry.name))
        if live:
            break
    return sorted(live)


def owned_workers_live(row: dict) -> bool:
    """A session leader is insufficient to prove distributed child termination."""
    from scripts.onestep_avatar.execution.queue_protocol import (  # noqa: PLC0415 -- canonical environment field
        TOKEN_ENV,
    )

    token = row.get("environment_changes", {}).get(TOKEN_ENV)
    if row.get("process_ledger") is not None:
        from scripts.onestep_avatar.execution.process_registry import (  # noqa: PLC0415 -- targeted own tree
            inspect_record,
        )

        observed = inspect_record(Path(row["process_ledger"]), token)
        return not observed["complete"] or observed["workers_live"]
    return bool(
        (row.get("child_session") is not None and live_session_processes(row["child_session"]))
        or (token is not None and live_attempt_processes(token, started_ticks=row.get("attempt_started_ticks")))
    )


def live_session_processes(session_id: int, *, proc_root: Path = Path("/proc")) -> list[int]:
    """Find live members of an owned Linux session; unreadable handles fail closed."""
    if type(session_id) is not int or session_id <= 0:
        raise ValueError("child session requires a positive ID")
    live = set()
    # Reobserve an empty session rather than infer termination from its leader.
    for _ in range(2):
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            fields = raw[raw.rfind(")") + 1 :].split()
            if len(fields) < 20:
                raise ValueError("malformed process stat during child-session inspection")
            if int(fields[3]) == session_id and fields[0] not in ("Z", "X", "x"):
                live.add(int(entry.name))
        if live:
            break
    return sorted(live)


def process_identity(pid: int, *, proc_root: Path = Path("/proc")) -> dict | None:
    """Read a stable Linux process handle; I/O denial never establishes death."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("process identity requires a positive PID")
    root = proc_root / str(pid)

    def read_stat() -> tuple[str, int]:
        raw = (root / "stat").read_text()
        closing = raw.rfind(")")
        fields = raw[closing + 1 :].split()
        if closing < 0 or len(fields) < 20:
            raise ValueError("malformed process stat record")
        return fields[0], int(fields[19])

    try:
        before = read_stat()
        command = (root / "cmdline").read_bytes()
        after = read_stat()
    except FileNotFoundError:
        return None
    if before[1] != after[1]:
        raise ValueError("process identity changed during inspection")
    return {
        "pid": pid,
        "start_ticks": after[1],
        "command": [os.fsdecode(value) for value in command.rstrip(b"\0").split(b"\0")] if command else [],
        "terminal": after[0] in ("Z", "X", "x"),
    }


def process_alive(pid: int | None) -> bool:
    """Probe the recorded process; permission denial means it remains live."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        identity = process_identity(pid)
    except (OSError, ValueError):
        return True  # An unreadable handle is not proof of death.
    return not (identity is None or identity["terminal"])


def inspect_child(row: dict) -> str:
    """Classify a saved child without adopting a reused or unidentified live PID."""
    pid = row.get("child_pid")
    if pid is None:
        from scripts.onestep_avatar.execution.queue_launch import inspect_unapproved_launch  # noqa: PLC0415

        inspect_unapproved_launch(row)
        return "unapproved"
    saved = row.get("child_identity")
    if saved is None and row.get("launch_protocol") == LAUNCH_PROTOCOL:
        from scripts.onestep_avatar.execution.queue_launch import inspect_unapproved_launch  # noqa: PLC0415

        inspect_unapproved_launch(row)
        return "unapproved"
    current = process_identity(pid)
    if current is None:
        if owned_workers_live(row):
            return "live"
        return "missing"
    if not isinstance(saved, dict) or any(saved.get(key) != current[key] for key in ("pid", "start_ticks")):
        raise ValueError("queue child identity differs; recovery is required")
    if (
        not isinstance(saved.get("command"), list)
        or not saved["command"]
        or any(not isinstance(value, str) for value in saved["command"])
    ):
        raise ValueError("queue child identity differs; recovery is required")
    # Linux clears cmdline after exit while an unreaped zombie retains its PID/ticks.
    if current["command"] != saved.get("command") and not (current["terminal"] and current["command"] == []):
        from scripts.onestep_avatar.execution.queue_launch import verify_command_transition  # noqa: PLC0415

        verify_command_transition(row, current)
    if owned_workers_live(row):
        return "live"
    return "terminal" if current["terminal"] else "live"
