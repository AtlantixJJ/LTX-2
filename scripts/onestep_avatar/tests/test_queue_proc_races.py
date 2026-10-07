"""Process disappearance is confirmed from handles; permission denial alone never proves death."""

from pathlib import Path

import pytest

from scripts.onestep_avatar import queue
from scripts.onestep_avatar.queue_protocol import TOKEN_ENV


def stat_text(state: str = "S", tick: int = 200) -> str:
    """Make Linux stat fields including the inspected state, session and start tick."""
    return "42 (worker) " + " ".join([state, "0", "0", "999", *(["0"] * 15), str(tick)])


@pytest.mark.parametrize("scan", ["session", "token"])
def test_disappearing_stat_with_esrch_is_reobserved_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scan: str
) -> None:
    entry = tmp_path / "42"
    entry.mkdir()
    original = Path.read_text

    def gone(path: Path, *args, **kwargs) -> str:
        if path == entry / "stat":
            raise ProcessLookupError("process disappeared during stat read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", gone)
    if scan == "session":
        assert queue.live_session_processes(999, proc_root=tmp_path) == []
    else:
        assert queue.live_attempt_processes("b" * 32, started_ticks=100, proc_root=tmp_path) == []


@pytest.mark.parametrize("transition", ["terminal", "missing", "reused", "live"])
def test_environment_denial_requires_same_terminal_or_absent_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    entry = tmp_path / "42"
    entry.mkdir()
    stat = entry / "stat"
    stat.write_text(stat_text())
    (entry / "environ").write_bytes((TOKEN_ENV + "=" + "b" * 32).encode())
    original = Path.read_bytes

    def denied(path: Path) -> bytes:
        if path != entry / "environ":
            return original(path)
        if transition == "terminal":
            stat.write_text(stat_text("Z"))
        elif transition == "missing":
            stat.unlink()
        elif transition == "reused":
            stat.write_text(stat_text("Z", 201))
        raise PermissionError("environment unavailable")

    monkeypatch.setattr(Path, "read_bytes", denied)
    if transition in ("terminal", "missing"):
        assert queue.live_attempt_processes("b" * 32, started_ticks=100, proc_root=tmp_path) == []
    else:
        with pytest.raises(PermissionError):
            queue.live_attempt_processes("b" * 32, started_ticks=100, proc_root=tmp_path)


def test_brief_live_denial_retries_then_identifies_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "42"
    entry.mkdir()
    (entry / "stat").write_text(stat_text())
    (entry / "environ").write_bytes((TOKEN_ENV + "=" + "b" * 32).encode())
    original = Path.read_bytes
    attempts = []

    def briefly_denied(path: Path) -> bytes:
        if path == entry / "environ":
            attempts.append(path)
            if len(attempts) < 3:
                raise PermissionError("brief exit/exec transition")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", briefly_denied)
    assert queue.live_attempt_processes("b" * 32, started_ticks=100, proc_root=tmp_path) == [42]
    assert len(attempts) == 3
