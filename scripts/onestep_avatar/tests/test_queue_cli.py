"""Select the requested journal without writing or claiming devices during review."""

import json
from pathlib import Path

import pytest

from scripts.onestep_avatar.execution import queue


@pytest.mark.parametrize("mode", ["--execute", "--foreground", "--recover"])
@pytest.mark.parametrize("explicit_state", [False, True])
def test_cli_modes_use_jobs_directory_state_unless_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    explicit_state: bool,
) -> None:
    jobs = tmp_path / "run" / "jobs.json"
    selected_state = tmp_path / "custom.json" if explicit_state else jobs.parent / "state.json"
    observed = []

    def execute(jobs_path: Path, state_path: Path, _ledger: Path, **_options: object) -> int:
        observed.append((jobs_path, state_path))
        return 0

    def recover(jobs_path: Path, state_path: Path) -> dict:
        observed.append((jobs_path, state_path))
        return {"recovered": False, "jobs": []}

    monkeypatch.setattr(queue, "execute_jobs", execute)
    monkeypatch.setattr(queue, "recover_jobs", recover)
    arguments = ["--jobs", str(jobs), mode]
    if explicit_state:
        arguments += ["--state", str(selected_state)]
    if mode != "--recover":
        arguments += ["--process-ledger", str(tmp_path / "ledger.json")]
    if mode == "--execute":
        arguments += ["--once"]
    assert queue.main(arguments) == 0
    assert observed == [(jobs, selected_state)]
    assert not list(tmp_path.iterdir())
    capsys.readouterr()


def test_dry_run_reads_default_journal_beside_relative_jobs_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    jobs = directory / "jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": []}))
    state = directory / "state.json"
    state.write_text(json.dumps({"schema_version": 1, "job_order": [], "jobs": {}, "owner_pid": 123}))
    original = {path: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.chdir(tmp_path)
    assert queue.main(["--jobs", "run/jobs.json", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"] == []
    assert {path: path.read_bytes() for path in directory.iterdir()} == original
    assert not (tmp_path / "state.json").exists()


def test_explicit_state_overrides_a_malformed_default_journal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": []}))
    default_state = tmp_path / "state.json"
    default_state.write_text("not a journal")
    explicit = tmp_path / "other.json"
    assert queue.main(["--jobs", str(jobs), "--state", str(explicit), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"] == []
    assert not explicit.exists()
    assert default_state.read_text() == "not a journal"


def test_dry_run_with_no_journal_creates_no_default_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": []}))
    assert queue.main(["--jobs", str(jobs), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"] == []
    assert list(tmp_path.iterdir()) == [jobs]


def test_no_startup_retries_reaches_execution(tmp_path, monkeypatch):
    observed = {}

    def execute(*args, **options):
        observed.update(options)
        return 0

    monkeypatch.setattr(queue, "execute_jobs", execute)
    assert (
        queue.main(
            [
                "--jobs",
                str(tmp_path / "jobs.json"),
                "--execute",
                "--loop",
                "--process-ledger",
                str(tmp_path / "ledger.json"),
                "--no-startup-retries",
            ]
        )
        == 0
    )
    assert observed["startup_retries"] is False


def test_disabled_startup_retry_propagates_oom(tmp_path, monkeypatch):
    from scripts.onestep_avatar.execution import attempt, clock, completion, process_registry

    job = {"id": "train", "kind": "train", "dependencies": []}
    monkeypatch.setattr(completion, "ready_jobs", lambda *args: [job])
    monkeypatch.setattr(process_registry, "gpu_memory", lambda **kwargs: dict.fromkeys(range(4), 0))

    class Claims:
        token = "test"

        def __init__(self, *args):
            pass

        def choose(self, *args, **kwargs):
            return (0, 1, 2, 3)

        def acquire(self, *args, **kwargs):
            return True

    monkeypatch.setattr(process_registry, "ProcessRegistry", Claims)

    def fail(*args, **kwargs):
        raise clock.QueueChildError("CUDA out of memory")

    monkeypatch.setattr(attempt, "run_child", fail)

    def forbidden(*args):
        pytest.fail("must not retry")

    monkeypatch.setattr(attempt, "retry_startup_failure", forbidden)
    with pytest.raises(clock.QueueChildError, match="out of memory"):
        queue.dispatch_ready(
            [job], {"jobs": {"train": {}}}, tmp_path / "state.json", tmp_path / "ledger.json", startup_retries=False
        )
