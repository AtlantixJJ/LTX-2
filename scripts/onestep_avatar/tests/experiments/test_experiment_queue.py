"""Literal experiment dispatch binds parsers, spec bytes and saved completion."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.experiments import sigma_sweep
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.experiments.test_sigma_sweep import sweep_queue_job

# Reuse the real saved-media fixture without shadowing an imported fixture name.
pytest_plugins = ("scripts.onestep_avatar.tests.experiments.test_sigma_sweep",)


def test_literal_experiment_table_has_only_reviewed_current_owner() -> None:
    assert queue.EXPERIMENTS == {"sigma_sweep": "scripts.onestep_avatar.experiments.sigma_sweep"}


@pytest.mark.parametrize("prepared_input", [False, True])
def test_experiment_raw_and_prepared_round_trips_bind_spec_and_descriptor(
    saved_sweep: tuple, prepared_input: bool
) -> None:
    _, path, output, _ = saved_sweep
    initial = sweep_queue_job(path, output)
    raw = deepcopy(initial)
    if not prepared_input:
        raw.pop("sha256")
    assert queue.prepare_job(raw, path.parent) == initial
    assert initial["spec_sha256"] == sha256(path)
    command, _ = queue.job_command(initial, (0,))
    assert command[2] == queue.EXPERIMENTS["sigma_sweep"]
    changed = deepcopy(initial)
    changed["completion"]["manifest"] = str(output / "different.json")
    with pytest.raises(ValueError, match="completion manifest differs"):
        queue.prepare_job(changed, path.parent)


@pytest.mark.parametrize("selector", ["unknown", "", "scripts.onestep_avatar.experiments.sigma_sweep", None, 1])
def test_unknown_selector_refuses_before_input_reads_imports_claims_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: object
) -> None:
    spec = tmp_path / "missing.json"
    output = tmp_path / "output"
    raw = {
        "id": "unknown",
        "kind": "experiment",
        "experiment": selector,
        "spec": str(spec),
        "spec_sha256": "a" * 64,
        "arguments": ["--spec", str(spec), "--output", str(output)],
        "output": str(output),
        "completion": {"manifest": str(output / "manifest.json")},
    }
    monkeypatch.setattr(queue.importlib, "import_module", lambda *_args: pytest.fail("imported unknown selector"))
    before = list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="selector is unknown"):
        queue.prepare_job(raw, tmp_path)
    assert list(tmp_path.iterdir()) == before


@pytest.mark.parametrize("field", ["experiment", "spec", "spec_sha256"])
def test_required_experiment_identity_fields_are_not_inferred(saved_sweep: tuple, field: str) -> None:
    _, path, output, _ = saved_sweep
    job = sweep_queue_job(path, output)
    job.pop(field)
    with pytest.raises(ValueError, match=r"selector|specification"):
        queue.prepare_job(job, path.parent)


def test_spec_claim_and_prepared_digest_refuse_mutation(saved_sweep: tuple) -> None:
    _, path, output, _ = saved_sweep
    job = sweep_queue_job(path, output)
    bad_claim = {**job, "spec_sha256": "a" * 64}
    with pytest.raises(ValueError, match="specification changed"):
        queue.prepare_job(bad_claim, path.parent)
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    updated = {**job, "spec_sha256": sha256(path)}
    with pytest.raises(ValueError, match="prepared job identity"):
        queue.prepare_job(updated, path.parent)
    for verify in (lambda: queue.job_command(job, (0,)), lambda: queue.verify_completion(job)):
        with pytest.raises(ValueError, match="specification changed"):
            verify()
    path.write_bytes(original)
    assert queue.prepare_job(job, path.parent) == job


def test_retired_sweep_kind_remains_data_and_cannot_be_dispatched(saved_sweep: tuple) -> None:
    _, path, output, _ = saved_sweep
    job = sweep_queue_job(path, output)
    old = {**job, "kind": "sigma_sweep"}
    before = sorted(p.name for p in path.parent.iterdir())
    with pytest.raises(ValueError, match="supported package owner"):
        queue.prepare_job(old, path.parent)
    assert sorted(p.name for p in path.parent.iterdir()) == before


def test_ordinary_job_never_looks_up_or_imports_experiments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue, "EXPERIMENTS", None)
    monkeypatch.setattr(
        queue.importlib, "import_module", lambda *_args: pytest.fail("ordinary job imported experiment")
    )
    jobs_file = tmp_path / "saved_jobs.json"
    jobs_file.write_text('{"schema_version": 1, "jobs": []}')
    output = tmp_path / "output"
    raw = {
        "id": "ordinary",
        "kind": "decode",
        "arguments": ["--jobs", str(jobs_file), "--output", str(output)],
        "output": str(output),
        "completion": {"manifest": str(output / "manifest.json")},
    }
    prepared = queue.prepare_job(raw, tmp_path)
    command, _ = queue.job_command(prepared, (0,))
    assert command[2] == "scripts.onestep_avatar.decode_saved"
    assert not queue.verify_completion(prepared)
    assert not output.exists()


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        [],
        {},
        {"manifest": 1},
        {"manifest": ""},
        {"records": []},
        {"records": [1]},
        {"records": [""]},
        {"unknown": "data"},
    ],
)
def test_malformed_public_parser_completion_descriptor_refused_before_writes(
    saved_sweep: tuple, monkeypatch: pytest.MonkeyPatch, descriptor: object
) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    parser = sigma_sweep.parse_args

    def malformed(argv: list[str]) -> object:
        args = parser(argv)
        args.completion = descriptor
        return args

    monkeypatch.setattr(sigma_sweep, "parse_args", malformed)
    with pytest.raises(ValueError, match="descriptor is malformed"):
        queue.prepare_job(job, path.parent)
    assert not output.exists()
    assert not calls


def test_matching_parser_and_job_descriptor_still_cannot_escape_output(
    saved_sweep: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    escaped = {"manifest": str(output.parent / "escape.json")}
    job["completion"] = escaped
    parser = sigma_sweep.parse_args

    def changed(argv: list[str]) -> object:
        args = parser(argv)
        args.completion = escaped
        return args

    monkeypatch.setattr(sigma_sweep, "parse_args", changed)
    with pytest.raises(ValueError, match="escapes its output"):
        queue.prepare_job(job, path.parent)
    assert not output.exists()
    assert not calls


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        [],
        {},
        {"manifest": 1},
        {"manifest": ""},
        {"records": []},
        {"records": [1]},
        {"records": [""]},
        {"unknown": "data"},
    ],
)
def test_malformed_job_descriptor_refused_before_owner_import_or_writes(
    saved_sweep: tuple, monkeypatch: pytest.MonkeyPatch, descriptor: object
) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    job["completion"] = descriptor
    monkeypatch.setattr(queue.importlib, "import_module", lambda *_args: pytest.fail("imported malformed job"))
    with pytest.raises(ValueError, match="descriptor is malformed"):
        queue.prepare_job(job, path.parent)
    assert not output.exists()
    assert not calls


def test_relative_spec_and_output_paths_normalize_before_prepared_round_trip(saved_sweep: tuple) -> None:
    _, path, output, _ = saved_sweep
    absolute = sweep_queue_job(path, output)
    relative = deepcopy(absolute)
    relative.pop("sha256")
    relative["spec"] = path.name
    relative["output"] = output.name
    relative["arguments"] = ["--spec", path.name, "--output", output.name]
    relative["completion"] = {"manifest": str(Path(output.name) / "manifest.json")}
    normalized = queue.prepare_job(relative, path.parent)
    assert normalized == absolute
    assert queue.prepare_job(normalized, path.parent) == normalized


def test_declared_spec_cannot_differ_from_command_argument(saved_sweep: tuple) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    job["spec"] = str(path.parent / "another.json")
    with pytest.raises(ValueError, match="specification differs from command"):
        queue.prepare_job(job, path.parent)
    assert not output.exists()
    assert not calls
