"""Own-process dispatch, claims and child lifecycle controls for the package queue."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import torch
from PIL import Image

from ltx_trainer.video_utils import save_video
from scripts.onestep_avatar import decode_saved, evaluate, media
from scripts.onestep_avatar.execution import queue, supervision
from scripts.onestep_avatar.execution.process_registry import ProcessRegistry
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.claims_fixture import GPUClaims
from scripts.onestep_avatar.tests.test_previews import _completed
from scripts.prune.core import model_registry


def test_inventory_requires_all_allowed_devices() -> None:
    assert queue.parse_gpu_memory("\n".join(f"{gpu}, 0" for gpu in range(8))) == dict.fromkeys(range(8), 0)
    for text in ("", "0, 0", "0, unavailable", "\n".join(f"{gpu}, 0" for gpu in [0, 1, 2, 3, 4, 5, 5])):
        with pytest.raises(ValueError, match="GPU"):
            queue.parse_gpu_memory(text)


def test_job_list_requires_explicit_mode_and_matching_outputs() -> None:
    job = {
        "id": "first",
        "kind": "evaluate",
        "arguments": ["--mode", "causal", "--output", "/result"],
        "output": "/result",
        "dependencies": [],
        "completion": {"records": ["/result/result.json"]},
    }
    assert queue.validate_job_list({"schema_version": 1, "jobs": [job]}) == [job]
    for replacement in (
        {"arguments": ["--output", "/result"]},
        {"output": "/other"},
        {"dependencies": ["missing"]},
        {"completion": {}},
        {"kind": "shell"},
    ):
        with pytest.raises(ValueError, match="queue"):
            queue.validate_job_list({"schema_version": 1, "jobs": [{**job, **replacement}]})
    with pytest.raises(ValueError, match="unique"):
        queue.validate_job_list({"schema_version": 1, "jobs": [job, job]})


def test_prepared_jobs_reuse_owner_parser_and_hash_resolved_settings(tmp_path: Path) -> None:
    job = {
        "id": "first",
        "kind": "evaluate",
        "arguments": [
            "--mode",
            "causal",
            "--subset",
            "membership.json",
            "--output",
            "result",
            "--schedule",
            "0.725",
            "0",
        ],
        "output": "result",
        "dependencies": [],
        "completion": {"records": ["result/result.json"]},
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    first = queue.prepare_jobs(path)[0]
    assert first["output"] == str(tmp_path / "result")
    assert str(tmp_path / "membership.json") in first["arguments"]
    assert queue.prepare_jobs(path)[0]["sha256"] == first["sha256"]
    assert not (tmp_path / "result").exists()
    job["arguments"] += ["--seed", "99"]
    path.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    assert queue.prepare_jobs(path)[0]["sha256"] != first["sha256"]
    job["arguments"] += ["--gpu-id", "6"]
    path.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    with pytest.raises(ValueError, match="forbidden execution override"):
        queue.prepare_jobs(path)


def test_training_launch_identity_pins_config_bytes(tmp_path: Path) -> None:
    config = tmp_path / "fsdp.yaml"
    config.write_text("num_processes: 4\n")
    job = {
        "id": "train",
        "kind": "train",
        "output": "result",
        "processes": 4,
        "accelerate_config": "fsdp.yaml",
        "port": 29503,
        "arguments": ["--mode", "bidirectional", "--subset", "membership.json", "--output", "result", "--steps", "2"],
        "completion": {"checkpoint": "result/checkpoints/final.safetensors", "step": 2},
    }
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    first = queue.prepare_jobs(path)[0]
    assert first["accelerate_config_sha256"] == sha256(config)
    queue.job_command(first, (0, 1, 2, 3))
    config.write_text("num_processes: 4\nfsdp_forward_prefetch: true\n")
    assert queue.prepare_jobs(path)[0]["sha256"] != first["sha256"]
    with pytest.raises(ValueError, match="configuration changed"):
        queue.job_command(first, (0, 1, 2, 3))
    config.unlink()
    with pytest.raises(ValueError, match="configuration is missing"):
        queue.prepare_jobs(path)
    assert not (tmp_path / "result").exists()


def test_decoder_job_identity_pins_list_contents(tmp_path: Path) -> None:
    saved = tmp_path / "decode.json"
    saved.write_text(json.dumps({"jobs": [{"id": "first", "latent": "source.pt"}]}))
    job = {
        "id": "decode",
        "kind": "decode",
        "output": "result",
        "arguments": ["--jobs", "decode.json", "--output", "result"],
        "completion": {"manifest": "result/manifest.json"},
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    first = queue.prepare_jobs(path)[0]
    queue.job_command(first, (5,))
    saved.write_text(json.dumps({"jobs": [{"id": "changed", "latent": "other.pt"}]}))
    assert queue.prepare_jobs(path)[0]["sha256"] != first["sha256"]
    with pytest.raises(ValueError, match="decoder job list changed"):
        queue.job_command(first, (5,))
    saved.unlink()
    with pytest.raises(ValueError, match="decoder job list is missing"):
        queue.prepare_jobs(path)
    assert not (tmp_path / "result").exists()


def test_commands_mask_physical_devices_and_use_fixed_package_owners(tmp_path: Path) -> None:

    job = {"kind": "evaluate", "arguments": ["--mode", "causal"]}
    command, environment = queue.job_command(job, (5,))
    assert command == [sys.executable, "-m", "scripts.onestep_avatar.evaluate", "--mode", "causal", "--gpu-id", "0"]
    assert environment == {"CUDA_VISIBLE_DEVICES": "5"}
    config = tmp_path / "fsdp.yaml"
    config.write_text("num_processes: 4\n")
    training = {
        "kind": "train",
        "arguments": [
            "--mode",
            "bidirectional",
            "--subset",
            str(tmp_path / "membership.json"),
            "--output",
            str(tmp_path / "run"),
        ],
        "processes": 4,
        "port": 29500,
        "accelerate_config": str(config),
        "accelerate_config_sha256": sha256(config),
    }
    command, environment = queue.job_command(training, (0, 1, 2, 3))
    assert command[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    assert "scripts.onestep_avatar.train" in command
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    for gpus in ((8,), (4, 5), (5, 5)):
        with pytest.raises(ValueError, match=r"GPU|four"):
            queue.job_command(job, gpus)


@pytest.mark.usefixtures("controlled_evaluation_conditions")
def test_completion_verifies_raw_bytes_and_rejects_unrelated_outputs(tmp_path: Path) -> None:

    output = tmp_path / "result"
    job = {
        "kind": "evaluate",
        "output": str(output),
        "arguments": ["--mode", "causal"],
        "completion": {"records": [str(output / "result.json")]},
    }
    assert not queue.verify_completion(job)
    evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {"mode": "causal"}, output)
    assert queue.verify_completion(job)
    (output / "generated.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        queue.verify_completion(job)
    job["completion"]["records"] = [str(tmp_path / "outside.json")]
    with pytest.raises(ValueError, match="escapes"):
        queue.verify_completion(job)


def test_training_is_pending_without_completed_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.safetensors"
    checkpoint.write_bytes(b"incomplete")
    job = {"kind": "train", "output": str(tmp_path), "completion": {"checkpoint": str(checkpoint), "step": 7}}
    assert not queue.verify_completion(job)


@pytest.mark.usefixtures("controlled_training_conditions")
def test_checkpoint_completion_checks_real_contract_and_marker(tmp_path: Path) -> None:
    checkpoint = _completed(tmp_path)
    job = {
        "kind": "train",
        "output": str(tmp_path),
        "arguments": ["--mode", "bidirectional"],
        "completion": {"checkpoint": str(checkpoint), "step": 100},
    }
    assert queue.verify_completion(job)
    job["arguments"] = ["--mode", "causal"]
    with pytest.raises(ValueError, match="wrong mode"):
        queue.verify_completion(job)
    job["arguments"] = ["--mode", "bidirectional"]
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="marker is invalid"):
        queue.verify_completion(job)


def test_state_preserves_job_identities_and_aborted_transaction(tmp_path: Path) -> Never:
    path = tmp_path / "state.json"
    jobs = [{"id": "first", "sha256": "a" * 64}]
    with queue.queue_state(path, jobs) as state:
        state["jobs"]["first"]["state"] = "complete"
    original = path.read_bytes()
    with pytest.raises(RuntimeError), queue.queue_state(path, jobs) as state:  # noqa: PT012 -- assert rollback and lifecycle reporting across the whole transaction
        state["jobs"]["first"]["state"] = "failed"
        raise RuntimeError("interrupted transaction")
    assert path.read_bytes() == original
    appended = [*jobs, {"id": "second", "sha256": "b" * 64}]
    assert queue.read_queue_state(path, appended)["jobs"]["second"]["state"] == "pending"
    assert path.read_bytes() == original
    for changed in ([{"id": "first", "sha256": "c" * 64}], [], list(reversed(appended))):
        with pytest.raises(ValueError, match="queue"):
            queue.read_queue_state(path, changed)


def test_state_protects_surviving_child_without_live_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    jobs = [{"id": "first", "sha256": "a" * 64}]
    state = {
        "schema_version": 1,
        "job_order": ["first"],
        "owner_pid": 99,
        "jobs": {"first": {"sha256": "a" * 64, "state": "running", "child_pid": 98, "attempts": []}},
    }
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    monkeypatch.setattr(queue, "process_alive", lambda pid: pid == 98)
    with pytest.raises(ValueError, match="surviving child"), queue.queue_state(path, jobs):
        pytest.fail("surviving child permitted another owner")
    assert path.read_bytes() == original


def test_decoder_completion_checks_inventory_and_every_output_hash(  # noqa: PLR0915 -- one complete fixture lifecycle with all assertions and cleanup
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    source = tmp_path / "encoded.pt"
    latent = torch.zeros(1, 2, 2, 2, 2)
    torch.save(latent, source)
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"VAE identity fixture")
    model = SimpleNamespace(
        paths=SimpleNamespace(video_vae=lambda: vae),
        caps=SimpleNamespace(latent_channels=2),
        scale_factors=SimpleNamespace(time=8, height=32, width=32),
    )
    monkeypatch.setattr(model_registry, "resolve", lambda *_args: model)
    requested = {"id": "case", "latent": str(source), "sha256": sha256(source)}
    jobs_path = tmp_path / "decode_jobs.json"
    jobs_path.write_text(json.dumps({"jobs": [requested]}))
    video = tmp_path / "case.mp4"
    save_video(torch.zeros(9, 3, 64, 64), video, fps=30, video_format="FCHW")
    sample = tmp_path / "case_f000.png"
    Image.new("RGB", (64, 64)).save(sample)
    row = {
        "id": "case",
        "input": requested,
        "frames": 9,
        "video": video.name,
        "video_sha256": sha256(video),
        "samples": [{"frame": 0, "file": sample.name, "sha256": sha256(sample)}],
    }
    args = decode_saved.parse_args(["--jobs", str(jobs_path), "--output", str(tmp_path)])
    settings = media.native_decoder_settings()
    row["decoder"] = {
        "model": args.model,
        "vae_path": str(vae),
        "vae_sha256": sha256(vae),
        "seed": args.seed,
        "settings": settings,
    }
    row["source_code_sha256"] = sha256(Path(decode_saved.__file__))
    row["software"] = decode_saved.software.capture("decoding")
    row["decode_key"] = media.decode_key(
        sha256(source), sha256(vae), list(latent.shape), "native_decode_video", args.seed, settings
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [row], "comparisons": []}))
    job = {
        "kind": "decode",
        "output": str(tmp_path),
        "arguments": ["--jobs", str(jobs_path), "--output", str(tmp_path)],
        "completion": {"manifest": str(manifest)},
    }
    assert queue.verify_completion(job)
    with pytest.raises(ValueError, match="frame count or playback rate differs"):
        media.verify_saved_video(video, 8, 30)
    with pytest.raises(ValueError, match="frame count or playback rate differs"):
        media.verify_saved_video(video, 9, 24)
    Image.new("RGB", (32, 64)).save(sample)
    row["samples"][0]["sha256"] = sha256(sample)
    manifest.write_text(json.dumps({"jobs": [row], "comparisons": []}))
    with pytest.raises(ValueError, match="dimensions differ from video"):
        queue.verify_completion(job)
    Image.new("RGB", (64, 64)).save(sample)
    row["samples"][0]["sha256"] = sha256(sample)
    manifest.write_text(json.dumps({"jobs": [row], "comparisons": []}))
    model.scale_factors.width = 64
    with pytest.raises(ValueError, match="dimensions differ from selected model"):
        queue.verify_completion(job)
    model.scale_factors.width = 32
    model.caps.latent_channels = 4
    with pytest.raises(ValueError, match="invalid shape/content"):
        queue.verify_completion(job)
    model.caps.latent_channels = 2
    row["decoder"]["seed"] += 1
    manifest.write_text(json.dumps({"jobs": [row], "comparisons": []}))
    with pytest.raises(ValueError, match="decoder identity differs"):
        queue.verify_completion(job)
    row["decoder"]["seed"] -= 1
    row["frames"] += 1
    manifest.write_text(json.dumps({"jobs": [row], "comparisons": []}))
    with pytest.raises(ValueError, match="coverage differs"):
        queue.verify_completion(job)
    row["frames"] -= 1
    manifest.write_text(json.dumps({"jobs": [row], "comparisons": []}))
    sample.write_bytes(b"changed sample")
    with pytest.raises(ValueError, match="artifact content changed"):
        queue.verify_completion(job)
    sample.unlink()
    assert not queue.verify_completion(job)
    manifest.write_text(json.dumps({"jobs": [], "comparisons": []}))
    with pytest.raises(ValueError, match="inventory differs"):
        queue.verify_completion(job)


@pytest.mark.usefixtures("controlled_evaluation_conditions")
def test_receipt_rejects_rewritten_result_even_when_new_output_hash_matches(tmp_path: Path) -> None:

    evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {"mode": "causal"}, tmp_path)
    job = {
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(tmp_path),
        "arguments": ["--mode", "causal"],
        "completion": {"records": [str(tmp_path / "result.json")]},
    }
    receipt = queue.completion_receipt(job)
    assert queue.verify_receipt(job, receipt)
    with pytest.raises(ValueError, match="another job"):
        queue.verify_receipt({**job, "sha256": "b" * 64}, receipt)
    record = json.loads((tmp_path / "result.json").read_text())
    torch.save(torch.ones(1, 2, 7, 2, 2), tmp_path / "generated.pt")
    record["output"]["sha256"] = sha256(tmp_path / "generated.pt")
    (tmp_path / "result.json").write_text(json.dumps(record))
    assert queue.verify_completion(job)
    with pytest.raises(ValueError, match="receipt evidence changed"):
        queue.verify_receipt(job, receipt)


@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.usefixtures("controlled_queue_launch", "controlled_evaluation_conditions")
def test_child_requires_verified_outputs_and_releases_reaped_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publish: bool
) -> None:

    output = tmp_path / "result"
    claims = GPUClaims(tmp_path / "claims")
    assert claims.acquire((5,), job="case")
    job = {
        "id": "case",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(output),
        "arguments": ["--mode", "causal", "--output", str(output)],
        "completion": {"records": [str(output / "result.json")]},
    }

    def launch(command: list[str], **kwargs) -> SimpleNamespace:
        assert command[-2:] == ["--gpu-id", "0"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "5"
        if publish:
            evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {"mode": "causal"}, output)
        return SimpleNamespace(pid=99999, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(queue, "process_alive", lambda _pid: False)
    state_path = tmp_path / "state.json"
    if publish:
        result = queue.run_child(job, claims, tmp_path / "child.log", state_path=state_path, jobs=[job])
        assert result["returncode"] == 0
        assert result["receipt"]["job_sha256"] == job["sha256"]
    else:
        with pytest.raises(ValueError, match="incomplete queue outputs"):
            queue.run_child(job, claims, tmp_path / "child.log", state_path=state_path, jobs=[job])
    row = json.loads(state_path.read_text())["jobs"]["case"]
    assert row["state"] == ("complete" if publish else "failed")
    assert row["child_pid"] == 99999
    assert len(row["attempts"]) == 1
    assert row["attempts"][0]["child_pid"] == 99999
    if publish:
        assert queue.verify_receipt(job, row["receipt"])
    assert not claims.owned
    assert not (tmp_path / "claims/5").exists()


def test_refused_child_preserves_existing_running_journal(tmp_path: Path) -> None:
    job = {
        "id": "case",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(tmp_path / "output"),
        "arguments": ["--mode", "causal"],
    }
    path = tmp_path / "state.json"
    with queue.queue_state(path, [job]) as state:
        state["jobs"]["case"].update(state="running", child_pid=12345)
    original = path.read_bytes()
    claims = GPUClaims(tmp_path / "claims")
    assert claims.acquire((5,), job="case")
    with pytest.raises(ValueError, match="pending saved job"):
        queue.run_child(job, claims, tmp_path / "child.log", state_path=path, jobs=[job])
    assert path.read_bytes() == original
    assert not claims.owned


def test_interrupted_owner_preserves_claim_for_live_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claims = GPUClaims(tmp_path / "claims")
    assert claims.acquire((5,), job="case")
    job = {
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(tmp_path / "result"),
        "arguments": ["--mode", "causal"],
    }
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=99999, poll=lambda: None))
    monkeypatch.setattr(supervision, "supervise", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        queue.run_child(job, claims, tmp_path / "child.log")
    assert claims.owned == {5}
    assert json.loads((tmp_path / "claims/5").read_text())["child_pid"] == 99999


def test_queue_dry_run_is_read_only_and_reports_planned_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs_path = tmp_path / "jobs.json"
    state_path = tmp_path / "not_created/state.json"
    job = {
        "id": "case",
        "kind": "evaluate",
        "output": "result",
        "dependencies": [],
        "arguments": [
            "--mode",
            "causal",
            "--subset",
            "membership.json",
            "--output",
            "result",
            "--schedule",
            "0.725",
            "0",
        ],
        "completion": {"records": ["result/result.json"]},
    }
    jobs_path.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    original = jobs_path.read_bytes()
    assert queue.main(["--jobs", str(jobs_path), "--state", str(state_path), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["validation"] == "arguments_and_identity_only"
    assert not result["gpu_availability_checked"]
    assert result["jobs"][0]["planned_physical_gpus"] == [0]
    assert result["jobs"][0]["command"][-2:] == ["--gpu-id", "0"]
    assert jobs_path.read_bytes() == original
    assert not state_path.parent.exists()
    assert not (tmp_path / "result").exists()
    with pytest.raises(SystemExit):
        queue.main(["--jobs", str(jobs_path), "--state", str(state_path), "--once"])
    assert not state_path.parent.exists()


def test_queue_execute_once_uses_shared_own_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = {"id": "case", "kind": "evaluate", "output": str(tmp_path / "result"), "sha256": "a" * 64}
    state = {"jobs": {"case": {"state": "pending", "attempts": []}}}
    seen = {}

    ledger = tmp_path / "own_processes.json"

    def dispatch(job: dict, claims: object, log: Path, **kwargs) -> None:
        seen.update(owned=set(claims.owned), log=log, kwargs=kwargs)
        assert isinstance(claims, ProcessRegistry)
        assert claims.path == ledger.resolve()
        row = json.loads(ledger.read_text())["attempts"][claims.token]
        assert row["job"] == job["id"]
        assert row["state"] == "active"
        assert row["gpus"] == [0]
        assert row["owner"] == queue.process_identity(os.getpid())
        claims.release()

    monkeypatch.setattr(queue, "prepare_jobs", lambda _: [job])
    monkeypatch.setattr(queue, "read_queue_state", lambda *_: state)
    monkeypatch.setattr(queue, "ready_jobs", lambda *_: [job])
    monkeypatch.setattr(queue, "run_child", dispatch)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"stdout": "\n".join(f"{gpu}, 0" for gpu in range(8))})(),
    )

    assert (
        queue.main(
            [
                "--jobs",
                "jobs.json",
                "--state",
                str(tmp_path / "state.json"),
                "--execute",
                "--once",
                "--process-ledger",
                str(ledger),
            ]
        )
        == 0
    )
    assert seen["owned"] == {0}
    assert seen["log"].parent == tmp_path / "logs"
    assert seen["kwargs"]["state_path"] == tmp_path / "state.json"
    assert all(row["state"] == "closed" for row in json.loads(ledger.read_text())["attempts"].values())


def test_queue_execute_requires_shared_process_ledger_before_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_args) -> None:
        pytest.fail("invalid execution request prepared jobs")

    monkeypatch.setattr(queue, "prepare_jobs", forbidden)
    with pytest.raises(SystemExit):
        queue.main(["--jobs", "jobs.json", "--state", str(tmp_path / "state.json"), "--execute", "--once"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(("state", "terminal"), [("S", False), ("Z", True)])
def test_process_identity_handles_names_and_terminal_states(tmp_path: Path, state: str, terminal: bool) -> None:
    root = tmp_path / "42"
    root.mkdir()
    fields = [state, *(["0"] * 18), "987654", "0"]
    (root / "stat").write_text("42 (worker (with spaces)) " + " ".join(fields))
    (root / "cmdline").write_bytes(b"python\0-m\0scripts.onestep_avatar.evaluate\0")
    assert queue.process_identity(42, proc_root=tmp_path) == {
        "pid": 42,
        "start_ticks": 987654,
        "command": ["python", "-m", "scripts.onestep_avatar.evaluate"],
        "terminal": terminal,
    }
    assert queue.process_identity(43, proc_root=tmp_path) is None


def test_process_alive_does_not_treat_unreadable_identity_as_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue.os, "kill", lambda *_args: None)
    monkeypatch.setattr(queue, "process_identity", lambda *_args: (_ for _ in ()).throw(PermissionError()))
    assert queue.process_alive(42)
    monkeypatch.setattr(queue, "process_identity", lambda *_args: {"terminal": True})
    assert not queue.process_alive(42)


def test_process_identity_inspects_current_handle() -> None:
    identity = queue.process_identity(os.getpid())
    assert identity["pid"] == os.getpid()
    assert identity["start_ticks"] > 0
    assert identity["command"]
    assert not identity["terminal"]


def test_child_inspection_rejects_reused_pid_and_unrecorded_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {"pid": 42, "start_ticks": 100, "command": ["python", "-m", "owner"], "terminal": False}
    monkeypatch.setattr(queue, "process_identity", lambda _pid: identity)
    row = {"child_pid": 42, "child_identity": dict(identity)}
    assert queue.inspect_child(row) == "live"
    identity["terminal"] = True
    assert queue.inspect_child(row) == "terminal"
    identity["start_ticks"] += 1
    with pytest.raises(ValueError, match="identity differs"):
        queue.inspect_child(row)
    with pytest.raises(ValueError, match="identity differs"):
        queue.inspect_child({"child_pid": 42})
    with pytest.raises(ValueError, match="no recorded PID"):
        queue.inspect_child({})
    monkeypatch.setattr(queue, "process_identity", lambda _pid: None)
    assert queue.inspect_child(row) == "missing"


@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.usefixtures("controlled_evaluation_conditions")
def test_dead_owner_recovery_requires_explicit_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publish: bool
) -> None:

    output = tmp_path / "output"
    job = {
        "id": "case",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(output),
        "arguments": ["--mode", "causal"],
        "completion": {"records": [str(output / "result.json")]},
    }
    path = tmp_path / "state.json"
    state = {
        "schema_version": 1,
        "job_order": ["case"],
        "owner_pid": 99,
        "jobs": {"case": {"sha256": job["sha256"], "state": "running", "child_pid": 98, "attempts": []}},
    }
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    monkeypatch.setattr(queue, "process_alive", lambda _pid: False)
    monkeypatch.setattr(queue, "process_identity", lambda _pid: None)
    if publish:
        evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {"mode": "causal"}, output)
    with pytest.raises(ValueError, match="explicit recovery"), queue.queue_state(path, [job]):
        pytest.fail("interrupted job silently adopted")
    assert path.read_bytes() == original
    with queue.queue_state(path, [job], recover=True) as recovered:
        row = recovered["jobs"]["case"]
        assert row["state"] == ("complete" if publish else "failed")
        assert row["child_pid"] == 98
        if publish:
            assert queue.verify_receipt(job, row["receipt"])
    assert json.loads(path.read_text())["jobs"]["case"]["state"] == row["state"]


def test_recovery_preserves_unknown_launch_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    job = {"id": "case", "sha256": "a" * 64}
    state = {
        "schema_version": 1,
        "job_order": ["case"],
        "owner_pid": 99,
        "jobs": {"case": {"sha256": job["sha256"], "state": "running", "child_pid": None, "attempts": []}},
    }
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    monkeypatch.setattr(queue, "process_alive", lambda _pid: False)
    with pytest.raises(ValueError, match="no recorded PID"), queue.queue_state(path, [job], recover=True):
        pytest.fail("unresolved launch permitted takeover")
    assert path.read_bytes() == original


@pytest.mark.usefixtures("controlled_evaluation_conditions")
def test_ready_jobs_require_unchanged_completed_dependencies(tmp_path: Path) -> None:

    output = tmp_path / "source"
    evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {"mode": "causal"}, output)
    first = {
        "id": "first",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(output),
        "arguments": ["--mode", "causal"],
        "completion": {"records": [str(output / "result.json")]},
    }
    train = {"id": "train", "kind": "train", "sha256": "b" * 64, "dependencies": ["first"]}
    decode = {"id": "decode", "kind": "decode", "sha256": "c" * 64, "dependencies": ["first"]}
    jobs = [first, train, decode]
    state = {"jobs": {job["id"]: {"sha256": job["sha256"], "state": "pending"} for job in jobs}}
    assert queue.ready_jobs(jobs, state) == [first]
    receipt = queue.completion_receipt(first)
    state["jobs"]["first"].update(state="complete", receipt=receipt)
    original = json.dumps(state)
    assert queue.ready_jobs(jobs, state) == [decode, train]
    assert json.dumps(state) == original
    raw = (output / "generated.pt").read_bytes()
    (output / "generated.pt").unlink()
    assert queue.ready_jobs(jobs, state) == []
    assert json.dumps(state) == original
    (output / "generated.pt").write_bytes(raw)
    (output / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt evidence changed"):
        queue.ready_jobs(jobs, state)


@pytest.mark.usefixtures("controlled_evaluation_conditions")
def test_ready_jobs_verify_only_receipts_that_gate_pending_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "source"
    evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {"mode": "causal"}, output)
    first = {
        "id": "first",
        "kind": "evaluate",
        "sha256": "a" * 64,
        "output": str(output),
        "arguments": ["--mode", "causal"],
        "completion": {"records": [str(output / "result.json")]},
    }
    other = {"id": "other", "kind": "decode", "sha256": "c" * 64}
    jobs = [first, other]
    state = {"jobs": {job["id"]: {"sha256": job["sha256"], "state": "pending"} for job in jobs}}
    state["jobs"]["first"].update(state="complete", receipt=queue.completion_receipt(first))
    checked = []
    original = queue.verify_receipt
    def counted(job: dict, receipt: dict) -> bool:
        checked.append(job["id"])
        return original(job, receipt)

    monkeypatch.setattr(queue, "verify_receipt", counted)
    # No pending job depends on "first", so its receipt is not rehashed before dispatch.
    (output / "result.json").write_text("{}")
    assert queue.ready_jobs(jobs, state) == [other]
    assert checked == []
    # A completed job without a receipt is still refused.
    del state["jobs"]["first"]["receipt"]
    with pytest.raises(ValueError, match="no verified receipt"):
        queue.ready_jobs(jobs, state)


@pytest.mark.parametrize("changed_receipt", [False, True])
def test_persistent_child_cannot_start_with_pending_or_changed_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_receipt: bool
) -> None:
    source = {"id": "source", "kind": "evaluate", "sha256": "a" * 64}
    job = {
        "id": "case",
        "kind": "evaluate",
        "sha256": "b" * 64,
        "output": str(tmp_path / "output"),
        "arguments": ["--mode", "causal"],
        "dependencies": ["source"],
    }
    path = tmp_path / "state.json"
    prior_receipt = {"job_sha256": source["sha256"], "evidence": []}
    with queue.queue_state(path, [source, job]) as state:
        if changed_receipt:
            state["jobs"]["source"].update(state="complete", receipt={**prior_receipt, "changed": True})
    original = path.read_bytes()
    claims = GPUClaims(tmp_path / "claims")
    assert claims.acquire((5,), job="case")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("blocked child started"))
    with pytest.raises(ValueError, match="dependencies are not verified"):
        queue.run_child(job, claims, tmp_path / "child.log", state_path=path, jobs=[source, job],
                        dependency_receipts={"source": prior_receipt} if changed_receipt else None)
    assert path.read_bytes() == original
    assert not claims.owned


@pytest.mark.parametrize(
    "row", [None, [], {"attempts": None}, {"attempts": [None]}, {"attempts": [], "child_pid": True}]
)
def test_corrupt_state_rows_cannot_become_new_pending_jobs(tmp_path: Path, row: object) -> None:
    path = tmp_path / "state.json"
    job = {"id": "case", "sha256": "a" * 64}
    if isinstance(row, dict):
        row = {"sha256": job["sha256"], "state": "pending", **row}
    path.write_text(json.dumps({"schema_version": 1, "job_order": ["case"], "jobs": {"case": row}}))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="queue"), queue.queue_state(path, [job]):
        pytest.fail("corrupt state permitted ownership")
    assert path.read_bytes() == original
