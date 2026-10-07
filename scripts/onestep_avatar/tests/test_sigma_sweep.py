"""Saved sigma-sweep decoding has one package owner and checks inputs before native sessions."""

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import media, queue, sigma_sweep
from scripts.onestep_avatar.hashing import sha256
from scripts.prune.core import model_registry as registry


@pytest.fixture
def saved_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    vae = tmp_path / "vae.bin"
    vae.write_bytes(b"controlled VAE identity")
    spec = {
        "schema_version": 1,
        "model": "2.5",
        "tag": "matched",
        "fps": 30,
        "decode_seed": 42,
        "vae": {"path": str(vae), "sha256": sha256(vae)},
        "cells": [],
    }
    for role in ("capture", "guide"):
        path = tmp_path / f"{role}.pt"
        torch.save({"schema_version": 2, "master": torch.full((3, 18, 2, 2), 0.1).bfloat16(), "fps": 30}, path)
        spec[role] = {"path": str(path), "sha256": sha256(path)}
    for sigma in sigma_sweep.LEVELS:
        for arm in ("d0", "d1"):
            path = tmp_path / f"{sigma}_{arm}.pt"
            torch.save(torch.full((1, 3, 17, 2, 2), 0.1).bfloat16(), path)
            spec["cells"].append({"sigma": sigma, "arm": arm, "path": str(path), "sha256": sha256(path)})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(
        registry,
        "resolve",
        lambda *_a: SimpleNamespace(
            caps=SimpleNamespace(latent_channels=3), paths=SimpleNamespace(video_vae=lambda: vae)
        ),
    )
    calls = []

    def session(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls.append("session")
        return SimpleNamespace(device=torch.device("cpu"), decoder=lambda: nullcontext(None))

    def decode(_session: object, latent: torch.Tensor, _decoder: object, seed: int) -> torch.Tensor:
        calls.append((latent.clone(), seed))
        return (torch.arange(129).float().view(129, 1, 1, 1) / 1000 + latent.float().mean()).expand(129, 3, 2, 2)

    monkeypatch.setattr(media, "open_decoder_session", session)
    monkeypatch.setattr(media, "decode", decode)
    return spec, path, tmp_path / "output", calls


def sweep_queue_job(path: Path, output: Path) -> dict:
    """Prepare the real queue command through its public parser and identity owner."""
    job = {"id": "matched", "kind": "sigma_sweep",
           "arguments": ["--spec", str(path), "--output", str(output)],
           "output": str(output), "completion": {"manifest": str(output / "manifest.json")}}
    jobs = path.parent / "queue_jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    return queue.prepare_jobs(jobs)[0]


def test_queue_sweep_command_and_complete_receipt_use_real_owner(saved_sweep: tuple) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    command, environment = queue.job_command(job, (4,))
    assert command[2:4] == ["scripts.onestep_avatar.sigma_sweep", "--spec"]
    assert command[-2:] == ["--gpu-id", "0"]
    assert environment == {"CUDA_VISIBLE_DEVICES": "4"}
    assert not queue.verify_completion(job)
    assert not calls
    sigma_sweep.execute(path, output, gpu_id=4)
    before = len(calls)
    receipt = queue.completion_receipt(job)
    assert queue.verify_receipt(job, receipt)
    assert len(receipt["evidence"]) == 1
    assert len(calls) == before
    # A movie outside a report sheet is still part of completion verification.
    movie = output / "videos" / "sigma0.725000_d1.mp4"
    movie.write_bytes(movie.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=r"content changed|bytes changed|hash"):
        queue.verify_receipt(job, receipt)
    assert len(calls) == before


def test_queue_sweep_refuses_changed_spec_before_launch_and_completion(saved_sweep: tuple) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="specification changed"):
        queue.job_command(job, (5,))
    with pytest.raises(ValueError, match="specification changed"):
        queue.verify_completion(job)
    assert not calls


@pytest.mark.parametrize("flag", ["--verify", "--gpu-id=6"])
def test_queue_sweep_rejects_execution_overrides(saved_sweep: tuple, flag: str) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    job["arguments"].append(flag)
    jobs = path.parent / "bad_jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    with pytest.raises(ValueError, match="forbidden execution override"):
        queue.prepare_jobs(jobs)
    assert not calls


def test_queue_sweep_refuses_wrong_completion_destination(saved_sweep: tuple) -> None:
    _, path, output, calls = saved_sweep
    job = sweep_queue_job(path, output)
    job["completion"]["manifest"] = str(output / "another.json")
    jobs = path.parent / "bad_jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    with pytest.raises(ValueError, match="completion manifest differs"):
        queue.prepare_jobs(jobs)
    assert not calls


def test_ten_matched_decodes_preserve_metrics_controls_and_real_media(saved_sweep: tuple) -> None:
    spec, path, output, calls = saved_sweep
    manifest = sigma_sweep.execute(path, output, gpu_id=4)
    assert calls[0] == "session"
    assert len(calls) == 11
    assert all(seed == 42 and tuple(latent.shape) == (1, 3, 17, 2, 2) for latent, seed in calls[1:])
    assert len(manifest["outputs"]) == 10
    assert manifest["spec"] == spec
    for row in manifest["outputs"].values():
        assert sha256(Path(row["video"])) == row["sha256"]
        assert [sample["frame"] for sample in row["samples"]] == list(sigma_sweep.SAMPLES)
        assert all(sha256(Path(sample["path"])) == sample["sha256"] for sample in row["samples"])
    media.verify_saved_video(Path(manifest["outputs"]["capture"]["video"]), 129, 30)
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(metrics["cells"]) == 8
    assert metrics["sigma1_d0_equals_d1"]
    assert metrics["sigma1_d0_vs_d1_max_abs"] == 0
    assert all(cell["c0_equal"] and cell["motion_over_capture"] == 1 for cell in metrics["cells"])
    assert (output / "manifest.json").is_file()
    before = len(calls)
    assert sigma_sweep.verify_completion(path, output) == manifest
    sigma_sweep.main(["--spec", str(path), "--output", str(output), "--verify"])
    assert len(calls) == before


@pytest.mark.parametrize("defect", ["hash", "inventory", "nan", "short", "fps", "vae", "output"])
def test_bad_saved_inputs_fail_before_session_or_output(saved_sweep: tuple, defect: str) -> None:
    spec, path, output, calls = saved_sweep
    if defect == "hash":
        spec["cells"][0]["sha256"] = "0" * 64
    elif defect == "inventory":
        spec["cells"].pop()
    elif defect in ("nan", "short"):
        cell = spec["cells"][0]
        tensor = torch.load(cell["path"], weights_only=True)
        if defect == "nan":
            tensor[0, 0, 0, 0, 0] = float("nan")
        else:
            tensor = tensor[:, :, :16]
        torch.save(tensor, cell["path"])
        cell["sha256"] = sha256(Path(cell["path"]))
    elif defect == "fps":
        spec["fps"] = 25
    elif defect == "vae":
        spec["vae"]["sha256"] = "0" * 64
    else:
        output.mkdir()
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="sigma sweep"):
        sigma_sweep.execute(path, output, gpu_id=4)
    assert not calls
    assert not output.exists() or not list(output.iterdir())


def test_input_change_during_decode_prevents_publication(saved_sweep: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, path, output, _ = saved_sweep
    original = media.decode

    def changed(*args: object) -> torch.Tensor:
        pixels = original(*args)
        Path(spec["cells"][0]["path"]).write_bytes(b"changed during decoding")
        return pixels

    monkeypatch.setattr(media, "decode", changed)
    with pytest.raises(ValueError, match="changed during decoding"):
        sigma_sweep.execute(path, output, gpu_id=4)
    assert not output.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "video",
        "sample",
        "inventory",
        "pixel_digest",
        "geometry",
        "metric_inventory",
        "metric_control",
        "nonfinite_metric",
    ],
)
def test_saved_completion_rejects_changed_media_and_controls(saved_sweep: tuple, defect: str) -> None:
    _, path, output, calls = saved_sweep
    manifest = sigma_sweep.execute(path, output, gpu_id=4)
    row = manifest["outputs"]["sigma0.725000_d0"]
    if defect == "video":
        Path(row["video"]).write_bytes(b"changed encoded movie")
    elif defect == "sample":
        Path(row["samples"][8]["path"]).write_bytes(b"changed unused D0 frame 66")
    elif defect == "inventory":
        manifest["outputs"].pop("guide")
    elif defect == "pixel_digest":
        row["pixel_sha256"] = None
    elif defect == "geometry":
        row["width"] = 20
    else:
        metrics = json.loads((output / "metrics.json").read_text())
        if defect == "metric_inventory":
            metrics["cells"].pop()
        elif defect == "metric_control":
            metrics["sigma1_d0_equals_d1"] = False
        else:
            metrics["cells"][0]["mean_boundary_ratio"] = float("nan")
        (output / "metrics.json").write_text(json.dumps(metrics))
        manifest["metrics"]["sha256"] = sha256(output / "metrics.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    before = len(calls)
    with pytest.raises(ValueError, match=r"sigma sweep"):
        sigma_sweep.verify_completion(path, output)
    assert len(calls) == before


def test_short_movie_prevents_complete_manifest(saved_sweep: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    from ltx_trainer import video_utils  # noqa: PLC0415 -- controlled writer failure

    _, path, output, _ = saved_sweep
    original = video_utils.save_video

    def shortened(pixels: torch.Tensor, destination: Path, **settings: object) -> None:
        original(pixels[:5], destination, **settings)

    monkeypatch.setattr(video_utils, "save_video", shortened)
    with pytest.raises(ValueError, match="frame count"):
        sigma_sweep.execute(path, output, gpu_id=4)
    assert (output / "videos/capture.mp4").exists()
    assert not (output / "manifest.json").exists()


def test_report_refuses_changed_movie_outside_selected_samples(saved_sweep: tuple) -> None:
    _, path, output, calls = saved_sweep
    manifest = sigma_sweep.execute(path, output, gpu_id=4)
    source = Path(__file__).resolve().parents[4] / "expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926/sheets.py"
    spec = importlib.util.spec_from_file_location("saved_sweep_report", source)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    Path(manifest["outputs"]["sigma0.725000_d0"]["video"]).write_bytes(b"changed movie")
    before = len(calls)
    destination = output.parent / "refused_report"
    with pytest.raises(ValueError, match="saved media path or bytes changed"):
        report.build(output / "manifest.json", destination)
    assert not destination.exists()
    assert len(calls) == before
