"""Real small CPU causal results feed the sweep decoder only through scientific completion."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.experiments import sigma_sweep
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.test_evaluation_completion import completed  # noqa: F401 -- shared CPU model fixture
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- transitive fixture
from scripts.onestep_avatar.tests.test_training_preflight import checked_settings  # noqa: F401 -- transitive fixture
from scripts.prune.core import model_registry


@pytest.fixture
def completed_sweep(completed: tuple, monkeypatch: pytest.MonkeyPatch) -> tuple:  # noqa: F811 -- imported pytest fixture
    _, _, settings, membership, calls = completed
    specification = evaluate.backbone.resolve("2.5", "distilled")
    specification.sigmas = [0.421875, 0.725, 0.909375, 1.0]
    vae = Path(specification.paths.video_vae())
    vae.write_bytes(b"controlled VAE")
    monkeypatch.setattr(model_registry, "resolve", lambda *_a: specification)
    for source in membership["sources"]:
        view = Path(membership["corpus_root"]) / source["relative_dir"]
        capture_path = view / dataset.capture_bundle_name("white")
        capture = torch.load(capture_path, weights_only=True)
        capture["master"] = torch.ones(2, 17, 2, 2, dtype=torch.bfloat16)
        torch.save(capture, capture_path)
        source.update(shape=[2, 17, 2, 2], capture_latent_sha256=sha256(capture_path))
        render = view / dataset.render_name("white")
        render.write_bytes(b"controlled guide")
        sidecar = view / dataset.render_metadata_name("white")
        sidecar.write_text(json.dumps({"objective": "white",
                                      "compositing_version": dataset.GUIDE_COMPOSITING_VERSION}))
        encoding = {**source["capture_encode_record"], "input_fingerprint": sha256(render)}
        guide_path = view / dataset.guide_bundle_name("white")
        torch.save({**capture, **encoding, "master": capture["master"] + 1}, guide_path)
        source.update(guide_latent_sha256=sha256(guide_path), guide_encode_record=encoding,
                      guide_sidecar_sha256=sha256(sidecar))
    membership["sha256"] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    source = membership["sources"][0]
    view = Path(membership["corpus_root"]) / source["relative_dir"]
    noise_path = settings.output.parent / "shared_noise.pt"
    torch.save(torch.ones(1, 68, 2, dtype=torch.bfloat16), noise_path)
    spec = {"schema_version": 2, "model": "2.5", "tag": "controlled", "decode_seed": 42,
            "fps": 30, "vae": {"path": str(vae), "sha256": sha256(vae)}, "cells": []}
    for role in ("capture", "guide"):
        path = view / (dataset.capture_bundle_name("white") if role == "capture"
                       else dataset.guide_bundle_name("white"))
        spec[role] = {"path": str(path), "sha256": sha256(path)}
    for sigma, (_, count) in sigma_sweep.LEVELS.items():
        schedule = {1: [sigma, 0], 2: [0.725, 0.421875, 0],
                    3: [0.909375, 0.725, 0.421875, 0],
                    8: [1.0, 0.99, 0.95, 0.909375, 0.8, 0.725, 0.6, 0.421875, 0]}[count]
        specification.sigmas = sorted(set(specification.sigmas + schedule[:-1]))
        for arm in ("d0", "d1"):
            output = settings.output.parent / f"run_{sigma}_{arm}"
            command = ["--mode", "causal", "--subset", str(settings.subset), "--output", str(output),
                       "--model", "2.5", "--variant", "distilled", "--guide-mode", arm,
                       "--source", source["relative_dir"], "--noise-file", str(noise_path),
                       "--span-latent-frames", "17", "--blocks-per-sample", "8",
                       "--schedule", *map(str, schedule), "--seed", "42"]
            evaluate.execute_evaluation(evaluate.parse_args(command))
            record = output / "case_0000/variant_000/result.json"
            job = {"id": f"{sigma}_{arm}", "kind": "evaluate", "arguments": command,
                   "output": str(output), "completion": {"records": [str(record)]}}
            spec["cells"].append({"sigma": sigma, "arm": arm, "evaluation_job": job})
    path = settings.output.parent / "decode_spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(sigma_sweep.media, "open_decoder_session",
                        lambda *_a, **_k: pytest.fail("resolution opened a decoder"))
    return spec, path, settings.output.parent / "decoded", calls


def test_actual_causal_results_resolve_without_models_or_writes(completed_sweep: tuple) -> None:
    spec, path, output, calls = completed_sweep
    before = len(calls)
    normalized, tensors, hashes, _ = sigma_sweep.prepare(path, output)
    assert len(tensors) == 10
    assert len(calls) == before
    assert not output.exists()
    assert json.loads(path.read_text()) == spec
    for cell in normalized["cells"]:
        assert sha256(Path(cell["path"])) == cell["sha256"]
        assert cell["path"] in hashes
    assert sum(name.endswith("result.json") for name in hashes) == 8
    assert torch.equal(tensors["sigma1.000000_d0"], tensors["sigma1.000000_d1"])


@pytest.mark.parametrize("defect", ["seed", "arm", "noise", "record", "masters"])
def test_changed_requested_or_saved_conditions_fail_before_decoder(completed_sweep: tuple, defect: str) -> None:
    spec, path, output, calls = completed_sweep
    cell = spec["cells"][-1]
    job = cell["evaluation_job"]
    if defect in ("seed", "arm"):
        flag = "--seed" if defect == "seed" else "--guide-mode"
        job["arguments"][job["arguments"].index(flag) + 1] = "43" if defect == "seed" else "d0"
    elif defect == "masters":
        spec["guide"]["sha256"] = "0" * 64
    elif defect == "noise":
        target = Path(job["output"]) / "case_0000/noise.pt"
        torch.save(torch.load(target, weights_only=True) + 1, target)
    else:
        target = Path(job["completion"]["records"][0])
        record = json.loads(target.read_text())
        record["seed"] = 43
        target.write_text(json.dumps(record))
    path.write_text(json.dumps(spec))
    before = len(calls)
    with pytest.raises(ValueError, match=r"sweep|queue evaluation"):
        sigma_sweep.prepare(path, output)
    assert len(calls) == before
    assert not output.exists()


def test_result_bound_decode_and_receipt_reverify_all_generation_evidence(
    completed_sweep: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path, output, calls = completed_sweep
    monkeypatch.setattr(sigma_sweep.media, "open_decoder_session", lambda *_a, **_k: SimpleNamespace(
        decoder=lambda: nullcontext(None)))
    monkeypatch.setattr(sigma_sweep.media, "decode", lambda _s, latent, _d, _seed:
                        (torch.arange(129).float().view(129, 1, 1, 1) / 1000
                         + latent.float().mean().sigmoid() / 2).expand(129, 3, 2, 2))
    manifest = sigma_sweep.execute(path, output, gpu_id=4)
    assert manifest["spec"]["schema_version"] == 2
    job = {"id": "decode", "kind": "experiment", "experiment": "sigma_sweep",
           "spec": str(path), "spec_sha256": sha256(path), "arguments": ["--spec", str(path), "--output", str(output)],
           "output": str(output), "completion": {"manifest": str(output / "manifest.json")}}
    jobs = path.parent / "decoding_jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    prepared = queue.prepare_jobs(jobs)[0]
    before = len(calls)
    receipt = queue.completion_receipt(prepared)
    assert queue.verify_receipt(prepared, receipt)
    assert len(calls) == before
    spec = json.loads(path.read_text())
    text = Path(spec["cells"][0]["evaluation_job"]["output"]) / "text.pt"
    torch.save(torch.load(text, weights_only=True) + 1, text)
    with pytest.raises(ValueError, match=r"queue evaluation|sweep"):
        queue.verify_receipt(prepared, receipt)
    assert len(calls) == before


def test_missing_result_refuses_before_decoder(completed_sweep: tuple) -> None:
    spec, path, output, calls = completed_sweep
    Path(spec["cells"][0]["evaluation_job"]["completion"]["records"][0]).unlink()
    before = len(calls)
    with pytest.raises(FileNotFoundError):
        sigma_sweep.prepare(path, output)
    assert len(calls) == before
    assert not output.exists()


def test_decoder_readiness_requires_all_eight_unchanged_receipts(completed_sweep: tuple) -> None:
    spec, path, output, calls = completed_sweep
    generation = [cell["evaluation_job"] for cell in spec["cells"]]
    decode = {"id": "decode", "kind": "experiment", "experiment": "sigma_sweep",
           "spec": str(path), "spec_sha256": sha256(path), "arguments": ["--spec", str(path), "--output", str(output)],
              "output": str(output), "dependencies": [job["id"] for job in generation],
              "completion": {"manifest": str(output / "manifest.json")}}
    jobs_path = path.parent / "pipeline_jobs.json"
    jobs_path.write_text(json.dumps({"schema_version": 1, "jobs": [*generation, decode]}))
    jobs = queue.prepare_jobs(jobs_path)
    state = {"jobs": {job["id"]: {"sha256": job["sha256"], "state": "pending"} for job in jobs}}
    before = len(calls)
    for job in jobs[:-2]:
        state["jobs"][job["id"]].update(state="complete", receipt=queue.completion_receipt(job))
    assert [job["id"] for job in queue.ready_jobs(jobs, state)] == [generation[-1]["id"]]
    last = jobs[-2]
    state["jobs"][last["id"]].update(state="complete", receipt=queue.completion_receipt(last))
    assert [job["id"] for job in queue.ready_jobs(jobs, state)] == ["decode"]
    text = Path(generation[0]["output"]) / "text.pt"
    torch.save(torch.load(text, weights_only=True) + 1, text)
    with pytest.raises(ValueError, match="receipt evidence changed"):
        queue.ready_jobs(jobs, state)
    assert len(calls) == before


def test_late_evidence_change_refuses_before_decoder(
    completed_sweep: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path, output, calls = completed_sweep
    verify = queue.verify_completion

    def change_after_verification(job: dict) -> bool:
        result = verify(job)
        text = Path(job["output"]) / "text.pt"
        torch.save(torch.load(text, weights_only=True) + 1, text)
        return result

    monkeypatch.setattr(queue, "verify_completion", change_after_verification)
    before = len(calls)
    with pytest.raises(ValueError, match="evidence changed during verification"):
        sigma_sweep.prepare(path, output)
    assert len(calls) == before
    assert not output.exists()
