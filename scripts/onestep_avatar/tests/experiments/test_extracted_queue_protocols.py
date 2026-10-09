"""Queued extracted diagnostics retain full raw-control verification and relative path parsing."""

from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ltx_core.model.transformer.model import X0Model
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.experiments import causality, fusion_parity
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.experiments.test_causality_diagnostic import fixture_inputs
from scripts.onestep_avatar.tests.test_causal_core import CHANNELS, SCALE, _context, _model


def job(selector: str, spec: Path, output: Path) -> dict:
    return {
        "id": selector,
        "kind": "experiment",
        "experiment": selector,
        "spec": str(spec),
        "spec_sha256": sha256(spec),
        "arguments": ["--spec", str(spec), "--output", str(output)],
        "output": str(output),
        "completion": {"manifest": str(output / "manifest.json")},
    }


@pytest.mark.parametrize("selector", ["causality", "fusion_parity"])
def test_relative_job_paths_normalize_before_spec_reads(selector: str, tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    output = tmp_path / "output"
    raw = job(selector, spec, output)
    absolute = queue.prepare_job(raw, tmp_path)
    relative = deepcopy(raw)
    relative.update(
        spec=spec.name,
        output=output.name,
        arguments=["--spec", spec.name, "--output", output.name],
        completion={"manifest": str(Path(output.name) / "manifest.json")},
    )
    assert queue.prepare_job(relative, tmp_path) == absolute
    assert queue.prepare_job(absolute, tmp_path) == absolute
    command, _environment = queue.job_command(absolute, (0,))
    assert command[2] == queue.EXPERIMENTS[selector]
    assert not queue.verify_completion(absolute)
    assert not output.exists()
    spec.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="specification changed"):
        queue.job_command(absolute, (0,))
    with pytest.raises(ValueError, match="specification changed"):
        queue.verify_completion(absolute)


@pytest.fixture
def completed_eight_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    import scripts.prune.core.session as sessions  # noqa: PLC0415 -- native fixture session dependency

    dev, adapter, view, _capture, _guide = fixture_inputs(tmp_path, monkeypatch)
    capture = torch.randn(CHANNELS, 17, 2, 2, generator=torch.Generator().manual_seed(8))
    guide = torch.randn(CHANNELS, 17, 2, 2, generator=torch.Generator().manual_seed(9))
    guide[0, 0, 0, 0] = 0.0
    monkeypatch.setattr(
        dataset,
        "load_training_master",
        lambda path: (capture if path.name == dataset.capture_bundle_name("white") else guide, 30),
    )
    monkeypatch.setattr(causality.checkpoints, "check_adapter_conditions", lambda *_a, **_k: None)
    session = SimpleNamespace(
        device=torch.device("cpu"),
        context=_context().to(torch.bfloat16),
        model=SimpleNamespace(scale_factors=SCALE, caps=SimpleNamespace(latent_channels=CHANNELS)),
        transformer=lambda *_a, **_k: nullcontext(X0Model(_model().to(torch.bfloat16))),
    )
    monkeypatch.setattr(sessions, "open_session", lambda *_a, **_k: session)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "eight_block",
                "checkpoint": str(adapter),
                "view": str(view),
                "sigma": 0.421875,
            }
        )
    )
    output = tmp_path / "output"
    causality.main(["--spec", str(spec), "--output", str(output)])
    prepared = queue.prepare_job(job("causality", spec, output), tmp_path)
    monkeypatch.setattr(sessions, "open_session", lambda *_a, **_k: pytest.fail("completion opened native weights"))
    assert queue.verify_completion(prepared)
    assert len(queue.completion_receipt(prepared)["evidence"]) == 4
    assert dev.is_file()
    return prepared, output


@pytest.mark.parametrize("defect", ["raw_bytes", "missing", "summary_rehashed"])
def test_eight_block_completion_rechecks_full_saved_controls(completed_eight_block: tuple, defect: str) -> None:
    prepared, output = completed_eight_block
    raw = output / "raw_outputs.pt"
    if defect == "missing":
        raw.unlink()
        with pytest.raises(FileNotFoundError):
            queue.verify_completion(prepared)
    elif defect == "raw_bytes":
        original = raw.read_bytes()
        raw.write_bytes(original + b"changed serialization")
        with pytest.raises(ValueError, match="artifacts differ"):
            queue.verify_completion(prepared)
        raw.write_bytes(original)
        assert queue.verify_completion(prepared)
    else:
        result = output / "result.json"
        data = json.loads(result.read_text())
        data["earlier_blocks_bit_identical"] = False
        result.write_text(json.dumps(data))
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][str(result.resolve())] = sha256(result)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="diagnostic differs"):
            queue.verify_completion(prepared)


@pytest.fixture
def completed_fusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    run, view = tmp_path / "run", tmp_path / "view"
    (run / "checkpoints").mkdir(parents=True)
    view.mkdir()
    dev = tmp_path / "dev.safetensors"
    paths = [
        run / "config.json",
        dev,
        run / "checkpoints/lora_weights_step_00000.safetensors",
        run / "checkpoints/lora_weights_step_00001.safetensors",
        view / dataset.capture_bundle_name("white"),
        view / dataset.guide_bundle_name("white"),
    ]
    for path in paths:
        path.write_bytes(b"pinned scientific input")
    monkeypatch.setattr(fusion_parity.backbone, "transformer_path", lambda *_a: dev)
    capture = torch.ones(CHANNELS, 17, 2, 2)
    monkeypatch.setattr(dataset, "load_training_master", lambda _path: (capture, 30))
    outputs = {
        name: torch.full((1, 12, CHANNELS), value)
        for name, value in [("bare", 1.0), ("step0", 1.0), ("fused1", 1.18), ("peft0", 1.0), ("peft1", 1.2)]
    }
    for value in outputs.values():
        value[:, :4] = 1.0

    def execute(_run: Path, _view: Path, output: Path, *, gpu_id: int, step: int, output_tensors_path: Path) -> dict:
        assert gpu_id == 0
        assert step == 1
        output.parent.mkdir()
        torch.save(outputs, output_tensors_path)
        result = {
            "view": str(view),
            "trained_step": 1,
            "sigma": 0.421875,
            "block": 0,
            "noise_seed": 42,
            **fusion_parity.fusion_parity_metrics(outputs),
            "input_sha256": {str(path.resolve()): sha256(path) for path in paths},
        }
        output.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(fusion_parity, "evaluate_fusion_parity", execute)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema_version": 1, "protocol": "fusion_parity", "run": str(run), "view": str(view), "step": 1})
    )
    output = tmp_path / "output"
    fusion_parity.main(["--spec", str(spec), "--output", str(output)])
    prepared = queue.prepare_job(job("fusion_parity", spec, output), tmp_path)
    monkeypatch.setattr(
        fusion_parity, "evaluate_fusion_parity", lambda *_a, **_k: pytest.fail("completion loaded model")
    )
    assert queue.verify_completion(prepared)
    assert len(queue.completion_receipt(prepared)["evidence"]) == 3
    return prepared, output


@pytest.mark.parametrize("field", ["capture_sha256", "guide_sha256", "text_sha256", "original_noise", "mixed_noise"])
def test_eight_block_completion_binds_every_input_digest(completed_eight_block: tuple, field: str) -> None:
    prepared, output = completed_eight_block
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    if field in ("original_noise", "mixed_noise"):
        result["records"][0 if field == "original_noise" else 1]["noise_sha256"] = "0" * 64
    else:
        result[field] = "0" * 64
    result_path.write_text(json.dumps(result))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][str(result_path.resolve())] = sha256(result_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest differs from raw inputs"):
        queue.verify_completion(prepared)


@pytest.mark.parametrize(
    "defect", ["missing", "nonfinite", "shape", "earlier_noise", "unchanged_later", "master_signed_zero"]
)
def test_eight_block_completion_rechecks_raw_inputs(completed_eight_block: tuple, defect: str) -> None:
    prepared, output = completed_eight_block
    raw_path = output / "raw_inputs.pt"
    if defect == "missing":
        raw_path.unlink()
        with pytest.raises(FileNotFoundError):
            queue.verify_completion(prepared)
        return
    inputs = torch.load(raw_path, weights_only=True)
    if defect == "nonfinite":
        inputs["text"][0, 0, 0] = float("nan")
    elif defect == "shape":
        inputs["original_noise"] = inputs["original_noise"][:, :-1]
    elif defect == "earlier_noise":
        inputs["mixed_noise"][:, 0] += 1
    elif defect == "master_signed_zero":
        original = inputs["guide"].clone()
        inputs["guide"][0, 0, 0] = -0.0
        assert torch.equal(original, inputs["guide"])
    else:
        inputs["mixed_noise"] = inputs["original_noise"].clone()
    torch.save(inputs, raw_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][str(raw_path.resolve())] = sha256(raw_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"raw input|raw noise"):
        queue.verify_completion(prepared)


def test_eight_block_completion_requires_native_saved_output_precision(completed_eight_block: tuple) -> None:
    prepared, output = completed_eight_block
    raw_path = output / "raw_outputs.pt"
    outputs = torch.load(raw_path, weights_only=True)
    torch.save([value.double() for value in outputs], raw_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][str(raw_path.resolve())] = sha256(raw_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="saved output shape, c0 or values differ"):
        queue.verify_completion(prepared)


@pytest.mark.parametrize("defect", ["shape", "c0", "precision"])
def test_fusion_completion_rechecks_block_geometry_and_c0(completed_fusion: tuple, defect: str) -> None:
    prepared, output = completed_fusion
    raw_path = output / "raw_outputs.pt"
    outputs = torch.load(raw_path, weights_only=True)
    if defect == "shape":
        outputs = {name: value[:, :-1] for name, value in outputs.items()}
    elif defect == "c0":
        outputs["fused1"][:, 0] += 1
    else:
        outputs = {name: value.double() for name, value in outputs.items()}
    torch.save(outputs, raw_path)
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    result.update(fusion_parity.fusion_parity_metrics(outputs))
    result_path.write_text(json.dumps(result))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in (raw_path, result_path):
        manifest["artifacts"][str(path.resolve())] = sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape, precision or clean c0"):
        queue.verify_completion(prepared)


@pytest.mark.parametrize("defect", ["missing", "metric_rehashed", "input"])
def test_fusion_completion_recomputes_metrics_and_input_identity(completed_fusion: tuple, defect: str) -> None:
    prepared, output = completed_fusion
    if defect == "missing":
        (output / "raw_outputs.pt").unlink()
        with pytest.raises(FileNotFoundError):
            queue.verify_completion(prepared)
    elif defect == "input":
        args = fusion_parity.read_spec(fusion_parity.parse_args(prepared["arguments"]))
        (args.run / "config.json").write_bytes(b"changed scientific input")
        with pytest.raises(ValueError, match="input bytes differ"):
            queue.verify_completion(prepared)
    else:
        result_path = output / "result.json"
        result = json.loads(result_path.read_text())
        result["rel_l2_effect_fused_vs_effect_peft"] = 0
        result_path.write_text(json.dumps(result))
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][str(result_path.resolve())] = sha256(result_path)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="metrics differ"):
            queue.verify_completion(prepared)
