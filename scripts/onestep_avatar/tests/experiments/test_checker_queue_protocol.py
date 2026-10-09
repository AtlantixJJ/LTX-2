"""Saved scientific arrays and strict queue specs, without models or CUDA."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import hashing
from scripts.onestep_avatar.experiments import adapter_effect_check as effect
from scripts.onestep_avatar.experiments import continuation_check as continuation
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.experiments import stock_parity as stock
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone, causal, common
from scripts.onestep_avatar.tests.experiments.test_adapter_effect_check import saved_effect as create_saved_effect
from scripts.onestep_avatar.tests.experiments.test_continuation_check import inputs as continuation_inputs
from scripts.onestep_avatar.tests.experiments.test_stock_parity import Geometry, reference_case
from scripts.onestep_avatar.tests.test_causal_core import CHANNELS, EDGE, FPS, _model
from scripts.onestep_avatar.training import resources


def write_spec(path: Path, selector: str, arguments: list[str]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "protocol": selector, "arguments": arguments}))
    return path


@pytest.fixture(params=[False, True], ids=["one_update", "pilot"])
def saved_effect(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple:
    return create_saved_effect.__wrapped__(tmp_path, request, monkeypatch)


@pytest.mark.parametrize("owner", [stock, effect, continuation])
def test_queue_outer_parse_does_not_open_relative_spec(
    owner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("outer parser must not open a scientific file")

    monkeypatch.setattr(Path, "read_text", refuse)
    args = owner.parse_args(["--spec", "input.json", "--output", "output"])
    assert args.spec == Path("input.json")
    assert args.output == Path("output")
    assert args.completion == {"manifest": str(Path("output/manifest.json").resolve())}
    for flag in ("--supervise", "--launch-record", "--process-ledger", "--phase"):
        with pytest.raises(SystemExit):
            owner.parse_args(["--spec", "input.json", "--output", "output", flag, "extra"])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("owner", "selector"),
    [(stock, "stock_parity"), (effect, "adapter_effect_check"), (continuation, "continuation_check")],
)
@pytest.mark.parametrize("flag", ["--output", "--gpu-id", "--dry-run", "--supervise", "--launch-record"])
def test_scientific_spec_rejects_execution_overrides(
    owner: ModuleType, selector: str, flag: str, tmp_path: Path
) -> None:
    spec = write_spec(tmp_path / "spec.json", selector, [flag, "0"])
    with pytest.raises(ValueError, match="override"):
        owner.read_spec(spec, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.fixture
def completed_stock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    root = tmp_path / "stock"
    root.mkdir()
    _grid, image, context, levels, protocol, record = reference_case(root)
    first, text = tmp_path / "image.pt", tmp_path / "text.json"
    first.write_bytes(b"CPU fixture image identity")
    text.write_text("{}")
    spec = write_spec(
        tmp_path / "stock.json",
        "stock_parity",
        ["--first-image", str(first), "--text-record", str(text), "--frames", "3", "--steps", "4"],
    )
    model = SimpleNamespace(scale_factors=Geometry.scale_factors, caps=SimpleNamespace(latent_channels=128))
    monkeypatch.setattr(
        stock,
        "prepare",
        lambda _args, **_kwargs: (model, image, context, 30.0, levels, "checked prompt", {}, protocol["software"]),
    )
    record["execution"] = {
        "paths": {
            name: {"forward_calls": 4, "seconds_including_traces": 1.0}
            for name in ("stock", "stock_repeat", "bidirectional", "bidirectional_default")
        }
    }
    (root / "result.json").write_text(json.dumps(record))
    (root / "protocol.json").write_text(json.dumps(protocol))
    comparison = root / "comparison"
    comparison.mkdir()
    video, poster = comparison / "comparison.mp4", comparison / "poster.png"
    video.write_bytes(b"bounded RGB fixture")
    poster.write_bytes(b"poster fixture")
    render = {
        "fps": 30.0,
        "software": protocol["software"],
        "panels": [{"role": name} for name in ("stock", "stock_repeat", "bidirectional", "bidirectional_default")],
        "outputs": {
            "video": {"path": str(video), "sha256": sha256(video)},
            "poster": {"path": str(poster), "sha256": sha256(poster)},
        },
    }
    (comparison / "rendering.json").write_text(json.dumps(render))
    monkeypatch.setattr(stock.Session, "__init__", lambda *_a, **_kw: pytest.fail("completion loaded a model"))
    spec_protocol.publish(stock.read_spec(spec, root), "stock_parity")
    return spec, root, record


def test_stock_completion_recomputes_four_actual_paths(completed_stock: tuple) -> None:
    spec, root, record = completed_stock
    before = {path: sha256(path) for path in root.rglob("*") if path.is_file()}
    assert stock.verify_completion(spec, root) == record
    assert set(stock.evidence_paths(spec, root)) == set(before)
    assert before == {path: sha256(path) for path in before}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("defect", ["raw_summary", "calls", "c0", "noise", "rgb_missing", "spec"])
def test_stock_completion_refuses_restated_invalid_scientific_evidence(completed_stock: tuple, defect: str) -> None:
    spec, root, record = completed_stock
    if defect == "raw_summary":
        record["raw"]["terminal_difference_only"] = False
    elif defect == "calls":
        record["execution"]["paths"]["stock"]["forward_calls"] = 3
    elif defect == "c0":
        path = root / "bidirectional.pt"
        value = torch.load(path, weights_only=True)
        value[:, :, :1] += 1
        torch.save(value, path)
        record["output_files"][path.name] = sha256(path)
    elif defect == "noise":
        path = root / "noise.pt"
        value = torch.load(path, weights_only=True)
        value += 1
        torch.save(value, path)
        record["output_files"][path.name] = sha256(path)
    elif defect == "rgb_missing":
        (root / "comparison/poster.png").unlink()
    else:
        spec.write_text(spec.read_text() + "\n")
    (root / "result.json").write_text(json.dumps(record))
    if defect != "spec":
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["artifacts"] = spec_protocol.inventory(root)
        (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(
        (ValueError, FileNotFoundError), match=r"stock|specification|protocol|artifact|bytes|RGB|native|noise"
    ):
        stock.verify_completion(spec, root)


def effect_spec_fixture(saved_effect: tuple, monkeypatch: pytest.MonkeyPatch) -> tuple:
    root, record, _noise = saved_effect
    selected = record["protocol"]["checkpoints"]
    job, inputs, update = (root.parent / name for name in ("job.json", "inputs.json", "update.json"))
    for path in (job, inputs, update):
        path.write_text("{}")
    arguments = [
        "--job",
        str(job),
        "--inputs",
        str(inputs),
        "--update-check",
        str(update),
        "--checkpoints",
        *(row["path"] for row in selected.values()),
    ]
    if "step60" in selected:
        arguments.extend(["--update-job", str(job)])
    spec = write_spec(root.parent / "effect_spec.json", "adapter_effect_check", arguments)
    prepared = {
        "job": {"id": "fixed-original-job"},
        "pins": record["protocol"]["input_files"],
        "checkpoints": {name: Path(row["path"]) for name, row in selected.items()},
        "budget": record["protocol"]["resource_budget"],
        "schedule": record["protocol"]["schedule"],
        "seed": record["protocol"]["seed"],
        "cases": [{"source": row["source"]} for row in record["cases"]],
    }
    record["protocol"]["job"] = prepared["job"]
    (root / "protocol.json").write_text(json.dumps(record["protocol"]))
    record["output_files"]["protocol.json"] = sha256(root / "protocol.json")
    (root / "result.json").write_text(json.dumps(record))
    monkeypatch.setattr(effect, "prepare", lambda _args, **_kwargs: prepared)
    spec_protocol.publish(effect.read_spec(spec, root), "adapter_effect_check")
    return spec, root, record, prepared


def test_adapter_completion_binds_actual_saved_reader(saved_effect: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, root, record, _prepared = effect_spec_fixture(saved_effect, monkeypatch)
    assert effect.verify_completion(spec, root) == record
    assert root / "manifest.json" in effect.evidence_paths(spec, root)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("defect", ["decode", "selected", "summary"])
def test_adapter_completion_rejects_restated_spec_or_scientific_result(
    saved_effect: tuple, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    spec, root, record, prepared = effect_spec_fixture(saved_effect, monkeypatch)
    if defect == "decode":
        document = json.loads(spec.read_text())
        document["arguments"].append("--decode")
        spec.write_text(json.dumps(document))
        (root / "manifest.json").unlink()
        spec_protocol.publish(effect.read_spec(spec, root), "adapter_effect_check")
    elif defect == "selected":
        prepared["checkpoints"] = {**prepared["checkpoints"], "zero": root.parent / "foreign.safetensors"}
    else:
        record["cases"][0]["comparison"]["passed"] = False
        (root / "result.json").write_text(json.dumps(record))
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["artifacts"] = spec_protocol.inventory(root)
        (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="adapter-effect"):
        effect.verify_completion(spec, root)


def test_continuation_retired_self_launch_and_preserved_phase_inventory(tmp_path: Path) -> None:
    assert not hasattr(continuation, "supervised_run")
    assert not hasattr(continuation, "check_launch_record")
    spec = write_spec(
        tmp_path / "continuation.json",
        "continuation_check",
        ["--phase", "capture", "--protocol", str(tmp_path / "protocol.json"), "--sigma", "1.0"],
    )
    assert continuation.resource_phases(spec) == ["continuation-case"]
    assert continuation.runtime_profile(spec)["sampled_total_device_limit_bytes"] == 48800 * (1 << 20)
    assert continuation.dependency_outputs(spec) == []
    snapshot = tmp_path / "capture/continuation.json"
    spec = write_spec(
        tmp_path / "reference.json", "continuation_check", ["--phase", "reference", "--snapshot", str(snapshot)]
    )
    assert continuation.dependency_outputs(spec) == [snapshot.parent.resolve()]


def test_adapter_phase_inventory_retains_original_order(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"cases": [{"source": "actor/views/view01_cam57"}]}))
    spec = write_spec(
        tmp_path / "effect.json",
        "adapter_effect_check",
        [
            "--job",
            str(tmp_path / "job.json"),
            "--inputs",
            str(inputs),
            "--update-check",
            str(tmp_path / "update.json"),
            "--checkpoints",
            str(tmp_path / "zero.safetensors"),
            str(tmp_path / "step1.safetensors"),
            "--decode",
        ],
    )
    phases = effect.resource_phases(spec)
    assert phases[:4] == [
        "load:actor/views/view01_cam57:reference:base",
        "sample:actor/views/view01_cam57:reference:base",
        "load:actor/views/view01_cam57:reference:zero",
        "sample:actor/views/view01_cam57:reference:zero",
    ]
    assert len(phases) == 19
    assert phases[-1] == "decode:actor/views/view01_cam57"
    assert effect.dependency_outputs(spec) == [tmp_path.resolve()]


def test_stock_reference_requires_its_exact_queue_dependency(tmp_path: Path) -> None:
    spec = write_spec(
        tmp_path / "stock.json",
        "stock_parity",
        [
            "--first-image",
            str(tmp_path / "image.pt"),
            "--text-record",
            str(tmp_path / "text.json"),
            "--reference-run",
            str(tmp_path / "old_stock"),
        ],
    )
    assert stock.dependency_outputs(spec) == [(tmp_path / "old_stock").resolve()]


@pytest.fixture(params=["capture", "reference"])
def completed_continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> tuple:
    """Real tiny native CPU K/V; mocked full-weight lineage and CUDA journal."""
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    _old_grid, geometry, capture, guide, noise, context = continuation_inputs()
    grid = common.ClipGrid.build(
        17, EDGE, EDGE, FPS, geometry, device=torch.device("cpu"), dtype=torch.bfloat16, latent_channels=CHANNELS
    )
    capture, guide, noise, context = (value.bfloat16() for value in (capture, guide, noise, context))
    model = _model(prompt_adaln=True).bfloat16()
    predict = common.denoised_from_velocity_model(model)
    with torch.no_grad():
        recorder = continuation.CacheRecorder(
            predict,
            grid,
            geometry,
            capture_root,
            sigma=1.0,
            capture=capture,
            guide=guide,
            noise=noise,
            context=context,
            teacher_forcing=False,
        )
        causal.sample(
            recorder,
            context,
            grid,
            guide,
            capture[:, :4],
            transformer=model,
            geometry=geometry,
            schedule=[1.0, 0.0],
            seed=42,
            epsilon=noise,
            teacher_tokens=capture,
            teacher_forcing=False,
        )
        recorder.finish(2)
    original = tmp_path / "original_protocol.json"
    original.write_text("{}")
    fixed = {
        "input_files": {
            name: {"path": str(original), "sha256": sha256(original), "tensor_sha256": hashing.tensor_sha256(value)}
            for name, value in [
                ("capture", capture),
                ("guide", guide),
                ("noise", noise),
                ("first_image", capture[:, :4]),
                ("text", context),
            ]
        }
    }
    prepared = {
        "software": {},
        "fixed_inputs": fixed,
        "protocol": {"path": str(original), "sha256": sha256(original)},
        "original_result": {"path": str(original), "sha256": sha256(original)},
        "source_snapshot": None,
        "control": None,
        "changed_noise": None,
        "budget": {
            "wall_seconds_per_case": 1800,
            "observed_gpu_memory_limit_mib": 48800,
            "allocated_byte_threshold": None,
        },
        "estimate": {},
        "arguments": ["fixture"],
    }
    phase = request.param
    root = capture_root if phase == "capture" else tmp_path / "reference"
    if phase == "reference":
        root.mkdir()
        snapshot_path = capture_root / "continuation.json"
        snapshot_path.write_text(json.dumps({"snapshots": recorder.snapshots}))
        prepared.update(
            source_snapshot={"path": str(snapshot_path), "sha256": sha256(snapshot_path)},
            snapshot={"snapshots": recorder.snapshots},
        )
        with torch.no_grad():
            observations, blocks = continuation.reference_blocks(
                predict,
                model,
                grid,
                geometry,
                capture_root,
                recorder.snapshots,
                context=context,
                sigma=1.0,
                output_root=root,
            )
        arguments = ["--phase", "reference", "--snapshot", str(snapshot_path)]
    else:
        observations, blocks = [], []
        arguments = ["--phase", "capture", "--protocol", str(original), "--sigma", "1.0"]
    spec = write_spec(tmp_path / "spec.json", "continuation_check", arguments)
    budget = resources.default_budget()
    record = {
        "schema_version": 1,
        "kind": "onestep_avatar.continuation_check",
        "phase": phase,
        "sigma": 1.0,
        "history": "generated",
        "state": "complete",
        "error": None,
        **{key: value for key, value in prepared.items() if key not in ("snapshot", "arguments", "estimate")},
        "allocation_estimate": {},
        "effective_resource_budget": budget,
        "resources": {
            "schema_version": 1,
            "phase": "continuation-case",
            "rank": 0,
            "device": "cuda:0",
            "elapsed_s": 0.1,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 1,
            "state": "passed",
            "error": None,
            "budget_sha256": budget["sha256"],
        },
        "sampled_total_device_memory": [{"physical_gpu": "0", "memory_used_mib": 1}],
        "snapshots": recorder.snapshots if phase == "capture" else [],
        "layerwise": observations,
        "reference_blocks": blocks,
        "control_results": [],
        "call_counts": {
            "actual_model_calls": 16 if phase == "capture" else 10,
            "ordinary_model_calls": 16 if phase == "capture" else 8,
            "diagnostic_model_calls": 0 if phase == "capture" else 2,
        },
    }
    record["output_files"] = spec_protocol.inventory(root)
    (root / "continuation.json").write_text(json.dumps(record))

    def preflight(args: Namespace, **_kwargs: object) -> dict:
        args.sigma = 1.0
        return prepared

    monkeypatch.setattr(continuation, "prepare", preflight)
    monkeypatch.setattr(continuation, "_check_prepared", lambda _record: None)
    monkeypatch.setattr(continuation.evaluate, "verify_evaluation_conditions", lambda *_a, **_k: None)
    monkeypatch.setattr(
        backbone, "resolve", lambda *_a: SimpleNamespace(caps=SimpleNamespace(num_layers=2, num_heads=2, head_dim=4))
    )
    spec_protocol.publish(continuation.read_spec(spec, root), "continuation_check")
    return spec, root, record


def test_continuation_completion_recomputes_actual_native_observations(completed_continuation: tuple) -> None:
    spec, root, record = completed_continuation
    assert continuation.verify_completion(spec, root) == record
    assert root / "manifest.json" in continuation.evidence_paths(spec, root)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("defect", ["call_count", "resource", "raw", "layers"])
def test_continuation_completion_refuses_restated_partial_or_changed_observations(
    completed_continuation: tuple, defect: str
) -> None:
    spec, root, record = completed_continuation
    if defect == "call_count":
        record["call_counts"]["diagnostic_model_calls"] += 1
    elif defect == "resource":
        record["resources"]["elapsed_s"] = 1801
    elif defect == "layers":
        rows = record["snapshots"] if record["phase"] == "capture" else record["layerwise"]
        if record["phase"] == "capture":
            rows[0]["layers"].pop()
        else:
            rows.pop()
    elif record["phase"] == "capture":
        row = record["snapshots"][0]["layers"][0]
        path = root / row["file"]["path"]
        values = torch.load(path, weights_only=True)
        values["k"][0, 0, 0] += 1
        torch.save(values, path)
        row["file"] = continuation.file_record(root, path)
        record["output_files"][row["file"]["path"]] = sha256(path)
    else:
        record["layerwise"][0]["k"]["rms_delta"] += 0.1
    (root / "continuation.json").write_text(json.dumps(record))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["artifacts"] = spec_protocol.inventory(root)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"continuation|native resource"):
        continuation.verify_completion(spec, root)
