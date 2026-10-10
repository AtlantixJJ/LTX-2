"""Queue ordinary owners reject incomplete or changed saved scientific evidence."""

from __future__ import annotations

import json
import statistics
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import bench, evaluate, hashing, infer, media, prepare_inputs, previews
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.model import adapters
from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B
from scripts.onestep_avatar.tests.test_evaluation_completion import completed as native_completed
from scripts.onestep_avatar.tests.test_infer import product_files as native_product_files
from scripts.onestep_avatar.tests.test_prepare_preview import preview_case as prepared_preview_case
from scripts.onestep_avatar.tests.test_previews import _completed, _fixed_inputs
from scripts.onestep_avatar.training import checkpoints, config, engine
from scripts.prune.core import preflight
from scripts.prune.core import session as sessions
from scripts.prune.data import prompt_cache as prompts

completed = native_completed
product_files = native_product_files
preview_case = prepared_preview_case


def _blocked_native(monkeypatch: pytest.MonkeyPatch) -> None:

    def forbidden(*_args, **_kwargs) -> None:
        pytest.fail("saved completion opened a native model boundary")

    monkeypatch.setattr(preflight, "check", forbidden)
    monkeypatch.setattr(sessions, "Session", forbidden)
    monkeypatch.setattr(prompts, "get_or_build", forbidden)
    monkeypatch.setattr(media, "decode", forbidden)


@pytest.mark.parametrize(
    ("owner", "arguments"),
    [
        (previews, ["--preview-job", "/not-read", "--output", "/unused"]),
        (
            infer,
            [
                "--mode",
                "bidirectional",
                "--guide",
                "/not-read",
                "--first-image",
                "/not-read",
                "--output",
                "/unused",
                "--schedule",
                "0.725",
                "0",
            ],
        ),
        (
            bench,
            ["--mode", "bidirectional", "--subset", "/not-read", "--output", "/unused", "--schedule", "0.725", "0"],
        ),
        (bench, ["--operation-timing", "--mode", "causal", "--output", "/unused"]),
    ],
)
def test_ordinary_parsers_read_no_scientific_files(
    owner: ModuleType, arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: pytest.fail("parse read a file"))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("parse loaded scientific tensors"))
    _blocked_native(monkeypatch)
    args = owner.parse_args(arguments)
    assert args.output == Path("/unused")
    assert args.gpu_id == 0


def test_preview_queue_identity_survives_state_writes_and_refuses_input_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, settings = _fixed_inputs(tmp_path)
    fixed = engine.read_preview_inputs(path, settings)
    job_path = engine.enqueue_preview(_completed(tmp_path), fixed, settings.output)
    job = json.loads(job_path.read_text())
    arguments = ["--preview-job", str(job_path), "--output", job["output"]]
    _blocked_native(monkeypatch)
    identity, before = previews.queue_identity(arguments), hashing.sha256(job_path)
    job.update(state="running", pid=123, updated_at=123, results=[{"path": "new-state"}])
    job_path.write_text(json.dumps(job))
    assert hashing.sha256(job_path) != before
    assert previews.queue_identity(arguments) == identity
    with pytest.raises(ValueError, match="output differs"):
        previews.queue_identity(["--preview-job", str(job_path), "--output", str(tmp_path / "elsewhere")])
    job["fixed_inputs"]["evaluation_arguments"].append("--include-base")
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="fixed preview record changed"):
        previews.queue_identity(arguments)


@pytest.fixture
def preview_complete(completed: tuple, preview_case: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple:
    _, _, _, _, native_calls = completed
    setup, settings, _ = preview_case
    args = setup()
    reference_path = args.references
    reference = json.loads(reference_path.read_text())
    reference["producer"].update(decoder_settings=media.native_decoder_settings(), decode_seed=42)
    reference_path.write_text(json.dumps(reference))

    fixed = prepare_inputs.prepare_preview(args)
    settings.lora_rank = settings.lora_alpha = 2
    specification = evaluate.backbone.resolve("2.5", "dev")
    settings.base_identity = evaluate.backbone.identity(specification.paths.transformer(), "dev", "2.5", full_hash=True)
    settings.base_identity["base_transformer_file"] = Path(specification.paths.transformer()).name
    membership = json.loads(settings.subset.read_text())
    plan = config.build_frame_plan(settings, membership, specification.scale_factors)
    contract = checkpoints.make_contract(settings, membership, plan, step=100)
    contract["adapter"]["tensor_shapes"] = {A: [2, 4], B: [4, 2]}
    checkpoint = tmp_path / "preview_adapter.safetensors"
    save_file(
        {A: torch.ones(2, 4), B: torch.zeros(4, 2)},
        checkpoint,
        metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)},
    )
    checkpoint.with_suffix(".complete.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path": str(checkpoint),
                "sha256": hashing.sha256(checkpoint),
                "state": "complete",
                "step": 100,
            }
        )
    )
    job_path = engine.enqueue_preview(checkpoint, fixed, settings.output)
    job = json.loads(job_path.read_text())
    arguments = ["--preview-job", str(job_path), "--output", job["output"]]
    old_session = sessions.Session

    def cpu_session(*a, **kw) -> object:
        value = old_session(*a, **kw)
        value.decoder = lambda: nullcontext(None)
        return value

    monkeypatch.setattr(sessions, "Session", cpu_session)
    monkeypatch.setattr(media, "decode", lambda *_a, **_k: torch.zeros(49, 3, 16, 16, dtype=torch.uint8))

    def save_media(_pixels: torch.Tensor, record: dict, output: Path) -> dict:
        output.mkdir(parents=True)
        outputs = {}
        for role, name in (("video", "comparison.mp4"), ("poster", "poster.png")):
            path = output / name
            path.write_bytes(b"controlled saved media " + role.encode())
            outputs[role] = {"path": str(path), "sha256": hashing.sha256(path)}
        result = {**record, "outputs": outputs}
        (output / "rendering.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(media, "save_render", save_media)
    assert previews.main(arguments) == 0
    assert "transformer" in native_calls
    root = Path(job["output"])
    _blocked_native(monkeypatch)
    assert previews.verify_completion(arguments, root)
    return arguments, root, job_path


def test_preview_completion_binds_rendered_and_raw_evidence(preview_complete: tuple) -> None:
    arguments, root, job_path = preview_complete
    saved = previews.verify_completion(arguments, root)
    assert saved["state"] == "complete"
    assert {"preview_completion.json", "generated.pt", "comparison.mp4", "poster.png"} <= {
        path.name for path in previews.evidence_paths(arguments, root)
    }
    identity = previews.queue_identity(arguments)
    job = json.loads(job_path.read_text())
    job["updated_at"] = 9999
    job_path.write_text(json.dumps(job))
    assert previews.queue_identity(arguments) == identity
    assert previews.verify_completion(arguments, root)
    marker = root / "preview_completion.json"
    before = hashing.sha256(marker)
    assert previews.main(arguments) == 0
    assert hashing.sha256(marker) == before
    marker.unlink()
    assert previews.main(arguments) == 0
    assert hashing.sha256(marker) == before


@pytest.mark.parametrize(
    "defect", ["missing_media", "raw_shape", "raw_precision", "raw_c0", "frame_mapping", "decoder"]
)
def test_preview_completion_rejects_rehashed_tampering(preview_complete: tuple, defect: str) -> None:
    arguments, root, job_path = preview_complete
    job = json.loads(job_path.read_text())
    record_path = Path(job["results"][0]["path"])
    record = json.loads(record_path.read_text())
    rendering_path = Path(job["renderings"][0]["path"])
    rendering = json.loads(rendering_path.read_text())
    if defect == "missing_media":
        Path(rendering["outputs"]["poster"]["path"]).unlink()
    elif defect.startswith("raw_"):
        path = Path(record["output"]["path"])
        value = torch.load(path, weights_only=True)
        if defect == "raw_shape":
            value = value[:, :, :-1]
        elif defect == "raw_precision":
            value = value.float()
        else:
            value[:, :, :1] += 1
        torch.save(value, path)
        record["output"].update(sha256=hashing.sha256(path), shape=list(value.shape))
        record_path.write_text(json.dumps(record))
        job["results"][0]["sha256"] = hashing.sha256(record_path)
        rendering["common_settings"]["result_records"] = job["results"]
    elif defect == "frame_mapping":
        rendering["source_frames"][0] += 1
    else:
        rendering["common_settings"]["decoder_records"][0]["decode_key"] = "a" * 64
    rendering_path.write_text(json.dumps(rendering))
    job["renderings"][0]["sha256"] = hashing.sha256(rendering_path)
    job_path.write_text(json.dumps(job))
    with pytest.raises((ValueError, FileNotFoundError)):
        previews.verify_completion(arguments, root)


@pytest.fixture(params=["bidirectional", "causal"])
def product_complete(product_files: tuple, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> tuple:

    source_arguments, _, _ = product_files
    arguments = list(source_arguments)
    arguments[arguments.index("--mode") + 1] = request.param
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=128,
        out_channels=128,
        num_layers=2,
        cross_attention_dim=8,
        caption_projection=torch.nn.Linear(4, 8),
    )
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    model = X0Model(model.to(dtype=torch.bfloat16).eval())
    monkeypatch.setattr(preflight, "check", lambda *_a, **_k: None)
    monkeypatch.setattr(prompts, "get_or_build", lambda *_a, **_k: torch.ones(1, 3, 4, dtype=torch.bfloat16))
    monkeypatch.setattr(sessions, "Session", lambda *_a, **_k: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(infer, "torch", SimpleNamespace(**{**vars(torch), "device": lambda *_a: torch.device("cpu")}))

    monkeypatch.setattr(adapters, "inference_transformer", lambda *_a, **_k: nullcontext(model))
    assert infer.main(arguments) == 0
    root = infer.parse_args(arguments).output
    _blocked_native(monkeypatch)
    return arguments, root


def test_product_completion_checks_actual_saved_inputs_model_free(product_complete: tuple) -> None:
    arguments, root = product_complete
    assert infer.verify_completion(arguments, root)["capture_reference"] is None
    assert {path.name for path in infer.evidence_paths(arguments, root)} == {
        "result.json",
        "generated.pt",
        "generation_inputs.pt",
    }


@pytest.mark.parametrize("defect", ["noise", "text", "precision", "nonfinite", "c0", "calls", "missing"])
def test_product_completion_rejects_changed_evidence(product_complete: tuple, defect: str) -> None:
    arguments, root = product_complete
    inputs_path, result_path = root / "generation_inputs.pt", root / "result.json"
    record = json.loads(result_path.read_text())
    if defect in ("noise", "text", "precision", "nonfinite"):
        inputs = torch.load(inputs_path, weights_only=True)
        role = "text" if defect == "text" else "noise"
        inputs[role] = (
            inputs[role] + 1
            if defect in ("noise", "text")
            else (inputs[role].float() if defect == "precision" else torch.full_like(inputs[role], float("nan")))
        )
        torch.save(inputs, inputs_path)
        record["generation_inputs"]["sha256"] = hashing.sha256(inputs_path)
    elif defect == "c0":
        path = root / "generated.pt"
        value = torch.load(path, weights_only=True)
        value[:, :, :1] += 1
        torch.save(value, path)
        record["output"]["sha256"] = hashing.sha256(path)
    elif defect == "calls":
        record["call_counts"]["model_calls"] += 1
    else:
        inputs_path.unlink()
    result_path.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        infer.verify_completion(arguments, root)


@pytest.fixture(params=["bidirectional", "causal"])
def benchmark_complete(completed: tuple, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> tuple:
    _, arguments, settings, _, _ = completed
    arguments = [*arguments, "--repetitions", "2", "--warmup", "1"]
    arguments[arguments.index("--mode") + 1] = request.param
    assert bench.main(arguments) == 0
    _blocked_native(monkeypatch)
    assert bench.verify_completion(arguments, settings.output)
    return arguments, settings.output


def test_benchmark_completion_checks_all_cases_model_free(benchmark_complete: tuple) -> None:
    arguments, root = benchmark_complete
    records = bench.verify_completion(arguments, root)["records"]
    assert len(records) == 2
    assert len(bench.evidence_paths(arguments, root)) == 7


@pytest.mark.parametrize(
    "defect", ["reps", "warmup", "elapsed", "summary", "coverage", "calls", "memory", "output", "missing"]
)
def test_benchmark_completion_rejects_measurement_tampering(benchmark_complete: tuple, defect: str) -> None:
    arguments, root = benchmark_complete
    path = sorted(root.glob("case_*/variant_*/result.json"))[0]
    record = json.loads(path.read_text())
    timing, row = record["benchmark"], record["benchmark"]["repetitions"][0]
    if defect == "reps":
        timing["repetitions"].pop()
    elif defect == "warmup":
        timing["warmup"] = 0
    elif defect == "elapsed":
        row["elapsed_s"] = float("nan")
    elif defect == "summary":
        timing["elapsed_s"]["median"] += 1
    elif defect == "coverage":
        row["covered_rgb_frames"] += 1
    elif defect == "calls":
        row["call_counts"]["model_calls"] += 1
    elif defect == "memory":
        row["memory_baseline_bytes"] = {"allocated": 0, "reserved": 0}
    elif defect == "output":
        row["output_sha256"] = "a" * 64
    else:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            bench.verify_completion(arguments, root)
        return
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"(?i)benchmark|operation|evaluation|software"):
        bench.verify_completion(arguments, root)


@pytest.fixture
def operation_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:

    transformer = tmp_path / "base.safetensors"
    transformer.write_bytes(b"operation fixture")
    model = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32), paths=SimpleNamespace(transformer=lambda: transformer)
    )
    monkeypatch.setattr(bench.model_registry, "resolve", lambda *_a, **_k: model)
    root = tmp_path / "output"
    root.mkdir()
    arguments = [
        "--operation-timing",
        "--mode",
        "causal",
        "--output",
        str(root),
        "--reps",
        "2",
        "--warmup",
        "1",
        "--latent-frames",
        "7",
        "--context-latent-frames",
        "0",
        "2",
    ]
    args = bench.parse_args(arguments)
    record = {
        "schema_version": 1,
        "kind": "onestep_avatar.operation_timing",
        "state": "complete",
        "settings": {
            "mode": "causal",
            "model": "2.5",
            "gpu_id": 0,
            "sigma0": 0.725,
            "reps": 2,
            "warmup": 1,
            "seed": 0,
            "context_latent_frames": [0, 2],
            "latent_frames": 7,
        },
        "software": software.capture("evaluation", "causal", extra_sources=("scripts/onestep_avatar/bench.py",)),
        "producer_source_sha256": hashing.sha256(Path(bench.__file__)),
        "transformer": {"path": str(transformer.resolve()), "sha256": hashing.sha256(transformer)},
        "geometry": {"edge": 1024, "latent_frames": 7, "block_latent_frames": 2},
        "measurements": {},
        "cache_details": {},
    }
    for depth, capacity, cached in ((0, 4096, 1024), (2, 6144, 3072)):
        record["measurements"][str(depth)] = {"denoise_s": [0.2, 0.4], "refresh_s": [0.1, 0.3]}
        record["cache_details"][str(depth)] = {"layers": 2, "inner_dim": 8, "capacity": capacity, "element_bytes": 2}
        for name, times in (("denoise", [0.2, 0.4]), ("refresh", [0.1, 0.3])):
            record[f"causal_{name}_ctx{depth}"] = {
                "query_tokens": 2048,
                "median_s": statistics.median(times),
                "mean_s": statistics.mean(times),
                "stdev_s": statistics.stdev(times),
                "reps": 2,
            }
        record[f"causal_total_ctx{depth}"] = {
            "per_chunk_s": statistics.median([0.2, 0.4]) + statistics.median([0.1, 0.3]),
            "cached_key_tokens": cached,
            "cache_gib": 2 * 2 * capacity * 8 * 2 / 2**30,
        }
    args.out.write_text(json.dumps(record))
    _blocked_native(monkeypatch)
    assert bench.verify_completion(arguments, root)
    return arguments, root, args.out


def test_operation_output_normalization_and_saved_completion(operation_complete: tuple, tmp_path: Path) -> None:
    arguments, root, path = operation_complete
    args = bench.parse_args(["--operation-timing", "--mode", "causal", "--out", str(tmp_path / "legacy.json")])
    assert args.output == tmp_path
    assert args.completion == {"records": [str((tmp_path / "legacy.json").resolve())]}
    assert bench.evidence_paths(arguments, root) == [path]


@pytest.mark.parametrize("defect", ["depth", "reps", "warmup", "times", "sum", "cache", "source"])
def test_operation_completion_rejects_incomplete_or_changed_protocol(operation_complete: tuple, defect: str) -> None:
    arguments, root, path = operation_complete
    record = json.loads(path.read_text())
    if defect == "depth":
        record["measurements"].pop("2")
    elif defect == "reps":
        record["measurements"]["0"]["denoise_s"].pop()
    elif defect == "warmup":
        record["settings"]["warmup"] += 1
    elif defect == "times":
        record["measurements"]["0"]["refresh_s"][0] = float("inf")
    elif defect == "sum":
        record["causal_total_ctx0"]["per_chunk_s"] += 1
    elif defect == "cache":
        record["cache_details"]["0"]["capacity"] += 1
    else:
        record["producer_source_sha256"] = "a" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"(?i)benchmark|operation|evaluation|software"):
        bench.verify_completion(arguments, root)
