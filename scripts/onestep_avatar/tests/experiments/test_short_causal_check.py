"""Exact short controls use real tiny BF16 LTX/PEFT paths; completion loads no model."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ltx_core.model.transformer.model import X0Model
from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import causality, spec_protocol
from scripts.onestep_avatar.experiments import continuation_check as continuation
from scripts.onestep_avatar.experiments import short_causal_check as short
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters
from scripts.onestep_avatar.tests.support.model import SCALE, _context, _geometry, _grid, _model
from scripts.onestep_avatar.training import config, numerics, resources


@pytest.fixture(scope="module")
def observed() -> dict:
    """Run every numerical path once; later tests mutate its saved CPU evidence."""
    grid = _grid(_geometry(8))
    generator = torch.Generator().manual_seed(26)
    tensors = {
        name: torch.randn(1, 28, 8, generator=generator).to(torch.bfloat16) for name in ("capture", "guide", "noise")
    }
    tensors["image"] = torch.full((1, 4, 8), 0.75, dtype=torch.bfloat16)
    for boundary in (3, 5):
        changed = tensors["noise"].clone()
        changed[:, boundary * 4 :] *= -1
        tensors[f"changed_{boundary}"] = changed
        short.validate_changed_noise(tensors["noise"], changed, boundary, 4)
    base = _model(prompt_adaln=True).to(torch.bfloat16)
    trained = adapters.attach(base, rank=2, alpha=2, target="attn", init_seed=9)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if ".lora_B." in name:
                parameter.fill_(0.125)
    context = _context().to(torch.bfloat16)
    settings = config.RunSettings("causal", Path("/unused/subset"), Path("/unused/output"), config.CausalSettings())
    rows, phases = {}, []

    @contextmanager
    def phase(name: str) -> Iterator[None]:
        phases.append(name)
        yield

    def save(name: str, output: torch.Tensor, record: dict) -> None:
        rows[name] = (output, record)

    summary, diagnostics = short.run_controls(X0Model(trained).eval(), context, grid, tensors, settings, phase, save)
    assert not torch.cuda.is_initialized()
    return {
        "grid": grid,
        "tensors": tensors,
        "context": context,
        "settings": settings,
        "rows": rows,
        "summary": summary,
        "diagnostics": diagnostics,
        "phases": phases,
    }


def test_nine_real_paths_preserve_independent_image_exact_controls_and_actual_calls(observed: dict) -> None:
    assert set(observed["rows"]) == set(short.NAMES)
    assert observed["phases"] == list(short.PHASES[1:])
    assert observed["summary"]["actual_model_calls"] == 54
    assert observed["summary"]["rollouts"] == 9
    assert observed["summary"]["same_input_repeat_bit_identical"]
    assert observed["summary"]["evaluation_product_bit_identical"]
    assert not torch.equal(observed["tensors"]["capture"][:, :4], observed["tensors"]["image"])
    image = observed["grid"].unpatchify_block(observed["tensors"]["image"], 1)
    for output, record in observed["rows"].values():
        assert output.dtype == torch.bfloat16
        assert not output.requires_grad
        assert torch.equal(output[:, :, :1], image)
        assert record["call_counts"] == short.CALLS
    for row in observed["summary"]["future_noise"]:
        assert row["earlier_bit_identical"]
        assert row["earlier_max_abs_delta"] == 0
        assert row["later_max_abs_delta"] > 0


@pytest.mark.parametrize("defect", ["earlier", "later", "unchanged", "dtype", "nan", "boundary", "tokens"])
def test_declared_noise_intervention_rejects_changed_protocol(observed: dict, defect: str) -> None:
    original = observed["tensors"]["noise"]
    changed = observed["tensors"]["changed_3"].clone()
    boundary, tokens = 3, 4
    if defect == "earlier":
        changed[:, 0] += 1
    elif defect == "later":
        changed[:, 12] += 1
    elif defect == "unchanged":
        changed = original.clone()
    elif defect == "dtype":
        changed = changed.float()
    elif defect == "nan":
        changed[:, 20] = float("nan")
    elif defect == "boundary":
        boundary = 4
    else:
        tokens = 3
    with pytest.raises(ValueError, match="short-control"):
        short.validate_changed_noise(original, changed, boundary, tokens)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def scientific_spec(tmp_path: Path) -> tuple[Path, Path]:
    budget = tmp_path / "budget.json"
    write_json(
        budget,
        {
            "wall_seconds_per_phase": 1800,
            "memory_limit_allocated_bytes": 48_000_000_000,
            "tolerance": {"relative_l2": 0.02, "absolute_rms": 1e-8},
        },
    )
    job = tmp_path / "job.json"
    write_json(
        job,
        {
            "arguments": [
                "--mode",
                "causal",
                "--subset",
                str(tmp_path / "subset.json"),
                "--output",
                str(tmp_path / "training"),
                "--resource-budget",
                str(budget),
            ]
        },
    )
    spec = tmp_path / "spec.json"
    write_json(
        spec,
        {
            "schema_version": 1,
            "protocol": "continuation_check",
            "arguments": [
                "--phase",
                "short_controls",
                "--job",
                str(job),
                "--inputs",
                str(tmp_path / "inputs.json"),
                "--update-check",
                str(tmp_path / "reference/result.json"),
                "--checkpoints",
                str(tmp_path / "zero.safetensors"),
                str(tmp_path / "step1.safetensors"),
                "--changed-noise-3",
                str(tmp_path / "changed3.pt"),
                "--changed-noise-5",
                str(tmp_path / "changed5.pt"),
            ],
        },
    )
    return spec, budget


def test_short_phase_keeps_strict_spec_dispatch_budget_and_dependencies(tmp_path: Path) -> None:
    spec, budget = scientific_spec(tmp_path)
    args = continuation.read_spec(spec, tmp_path / "output")
    assert args.phase == "short_controls"
    assert args.operation == "run"
    assert continuation.resource_phases(spec) == list(short.PHASES)
    assert continuation.runtime_profile(spec) == {
        "processes": 1,
        "generic_phase": False,
        "phases": list(short.PHASES),
        "resource_budget": resources.read_budget(budget),
    }
    assert continuation.dependency_outputs(spec) == [tmp_path / "reference"]
    saved = json.loads(spec.read_text())
    saved["arguments"] += ["--sigma", "1.0"]
    write_json(spec, saved)
    with pytest.raises(ValueError, match="scientific arguments"):
        continuation.read_spec(spec, tmp_path / "output")


def test_long_phase_retains_original_resource_meanings(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    write_json(
        spec,
        {
            "schema_version": 1,
            "protocol": "continuation_check",
            "arguments": ["--phase", "capture", "--protocol", str(tmp_path / "original.json"), "--sigma", "1.0"],
        },
    )
    assert continuation.resource_phases(spec) == ["continuation-case"]
    assert continuation.runtime_profile(spec) == {
        "generic_phase": False,
        "phases": ["continuation-case"],
        "processes": 1,
        "sampled_total_device_limit_bytes": 48800 * (1 << 20),
    }
    assert continuation.dependency_outputs(spec) == []


def test_long_phase_imports_exclude_short_only_scientific_owners(tmp_path: Path) -> None:
    """A fresh interpreter must not inherit this test module's explicit short imports."""
    specifications = []
    for phase, arguments in (
        ("capture", ["--protocol", str(tmp_path / "original.json"), "--sigma", "1.0"]),
        ("reference", ["--snapshot", str(tmp_path / "capture/continuation.json")]),
        (
            "control",
            [
                "--protocol",
                str(tmp_path / "original.json"),
                "--control-plan",
                str(tmp_path / "control_plan.json"),
                "--control-id",
                "declared-control",
            ],
        ),
    ):
        path = tmp_path / f"{phase}.json"
        write_json(
            path, {"schema_version": 1, "protocol": "continuation_check", "arguments": ["--phase", phase, *arguments]}
        )
        specifications.append(str(path))
    script = """
import json,sys,torch
from pathlib import Path
from scripts.onestep_avatar.experiments import continuation_check as check
excluded = ('scripts.onestep_avatar.experiments.short_causal_check',
            'scripts.onestep_avatar.experiments.adapter_effect_check', 'scripts.onestep_avatar.infer')
assert not any(name in sys.modules for name in excluded)
for name in json.loads(sys.argv[1]):
    spec=Path(name)
    check.read_spec(spec, spec.parent/'unused_output')
    assert check.resource_phases(spec)==['continuation-case']
    assert check.runtime_profile(spec)['sampled_total_device_limit_bytes']==48800*(1<<20)
    check.dependency_outputs(spec)
    assert not any(name in sys.modules for name in excluded)
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(specifications)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observed: dict) -> tuple:
    """Publish real tiny arrays plus explicitly synthetic resource records for CPU verifier tests."""
    spec, budget_path = scientific_spec(tmp_path)
    root = tmp_path / "output"
    args = continuation.read_spec(spec, root)
    root.mkdir()
    grid, values = observed["grid"], observed["tensors"]
    case = {
        name: grid.unpatchify_block(values[name], 1 if name == "image" else 7) for name in ("capture", "guide", "image")
    }
    case.update(noise=values["noise"], frames=7, fps=30, source="actor/view", requested={"fixed": True})
    checkpoint = tmp_path / "step1.safetensors"
    checkpoint.write_bytes(b"test source binding, not a model load")
    prepared = {
        "settings": observed["settings"],
        "cases": [case],
        "context": observed["context"],
        "specification": SimpleNamespace(scale_factors=SCALE),
        "schedule": [0.725, 0.0],
        "seed": 42,
        "budget": resources.read_budget(budget_path),
        "pins": {str(checkpoint): sha256(checkpoint), str(spec): sha256(spec)},
        "software": software.capture("evaluation", "causal", extra_sources=short.EXTRA_SOURCES),
        "checkpoints": {"step1": checkpoint},
        "numerics": numerics.capture(),
        "changed_noise": {b: values[f"changed_{b}"] for b in (3, 5)},
    }
    monkeypatch.setattr(short, "prepare", lambda _args, **_kwargs: prepared)
    binding = {
        "source": case["source"],
        "fps": case["fps"],
        "software": prepared["software"],
        "input_files": prepared["pins"],
        "conditions": case["requested"],
        "adapter": str(checkpoint),
        "adapter_sha256": sha256(checkpoint),
        "application_method": adapters.UNMERGED,
        "numerics": prepared["numerics"],
    }
    for name, (output, record) in observed["rows"].items():
        if not name.startswith("future_noise_"):
            evaluate.save_case(output, {**record, **binding}, root / name)
    for boundary, (outputs, diagnostic) in observed["diagnostics"].items():
        causality.save_future_noise_probe(outputs, diagnostic, binding, root / f"future_noise_{boundary}")
    write_json(root / "protocol.json", short.protocol(prepared))
    measurements = [
        {
            "schema_version": 1,
            "phase": phase,
            "rank": 0,
            "device": "cuda:0",
            "elapsed_s": 0.01,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
            "budget_sha256": prepared["budget"]["sha256"],
            "state": "passed",
            "error": None,
        }
        for phase in short.PHASES
    ]
    (root / "resources_rank0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in measurements))
    result = {
        "schema_version": 1,
        "state": "passed",
        "protocol": short.protocol(prepared),
        "comparison": observed["summary"],
        "resource_measurements": measurements,
        "output_files": spec_protocol.inventory(root),
    }
    write_json(root / "result.json", result)
    spec_protocol.publish(args, "continuation_check")

    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("saved-only completion attempted a numerical or model path")

    for owner, name in (
        (short, "execute"),
        (short, "run_controls"),
        (adapters, "inference_transformer"),
        (evaluate, "sample_case"),
    ):
        monkeypatch.setattr(owner, name, denied)
    return spec, root, prepared


def rebind_saved_inventory(spec: Path, root: Path) -> None:
    """Give tampered bytes a valid envelope so scientific rejection cannot rely on old hashes."""
    result = json.loads((root / "result.json").read_text())
    result["output_files"] = {
        name: digest for name, digest in spec_protocol.inventory(root).items() if name != "result.json"
    }
    write_json(root / "result.json", result)
    (root / "manifest.json").unlink()
    spec_protocol.publish(continuation.read_spec(spec, root), "continuation_check")


def test_saved_completion_recomputes_all_nine_arrays_without_model_or_sampler(saved: tuple) -> None:
    spec, root, _prepared = saved
    result = continuation.verify_completion(spec, root)
    assert result["state"] == "passed"
    assert result["comparison"]["actual_model_calls"] == 54
    assert len(continuation.evidence_paths(spec, root)) == 24
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "defect",
    [
        "repeat",
        "product",
        "c0",
        "future_earlier",
        "calls",
        "bool_count",
        "noise_hash",
        "context_hash",
        "conditions",
        "redirect",
        "phase",
        "protocol",
        "extra",
    ],
)
def test_saved_completion_rejects_rehashed_scientific_tampering(saved: tuple, defect: str) -> None:
    spec, root, _prepared = saved
    name = {
        "repeat": "repeat_b",
        "product": "product_original",
        "c0": "repeat_a",
        "future_earlier": "future_noise_3/changed",
    }.get(defect, "repeat_a")
    record_path = root / name / "result.json"
    row = json.loads(record_path.read_text())
    if defect in ("repeat", "product", "c0", "future_earlier"):
        path = root / name / "generated.pt"
        tensor = torch.load(path, weights_only=True)
        tensor[:, :, 0 if defect == "c0" else 1] += 1
        torch.save(tensor, path)
        row["output"]["sha256"] = sha256(path)
    elif defect == "calls":
        row["call_counts"]["model_calls"] = 5
    elif defect == "bool_count":
        row["call_counts"]["prime_calls"] = False
    elif defect == "noise_hash":
        row["noise_sha256"] = "0" * 64
    elif defect == "context_hash":
        row["text_sha256"] = "0" * 64
    elif defect == "conditions":
        row["conditions"] = {"fixed": False}
    elif defect == "redirect":
        row["output"]["path"] = str(root.parent / "elsewhere.pt")
    elif defect == "phase":
        journal = root / "resources_rank0.jsonl"
        journal.write_text("\n".join(journal.read_text().splitlines()[:-1]) + "\n")
    elif defect == "protocol":
        wrong = json.loads((root / "protocol.json").read_text())
        wrong["seed"] = 43
        write_json(root / "protocol.json", wrong)
    else:
        (root / "unrequested.pt").write_bytes(b"extra")
    write_json(record_path, row)
    rebind_saved_inventory(spec, root)
    with pytest.raises(ValueError, match=r"short-control|resource"):
        continuation.verify_completion(spec, root)


def test_saved_completion_rejects_changed_consumed_source_bytes(saved: tuple) -> None:
    spec, root, prepared = saved
    Path(prepared["checkpoints"]["step1"]).write_bytes(b"changed accepted matrices")
    with pytest.raises(ValueError, match="inputs changed"):
        continuation.verify_completion(spec, root)


def test_new_experiment_owners_are_excluded_from_ordinary_and_e4_software_profiles() -> None:
    from scripts.onestep_avatar.experiments import training_update_check  # noqa: PLC0415 -- explicit closure audit

    changed = {short.ENTRY, "scripts/onestep_avatar/experiments/continuation_check.py"}
    assert not changed.intersection(training_update_check.EXTRA_SOURCES)
    assert not changed.intersection(short.effect.EXTRA_SOURCES)


@pytest.mark.parametrize("defect", ["geometry", "context", "span", "budget"])
def test_native_preparation_rejects_weaker_short_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    spec, budget_path = scientific_spec(tmp_path)
    args = continuation.read_spec(spec, tmp_path / "output")
    settings = config.RunSettings("causal", tmp_path / "subset", tmp_path / "training", config.CausalSettings())
    case = {"frames": 7, "fps": 30, "guide": torch.zeros(1, 128, 7, 32, 32, dtype=torch.bfloat16)}
    prepared = {
        "settings": settings,
        "cases": [case],
        "schedule": [0.725, 0.0],
        "seed": 42,
        "budget": resources.read_budget(budget_path),
        "context": torch.zeros(1, 1024, 4096, dtype=torch.bfloat16),
    }
    if defect == "geometry":
        case["guide"] = torch.zeros(1, 8, 7, 2, 2, dtype=torch.bfloat16)
    elif defect == "context":
        prepared["context"] = torch.zeros(1, 3, 16, dtype=torch.bfloat16)
    elif defect == "span":
        settings.mode_settings = config.CausalSettings(span_latent_frames=7)
    else:
        prepared["budget"] = copy.deepcopy(prepared["budget"])
        prepared["budget"]["memory_limit_allocated_bytes"] -= 1
    monkeypatch.setattr(short.effect, "prepare", lambda *_args, **_kwargs: prepared)
    with pytest.raises(ValueError, match="short controls"):
        short.prepare(args)
    assert not args.output.exists()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", ["model", "wall", "memory"])
def test_failed_phase_retains_shared_measurement_before_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _spec, budget_path = scientific_spec(tmp_path)
    budget = resources.read_budget(budget_path)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda _device: 48_000_000_001 if failure == "memory" else 100
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 48_000_000_002)
    ticks = iter((0.0, 1801.0 if failure == "wall" else 1.0))
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(ticks))
    records = []

    def attempt() -> None:
        with short.measured_phase(tmp_path, torch.device("cuda:0"), budget, records, "repeat_a"):
            if failure == "model":
                raise RuntimeError("declared failure")

    with pytest.raises((ValueError, RuntimeError), match=r"declared failure|resource phase"):
        attempt()
    rows = [json.loads(line) for line in (tmp_path / "resources_rank0.jsonl").read_text().splitlines()]
    assert rows == records
    assert rows[0]["state"] == "failed"
    assert rows[0]["budget_sha256"] == budget["sha256"]
    assert not torch.cuda.is_initialized()
