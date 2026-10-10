"""Exact short causal controls through shared APIs; see doc/experiments/short_causal_check.md."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.onestep_avatar import evaluate, hashing, infer
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import adapter_effect_check as effect
from scripts.onestep_avatar.experiments import causality, spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, common
from scripts.onestep_avatar.training import checkpoints, config, numerics, resources

ENTRY = "scripts/onestep_avatar/experiments/short_causal_check.py"
IMPORTED_OWNER_SHA256 = sha256(Path(__file__))
EXTRA_SOURCES = (
    *effect.EXTRA_SOURCES,
    ENTRY,
    "scripts/onestep_avatar/experiments/continuation_check.py",
    "scripts/onestep_avatar/experiments/causality.py",
)
PHASES = (
    "load",
    "repeat_a",
    "repeat_b",
    "product_original",
    "future_noise_3",
    "product_changed_3",
    "future_noise_5",
    "product_changed_5",
)
CALLS = {"denoise_calls": 3, "prime_calls": 0, "refresh_calls": 3, "model_calls": 6}
NAMES = (
    "repeat_a",
    "repeat_b",
    "product_original",
    "future_noise_3/original",
    "future_noise_3/changed",
    "product_changed_3",
    "future_noise_5/original",
    "future_noise_5/changed",
    "product_changed_5",
)


def scientific_args(argv: list[str]) -> argparse.Namespace:
    """Accept this phase's own inputs; mixed long-continuation flags stay invalid."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=("short_controls",), required=True)
    for name in ("job", "inputs", "update-check", "changed-noise-3", "changed-noise-5", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_id < 0:
        parser.error("local GPU must be nonnegative")
    return args


def dependency_outputs(args: argparse.Namespace) -> list[Path]:
    return [args.update_check.parent.resolve()]


def runtime_profile(args: argparse.Namespace) -> dict:
    """Bind the original job's exact budget without reading future native output."""
    job = json.loads(args.job.read_text())
    settings = config.parse_settings(job["arguments"])
    budget = resources.read_budget(settings.resource_budget)
    if budget is None:
        raise ValueError("short controls require the accepted E4 resource budget")
    return {"processes": 1, "generic_phase": False, "phases": list(PHASES), "resource_budget": budget}


def prepare(args: argparse.Namespace, *, completion: bool = False) -> dict:
    """Use the public calibrated-input gate before any model or output mutation."""
    selected = argparse.Namespace(**vars(args), update_job=None, subset=None, decode=False, fused_diagnostic=False)
    prepared = effect.prepare(selected, require_fresh_output=not completion)
    settings, cases = prepared["settings"], prepared["cases"]
    if (
        settings.mode != "causal"
        or settings.model != "2.5"
        or settings.variant != "dev"
        or settings.objective != "white"
        or settings.guide_mode != "d1"
        or settings.mode_settings != config.CausalSettings()
        or prepared["schedule"] != [0.725, 0.0]
        or prepared["seed"] != 42
        or len(cases) != 1
        or cases[0]["frames"] != 7
        or cases[0]["fps"] != 30
        or tuple(cases[0]["guide"].shape) != (1, 128, 7, 32, 32)
    ):
        raise ValueError("short controls require the fixed seven-frame causal D1 white dev inputs")
    budget = prepared["budget"]
    if budget["wall_seconds_per_phase"] != 1800 or budget["memory_limit_allocated_bytes"] != 48_000_000_000:
        raise ValueError("short controls require unchanged E4 resource limits")
    if (
        prepared["context"].dtype != torch.bfloat16
        or tuple(prepared["context"].shape) != (1, 1024, 4096)
        or not torch.isfinite(prepared["context"]).all()
    ):
        raise ValueError("short controls require the actual saved BF16 training context")
    changed = {}
    for boundary in (3, 5):
        path = getattr(args, f"changed_noise_{boundary}")
        before = sha256(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        validate_changed_noise(cases[0]["noise"], value, boundary, 32 * 32)
        if sha256(path) != before:
            raise ValueError("changed short-control noise changed while reading")
        prepared["pins"][str(path.resolve())] = before
        changed[boundary] = value
    prepared["changed_noise"] = changed
    prepared["software"] = software.capture("evaluation", "causal", extra_sources=EXTRA_SOURCES)
    if prepared["software"]["sources"][ENTRY] != IMPORTED_OWNER_SHA256:
        raise ValueError("short-control owner bytes changed since this process imported them")
    prepared["pins"][str(args.spec.resolve())] = args.spec_sha256
    effect.check_current(prepared)
    return prepared


def validate_changed_noise(original: torch.Tensor, changed: torch.Tensor, boundary: int, tokens: int) -> None:
    """Check the predeclared intervention; never regenerate a CUDA random stream."""
    if (
        not isinstance(original, torch.Tensor)
        or not isinstance(changed, torch.Tensor)
        or original.ndim != 3
        or original.shape[0] != 1
        or changed.shape != original.shape
        or changed.dtype != torch.bfloat16
        or original.dtype != torch.bfloat16
        or not torch.isfinite(original).all()
        or not torch.isfinite(changed).all()
        or boundary not in (3, 5)
        or tokens < 1
        or original.shape[1] != 7 * tokens
    ):
        raise ValueError("changed short-control noise must be finite BF16 with the original shape")
    cutoff = boundary * tokens
    if (
        not torch.equal(original[:, :cutoff], changed[:, :cutoff])
        or not torch.equal(-original[:, cutoff:], changed[:, cutoff:])
        or torch.equal(original[:, cutoff:], changed[:, cutoff:])
    ):
        raise ValueError("short-control noise must negate only the declared later frames")


def compare_outputs(outputs: dict[str, torch.Tensor]) -> dict:
    """Exact repeated and cross-path controls, with later changes reported separately."""
    if set(outputs) != set(NAMES):
        raise ValueError("short-control output inventory differs")
    baseline = outputs["repeat_a"]
    if not torch.equal(baseline, outputs["repeat_b"]) or not torch.equal(baseline, outputs["product_original"]):
        raise ValueError("short-control repeated or evaluation/product output differs")
    future = []
    for boundary in (3, 5):
        original = outputs[f"future_noise_{boundary}/original"]
        changed = outputs[f"future_noise_{boundary}/changed"]
        product = outputs[f"product_changed_{boundary}"]
        if (
            not torch.equal(original, baseline)
            or not torch.equal(changed, product)
            or not torch.equal(original[:, :, :boundary], changed[:, :, :boundary])
        ):
            raise ValueError("short-control future or evaluation/product equality differs")
        future.append(
            {
                "boundary": boundary,
                "earlier_bit_identical": True,
                "earlier_max_abs_delta": 0.0,
                "evaluation_product_bit_identical": True,
                "later_max_abs_delta": float(
                    (original[:, :, boundary:].float() - changed[:, :, boundary:].float()).abs().max()
                ),
            }
        )
    return {"same_input_repeat_bit_identical": True, "evaluation_product_bit_identical": True, "future_noise": future}


def check_rollout(
    output: torch.Tensor,
    record: dict,
    *,
    shape: tuple,
    image: torch.Tensor,
    guide: torch.Tensor,
    noise: torch.Tensor,
    context: torch.Tensor,
    product: bool,
) -> None:
    """Check consumed values and actual calls before accepting any saved array."""
    if (
        not isinstance(output, torch.Tensor)
        or output.dtype != torch.bfloat16
        or tuple(output.shape) != shape
        or not torch.isfinite(output).all()
        or not torch.equal(output[:, :, :1], image.cpu().to(output.dtype))
    ):
        raise ValueError("short-control output shape, precision, finiteness or supplied c0 differs")
    expected = {
        "call_counts": CALLS,
        "mode": "causal",
        "guide_mode": "d1",
        "schedule": [0.725, 0.0],
        "schema_version": 2,
        "frames": 7,
        "mode_settings": asdict(config.CausalSettings()),
        "seed": 42,
        "guide_sha256": hashing.tensor_sha256(guide),
        "noise_sha256": hashing.tensor_sha256(noise),
        "text_sha256": hashing.tensor_sha256(context),
    }
    expected.update(
        {"global_sigma_dtype": "float32", "capture_reference": None}
        if product
        else {"history_mode": "cache", "kv_source": "refresh"}
    )
    counts = record.get("call_counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(CALLS)
        or any(type(value) is not int for value in counts.values())
        or type(record.get("schema_version")) is not int
        or type(record.get("frames")) is not int
        or type(record.get("seed")) is not int
        or any(record.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("short-control consumed inputs, schedule or actual call record differs")


@torch.no_grad()
def run_controls(
    model: torch.nn.Module,
    context: torch.Tensor,
    grid: common.ClipGrid,
    tensors: dict,
    settings: config.RunSettings,
    phase: Callable[[str], contextlib.AbstractContextManager],
    save: Callable[[str, torch.Tensor, dict], None],
) -> tuple[dict, dict]:
    """Run nine fresh-cache paths; own comparisons, never cache or sampler arithmetic."""
    capture = common.with_clean_prefix(tensors["capture"], tensors["image"])
    guide, image, noise = (tensors[name] for name in ("guide", "image", "noise"))
    shape = tuple(grid.unpatchify_block(guide, grid.latent_frames).shape)
    c0 = grid.unpatchify_block(image, 1).cpu()
    ordinary = {
        "mode": "causal",
        "mode_settings": settings.mode_settings,
        "guide_mode": "d1",
        "schedule": [0.725, 0.0],
        "seed": 42,
        "history_mode": "cache",
        "kv_source": "refresh",
    }
    outputs, diagnostics = {}, {}

    def accept(name: str, output: torch.Tensor, record: dict, epsilon: torch.Tensor) -> None:
        check_rollout(
            output,
            record,
            shape=shape,
            image=c0,
            guide=guide,
            noise=epsilon,
            context=context,
            product=name.startswith("product_"),
        )
        if not name.startswith("product_") and record.get("capture_sha256") != hashing.tensor_sha256(capture):
            raise ValueError("short-control evaluation did not consume the independent supplied image")
        if record.get("c0_sha256") != hashing.tensor_sha256(image):
            raise ValueError("short-control assembled c0 differs from the supplied image")
        outputs[name] = output
        save(name, output, record)

    with evaluate.measure_calls(model) as calls:
        for name in ("repeat_a", "repeat_b"):
            with phase(name):
                output, record = evaluate.sample_case(model, context, grid, capture, guide, noise, **ordinary)
            accept(name, output, record, noise)
        with phase("product_original"):
            output, record = infer.generate(
                model,
                context,
                grid,
                guide,
                image,
                mode="causal",
                settings=settings.mode_settings,
                schedule=[0.725, 0.0],
                seed=42,
                epsilon=noise,
            )
        accept("product_original", output, record, noise)
        for boundary in (3, 5):
            changed = tensors[f"changed_{boundary}"]
            with phase(f"future_noise_{boundary}"):
                pair, diagnostic = causality.probe_future_noise(
                    model, context, grid, capture, guide, noise, changed, change_start_frame=boundary, **ordinary
                )
            for name, output, record, epsilon in zip(
                ("original", "changed"), pair, diagnostic["records"], (noise, changed), strict=True
            ):
                accept(f"future_noise_{boundary}/{name}", output, record, epsilon)
            diagnostics[str(boundary)] = (pair, diagnostic)
            with phase(f"product_changed_{boundary}"):
                output, record = infer.generate(
                    model,
                    context,
                    grid,
                    guide,
                    image,
                    mode="causal",
                    settings=settings.mode_settings,
                    schedule=[0.725, 0.0],
                    seed=42,
                    epsilon=changed,
                )
            accept(f"product_changed_{boundary}", output, record, changed)
    if calls["model_calls"] != 54:
        raise ValueError("short controls require 54 actual native forwards")
    summary = {**compare_outputs(outputs), "actual_model_calls": calls["model_calls"], "rollouts": len(outputs)}
    return summary, diagnostics


@contextlib.contextmanager
def measured_phase(root: Path, device: torch.device, budget: dict, records: list[dict], name: str) -> Iterator[None]:
    measured = resources.Phase(device, name, 0, budget)
    error = None
    try:
        measured.start()
        yield
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        if measured.started is not None:
            record = measured.finish(error)
            records.append(record)
            with (root / "resources_rank0.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if error is None and record["state"] != "passed":
                raise ValueError(record["error"])


def protocol(prepared: dict) -> dict:
    case = prepared["cases"][0]
    return {
        "kind": "onestep_avatar.short_causal_controls",
        "schema_version": 1,
        "input_files": prepared["pins"],
        "software": prepared["software"],
        "numerics": prepared["numerics"],
        "resource_budget": prepared["budget"],
        "source": case["source"],
        "fps": case["fps"],
        "shape": list(case["guide"].shape),
        "mode_settings": asdict(prepared["settings"].mode_settings),
        "schedule": prepared["schedule"],
        "seed": prepared["seed"],
        "adapter": str(prepared["checkpoints"]["step1"].resolve()),
        "application_method": adapters.UNMERGED,
        "tensor_inputs": {name: hashing.tensor_sha256(case[name]) for name in ("capture", "guide", "image", "noise")},
        "text_sha256": hashing.tensor_sha256(prepared["context"]),
        "changed_noise": {str(b): hashing.tensor_sha256(value) for b, value in prepared["changed_noise"].items()},
        "scope": "seven encoded frames; generated history; no cache eviction or quality acceptance",
    }


def execute(args: argparse.Namespace, prepared: dict) -> dict:
    """Load once under the queue and publish only complete exact controls."""
    from scripts.prune.core import preflight  # noqa: PLC0415 -- CPU preparation opens no native driver
    from scripts.prune.core.session import Session  # noqa: PLC0415 -- no native session during CPU preparation

    device = torch.device(f"cuda:{args.gpu_id}")
    actual = numerics.apply(expected=prepared["numerics"], environment_required=True)
    effect.check_current(prepared)
    preflight.check(
        prepared["settings"].model, gpu_id=args.gpu_id, transformer_path=prepared["specification"].paths.transformer()
    )
    case, model_spec = prepared["cases"][0], prepared["specification"]
    scales = model_spec.scale_factors
    grid = common.ClipGrid.build(
        7,
        case["guide"].shape[3] * scales.height,
        case["guide"].shape[4] * scales.width,
        case["fps"],
        model_spec,
        device=device,
        dtype=torch.bfloat16,
        latent_channels=case["guide"].shape[1],
    )
    context = prepared["context"].to(device)
    tensors = {name: grid.patchify(case[name].to(device, torch.bfloat16)) for name in ("capture", "guide", "image")}
    tensors["noise"] = case["noise"].to(device)
    tensors.update({f"changed_{boundary}": value.to(device) for boundary, value in prepared["changed_noise"].items()})
    path, contract = prepared["checkpoints"]["step1"], prepared["contracts"]["step1"]
    identity = prepared["pins"][str(path.resolve())]
    checkpoints.recheck_adapter(path, contract, identity)
    args.output.mkdir(parents=True, exist_ok=False)
    record = protocol(prepared)
    dataset.atomic_write(args.output / "protocol.json", lambda p: p.write_text(json.dumps(record, indent=2) + "\n"))
    measurements = []
    session = Session(model_spec, device, "onestep_avatar.short_causal_controls", context)
    provenance = {
        "source": case["source"],
        "fps": case["fps"],
        "software": prepared["software"],
        "input_files": prepared["pins"],
        "conditions": case["requested"],
        "adapter": str(path.resolve()),
        "adapter_sha256": identity,
        "application_method": adapters.UNMERGED,
        "numerics": actual,
    }

    def phase(name: str) -> contextlib.AbstractContextManager:
        return measured_phase(args.output, device, prepared["budget"], measurements, name)

    def save(name: str, output: torch.Tensor, row: dict) -> None:
        if not name.startswith("future_noise_"):
            evaluate.save_case(output, {**row, **provenance}, args.output / name)

    try:
        with contextlib.ExitStack() as lifecycle:
            with phase("load"):
                model = lifecycle.enter_context(
                    adapters.inference_transformer(
                        session, path, contract, method=adapters.UNMERGED, adapter_sha256=identity
                    )
                )
            summary, diagnostics = run_controls(model, context, grid, tensors, prepared["settings"], phase, save)
            del model
        for boundary, (pair, diagnostic) in diagnostics.items():
            causality.save_future_noise_probe(pair, diagnostic, provenance, args.output / f"future_noise_{boundary}")
        gc.collect()
        torch.cuda.empty_cache()
        effect.check_current(prepared)
        numerics.validate(numerics.capture(), required=True, expected=prepared["numerics"])
        resources.validate_records(measurements, 1, list(PHASES), prepared["budget"])
        result = {
            "schema_version": 1,
            "state": "passed",
            "protocol": record,
            "comparison": summary,
            "resource_measurements": measurements,
            "output_files": spec_protocol.inventory(args.output),
        }
        dataset.atomic_write(args.output / "result.json", lambda p: p.write_text(json.dumps(result, indent=2) + "\n"))
        return result
    except BaseException as failure:
        failed = {"state": "failed", "error": f"{type(failure).__name__}: {failure}"}
        dataset.atomic_write(args.output / "failure.json", lambda p: p.write_text(json.dumps(failed, indent=2) + "\n"))
        raise


def verify_saved(args: argparse.Namespace, manifest: dict) -> dict:
    """Recompute every exact control from bound CPU arrays, with no model session."""
    prepared = prepare(args, completion=True)
    result = json.loads((args.output / "result.json").read_text())
    paths = {f"{name}/{filename}" for name in NAMES for filename in ("generated.pt", "result.json")}
    paths.update(
        {
            "protocol.json",
            "resources_rank0.jsonl",
            "result.json",
            "future_noise_3/future_noise.json",
            "future_noise_5/future_noise.json",
        }
    )
    if set(manifest["artifacts"]) != paths:
        raise ValueError("short-control complete artifact paths differ")
    expected_inventory = {name: digest for name, digest in manifest["artifacts"].items() if name != "result.json"}
    if (
        type(result.get("schema_version")) is not int
        or result.get("schema_version") != 1
        or result.get("state") != "passed"
        or result.get("protocol") != protocol(prepared)
        or result.get("output_files") != expected_inventory
    ):
        raise ValueError("short-control result or full artifact binding differs")
    if json.loads((args.output / "protocol.json").read_text()) != protocol(prepared):
        raise ValueError("short-control saved protocol differs")
    case = prepared["cases"][0]
    scales = prepared["specification"].scale_factors
    grid = common.ClipGrid.build(
        7,
        case["guide"].shape[3] * scales.height,
        case["guide"].shape[4] * scales.width,
        case["fps"],
        prepared["specification"],
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        latent_channels=case["guide"].shape[1],
    )
    guide, image = (grid.patchify(case[name]) for name in ("guide", "image"))
    capture = common.with_clean_prefix(grid.patchify(case["capture"]), image)
    arrays = {}
    for name in NAMES:
        row = json.loads((args.output / name / "result.json").read_text())
        path = args.output / name / "generated.pt"
        if row.get("state") != "complete" or row.get("output") != {
            "path": str(path),
            "sha256": sha256(path),
            "shape": list(case["guide"].shape),
        }:
            raise ValueError("short-control saved raw path or file identity differs")
        output = torch.load(path, map_location="cpu", weights_only=True)
        boundary = int(name.split("_")[-1].split("/")[0]) if name.startswith("product_changed_") else None
        if name.startswith("future_noise_") and name.endswith("/changed"):
            boundary = int(name.split("/")[0].split("_")[-1])
        noise = case["noise"] if boundary is None else prepared["changed_noise"][boundary]
        check_rollout(
            output,
            row,
            shape=tuple(case["guide"].shape),
            image=case["image"],
            guide=guide,
            noise=noise,
            context=prepared["context"],
            product=name.startswith("product_"),
        )
        if not name.startswith("product_") and row.get("capture_sha256") != hashing.tensor_sha256(capture):
            raise ValueError("short-control saved evaluation capture/c0 binding differs")
        bindings = {
            "software": prepared["software"],
            "input_files": prepared["pins"],
            "source": case["source"],
            "fps": case["fps"],
            "c0_sha256": hashing.tensor_sha256(image),
            "conditions": case["requested"],
            "adapter": str(prepared["checkpoints"]["step1"].resolve()),
            "adapter_sha256": prepared["pins"][str(prepared["checkpoints"]["step1"].resolve())],
            "application_method": adapters.UNMERGED,
            "numerics": prepared["numerics"],
        }
        if any(row.get(key) != value for key, value in bindings.items()):
            raise ValueError("short-control raw conditioning or producer binding differs")
        arrays[name] = output
    measured = {**compare_outputs(arrays), "actual_model_calls": 54, "rollouts": 9}
    if result.get("comparison") != measured:
        raise ValueError("short-control numerical comparison differs from saved arrays")
    for boundary in (3, 5):
        diagnostic = json.loads((args.output / f"future_noise_{boundary}/future_noise.json").read_text())
        rows = [
            json.loads((args.output / f"future_noise_{boundary}/{name}/result.json").read_text())
            for name in ("original", "changed")
        ]
        expected = measured["future_noise"][(boundary - 3) // 2]
        if (
            diagnostic.get("records") != rows
            or diagnostic.get("change_start_encoded_frame") != boundary
            or diagnostic.get("earlier_output_bit_identical") is not True
            or diagnostic.get("earlier_output_max_abs_delta") != 0
            or diagnostic.get("later_output_max_abs_delta") != expected["later_max_abs_delta"]
        ):
            raise ValueError("short-control future diagnostic differs from saved arrays")
    records, _identities = resources.read_records(args.output, 1)
    resources.validate_records(records, 1, list(PHASES), prepared["budget"])
    if result.get("resource_measurements") != records:
        raise ValueError("short-control resource summary differs from phase journals")
    effect.check_current(prepared)
    return result
