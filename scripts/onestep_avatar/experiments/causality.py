"""Preserve study protocols through public shared owners; see doc/experiments/causality.md."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from scripts.onestep_avatar import evaluate, hashing
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone, causal, common
from scripts.onestep_avatar.model.sampling import validate_schedule
from scripts.onestep_avatar.training import checkpoints

if TYPE_CHECKING:
    from scripts.prune.core.session import Session


def probe_future_noise(
    transformer: torch.nn.Module,
    context: torch.Tensor,
    grid: common.ClipGrid,
    capture: torch.Tensor,
    guide: torch.Tensor | None,
    noise: torch.Tensor,
    changed_noise: torch.Tensor,
    *,
    change_start_frame: int,
    **settings,
) -> tuple[list[torch.Tensor], dict]:
    """Change only later saved noise and measure the earlier generated encoding."""
    if not 0 < change_start_frame < grid.latent_frames:
        raise ValueError("future-noise boundary must leave nonempty earlier and later regions")
    if settings.get("mode") == "causal":
        mode = settings["mode_settings"]
        plan = causal.CausalGeometry(
            grid.tools.scale_factors, mode.block_latent_frames, mode.context_latent_frames
        ).plan(grid.latent_frames)
        if change_start_frame not in [end for _, end in plan[:-1]]:
            raise ValueError("causal future-noise boundary must separate completed blocks")
    boundary = change_start_frame * grid.tokens_per_latent_frame
    source = common.source_for(capture, guide, settings["guide_mode"])
    if noise.shape != source.shape or not torch.isfinite(noise).all() or not torch.isfinite(changed_noise).all():
        raise ValueError("future-noise tensors must be finite and match the complete source")
    if noise.shape != changed_noise.shape or noise.dtype != changed_noise.dtype or noise.device != changed_noise.device:
        raise ValueError("future-noise tensors must have identical shape, dtype and device")
    if hashing.tensor_sha256(noise[:, :boundary]) != hashing.tensor_sha256(changed_noise[:, :boundary]):
        raise ValueError("future-noise diagnostic changed earlier noise")
    if torch.equal(noise[:, boundary:], changed_noise[:, boundary:]):
        raise ValueError("future-noise diagnostic requires changed later noise")
    outputs, records = [], []
    for epsilon in (noise, changed_noise):
        output, record = evaluate.sample_case(transformer, context, grid, capture, guide, epsilon, **settings)
        outputs.append(output)
        records.append(record)
    left, right = (output[:, :, :change_start_frame] for output in outputs)
    return outputs, {
        "change_start_encoded_frame": change_start_frame,
        "earlier_output_bit_identical": torch.equal(left, right),
        "earlier_output_max_abs_delta": float((left.float() - right.float()).abs().max()),
        "later_output_max_abs_delta": float(
            (outputs[0][:, :, change_start_frame:].float() - outputs[1][:, :, change_start_frame:].float()).abs().max()
        ),
        "records": records,
        "scope": "fixed model inputs; only noise at/after the encoded boundary differs",
    }


def save_future_noise_probe(outputs: list[torch.Tensor], diagnostic: dict, provenance: dict, destination: Path) -> dict:
    """Publish both raw results before a diagnostic, preserving their distinct noise hashes."""
    if len(outputs) != 2 or len(diagnostic.get("records", [])) != 2:
        raise ValueError("future-noise publication requires two outputs and records")
    saved = [
        evaluate.save_case(encoded, {**raw, **provenance}, destination / name)
        for encoded, raw, name in zip(outputs, diagnostic["records"], ("original", "changed"), strict=True)
    ]
    completed = {**diagnostic, "records": saved}
    atomic_write(
        destination / "future_noise.json",
        lambda temporary: temporary.write_text(json.dumps(completed, indent=2) + "\n"),
    )
    return completed


@torch.no_grad()
def causality_probe(
    transformer: torch.nn.Module,
    session: Session,
    capture: torch.Tensor,
    guide: torch.Tensor,
    fps: float,
    sigma: float,
    *,
    output_observer: Callable[[list[torch.Tensor]], None] | None = None,
    input_observer: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> dict:
    """Keep global noise mapping and execute exactly eight generated-history blocks."""
    from scripts.prune.core.session import DTYPE  # noqa: PLC0415

    levels = list(validate_schedule([sigma, 0.0]))
    if capture.shape != guide.shape or capture.ndim != 4 or capture.shape[1] < 17:
        raise ValueError("causality diagnostic requires matching masters and eight complete blocks")
    if any(not torch.isfinite(value).all() for value in (capture, guide)):
        raise ValueError("causality diagnostic masters must be finite")
    geometry = causal.deployed_geometry(session.model.scale_factors)
    _, frames, height, width = capture.shape
    grid = common.ClipGrid.build(
        frames,
        height * geometry.scale_factors.height,
        width * geometry.scale_factors.width,
        fps,
        geometry,
        device=session.device,
        dtype=DTYPE,
        latent_channels=session.model.caps.latent_channels,
    )
    plan = geometry.plan(frames)[:8]
    target = grid.patchify(capture.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    source = grid.patchify(guide.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    original = common.epsilon_block(target, 42)
    alternate = common.epsilon_block(target, 99)
    first = plan[3][1]
    boundary = first * grid.tokens_per_latent_frame
    mixed = torch.cat((original[:, :boundary], alternate[:, boundary:]), dim=1)
    if input_observer is not None:
        input_observer(
            {
                name: value.detach().cpu()
                for name, value in {
                    "capture": target,
                    "guide": source,
                    "text": session.context,
                    "original_noise": original,
                    "mixed_noise": mixed,
                }.items()
            }
        )
    outputs, records = [], []
    for epsilon in (original, mixed):
        with evaluate.measure_calls(transformer) as measured:
            tokens, counts = causal.sample(
                common.denoised_from_x0_model(transformer),
                session.context,
                grid,
                source,
                target[:, : grid.tokens_per_latent_frame],
                transformer=transformer,
                geometry=geometry,
                schedule=levels,
                seed=42,
                epsilon=epsilon,
                blocks=plan,
                teacher_forcing=False,
                history_mode="cache",
                kv_source="refresh",
            )
        covered = plan[-1][1]
        outputs.append(
            grid.unpatchify_block(tokens[:, : covered * grid.tokens_per_latent_frame], covered).float().cpu()
        )
        if not torch.isfinite(outputs[-1]).all():
            raise ValueError("causality diagnostic produced nonfinite output")
        records.append({"noise_sha256": hashing.tensor_sha256(epsilon), "call_counts": {**counts, **measured}})
    if output_observer is not None:
        output_observer(outputs)
    left, right = outputs
    later_delta = float((left[:, :, first:] - right[:, :, first:]).abs().max())
    return {
        "shared_noise_blocks": [0, 1, 2, 3],
        "changed_noise_blocks": [4, 5, 6, 7],
        "latent_frames_compared_equal": [0, first],
        "earlier_blocks_bit_identical": bool(torch.equal(left[:, :, :first], right[:, :, :first])),
        "later_blocks_max_abs_diff": later_delta,
        "later_blocks_changed": later_delta > 0,
        "records": records,
        "capture_sha256": hashing.tensor_sha256(target),
        "guide_sha256": hashing.tensor_sha256(source),
        "text_sha256": hashing.tensor_sha256(session.context),
    }


def evaluate_causality(
    checkpoint: Path,
    view: Path,
    output: Path,
    *,
    gpu_id: int,
    sigma: float,
    output_tensors_path: Path | None = None,
    input_tensors_path: Path | None = None,
) -> dict:
    """Check scientific inputs before weights, then publish the historical comparison."""
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: PLC0415
    from scripts.prune.core.session import open_session  # noqa: PLC0415

    validate_schedule([sigma, 0.0])
    if output.exists():
        raise ValueError("causality diagnostic requires a fresh output")
    dev = backbone.transformer_path("2.5", "dev")
    capture_path = view / dataset.capture_bundle_name("white")
    guide_path = view / dataset.guide_bundle_name("white")
    paths = [dev, checkpoint, capture_path, guide_path]
    if any(not path.is_file() for path in paths):
        raise ValueError("causality diagnostic input file is missing")
    capture, fps = dataset.load_training_master(capture_path)
    guide, guide_fps = dataset.load_training_master(guide_path)
    if capture.shape != guide.shape or fps != guide_fps or capture.shape[1] < 17:
        raise ValueError("causality diagnostic requires matching masters, fps and eight complete blocks")
    meta = checkpoints.read_adapter_metadata(checkpoint)
    checkpoints.check_adapter_conditions(
        meta,
        base=backbone.identity(dev, "dev", "2.5"),
        objective="white",
        guide_mode="d1",
        schedule=[sigma, 0.0],
        geometry={"block_latent_frames": 2, "context_latent_frames": 8, "sink_latent_frames": 1},
        teacher_forcing=False,
    )
    identities = {str(path.resolve()): sha256(path) for path in paths}
    session = open_session(
        argparse.Namespace(model="2.5", gpu_id=gpu_id, seed=42),
        script="onestep_avatar.experiments.causality",
        transformer_path=dev,
    )
    loras = (LoraPathStrengthAndSDOps(str(checkpoint), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)
    raw_outputs = []
    raw_inputs = {}
    with session.transformer(dev, loras=loras) as transformer:
        diagnostic = causality_probe(
            transformer,
            session,
            capture,
            guide,
            fps,
            sigma,
            **({"output_observer": raw_outputs.extend} if output_tensors_path is not None else {}),
            **({"input_observer": raw_inputs.update} if input_tensors_path is not None else {}),
        )
    del transformer
    torch.cuda.empty_cache()
    if any(sha256(path) != identities[str(path.resolve())] for path in paths):
        raise ValueError("causality diagnostic inputs changed during execution")
    if output_tensors_path is not None:
        output_tensors_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(output_tensors_path, lambda p: torch.save(raw_outputs, p))
    if input_tensors_path is not None:
        input_tensors_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(input_tensors_path, lambda p: torch.save(raw_inputs, p))
    result = {
        "checkpoint": str(checkpoint),
        "view": str(view),
        "sigma": sigma,
        **diagnostic,
        "input_sha256": identities,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"))
    return result


EXTRA_SOURCES = ("scripts/onestep_avatar/experiments/__init__.py", "scripts/onestep_avatar/experiments/causality.py")


def parse_future_args(argv: list[str]) -> argparse.Namespace:
    """Own intervention options; pass only ordinary arguments to the shared parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--changed-noise-file", type=Path)
    parser.add_argument("--future-noise-start", type=int)
    intervention, remaining = parser.parse_known_args(argv)
    args = evaluate.parse_args(remaining)
    if (intervention.changed_noise_file is None) != (intervention.future_noise_start is None):
        parser.error("future-noise probe requires both changed noise and an encoded boundary")
    if intervention.changed_noise_file is not None and (args.noise_file is None or intervention.future_noise_start < 1):
        parser.error("future-noise probe requires saved original noise and a positive boundary")
    args.changed_noise_file = intervention.changed_noise_file
    args.future_noise_start = intervention.future_noise_start
    args.ordinary_arguments = remaining
    return args


def prepare_evaluation(args: argparse.Namespace, *, require_fresh_output: bool = True) -> tuple:
    """Complete intervention-byte checks before any native session can open."""
    prepared = evaluate.prepare_evaluation(args, require_fresh_output=require_fresh_output)
    args.changed_noise = None
    if args.changed_noise_file is None:
        return prepared
    changed = torch.load(args.changed_noise_file, map_location="cpu", weights_only=True)
    if not isinstance(changed, torch.Tensor) or changed.dtype != torch.bfloat16 or not torch.isfinite(changed).all():
        raise ValueError("changed comparison noise must be a finite native-bf16 tensor")
    specification, _variants, cases, _membership = prepared
    for video, frames, _requested, _adapters in cases:
        start = args.future_noise_start
        boundary = start * video.z_y.shape[2] * video.z_y.shape[3]
        expected = (1, frames * video.z_y.shape[2] * video.z_y.shape[3], video.z_y.shape[0])
        if not 0 < start < frames or tuple(changed.shape) != expected:
            raise ValueError("future-noise boundary/shape differs from selected input")
        if args.mode == "causal":
            plan = causal.CausalGeometry(
                specification.scale_factors,
                args.mode_settings.block_latent_frames,
                args.mode_settings.context_latent_frames,
            ).plan(frames)
            if start not in [end for _, end in plan[:-1]]:
                raise ValueError("causal future-noise boundary must separate completed blocks")
        if hashing.tensor_sha256(args.saved_noise[:, :boundary]) != hashing.tensor_sha256(changed[:, :boundary]):
            raise ValueError("future-noise diagnostic changed earlier noise")
        if torch.equal(args.saved_noise[:, boundary:], changed[:, boundary:]):
            raise ValueError("future-noise diagnostic requires changed later noise")
    args.changed_noise = changed
    return prepared


def verify_evaluation_conditions(arguments: list[str], record_paths: list[Path]) -> None:
    """Verify both branches through ordinary invariants, then recompute the pair diagnostic."""
    args = parse_future_args(arguments)
    prepare_evaluation(args, require_fresh_output=False)
    if args.changed_noise is None:
        evaluate.verify_evaluation_conditions(args.ordinary_arguments, record_paths, extra_sources=EXTRA_SOURCES)
        return

    def branches(destination: Path) -> list[tuple[Path, torch.Tensor | None]]:
        changed = torch.load(destination.parent / "changed_noise.pt", map_location="cpu", weights_only=True)
        if not isinstance(changed, torch.Tensor) or hashing.tensor_sha256(changed) != hashing.tensor_sha256(
            args.changed_noise
        ):
            raise ValueError("saved changed noise differs from requested input")
        return [(destination / "original", None), (destination / "changed", changed)]

    def validate(destination: Path, records: list[dict], predictions: list[torch.Tensor]) -> None:
        diagnostic = json.loads((destination / "future_noise.json").read_text())
        boundary = args.future_noise_start
        left, right = (value[:, :, :boundary] for value in predictions)
        expected = {
            "change_start_encoded_frame": boundary,
            "records": records,
            "earlier_output_bit_identical": torch.equal(left, right),
            "earlier_output_max_abs_delta": float((left.float() - right.float()).abs().max()),
            "later_output_max_abs_delta": float(
                (predictions[0][:, :, boundary:].float() - predictions[1][:, :, boundary:].float()).abs().max()
            ),
        }
        if any(key not in diagnostic or diagnostic[key] != value for key, value in expected.items()):
            raise ValueError("future-noise diagnostic differs from saved results")

    evaluate.verify_evaluation_conditions(
        args.ordinary_arguments,
        record_paths,
        branch_provider=branches,
        records_validator=validate,
        extra_sources=EXTRA_SOURCES,
    )


def evaluation_evidence_paths(arguments: list[str], record_paths: list[Path]) -> list[Path]:
    """Own the paired publication inventory; ordinary evaluation knows no branches."""
    args = parse_future_args(arguments)
    if args.changed_noise_file is None:
        return evaluate.evaluation_evidence_paths(args.ordinary_arguments, record_paths)
    paths = set(record_paths)
    paths.add(args.output / "text.pt")
    if args.cfg != 1:
        paths.add(args.output / "negative_text.pt")
    for record in record_paths:
        variant = record.parent.parent
        paths.update(
            (
                record.parent / "generated.pt",
                variant.parent / "noise.pt",
                variant.parent / "changed_noise.pt",
                variant / "future_noise.json",
            )
        )
    return sorted(paths)


def execute_evaluation(args: argparse.Namespace) -> int:
    """Intervene through the one shared model/runtime owner and explicit publication hook."""
    prepare_evaluation(args)
    if args.changed_noise is None:
        return evaluate.execute_evaluation(args, extra_sources=EXTRA_SOURCES)
    pending = {}

    def runner(transformer, context, grid, capture, guide, noise, **settings):  # noqa: ANN001, ANN202
        outputs, diagnostic = probe_future_noise(
            transformer,
            context,
            grid,
            capture,
            guide,
            noise,
            args.changed_noise.to(device=noise.device),
            change_start_frame=args.future_noise_start,
            **settings,
        )
        pending.update(outputs=outputs, diagnostic=diagnostic)
        return outputs[0], diagnostic["records"][0]

    def publish(_output: torch.Tensor, record: dict, destination: Path) -> dict:
        atomic_write(
            destination.parent / "changed_noise.pt", lambda temporary: torch.save(args.changed_noise, temporary)
        )
        diagnostic = pending["diagnostic"]
        provenance = {key: value for key, value in record.items() if key not in diagnostic["records"][0]}
        return save_future_noise_probe(pending["outputs"], diagnostic, provenance, destination)

    return evaluate.execute_evaluation(
        args, sample_runner=runner, result_publisher=publish, extra_sources=EXTRA_SOURCES
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a pinned queue specification or the preserved direct diagnostic interface."""
    arguments = sys.argv[1:] if argv is None else argv
    if "--spec" not in arguments:
        if "--causality" not in arguments:
            return parse_future_args(arguments)
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--causality", action="store_true", required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--view", type=Path, required=True)
        parser.add_argument("--sigma", type=float, required=True)
        parser.add_argument("--gpu-id", type=int, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return parser.parse_args(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    outer = parser.parse_args(arguments)
    outer.completion = {"manifest": str(outer.output.resolve() / "manifest.json")}
    return outer


def read_spec(outer: argparse.Namespace) -> argparse.Namespace:
    """Read scientific settings only after queue path normalization or direct CLI parsing."""
    spec = json.loads(outer.spec.read_text())
    if spec.get("schema_version") != 1 or spec.get("protocol") not in ("future_noise", "eight_block"):
        raise ValueError("causality requires a version-one future_noise or eight_block specification")
    if spec["protocol"] == "future_noise":
        if (
            set(spec) != {"schema_version", "protocol", "arguments"}
            or not isinstance(spec["arguments"], list)
            or any(not isinstance(token, str) for token in spec["arguments"])
        ):
            raise ValueError("future-noise specification has an invalid field inventory")
        if any(
            token.split("=", 1)[0] in ("--output", "--gpu-id", "--dry-run", "--spec", "--help")
            for token in spec["arguments"]
        ):
            raise ValueError("scientific specification contains an execution override")
        args = parse_future_args([*spec["arguments"], "--output", str(outer.output), "--gpu-id", str(outer.gpu_id)])
        for field, value in vars(args).items():
            if field == "output":
                continue
            values = [value] if isinstance(value, Path) else value if isinstance(value, list) else []
            if any(isinstance(item, Path) and not item.is_absolute() for item in values):
                raise ValueError("scientific specification paths must be absolute")
    else:
        if set(spec) != {"schema_version", "protocol", "checkpoint", "view", "sigma"}:
            raise ValueError("eight-block specification has an invalid field inventory")
        args = argparse.Namespace(
            checkpoint=Path(spec["checkpoint"]), view=Path(spec["view"]), sigma=spec["sigma"], causality=True
        )
        validate_schedule([args.sigma, 0])
        if not args.checkpoint.is_absolute() or not args.view.is_absolute():
            raise ValueError("scientific specification paths must be absolute")
    args.spec, args.output, args.gpu_id, args.dry_run = outer.spec, outer.output, outer.gpu_id, outer.dry_run
    args.protocol, args.completion = spec["protocol"], outer.completion
    return args


def future_record_paths(args: argparse.Namespace) -> list[Path]:
    """Derive exact case/variant inventory from checked requests."""
    _specification, variants, cases, _membership = prepare_evaluation(args, require_fresh_output=False)
    result = []
    for index in range(len(cases)):
        for variant in range(len(variants)):
            destination = args.output / f"case_{index:04d}" / f"variant_{variant:03d}"
            folders = (
                [destination] if args.changed_noise_file is None else [destination / b for b in ("original", "changed")]
            )
            result.extend(folder / "result.json" for folder in folders)
    return result


def publish_manifest(args: argparse.Namespace, paths: list[Path], *, mode: str) -> dict:
    """Bind every saved result byte after scientific verification, without restamping old runs."""
    manifest = {
        "schema_version": 1,
        "kind": "onestep_avatar.causality",
        "protocol": args.protocol,
        "spec_sha256": sha256(args.spec),
        "software": software.capture("evaluation", mode, extra_sources=EXTRA_SOURCES),
        "artifacts": {str(path.resolve()): sha256(path) for path in sorted(set(paths))},
    }
    atomic_write(
        args.output / "manifest.json", lambda p: p.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    )
    return manifest


def _verify_eight_block_inputs(args: argparse.Namespace, root: Path, result: dict) -> tuple[torch.Tensor, float]:
    """Check full masters, actual saved native inputs and recorded tensor digests."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: PLC0415 -- model-free tensor layout

    inputs = torch.load(root / "raw_inputs.pt", map_location="cpu", weights_only=True)
    capture, fps = dataset.load_training_master(args.view / dataset.capture_bundle_name("white"))
    guide, guide_fps = dataset.load_training_master(args.view / dataset.guide_bundle_name("white"))
    if capture.shape != guide.shape or fps != guide_fps or capture.shape[1] < 17:
        raise ValueError("eight-block completion requires matching full masters and frame rates")
    patchifier = VideoLatentPatchifier(patch_size=1)
    target = patchifier.patchify(capture.unsqueeze(0).to(torch.bfloat16))
    source = patchifier.patchify(guide.unsqueeze(0).to(torch.bfloat16))
    required = {"capture", "guide", "text", "original_noise", "mixed_noise"}
    if (
        not isinstance(inputs, dict)
        or set(inputs) != required
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bfloat16
            or value.numel() == 0
            or not torch.isfinite(value).all()
            for value in inputs.values()
        )
    ):
        raise ValueError("eight-block completion requires finite native-bf16 raw inputs")
    if (
        any(inputs[name].shape != target.shape for name in required - {"text"})
        or hashing.tensor_sha256(inputs["capture"]) != hashing.tensor_sha256(target)
        or hashing.tensor_sha256(inputs["guide"]) != hashing.tensor_sha256(source)
        or inputs["text"].ndim != 3
        or inputs["text"].shape[0] != 1
    ):
        raise ValueError("eight-block raw input shapes or master bytes differ")
    boundary = 9 * capture.shape[2] * capture.shape[3]
    original, mixed = inputs["original_noise"], inputs["mixed_noise"]
    if hashing.tensor_sha256(original[:, :boundary]) != hashing.tensor_sha256(mixed[:, :boundary]) or torch.equal(
        original[:, boundary:], mixed[:, boundary:]
    ):
        raise ValueError("eight-block raw noise must keep earlier bytes and change later noise")
    expected_hashes = {f"{name}_sha256": hashing.tensor_sha256(inputs[name]) for name in ("capture", "guide", "text")}
    rows = result.get("records", [])
    if (
        any(result.get(name) != digest for name, digest in expected_hashes.items())
        or len(rows) != 2
        or any(
            row.get("noise_sha256") != hashing.tensor_sha256(inputs[name])
            for row, name in zip(rows, ("original_noise", "mixed_noise"), strict=True)
        )
    ):
        raise ValueError("eight-block saved input or noise digest differs from raw inputs")
    return capture, fps


def _verify_eight_block_outputs(args: argparse.Namespace, root: Path) -> list[Path]:
    """Check the preserved output control and calls after verifying its saved inputs."""
    paths = [root / "result.json", root / "raw_outputs.pt", root / "raw_inputs.pt"]
    result = json.loads(paths[0].read_text())
    outputs = torch.load(paths[1], map_location="cpu", weights_only=True)
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError("eight-block completion requires both raw outputs")
    capture, fps = _verify_eight_block_inputs(args, root, result)
    if fps <= 0 or any(
        not isinstance(v, torch.Tensor)
        or v.dtype != torch.float32
        or list(v.shape) != [1, *capture[:, :17].shape]
        or not torch.isfinite(v).all()
        or not torch.equal(v[:, :, 0], capture[:, 0].unsqueeze(0).to(torch.bfloat16).float())
        for v in outputs
    ):
        raise ValueError("eight-block saved output shape, c0 or values differ")
    delta = float((outputs[0][:, :, 9:] - outputs[1][:, :, 9:]).abs().max())
    expected = {
        "checkpoint": str(args.checkpoint),
        "view": str(args.view),
        "sigma": args.sigma,
        "latent_frames_compared_equal": [0, 9],
        "shared_noise_blocks": [0, 1, 2, 3],
        "changed_noise_blocks": [4, 5, 6, 7],
        "earlier_blocks_bit_identical": torch.equal(outputs[0][:, :, :9], outputs[1][:, :, :9]),
        "later_blocks_max_abs_diff": delta,
        "later_blocks_changed": delta > 0,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("eight-block saved diagnostic differs from raw outputs")
    if not expected["earlier_blocks_bit_identical"] or delta <= 0:
        raise ValueError("eight-block causal control failed or was insensitive")
    identities = result.get("input_sha256", {})
    expected_inputs = [
        backbone.transformer_path("2.5", "dev"),
        args.checkpoint,
        args.view / dataset.capture_bundle_name("white"),
        args.view / dataset.guide_bundle_name("white"),
    ]
    if identities != {str(path.resolve()): sha256(path) for path in expected_inputs}:
        raise ValueError("eight-block input identities differ")
    rows = result.get("records", [])
    if len(rows) != 2 or any(
        row.get("call_counts", {}).get(key) != count
        for row in rows
        for key, count in (("model_calls", 16), ("denoise_calls", 8), ("refresh_calls", 8))
    ):
        raise ValueError("eight-block execution inventory is incomplete")
    return paths


def verify_completion(spec: Path, root: Path) -> dict:
    """Recompute the experiment controls from saved tensors; never load native weights."""
    args = read_spec(parse_args(["--spec", str(spec), "--output", str(root)]))
    manifest = json.loads((root / "manifest.json").read_text())
    if args.protocol == "future_noise":
        records = future_record_paths(args)
        verify_evaluation_conditions(
            args.ordinary_arguments
            + (
                [
                    "--changed-noise-file",
                    str(args.changed_noise_file),
                    "--future-noise-start",
                    str(args.future_noise_start),
                ]
                if args.changed_noise_file is not None
                else []
            ),
            records,
        )
        paths = evaluation_evidence_paths(
            args.ordinary_arguments
            + (
                [
                    "--changed-noise-file",
                    str(args.changed_noise_file),
                    "--future-noise-start",
                    str(args.future_noise_start),
                ]
                if args.changed_noise_file is not None
                else []
            ),
            records,
        )
        mode = args.mode
    else:
        paths = _verify_eight_block_outputs(args, root)
        mode = "causal"
    expected = {
        "schema_version": 1,
        "kind": "onestep_avatar.causality",
        "protocol": args.protocol,
        "spec_sha256": sha256(spec),
        "software": software.capture("evaluation", mode, extra_sources=EXTRA_SOURCES),
        "artifacts": {str(path.resolve()): sha256(path) for path in sorted(set(paths))},
    }
    if manifest != expected:
        raise ValueError("causality completion specification, software or artifacts differ")
    software.check_current(manifest["software"])
    return manifest


def evidence_paths(spec: Path, root: Path) -> list[Path]:
    """Return the fully verified byte inventory to the generic queue receipt."""
    manifest = verify_completion(spec, root)
    return sorted([root / "manifest.json", *map(Path, manifest["artifacts"])])


def main(argv: list[str] | None = None) -> int:
    """Run only the explicitly selected protocol."""
    args = parse_args(argv)
    if not hasattr(args, "spec"):
        if getattr(args, "causality", False):
            evaluate_causality(args.checkpoint, args.view, args.output, gpu_id=args.gpu_id, sigma=args.sigma)
            return 0
        return execute_evaluation(args)
    args = read_spec(args)
    producer_software = software.capture(
        "evaluation", args.mode if args.protocol == "future_noise" else "causal", extra_sources=EXTRA_SOURCES
    )
    spec_hash = sha256(args.spec)
    if args.protocol == "future_noise":
        execute_evaluation(args)
        if args.dry_run:
            return 0
        paths = evaluation_evidence_paths(
            args.ordinary_arguments
            + (
                [
                    "--changed-noise-file",
                    str(args.changed_noise_file),
                    "--future-noise-start",
                    str(args.future_noise_start),
                ]
                if args.changed_noise_file is not None
                else []
            ),
            future_record_paths(args),
        )
        verify_evaluation_conditions(
            args.ordinary_arguments
            + (
                [
                    "--changed-noise-file",
                    str(args.changed_noise_file),
                    "--future-noise-start",
                    str(args.future_noise_start),
                ]
                if args.changed_noise_file is not None
                else []
            ),
            future_record_paths(args),
        )
        if sha256(args.spec) != spec_hash:
            raise ValueError("causality specification changed during execution")
        software.check_current(producer_software)
        publish_manifest(args, paths, mode=args.mode)
    else:
        if args.output.exists():
            raise ValueError("eight-block experiment requires a fresh output directory")
        if args.dry_run:
            raise ValueError("eight-block dry run is queue command preparation only")
        evaluate_causality(
            args.checkpoint,
            args.view,
            args.output / "result.json",
            gpu_id=args.gpu_id,
            sigma=args.sigma,
            output_tensors_path=args.output / "raw_outputs.pt",
            input_tensors_path=args.output / "raw_inputs.pt",
        )
        if sha256(args.spec) != spec_hash:
            raise ValueError("causality specification changed during execution")
        software.check_current(producer_software)
        publish_manifest(
            args,
            [args.output / "result.json", args.output / "raw_outputs.pt", args.output / "raw_inputs.pt"],
            mode="causal",
        )
    verify_completion(args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
