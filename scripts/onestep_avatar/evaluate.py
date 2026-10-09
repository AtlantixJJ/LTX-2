"""Check and execute same-input comparisons; see doc/evaluate.md."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.onestep_avatar import hashing, metrics
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters as adapter_loader
from scripts.onestep_avatar.model import backbone, bidirectional, causal, common
from scripts.onestep_avatar.model.sampling import validate_schedule
from scripts.onestep_avatar.training import checkpoints
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings


def check_adapter(path: Path | None, requested: dict, *, override: bool = False, product: bool = False) -> dict:
    """Read metadata and actual matrices before a transformer is opened."""
    if path is None:
        return {"adapter": None, "application_method": "base", "overrides": []}
    record = checkpoints.read_contract(path)
    checkpoints.validate_adapter_tensors(path, record)
    differences = checkpoints.check_contract(record, requested, override=override, product=product)
    return {
        "adapter": str(path),
        "adapter_sha256": sha256(path),
        "contract": record,
        "application_method": requested["application_method"],
        "overrides": differences,
    }


def validate_comparison(records: list[dict], changed_factor: str) -> None:
    """Require recorded same-input evidence except for the single declared factor."""
    allowed = {
        "adapter": {"adapter_sha256", "adapter", "application_method"},
        "guide_mode": {"guide_mode", "conditions.task.guide_mode"},
        "schedule": {"schedule", "conditions.schedule"},
        "history": {"mode_settings.teacher_forcing", "conditions.mode_settings.teacher_forcing"},
        "history_mode": {"history_mode", "conditions.history_mode"},
        "kv_source": {"kv_source", "conditions.kv_source"},
        "guidance": {"guidance"},
        "application_method": {"application_method", "conditions.application_method"},
        "mode": {"mode", "mode_settings", "conditions.mode", "conditions.mode_settings"},
    }
    if changed_factor not in allowed or len(records) < 2:
        raise ValueError("comparison requires two records and one supported changed factor")
    if changed_factor == "adapter":
        methods = {record.get("application_method") for record in records if record.get("adapter") is not None}
        if len(methods) > 1:
            raise ValueError("adapter comparison changes the application method")

    def evidence(record: dict) -> dict:
        required = (
            "capture_sha256",
            "guide_sha256",
            "c0_sha256",
            "noise_sha256",
            "text_sha256",
            "source",
            "fps",
            "frames",
            "mode",
            "mode_settings",
            "guide_mode",
            "schedule",
            "conditions",
            "application_method",
            "adapter",
        )
        if any(key not in record for key in required):
            raise ValueError("comparison record lacks fixed-input evidence")
        selected = {key: record[key] for key in required}
        if record["mode"] == "causal" and any(key not in record for key in ("history_mode", "kv_source")):
            raise ValueError("causal comparison lacks history diagnostic settings")
        selected.update(history_mode=record.get("history_mode"), kv_source=record.get("kv_source"))
        selected["guidance"] = record.get("guidance")
        selected["adapter_sha256"] = record.get("adapter_sha256")
        for path in allowed[changed_factor]:
            keys = path.split(".")
            node = selected
            # Copy each traversed dictionary: never mutate the saved executed record.
            for key in keys[:-1]:
                if not isinstance(node.get(key), dict):
                    raise ValueError(f"comparison record lacks {path}")
                node[key] = dict(node[key])
                node = node[key]
            node.pop(keys[-1], None)
        return selected

    baseline = evidence(records[0])
    for record in records[1:]:
        if evidence(record) != baseline:
            raise ValueError("comparison changes fixed inputs or a second factor")


@contextmanager
def measure_calls(transformer: torch.nn.Module) -> Iterator[dict]:
    """Count actual underlying forwards, including every guidance pass."""
    measured = {"model_calls": 0}

    def count(_module: torch.nn.Module, _args: tuple, _output: object) -> None:
        measured["model_calls"] += 1

    hook = common.base_model(transformer).register_forward_hook(count)
    try:
        yield measured
    finally:
        hook.remove()


@torch.no_grad()
def sample_case(  # noqa: PLR0913 -- explicit checked model inputs and selected mode
    transformer: torch.nn.Module,
    context: torch.Tensor,
    grid: common.ClipGrid,
    capture: torch.Tensor,
    guide: torch.Tensor | None,
    epsilon: torch.Tensor,
    *,
    mode: str,
    mode_settings: BidirectionalSettings | CausalSettings,
    guide_mode: str,
    schedule: list[float],
    seed: int,
    history_mode: str = "cache",
    kv_source: str = "refresh",
    predict_x0=None,  # noqa: ANN001 -- optional guidance already constructed by the caller
) -> tuple[torch.Tensor, dict]:
    """Both modes consume the same source, c0, text and saved full-range noise."""
    levels = list(validate_schedule(schedule))
    source = common.source_for(capture, guide, guide_mode)
    if source.shape != epsilon.shape or not torch.isfinite(epsilon).all():
        raise ValueError("saved noise must be finite and match the full input token range")
    if mode not in ("bidirectional", "causal"):
        raise ValueError("mode must be bidirectional or causal")
    if (mode == "bidirectional") != isinstance(mode_settings, BidirectionalSettings):
        raise ValueError("settings do not match the explicitly selected mode")
    c0 = capture[:, : grid.tokens_per_latent_frame]
    predict_x0 = common.denoised_from_x0_model(transformer) if predict_x0 is None else predict_x0
    started = time.perf_counter()
    with measure_calls(transformer) as measured:
        if mode == "bidirectional":
            tokens, counts = bidirectional.sample(
                predict_x0, context, grid, source, c0, schedule=levels, seed=seed, epsilon=epsilon
            )
            frames = grid.latent_frames
            counts["model_calls"] = counts["denoise_calls"]
        else:
            geometry = causal.CausalGeometry(
                grid.tools.scale_factors, mode_settings.block_latent_frames, mode_settings.context_latent_frames
            )
            blocks = geometry.plan(grid.latent_frames)
            tokens, counts = causal.sample(
                predict_x0,
                context,
                grid,
                source,
                c0,
                transformer=transformer,
                geometry=geometry,
                schedule=levels,
                seed=seed,
                epsilon=epsilon,
                teacher_tokens=capture,
                teacher_forcing=mode_settings.teacher_forcing,
                history_mode=history_mode,
                kv_source=kv_source,
            )
            frames = blocks[-1][1]
    counts.update(measured)
    output = grid.unpatchify_block(tokens[:, : frames * grid.tokens_per_latent_frame], frames).cpu()
    target = grid.unpatchify_block(capture[:, : frames * grid.tokens_per_latent_frame], frames).cpu()
    return output, {
        "schema_version": 2,
        "mode": mode,
        "mode_settings": asdict(mode_settings),
        "guide_mode": guide_mode,
        "schedule": levels,
        "seed": seed,
        "frames": frames,
        "capture_sha256": hashing.tensor_sha256(capture),
        "guide_sha256": None if guide is None else hashing.tensor_sha256(guide),
        "c0_sha256": hashing.tensor_sha256(c0),
        "noise_sha256": hashing.tensor_sha256(epsilon),
        "text_sha256": hashing.tensor_sha256(context),
        "call_counts": counts,
        "elapsed_s": time.perf_counter() - started,
        "metrics": metrics.encoded_metrics(output, target),
        **({"history_mode": history_mode, "kv_source": kv_source} if mode == "causal" else {}),
    }


def save_case(output: torch.Tensor, record: dict, destination: Path) -> dict:
    """Publish encoding first, then a JSON record with its actual serialized hash."""
    destination.mkdir(parents=True, exist_ok=True)
    encoding = destination / "generated.pt"
    atomic_write(encoding, lambda temporary: torch.save(output, temporary))
    completed = {
        **record,
        "output": {"path": str(encoding), "sha256": sha256(encoding), "shape": list(output.shape)},
        "state": "complete",
    }
    atomic_write(
        destination / "result.json",
        lambda temporary: temporary.write_text(json.dumps(completed, indent=2, allow_nan=False) + "\n"),
    )
    return completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: PLR0912, PLR0915 -- explicit mode/diagnostic argument gates
    """Require execution mode and preserve explicit versus omitted causal options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bidirectional", "causal"), required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--frame-plan", type=Path)
    parser.add_argument("--split", choices=("train", "held_out", "validation", "test"))
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--model", default="2.5")
    parser.add_argument("--variant", choices=backbone.VARIANTS, default=backbone.DEFAULT_VARIANT)
    parser.add_argument("--guide-mode", choices=("d0", "d1"), default="d1")
    parser.add_argument("--schedule", nargs="+", type=float, required=True)
    parser.add_argument("--span-latent-frames", type=int)
    parser.add_argument("--output-latent-frames", type=int,
                        help="causal physical prefix; keep the adapter's recorded training span unchanged")
    parser.add_argument("--block-latent-frames", type=int)
    parser.add_argument("--blocks-per-sample", type=int)
    parser.add_argument("--context-latent-frames", type=int)
    parser.add_argument("--teacher-forcing", action="store_true", default=None)
    parser.add_argument("--history-mode", choices=("cache", "recompute", "joint"))
    parser.add_argument("--kv-source", choices=("refresh", "denoise"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-file", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--stg", type=float, default=0.0)
    parser.add_argument("--stg-blocks", type=int, nargs="+", default=[])
    parser.add_argument("--rescale", type=float, default=0.0)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--research-override", action="store_true")
    parser.add_argument("--adapter-application", choices=adapter_loader.METHODS, default=adapter_loader.UNMERGED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.cfg) or args.cfg < 0:
        parser.error("CFG scale must be finite and nonnegative")
    if not math.isfinite(args.stg) or args.stg < 0 or not math.isfinite(args.rescale) or not 0 <= args.rescale <= 1:
        parser.error("STG must be finite/nonnegative and rescale must be in [0,1]")
    if args.stg > 0 and not args.stg_blocks:
        parser.error("positive STG requires explicit perturbation blocks")
    if len(set(args.stg_blocks)) != len(args.stg_blocks) or any(block < 0 for block in args.stg_blocks):
        parser.error("STG blocks must be unique nonnegative indices")
    if args.stg == 0 and args.stg_blocks:
        parser.error("perturbation blocks require positive STG")
    if args.negative_prompt is not None and args.cfg == 1.0:
        parser.error("negative prompt requires non-unit CFG")
    args.schedule = list(validate_schedule(args.schedule))
    if args.span_latent_frames is not None and args.span_latent_frames < 1:
        parser.error("--span-latent-frames must be positive")
    if args.output_latent_frames is not None:
        if args.output_latent_frames < 1:
            parser.error("--output-latent-frames must be positive")
        if args.mode != "causal":
            parser.error("--output-latent-frames is only supported for causal mode")
        if args.span_latent_frames is not None and args.span_latent_frames != args.output_latent_frames:
            parser.error("explicit output and training span lengths must match")
    if args.mode == "bidirectional":
        if any(
            getattr(args, key) is not None
            for key in (
                "block_latent_frames",
                "blocks_per_sample",
                "context_latent_frames",
                "teacher_forcing",
                "history_mode",
                "kv_source",
            )
        ):
            parser.error("bidirectional mode rejects block/cache/history options")
        args.mode_settings = BidirectionalSettings(args.span_latent_frames)
    else:
        args.mode_settings = CausalSettings(
            2 if args.block_latent_frames is None else args.block_latent_frames,
            3 if args.blocks_per_sample is None else args.blocks_per_sample,
            8 if args.context_latent_frames is None else args.context_latent_frames,
            bool(args.teacher_forcing),
            args.span_latent_frames,
        )
        if args.mode_settings.block_latent_frames < 1 or args.mode_settings.blocks_per_sample < 1:
            parser.error("causal block length and K must be positive")
        if not 0 <= args.mode_settings.context_latent_frames <= causal.MAX_CONTEXT_LATENT_FRAMES:
            parser.error("causal history depth is outside the supported range")
        args.history_mode = "cache" if args.history_mode is None else args.history_mode
        args.kv_source = "refresh" if args.kv_source is None else args.kv_source
        if args.kv_source == "denoise" and (args.history_mode != "cache" or args.mode_settings.teacher_forcing):
            parser.error("denoise K/V requires cached generated history")
    return args


def prepare_evaluation(args: argparse.Namespace, *, require_fresh_output: bool = True) -> tuple:  # noqa: PLR0912, PLR0915 -- ordered pre-weight input/adapter gates
    """All data, schedule, output and adapter gates precede model/text sessions."""
    membership = json.loads(args.subset.read_text())
    subset.validate_membership(membership)
    if args.frame_plan is not None:
        plan = json.loads(args.frame_plan.read_text())
        if (plan.get("schema_version") != 2 or plan.get("membership_sha256") != membership["sha256"]
                or plan.get("mode") != args.mode or subset.record_hash(plan) != plan.get("sha256")):
            raise ValueError("evaluation frame plan differs from the fixed membership or mode")
    store = dataset.ClipStore(membership, args.corpus_root)
    store.verify(require_guide=args.guide_mode == "d1")
    available = [
        source for source, record in store.sources.items()
        if args.split is None or record["split"] == args.split
    ]
    ids = args.source or available
    if not ids:
        raise ValueError("source selection is empty for the requested fixed-video split")
    if len(set(ids)) != len(ids) or any(source not in store.sources for source in ids):
        raise ValueError("source selection contains duplicates or unknown videos")
    if args.split is not None and any(store.sources[source]["split"] != args.split for source in ids):
        raise ValueError("source selection differs from the requested fixed-video split")
    if require_fresh_output and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("evaluation output is already used; choose a new directory")
    specification = backbone.resolve(args.model, args.variant)
    if args.stg > 0 and any(block >= specification.caps.num_layers for block in args.stg_blocks):
        raise ValueError("STG block index exceeds the selected base layer count")
    if args.variant == "distilled" and any(level not in specification.sigmas for level in args.schedule[:-1]):
        raise ValueError("evaluation schedule is outside the distilled base grid")
    from scripts.onestep_avatar.corpus.precompute import file_fingerprint  # noqa: PLC0415 -- producer identity

    vae_fingerprint = file_fingerprint(Path(specification.paths.video_vae()))
    for source in ids:
        for role in ("capture", "guide") if args.guide_mode == "d1" else ("capture",):
            if store.sources[source].get(f"{role}_encode_record", {}).get("vae_fingerprint") != vae_fingerprint:
                raise ValueError(f"{source}: {role} encoding VAE differs from the selected base VAE")
    base = backbone.identity(specification.paths.transformer(), args.variant, args.model, full_hash=True)
    variants = ([None] if not args.checkpoint or args.include_base else []) + args.checkpoint
    cases = []
    saved_noise = (
        None if args.noise_file is None else torch.load(args.noise_file, map_location="cpu", weights_only=True)
    )
    if saved_noise is not None and (not isinstance(saved_noise, torch.Tensor) or not torch.isfinite(saved_noise).all()):
        raise ValueError("saved comparison noise must be a finite tensor")
    if saved_noise is not None and saved_noise.dtype != torch.bfloat16:
        raise ValueError("saved comparison noise must use the native bf16 dtype")
    args.saved_noise = saved_noise
    if saved_noise is not None and len(ids) != 1:
        raise ValueError("a saved noise file belongs to exactly one selected video")
    for source in ids:
        video = store.load(source, require_guide=args.guide_mode == "d1")
        output_frames = getattr(args, "output_latent_frames", None)
        frames = (output_frames if output_frames is not None else
                  video.z_y.shape[1] if args.span_latent_frames is None else args.span_latent_frames)
        if not 1 <= frames <= video.z_y.shape[1]:
            raise ValueError("requested range does not fit a selected video")
        if args.mode == "causal":
            geometry = causal.CausalGeometry(
                specification.scale_factors,
                args.mode_settings.block_latent_frames,
                args.mode_settings.context_latent_frames,
            )
            plan = geometry.plan(frames)
            if not plan:
                raise ValueError("selected range has no complete causal block")
            if output_frames is not None and plan[-1][1] != output_frames:
                raise ValueError("explicit output range requires complete causal blocks with that exact frame count")
            frames = plan[-1][1]
        if video.z_y.shape[0] != specification.caps.latent_channels:
            raise ValueError("encoded channels differ from the selected base model")
        expected_noise = (1, frames * video.z_y.shape[2] * video.z_y.shape[3], video.z_y.shape[0])
        if saved_noise is not None and tuple(saved_noise.shape) != expected_noise:
            raise ValueError("saved noise shape differs from selected input tokens")
        if ((frames - 1) * specification.scale_factors.time + 1) / video.fps > common.MAX_ROPE_SECONDS:
            raise ValueError("selected range exceeds the model position limit")
        requested = {
            "application_method": args.adapter_application,
            "global_sigma_dtype": common.SIGMA_PRECISION,
            "mode": args.mode,
            "mode_settings": asdict(args.mode_settings),
            "schedule": args.schedule,
            "model": {"version": args.model, "variant": args.variant, "base_sha256": base["base_transformer_sha256"]},
            "task": {
                "guide_mode": args.guide_mode,
                "objective": store.objective,
                "first_frame_conditioning": "clean_c0_v1",
                "loss": "full_frame_x0_mse",
                "split": args.split,
            },
            "shape": {
                "channels": video.z_y.shape[0],
                "height": video.z_y.shape[2],
                "width": video.z_y.shape[3],
                "frames": frames,
            },
        }
        if args.mode == "causal":
            requested.update(history_mode=args.history_mode, kv_source=args.kv_source)
        adapters = [check_adapter(path, requested, override=args.research_override) for path in variants]
        cases.append((video, frames, requested, adapters))
    return specification, variants, cases, membership


def verify_evaluation_conditions(  # noqa: PLR0912, PLR0915 -- all scientific evidence must agree
    arguments: list[str], record_paths: list[Path], *,
    branch_provider: Callable[[Path], list[tuple[Path, torch.Tensor | None]]] | None = None,
    records_validator: Callable[[Path, list[dict], list[torch.Tensor]], None] | None = None,
    extra_sources: tuple[str, ...] = (),
) -> None:
    """Verify queued results against current inputs, without opening model sessions."""
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT  # noqa: PLC0415
    from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE  # noqa: PLC0415 -- constants only

    if (branch_provider is not None or records_validator is not None) and (not extra_sources
            or (branch_provider is not None and not callable(branch_provider))
            or (records_validator is not None and not callable(records_validator))):
        raise ValueError("custom evaluation verification requires explicit source owners")
    args = parse_args(arguments)
    specification, variants, cases, membership = prepare_evaluation(args, require_fresh_output=False)

    def branches(destination: Path) -> list[tuple[Path, torch.Tensor | None]]:
        rows = [(destination, None)] if branch_provider is None else branch_provider(destination)
        if not rows or any(not folder.resolve().is_relative_to(destination.resolve()) for folder, _ in rows):
            raise ValueError("evaluation verification branch escapes its variant")
        return rows
    expected_paths = []
    for case_index in range(len(cases)):
        for variant_index in range(len(variants)):
            destination = args.output / f"case_{case_index:04d}" / f"variant_{variant_index:03d}"
            expected_paths.extend(folder / "result.json" for folder, _ in branches(destination))
    resolved = [path.resolve() for path in record_paths]
    if len(set(resolved)) != len(resolved) or set(resolved) != {path.resolve() for path in expected_paths}:
        raise ValueError("queue evaluation result inventory differs from requested cases")

    def tensor(path: Path) -> torch.Tensor:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(value, torch.Tensor) or value.dtype != DTYPE or value.numel() == 0
                or not torch.isfinite(value).all()):
            raise ValueError("queue evaluation saved input must be a finite native-bf16 tensor")
        return value

    context = tensor(args.output / "text.pt")
    negative = tensor(args.output / "negative_text.pt") if args.cfg != 1.0 else None
    guidance = {
        "cfg": args.cfg, "stg": args.stg, "stg_blocks": args.stg_blocks, "rescale": args.rescale,
        "negative_text_sha256": None if negative is None else hashing.tensor_sha256(negative),
    }
    prompt = DEFAULT_PROMPT if args.prompt is None else args.prompt
    negative_prompt = (DEFAULT_NEGATIVE_PROMPT if args.negative_prompt is None else args.negative_prompt)
    for case_index, (video, frames, requested, adapters) in enumerate(cases):
        grid = common.ClipGrid.build(
            frames, video.z_y.shape[2] * specification.scale_factors.height,
            video.z_y.shape[3] * specification.scale_factors.width, video.fps, specification,
            device=torch.device("cpu"), dtype=DTYPE, latent_channels=specification.caps.latent_channels,
        )
        capture = grid.patchify(video.z_y[:, :frames].unsqueeze(0).to(dtype=DTYPE))
        guide = None if video.z_g is None else grid.patchify(video.z_g[:, :frames].unsqueeze(0).to(dtype=DTYPE))
        c0 = capture[:, :grid.tokens_per_latent_frame]
        case_dir = args.output / f"case_{case_index:04d}"
        noise = tensor(case_dir / "noise.pt")
        if noise.shape != capture.shape or (args.saved_noise is not None and not torch.equal(noise, args.saved_noise)):
            raise ValueError("queue evaluation saved noise differs from requested input")
        fixed = {
            "schema_version": 2, "state": "complete", "mode": args.mode,
            "mode_settings": asdict(args.mode_settings), "guide_mode": args.guide_mode,
            "schedule": args.schedule, "seed": args.seed, "frames": frames,
            "conditions": requested, "source": video.source, "fps": video.fps,
            "membership_sha256": membership["sha256"], "input_file_hashes": video.hashes,
            "capture_sha256": hashing.tensor_sha256(capture),
            "guide_sha256": None if guide is None else hashing.tensor_sha256(guide),
            "c0_sha256": hashing.tensor_sha256(c0), "text_sha256": hashing.tensor_sha256(context),
            "prompt": prompt, "guidance": guidance,
            "negative_prompt": negative_prompt if negative is not None else None,
            "producer_source_sha256": sha256(Path(__file__)),
            "software": software.capture("evaluation", args.mode, extra_sources=extra_sources),
        }
        if args.mode == "causal":
            fixed.update(history_mode=args.history_mode, kv_source=args.kv_source)
        for variant_index, adapter in enumerate(adapters):
            destination = case_dir / f"variant_{variant_index:03d}"
            branch_records, predictions = [], []
            for folder, branch_noise in branches(destination):
                epsilon = noise if branch_noise is None else branch_noise
                if epsilon.shape != noise.shape or epsilon.dtype != noise.dtype or not torch.isfinite(epsilon).all():
                    raise ValueError("evaluation verification branch noise differs from the source geometry")
                record = json.loads((folder / "result.json").read_text())
                software.check_current(record.get("software"))
                expected = {**fixed, **adapter, "noise_sha256": hashing.tensor_sha256(epsilon)}
                if any(key not in record or record[key] != value for key, value in expected.items()):
                    raise ValueError("queue evaluation scientific settings or input evidence differ")
                encoded = folder / "generated.pt"
                if Path(record.get("output", {}).get("path", "")).resolve() != encoded.resolve():
                    raise ValueError("queue evaluation encoding path differs from requested variant")
                if record.get("output", {}).get("sha256") != sha256(encoded):
                    raise ValueError("evaluation encoding bytes differ from the published record")
                prediction = tensor(encoded)
                if record.get("output", {}).get("shape") != list(prediction.shape):
                    raise ValueError("evaluation encoding shape differs from the published record")
                if list(prediction.shape) != [1, *video.z_y[:, :frames].shape]:
                    raise ValueError("queue evaluation encoding shape differs from requested video")
                if not torch.equal(prediction[:, :, :1], video.z_y[:, :1].unsqueeze(0).to(dtype=DTYPE)):
                    raise ValueError("queue evaluation changed clean first-image input")
                branch_records.append(record)
                predictions.append(prediction)
            if records_validator is not None:
                records_validator(destination, branch_records, predictions)


def evaluation_evidence_paths(arguments: list[str], record_paths: list[Path]) -> list[Path]:
    """Inventory artifacts after scientific verification, for byte-bound receipts."""
    args = parse_args(arguments)
    paths = set(record_paths)
    paths.add(args.output / "text.pt")
    if args.cfg != 1.0:
        paths.add(args.output / "negative_text.pt")
    for record in record_paths:
        paths.add(record.parent / "generated.pt")
        paths.add(record.parent.parent / "noise.pt")
    return sorted(paths)


def execute_evaluation(  # noqa: PLR0912, PLR0915 -- explicit sampler and native session orchestration
    args: argparse.Namespace, sample_runner: Callable[..., tuple[torch.Tensor, dict]] | None = None, *,
    preview_tensor_validator: Callable[[dict, dict[str, torch.Tensor | None]], None] | None = None,
    result_publisher: Callable[[torch.Tensor, dict, Path], object] | None = None,
    extra_sources: tuple[str, ...] = (),
) -> int:
    """Execute ordinary sampling; a fixed preview requires its canonical tensor validator."""
    if getattr(args, "preview_fixed", None) is not None and not callable(preview_tensor_validator):
        raise ValueError("fixed preview requires its explicit preview tensor validator")
    if result_publisher is not None and (not callable(result_publisher) or not extra_sources):
        raise ValueError("custom evaluation publication requires explicit source owners")
    producer_source = sha256(Path(__file__))
    producer_software = software.capture("evaluation", args.mode, extra_sources=extra_sources)
    specification, variants, cases, membership = prepare_evaluation(args)
    if (getattr(args, "preview_fixed", None) is not None and args.cfg != 1.0
            and 'negative_text' not in args.preview_fixed.get('input_files', {})):
        raise ValueError("fixed previews require pinned negative text before enabling CFG")
    if args.dry_run:
        print(  # noqa: T201 -- dry-run output
            json.dumps(
                {
                    "cases": [requested for _, _, requested, _ in cases],
                    "variants": [str(path) if path is not None else "base" for path in variants],
                },
                indent=2,
            )
        )
        return 0
    # Repeat pinned adapter checks before text or native weights can open.
    for _, _, _, case_adapters in cases:
        for index, checkpoint in enumerate(variants):
            if checkpoint is not None:
                checkpoints.recheck_adapter(
                    checkpoint, case_adapters[index]["contract"], case_adapters[index]["adapter_sha256"]
                )
    # Heavy native handles are imported only after every cheap condition has passed.
    software.check_current(producer_software)
    from scripts.prune.core import preflight  # noqa: PLC0415
    from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE, Session  # noqa: PLC0415
    from scripts.prune.data import prompt_cache  # noqa: PLC0415

    preflight.check(args.model, gpu_id=args.gpu_id, transformer_path=specification.paths.transformer())
    device = torch.device(f"cuda:{args.gpu_id}")
    prompt = DEFAULT_PROMPT if args.prompt is None else args.prompt
    fixed = getattr(args, "preview_fixed", None)
    context = (
        prompt_cache.get_or_build(specification, prompt, DTYPE, device)
        if fixed is None
        else torch.load(Path(fixed["input_files"]["text"]["path"]), map_location="cpu", weights_only=True)
    )
    if fixed is not None:
        if not isinstance(context, torch.Tensor) or context.dtype != DTYPE or not torch.isfinite(context).all():
            raise ValueError("fixed preview text must be a finite native-bf16 tensor")
        context = context.to(device=device)
    session = Session(specification, device, "onestep_avatar.evaluate", context)
    negative_context, guider = None, None
    if args.cfg != 1.0 or args.stg != 0 or args.rescale != 0:
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams  # noqa: PLC0415
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT  # noqa: PLC0415

        if args.cfg == 1.0:
            negative_context = None
        elif fixed is not None:
            negative_context = torch.load(Path(fixed['input_files']['negative_text']['path']), map_location=device,
                                          weights_only=True)
            if (not isinstance(negative_context, torch.Tensor) or negative_context.dtype != DTYPE
                    or not torch.isfinite(negative_context).all()):
                raise ValueError('fixed preview negative text must be a finite native-bf16 tensor')
        else:
            negative_context = prompt_cache.get_or_build(
                specification,
                DEFAULT_NEGATIVE_PROMPT if args.negative_prompt is None else args.negative_prompt,
                DTYPE,
                device,
            )
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=args.cfg, stg_scale=args.stg, stg_blocks=args.stg_blocks, rescale_scale=args.rescale
            ),
            negative_context=negative_context,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    saved_context = context.cpu()
    atomic_write(args.output / "text.pt", lambda temporary: torch.save(saved_context, temporary))
    if negative_context is not None:
        saved_negative = negative_context.cpu()
        atomic_write(args.output / "negative_text.pt", lambda temporary: torch.save(saved_negative, temporary))
    for case_index, (video, frames, requested, adapters) in enumerate(cases):
        grid = common.ClipGrid.build(
            frames,
            video.z_y.shape[2] * specification.scale_factors.height,
            video.z_y.shape[3] * specification.scale_factors.width,
            video.fps,
            specification,
            device=device,
            dtype=DTYPE,
            latent_channels=specification.caps.latent_channels,
        )
        capture = grid.patchify(video.z_y[:, :frames].unsqueeze(0).to(device=device, dtype=DTYPE))
        guide = (
            None
            if video.z_g is None
            else grid.patchify(video.z_g[:, :frames].unsqueeze(0).to(device=device, dtype=DTYPE))
        )
        epsilon = (
            common.epsilon_block(capture, args.seed) if args.saved_noise is None else args.saved_noise.to(device=device)
        )
        if fixed is not None:
            preview_tensor_validator(
                fixed,
                {
                    "capture": capture,
                    "guide": guide,
                    "first_image": capture[:, : grid.tokens_per_latent_frame],
                    "text": context,
                    "negative_text": negative_context,
                    "noise": epsilon,
                },
            )
        destination = args.output / f"case_{case_index:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        saved_noise = epsilon.cpu()
        atomic_write(
            destination / "noise.pt", lambda temporary, saved_noise=saved_noise: torch.save(saved_noise, temporary)
        )
        for index, checkpoint in enumerate(variants):
            software.check_current(producer_software)
            if checkpoint is not None:
                checkpoints.recheck_adapter(checkpoint, adapters[index]["contract"],
                                            adapters[index]["adapter_sha256"])
            with adapter_loader.inference_transformer(
                session, checkpoint, adapters[index].get("contract"), method=args.adapter_application,
                adapter_sha256=adapters[index].get("adapter_sha256")
            ) as transformer:
                sampler = sample_case if sample_runner is None else sample_runner
                result = sampler(
                    transformer,
                    context,
                    grid,
                    capture,
                    guide,
                    epsilon,
                    mode=args.mode,
                    mode_settings=args.mode_settings,
                    guide_mode=args.guide_mode,
                    schedule=args.schedule,
                    seed=args.seed,
                    history_mode=args.history_mode or "cache",
                    kv_source=args.kv_source or "refresh",
                    predict_x0=(
                        None
                        if guider is None
                        else common.guided_denoised_from_x0_model(transformer, guider, negative_context)
                    ),
                )
            # The context manager cannot release the caller's retained reference.
            # Drop it before loading the next full resident checkpoint.
            del transformer
            output, record = result
            record = dict(record)
            record.update(
                conditions=requested,
                source=video.source,
                fps=video.fps,
                membership_sha256=membership["sha256"],
                input_file_hashes=video.hashes,
                prompt=prompt,
                negative_prompt=(
                    (DEFAULT_NEGATIVE_PROMPT if args.negative_prompt is None else args.negative_prompt)
                    if negative_context is not None else None
                ),
                producer_source_sha256=producer_source,
                software=producer_software,
                guidance={
                    "cfg": args.cfg,
                    "stg": args.stg,
                    "stg_blocks": args.stg_blocks,
                    "rescale": args.rescale,
                    "negative_text_sha256": (
                        None if negative_context is None else hashing.tensor_sha256(negative_context)
                    ),
                },
                **adapters[index],
            )
            variant_output = destination / f"variant_{index:03d}"
            if sha256(Path(__file__)) != producer_source:
                raise ValueError("evaluation producer source changed before result publication")
            software.check_current(producer_software)
            publisher = save_case if result_publisher is None else result_publisher
            publisher(output, record, variant_output)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Execute only ordinary explicitly selected evaluation."""
    return execute_evaluation(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
