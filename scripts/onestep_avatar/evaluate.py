"""Check and execute same-input comparisons; see doc/evaluate.md."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters as adapter_loader
from scripts.onestep_avatar.model import backbone, bidirectional, causal, common
from scripts.onestep_avatar.model.sampling import validate_schedule
from scripts.onestep_avatar.training import checkpoints
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings
from scripts.onestep_avatar import hashing, metrics






def saved_latent_metrics(
    output: torch.Tensor, capture: torch.Tensor, guide: torch.Tensor, *, long: bool = False
) -> dict:
    """Preserve historical generated-frame measurements separately from training loss."""
    if (
        output.ndim != 4
        or output.shape != capture.shape
        or output.shape != guide.shape
        or output.shape[1] < 3
        or output.shape[1] % 2 != 1
        or min(output.shape[2:]) < 2
        or any(not torch.isfinite(value).all() for value in (output, capture, guide))
    ):
        raise ValueError("saved metrics require finite matching C,F,H,W complete two-frame blocks")
    output, capture, guide = (value.float() for value in (output, capture, guide))
    frames = output.shape[1]
    if not long and frames != 17:
        raise ValueError("short saved metrics require exactly 17 encoded frames")
    blocks = [(1, 3)] + [(start, start + 2) for start in range(3, frames, 2)]

    def detail(value: torch.Tensor) -> float:
        return float(
            (value[:, :, 1:] - value[:, :, :-1]).abs().mean() + (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean()
        )

    def ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            raise ValueError("saved metric ratio has a zero denominator")
        return numerator / denominator

    result = {
        "c0_exact": bool(torch.equal(output[:, 0], capture[:, 0])),
        "per_block_mse": [float((output[:, a:b] - capture[:, a:b]).square().mean()) for a, b in blocks],
    }
    if long:
        return {
            **result,
            "latent_frames": frames,
            "per_block_guide_mse": [float((guide[:, a:b] - capture[:, a:b]).square().mean()) for a, b in blocks],
            "per_block_detail_ratio": [ratio(detail(output[:, a:b]), detail(capture[:, a:b])) for a, b in blocks],
        }
    boundaries = [end for _, end in blocks[:-1]]

    def seam(value: torch.Tensor) -> float:
        steps = (value[:, 1:] - value[:, :-1]).square().mean(dim=(0, 2, 3))
        across = torch.stack([steps[end - 1] for end in boundaries]).mean()
        inside = torch.stack([steps[t - 1] for t in range(2, frames) if t not in boundaries]).mean()
        if float(inside) == 0:
            raise ValueError("saved metric ratio has a zero denominator")
        return float(across / inside)

    def motion(value: torch.Tensor) -> float:
        return float((value[:, 2:] - value[:, 1:-1]).abs().mean())

    return {
        **result,
        "capture_mse": float((output[:, 1:] - capture[:, 1:]).square().mean()),
        "guide_mse": float((guide[:, 1:] - capture[:, 1:]).square().mean()),
        "motion_ratio": ratio(motion(output), motion(capture)),
        "detail_ratio": ratio(detail(output[:, 1:]), detail(capture[:, 1:])),
        "guide_detail_ratio": ratio(detail(guide[:, 1:]), detail(capture[:, 1:])),
        "seam_ratio": seam(output),
        "capture_seam_ratio": seam(capture),
    }


def measure_saved_probe(directory: Path, *, long: bool = False) -> dict:
    """Measure verified saved historical encodings, without any model session."""
    manifest = json.loads((directory / "manifest.json").read_text())
    rows, seen = [], set()
    for video in manifest["videos"]:
        artifacts = video["artifacts"]
        seed = artifacts.get("seed", manifest.get("seed"))
        key = (artifacts["view"], seed)
        if key in seen:
            continue
        seen.add(key)
        capture, _ = dataset.load_training_master(Path(artifacts["capture"]))
        guide, _ = dataset.load_training_master(Path(artifacts["guide"]))
        for latent in artifacts["latents"]:
            path = directory / latent["path"]
            if sha256(path) != latent["sha256"]:
                raise ValueError(f"saved encoding content changed: {path}")
            output = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(output, torch.Tensor) or output.ndim != 5 or output.shape[0] != 1:
                raise ValueError("saved output must be a single B,C,F,H,W tensor")
            frames = output.shape[2]
            view = Path(artifacts["view"])
            row = {
                "view": f"{view.parent.parent.parent.name}/{view.parent.parent.name}/{view.name}",
                "seed": seed,
                **saved_latent_metrics(output[0], capture[:, :frames], guide[:, :frames], long=long),
            }
            if not long:
                row.update(
                    sigma=latent["sigma"],
                    arm=latent["arm"],
                    latent_sha256=latent["sha256"],
                    epsilon_sha256=artifacts["epsilon_sha256"],
                )
            rows.append(row)
    result = {"probe": str(directory), "checkpoint": manifest.get("checkpoint"), "rows": rows}
    if not long:
        result.update(
            off_condition=manifest.get("off_condition", False),
            model_variant=manifest.get("model_variant"),
            schedule=manifest["videos"][0]["schedule"] if manifest["videos"] else None,
        )
    atomic_write(
        directory / ("metrics_long.json" if long else "metrics.json"),
        lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"),
    )
    return result








def sigma_sweep_boundary_metrics(video, capture, guide, mask) -> dict:  # noqa: ANN001 -- historical NumPy RGB arrays
    """Preserve historical fixed-129-frame boundary scores, with explicit undefined ratios."""
    import numpy as np  # noqa: PLC0415 -- historical metric arithmetic

    arrays = (video, capture, guide)
    if (any(not isinstance(value, np.ndarray) for value in arrays)
            or video.ndim != 4 or video.shape[0] != 129 or video.shape[-1] != 3
            or min(video.shape[1:3]) < 1 or any(value.shape != video.shape for value in arrays)):
        raise ValueError("sigma sweep requires aligned nonempty 129,H,W,3 RGB arrays")
    if any(not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()
           or value.min() < 0 or value.max() > 1 for value in arrays):
        raise ValueError("sigma sweep requires finite floating pixels in [0,1]")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != video.shape[:3]:
        raise ValueError("sigma sweep requires an aligned boolean foreground mask")
    video, capture, guide = [value.astype(np.float32, copy=False) if value.dtype.itemsize < 4 else value
                             for value in arrays]
    boundaries, post_eviction = (17, 33, 49, 65, 81, 97, 113), {81, 97, 113}
    indices = {boundary - 1 for boundary in boundaries}

    def errors(left, right, selected):  # noqa: ANN001, ANN202 -- checked NumPy inputs
        delta = np.abs(left - right).mean(-1)
        return (delta * selected).sum((1, 2)) / selected.sum((1, 2)).clip(1)

    def ratio(numerator: float, denominator: float) -> float | None:
        if denominator == 0:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(numerator / denominator)
        if not math.isfinite(result):
            raise ValueError("sigma sweep ratio is nonfinite")
        return result

    def mean(values: list) -> float | None:
        return None if any(value is None for value in values) else float(np.mean(values))

    motion, capture_motion = (
        metrics.masked_rgb_transition_steps(video, mask),
        metrics.masked_rgb_transition_steps(capture, mask),
    )
    rows = []
    for boundary in boundaries:
        index = boundary - 1
        local = [motion[j] for j in range(index - 4, index + 5) if j != index and j not in indices]
        local_capture = np.delete(capture_motion[index - 4:index + 5], 4)
        local_ratio = ratio(motion[index], np.mean(local))
        capture_ratio = ratio(capture_motion[index], local_capture.mean())
        rows.append({"boundary": boundary, "post_eviction": boundary in post_eviction,
                     "step": float(motion[index]), "ratio_to_local_interior": local_ratio,
                     "capture_ratio": capture_ratio,
                     "ratio_status": "undefined_zero_local_motion" if local_ratio is None else "defined",
                     "capture_ratio_status": "undefined_zero_local_motion" if capture_ratio is None else "defined"})
    motion_ratio = ratio(motion[16:].mean(), capture_motion[16:].mean())
    return {
        "per_boundary": rows,
        "mean_boundary_ratio": mean([row["ratio_to_local_interior"] for row in rows]),
        "mean_boundary_ratio_pre_eviction": mean([row["ratio_to_local_interior"] for row in rows
                                                 if not row["post_eviction"]]),
        "mean_boundary_ratio_post_eviction": mean([row["ratio_to_local_interior"] for row in rows
                                                  if row["post_eviction"]]),
        "interior_step": float(np.mean([motion[i] for i in range(16, 128) if i not in indices])),
        "motion_over_capture": motion_ratio,
        "motion_ratio_status": "undefined_zero_capture_motion" if motion_ratio is None else "defined",
        "err_vs_capture": float(errors(video[1:], capture[1:], mask[1:]).mean()),
        "err_vs_guide": float(errors(video[1:], guide[1:], mask[1:]).mean()),
        "drift_vs_c0_last": float(errors(video[128:], video[:1], mask[128:])[0]),
        "per_frame_err_vs_capture": errors(video, capture, mask).round(4).tolist(),
        "per_transition_step": motion.round(4).tolist(),
    }












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
    if not torch.equal(noise[:, :boundary], changed_noise[:, :boundary]):
        raise ValueError("future-noise diagnostic changed earlier noise")
    if torch.equal(noise[:, boundary:], changed_noise[:, boundary:]):
        raise ValueError("future-noise diagnostic requires changed later noise")
    outputs, records = [], []
    for epsilon in (noise, changed_noise):
        output, record = sample_case(transformer, context, grid, capture, guide, epsilon, **settings)
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


def save_future_noise_probe(outputs: list[torch.Tensor], diagnostic: dict, provenance: dict, destination: Path) -> dict:
    """Publish both raw results before a diagnostic, preserving their distinct noise hashes."""
    if len(outputs) != 2 or len(diagnostic.get("records", [])) != 2:
        raise ValueError("future-noise publication requires two outputs and records")
    saved = [
        save_case(encoded, {**raw, **provenance}, destination / name)
        for encoded, raw, name in zip(outputs, diagnostic["records"], ("original", "changed"), strict=True)
    ]
    completed = {**diagnostic, "records": saved}
    atomic_write(
        destination / "future_noise.json",
        lambda temporary: temporary.write_text(json.dumps(completed, indent=2) + "\n"),
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
    parser.add_argument("--changed-noise-file", type=Path)
    parser.add_argument("--future-noise-start", type=int)
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
    if (args.changed_noise_file is None) != (args.future_noise_start is None):
        parser.error("future-noise probe requires both changed noise and an encoded boundary")
    if args.changed_noise_file is not None and (args.noise_file is None or args.future_noise_start < 1):
        parser.error("future-noise probe requires saved original noise and a positive boundary")
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
    args.changed_noise = (
        None
        if args.changed_noise_file is None
        else torch.load(args.changed_noise_file, map_location="cpu", weights_only=True)
    )
    if args.changed_noise is not None and (
        not isinstance(args.changed_noise, torch.Tensor)
        or args.changed_noise.dtype != torch.bfloat16
        or not torch.isfinite(args.changed_noise).all()
    ):
        raise ValueError("changed comparison noise must be a finite native-bf16 tensor")
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
        if args.changed_noise is not None:
            start = args.future_noise_start
            boundary = start * video.z_y.shape[2] * video.z_y.shape[3]
            if not 0 < start < frames or tuple(args.changed_noise.shape) != expected_noise:
                raise ValueError("future-noise boundary/shape differs from selected input")
            if args.mode == "causal" and start not in [end for _, end in plan[:-1]]:
                raise ValueError("causal future-noise boundary must separate completed blocks")
            if not torch.equal(saved_noise[:, :boundary], args.changed_noise[:, :boundary]):
                raise ValueError("future-noise diagnostic changed earlier noise")
            if torch.equal(saved_noise[:, boundary:], args.changed_noise[:, boundary:]):
                raise ValueError("future-noise diagnostic requires changed later noise")
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


def verify_evaluation_conditions(arguments: list[str], record_paths: list[Path]) -> None:  # noqa: PLR0912, PLR0915 -- all scientific evidence must agree
    """Verify queued results against current inputs, without opening model sessions."""
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT  # noqa: PLC0415
    from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE  # noqa: PLC0415 -- constants only

    args = parse_args(arguments)
    specification, variants, cases, membership = prepare_evaluation(args, require_fresh_output=False)
    expected_paths = []
    for case_index in range(len(cases)):
        for variant_index in range(len(variants)):
            destination = args.output / f"case_{case_index:04d}" / f"variant_{variant_index:03d}"
            expected_paths.extend(
                [destination / "result.json"] if args.changed_noise_file is None
                else [destination / branch / "result.json" for branch in ("original", "changed")]
            )
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
        changed = None if args.changed_noise is None else tensor(case_dir / "changed_noise.pt")
        if changed is not None and not torch.equal(changed, args.changed_noise):
            raise ValueError("queue evaluation saved changed noise differs from requested input")
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
            "software": software.capture("evaluation", args.mode),
        }
        if args.mode == "causal":
            fixed.update(history_mode=args.history_mode, kv_source=args.kv_source)
        for variant_index, adapter in enumerate(adapters):
            destination = case_dir / f"variant_{variant_index:03d}"
            branch_records, predictions = [], []
            branches = [(destination, noise)] if changed is None else [
                (destination / "original", noise), (destination / "changed", changed)]
            for folder, epsilon in branches:
                record = json.loads((folder / "result.json").read_text())
                software.check_current(record.get("software"))
                expected = {**fixed, **adapter, "noise_sha256": hashing.tensor_sha256(epsilon)}
                if any(key not in record or record[key] != value for key, value in expected.items()):
                    raise ValueError("queue evaluation scientific settings or input evidence differ")
                encoded = folder / "generated.pt"
                if Path(record.get("output", {}).get("path", "")).resolve() != encoded.resolve():
                    raise ValueError("queue evaluation encoding path differs from requested variant")
                prediction = tensor(encoded)
                if list(prediction.shape) != [1, *video.z_y[:, :frames].shape]:
                    raise ValueError("queue evaluation encoding shape differs from requested video")
                if not torch.equal(prediction[:, :, :1], video.z_y[:, :1].unsqueeze(0).to(dtype=DTYPE)):
                    raise ValueError("queue evaluation changed clean first-image input")
                branch_records.append(record)
                predictions.append(prediction)
            if changed is not None:
                diagnostic = json.loads((destination / "future_noise.json").read_text())
                boundary = args.future_noise_start
                left, right = (value[:, :, :boundary] for value in predictions)
                expected_diagnostic = {
                    "change_start_encoded_frame": boundary, "records": branch_records,
                    "earlier_output_bit_identical": torch.equal(left, right),
                    "earlier_output_max_abs_delta": float((left.float()-right.float()).abs().max()),
                    "later_output_max_abs_delta": float((predictions[0][:, :, boundary:].float()
                                                         -predictions[1][:, :, boundary:].float()).abs().max()),
                }
                if any(key not in diagnostic or diagnostic[key] != value
                       for key, value in expected_diagnostic.items()):
                    raise ValueError("queue evaluation future-noise diagnostic differs from saved results")


def evaluation_evidence_paths(arguments: list[str], record_paths: list[Path]) -> list[Path]:
    """Inventory artifacts after scientific verification, for byte-bound receipts."""
    args = parse_args(arguments)
    paths = set(record_paths)
    paths.add(args.output / "text.pt")
    if args.cfg != 1.0:
        paths.add(args.output / "negative_text.pt")
    for record in record_paths:
        paths.add(record.parent / "generated.pt")
        variant = record.parent if args.changed_noise_file is None else record.parent.parent
        paths.add(variant.parent / "noise.pt")
        if args.changed_noise_file is not None:
            paths.update((variant.parent / "changed_noise.pt", variant / "future_noise.json"))
    return sorted(paths)






























def execute_evaluation(  # noqa: PLR0912, PLR0915 -- explicit sampler and native session orchestration
    args: argparse.Namespace, sample_runner: Callable[..., tuple[torch.Tensor, dict]] | None = None, *,
    preview_tensor_validator: Callable[[dict, dict[str, torch.Tensor | None]], None] | None = None,
) -> int:
    """Execute ordinary sampling; a fixed preview requires its canonical tensor validator."""
    if getattr(args, "preview_fixed", None) is not None and not callable(preview_tensor_validator):
        raise ValueError("fixed preview requires its explicit preview tensor validator")
    if sample_runner is not None and args.changed_noise_file is not None:
        raise ValueError("future-noise probes reject a custom sample runner")
    producer_source = sha256(Path(__file__))
    producer_software = software.capture("evaluation", args.mode)
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
        changed_noise = None if args.changed_noise is None else args.changed_noise.to(device=device)
        if changed_noise is not None:
            atomic_write(destination / "changed_noise.pt", lambda temporary: torch.save(args.changed_noise, temporary))
        for index, checkpoint in enumerate(variants):
            software.check_current(producer_software)
            if checkpoint is not None:
                checkpoints.recheck_adapter(checkpoint, adapters[index]["contract"],
                                            adapters[index]["adapter_sha256"])
            with adapter_loader.inference_transformer(
                session, checkpoint, adapters[index].get("contract"), method=args.adapter_application,
                adapter_sha256=adapters[index].get("adapter_sha256")
            ) as transformer:
                sampler = (
                    (sample_case if sample_runner is None else sample_runner)
                    if changed_noise is None
                    else probe_future_noise
                )
                result = sampler(
                    transformer,
                    context,
                    grid,
                    capture,
                    guide,
                    epsilon,
                    *((changed_noise,) if changed_noise is not None else ()),
                    **({"change_start_frame": args.future_noise_start} if changed_noise is not None else {}),
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
            output, record = result if changed_noise is None else (result[0][0], result[1]["records"][0])
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
            if changed_noise is None:
                save_case(output, record, variant_output)
            else:
                outputs, diagnostic = result
                common_record = {key: value for key, value in record.items() if key not in diagnostic["records"][0]}
                save_future_noise_probe(outputs, diagnostic, common_record, variant_output)
    return 0


def fusion_probe_block(transformer, kind: str, session, capture: torch.Tensor, guide: torch.Tensor, fps: float) -> torch.Tensor:  # noqa: ANN001, E501 -- native external model/session handles
    """Preserve the historical diagnostic's one empty-cache block, input and seed."""
    from scripts.prune.core.session import DTYPE  # noqa: PLC0415 -- native training dtype

    geometry = causal.deployed_geometry(session.model.scale_factors)
    _, frames, height, width = capture.shape
    grid = common.ClipGrid.build(
        frames, height * geometry.scale_factors.height, width * geometry.scale_factors.width,
        fps, geometry, device=session.device, dtype=DTYPE, latent_channels=session.model.caps.latent_channels,
    )
    span = geometry.plan(frames)[0]
    lo, hi = grid.token_span(*span)
    target = grid.patchify(capture.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    source = grid.patchify(guide.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    c0 = target[:, :grid.tokens_per_latent_frame]
    inner = common.base_model(transformer)
    cache = causal.BlockCache.allocate(
        grid, geometry, num_layers=len(inner.transformer_blocks), inner_dim=inner.inner_dim,
        device=session.device, dtype=DTYPE,
    )
    noisy = common.with_clean_prefix(common.noise_block(source[:, lo:hi], 0.421875, 42), c0)
    return causal.fusion_parity_block(
        transformer, kind, grid, cache, noisy, session.context, 0.421875, span,
        clean_prefix_tokens=c0.shape[1],
    ).float().cpu()


def fusion_parity_metrics(outputs: dict[str, torch.Tensor]) -> dict:
    """Compare raw outputs and adapter effects without hiding undefined ratios."""
    if set(outputs) != {"bare", "step0", "fused1", "peft0", "peft1"}:
        raise ValueError("fusion parity requires all five model cases")
    reference = outputs["bare"]
    if reference.numel() == 0 or any(
        value.shape != reference.shape or not torch.isfinite(value).all() for value in outputs.values()
    ):
        raise ValueError("fusion parity outputs must have matching shapes and finite values")

    def relative(delta: torch.Tensor, base: torch.Tensor) -> float | None:
        denominator = float(base.norm())
        return None if denominator == 0 else float(delta.norm() / base.norm())

    effect_peft = outputs["peft1"] - outputs["peft0"]
    effect_fused = outputs["fused1"] - reference
    gap = relative(effect_fused - effect_peft, effect_peft)
    return {
        "step0_equals_bare_bitwise": bool(torch.equal(outputs["step0"], reference)),
        "rel_l2_step0_vs_bare": relative(outputs["step0"] - reference, reference),
        "rel_l2_training_path_vs_probe_path_no_adapter": relative(outputs["peft0"] - reference, reference),
        "rel_l2_adapter_effect_peft": relative(effect_peft, outputs["peft0"]),
        "rel_l2_adapter_effect_fused": relative(effect_fused, reference),
        "rel_l2_effect_fused_vs_effect_peft": gap,
        "rel_l2_fused1_vs_peft1": relative(outputs["fused1"] - outputs["peft1"], outputs["peft1"]),
        "tolerance_rel_l2_effect": 0.2,
        "fused_vs_peft_within_tolerance": gap is not None and gap <= 0.2,
        "effect_ratio_status": "undefined_zero_peft_effect" if gap is None else "defined",
    }


def evaluate_fusion_parity(run: Path, view: Path, output: Path, *, gpu_id: int, step: int = 1) -> dict:
    """Own the five historical fusion cases and publish only completed diagnostic results."""
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: PLC0415
    from ltx_trainer.model_loader import load_transformer  # noqa: PLC0415
    from scripts.onestep_avatar.model.adapters import LORA_TARGETS  # noqa: PLC0415
    from scripts.prune.core.session import DTYPE, open_session  # noqa: PLC0415

    if type(step) is not int or step < 1 or output.exists():
        raise ValueError("fusion diagnostic requires a positive trained step and fresh output")
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    if (config.get("lora_target") not in LORA_TARGETS
            or type(config.get("lora_rank")) is not int or config["lora_rank"] < 1):
        raise ValueError("fusion diagnostic requires a valid saved LoRA configuration")
    alpha = config.get("lora_alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("fusion diagnostic requires a positive saved LoRA alpha")
    if alpha != config["lora_rank"]:
        raise ValueError("fusion diagnostic requires alpha equal to rank for the fused path")
    dev = backbone.transformer_path("2.5", "dev")
    step0 = run / "checkpoints/lora_weights_step_00000.safetensors"
    trained = run / f"checkpoints/lora_weights_step_{step:05d}.safetensors"
    capture_path, guide_path = view / dataset.capture_bundle_name("white"), view / dataset.guide_bundle_name("white")
    paths = [config_path, dev, step0, trained, capture_path, guide_path]
    if any(not path.is_file() for path in paths):
        raise ValueError("fusion diagnostic input file is missing")
    capture, fps = dataset.load_training_master(capture_path)
    guide, guide_fps = dataset.load_training_master(guide_path)
    if capture.shape != guide.shape or fps != guide_fps or capture.shape[1] < 3:
        raise ValueError("fusion diagnostic requires matching capture/guide geometry, fps and block-zero coverage")
    identities = {str(path.resolve()): sha256(path) for path in paths}
    session = open_session(argparse.Namespace(model="2.5", gpu_id=gpu_id, seed=42),
                           script="onestep_avatar.evaluate.fusion_parity", transformer_path=dev)
    outputs = {}
    for name, adapters in (("bare", ()), ("step0", (step0,)), ("fused1", (trained,))):
        loras = tuple(LoraPathStrengthAndSDOps(str(path), 1.0, LTXV_LORA_COMFY_RENAMING_MAP) for path in adapters)
        with session.transformer(dev, loras=loras) as transformer:
            outputs[name] = fusion_probe_block(transformer, "x0", session, capture, guide, fps)
        del transformer
        torch.cuda.empty_cache()
    model = load_transformer(checkpoint_path=str(dev), device=session.device, dtype=DTYPE, video_only=True)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=alpha,
                           target_modules=LORA_TARGETS[config["lora_target"]], lora_dropout=0.0))
    model.eval()
    for name, path in (("peft0", step0), ("peft1", trained)):
        checkpoints.load_stage_init(model, path)
        outputs[name] = fusion_probe_block(model, "velocity", session, capture, guide, fps)
    del model
    torch.cuda.empty_cache()
    if any(sha256(path) != identities[str(path.resolve())] for path in paths):
        raise ValueError("fusion diagnostic inputs changed during execution")
    result = {"view": str(view), "trained_step": step, "sigma": 0.421875, "block": 0, "noise_seed": 42,
              **fusion_parity_metrics(outputs), "input_sha256": identities}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"))
    return result


@torch.no_grad()
def causality_probe(transformer, session, capture: torch.Tensor, guide: torch.Tensor, fps: float, sigma: float) -> dict:  # noqa: ANN001 -- native model/session handles
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
        frames, height * geometry.scale_factors.height, width * geometry.scale_factors.width,
        fps, geometry, device=session.device, dtype=DTYPE, latent_channels=session.model.caps.latent_channels,
    )
    plan = geometry.plan(frames)[:8]
    target = grid.patchify(capture.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    source = grid.patchify(guide.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    original = common.epsilon_block(target, 42)
    alternate = common.epsilon_block(target, 99)
    first = plan[3][1]
    boundary = first * grid.tokens_per_latent_frame
    mixed = torch.cat((original[:, :boundary], alternate[:, boundary:]), dim=1)
    outputs, records = [], []
    for epsilon in (original, mixed):
        with measure_calls(transformer) as measured:
            tokens, counts = causal.sample(
                common.denoised_from_x0_model(transformer), session.context, grid, source,
                target[:, :grid.tokens_per_latent_frame], transformer=transformer,
                geometry=geometry, schedule=levels, seed=42, epsilon=epsilon, blocks=plan,
                teacher_forcing=False, history_mode="cache", kv_source="refresh",
            )
        covered = plan[-1][1]
        outputs.append(grid.unpatchify_block(tokens[:, :covered * grid.tokens_per_latent_frame], covered).float().cpu())
        if not torch.isfinite(outputs[-1]).all():
            raise ValueError("causality diagnostic produced nonfinite output")
        records.append({"noise_sha256": hashing.tensor_sha256(epsilon), "call_counts": {**counts, **measured}})
    left, right = outputs
    later_delta = float((left[:, :, first:] - right[:, :, first:]).abs().max())
    return {
        "shared_noise_blocks": [0, 1, 2, 3], "changed_noise_blocks": [4, 5, 6, 7],
        "latent_frames_compared_equal": [0, first],
        "earlier_blocks_bit_identical": bool(torch.equal(left[:, :, :first], right[:, :, :first])),
        "later_blocks_max_abs_diff": later_delta, "later_blocks_changed": later_delta > 0,
        "records": records, "capture_sha256": hashing.tensor_sha256(target),
        "guide_sha256": hashing.tensor_sha256(source), "text_sha256": hashing.tensor_sha256(session.context),
    }


def evaluate_causality(checkpoint: Path, view: Path, output: Path, *, gpu_id: int, sigma: float) -> dict:
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
        meta, base=backbone.identity(dev, "dev", "2.5"), objective="white", guide_mode="d1",
        schedule=[sigma, 0.0], geometry={"block_latent_frames": 2, "context_latent_frames": 8,
                                       "sink_latent_frames": 1}, teacher_forcing=False,
    )
    identities = {str(path.resolve()): sha256(path) for path in paths}
    session = open_session(argparse.Namespace(model="2.5", gpu_id=gpu_id, seed=42),
                           script="onestep_avatar.evaluate.causality", transformer_path=dev)
    loras = (LoraPathStrengthAndSDOps(str(checkpoint), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)
    with session.transformer(dev, loras=loras) as transformer:
        diagnostic = causality_probe(transformer, session, capture, guide, fps, sigma)
    del transformer
    torch.cuda.empty_cache()
    if any(sha256(path) != identities[str(path.resolve())] for path in paths):
        raise ValueError("causality diagnostic inputs changed during execution")
    result = {"checkpoint": str(checkpoint), "view": str(view), "sigma": sigma,
              **diagnostic, "input_sha256": identities}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"))
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if "--causality" in arguments:
        parser = argparse.ArgumentParser(description="Compare eight D1 blocks with changed later noise")
        parser.add_argument("--causality", action="store_true", required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--view", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--gpu-id", type=int, required=True)
        parser.add_argument("--sigma", type=float, required=True)
        args = parser.parse_args(arguments)
        evaluate_causality(args.checkpoint, args.view, args.output, gpu_id=args.gpu_id, sigma=args.sigma)
        return 0
    if "--fusion-parity" in arguments:
        parser = argparse.ArgumentParser(description="Compare PEFT and fused adapter effects on D1 block zero")
        parser.add_argument("--fusion-parity", action="store_true", required=True)
        parser.add_argument("--run", type=Path, required=True)
        parser.add_argument("--view", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--gpu-id", type=int, required=True)
        parser.add_argument("--step", type=int, default=1)
        args = parser.parse_args(arguments)
        evaluate_fusion_parity(args.run, args.view, args.output, gpu_id=args.gpu_id, step=args.step)
        return 0
    if "--saved-metrics" in arguments:
        parser = argparse.ArgumentParser(description="Measure saved encodings without loading models")
        parser.add_argument("--saved-metrics", nargs="+", type=Path, required=True)
        parser.add_argument("--long-metrics", action="store_true")
        args = parser.parse_args(arguments)
        for directory in args.saved_metrics:
            measure_saved_probe(directory, long=args.long_metrics)
        return 0
    return execute_evaluation(parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
