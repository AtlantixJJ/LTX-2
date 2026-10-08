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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from scripts.onestep_avatar import dataset, software, subset
from scripts.onestep_avatar.dataset import atomic_write
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters as adapter_loader
from scripts.onestep_avatar.model import backbone, bidirectional, causal, common
from scripts.onestep_avatar.model.sampling import validate_schedule
from scripts.onestep_avatar.training import checkpoints
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash shape, dtype and original tensor bytes, independently of serialization."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encoded_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Full-frame fp32 x0 MSE, with unchanged c0 retained in the denominator."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("encoded metrics require equal B,C,F,H,W tensors")
    error = (prediction.float() - target.float()).square()
    return {
        "definition": "mean((prediction-capture)^2) in fp32, including c0",
        "mse": float(error.mean()),
        "per_frame_mse": error.mean(dim=(0, 1, 3, 4)).tolist(),
        "frames": prediction.shape[2],
    }


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


def rgb_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Aligned unquantized FCHW float RGB measurements, without a synthetic mask."""
    _check_rgb_pair(prediction, target)
    error = (prediction.float() - target.float()).square()
    mse = error.mean(dim=(1, 2, 3)).tolist()
    total_mse = float(error.mean())
    return {
        "definition": "float RGB MSE and PSNR=-10*log10(MSE), before presentation compression",
        "per_frame_mse": mse,
        "per_frame_psnr": [None if x == 0 else -10 * math.log10(x) for x in mse],
        "exact_match": [x == 0 for x in mse],
        "frames": len(mse),
        "mse": total_mse,
        "psnr": None if total_mse == 0 else -10 * math.log10(total_mse),
        "all_exact_match": total_mse == 0,
    }


def _check_rgb_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if (
        prediction.shape != target.shape
        or prediction.ndim != 4
        or prediction.shape[1] != 3
        or any(size < 1 for size in prediction.shape)
    ):
        raise ValueError("RGB measurement requires matching nonempty F,3,H,W tensors")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1
        for value in (prediction, target)
    ):
        raise ValueError("RGB measurement requires finite floating pixels in [0,1]")


def masked_rgb_transition_steps(video, mask):  # noqa: ANN001, ANN201 -- historical NumPy RGB arrays
    """Measure absolute frame changes in the supplied union-foreground mask."""
    import numpy as np  # noqa: PLC0415 -- historical metric arithmetic

    if (not isinstance(video, np.ndarray) or video.ndim != 4 or video.shape[0] < 2
            or video.shape[-1] != 3 or min(video.shape[1:3]) < 1
            or not np.issubdtype(video.dtype, np.floating) or not np.isfinite(video).all()
            or video.min() < 0 or video.max() > 1):
        raise ValueError("transition measurement requires finite nonempty F,H,W,3 RGB in [0,1]")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != video.shape[:3]:
        raise ValueError("transition measurement requires an aligned boolean foreground mask")
    value = video.astype(np.float32, copy=False) if video.dtype.itemsize < 4 else video
    delta = np.abs(value[1:] - value[:-1]).mean(-1)
    selected = mask[1:] | mask[:-1]
    return (delta * selected).sum((1, 2)) / selected.sum((1, 2)).clip(1)


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

    motion, capture_motion = masked_rgb_transition_steps(video, mask), masked_rgb_transition_steps(capture, mask)
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


def subject_mask(path: Path, frames: int, height: int, width: int) -> torch.Tensor | None:
    """Replay the study's optional two-cell mask dilation for RGB QA only."""
    from scripts.onestep_avatar import mask_video  # noqa: PLC0415 -- CPU lossless mask reader

    if any(type(value) is not int or value < 1 for value in (frames, height, width)):
        raise ValueError("subject mask requires positive frame count and dimensions")
    if not path.is_file():
        return None
    raw = torch.from_numpy(mask_video.read_mask_video(path))
    if raw.dtype != torch.uint8 or raw.ndim != 3 or raw.shape[0] < frames or any(size < 1 for size in raw.shape):
        raise ValueError("subject mask does not cover the requested RGB frames")
    grid = raw[:frames].float()[:, None] / 255
    grid = torch.nn.functional.max_pool2d(grid, 5, stride=1, padding=2)
    return torch.nn.functional.interpolate(grid, size=(height, width), mode="nearest")[:, 0] > 0.5


def subject_rgb_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """Measure aligned supplied subject pixels before presentation compression."""
    _check_rgb_pair(prediction, target)
    if mask.dtype != torch.bool or mask.shape != (prediction.shape[0], *prediction.shape[2:]) or not mask.any():
        raise ValueError("subject mask must be nonempty boolean F,H,W aligned with RGB")
    error = (prediction.float() - target.float()).square().mean(dim=1)
    mse = float(error[mask.to(error.device)].mean())
    return {
        "mse": mse,
        "psnr": None if mse == 0 else -10 * math.log10(mse),
        "exact_match": mse == 0,
        "selected_pixels": int(mask.sum()),
    }


def _lpips_batches(
    model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor, device: torch.device, batch: int
) -> Iterator[torch.Tensor]:
    """Share input/model gates while retaining native per-batch score tensors."""
    _check_rgb_pair(prediction, target)
    if type(batch) is not int or batch < 1:
        raise ValueError("perceptual batch size must be a positive integer")
    for start in range(0, len(prediction), batch):
        left = prediction[start : start + batch].float().to(device) * 2 - 1
        right = target[start : start + batch].float().to(device) * 2 - 1
        scores = model(left, right)
        if not isinstance(scores, torch.Tensor) or scores.numel() != len(left) or not torch.isfinite(scores).all():
            raise ValueError("perceptual model must return one finite score per frame")
        yield scores


@torch.no_grad()
def lpips_frame_scores(
    model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor, device: torch.device, batch: int = 16
) -> list[float]:
    """Return checked aligned scores; the caller chooses c0 exclusion and averaging."""
    return [value for scores in _lpips_batches(model, prediction, target, device, batch)
            for value in scores.flatten().tolist()]


@torch.no_grad()
def lpips_distance(
    model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor, device: torch.device, batch: int = 8
) -> float:
    """Preserve the scalar path's per-batch native sums and frame weighting."""
    total = 0.0
    for scores in _lpips_batches(model, prediction, target, device, batch):
        total += float(scores.sum())
    return total / len(prediction)


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
        "capture_sha256": tensor_sha256(capture),
        "guide_sha256": None if guide is None else tensor_sha256(guide),
        "c0_sha256": tensor_sha256(c0),
        "noise_sha256": tensor_sha256(epsilon),
        "text_sha256": tensor_sha256(context),
        "call_counts": counts,
        "elapsed_s": time.perf_counter() - started,
        "metrics": encoded_metrics(output, target),
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


def check_preview_reference_bundle(fixed: dict) -> dict | None:
    """Bind checked prepared RGB to the fixed capture and guide producers."""
    producer_inputs = fixed.get('producer_inputs', {})
    if not isinstance(producer_inputs, dict):
        raise ValueError('preview preparation input identities must be a mapping')
    for source in producer_inputs.values():
        if (not isinstance(source, dict) or not isinstance(source.get('path'), str)
                or not Path(source['path']).is_absolute() or sha256(Path(source['path'])) != source.get('sha256')):
            raise ValueError('preview preparation input changed or lacks an absolute identity')
    identity = fixed.get("reference_bundle")
    if identity is None:
        return None
    from scripts.onestep_avatar.media import load_training_references  # noqa: PLC0415 -- saved RGB only

    path = Path(identity["path"])
    if not path.is_absolute() or sha256(path) != identity["sha256"]:
        raise ValueError("preview reference bundle manifest changed")
    _, producer = load_training_references(path)
    if producer["capture_encoding_sha256"] != fixed["input_files"]["capture"]["sha256"]:
        raise ValueError("preview reference bundle uses a different capture encoding")
    args = parse_args([*fixed["evaluation_arguments"], "--output", "/unused-preview-reference-check"])
    if args.source != [producer["source"]]:
        raise ValueError("preview reference bundle requires its one explicit source selection")
    if args.guide_mode == "d1":
        if not producer.get("guide_rgb_sha256"):
            raise ValueError("D1 preview requires a checked guide reference")
        guide = torch.load(Path(fixed["input_files"]["guide"]["path"]), map_location="cpu", weights_only=True)
        if not isinstance(guide, dict) or guide.get("input_fingerprint") != producer["guide_rgb_sha256"]:
            raise ValueError("preview reference bundle uses a different guide render")
    return producer


def verify_preview_job(path: Path, *, verify_files: bool = True) -> dict:
    """Read only complete, unchanged checkpoints and pinned preview inputs."""
    job = json.loads(path.read_text())
    if job.get("schema_version") != 2 or job.get("kind") != "onestep_avatar.preview_job":
        raise ValueError("preview requires a version-two job record")
    fixed, checkpoint = job["fixed_inputs"], job["checkpoint"]
    if subset.record_hash(fixed) != fixed.get("sha256"):
        raise ValueError("fixed preview record changed")
    identity = hashlib.sha256((checkpoint["sha256"] + fixed["sha256"]).encode()).hexdigest()
    if job.get("id") != identity:
        raise ValueError("preview job identity changed")
    if not verify_files:
        return job
    for role, source in fixed["input_files"].items():
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"preview {role} file changed")
    check_preview_reference_bundle(fixed)
    adapter = Path(checkpoint["path"])
    marker = json.loads(adapter.with_suffix(".complete.json").read_text())
    if marker.get("state") != "complete" or sha256(adapter) != checkpoint["sha256"]:
        raise ValueError("preview checkpoint is incomplete or changed")
    if marker.get("sha256") != checkpoint["sha256"] or marker.get("step") != checkpoint["step"]:
        raise ValueError("preview completion marker differs from its pinned checkpoint")
    contract = checkpoints.read_contract(adapter)
    checkpoints.validate_adapter_tensors(adapter, contract)
    if contract["adapter"]["step"] != checkpoint["step"] or contract["mode"] != fixed["mode"]:
        raise ValueError("preview adapter step or mode differs from its job")
    return job


def _verify_preview_outputs(records: list[dict], job: dict, *, rendered: bool) -> None:  # noqa: PLR0912 -- all completion evidence gates
    if not records:
        raise ValueError("preview completion requires raw results and rendered outputs")
    for identity in records:
        path = Path(identity["path"])
        if sha256(path) != identity["sha256"]:
            raise ValueError("preview output record changed")
        record = json.loads(path.read_text())
        software.check_current(record.get("software"))
        if rendered:
            common_settings = record.get("common_settings", {})
            if (
                common_settings.get("preview_job_id") != job["id"]
                or common_settings.get("fixed_inputs_sha256") != job["fixed_inputs"]["sha256"]
            ):
                raise ValueError("preview rendering belongs to different fixed inputs")
            if common_settings.get("result_records") != job["results"]:
                raise ValueError("preview rendering does not identify its generated result records")
            outputs = record.get("outputs", {})
            if set(outputs) != {"video", "poster"}:
                raise ValueError("preview rendering lacks video or poster")
        else:
            if record.get("state") != "complete":
                raise ValueError("preview encoding is not complete")
            if record.get("mode") != job["fixed_inputs"]["mode"]:
                raise ValueError("preview result has the wrong mode")
            for role, key in (
                ("capture", "capture_sha256"),
                ("guide", "guide_sha256"),
                ("first_image", "c0_sha256"),
                ("text", "text_sha256"),
                ("noise", "noise_sha256"),
            ):
                expected = job["fixed_inputs"]["input_files"].get(role)
                if expected is not None and record.get(key) != expected.get("tensor_sha256"):
                    raise ValueError(f"preview result changed the fixed {role} tensor")
            if record.get("adapter") is not None and record.get("adapter_sha256") != job["checkpoint"]["sha256"]:
                raise ValueError("preview result uses a different checkpoint")
            outputs = {"encoding": record["output"]}
        for output in outputs.values():
            if sha256(Path(output["path"])) != output["sha256"]:
                raise ValueError("preview encoded/rendered output changed")


def set_preview_state(
    path: Path,
    state: str,
    *,
    error: str | None = None,
    results: list[dict] | None = None,
    renderings: list[dict] | None = None,
) -> dict:
    """Serialize job transitions; they never write to training or checkpoint files."""
    transitions = {
        "pending": {"running", "failed"},
        "failed": {"running"},
        "running": {"complete", "failed"},
        "complete": set(),
    }
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job = verify_preview_job(path, verify_files=state != "failed")
        if state not in transitions.get(job.get("state"), set()):
            raise ValueError("unsupported preview state transition")
        if job["state"] == "running" and job.get("pid") != os.getpid():
            try:
                os.kill(job["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("preview is owned by a live process")
        if state == "complete":
            _verify_preview_outputs(results or [], job, rendered=False)
            job["results"] = results
            _verify_preview_outputs(renderings or [], job, rendered=True)
            if not any(
                json.loads(Path(r["path"]).read_text()).get("adapter_sha256") == job["checkpoint"]["sha256"]
                for r in results
            ):
                raise ValueError("preview has no generated result from its pinned checkpoint")
            job.update(results=results, renderings=renderings)
        if state == "failed" and not error:
            raise ValueError("failed preview requires a recorded reason")
        job.update(state=state, pid=os.getpid(), updated_at=time.time(), error=error)
        atomic_write(path, lambda temporary: temporary.write_text(json.dumps(job, indent=2) + "\n"))
        return job


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
    from scripts.onestep_avatar.precompute import file_fingerprint  # noqa: PLC0415 -- producer identity

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
        "negative_text_sha256": None if negative is None else tensor_sha256(negative),
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
            "capture_sha256": tensor_sha256(capture), "guide_sha256": None if guide is None else tensor_sha256(guide),
            "c0_sha256": tensor_sha256(c0), "text_sha256": tensor_sha256(context),
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
                expected = {**fixed, **adapter, "noise_sha256": tensor_sha256(epsilon)}
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


def render_preview_outputs(path: Path, *, gpu_id: int) -> dict:  # noqa: PLR0915 -- ordered decoder/media lifecycle
    """Render owned saved preview results without another transformer call."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- saved-output rendering
    from scripts.prune.core.session import Session  # noqa: PLC0415 -- decoder-only session

    producer_software = software.capture("decoding")
    job = verify_preview_job(path)
    if job.get("state") != "running" or job.get("pid") != os.getpid():
        raise ValueError("preview rendering requires the current running owner")
    fixed = job["fixed_inputs"]
    if fixed.get("reference_bundle") is None:
        raise ValueError("preview rendering requires pinned reference pixels")
    references, producer = media.load_training_references(Path(fixed["reference_bundle"]["path"]))
    decoder_settings = media.native_decoder_settings()
    if producer.get("decoder_settings") != decoder_settings:
        raise ValueError("preview reference decoder settings differ from the current runtime")
    results = job.get("results", [])
    _verify_preview_outputs(results, job, rendered=False)
    records = [json.loads(Path(row["path"]).read_text()) for row in results]
    changed = [record for record in records if record.get("adapter_sha256") == job["checkpoint"]["sha256"]]
    base = [record for record in records if record.get("adapter") is None]
    if len(changed) != 1 or len(base) > 1 or len(records) != len(changed) + len(base):
        raise ValueError("preview rendering requires one pinned adapter and at most one base")
    for record in records:
        if (
            record.get("source") != producer["source"]
            or record.get("fps") != producer["fps"]
            or (record["frames"] - 1) * 8 + 1 != len(producer["source_frames"])
        ):
            raise ValueError("preview output source/timebase/coverage differs from its references")
    args = parse_args([*fixed["evaluation_arguments"], "--output", job["output"]])
    specification = backbone.resolve(args.model, args.variant)
    vae_hash = sha256(Path(specification.paths.video_vae()))
    if vae_hash != producer["vae_sha256"]:
        raise ValueError("preview rendering VAE differs from its fixed references")
    destination = Path(job["output"]) / f"render_attempt_{job['raw_attempt']:04d}"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("preview rendering output is already used")
    question = "Does the adapter change the output?"
    output_panels = [
        media.Panel(role, title, None, tuple(producer["source_frames"]),
                    value=("no adapter" if role == "baseline" else f"step {job['checkpoint']['step']}")
                    if selected else "", missing_reason="Not requested" if not selected else "")
        for role, selected, title in (("baseline", base, "Base output"), ("changed", changed, "Adapter output"))
    ]
    planned = references + output_panels
    try:
        media.layout_geometry(planned, question=question, layout="training")
        layout = "training"
    except ValueError:
        layout = media.compact_layout(planned, question=question, layout="training")
    software.check_current(producer_software)
    session = Session(specification, torch.device(f"cuda:{gpu_id}"), "onestep_avatar.preview_render", None)
    decode_records = []
    with session.decoder() as decoder:
        for role, selected, title in (("baseline", base, "Base output"), ("changed", changed, "Adapter output")):
            if not selected:
                references.append(media.Panel(role, title, None, missing_reason="Not requested"))
                continue
            record = selected[0]
            latent = torch.load(Path(record["output"]["path"]), map_location="cpu", weights_only=True)
            pixels = media.decode(session, latent, decoder, producer["decode_seed"])
            references.append(
                media.Panel(
                    role,
                    title,
                    pixels,
                    tuple(producer["source_frames"]),
                    value="no adapter" if role == "baseline" else f"step {job['checkpoint']['step']}",
                )
            )
            decode_records.append(
                {
                    "role": role,
                    "decode_key": media.decode_key(
                        record["output"]["sha256"],
                        vae_hash,
                        list(latent.shape),
                        "native_decode_video",
                        producer["decode_seed"],
                        decoder_settings,
                    ),
                }
            )
    pixels, rendering = media.render_panels(
        references,
        question=question,
        layout=layout,
        fps=producer["fps"],
        common_settings={
            "preview_job_id": job["id"],
            "fixed_inputs_sha256": fixed["sha256"],
            "result_records": results,
            "reference_bundle": fixed["reference_bundle"],
            "decoder_records": decode_records,
        },
    )
    rendering["software"] = producer_software
    media.save_render(pixels, rendering, destination)
    rendered_path = destination / "rendering.json"
    return set_preview_state(
        path,
        "complete",
        results=results,
        renderings=[{"path": str(rendered_path.resolve()), "sha256": sha256(rendered_path)}],
    )


def _saved_panel_path(item: dict, root: Path) -> tuple[Path, bool]:
    """Resolve the shared saved-output/master spelling without loading data."""
    reference = item.get("latent")
    if not isinstance(reference, str) or not reference:
        raise ValueError("saved comparison latent is missing")
    master = reference.startswith(("capture:", "guide:"))
    path = Path(reference.split(":", 1)[1] if master else reference)
    path = (root / path).resolve()
    return path, master


def saved_comparison_inputs_ready(spec_path: Path) -> bool:
    """Wait for saved panel files; invalid empty specifications fail rather than wait."""
    spec = json.loads(spec_path.read_text())
    comparisons = spec.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("saved comparison specification is empty")
    paths = []
    for comparison in comparisons:
        panels = comparison.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError("saved comparison requires panels")
        root = spec_path.resolve().parent
        if "reference_bundle" in comparison:
            reference_path = (root / comparison["reference_bundle"]).resolve()
            paths.append(reference_path)
            if reference_path.is_file():
                reference = json.loads(reference_path.read_text())
                paths.extend(Path(row["path"]) for row in reference.get("panels", []) if row.get("path") is not None)
        for panel in panels:
            if "reference_role" not in panel:
                paths.append(_saved_panel_path(panel, root)[0])
            if "result" in panel:
                paths.append((root / panel["result"]).resolve())
    return all(path.is_file() for path in paths)


def _saved_panel_input(item: dict, root: Path, span: int, fps: float) -> tuple[torch.Tensor, dict]:
    """Resolve an existing output or a checked master without changing recorded bytes."""
    path, master = _saved_panel_path(item, root)
    if not path.is_file():
        raise ValueError("saved comparison latent is missing")
    fingerprint = sha256(path)
    if master:
        latent, source_fps = dataset.load_training_master(path)
        if source_fps != fps:
            raise ValueError("saved comparison bundle fps differs from playback fps")
        if latent.shape[1] < span:
            raise ValueError("saved comparison master has insufficient frame coverage")
        latent = latent[:, :span].unsqueeze(0)
    else:
        latent = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[0] != 1
        or not latent.is_floating_point() or min(latent.shape) < 1 or not torch.isfinite(latent).all()
    ):
        raise ValueError("saved comparison requires a finite floating B,C,F,H,W latent with batch one")
    if latent.shape[2] != span:
        raise ValueError("saved comparison encoded geometry differs from requested coverage")
    if sha256(path) != fingerprint:
        raise ValueError("saved comparison latent changed while loading")
    return latent, {"path": str(path), "sha256": fingerprint, "shape": list(latent.shape), "master": master}


def _saved_comparison_inputs(  # noqa: PLR0912 -- all saved RGB/result gates precede decoder work
    comparison: dict, root: Path, model: str, seed: int
) -> list[tuple[torch.Tensor, dict]]:
    """Read latent-only panels or exactly matched, already-prepared RGB references."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- checked saved RGB reader
    from scripts.prune.core import model_registry  # noqa: PLC0415 -- geometry only

    span, fps = comparison.get("span", 17), comparison.get("fps", 30)
    panels = comparison["panels"]
    if "reference_bundle" not in comparison:
        if any("reference_role" in panel for panel in panels):
            raise ValueError("saved RGB panels require a reference bundle")
        return [_saved_panel_input(panel, root, span, fps) for panel in panels]
    if "view" in comparison:
        raise ValueError("saved RGB reference comparisons do not use legacy view metrics")
    path = (root / comparison["reference_bundle"]).resolve()
    manifest_hash = sha256(path)
    references, producer = media.load_training_references(path)
    manifest = json.loads(path.read_text())
    if sha256(path) != manifest_hash:
        raise ValueError("saved reference bundle changed while loading")
    software.check_current(producer.get("software"))
    scale = model_registry.resolve(model).scale_factors
    frames = 1 + (span - 1) * scale.time
    if (producer.get("fps") != fps or producer.get("source_frames") != list(range(frames))
            or producer.get("decode_seed") != seed
            or producer.get("decoder_settings") != media.native_decoder_settings()
            or producer.get("vae_sha256") != sha256(Path(model_registry.resolve(model).paths.video_vae()))):
        raise ValueError("saved reference source mapping, timebase or decoder settings differ")
    if (len(panels) < 4 or [item.get("reference_role") for item in panels[:3]] != ["recorded", "decoded", "guide"]
            or any("reference_role" in item for item in panels[3:])):
        raise ValueError("saved reference panels require recorded, decoded, guide before outputs")
    inputs, records = [], []
    for item, panel, row in zip(panels[:3], references, manifest["panels"], strict=True):
        pixels = panel.pixels
        if (item.get("role") != panel.role or "latent" in item or pixels is None
                or pixels.ndim != 4 or pixels.shape[:2] != (frames, 3)
                or not (pixels.is_floating_point() or pixels.dtype == torch.uint8)
                or not torch.isfinite(pixels).all()):
            raise ValueError("saved reference RGB values or panel roles are invalid")
        inputs.append((pixels, {"path": row["path"], "sha256": row["sha256"], "shape": list(pixels.shape),
                               "kind": "rgb_reference", "reference_bundle": {"path": str(path),
                                                                                "sha256": manifest_hash}}))
    for item in panels[3:]:
        latent, evidence = _saved_panel_input(item, root, span, fps)
        if "result" not in item:
            raise ValueError("saved reference output requires its executed result record")
        result_path = (root / item["result"]).resolve()
        result_hash = sha256(result_path)
        record = json.loads(result_path.read_text())
        software.validate(record.get("software"))
        if (record.get("state") != "complete" or record.get("source") != producer.get("source")
                or record.get("fps") != fps or record.get("frames") != span
                or record.get("input_file_hashes", {}).get("capture") != producer.get("capture_encoding_sha256")
                or record.get("input_file_hashes", {}).get("render") != producer.get("guide_rgb_sha256")
                or record.get("membership_sha256") != producer.get("membership_sha256")
                or record.get("conditions", {}).get("task", {}).get("objective") != producer.get("objective")
                or record.get("guide_mode") != "d1"
                or Path(record.get("output", {}).get("path", "")).resolve() != Path(evidence["path"])
                or record.get("output", {}).get("sha256") != evidence["sha256"]
                or record.get("output", {}).get("shape") != list(latent.shape)
                or tensor_sha256(VideoLatentPatchifier(patch_size=1).patchify(latent[:, :, :1]))
                != record.get("c0_sha256")):
            raise ValueError("saved output result differs from its RGB references or encoding")
        if any(pixels.shape[-2:] != (latent.shape[-2] * scale.height, latent.shape[-1] * scale.width)
               for pixels, _ in inputs[:3]):
            raise ValueError("saved reference RGB dimensions differ from output encoding")
        evidence["result"] = {"path": str(result_path), "sha256": result_hash}
        inputs.append((latent, evidence))
        records.append(record)
    if len(records) > 1:
        validate_comparison(records, comparison.get("changed_factor"))
        if any(record.get("software") != records[0].get("software") for record in records[1:]):
            raise ValueError("saved comparison output producers differ")
    return inputs


def _saved_input_bytes_current(inputs: list[tuple[torch.Tensor, dict]]) -> bool:
    """Check every bound pixel, latent, manifest and executed-result file once."""
    files = {}
    for _, record in inputs:
        for evidence in (record, record.get("reference_bundle"), record.get("result")):
            if evidence is not None:
                files[evidence["path"]] = evidence["sha256"]
    return all(sha256(Path(path)) == digest for path, digest in files.items())


def parse_saved_comparison_args(argv: list[str], *, require_gpu: bool = False) -> argparse.Namespace:
    """One parser for direct rendering and queue normalization."""
    parser = argparse.ArgumentParser(description="Render saved comparison latents with the package decoder owner")
    parser.add_argument("--render-saved-comparisons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=require_gpu, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def saved_comparison_identity(spec_path: Path, spec: dict, seed: int) -> dict:
    """Bind a saved render to actual decoder bytes and software, without model loading."""
    from scripts.onestep_avatar import media  # noqa: PLC0415
    from scripts.prune.core import model_registry  # noqa: PLC0415

    model = model_registry.resolve(spec.get("model", "2.5"))
    vae = Path(model.paths.video_vae()).resolve()
    return {
        "spec": str(spec_path.resolve()), "spec_sha256": sha256(spec_path), "seed": seed,
        "decoder": {"model": spec.get("model", "2.5"), "variant": spec.get("variant", "dev"),
                    "vae_path": str(vae), "vae_sha256": sha256(vae), "settings": media.native_decoder_settings()},
        "source_code_sha256": {"evaluate": sha256(Path(__file__)), "media": sha256(Path(media.__file__))},
        "software": software.capture("decoding"),
    }


def _save_comparison_variant(pixels: torch.Tensor, record: dict, output: Path, name: str) -> dict:
    """Save a shared-layout variant and bind its stable report-facing media names."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- common RGB media owner

    destination = output / name
    complete = media.save_render(pixels, record, destination)
    named_video, named_poster = output / f"{name}.mp4", output / f"{name}_poster.png"
    (destination / "comparison.mp4").replace(named_video)
    (destination / "poster.png").replace(named_poster)
    for field, path in (("video", named_video), ("poster", named_poster)):
        complete["outputs"][field] = {"path": str(path), "sha256": sha256(path)}
    (destination / "rendering.json").write_text(json.dumps(complete, indent=2) + "\n")
    return {"rendering": complete, "video": named_video.name, "poster": named_poster.name}


def render_saved_comparisons(  # noqa: PLR0912, PLR0915 -- preflight, decode, QA and publication in order
    spec_path: Path, output: Path, *, gpu_id: int, seed: int = 42
) -> dict:
    """Decode and render a saved comparison specification under package ownership.

    The spec contains only saved latent paths and panel metadata. This function
    opens the selected VAE session, never a transformer, and publishes media
    through the shared renderer. Report code receives the manifest afterward.
    """
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- shared preflight/decode/render owner

    spec_hash = sha256(spec_path)
    spec = json.loads(spec_path.read_text())
    if not isinstance(spec.get("comparisons"), list) or not spec["comparisons"]:
        raise ValueError("saved comparison specification is empty")
    if (output / "render_manifest.json").exists():
        raise ValueError("saved comparison output already exists")
    spec_root = spec_path.resolve().parent
    names = set()
    reserved_outputs = set()
    prepared = []
    compact_layouts = []
    for comparison in spec["comparisons"]:
        if not isinstance(comparison, dict) or not isinstance(comparison.get("name"), str) or not comparison["name"]:
            raise ValueError("saved comparison requires a nonempty name")
        if comparison["name"] in names:
            raise ValueError("saved comparison names must be unique")
        names.add(comparison["name"])
        if Path(comparison["name"]).name != comparison["name"] or comparison["name"] in (".", ".."):
            raise ValueError("saved comparison name must be one safe path component")
        for name in (comparison["name"], comparison["name"] + "_compact"):
            for target in (output / name, output / f"{name}.mp4", output / f"{name}_poster.png"):
                if target in reserved_outputs:
                    raise ValueError("saved comparison output names collide")
                reserved_outputs.add(target)
                if target.exists():
                    raise ValueError("saved comparison output already exists")
        span, fps = comparison.get("span", 17), comparison.get("fps", 30)
        if type(span) is not int or span < 1 or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("saved comparison span and fps must be positive")
        panels = comparison.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError("saved comparison requires panels")
        for panel in panels:
            if not isinstance(panel, dict) or not isinstance(panel.get("title"), str):
                raise ValueError("saved comparison panel metadata is invalid")
        inputs = _saved_comparison_inputs(comparison, spec_root, spec.get("model", "2.5"), seed)
        latents = [value for value, record in inputs if record.get("kind") != "rgb_reference"]
        if any(latent.shape != latents[0].shape for latent in latents):
            raise ValueError("saved comparison panels must have identical encoded geometry")
        if "view" in comparison and latents[0].shape[2] < 2:
            raise ValueError("saved comparison metrics require frames after the first image")
        metadata = [media.Panel(item.get("role", f"panel_{index}"), item["title"], None, (),
                                value=item.get("value", "")) for index, item in enumerate(panels)]
        question, layout = comparison.get("question", comparison["name"]), comparison.get("layout", "comparison")
        panel_size = tuple(comparison.get("panel_size", [400, 400]))
        media.layout_geometry(metadata, question=question, layout=layout, panel_size=panel_size,
                              viewing_width=comparison.get("viewing_width", 1280))
        compact_layouts.append(media.compact_layout(metadata, question=question, layout=layout, panel_size=panel_size))
        prepared.append(inputs)

    identity = saved_comparison_identity(spec_path, spec, seed)
    if identity["spec_sha256"] != spec_hash:
        raise ValueError("saved comparison specification changed during preflight")
    if any(not _saved_input_bytes_current(inputs) for inputs in prepared):
        raise ValueError("saved comparison inputs changed during preflight")
    software.check_current(identity["software"])
    session = media.open_decoder_session(
        spec.get("model", "2.5"), gpu_id, script="onestep_avatar.evaluate.saved_comparisons"
    )
    perceptual = None
    if any("view" in comparison for comparison in spec["comparisons"]):
        import lpips  # noqa: PLC0415 -- historical RGB QA only

        perceptual = lpips.LPIPS(net="alex", verbose=False).to(session.device).eval()
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with session.decoder() as decoder, torch.inference_mode():
        for comparison, inputs, compact_layout in zip(spec["comparisons"], prepared, compact_layouts, strict=True):
            panels = []
            for index, (item, (value, record)) in enumerate(zip(comparison["panels"], inputs, strict=True)):
                pixels = value if record.get("kind") == "rgb_reference" else media.decode(session, value, decoder, seed)
                panels.append(
                    media.Panel(
                        item.get("role", f"panel_{index}"),
                        item["title"],
                        pixels,
                        tuple(range(len(pixels))),
                        value=item.get("value", ""),
                    )
                )
            metrics = []
            if "view" in comparison:
                reference = panels[0].pixels
                view = (spec_root / comparison["view"]).resolve()
                mask = subject_mask(view / "capture_mask_crop.mp4", len(reference), *reference.shape[-2:])
                for item, panel in zip(comparison["panels"], panels, strict=True):
                    metrics.append({
                        "title": item["title"], "latent": item.get("latent"),
                        "psnr_full": rgb_metrics(panel.pixels[1:], reference[1:])["psnr"],
                        "psnr_subject": None if mask is None else subject_rgb_metrics(
                            panel.pixels[1:], reference[1:], mask[1:]
                        )["psnr"],
                        "lpips": lpips_distance(perceptual, panel.pixels[1:], reference[1:], session.device),
                    })
            rendered, record = media.render_panels(
                panels,
                question=comparison.get("question", comparison["name"]),
                layout=comparison.get("layout", "comparison"),
                fps=comparison.get("fps", 30),
                common_settings={"spec": str(spec_path.resolve()), "seed": seed},
                poster_frame=min(comparison.get("poster_frame", 96), len(panels[0].pixels) - 1),
                panel_size=tuple(comparison.get("panel_size", [400, 400])),
                viewing_width=comparison.get("viewing_width", 1280),
            )
            record["software"] = identity["software"]
            full = _save_comparison_variant(rendered, record, output, comparison["name"])
            del rendered
            compact_pixels, compact_record = media.render_panels(
                panels, question=comparison.get("question", comparison["name"]), layout=compact_layout,
                fps=comparison.get("fps", 30), panel_size=tuple(comparison.get("panel_size", [400, 400])),
                viewing_width=480, common_settings={"spec": str(spec_path.resolve()), "seed": seed},
                poster_frame=min(comparison.get("poster_frame", 96), len(panels[0].pixels) - 1),
            )
            compact_record["software"] = identity["software"]
            compact = _save_comparison_variant(compact_pixels, compact_record, output, comparison["name"] + "_compact")
            del compact_pixels
            manifest.append({**comparison, **full, "compact": compact, "inputs": [row for _, row in inputs],
                             "frames": len(panels[0].pixels), "metrics": metrics})
    if (saved_comparison_identity(spec_path, spec, seed) != identity
            or any(not _saved_input_bytes_current(inputs) for inputs in prepared)):
        raise ValueError("saved comparison inputs changed before publication")
    result = {"schema_version": 3, **identity, "comparisons": manifest, "results": manifest}
    atomic_write(
        output / "render_manifest.json",
        lambda temporary: temporary.write_text(json.dumps(result, indent=2) + "\n"),
    )
    return result


def _verify_comparison_variant(  # noqa: PLR0912 -- bound layout/media checks
    request: dict, row: dict, output: Path, *, frames: int, fps: float, spec_path: Path, seed: int
) -> bool:
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- single geometry owner

    rendering = row.get("rendering", {})
    software.check_current(rendering.get("software"))
    if (rendering.get("fps") != fps
            or rendering.get("source_frames") != list(range(frames))
            or rendering.get("layout") != request.get("layout", "comparison")
            or rendering.get("question") != request.get("question", request["name"])
            or rendering.get("poster_frame") != min(request.get("poster_frame", 96), frames - 1)
            or rendering.get("panel_size") != request.get("panel_size", [400, 400])
            or rendering.get("viewing_width") != request.get("viewing_width", 1280)
            or rendering.get("common_settings") != {"spec": str(spec_path.resolve()), "seed": seed}):
        raise ValueError("saved comparison completion coverage or settings differ")
    metadata = [media.Panel(item.get("role", f"panel_{index}"), item["title"], None, (),
                            value=item.get("value", "")) for index, item in enumerate(request["panels"])]
    geometry = media.layout_geometry(metadata, question=request.get("question", request["name"]),
                                     layout=request.get("layout", "comparison"),
                                     panel_size=tuple(request.get("panel_size", [400, 400])),
                                     viewing_width=request.get("viewing_width", 1280))
    if (rendering.get("font_size") != geometry["font_size"]
            or rendering.get("display_size") != geometry["display_size"]
            or rendering.get("source_times") != [value / fps for value in range(frames)]):
        raise ValueError("saved comparison completion geometry or readability differs")
    panels = [item for item in rendering.get("panels", []) if item.get("role") != "unused"]
    roles = [panel.get("role", f"panel_{index}") for index, panel in enumerate(request["panels"])]
    if rendering["layout"] in media.COMPARISON_COLUMNS:
        columns = min(media.COMPARISON_COLUMNS[rendering["layout"]], len(roles))
        roles += ["unused"] * (-len(roles) % columns)
        positions = [(index // columns, index % columns, role) for index, role in enumerate(roles)]
    else:
        positions = [(r, c, role) for r, line in enumerate(media.LAYOUTS[rendering["layout"]])
                     for c, role in enumerate(line)]
    if [(item.get("row"), item.get("column"), item.get("role"))
            for item in rendering.get("panels", [])] != positions:
        raise ValueError("saved comparison completion panel positions differ")
    if [item.get("title") for item in panels] != [item["title"] for item in request["panels"]]:
        raise ValueError("saved comparison completion rendered titles differ")
    for index, (panel, requested) in enumerate(zip(panels, request["panels"], strict=True)):
        digest = panel.get("pixels_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("saved comparison completion panel pixel hash is invalid")
        if (panel.get("role") != requested.get("role", f"panel_{index}")
                or panel.get("value") != requested.get("value", "")
                or panel.get("source_frames") != list(range(frames))):
            raise ValueError("saved comparison completion rendered panel settings differ")
    for field, suffix in (("video", ".mp4"), ("poster", "_poster.png")):
        name = row.get(field)
        if name != request["name"] + suffix or Path(name).name != name:
            raise ValueError("saved comparison completion media name differs")
        artifact = output / name
        if not artifact.resolve().is_relative_to(output.resolve()):
            raise ValueError("saved comparison completion media escapes output")
        if not artifact.is_file():
            return False
        recorded = rendering.get("outputs", {}).get(field, {})
        if (Path(recorded.get("path", "")).resolve() != artifact.resolve()
                or artifact.stat().st_size == 0 or sha256(artifact) != recorded.get("sha256")):
            raise ValueError("saved comparison completion media hash or path differs")
    return True


def verify_saved_comparison_completion(  # noqa: PLR0912 -- exact rendering evidence gates together
    spec_path: Path, output: Path, *, seed: int = 42
) -> bool:
    """Verify exact queued render evidence without invoking a decoder or model."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- canonical layout declarations
    from scripts.prune.core import model_registry  # noqa: PLC0415

    path = output / "render_manifest.json"
    if not path.is_file():
        return False
    spec, result = json.loads(spec_path.read_text()), json.loads(path.read_text())
    if not saved_comparison_inputs_ready(spec_path):
        return False
    identity = saved_comparison_identity(spec_path, spec, seed)
    if result.get("schema_version") != 3 or any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("saved comparison completion identity differs")
    expected, actual = spec.get("comparisons"), result.get("comparisons")
    if not expected or not isinstance(actual, list) or len(actual) != len(expected) or result.get("results") != actual:
        raise ValueError("saved comparison completion inventory differs")
    scale = model_registry.resolve(spec.get("model", "2.5")).scale_factors.time
    for request, row in zip(expected, actual, strict=True):
        if any(row.get(key) != value for key, value in request.items()):
            raise ValueError("saved comparison completion request fields differ")
        span, fps = request.get("span", 17), request.get("fps", 30)
        saved_inputs = row.get("inputs")
        if not isinstance(saved_inputs, list) or len(saved_inputs) != len(request["panels"]):
            raise ValueError("saved comparison completion input inventory differs")
        try:
            prepared = _saved_comparison_inputs(request, spec_path.resolve().parent, spec.get("model", "2.5"), seed)
        except FileNotFoundError:
            return False
        inputs = [record for _, record in prepared]
        if not _saved_input_bytes_current(prepared):
            raise ValueError("saved comparison completion input hash or path differs")
        del prepared
        if row.get("inputs") != inputs:
            raise ValueError("saved comparison completion input inventory differs")
        frames = 1 + (span - 1) * scale
        if row.get("frames") != frames:
            raise ValueError("saved comparison completion coverage or settings differ")
        if not _verify_comparison_variant(request, row, output, frames=frames, fps=fps, spec_path=spec_path, seed=seed):
            return False
        metadata = [media.Panel(item.get("role", f"panel_{index}"), item["title"], None, (),
                                value=item.get("value", "")) for index, item in enumerate(request["panels"])]
        layout = media.compact_layout(metadata, question=request.get("question", request["name"]),
                                      layout=request.get("layout", "comparison"),
                                      panel_size=tuple(request.get("panel_size", [400, 400])))
        compact_request = {**request, "name": request["name"] + "_compact",
                           "question": request.get("question", request["name"]), "layout": layout, "viewing_width": 480}
        compact = row.get("compact")
        if not isinstance(compact, dict):
            raise ValueError("saved comparison completion requires compact media")
        if not _verify_comparison_variant(compact_request, compact, output, frames=frames, fps=fps,
                                          spec_path=spec_path, seed=seed):
            return False
        def pixel_hashes(record: dict) -> dict:
            return {item["role"]: item.get("pixels_sha256") for item in record["panels"] if item["role"] != "unused"}
        if pixel_hashes(row["rendering"]) != pixel_hashes(compact["rendering"]):
            raise ValueError("saved comparison compact panel pixels differ")
    return True


def generate_preview(path: Path, *, gpu_id: int) -> list[dict]:
    """Run a pinned preview's raw stage; rendering is required for completion."""
    try:
        job = verify_preview_job(path)
    except Exception as error:
        identity = verify_preview_job(path, verify_files=False)
        if identity.get("state") == "pending":
            set_preview_state(path, "failed", error=f"{type(error).__name__}: {error}")
        raise
    job = set_preview_state(path, "running")
    try:
        output = Path(job["output"])
        attempt = 0
        while (output / f"attempt_{attempt:04d}").exists():
            attempt += 1
        destination = output / f"attempt_{attempt:04d}"
        args = parse_args(
            [
                *job["fixed_inputs"]["evaluation_arguments"],
                "--checkpoint",
                job["checkpoint"]["path"],
                "--output",
                str(destination),
                "--gpu-id",
                str(gpu_id),
            ]
        )
        args.preview_fixed = job["fixed_inputs"]
        execute_evaluation(args)
        results = [
            {"path": str(record.resolve()), "sha256": sha256(record)}
            for record in sorted(destination.glob("case_*/variant_*/result.json"))
        ]
        _verify_preview_outputs(results, job, rendered=False)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = verify_preview_job(path)
            if current.get("state") != "running" or current.get("pid") != os.getpid():
                raise ValueError("preview generation lost its job ownership")
            current.update(results=results, raw_attempt=attempt, updated_at=time.time())
            atomic_write(path, lambda temporary: temporary.write_text(json.dumps(current, indent=2) + "\n"))
        if job["fixed_inputs"].get("reference_bundle") is not None:
            render_preview_outputs(path, gpu_id=gpu_id)
        return results
    except Exception as error:
        set_preview_state(path, "failed", error=f"{type(error).__name__}: {error}")
        raise


def verify_preview_tensors(fixed: dict, tensors: dict[str, torch.Tensor | None]) -> None:
    """Reject changed execution tensors before opening a transformer."""
    for role, identity in fixed["input_files"].items():
        if role == "subset":
            continue
        tensor = tensors.get(role)
        if tensor is None or tensor_sha256(tensor) != identity.get("tensor_sha256"):
            raise ValueError(f"preview execution changed the fixed {role} tensor")


def execute_evaluation(args: argparse.Namespace, sample_runner=None) -> int:  # noqa: ANN001, PLR0912, PLR0915 -- explicit package sampler and native session orchestration
    """Execute already parsed settings after all data and adapter preflight."""
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
            verify_preview_tensors(
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
            with adapter_loader.inference_transformer(
                session, checkpoint, adapters[index].get("contract"), method=args.adapter_application
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
                    "negative_text_sha256": None if negative_context is None else tensor_sha256(negative_context),
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
    from scripts.onestep_avatar.training.config import LORA_TARGETS  # noqa: PLC0415
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
        records.append({"noise_sha256": tensor_sha256(epsilon), "call_counts": {**counts, **measured}})
    left, right = outputs
    later_delta = float((left[:, :, first:] - right[:, :, first:]).abs().max())
    return {
        "shared_noise_blocks": [0, 1, 2, 3], "changed_noise_blocks": [4, 5, 6, 7],
        "latent_frames_compared_equal": [0, first],
        "earlier_blocks_bit_identical": bool(torch.equal(left[:, :, :first], right[:, :, :first])),
        "later_blocks_max_abs_diff": later_delta, "later_blocks_changed": later_delta > 0,
        "records": records, "capture_sha256": tensor_sha256(target),
        "guide_sha256": tensor_sha256(source), "text_sha256": tensor_sha256(session.context),
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
    if any(argument == "--preview-job" or argument.startswith("--preview-job=") for argument in arguments):
        parser = argparse.ArgumentParser(
            description="Generate pinned raw preview outputs; rendering completes the job."
        )
        parser.add_argument("--preview-job", type=Path, required=True)
        parser.add_argument("--gpu-id", type=int, required=True)
        args = parser.parse_args(arguments)
        generate_preview(args.preview_job, gpu_id=args.gpu_id)
        return 0
    if any(value.split("=", 1)[0] == "--render-saved-comparisons" for value in arguments):
        args = parse_saved_comparison_args(arguments, require_gpu=True)
        render_saved_comparisons(args.render_saved_comparisons, args.output, gpu_id=args.gpu_id, seed=args.seed)
        return 0
    return execute_evaluation(parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
