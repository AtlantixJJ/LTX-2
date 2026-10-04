"""Bounded CPU ridge reconstruction and strict calibration-only sample-cache validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.prune.core import provenance
from scripts.prune.data import whole_clip
from scripts.prune.score import export_pruned, hooks, token_sampling

CACHE_FORMAT = "whole_clip_d0_ffn_reconstruction_samples_v1"
TARGET = "fp32_source_ffn_projection_without_bias_v1"
SELECTION = "midpoint_subsample_v1"


def fit_output_projection(
    retained_features: torch.Tensor, teacher_output: torch.Tensor, source_retained_weight: torch.Tensor,
    *, ridge_lambda: float, max_samples: int = 512, output_chunk: int = 256, max_memory_bytes: int = 1 << 30,
) -> dict:
    """Fit an FP64 correction around sliced source weights; inputs and targets exclude bias."""
    tensors = (retained_features, teacher_output, source_retained_weight)
    if any(tensor.ndim != 2 or tensor.device.type != "cpu" or not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("ridge inputs must be two-dimensional floating CPU tensors")
    samples, retained = retained_features.shape
    outputs = teacher_output.shape[1]
    if samples < 1 or retained < 1 or outputs < 1 or teacher_output.shape[0] != samples or (
        source_retained_weight.shape != (outputs, retained)
    ):
        raise ValueError("ridge sample/feature/output dimensions are incompatible or empty")
    if type(ridge_lambda) not in (int, float) or not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    if any(type(value) is not int or value < 1 for value in (max_samples, output_chunk, max_memory_bytes)):
        raise ValueError("sample, chunk and memory caps must be positive integers")
    if samples > max_samples:
        raise ValueError("sample count exceeds the declared ridge cap")
    chunk = min(outputs, output_chunk)
    # Include caller tensor storage, result, FP64 features and conservative factor/chunk headroom.
    estimated = sum(tensor.numel() * tensor.element_size() for tensor in tensors) + 8 * (
        samples * retained + outputs * retained + 4 * samples * samples +
        4 * samples * chunk + 3 * retained * chunk
    )
    if estimated > max_memory_bytes:
        raise ValueError(f"ridge estimated memory {estimated} exceeds cap {max_memory_bytes}")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("ridge inputs must be finite")
    with torch.no_grad():
        features = retained_features.detach().to(torch.float64)
        gram = features @ features.T
        gram.diagonal().add_(samples * ridge_lambda)
        factor, info = torch.linalg.cholesky_ex(gram)
        if info.item() != 0:
            raise ValueError("ridge dual matrix is not positive definite; increase ridge_lambda")
        fitted = torch.empty((outputs, retained), dtype=torch.float64)
        before, after = 0.0, 0.0
        for lo in range(0, outputs, chunk):
            hi = min(outputs, lo + chunk)
            original = source_retained_weight[lo:hi].detach().to(torch.float64)
            target = teacher_output[:, lo:hi].detach().to(torch.float64)
            residual = target - features @ original.T
            before += residual.square().sum().item()
            dual = torch.cholesky_solve(residual, factor)
            fitted[lo:hi] = original + (features.T @ dual).T
            error = target - features @ fitted[lo:hi].T
            after += error.square().sum().item()
    if not torch.isfinite(fitted).all():
        raise ValueError("ridge fitted weight contains nonfinite values")
    return {
        "weight": fitted,
        "diagnostics": {
            "method": "dual_ridge_correction_v1", "samples": samples, "retained_channels": retained,
            "output_channels": outputs, "ridge_lambda": ridge_lambda, "objective_normalization": "mean_over_samples",
            "solve_dtype": "torch.float64", "output_chunk": chunk, "estimated_memory_bytes": estimated,
            "max_memory_bytes": max_memory_bytes, "max_samples": max_samples,
            "local_mse_before": before / (samples * outputs), "local_mse_after": after / (samples * outputs),
            "bias": "unchanged; targets exclude source bias", "qualification": "local_calibration_diagnostic_only",
        },
    }


def midpoint_subsample(indices: torch.Tensor, count: int) -> torch.Tensor:
    """Select a predeclared bounded quota from each case's ordered sampler indices."""
    if (indices.ndim != 1 or indices.dtype != torch.int64 or type(count) is not int or
            not 1 <= count <= indices.numel()):
        raise ValueError("invalid calibration sample quota or integer index vector")
    offsets = ((2 * torch.arange(count, dtype=torch.int64, device=indices.device) + 1) * indices.numel()) // (2 * count)
    return indices[offsets]


def validate_cache_manifest(  # noqa: PLR0912, PLR0915
    path: str | Path, *, baseline: dict | None = None, max_samples: int = 512,
    max_payload_bytes: int = 256 << 20, retained_alignment: int = 128,
) -> tuple[dict, dict]:
    """Validate cache identity/scope before loading tensors; does not certify how features were collected."""
    if any(type(value) is not int or value < 1 for value in (max_samples, max_payload_bytes, retained_alignment)):
        raise ValueError("cache caps and retained alignment must be positive integers")
    path = Path(path)
    cache = json.loads(path.read_bytes())
    if (not isinstance(cache, dict) or cache.get("cache_format") != CACHE_FORMAT or
            cache.get("target") != TARGET or cache.get("selection") != SELECTION):
        raise ValueError("invalid FFN reconstruction cache format or projection target")
    stamp = cache.get("provenance")
    if not isinstance(stamp, dict) or stamp.get("task") != whole_clip.TASK:
        raise ValueError("FFN reconstruction cache requires native calibration provenance")
    whole_clip.validate_native_provenance(stamp, baseline)
    views, sigmas = stamp.get("calibration_views"), stamp.get("sigmas")
    if (not isinstance(views, list) or not views or len(set(views)) != len(views) or
            not isinstance(sigmas, list) or not sigmas or len(set(sigmas)) != len(sigmas)):
        raise ValueError("FFN reconstruction cache has incomplete calibration scope")
    source = Path(cache.get("source_checkpoint", ""))
    if not source.is_file() or provenance.checkpoint_fingerprint(source) != stamp.get("transformer_fingerprint"):
        raise ValueError("FFN reconstruction cache source checkpoint changed or is unavailable")
    mask_path = Path(cache.get("mask_path", ""))
    if not mask_path.is_file() or provenance.file_sha256(mask_path) != cache.get("mask_sha256"):
        raise ValueError("FFN reconstruction cache width-mask content changed or is unavailable")
    masks, _ = hooks.read_mask_artifact(
        mask_path, model_key=stamp["model_key"], fingerprint=stamp["transformer_fingerprint"],
        widths=export_pruned.checkpoint_mask_widths(source), baseline=baseline,
    )
    mask_provenance = json.loads(mask_path.read_bytes())["provenance"]
    if stamp != mask_provenance:
        raise ValueError("cache and width mask calibration distributions differ")
    branch = cache.get("branch")
    if not isinstance(branch, str) or not branch.endswith(".ff") or branch not in masks:
        raise ValueError("cache must identify a complete native FFN mask branch")
    keep = [index for index, value in enumerate(masks[branch]) if value]
    retained_indices = cache.get("retained_indices")
    if (not isinstance(retained_indices, list) or any(type(index) is not int for index in retained_indices) or
            retained_indices != keep or len(keep) % retained_alignment):
        raise ValueError("cache retained indices differ from source-order mask or aligned width")
    layer = int(branch.split(".")[0])
    projection = f"{export_pruned.PREFIX}.{layer}.ff.net.2.weight"
    with safe_open(source, framework="pt", device="cpu") as handle:
        config = json.loads(handle.metadata()["config"])["transformer"]
        if config.get("pruning") is not None:
            raise ValueError("FFN reconstruction cache requires an unpruned source checkpoint")
        shape = handle.get_slice(projection).get_shape()
    if shape[1] != len(masks[branch]):
        raise ValueError("source FFN projection width differs from cache mask")
    records = cache.get("cases")
    expected_cases = [(view, sigma) for view in views for sigma in sigmas]
    if not isinstance(records, list) or len(records) != len(expected_cases):
        raise ValueError("cache must cover every calibration view/sigma exactly once")
    seen, cursor, quotas = set(), 0, []
    pinned_manifest = whole_clip.load_manifest(Path(stamp["baseline_manifest"]).parent)
    pinned_rows = whole_clip.records(pinned_manifest)
    for case in records:
        if not isinstance(case, dict):
            raise ValueError("cache cases must be structured calibration records")
        key = (case.get("view"), case.get("sigma"))
        if key not in expected_cases or key in seen:
            raise ValueError("cache case is duplicate or outside calibration scope")
        seen.add(key)
        sampling = case.get("sampling", {})
        frames, height, width = (sampling.get(name) for name in ("latent_frames", "latent_height", "latent_width"))
        if any(type(value) is not int or value < 1 for value in (frames, height, width)):
            raise ValueError("cache case lacks actual latent geometry")
        if pinned_rows[key]["artifacts"]["blocks"] != [[0, frames]]:
            raise ValueError("cache frame count differs from pinned native whole-clip geometry")
        _validate_case_geometry(
            pinned_rows[key]["artifacts"], Path(stamp["baseline_manifest"]).parent,
            frames, height, width, max_payload_bytes,
        )
        full_indices = token_sampling.sample_indices(
            frames * height * width, height, width, sampling.get("spatial_stride"), torch.device("cpu"),
            sampler=sampling.get("sampler"),
        )
        expected_sampling = token_sampling.sampling_record(
            full_indices, tokens=frames * height * width, height=height, width=width,
            stride=sampling["spatial_stride"], sampler=sampling["sampler"],
        )
        if sampling != expected_sampling:
            raise ValueError("cache sampler/hash/geometry record differs from deterministic selection")
        indices = case.get("token_indices")
        if not isinstance(indices, list) or not indices or any(type(index) is not int for index in indices):
            raise ValueError("cache case must pin actual integer token indices")
        selected = midpoint_subsample(full_indices, len(indices))
        if (indices != selected.tolist() or
                case.get("token_indices_sha256") != token_sampling.index_sha256(selected)):
            raise ValueError("cache token indices differ from quota/sampler or include clean c0")
        if case.get("row_slice") != [cursor, cursor + len(indices)]:
            raise ValueError("cache sample rows do not form contiguous case slices")
        cursor += len(indices)
        quotas.append(len(indices))
    if len(set(quotas)) != 1 or cursor > max_samples:
        raise ValueError("cache quotas must be equal per case and fit the total sample cap")
    payload = Path(cache.get("payload", ""))
    if not payload.is_file() or payload.stat().st_size > max_payload_bytes:
        raise ValueError("cache tensor payload unavailable or exceeds byte cap")
    if provenance.file_sha256(payload) != cache.get("payload_sha256"):
        raise ValueError("cache tensor payload content changed")
    return cache, {"samples": cursor, "retained_channels": len(keep), "output_channels": shape[0],
                   "manifest_sha256": provenance.file_sha256(path), "payload_bytes": payload.stat().st_size}


def _validate_case_geometry(
    artifacts: dict, baseline_root: Path, frames: int, height: int, width: int, max_bytes: int,
) -> None:
    """Read tensor metadata only; calibration cache H/W must describe the pinned native capture."""
    capture = Path(artifacts.get("capture", ""))
    epsilon = baseline_root / artifacts.get("epsilon", "")
    if not capture.is_file() or capture.stat().st_size > max_bytes:
        raise ValueError("pinned calibration capture unavailable or exceeds byte cap")
    if provenance.file_sha256(capture) != artifacts.get("capture_sha256"):
        raise ValueError("pinned calibration capture content changed")
    record = torch.load(capture, map_location="meta", weights_only=True)
    master = record.get("master") if isinstance(record, dict) else None
    if (not isinstance(master, torch.Tensor) or master.ndim != 4 or master.dtype != torch.bfloat16 or
            tuple(master.shape[1:]) != (frames, height, width) or record.get("schema_version") != 2 or
            master.numel() * master.element_size() > max_bytes or record.get("fps") != artifacts.get("fps")):
        raise ValueError("cache latent geometry/fps differs from pinned BF16 calibration capture")
    if (not epsilon.is_file() or epsilon.stat().st_size > max_bytes or
            provenance.file_sha256(epsilon) != artifacts.get("epsilon_sha256")):
        raise ValueError("pinned calibration epsilon content changed or exceeds byte cap")
    noise = torch.load(epsilon, map_location="meta", weights_only=True)
    if (not isinstance(noise, torch.Tensor) or noise.dtype != torch.bfloat16 or
            tuple(noise.shape) != (1, frames * height * width, master.shape[0])):
        raise ValueError("cache token geometry differs from pinned BF16 calibration epsilon")


def load_calibration_cache(
    path: str | Path, *, baseline: dict | None = None, max_samples: int = 512,
    max_payload_bytes: int = 256 << 20, retained_alignment: int = 128,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load only content-verified finite FP32 CPU features/targets for the declared calibration cases."""
    cache, dimensions = validate_cache_manifest(
        path, baseline=baseline, max_samples=max_samples, max_payload_bytes=max_payload_bytes,
        retained_alignment=retained_alignment,
    )
    expected = {"retained_features": (dimensions["samples"], dimensions["retained_channels"]),
                "teacher_output": (dimensions["samples"], dimensions["output_channels"])}
    if sum(math.prod(shape) * 4 for shape in expected.values()) > max_payload_bytes:
        raise ValueError("cache declared tensor allocation exceeds payload byte cap")
    meta_payload = torch.load(cache["payload"], map_location="meta", weights_only=True)
    _validate_payload(meta_payload, expected, finite=False)
    payload = torch.load(cache["payload"], map_location="cpu", weights_only=True)
    _validate_payload(payload, expected, finite=True)
    # Detect a substituted/mutated manifest or payload while the verified data was loading.
    if (provenance.file_sha256(path) != dimensions["manifest_sha256"] or
            provenance.file_sha256(cache["payload"]) != cache["payload_sha256"]):
        raise ValueError("cache manifest or payload changed during load")
    whole_clip.validate_native_provenance(cache["provenance"], baseline)
    if (provenance.checkpoint_fingerprint(cache["source_checkpoint"]) !=
            cache["provenance"]["transformer_fingerprint"] or
            provenance.file_sha256(cache["mask_path"]) != cache["mask_sha256"]):
        raise ValueError("cache source or mask changed during load")
    return payload, {"cache": cache, "dimensions": dimensions}


def _validate_payload(payload: dict, expected: dict, *, finite: bool) -> None:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError("cache payload must contain only retained_features and teacher_output")
    for name, shape in expected.items():
        value = payload[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape or
                (finite and not torch.isfinite(value).all())):
            raise ValueError(f"cache {name} must be finite FP32 data with declared dimensions")
