"""Validated saved inputs for native bidirectional whole-clip D0 experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar import causal_core
from scripts.onestep_avatar.train import _load_training_master
from scripts.prune.core import provenance, session

TASK = "whole_clip_d0"


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    if not manifest.get("videos"):
        raise ValueError("saved D0 manifest has no video/sigma records")
    return manifest


def validate_manifest(manifest: dict) -> None:  # noqa: PLR0912
    if not manifest.get("whole_clip") or manifest.get("trajectory_only"):
        raise ValueError("expected a whole-clip D0 rollout with saved predictions")
    if manifest.get("attention") != "full_bidirectional":
        raise ValueError("expected full bidirectional attention")
    if manifest.get("objective") != "white":
        raise ValueError("expected the white capture objective")
    if manifest.get("latent_dtype") != "torch.bfloat16":
        raise ValueError("native D0 reconstruction requires torch.bfloat16 latent dtype")
    guidance = manifest.get("guidance", {})
    if (guidance.get("cfg") != 1 or guidance.get("stg") != 0 or
            guidance.get("rescale") != 0 or guidance.get("passes_per_step") != 1 or
            guidance.get("negative_prompt") is not None or guidance.get("stg_blocks")):
        raise ValueError("native D0 reconstruction supports only unguided single-pass CFG 1 / STG 0")
    if manifest.get("checkpoint") is not None or manifest.get("model", {}).get("checkpoint") is not None:
        raise ValueError("native D0 reconstruction does not support LoRA checkpoints")
    if type(manifest.get("seed")) is not int:
        raise ValueError("native D0 manifest requires an integer seed")
    sigmas = manifest.get("sigmas")
    if (not isinstance(sigmas, list) or not sigmas or len(set(sigmas)) != len(sigmas) or
            any(type(s) not in (int, float) or not math.isfinite(s) or not 0 < s <= 1 for s in sigmas)):
        raise ValueError("native D0 sigmas must be distinct finite values in (0, 1]")
    if not manifest.get("videos"):
        return  # The pair validator also supports empty synthetic manifests in CPU tests.
    rows = records(manifest)
    if len(rows) != len(manifest["videos"]):
        raise ValueError("duplicate view/sigma pair in manifest")
    for (view, sigma), row in rows.items():
        if row["schedule"] != [sigma, 0.0]:
            raise ValueError(f"{view}: expected a one-step [sigma, 0] schedule")
        if sigma not in manifest["sigmas"]:
            raise ValueError(f"{view}: sigma is absent from the manifest levels")


def actor_identity(view: str) -> str:
    """Identify a capture subject across views, resolving filesystem aliases."""
    path = Path(view).resolve()
    for parent in path.parents:
        if parent.name == "views":
            return str(parent.parent)
    return str(path.parent if path.name.startswith("view") and path.parent.name else path)


def native_provenance(root: Path, manifest: dict, views: list[str], sigmas: list[float]) -> dict:
    """Pin the immutable calibration manifest and its model-facing distribution."""
    validate_manifest(manifest)
    rows = records(manifest)
    return {
        "task": TASK, "attention": manifest["attention"], "objective": manifest["objective"],
        "conditioning": "clean_capture_frame_0", "calibration_views": views, "sigmas": sigmas,
        "seed": manifest["seed"], "text_context": manifest["text_context"],
        "geometry": manifest["geometry"], "guidance": manifest["guidance"],
        "latent_dtype": manifest["latent_dtype"], "model_key": manifest["model"]["model_key"],
        "transformer_fingerprint": manifest["model"]["transformer_fingerprint"],
        "video_vae_fingerprint": manifest["model"]["video_vae_fingerprint"],
        "baseline_manifest": str((root / "manifest.json").resolve()),
        "baseline_manifest_sha256": provenance.file_sha256(root / "manifest.json"),
        "calibration_inputs": [
            {"view": view, "sigma": sigma, **{field: rows[(view, sigma)]["artifacts"][field]
             for field in ("capture_sha256", "epsilon_sha256", "fps", "blocks")}}
            for view in views for sigma in sigmas
        ],
    }


def validate_native_provenance(stamp: dict, baseline: dict | None = None) -> None:
    """Fail closed on mutable calibration inputs or a substituted distribution."""
    try:
        path = Path(stamp["baseline_manifest"])
        if provenance.file_sha256(path) != stamp["baseline_manifest_sha256"]:
            raise ValueError("native calibration manifest content changed")
        recorded = load_manifest(path.parent)
        expected = native_provenance(path.parent, recorded, stamp["calibration_views"], stamp["sigmas"])
        if any(stamp.get(key) != value for key, value in expected.items()):
            raise ValueError("native mask distribution differs from pinned calibration manifest")
        if baseline is not None:
            validate_manifest(baseline)
            for key in ("seed", "text_context", "geometry", "guidance", "latent_dtype", "attention", "objective"):
                if stamp[key] != baseline[key]:
                    raise ValueError(f"native mask distribution differs from baseline: {key}")
            for key in ("model_key", "transformer_fingerprint", "video_vae_fingerprint"):
                if stamp[key] != baseline["model"][key]:
                    raise ValueError(f"native mask model differs from baseline: {key}")
            rows = records(baseline)
            for item in stamp["calibration_inputs"]:
                row = rows[(item["view"], item["sigma"])]["artifacts"]
                if any(item[key] != row[key] for key in ("capture_sha256", "epsilon_sha256", "fps", "blocks")):
                    raise ValueError("native mask calibration inputs differ from baseline")
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("incomplete native D0 mask provenance or unavailable pinned manifest") from exc


def records(manifest: dict) -> dict[tuple[str, float], dict]:
    return {(row["view"], float(row["sigma"])): row for row in manifest["videos"]}


def latent_path(root: Path, row: dict) -> Path:
    match = [item for item in row["artifacts"]["latents"]
             if item["arm"] == "d0" and item["sigma"] == row["sigma"]]
    if len(match) != 1:
        raise ValueError(f"expected one D0 latent for {row['view']} at sigma {row['sigma']}")
    path = root / match[0]["path"]
    if provenance.file_sha256(path) != match[0]["sha256"]:
        raise ValueError("saved D0 latent content changed")
    return path


def verify_pair(base: dict, candidate: dict) -> None:
    validate_manifest(base)
    validate_manifest(candidate)
    for key in ("objective", "sigmas", "seed", "trajectory_only", "geometry", "attention",
                "latent_dtype", "text_context", "guidance"):
        if base[key] != candidate[key]:
            raise ValueError(f"unmatched manifest field: {key}")
    for key in ("model_key", "video_vae_fingerprint"):
        if base["model"][key] != candidate["model"][key]:
            raise ValueError(f"unmatched model field: {key}")
    if base["model"]["transformer_fingerprint"] == candidate["model"]["transformer_fingerprint"]:
        raise ValueError("transformer fingerprints are equal; no pruning comparison")
    base_rows, candidate_rows = records(base), records(candidate)
    if base_rows.keys() != candidate_rows.keys():
        raise ValueError("views or sigma levels differ")
    for key, row in base_rows.items():
        other = candidate_rows[key]
        for field in ("capture_sha256", "fps", "blocks"):
            if row["artifacts"][field] != other["artifacts"][field]:
                raise ValueError(f"unmatched {field} for {key}")


def verify_candidate(base: dict, candidate: dict) -> dict:
    """Bind a paired candidate to its actual export, source and mask content."""
    # Hooks consume this input contract; defer this validation-only dependency.
    from scripts.prune.score import export_pruned, hooks  # noqa: PLC0415

    verify_pair(base, candidate)
    path = Path(candidate["model"]["transformer_path"])
    if provenance.checkpoint_fingerprint(path) != candidate["model"]["transformer_fingerprint"]:
        raise ValueError("candidate checkpoint changed since saved rollout")
    with safe_open(path, framework="pt", device="cpu") as handle:
        config = json.loads((handle.metadata() or {}).get("config", "{}"))
    pruning = config.get("transformer", {}).get("pruning", {})
    if (pruning.get("task") != TASK or
            pruning.get("source_transformer_fingerprint") != base["model"]["transformer_fingerprint"] or
            pruning.get("model_key") != base["model"]["model_key"]):
        raise ValueError("candidate export task/source differs from native whole-clip D0")
    mask_path = Path(pruning.get("masks", ""))
    if not mask_path.is_file() or provenance.file_sha256(mask_path) != pruning.get("mask_sha256"):
        raise ValueError("candidate export mask content differs or is unavailable")
    hooks.read_mask_artifact(
        mask_path, model_key=base["model"]["model_key"], fingerprint=base["model"]["transformer_fingerprint"],
        widths=export_pruned.checkpoint_mask_widths(base["model"]["transformer_path"]),
        expected_task=TASK, baseline=base,
    )
    return pruning


def load_epsilon(root: Path, row: dict) -> torch.Tensor:
    path = root / row["artifacts"]["epsilon"]
    if provenance.file_sha256(path) != row["artifacts"]["epsilon_sha256"]:
        raise ValueError("saved epsilon content changed")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != session.DTYPE or not torch.isfinite(tensor).all():
        raise ValueError("saved epsilon must be a finite BF16 tensor")
    return tensor


def verify_saved_noise(base_root: Path, candidate_root: Path, base_row: dict, candidate_row: dict) -> None:
    if not torch.equal(load_epsilon(base_root, base_row), load_epsilon(candidate_root, candidate_row)):
        raise ValueError(f"saved noise tensors differ for {base_row['view']} sigma={base_row['sigma']}")


def build_input(root: Path, manifest: dict, *, view: str, sigma: float,
                current: session.Session) -> tuple[causal_core.ClipGrid, Modality, torch.Tensor, dict]:
    """Rebuild the baseline D0 modality without requiring a pruned candidate."""
    validate_manifest(manifest)
    row = records(manifest)[(view, sigma)]
    capture_path = Path(row["artifacts"]["capture"])
    if provenance.file_sha256(capture_path) != row["artifacts"]["capture_sha256"]:
        raise ValueError("capture changed since saved baseline rollout")
    capture, fps = _load_training_master(capture_path)
    if capture.dtype != session.DTYPE or capture.ndim != 4 or not torch.isfinite(capture).all():
        raise ValueError("capture must be a finite BF16 C,T,H,W latent")
    if provenance.checkpoint_fingerprint(current.model.paths.video_vae()) != manifest["model"]["video_vae_fingerprint"]:
        raise ValueError("session VAE differs from saved baseline")
    if fps != row["artifacts"]["fps"]:
        raise ValueError("capture fps changed since saved rollout")
    epsilon = load_epsilon(root, row)
    _, latent_frames, height, width = capture.shape
    geometry = causal_core.deployed_geometry(
        current.model.scale_factors, block_latent_frames=latent_frames - 1,
        context_latent_frames=manifest["geometry"]["context_latent_frames"],
    )
    grid = causal_core.ClipGrid.build(
        latent_frames, height * current.model.scale_factors.height,
        width * current.model.scale_factors.width, fps, geometry,
        device=current.device, dtype=session.DTYPE,
        latent_channels=current.model.caps.latent_channels,
    )
    if geometry.as_dict() != manifest["geometry"] or geometry.plan(latent_frames) != [(0, latent_frames)]:
        raise ValueError("geometry does not match the saved whole-clip run")
    if row["artifacts"]["blocks"] != [[0, latent_frames]]:
        raise ValueError("saved block plan does not cover the complete capture exactly once")
    source = grid.patchify(capture.unsqueeze(0).to(device=current.device, dtype=session.DTYPE))
    prompt_hash = hashlib.sha256(manifest["text_context"]["prompt"].encode()).hexdigest()
    context_bytes = current.context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    context = hashlib.sha256(context_bytes).hexdigest()
    if prompt_hash != manifest["text_context"]["prompt_sha256"] or context != manifest["text_context"]["sha256"]:
        raise ValueError("session text context differs from saved baseline")
    if epsilon.shape != source.shape:
        raise ValueError("saved epsilon does not fit the captured token grid")
    noisy = causal_core.mix_block_noise(source, epsilon.to(current.device), sigma)
    c0 = source[:, :grid.tokens_per_latent_frame]
    state = causal_core.with_clean_prefix(noisy, c0)
    ids = causal_core.block_ids_for([(0, latent_frames, 0)], grid.tokens_per_latent_frame).to(current.device)
    modality = causal_core.block_modality(
        grid, state, current.context, sigma,
        token_slices=[grid.token_span(0, latent_frames)],
        attention_mask=causal_core.block_causal_mask(ids),
        clean_prefix_tokens=grid.tokens_per_latent_frame,
    )
    return grid, modality, c0, row
