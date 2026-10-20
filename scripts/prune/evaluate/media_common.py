"""Shared decoded-media helpers for pruning comparisons.

`source_pixels` replays the recorded crop and white matte of a source view.
`region_lpips` scores fixed GT regions on generated frames; c0 is excluded.
`pixel_metrics` reports weighted MSE, PSNR and motion amount on the same frames.
`saved_video`, `preserve_source` and `verify_comparison` publish and probe media.
They never decode latents or open a model; decoding stays with each owner.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.onestep_avatar import media
from scripts.onestep_avatar.metrics import lpips_distance
from scripts.prune.core import provenance


def region_lpips(model, student, reference, weights, device, *, boxes=None, valid=None):
    """Use GT coverage and boxes only. Exclude c0 and invalid small regions."""
    if student.shape != reference.shape or weights.shape != (len(student), *student.shape[-2:]):
        raise ValueError("LPIPS fixed region geometry differs")
    left, right, frame_ids = [], [], []
    for f in range(1, len(student)):
        if valid is not None and not valid[f]:
            continue
        w = weights[f]
        if boxes is None:
            ys, xs = torch.where(w > 0)
            if not len(xs):
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        else:
            x0, y0, x1, y1 = [int(v) for v in boxes[f]]
        if min(x1 - x0, y1 - y0) < 32:
            continue
        for value, dest in ((student, left), (reference, right)):
            pixels = value[f].float() / 255
            masked = pixels * w[None] + (1 - w[None])
            crop = masked[:, y0:y1, x0:x1].unsqueeze(0)
            if crop.numel() == 0:
                raise ValueError("reviewed GT box lies outside pixel grid")
            resized = torch.nn.functional.interpolate(
                crop, (256, 256), mode="bilinear", align_corners=False, antialias=True
            )
            # FP32 interpolation can overshoot white by a few ulps. Preserve the
            # metric's strict RGB contract, and reject larger input errors.
            if not torch.isfinite(resized).all() or resized.min() < -1e-6 or resized.max() > 1 + 1e-6:
                raise ValueError("RGB resize exceeds bounded interpolation roundoff")
            dest.append(resized[0].clamp(0, 1))
        frame_ids.append(f)
    if not left:
        return {"lpips": None, "valid_frames": 0, "frame_ids": [], "status": "missing_coverage"}
    return {
        "lpips": lpips_distance(model, torch.stack(left), torch.stack(right), device, batch=8),
        "valid_frames": len(frame_ids),
        "frame_ids": frame_ids,
        "status": "scored",
    }


def pixel_metrics(student, reference, weights):
    if student.shape != reference.shape or student.ndim != 4 or weights.shape != (len(student), *student.shape[-2:]):
        raise ValueError("region pixels and fixed GT coverage differ")
    if not torch.isfinite(student).all() or not torch.isfinite(reference).all() or not torch.isfinite(weights).all():
        raise ValueError("nonfinite pixel input")
    s, r, w = student[1:].float() / 255, reference[1:].float() / 255, weights[1:].float()
    mass = w.sum()
    if mass <= 0:
        raise ValueError("required decoded region coverage is empty")
    mse = float(((s - r).square() * w[:, None]).sum() / (3 * mass))
    union = torch.maximum(w[1:], w[:-1])
    motion = float(((s[1:] - s[:-1]).abs() * union[:, None]).sum() / (3 * union.sum())) if len(s) > 1 else None
    return {
        "mse": mse,
        "psnr": float(-10 * np.log10(mse)) if mse else None,
        "motion_amount": motion,
        "lpips": None,
        "lpips_status": "requires_pinned_metric_and_valid_GT_face_hand_tracks",
        "valid_frames": int((w.sum((1, 2)) > 0).sum()),
    }


def saved_video(pixels, path, fps):
    from ltx_trainer.video_utils import save_video

    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(pixels, path, fps=fps, video_format="FCHW")
    checked = media.verify_saved_video(path, len(pixels), fps)
    return {"path": str(path), "sha256": provenance.file_sha256(path), "verification": checked}


def source_pixels(source):
    """Replay the original crop and white matte through the checked public media owner."""
    resolved = Path(source["resolved_dir"])
    root = resolved.parents[len(Path(source["relative_dir"]).parts) - 1]
    if (root / source["relative_dir"]).resolve() != resolved.resolve():
        raise ValueError("recorded RGB root differs from audited view")
    actual = {**source, "box_xyxy": source["capture_encode_record"]["box_xyxy"]}
    return media.recorded_capture_rgb(actual, root, "white", source["n_latent_frames"])


def preserve_source(source, destination, expected=None):
    """Pin complete source bytes; interrupted copies never publish a partial file."""
    source, destination = Path(source), Path(destination)
    digest = provenance.file_sha256(source)
    if expected is not None and digest != expected:
        raise ValueError("source reference differs from audited content")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        fd, name = tempfile.mkstemp(dir=destination.parent, prefix=".source-")
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as original:
                shutil.copyfileobj(original, output)
                output.flush()
                os.fsync(output.fileno())
            if provenance.file_sha256(name) != digest or provenance.file_sha256(source) != digest:
                raise ValueError("source reference changed while copying")
            try:
                os.link(name, destination)
            except FileExistsError:
                pass  # A concurrent identical copy is checked below.
        finally:
            Path(name).unlink(missing_ok=True)
    if provenance.file_sha256(destination) != digest:
        raise ValueError("preserved source reference differs; retain it for diagnosis")
    return {"path": str(destination), "sha256": digest}


def verify_comparison(record):
    """Probe the actual combined MP4 and PNG before publishing their rendering record."""
    outputs = record["outputs"]
    video, poster = Path(outputs["video"]["path"]), Path(outputs["poster"]["path"])
    checked = media.verify_saved_video(video, len(record["source_frames"]), record["fps"])
    width, height = record["display_size"]
    if int(checked["width"]) != width or int(checked["height"]) != height:
        raise ValueError("comparison video dimensions differ from recorded layout")
    media.verify_saved_png(poster, width, height)
    if any(provenance.file_sha256(item["path"]) != item["sha256"] for item in outputs.values()):
        raise ValueError("comparison output changed during verification")
    outputs["video"]["verification"] = checked
    return record
