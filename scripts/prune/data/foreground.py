"""Verify recorded cropped mattes and GT region coordinates; pool through corpus owners."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from scripts.onestep_avatar.corpus import dataset, geometry, mask_video
from scripts.prune.core.provenance import file_sha256
from scripts.prune.core.records import tensor_hash


def person_coverage(view, record, *, scale_factors=(8, 32, 32), replay=True):
    if tuple(scale_factors) != (8, 32, 32):
        raise ValueError("foreground requires registered temporal/spatial scales 8/32/32")
    path = view / dataset.CAPTURE_MASK_CROP_NAME
    if not path.is_file():
        raise FileNotFoundError(f"S0 canonical foreground gap: {path}; repair through corpus.precompute")
    mask = mask_video.read_mask_video(path)
    count = record["pixel_frames"]
    frames = (count - 1) // 8 + 1
    if (count - 1) % 8 or len(mask) < count or mask.shape[1:] != (dataset.ALPHA_GRID, dataset.ALPHA_GRID):
        raise ValueError("canonical mask coverage or producer resolution differs")
    probe = cv2.VideoCapture(str(path))
    fps = probe.get(cv2.CAP_PROP_FPS)
    probe.release()
    if fps != record["fps"]:
        raise ValueError("canonical matte fps differs")
    source_path = view / dataset.CAPTURE_MASK_NAME
    source_hash = file_sha256(source_path)
    if replay:
        reader = cv2.VideoCapture(str(source_path))
        try:
            for i in range(count):
                ok, frame = reader.read()
                if not ok:
                    raise ValueError("source matte lacks complete frame coverage")
                crop = geometry.crop_from_canvas(frame[..., 0], tuple(record["box_xyxy"]))
                expected = cv2.resize(crop, (dataset.ALPHA_GRID, dataset.ALPHA_GRID), interpolation=cv2.INTER_AREA)
                if not np.array_equal(mask[i], expected):
                    raise ValueError(f"canonical matte does not replay recorded crop at frame {i}")
        finally:
            reader.release()
    grid = mask_video.pool_to_latent_grid(
        mask[:count], latent_frames=frames, latent_height=32, latent_width=32, time_scale=8
    ).float()
    if not torch.isfinite(grid).all() or grid[1:].sum() <= 0:
        raise ValueError("empty or nonfinite canonical foreground")
    identity = {
        "canonical_path": str(path),
        "canonical_sha256": file_sha256(path),
        "source_sha256": source_hash,
        "crop": record["box_xyxy"],
        "fps": fps,
        "pixel_frames": count,
        "canonical_resolution": list(mask.shape[1:]),
        "pool_dtype": "torch.float16 promoted to torch.float32",
        "grid_sha256": tensor_hash(grid),
        "scales": list(scale_factors),
    }
    return mask[:count], grid, identity


def region_coverage(track, matte, record, *, schema):
    """Verified boxes in source pixels; never infer keypoint identities."""
    if (
        schema.get("format") != "gt_region_boxes_v1"
        or schema.get("coordinates") != "source_pixels"
        or schema.get("reviewed_on_gt") is not True
        or not schema.get("producer_sha256")
    ):
        raise ValueError("GT face/hand schema must be pinned and reviewed")
    boxes = np.asarray(track["boxes"], dtype=np.float64)
    valid = np.asarray(track["valid"], dtype=bool)
    if boxes.shape != (len(matte), 4) or valid.shape != (len(matte),):
        raise ValueError("GT track frame coverage differs")
    x0, y0, x1, y1 = record["box_xyxy"]
    edge = record["edge"]
    crop_boxes = (boxes - np.array([x0, y0, x0, y0])) * edge / (x1 - x0)
    widths = crop_boxes[:, 2:] - crop_boxes[:, :2]
    valid &= np.isfinite(crop_boxes).all(1) & (widths >= 32).all(1)
    crop_boxes[:, :2] -= 0.25 * widths
    crop_boxes[:, 2:] += 0.25 * widths
    crop_boxes = np.clip(crop_boxes, 0, edge)
    valid &= np.isfinite(crop_boxes).all(1) & ((crop_boxes[:, 2:] - crop_boxes[:, :2]) >= 32).all(1)
    crop_boxes = np.nan_to_num(crop_boxes, nan=0.0, posinf=0.0, neginf=0.0)
    covered = np.zeros_like(matte)
    h, w = matte.shape[1:]
    for i in np.flatnonzero(valid):
        a, b, c, d = crop_boxes[i] * np.array([w, h, w, h]) / edge
        left, top, right, bottom = int(np.floor(a)), int(np.floor(b)), int(np.ceil(c)), int(np.ceil(d))
        covered[i, top:bottom, left:right] = matte[i, top:bottom, left:right]
        valid[i] &= bool(covered[i].any())
    grid = mask_video.pool_to_latent_grid(
        covered, latent_frames=(len(matte) - 1) // 8 + 1, latent_height=32, latent_width=32, time_scale=8
    ).float()
    return grid, {
        "valid_frames": int(valid[1:].sum()),
        "absent": not bool(grid[1:].sum()),
        "crop_boxes": crop_boxes.tolist(),
        "valid": valid.tolist(),
        "grid_sha256": tensor_hash(grid),
        "schema": schema,
    }


def gt_statistics(rgb, matte):
    """Person area and adjacent change amount, using GT alone."""
    pixels = rgb.float() / 255
    if pixels.ndim != 4 or pixels.shape[0] != len(matte) or len(matte) < 2:
        raise ValueError("invalid RGB/matte coverage")
    masks = torch.from_numpy(matte).float().unsqueeze(1) / 255
    masks = torch.nn.functional.interpolate(masks, size=pixels.shape[-2:], mode="area")
    union = torch.maximum(masks[1:], masks[:-1])
    masses = union.sum((1, 2, 3))
    if (masses <= 0).any():
        raise ValueError("empty GT person pair")
    change = ((pixels[1:] - pixels[:-1]).abs() * union).sum((1, 2, 3)) / (3 * masses)
    return {"person_area": float(masks[1:].mean()), "motion_amount": float(change.mean())}
