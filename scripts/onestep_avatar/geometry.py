"""Pixel-space crop geometry for the ARGAvatar guidance corpus.

Per ``plans/2026-09-10-ltx25-one-step-argavatar-lora.md`` SS4.5: one square crop box per
(clip, view), fixed for the whole clip, sized to contain the subject across every frame. A
per-frame rect would inject camera motion the capture never had, and the temporal RoPE would
learn it -- fixed-for-the-clip is not a simplification.

Pure and dependency-free (numpy only) so it is unit-testable without a GPU or the dataset --
see ``tests/test_geometry.py``.
"""

from __future__ import annotations

import numpy as np

# The requested 1.20 square is shifted/clamped into the original canvas: the
# capture target never contains synthetic padding pixels.
PADDING_FACTOR = 1.20
OUT_SIZE = 1024

XYXY = tuple[float, float, float, float]


def valid_bbox_mask(bbox_xyxy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``valid & ~isnan(xyxy).any(1)`` -- never filter on ``valid`` alone.

    A small fraction of frames (2/200 views in the 09-05 sample, 0/300 in this corpus) carry
    NaN bbox rows even where the dataset's own ``valid`` flag is True.
    """
    return np.asarray(valid, dtype=bool) & ~np.isnan(bbox_xyxy).any(axis=1)


def union_bbox(bbox_xyxy: np.ndarray, valid: np.ndarray) -> XYXY:
    """The union of every valid, finite frame bbox over a whole clip."""
    mask = valid_bbox_mask(bbox_xyxy, valid)
    if not mask.any():
        raise ValueError("no valid, finite bbox in this clip/view")
    boxes = bbox_xyxy[mask]
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    return float(x0), float(y0), float(x1), float(y1)


def square_crop_box(union_xyxy: XYXY, pad_factor: float = PADDING_FACTOR) -> XYXY:
    """One square box centred on ``union_xyxy``'s centre, padded by ``pad_factor``.

    Call :func:`fit_square_to_canvas` with the source dimensions before extracting
    pixels. That keeps the requested padding around the bbox where possible while
    staying entirely inside the original camera canvas.
    """
    x0, y0, x1, y1 = union_xyxy
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * pad_factor / 2.0
    return cx - half, cy - half, cx + half, cy + half


def canonical_crop_box(
    bbox_xyxy: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    pad_factor: float = PADDING_FACTOR,
) -> XYXY:
    """The full SS4.5 box in one call: union -> padded square -> fitted into the canvas.

    This is the arithmetic ``precompute.py --capture-only`` runs to produce the box it
    records in ``capture_latent_manifest.json``, reproduced here so a consumer can *check*
    the recorded box against the current ``bbox.npy``. It is a guard, not a second producer:
    the manifest's box is what the capture latents were encoded with, so on disagreement the
    manifest wins and the caller should report that ``bbox.npy`` has changed underneath it.
    """
    square = square_crop_box(union_bbox(bbox_xyxy, valid), pad_factor)
    return fit_square_to_canvas(square, width, height)


def effective_pad_factor(box_xyxy: XYXY, bbox_xyxy: np.ndarray, valid: np.ndarray) -> float:
    """``side / max(w, h)`` of the subject union -- the padding the canvas actually allowed.

    Equals ``pad_factor`` where the square fits (87.1 % of views), less where the canvas
    capped it (12.9 %), and **below 1.0** where the subject itself is wider than the canvas
    (0.4 %) -- the last case means the box clips the subject and the view should be excluded.
    """
    ux0, uy0, ux1, uy1 = union_bbox(bbox_xyxy, valid)
    union_side = max(ux1 - ux0, uy1 - uy0)
    if union_side <= 0:
        raise ValueError("degenerate subject union")
    return float(box_xyxy[2] - box_xyxy[0]) / float(union_side)


def fit_square_to_canvas(box_xyxy: XYXY, width: int, height: int) -> XYXY:
    """Shift a requested square into the source canvas without synthesizing pixels."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid canvas {width}x{height}")
    x0, y0, x1, y1 = box_xyxy
    requested_side = max(x1 - x0, y1 - y0)
    if requested_side <= 0:
        raise ValueError(f"invalid crop box {box_xyxy}")
    side = min(round(requested_side), width, height)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(round(center_x - side / 2), 0), width - side)
    top = min(max(round(center_y - side / 2), 0), height - side)
    return float(left), float(top), float(left + side), float(top + side)


def crop_from_canvas(frame: np.ndarray, box_xyxy: XYXY) -> np.ndarray:
    """Crop a fitted box without padding; all returned pixels came from ``frame``."""
    x0, y0, x1, y1 = (round(value) for value in box_xyxy)
    height, width = frame.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop {box_xyxy} lies outside canvas {width}x{height}")
    return frame[y0:y1, x0:x1]


def crop_with_padding(
    frame: np.ndarray, box_xyxy: XYXY, pad_value: int | float = 255
) -> np.ndarray:
    """Crop ``frame`` (``H x W`` or ``H x W x C``) to ``box_xyxy``, padding out-of-frame
    regions with ``pad_value``.

    ``box_xyxy`` is allowed to extend past ``frame``'s bounds in any direction -- the box is
    fixed for the whole clip and sized to the clip's own subject extent, not this frame's.
    Pass ``pad_value=255`` for an RGB frame composited onto white (matching the render's own
    white background) and ``pad_value=0`` for a mask/alpha channel (no subject off-frame).
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy)
    out_w, out_h = x1 - x0, y1 - y0
    h, w = frame.shape[:2]
    out = np.full((out_h, out_w, *frame.shape[2:]), pad_value, dtype=frame.dtype)

    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, w), min(y1, h)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out  # box entirely outside the frame

    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out
