#!/usr/bin/env python
"""Render ARGAvatar guides -- the B2 stage of
``plans/2026-09-10-ltx25-one-step-argavatar-lora.md``.

**This stage consumes ``precompute.py --capture-only``'s output; it never re-derives it.**
The capture side runs first, encodes each view's target latents from a square crop of the raw
``rgb.mp4``, and records the exact box it used in ``capture_latent_manifest.json``. This script
renders the guide into *that recorded box*. There is therefore one producer of the crop box and
one consumer, rather than two implementations that have to agree -- which is what previously
went wrong (the two used different off-canvas rules and desynced 12.9 % of views).

Per (clip, driving view ``D``):

1. ``motion.build_motion`` converts ``pose3d.npy[D]`` to a ``sam3db``-format motion file
   (SS3.1's three fixes).
2. The crop box comes from ``dataset.CaptureManifest.box_for(D)`` (SS4.5). A view the capture
   pass has not reached yet is skipped as not-yet-ready (and named as an error if ``--clips``
   asked for it explicitly) -- never a cue to compute a box locally. The box is cross-checked
   against ``geometry.canonical_crop_box`` on the current ``bbox.npy``, so a corpus re-ingest
   that moved a bbox is caught here instead of becoming a silently misaligned pair.
3. The avatar is reconstructed once per clip from ``R``'s own frame-0 crops (SS4.1's default
   ``R = {view00, view02, view04}``), each cropped to *that view's own* recorded box.
4. ``D``'s motion is rendered into the *same* box (ARGAvatar's ``render_motion_frame_window``,
   via ``pipeline.render_motion_window`` -- ~19x cheaper than rendering the full frame and
   cropping after).
5. IoU (render alpha vs ``mask.mp4``, cropped/resized identically) is computed here as a cheap
   read-only QA number -- SSB1's decisive alignment check -- and the alpha is persisted
   (``argavatar_alpha.mp4``, lossless gray; see ``mask_video.py``) for SS4.3 row 1's masked
   loss. A legacy ``.npy`` is still read, and ``--migrate-alpha`` converts one.
6. **The guide is composited in pixel space** (SS1.2): ``guide_t = render_t * alpha_t +
   background * (1 - alpha_t)``, using the render's own (continuous, unthresholded) alpha.
   ``--objective`` chooses the background, and that is the ONLY thing it chooses here:
   ``bg`` blends the crop of ``D``'s own ``rgb.mp4`` frame 0 (the product's real background),
   ``white`` blends a white frame (the render on white, for the arm that isolates the subject
   gap). Done here, in the same pass over the RGBA frames, because the full-resolution alpha
   only exists transiently in this loop.
7. The composited result is encoded and persisted, atomically, as
   ``clip.view_dir(D)/`` + ``dataset.render_name(objective)`` -- the exact path/name
   ``LTX-2/scripts/onestep_avatar/precompute.py``'s ``discover_pairs()`` expects as the
   sibling of that view's capture bundle. **This file is the guide, not the raw render** --
   the render on its own background is persisted only when that IS the objective.

``--visualize`` additionally builds ``qa/overlay_view<D>.mp4``, a dissolve of the capture over
the render. The capture track is cropped from ``rgb.mp4`` on the fly with the same recorded
box: no persisted ``capture.mp4`` is needed or produced, since latents are the artifact.

Runs in the ``argavatar`` conda env (needs ARGAvatar's own ``xlib``/``scripts.inference``,
which import cleanly only when the process's cwd is ARGAvatar's repo root -- this script
resolves every workspace-side path to an absolute ``Path`` *before* chdir-ing there).

    conda activate argavatar
    python -m scripts.onestep_avatar.build_guidance --limit 8 --visualize   # review batch first
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.onestep_avatar import dataset, mask_video, geometry, motion, qa
from scripts.onestep_avatar.dataset import ClipRef

DEFAULT_ARGAVATAR_ROOT = Path("/home/jianjinx/data2/ARG-Avatar")
# Not the config's own default (`checkpoints/ARGAvatar-Final.pth`) -- that file's content has
# since diverged from this training run's PT_40000 (different md5, same size). Pin explicitly.
DEFAULT_CHECKPOINT_PATH = Path(
    "/home/jianjinx/data2/SAM3DGS/expr/"
    "MV-UV-RNU-HR-FACRoPE-None-L2-PoseDep-TokenCascade-SH1-DS-GSplat/PT_40000.pth"
)
DEFAULT_CONFIG_NAME = "configs/ARGAvatar-Final.yaml"

# SS4.1 of the 09-05 plan: reconstruction inputs 90 degrees apart, driving views maximally
# distant from every input -- so the render is never trivially good from having "seen" D.
DEFAULT_RECON_VIEWS = (0, 2, 4)      # front, right, back
DEFAULT_DRIVING_VIEWS = (1, 5)       # front-right, back-left

ENCODE_ARGS = ["-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p"]
# Artifact names come from dataset.py, which owns the objective -> filename mapping for
# both trees (SS1.2). Nothing here spells a corpus filename itself.
# The render's own alpha (SS4.4 "alpha comes free"), persisted for SS4.3 row 1's masked loss.
ALPHA_NAME = dataset.ALPHA_NAME          # "argavatar_alpha.mp4"
ALPHA_STEM = dataset.ALPHA_STEM          # suffix-less, for the legacy-.npy fallback
# Alpha is stored as an area-fraction grid, not at full 1024**2: the loss consumes it at the
# VAE's latent resolution (32x32 at this geometry), so persisting 1024**2 would be ~157 MB
# per view to throw 99.9 % of away. 256 is a clean multiple of every plausible latent grid,
# so `precompute.py` can average-pool it down without ever resampling to a non-integer ratio,
# and uint8 (1/255 of a cell's area) is finer than the downsample itself. The grid is 9.8 MB
# raw for a 150-frame clip; stored as lossless gray MP4 (mask_video.py) that is 0.23 MB.
ALPHA_GRID = 256


def read_frames(path: Path):
    """Yield BGR ``HxWxC`` uint8 frames from a video, one at a time (never the whole clip --
    a 3000x4096 clip is ~50 MB/frame in memory, and a 225-frame clip would be several GB)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def encode_frames(frame_dir: Path, out_path: Path, fps: float) -> list[str]:
    """ffmpeg-encode ``%06d.png`` frames in ``frame_dir`` to ``out_path``. Returns the exact
    argv (minus ``ffmpeg`` itself) so callers can record it verbatim in the sidecar metadata."""
    args = [
        "-y", "-loglevel", "error",
        "-framerate", f"{fps:.8f}".rstrip("0").rstrip("."),
        "-i", str(frame_dir / "%06d.png"),
        *ENCODE_ARGS,
        str(out_path),
    ]
    subprocess.run(["ffmpeg", *args], check=True)
    return args


def read_cropped_masks(clip: ClipRef, view_idx: int, box: geometry.XYXY, out_size: int):
    """Stream ``mask.mp4[view_idx]``, cropped to ``box`` and resized -- read-only, for IoU.

    ``crop_from_canvas`` rather than ``crop_with_padding``: a manifest box is inside the
    canvas by construction, so a box that needs padding here did not come from the manifest
    and the IoU would be computed against a different region than the capture latent.
    """
    for mask_bgr in read_frames(clip.mask_path(view_idx)):
        mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        mask_c = geometry.crop_from_canvas(mask, box)
        yield cv2.resize(mask_c, (out_size, out_size), interpolation=cv2.INTER_AREA)


def read_cropped_capture(clip: ClipRef, view_idx: int, box: geometry.XYXY, out_size: int):
    """Stream ``rgb.mp4[view_idx]`` cropped to ``box`` and resized -- the capture track.

    Exactly what ``precompute.py --capture-only`` feeds the VAE, reconstructed on the fly for
    the ``--visualize`` overlay. Nothing is persisted: the capture's durable form is its
    latent bundle, and re-encoding a video of it would only add a second generation of h264.
    """
    for frame_bgr in read_frames(clip.rgb_path(view_idx)):
        cropped = geometry.crop_from_canvas(frame_bgr, box)
        yield cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_AREA)


def build_recon_crop(clip: ClipRef, view_idx: int, out_size: int, box: geometry.XYXY) -> np.ndarray:
    """Frame 0 of ``view_idx``, cropped to *that view's own* recorded box and resized.

    Only frame 0 is needed -- reconstruction is single/multi-photo, not video (SS4.1: R
    views' "frame-0 crops"). The box is the caller's, resolved from the capture manifest, so
    the reconstruction inputs are the same pixel regions the capture latents were built from.
    """
    frame_bgr = next(read_frames(clip.rgb_path(view_idx)))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cropped = geometry.crop_from_canvas(frame_rgb, box)
    return cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _atomic_write_video(frame_dir: Path, output: Path, fps: float) -> list[str]:
    """Encode into a same-directory temp file, then rename -- so a reader (``precompute.py``,
    or a re-run of this script) never observes a half-written ``argavatar_render.mp4``."""
    temp_video = output.with_name(f".{output.stem}.tmp.{os.getpid()}{output.suffix}")
    args = encode_frames(frame_dir, temp_video, fps)
    temp_video.replace(output)
    return args


def composite_guide_frame(
    render_bgr: np.ndarray, alpha: np.ndarray, background_bgr: np.ndarray
) -> np.ndarray:
    """SS1.2's pixel-space composite: ``render * alpha + background * (1 - alpha)``.

    ONE function for both objectives -- they differ only in what ``background_bgr`` is:

    * ``bg``    -- frame 0 of the driving view, cropped to the same box. The product.
    * ``white`` -- a white frame, so the guide is the render on white and the target is the
      capture matted to white. Both sides then agree on the background by construction.

    Writing it as one blend rather than two branches is the point: an objective is a choice
    of background, not a second guide-construction code path that could drift from this one.

    ``alpha`` is the render's own uint8 [0, 255] coverage, used continuous (not thresholded)
    so the boundary is a soft blend rather than a hard-edged 32x-latent-cell quantisation
    (SS1.2's reason 1 against a latent-space blend). Pure and shape-only, so it is unit
    tested directly rather than only through the full render pipeline.
    """
    alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]
    blended = render_bgr.astype(np.float32) * alpha_f + background_bgr.astype(np.float32) * (1.0 - alpha_f)
    return blended.round().clip(0, 255).astype(np.uint8)


def guide_background(
    objective: str, clip: ClipRef, driving_view: int, box: geometry.XYXY, out_size: int
) -> np.ndarray:
    """The frame ``composite_guide_frame`` blends behind the render, for one objective.

    Read once per view, outside the per-frame loop -- it is constant over the clip in both
    objectives (SS1.2: the background is *given*, which is exactly why it does not move).
    """
    if objective == "white":
        return np.full((out_size, out_size, 3), 255, dtype=np.uint8)
    return next(read_cropped_capture(clip, driving_view, box, out_size))


def _atomic_write_json(payload: dict, output: Path) -> None:
    temp_path = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n")
    temp_path.replace(output)


@dataclass(frozen=True)
class BoxOfRecord:
    """The manifest's crop box for one view, plus what it cost to fit it in the canvas."""

    xyxy: geometry.XYXY
    effective_pad_factor: float
    clipped_subject: bool


def resolve_box(
    manifest: dataset.CaptureManifest, clip: ClipRef, view_idx: int, pad_factor: float, out_size: int
) -> BoxOfRecord:
    """The box this view's capture latents were encoded with, validated against ``bbox.npy``.

    The manifest is the authority -- ``precompute.py --capture-only`` already encoded a
    specific pixel region and there is no way to re-negotiate it after the fact. The local
    recomputation here is a *guard*: it catches a corpus re-ingest that moved a bbox, or a
    ``--pad-factor`` that disagrees with the capture pass, at the point where it is still
    unambiguous which side is stale. On disagreement this raises rather than rendering into
    a box the target latents do not share.
    """
    if manifest.pad_factor != pad_factor:
        raise ValueError(
            f"--pad-factor {pad_factor} disagrees with {dataset.CAPTURE_MANIFEST_NAME}'s "
            f"pad_factor={manifest.pad_factor}; the capture latents were encoded at the "
            f"manifest's value, so re-run this stage with --pad-factor {manifest.pad_factor} "
            f"(or re-run the capture pass)"
        )
    if manifest.edge != out_size:
        raise ValueError(
            f"--out-size {out_size} disagrees with {dataset.CAPTURE_MANIFEST_NAME}'s "
            f"edge={manifest.edge}; guide and target must be the same geometry"
        )

    box = manifest.box_for(clip.view_dir(view_idx))
    bbox = np.load(clip.bbox_path(view_idx), allow_pickle=True).item()
    view_meta = clip.view_meta(view_idx)
    expected = geometry.canonical_crop_box(
        bbox["xyxy"], bbox["valid"], view_meta["width"], view_meta["height"], pad_factor
    )
    if any(abs(a - b) > 1e-3 for a, b in zip(box, expected)):
        raise ValueError(
            f"{clip.name} view{view_idx:02d}: {dataset.CAPTURE_MANIFEST_NAME} records crop box "
            f"{box} but the current bbox.npy yields {expected} -- bbox.npy has changed since "
            f"the capture latents were encoded. Re-encode this view before rendering it"
        )

    effective = geometry.effective_pad_factor(box, bbox["xyxy"], bbox["valid"])
    return BoxOfRecord(xyxy=box, effective_pad_factor=effective, clipped_subject=effective < 1.0)


def padded_fraction(box_xyxy: geometry.XYXY, frame_width: float, frame_height: float) -> float:
    """Fraction of the crop box's area lying outside the source frame.

    **Zero for every manifest box**, since the capture pass fits the square into the canvas
    rather than inventing padding. Recorded anyway as a cheap invariant: a non-zero value
    here means the box did not come from the manifest, and the pair is not aligned."""
    x0, y0, x1, y1 = box_xyxy
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return 0.0
    inside_w = max(min(x1, frame_width) - max(x0, 0.0), 0.0)
    inside_h = max(min(y1, frame_height) - max(y0, 0.0), 0.0)
    return 1.0 - (inside_w * inside_h) / area


def _render_is_complete(
    output: Path, metadata_path: Path, expected_frames: int, out_size: int, objective: str
) -> bool:
    """Per-view resumability: a render counts as built only if its sidecar and its video
    agree on frame count and geometry, so a killed ffmpeg never looks finished."""
    if not output.is_file() or not metadata_path.is_file():
        return False
    if not mask_video.mask_exists(output.parent / ALPHA_STEM):
        # A render built before the alpha existed is incomplete, not merely old: SS4.3 row 1's
        # masked loss has no mask without it, and the alpha only exists inside the render's
        # own temp frames, so it cannot be back-filled without re-rendering.
        return False
    try:
        record = json.loads(metadata_path.read_text())
        reader = cv2.VideoCapture(str(output))
        frames = round(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        width = round(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reader.release()
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        record.get("n_frames") == expected_frames
        and record.get("alpha_grid") == ALPHA_GRID
        # SS1.2: which background is behind the render is what an objective IS, so a
        # sidecar that does not name this objective describes a different artifact -- stale
        # in the same sense as a render at the wrong box, and rebuilt rather than re-tagged.
        # A pre-objective sidecar carries ``composited: true`` and no ``objective``; that is
        # exactly the ``bg`` render, so it is accepted as one rather than re-rendered.
        and record.get("objective", "bg" if record.get("composited") else None) == objective
        and (frames, width, height) == (expected_frames, out_size, out_size)
    )


def write_overlay(
    render_video: Path, clip: ClipRef, driving_view: int, box: geometry.XYXY, out_path: Path,
    out_size: int, fps: float,
) -> None:
    """A 0.6/0.4 dissolve of capture-over-render -- misalignment shows as ghosting or doubled
    edges. The capture side is cropped from ``rgb.mp4`` with the manifest box on the fly, so
    the overlay needs no persisted capture video and shows exactly the region the target
    latents were encoded from. Purely a review aid; not part of the training contract."""
    with tempfile.TemporaryDirectory(prefix="overlay_") as tmp:
        tmp_dir = Path(tmp)
        frames = zip(read_frames(render_video), read_cropped_capture(clip, driving_view, box, out_size))
        for i, (render_bgr, capture_bgr) in enumerate(frames, start=1):
            blended = cv2.addWeighted(capture_bgr, 0.6, render_bgr, 0.4, 0.0)
            cv2.imwrite(str(tmp_dir / f"{i:06d}.png"), blended)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_video(tmp_dir, out_path, fps)


IOU_PERCENTILES = (0, 10, 50, 90, 100)


@dataclass
class PairResult:
    clip_name: str
    driving_view: int
    box_xyxy: tuple
    fps: float
    n_frames: int
    iou: dict[int, float]  # percentile -> value, keys = IOU_PERCENTILES
    render_path: str


def render_pair(
    clip: ClipRef, driving_view: int, box_record: BoxOfRecord, avatar, pipeline, out_size: int,
    pad_factor: float, force: bool, objective: str = dataset.DEFAULT_OBJECTIVE,
) -> PairResult:
    pose3d = np.load(clip.pose3d_path(driving_view), allow_pickle=True).item()
    bbox = np.load(clip.bbox_path(driving_view), allow_pickle=True).item()
    box = box_record.xyxy  # the manifest's, never recomputed here
    fps = clip.fps()
    expected_frames = clip.n_frames()

    view_dir = clip.view_dir(driving_view)
    output = view_dir / dataset.render_name(objective)
    metadata_path = view_dir / dataset.render_metadata_name(objective)

    if not force and _render_is_complete(output, metadata_path, expected_frames, out_size, objective):
        record = json.loads(metadata_path.read_text())
        iou = {p: record[f"iou_p{p}"] for p in IOU_PERCENTILES}
        return PairResult(
            clip_name=clip.name, driving_view=driving_view, box_xyxy=tuple(record["crop_box_xyxy"]),
            fps=fps, n_frames=record["n_frames"], iou=iou, render_path=str(output),
        )

    # -- S1: motion file (SS3.1 conversions) -------------------------------------------------
    view_meta = clip.view_meta(driving_view)
    sam3db = motion.build_motion(pose3d, bbox, frame_height=view_meta["height"])

    with tempfile.TemporaryDirectory(prefix=f"{clip.name}_view{driving_view:02d}_") as tmp:
        tmp_dir = Path(tmp)
        motion_path = tmp_dir / "motion.pth"
        torch.save(sam3db, motion_path)

        # -- S2: render D into the box (ARGAvatar, via pipeline.render_motion_window) --------
        render_frames_dir = tmp_dir / "render_frames"
        pipeline.render_motion_window(avatar, str(motion_path), box, (out_size, out_size), str(render_frames_dir))
        render_paths = sorted(render_frames_dir.glob("*.png"))
        if len(render_paths) != expected_frames:
            raise RuntimeError(
                f"{clip.name} view{driving_view:02d}: rendered {len(render_paths)} frames, "
                f"meta.json says {expected_frames}"
            )

        # The background this objective blends behind the render (SS1.2). Constant over the
        # clip in both objectives, so it is read once, outside the per-frame loop.
        background_bgr = guide_background(objective, clip, driving_view, box, out_size)

        # -- QA + alpha + composite: one pass over the RGBA frames does all three -----------
        # The render alpha vs mask.mp4 IoU is SSB1's decisive number; the alpha channel is
        # also what the masked loss (SS4.3 row 1) needs and what SS3.4's composite blends
        # with. Reading the PNGs a second time for any of these would cost another full
        # decode of the window for nothing.
        ious = []
        alpha_grid = np.empty((len(render_paths), ALPHA_GRID, ALPHA_GRID), dtype=np.uint8)
        # strict=True: `alpha_grid` is np.empty, so a mask.mp4 with fewer frames than the
        # render would leave its tail UNINITIALISED and then persist that as the view's
        # alpha -- silently, since the MP4 would look complete. A frame-count disagreement
        # between rgb.mp4 and mask.mp4 is a corpus defect; raising here makes it this pair's
        # exclusion (main() catches per pair) instead of a poisoned mask.
        for i, (r_path, mask_r) in enumerate(
            zip(render_paths, read_cropped_masks(clip, driving_view, box, out_size), strict=True)
        ):
            render_rgba = cv2.imread(str(r_path), cv2.IMREAD_UNCHANGED)
            alpha = render_rgba[..., 3]
            ious.append(qa.mask_iou(alpha, mask_r))
            # INTER_AREA is the area fraction of alpha inside each destination cell -- the
            # right reduction for a coverage mask, and the same one the capture crop uses.
            alpha_grid[i] = cv2.resize(alpha, (ALPHA_GRID, ALPHA_GRID), interpolation=cv2.INTER_AREA)

            # Built HERE, in pixel space, while the full-resolution alpha is still live --
            # deferring this to a later pass would cost a full re-render (SS4.5). Overwriting
            # the PNG in place is what the encoder below then picks up -- a composited frame
            # IS the guide, not a second artifact.
            composite = composite_guide_frame(render_rgba[..., :3], alpha, background_bgr)
            cv2.imwrite(str(r_path), composite)
        ious = np.asarray(ious, dtype=np.float64)

        # Lossless gray MP4, not a raw array: ~42x smaller, bit-exact, and the soft
        # silhouette edge (1.4 % of pixels, and the part that matters) survives untouched.
        # See mask_video.py for the measurements and why lossy was rejected.
        mask_video.write_mask_video(alpha_grid, view_dir / ALPHA_NAME, fps=round(fps))
        encode_args = _atomic_write_video(render_frames_dir, output, fps)

    x0, y0, x1, y1 = box
    iou = {p: float(np.percentile(ious, p)) for p in IOU_PERCENTILES}
    metadata = {
        "clip": clip.name,
        "driving_view": driving_view,
        "crop_box_xyxy": [x0, y0, x1, y1],
        "crop_box_source": dataset.CAPTURE_MANIFEST_NAME,
        "padding_factor": pad_factor,
        # What the canvas actually allowed: == padding_factor where the square fits, less
        # where it was capped, and < 1.0 where the subject itself does not fit (SS4.5's
        # 0.4 % tail, which must be excluded from training rather than trained on).
        "effective_padding_factor": box_record.effective_pad_factor,
        "clipped_subject": box_record.clipped_subject,
        "out_size": out_size,
        "alpha": ALPHA_NAME,
        # Lossless gray MP4 since 2026-09-15 (mask_video.py). Recorded so a reader can tell
        # a migrated view from one that predates the format without opening the file.
        "alpha_format": "mp4_lossless_gray",
        "alpha_grid": ALPHA_GRID,
        # SS1.2: which objective this guide was built for -- i.e. what sits behind the
        # render. Checked by _render_is_complete, so a guide built for the other objective
        # is rebuilt rather than silently trained against.
        "objective": objective,
        "fps": fps,
        "n_frames": len(ious),
        "padded_fraction": padded_fraction(box, view_meta["width"], view_meta["height"]),
        **{f"iou_p{p}": v for p, v in iou.items()},
        "render_encode_invocation": ["ffmpeg", *encode_args],
    }
    _atomic_write_json(metadata, metadata_path)

    return PairResult(
        clip_name=clip.name, driving_view=driving_view, box_xyxy=box, fps=fps,
        n_frames=metadata["n_frames"], iou=iou, render_path=str(output),
    )


def build_pipeline(argavatar_root: Path, checkpoint_path: Path, device_str: str):
    """Import and construct ``ARGAvatarPipeline`` -- must run with cwd == argavatar_root, so
    the caller has already chdir'd there before calling this (importing ARGAvatar's own
    ``scripts.inference`` submodule requires it on ``sys.path``, done by the caller too)."""
    from scripts.inference.pipeline import ARGAvatarPipeline

    device = torch.device(device_str)
    return ARGAvatarPipeline.build(
        config=str(argavatar_root / DEFAULT_CONFIG_NAME),
        checkpoint_path=str(checkpoint_path),
        device=device,
    )


def reconstruct_avatar(
    clip: ClipRef, recon_views, manifest: dataset.CaptureManifest, pipeline, out_size: int,
    pad_factor: float, work_dir: Path,
):
    """Reconstruct from ``R``'s frame-0 crops, each at that view's own manifest box.

    The reconstruction views are capture sources too, so their boxes are on record like any
    other -- resolving them the same way keeps a single crop rule across the whole stage.
    """
    image_paths = []
    for view_idx in recon_views:
        box_record = resolve_box(manifest, clip, view_idx, pad_factor, out_size)
        cropped = build_recon_crop(clip, view_idx, out_size, box_record.xyxy)
        path = work_dir / f"recon_view{view_idx:02d}.png"
        Image.fromarray(cropped).save(path)
        image_paths.append(str(path))
    return pipeline.reconstruct(image_paths, name=clip.name)


def migrate_alpha(corpus_root: Path, *, prune: bool, dry_run: bool) -> dict[str, int]:
    """Re-encode legacy ``argavatar_alpha.npy`` grids as lossless MP4, verifying each one.

    The round trip is bit-exact (see ``mask_video``), so this is a pure storage migration --
    but it is *verified* rather than trusted: the MP4 is decoded and compared against the
    array it came from before anything is removed, and a mismatch leaves both files in place
    and is counted as a failure. ``--prune-npy`` is what actually deletes, and only after
    that check passes, because the raw grids are 42x the size and re-deriving one costs a
    full re-render.
    """
    counts = {"converted": 0, "already_mp4": 0, "failed": 0, "removed_npy": 0}
    for legacy in sorted(corpus_root.glob(f"Part_*/*/views/*/{dataset.ALPHA_STEM}.npy")):
        video = legacy.with_suffix(".mp4")
        if video.is_file():
            # Counted the same way whether or not this is a dry run: a dry run that reported
            # an already-migrated view as a pending "conversion" would not describe the run it
            # is previewing, which is the only thing it is for.
            counts["already_mp4"] += 1
            if prune and not dry_run:
                legacy.unlink()
                counts["removed_npy"] += 1
            continue
        try:
            grid = np.load(legacy)
            if dry_run:
                counts["converted"] += 1
                continue
            mask_video.write_mask_video(grid, video)
            if not np.array_equal(mask_video.read_mask_video(video), grid):
                video.unlink(missing_ok=True)
                raise ValueError("round trip was not bit-exact")
            counts["converted"] += 1
            if prune:
                legacy.unlink()
                counts["removed_npy"] += 1
        except Exception as exc:  # noqa: BLE001 -- one bad view must not sink the migration.
            print(f"FAILED {legacy}: {exc}")
            counts["failed"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-root", type=Path, default=dataset.DEFAULT_CORPUS_ROOT)
    p.add_argument("--clips", type=str, default=None, help="comma-separated clip names (e.g. Part_1_0008_01); default: all done clips")
    p.add_argument("--recon-views", type=int, nargs="+", default=list(DEFAULT_RECON_VIEWS))
    p.add_argument("--driving-views", type=int, nargs="+", default=list(DEFAULT_DRIVING_VIEWS))
    p.add_argument("--out-size", type=int, default=geometry.OUT_SIZE)
    p.add_argument("--pad-factor", type=float, default=geometry.PADDING_FACTOR)
    p.add_argument("--limit", type=int, default=0, help="stop after this many (clip, view) pairs; 0 = no limit")
    p.add_argument("--visualize", action="store_true", help="also build qa/overlay_view<D>.mp4 (capture cropped from rgb.mp4 on the fly)")
    p.add_argument("--force", action="store_true", help="rebuild pairs whose guide render already exists")
    p.add_argument(
        "--objective",
        choices=dataset.OBJECTIVES,
        default=dataset.DEFAULT_OBJECTIVE,
        help="SS1.2. bg (default): the product -- the render composited over the clip's real "
        "first frame, written as argavatar_render.mp4. white: the render on white, written "
        "as argavatar_render_white.mp4, whose paired target is the capture matted to white "
        "(precompute.py --objective white). The two are separate artifacts beside the same "
        "view, so building one never invalidates the other.",
    )
    p.add_argument("--argavatar-root", type=Path, default=DEFAULT_ARGAVATAR_ROOT)
    p.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dry-run", action="store_true", help="print the plan (clips x views) and exit, no GPU work")
    p.add_argument(
        "--migrate-alpha",
        action="store_true",
        help="Re-encode legacy argavatar_alpha.npy grids as lossless MP4 (~42x smaller, "
        "bit-exact) and exit. No GPU, no renderer, no ARGAvatar import. Verifies each round "
        "trip before counting it; add --prune-npy to delete the originals afterwards.",
    )
    p.add_argument(
        "--prune-npy",
        action="store_true",
        help="With --migrate-alpha: delete each .npy once its MP4 is verified bit-exact.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.migrate_alpha:
        # Deliberately before every import-heavy step below: this needs no renderer, no GPU,
        # and no cwd change, so it must run on a machine that has none of them.
        counts = migrate_alpha(
            args.corpus_root.resolve(), prune=args.prune_npy, dry_run=args.dry_run
        )
        print(json.dumps({**counts, "dry_run": args.dry_run}, indent=2))
        return
    # Resolved before the chdir below (ARGAvatar's own imports need cwd == its repo root) --
    # a relative --corpus-root would otherwise silently re-resolve against the wrong directory.
    args.corpus_root = args.corpus_root.resolve()

    # The crop box of record. Loading it here rather than per-pair both fails fast on a corpus
    # the capture pass has never run over, and keeps the 10 MB parse to once per run.
    manifest = dataset.CaptureManifest.load(args.corpus_root)

    if args.clips:
        names = set(args.clips.split(","))
        clips = [c for c in dataset.list_clips(args.corpus_root) if c.name in names]
    else:
        clips = dataset.list_clips(args.corpus_root)

    def is_done(clip: ClipRef, d: int) -> bool:
        view_dir = clip.view_dir(d)
        return _render_is_complete(
            view_dir / dataset.render_name(args.objective),
            view_dir / dataset.render_metadata_name(args.objective),
            clip.n_frames(),
            args.out_size,
            args.objective,
        )

    def is_encoded(clip: ClipRef, d: int) -> bool:
        """Every view this clip needs -- driving *and* reconstruction -- must be on record."""
        return all(manifest.has(clip.view_dir(v)) for v in (d, *args.recon_views))

    all_pairs = [(c, d) for c in clips for d in args.driving_views]
    # A pair whose capture latents do not exist yet is not an error here, only not-yet-ready:
    # the capture pass runs for days over the whole corpus and this stage is meant to chase it.
    # (Asking for a specific --clips set that is not ready IS an error -- see below.)
    ready_pairs = [(c, d) for c, d in all_pairs if is_encoded(c, d)]
    waiting = len(all_pairs) - len(ready_pairs)
    if args.clips and waiting:
        not_ready = [f"{c.name} view{d:02d}" for c, d in all_pairs if not is_encoded(c, d)]
        raise SystemExit(
            f"--clips selected {len(not_ready)} (clip, view) pairs the capture pass has not "
            f"encoded yet: {', '.join(not_ready[:8])}{' ...' if len(not_ready) > 8 else ''}. "
            f"Run `precompute.py --capture-only` over them first"
        )

    skipped = 0 if args.force else sum(1 for c, d in ready_pairs if is_done(c, d))
    pending_pairs = ready_pairs if args.force else [(c, d) for c, d in ready_pairs if not is_done(c, d)]
    if args.limit:
        pending_pairs = pending_pairs[: args.limit]

    print(
        f"{len(clips)} clips selected, {len(all_pairs)} (clip, driving view) pairs total, "
        f"{waiting} waiting on capture latents, {skipped} already built, "
        f"{len(pending_pairs)} pending"
    )
    if args.dry_run:
        for c, d in pending_pairs:
            print(f"  {c.name} view{d:02d}")
        return
    if not pending_pairs:
        print("nothing to do (use --force to rebuild existing pairs)")
        return

    if not args.argavatar_root.is_dir():
        raise FileNotFoundError(f"--argavatar-root {args.argavatar_root} does not exist")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"--checkpoint-path {args.checkpoint_path} does not exist")

    argavatar_root = args.argavatar_root.resolve()
    checkpoint_path = args.checkpoint_path.resolve()

    # ARGAvatar's own modules assume cwd == its repo root (relative config/checkpoint
    # fallbacks, `sys.path.insert(0, ".")` in its own scripts) -- resolve every
    # workspace-side path to absolute first, then chdir.
    os.chdir(argavatar_root)
    sys.path.insert(0, str(argavatar_root))

    pipeline = build_pipeline(argavatar_root, checkpoint_path, args.device)

    by_clip: dict[str, list[int]] = {}
    for clip, driving_view in pending_pairs:
        by_clip.setdefault(clip.name, []).append(driving_view)

    done = 0
    failed: list[str] = []
    for clip in clips:
        pending = by_clip.get(clip.name)
        if not pending:
            continue

        # A batch of `--limit N` pairs must not die on the first bad one: a single clip's
        # pose-tracking gap (build_motion's own data-quality gate) or a reconstruction failure
        # is a per-clip exclusion, the same way SS4.5's 0.4 % clipped-subject tail is -- not a
        # reason to lose every remaining pair the review batch was meant to produce. Failures
        # are collected and reported, never silently swallowed.
        try:
            with tempfile.TemporaryDirectory(prefix=f"{clip.name}_recon_") as recon_tmp:
                avatar = reconstruct_avatar(
                    clip, args.recon_views, manifest, pipeline, args.out_size, args.pad_factor, Path(recon_tmp)
                )
                for d in pending:
                    box_record = resolve_box(manifest, clip, d, args.pad_factor, args.out_size)
                    try:
                        result = render_pair(
                            clip, d, box_record, avatar, pipeline, args.out_size, args.pad_factor,
                            args.force, args.objective,
                        )
                        iou_str = " ".join(f"p{p}={v:.3f}" for p, v in sorted(result.iou.items()))
                        clipped = " CLIPPED-SUBJECT" if box_record.clipped_subject else ""
                        print(
                            f"{clip.name} view{d:02d}: IoU[{iou_str}] ({result.n_frames} frames, "
                            f"pad={box_record.effective_pad_factor:.3f}{clipped}) -> {result.render_path}"
                        )
                        if args.visualize:
                            overlay_path = clip.dir / "qa" / f"overlay_view{d:02d}.mp4"
                            write_overlay(
                                Path(result.render_path), clip, d, box_record.xyxy, overlay_path,
                                args.out_size, result.fps,
                            )
                        done += 1
                    except Exception as exc:  # noqa: BLE001 -- one bad pair must not sink the batch.
                        label = f"{clip.name} view{d:02d}"
                        print(f"SKIPPED {label}: {exc}")
                        failed.append(label)
        except Exception as exc:  # noqa: BLE001 -- reconstruction failure excludes the whole clip.
            for d in pending:
                label = f"{clip.name} view{d:02d}"
                print(f"SKIPPED {label} (reconstruction failed): {exc}")
                failed.append(label)

    print(f"done: {done} pairs rendered, {skipped} already existed, {len(failed)} skipped (use --force to rebuild)")
    if failed:
        print("skipped pairs: " + ", ".join(failed))


if __name__ == "__main__":
    main()
