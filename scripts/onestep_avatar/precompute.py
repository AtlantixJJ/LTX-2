"""VAE-encode ARGAvatar/capture training windows.

The output root is directly consumable by ``FlexibleStrategy`` once its guided-init
configuration is enabled:

* ``target_latents/`` contains the capture latents (the loss target);
* ``init_latents/`` contains the ARGAvatar-render latents (the noising source);
* ``carryover_masks/`` contains a binary latent-grid mask with only latent frame 1
  enabled.  This is the frozen AR carryover condition.

**Every path uses one continuous VAE encode per source, sliced per window.** Revised
2026-09-11 (plan §4.4): a genuine causal
keyframe only ever exists at latent frame 0 of a truly continuous encode, and nothing
re-keys mid-rollout past a clip's first window -- so independently re-encoding every
window was manufacturing an artificial fresh keyframe the deployed AR rollout never
actually has. ``encode_capture_jobs`` VAE-encodes each source ONCE, continuously
(``crop_source`` decodes the whole needed pixel range in one pass), and slices every
window's latent directly out of that one encode. Window 0 gets a genuine keyframe for
free (the master's own frame 0); every later window gets the same multi-frame slot 0 a
true continuous rollout already has. Verified empirically: window 0 sliced vs.
independently encoded differs by ~0.1% (bf16 noise floor); a mid-clip window differs by
~24% from the old independent-encode approach, confirming the old construction was
wrong, not just redundant.

The paired path (``encode_jobs``, no ``--capture-only``) applies the same rule to the guide
render, and does not re-encode the capture at all: ``z_y`` is copied out of the bundle
``--capture-only`` already wrote for that view. One producer per tensor -- the crop box and
the target latents both come from the capture pass, and nothing downstream re-derives them.

``--capture-only`` writes one consolidated ``ltx_vae_latent.pt`` per source view
(``views/<view>/ltx_vae_latent.pt``, a ``{"windows": {index: record}}`` dict) instead
of one file per window -- a single atomic save per source, never per-window files.

Run from ``LTX-2`` using the ``ltx`` conda environment, for example::

    conda run -n ltx python -m scripts.onestep_avatar.precompute --model 2.5 --gpu-id 0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import logging
import multiprocessing
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.prune.core import ltx_adapter, model_registry, refine_task
from scripts.prune.core.refine_core import WindowGeometry
from scripts.prune.core.session import DTYPE

SCHEMA_VERSION = 1
DEFAULT_CORPUS_ROOT = model_registry.WORKSPACE_ROOT / "data" / "AnimatableHuman" / "DNARenderingVideo"
DEFAULT_OUTPUT_ROOT = model_registry.WORKSPACE_ROOT / "expr" / "onestep_avatar" / "precomputed"
# Each worker holds one source's raw-frame batch (~3000x4096 px, ~1-2 GB) plus its
# accumulated resized crops until the whole source returns. `os.cpu_count()` (e.g. 48
# on this workstation) workers at that footprint can spike host RAM by 50-100+ GB on
# top of other users' jobs, which is exactly what killed the first --capture-only run
# on 2026-09-10. Default low; raise explicitly only after checking `free -h` headroom.
DEFAULT_CROP_WORKERS = 6
# Written by --capture-only at the corpus root; the crop box of record for every view.
CAPTURE_MANIFEST_NAME = "capture_latent_manifest.json"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pair:
    """A guide render plus the capture bundle it is paired with, relative to the corpus root.

    The capture side is the ``ltx_vae_latent.pt`` bundle ``--capture-only`` already wrote, not
    a capture video: ``z_y`` is copied out of it verbatim rather than re-encoded, so the paired
    outputs and the bundle are the same tensors by construction.
    """

    relative_dir: str
    guide: str
    bundle: str
    guide_sha256: str
    bundle_sha256: str


@dataclass(frozen=True)
class WindowJob:
    pair: Pair
    index: int
    start: int
    end: int
    fps: float
    height: int
    width: int

    @property
    def relative_path(self) -> Path:
        return Path(self.pair.relative_dir) / f"window_{self.index:04d}.pt"


@dataclass(frozen=True)
class CaptureSource:
    """A raw DNARendering RGB view whose target latents can be prepared alone."""

    relative_dir: str
    rgb: str
    bbox: str
    rgb_fingerprint: str


@dataclass(frozen=True)
class CaptureJob:
    source: CaptureSource
    index: int
    start: int
    end: int
    fps: float
    box_xyxy: tuple[float, float, float, float]

    @property
    def relative_path(self) -> str:
        """A human-readable window label for logs/errors, not a real output path."""
        return f"{self.source.relative_dir}#window_{self.index:04d}"


def bundle_path(source: CaptureSource) -> Path:
    """The single consolidated latent file for one view -- every window, one file.

    Persisted beside the source view, not in an experiment-output tree, and written
    once per source (whole-source atomic save), never incrementally per window.
    """
    return Path(source.rgb).parent / "ltx_vae_latent.pt"


class VideoReader:
    """Small RGB frame reader over OpenCV, with decord's ``get_batch``/``get_avg_fps`` shape.

    The checked-in ``ltx`` environment has OpenCV but not decord, so this adapter avoids a
    hidden preprocessing-only dependency. Frames come back as ``[F, H, W, C]`` uint8 RGB; the
    ``/127.5 - 1`` normalization is the caller's, matching ``refine_core``'s.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        probe = cv2.VideoCapture(self.path)
        if not probe.isOpened():
            raise ValueError(f"cannot open video {path}")
        self._length = round(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(probe.get(cv2.CAP_PROP_FPS))
        probe.release()
        if self._length <= 0 or self._fps <= 0:
            raise ValueError(f"{path}: invalid frame count ({self._length}) or fps ({self._fps})")

    def __len__(self) -> int:
        return self._length

    def get_avg_fps(self) -> float:
        return self._fps

    def get_batch(self, indices: range | list[int]) -> torch.Tensor:
        wanted = list(indices)
        if not wanted:
            raise ValueError("cannot read an empty frame batch")
        if min(wanted) < 0 or max(wanted) >= self._length:
            raise IndexError(f"frame request [{min(wanted)}, {max(wanted)}] outside 0..{self._length - 1}")
        capture = cv2.VideoCapture(self.path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(wanted))
        frames: dict[int, torch.Tensor] = {}
        for index in range(min(wanted), max(wanted) + 1):
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise ValueError(f"{self.path}: could not decode frame {index}")
            if index in wanted:
                frames[index] = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return torch.stack([frames[index] for index in wanted])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(destination)


def atomic_json_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def manifest_boxes(corpus_root: Path) -> dict[str, tuple[float, float, float, float]]:
    """The crop box this run recorded per view, read back from ``--capture-only``'s manifest.

    Read as plain JSON rather than through the workspace-side ``dataset.CaptureManifest``:
    that module lives in the other tree and the other conda env, and this is four lines.
    """
    path = corpus_root / CAPTURE_MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run --capture-only over this corpus first")
    record = json.loads(path.read_text())
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for window in record["windows"]:
        relative_dir = window["bundle"].rsplit("/", 1)[0]
        boxes.setdefault(relative_dir, tuple(float(v) for v in window["box_xyxy"]))
    return boxes


def discover_pairs(corpus_root: Path) -> list[Pair]:
    """Find views with a guide render, a capture bundle, and a render built at the right box.

    Two rejections, and the difference between them matters:

    * **No bundle** -- the capture pass has not reached this view. Not a pair yet; skipped
      silently, since this stage is meant to chase a capture run that takes days.
    * **A render whose sidecar box disagrees with the manifest** -- the render was built
      against a different pixel region than the capture latents, so the two are not the same
      moment in space and training on them would teach a shift. That is stale output, not
      work in progress, so it raises: the fix is to re-render, and a silent skip would just
      make the pair quietly disappear from the corpus instead.
    """
    boxes = manifest_boxes(corpus_root)
    guides = sorted(corpus_root.glob("Part_*/*/views/*/argavatar_render.mp4"))
    pairs: list[Pair] = []
    stale: list[str] = []
    for guide in guides:
        bundle = guide.with_name("ltx_vae_latent.pt")
        if not bundle.is_file():
            continue
        directory = str(guide.parent.relative_to(corpus_root))

        sidecar = guide.with_name("argavatar_render.json")
        if not sidecar.is_file():
            stale.append(f"{directory} (no argavatar_render.json)")
            continue
        rendered_box = tuple(float(v) for v in json.loads(sidecar.read_text())["crop_box_xyxy"])
        recorded_box = boxes.get(directory)
        if recorded_box is None:
            stale.append(f"{directory} (bundle exists but the manifest has no box for it)")
            continue
        if any(abs(a - b) > 1e-3 for a, b in zip(rendered_box, recorded_box, strict=True)):
            stale.append(f"{directory} (rendered at {rendered_box}, capture encoded {recorded_box})")
            continue

        pairs.append(
            Pair(
                relative_dir=directory,
                guide=str(guide),
                bundle=str(bundle),
                guide_sha256=sha256(guide),
                bundle_sha256=sha256(bundle),
            )
        )
    if stale:
        listed = "\n  ".join(stale[:8])
        raise SystemExit(
            f"{len(stale)} render(s) do not match the crop box their capture latents were "
            f"encoded with:\n  {listed}"
            + (f"\n  ... and {len(stale) - 8} more" if len(stale) > 8 else "")
            + "\nRe-run build_guidance.py --force for these views; it renders into the "
            "manifest's box."
        )
    return pairs


def _fit_square_to_canvas(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Keep the padded bbox square entirely in the original camera canvas.

    The requested bbox padding is retained where possible by shifting the square;
    when it is wider than the canvas it is capped.  No white/constant border is
    ever invented for the capture target.
    """
    x0, y0, x1, y1 = box
    side = min(round(max(x1 - x0, y1 - y0)), width, height)
    if side <= 0:
        raise ValueError(f"invalid square crop {box}")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(round(cx - side / 2), 0), width - side)
    top = min(max(round(cy - side / 2), 0), height - side)
    return float(left), float(top), float(left + side), float(top + side)


def discover_capture_sources(corpus_root: Path, views: set[int]) -> list[CaptureSource]:
    """Discover selected raw views; no processed capture video is required."""
    sources: list[CaptureSource] = []
    for rgb in sorted(corpus_root.glob("Part_*/*/views/view*_cam*/rgb.mp4")):
        name = rgb.parent.name
        try:
            view = int(name.removeprefix("view")[:2])
        except ValueError:
            continue
        bbox = rgb.with_name("bbox.npy")
        if view not in views or not bbox.is_file():
            continue
        sources.append(
            CaptureSource(
                relative_dir=str(rgb.parent.relative_to(corpus_root)),
                rgb=str(rgb),
                bbox=str(bbox),
                rgb_fingerprint=f"size={rgb.stat().st_size};mtime_ns={rgb.stat().st_mtime_ns}",
            )
        )
    return sources


def _capture_box(source: CaptureSource, height: int, width: int, pad_factor: float) -> tuple[float, float, float, float]:
    bbox = np.load(source.bbox, allow_pickle=True).item()
    xyxy = np.asarray(bbox["xyxy"], dtype=np.float64)
    valid = np.asarray(bbox["valid"], dtype=bool) & ~np.isnan(xyxy).any(axis=1)
    if not valid.any():
        raise ValueError(f"{source.relative_dir}: no valid finite bbox")
    union = xyxy[valid]
    x0, y0 = union[:, 0].min(), union[:, 1].min()
    x1, y1 = union[:, 2].max(), union[:, 3].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * pad_factor / 2
    return _fit_square_to_canvas((cx - half, cy - half, cx + half, cy + half), width, height)


def plan_source(source: CaptureSource, geometry: WindowGeometry, pad_factor: float) -> list[CaptureJob]:
    """Plan one source's windows directly from original RGB and bbox coordinates.

    A module-level function so it can run in a worker process: one ``cv2.VideoCapture``
    open plus a single-frame decode per source, which is cheap in isolation but was
    previously done for every source strictly one at a time.
    """
    reader = VideoReader(source.rgb)
    first = reader.get_batch([0])
    _, height, width, _ = first.shape
    box = _capture_box(source, int(height), int(width), pad_factor)
    fps = reader.get_avg_fps()
    return [CaptureJob(source, index, start, end, fps, box) for index, (start, end) in enumerate(geometry.plan(len(reader)))]


def enumerate_capture_jobs(
    sources: list[CaptureSource], geometry: WindowGeometry, pad_factor: float, *, max_workers: int | None = None
) -> list[CaptureJob]:
    """Plan target windows for every source, in parallel.

    Each source only needs one frame decoded to plan its windows, but there can be
    hundreds of sources; running them one at a time serializes hundreds of small
    ``cv2.VideoCapture`` opens for no reason, since sources are independent.
    """
    jobs: list[CaptureJob] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
        for source_jobs in pool.map(plan_source, sources, itertools.repeat(geometry), itertools.repeat(pad_factor)):
            jobs.extend(source_jobs)
    return jobs


def crop_source(source: CaptureSource, last_needed: int, box_xyxy: tuple[float, float, float, float], edge: int) -> np.ndarray:
    """Decode, crop, and resize a source's needed pixel range in one sequential pass.

    Returns ONE array ``(last_needed, edge, edge, 3)`` uint8 -- every frame decoded,
    cropped, and resized exactly once, never duplicated across windows. This is what
    makes the single-encode construction in ``encode_capture_jobs`` possible (revised
    2026-09-11, see the module docstring and plan §4.4): the whole array is VAE-encoded
    once, and every window's latent is *sliced* from that one encode rather than
    re-derived from a re-cropped, re-encoded pixel range.

    One sequential decode pass, not one ``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, ...)``
    seek per window. These sources are h264 with extremely sparse keyframes (often a
    single I-frame at frame 0, all P/B after) -- OpenCV/FFmpeg can only start decoding
    from a keyframe, so seeking to a later start silently re-decodes (and discards)
    every frame from 0 up to that point. It is also more correct: CAP_PROP_POS_FRAMES
    seeking on B-frame content (this stream has B-frames) is a known source of
    off-by-a-few-frames errors in OpenCV; true sequential ``.read()`` decode has no such
    ambiguity.

    Runs in a worker process: pure CPU/numpy, no torch device involved, so many
    sources can be cropped concurrently while the GPU VAE encoder works through
    whichever source finished cropping first.
    """
    x0, y0, x1, y1 = (round(value) for value in box_xyxy)
    capture = cv2.VideoCapture(source.rgb)
    if not capture.isOpened():
        raise ValueError(f"cannot open video {source.rgb}")
    resized_frames: list[np.ndarray] = []
    try:
        for index in range(last_needed):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"{source.rgb}: could not decode frame {index}")
            cropped = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
            if cropped.shape[:2] != (y1 - y0, x1 - x0):
                raise RuntimeError(f"{source.relative_dir} frame {index}: crop escaped source canvas")
            resized_frames.append(cv2.resize(cropped, (edge, edge), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()

    return np.stack(resized_frames)


def write_cropped_capture_video(source: CaptureSource, jobs: list[CaptureJob], frames: np.ndarray, fps: float) -> None:
    """Persist an explicitly requested per-window capture preview beside the source view.

    Not part of the default pipeline (the plan treats latents as the artifact and
    capture video as optional QA), so this is only called when ``--keep-capture-video``
    is passed. ``frames`` is the one continuous array ``crop_source`` returned; each
    window's preview is a slice of it, not a separately decoded crop.
    """
    edge = frames.shape[1]
    for job in jobs:
        crop = frames[job.start : job.end]
        output = Path(source.rgb).parent / "capture_crop" / f"window_{job.index:04d}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}{output.suffix}")
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (edge, edge))
        if not writer.isOpened():
            raise RuntimeError(f"could not open capture-preview writer for {output}")
        try:
            for frame in crop:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        temporary.replace(output)


def write_capture_qa(source: CaptureSource, box: tuple[float, float, float, float], edge: int) -> Path:
    """Write an explicitly requested QA preview; never a training artifact."""
    output = Path(source.rgb).parents[2] / "qa" / f"capture_{Path(source.rgb).parent.name}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}{output.suffix}")
    capture = cv2.VideoCapture(source.rgb)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), capture.get(cv2.CAP_PROP_FPS), (edge, edge)
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"could not open QA reader/writer for {source.rgb}")
    x0, y0, x1, y1 = (round(value) for value in box)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.resize(frame[y0:y1, x0:x1], (edge, edge), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()
    temporary.replace(output)
    return output


def _video_info(path: Path) -> tuple[int, float, int, int]:
    reader = VideoReader(path)
    if len(reader) == 0:
        raise ValueError(f"{path}: no video frames")
    frame = reader.get_batch([0])
    _, height, width, channels = frame.shape
    if channels != 3:
        raise ValueError(f"{path}: expected RGB video, got {channels} channels")
    return len(reader), float(reader.get_avg_fps()), int(height), int(width)


def load_capture_bundle(pair: Pair) -> dict[int, dict[str, object]]:
    """The capture bundle's per-window records, keyed by window index."""
    bundle = torch.load(pair.bundle, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("windows"), dict):
        raise ValueError(f"{pair.relative_dir}: {pair.bundle} is not a capture latent bundle")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{pair.relative_dir}: bundle schema_version={bundle.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}; re-run --capture-only for this view"
        )
    return bundle["windows"]


def enumerate_jobs(pairs: list[Pair], geometry: WindowGeometry) -> list[WindowJob]:
    """Enumerate windows from the guide, and check each against the capture bundle's own plan.

    The window plan is a pure function of frame count, so the guide and the capture agree as
    long as they cover the same frames. Checking the bundle's recorded ``start``/``end`` here
    is what turns "they should agree" into "they do": a render that lost or gained frames
    against its capture is caught before anything is encoded, rather than producing a pair
    whose ``z_g`` and ``z_y`` describe different moments in the clip.
    """
    jobs: list[WindowJob] = []
    for pair in pairs:
        frames, fps, height, width = _video_info(Path(pair.guide))
        if height % geometry.scale_factors.height or width % geometry.scale_factors.width:
            raise ValueError(
                f"{pair.relative_dir}: {width}x{height} is not divisible by the VAE spatial factors "
                f"{geometry.scale_factors.width}x{geometry.scale_factors.height}; rebuild guidance at a valid edge"
            )
        windows = load_capture_bundle(pair)
        plan = list(enumerate(geometry.plan(frames)))
        if len(plan) != len(windows):
            raise ValueError(
                f"unaligned pair {pair.relative_dir}: the guide's {frames} frames plan "
                f"{len(plan)} windows but the capture bundle holds {len(windows)}; the render "
                f"and the capture cover different frame ranges"
            )
        for index, (start, end) in plan:
            record = windows.get(index)
            if record is None:
                raise ValueError(f"{pair.relative_dir}: capture bundle has no window {index}")
            if (record.get("start"), record.get("end")) != (start, end):
                raise ValueError(
                    f"{pair.relative_dir} window {index}: guide plans pixels "
                    f"[{start}:{end}) but the capture bundle records "
                    f"[{record.get('start')}:{record.get('end')})"
                )
            if record.get("fps") != fps:
                raise ValueError(
                    f"{pair.relative_dir} window {index}: guide fps {fps} disagrees with the "
                    f"capture bundle's {record.get('fps')}"
                )
            jobs.append(WindowJob(pair, index, start, end, fps, height, width))
    return jobs


def carryover_mask(latent_frames: int, latent_height: int, latent_width: int) -> torch.Tensor:
    """Return the trainer mask that freezes exactly regular latent frame index 1."""
    if latent_frames < 2:
        raise ValueError("a carryover mask requires a keyframe plus latent frame 1")
    mask = torch.zeros((latent_frames, latent_height, latent_width), dtype=torch.float32)
    mask[1].fill_(1.0)
    return mask


def latent_record(latent: torch.Tensor, *, fps: float) -> dict[str, object]:
    """Serialize one non-patchified latent in the trainer's canonical format."""
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"expected VAE latent [1, C, F, H, W], got {tuple(latent.shape)}")
    _, _channels, frames, height, width = latent.shape
    return {
        "latents": latent.squeeze(0).detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "num_frames": int(frames),
        "height": int(height),
        "width": int(width),
        "fps": float(fps),
    }


def mask_record(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"mask": mask.contiguous()}


def manifest(
    model: model_registry.RefinerModel, geometry: WindowGeometry, pairs: list[Pair], jobs: list[WindowJob]
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "one_step_argavatar_vae_windows",
        "model": {
            "key": model.key,
            "video_vae": model.paths.video_vae(),
            "video_vae_sha256": sha256(Path(model.paths.video_vae())),
            "scale_factors": list(model.scale_factors),
            "scale_factors_source": model.scale_factors_source,
        },
        "geometry": geometry.as_dict(),
        "outputs": {
            "target_latents": "capture loss target z_y, copied verbatim from each view's ltx_vae_latent.pt",
            "init_latents": "ARGAvatar guide noising source z_g, sliced from one continuous encode per render",
            "carryover_masks": "latent frame 1 is clean and excluded from loss",
        },
        "pairs": [asdict(pair) for pair in pairs],
        "windows": [
            {
                "path": str(job.relative_path),
                "start": job.start,
                "end": job.end,
                "fps": job.fps,
                "height": job.height,
                "width": job.width,
            }
            for job in jobs
        ],
    }


def _existing_record_is_current(path: Path, expected: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        record = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return (
        isinstance(record, dict)
        and tuple(record.get("latents", torch.empty(0)).shape) == expected["shape"]
        and record.get("fps") == expected["fps"]
    )


def _bundle_is_current(path: Path, source_jobs: list[CaptureJob], model: model_registry.RefinerModel, edge: int) -> bool:
    """Whole-source check: the bundle is current only if every window in it matches.

    There is no partial resume within a source -- the bundle is one atomic save, so a
    source is either fully current or fully redone.
    """
    if not path.is_file():
        return False
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    if not isinstance(bundle, dict) or not isinstance(bundle.get("windows"), dict):
        return False
    windows = bundle["windows"]
    for job in source_jobs:
        record = windows.get(job.index)
        expected = _expected_capture_shape(model, job, edge)
        if not (
            isinstance(record, dict)
            and tuple(record.get("latents", torch.empty(0)).shape) == expected["shape"]
            and record.get("fps") == expected["fps"]
        ):
            return False
    return True


def encode_jobs(
    model: model_registry.RefinerModel,
    jobs: list[WindowJob],
    output_root: Path,
    *,
    gpu_id: int,
    overwrite: bool,
) -> tuple[int, int]:
    """Write each window's ``z_g`` and ``z_y`` for the paired objective.

    Two rules, both of them the 2026-09-11 revision of plan SS4.4 applied here:

    * ``z_g`` comes from ONE continuous VAE encode of the whole guide render, sliced per
      window -- never an independent per-window encode. Re-encoding each window separately
      manufactures a fresh causal keyframe at every window's local frame 0, which the
      deployed AR rollout never produces past a clip's first window (measured: ~24 % off on a
      mid-clip window). The guide is a continuous video like any other, so it gets the same
      construction ``encode_capture_jobs`` gives the capture.
    * ``z_y`` is COPIED from the capture bundle ``--capture-only`` already wrote. Re-encoding
      the capture here would be a second producer of the same tensor, free to disagree with
      the bundle the rest of the pipeline reads; ``enumerate_jobs`` has already checked that
      the two cover the same frames.
    """
    device = torch.device(f"cuda:{gpu_id}")
    time_scale = model.scale_factors.time
    completed = skipped = 0

    by_pair: dict[str, list[WindowJob]] = {}
    for job in jobs:
        by_pair.setdefault(job.pair.relative_dir, []).append(job)

    def window_paths(job: WindowJob) -> tuple[Path, Path, Path]:
        return (
            output_root / "target_latents" / job.relative_path,
            output_root / "init_latents" / job.relative_path,
            output_root / "carryover_masks" / job.relative_path,
        )

    def expected_shape(job: WindowJob) -> dict[str, object]:
        latent_frames = (job.end - job.start - 1) // time_scale + 1
        return {
            "shape": (
                model.caps.latent_channels,
                latent_frames,
                job.height // model.scale_factors.height,
                job.width // model.scale_factors.width,
            ),
            "fps": job.fps,
        }

    pending: list[list[WindowJob]] = []
    for pair_jobs in by_pair.values():
        outstanding = []
        for job in pair_jobs:
            target_out, init_out, mask_out = window_paths(job)
            expected = expected_shape(job)
            current = all(_existing_record_is_current(path, expected) for path in (target_out, init_out))
            if not overwrite and current and mask_out.is_file():
                skipped += 1
            else:
                outstanding.append(job)
        if outstanding:
            # The whole guide is encoded in one pass, so a pair with any outstanding window
            # re-encodes all of it; only the outstanding windows are written.
            pending.append(outstanding)

    if not pending:
        return completed, skipped

    with ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder:
        for pair_jobs in pending:
            pair = pair_jobs[0].pair
            capture_windows = load_capture_bundle(pair)
            last_needed = max(job.end for job in pair_jobs)

            reader = VideoReader(pair.guide)
            frames = reader.get_batch(range(last_needed))  # [F, H, W, C] uint8
            video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=DTYPE)
            pixels = video / 127.5 - 1.0
            with torch.no_grad():
                master = encoder.tiled_encode(pixels, None)  # ONE encode for the whole guide

            for job in pair_jobs:
                target_out, init_out, mask_out = window_paths(job)
                expected = expected_shape(job)
                first = job.start // time_scale
                latent_frames = (job.end - job.start - 1) // time_scale + 1
                z_g = master[:, :, first : first + latent_frames]
                if tuple(z_g.shape[1:]) != expected["shape"]:
                    raise RuntimeError(
                        f"{job.relative_path}: sliced guide latent {tuple(z_g.shape)} does not "
                        f"match expected {expected['shape']}"
                    )
                target_record = capture_windows[job.index]
                if tuple(target_record["latents"].shape) != expected["shape"]:
                    raise RuntimeError(
                        f"{job.relative_path}: capture bundle latent "
                        f"{tuple(target_record['latents'].shape)} does not match the guide's "
                        f"{expected['shape']}"
                    )
                atomic_torch_save(dict(target_record), target_out)
                atomic_torch_save(latent_record(z_g, fps=job.fps), init_out)
                atomic_torch_save(
                    mask_record(carryover_mask(*expected["shape"][1:])), mask_out
                )
                completed += 1

            del pixels, video, master
            LOGGER.info(
                "encoded guide source=%s windows=%d (1 VAE call), capture copied from %s",
                pair.relative_dir,
                len(pair_jobs),
                Path(pair.bundle).name,
            )
    return completed, skipped


def _expected_capture_shape(model: model_registry.RefinerModel, job: CaptureJob, edge: int) -> dict[str, object]:
    latent_frames = (job.end - job.start - 1) // model.scale_factors.time + 1
    return {
        "shape": (
            model.caps.latent_channels,
            latent_frames,
            edge // model.scale_factors.height,
            edge // model.scale_factors.width,
        ),
        "fps": job.fps,
    }


def encode_capture_jobs(
    model: model_registry.RefinerModel,
    jobs: list[CaptureJob],
    *,
    gpu_id: int,
    edge: int,
    overwrite: bool,
    max_crop_workers: int | None = None,
    keep_capture_video: bool = False,
) -> tuple[int, int]:
    """Write ``z_y`` from ONE continuous per-source VAE encode, sliced per window.

    Revised 2026-09-11 (plan §4.4): a genuine causal keyframe only ever exists at
    latent frame 0 of a truly continuous encode. Window 0 of a source gets one for
    free by construction (the master's own frame 0); every later window's slot 0 is
    naturally a regular multi-frame block, matching what the deployed AR rollout
    already has past its first window -- there is no independent per-window
    re-encode, and therefore no artificial re-keyed frame 0 to manufacture. Verified
    empirically: window 0 sliced vs. independently encoded differs by ~0.1% (bf16
    noise floor); a mid-clip window differs by ~24% from its old independent-encode
    counterpart, confirming the old per-window encoding was fabricating data the
    deployed rollout never produces.

    Cropping (decode + crop + resize, pure CPU/numpy) runs for many sources
    concurrently in a worker pool, each returning ONE array for its whole source
    (never duplicated per window). The single GPU VAE encoder in this process
    encodes that whole array once, slices every window's latent out of the result,
    and moves to the next completed source. The pool is entered before the VAE
    encoder so worker processes never fork after this process has touched CUDA.
    """
    device = torch.device(f"cuda:{gpu_id}")
    time_scale = model.scale_factors.time
    completed = skipped = 0

    by_source_dir: dict[str, list[CaptureJob]] = {}
    for job in jobs:
        by_source_dir.setdefault(job.source.relative_dir, []).append(job)

    pending: list[tuple[CaptureSource, list[CaptureJob]]] = []
    for source_jobs in by_source_dir.values():
        source = source_jobs[0].source
        if any(job.box_xyxy != source_jobs[0].box_xyxy for job in source_jobs):
            raise ValueError(f"{source.relative_dir}: windows of one source must share one fixed crop box")
        if not overwrite and _bundle_is_current(bundle_path(source), source_jobs, model, edge):
            skipped += len(source_jobs)
            continue
        pending.append((source, source_jobs))

    if not pending:
        return completed, skipped

    context = multiprocessing.get_context("spawn")
    with (
        # max_tasks_per_child=1: recycle the worker after every source. Each source's
        # raw-frame batch (~1-2 GB) is bounded per task, but numpy/cv2 do not reliably
        # return freed heap to the OS between tasks in a long-lived worker -- recycling
        # forces the OS to actually reclaim it rather than letting RSS creep.
        concurrent.futures.ProcessPoolExecutor(max_workers=max_crop_workers, mp_context=context, max_tasks_per_child=1) as pool,
        ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder,
    ):
        futures = {
            pool.submit(crop_source, source, max(job.end for job in source_jobs), source_jobs[0].box_xyxy, edge): (
                source,
                source_jobs,
            )
            for source, source_jobs in pending
        }
        for future in concurrent.futures.as_completed(futures):
            source, source_jobs = futures[future]
            frames = future.result()  # (last_needed, edge, edge, 3), one array for the whole source
            if keep_capture_video:
                write_cropped_capture_video(source, source_jobs, frames, source_jobs[0].fps)

            video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=DTYPE)
            pixels = video / 127.5 - 1.0
            with torch.no_grad():
                master = encoder.tiled_encode(pixels, None)  # ONE encode for the whole source

            windows: dict[int, dict[str, object]] = {}
            for job in source_jobs:
                first = job.start // time_scale
                latent_frames = (job.end - job.start - 1) // time_scale + 1
                z_y = master[:, :, first : first + latent_frames]
                expected = _expected_capture_shape(model, job, edge)
                if tuple(z_y.shape[1:]) != expected["shape"]:
                    raise RuntimeError(
                        f"{job.relative_path}: sliced latent {tuple(z_y.shape)} does not match expected {expected['shape']}"
                    )
                record = latent_record(z_y, fps=job.fps)
                record["start"] = job.start
                record["end"] = job.end
                windows[job.index] = record
                completed += 1
            del pixels, master
            # One atomic save per source: the bundle is all-or-nothing, never a partial file.
            atomic_torch_save(
                {"schema_version": SCHEMA_VERSION, "source": source.relative_dir, "windows": windows},
                bundle_path(source),
            )
            LOGGER.info(
                "encoded capture target source=%s windows=%d (1 VAE call) -> %s",
                source.relative_dir,
                len(source_jobs),
                bundle_path(source),
            )
    return completed, skipped


def main() -> int:  # noqa: PLR0912, PLR0915 -- two explicit CLI modes share parser/provenance.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-frames", type=int, default=refine_task.WINDOW_FRAMES)
    parser.add_argument("--overlap-frames", type=int, default=refine_task.OVERLAP_FRAMES)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Encode z_y directly from raw rgb.mp4 crops; persists latents only, no capture video.",
    )
    parser.add_argument("--views", type=int, nargs="+", default=[1, 5], help="Raw capture views for --capture-only.")
    parser.add_argument("--edge", type=int, default=1024, help="Square raw-capture crop edge for --capture-only.")
    parser.add_argument("--pad-factor", type=float, default=1.20, help="BBox padding factor for --capture-only.")
    parser.add_argument("--visualize-qa", type=int, help="Write this many view previews under qa/ and exit.")
    parser.add_argument("--limit", type=int, help="Encode at most this many windows after deterministic ordering.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode every selected window.")
    parser.add_argument(
        "--crop-workers",
        type=int,
        default=DEFAULT_CROP_WORKERS,
        help="Parallel worker processes for window planning/cropping (--capture-only). "
        f"Default: {DEFAULT_CROP_WORKERS} (NOT os.cpu_count() -- see DEFAULT_CROP_WORKERS docstring).",
    )
    parser.add_argument(
        "--keep-capture-video",
        action="store_true",
        help="Also write each window's cropped capture preview under capture_crop/ (--capture-only). QA only.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.edge <= 0 or args.edge % 32:
        parser.error("--edge must be a positive multiple of 32")
    if args.pad_factor < 1:
        parser.error("--pad-factor must be at least 1")
    if args.visualize_qa is not None and args.visualize_qa <= 0:
        parser.error("--visualize-qa must be positive")
    if any(view < 0 or view > 7 for view in args.views):
        parser.error("--views must be in [0, 7]")

    model = model_registry.resolve(args.model)
    geometry = WindowGeometry(args.window_frames, args.overlap_frames, model.scale_factors)
    if args.capture_only:
        sources = discover_capture_sources(args.corpus_root, set(args.views))
        if not sources:
            raise SystemExit(f"No selected raw rgb.mp4 + bbox.npy views under {args.corpus_root}")
        all_jobs = enumerate_capture_jobs(sources, geometry, args.pad_factor, max_workers=args.crop_workers)
        if args.visualize_qa is not None:
            by_source = {job.source.relative_dir: job for job in all_jobs if job.index == 0}
            selected: list[CaptureSource] = []
            actors: set[str] = set()
            for source in sources:
                actor = str(json.loads((Path(source.rgb).parents[2] / "meta.json").read_text())["actor"]["id"])
                if actor not in actors:
                    selected.append(source)
                    actors.add(actor)
                if len(selected) == args.visualize_qa:
                    break
            for source in selected:
                print(write_capture_qa(source, by_source[source.relative_dir].box_xyxy, args.edge))  # noqa: T201
            return 0
        jobs = all_jobs if args.limit is None else all_jobs[: args.limit]
        capture_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "one_step_raw_capture_target_latents",
            "model": {
                "key": model.key,
                "video_vae": model.paths.video_vae(),
                "scale_factors": list(model.scale_factors),
            },
            "geometry": geometry.as_dict(),
            "edge": args.edge,
            "pad_factor": args.pad_factor,
            "views": args.views,
            "sources": [asdict(source) for source in sources],
            "windows": [
                {
                    "bundle": str(Path(job.source.relative_dir) / "ltx_vae_latent.pt"),
                    "index": job.index,
                    "start": job.start,
                    "end": job.end,
                    "box_xyxy": job.box_xyxy,
                }
                for job in all_jobs
            ],
        }
        manifest_path = args.corpus_root / CAPTURE_MANIFEST_NAME
        if args.dry_run:
            print(json.dumps({"sources": len(sources), "windows": len(jobs), "manifest": str(manifest_path)}, indent=2))  # noqa: T201
            return 0
        atomic_json_save(capture_manifest, manifest_path)
        completed, skipped = encode_capture_jobs(
            model,
            jobs,
            gpu_id=args.gpu_id,
            edge=args.edge,
            overwrite=args.overwrite,
            max_crop_workers=args.crop_workers,
            keep_capture_video=args.keep_capture_video,
        )
        print(  # noqa: T201 -- CLI completion summary.
            f"Capture target VAE precompute complete: encoded={completed}, skipped={skipped}, manifest={manifest_path}"
        )
        return 0
    pairs = discover_pairs(args.corpus_root)
    if not pairs:
        raise SystemExit(
            f"No views with both argavatar_render.mp4 and ltx_vae_latent.pt under {args.corpus_root}. "
            "Run --capture-only first, then build_guidance.py and its visual review gate."
        )
    all_jobs = enumerate_jobs(pairs, geometry)
    jobs = all_jobs
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("Pairs exist but none has a full deployment window.")
    # The freeze covers the complete corpus even when this invocation encodes just
    # a review shard.  A later larger --limit can then safely resume the same root.
    frozen_manifest = manifest(model, geometry, pairs, all_jobs)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        current = json.loads(manifest_path.read_text())
        if current != frozen_manifest:
            raise SystemExit(f"{manifest_path} differs from this input set; use a new --output-root or --overwrite.")
    if args.dry_run:
        print(  # noqa: T201 -- CLI's requested machine-readable plan.
            json.dumps(
                {
                    "pairs": len(pairs),
                    "windows": len(jobs),
                    "total_windows": len(all_jobs),
                    "manifest": str(manifest_path),
                },
                indent=2,
            )
        )
        return 0
    atomic_json_save(frozen_manifest, manifest_path)
    completed, skipped = encode_jobs(model, jobs, args.output_root, gpu_id=args.gpu_id, overwrite=args.overwrite)
    print(  # noqa: T201 -- CLI completion summary.
        f"VAE precompute complete: encoded={completed}, skipped={skipped}, manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
