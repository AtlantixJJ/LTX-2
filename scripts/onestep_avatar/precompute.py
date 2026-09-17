"""VAE-encode the ARGAvatar guide and the capture target, one continuous encode per view.

Every product this writes lives **beside the source video**, one per view, and the training
loop reads them directly -- there is no experiment-side latent tree any more:

* ``ltx_vae_latent.pt``            -- the capture master latent ``z_y`` (``--capture-only``);
* ``argavatar_ltx_vae_latent.pt``  -- the guide master latent ``z_g`` (the paired pass);
* ``argavatar_alpha.mp4`` and ``capture_mask_crop.mp4`` -- the two loss masks, stored
  losslessly at 256**2. Readers pool them to their latent geometry on demand; no derived
  latent-grid bundle is persisted.

**One continuous VAE encode per source, and the master is what is stored.** Revised
2026-09-11 (plan SS4.4): a genuine causal keyframe only ever exists at latent frame 0 of a
truly continuous encode, and nothing re-keys mid-rollout past a clip's first window -- so
independently re-encoding every window was manufacturing an artificial fresh keyframe the
deployed AR rollout never has. Verified empirically: window 0 sliced vs. independently
encoded differs by ~0.1 % (bf16 noise floor); a mid-clip window differs by ~24 %.

Revised again 2026-09-14: the per-window **slices are not stored either**. Under SS4.4's
causal block scheme a window is not a unit of anything, so the bundle IS the master and
``train.py`` slices the blocks it wants out of it. That drops the overlap duplication (every
latent frame was written twice, once per overlapping window) and lets the block geometry
change without re-encoding a single source.

``z_y`` has exactly one producer (``--capture-only``) and the paired pass does not re-encode
or copy it: the trainer reads that bundle directly.

Run from ``LTX-2`` using the ``ltx`` conda environment, for example::

    conda run -n ltx python -m scripts.onestep_avatar.precompute \
        --capture-only --objective bg white --model 2.5 --gpu-id 0
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
from typing import TypeVar

import cv2
import numpy as np
import torch

# Aliased: `geometry` is already the parameter name this module uses throughout for a
# WindowGeometry instance -- a different thing entirely (the k2 window plan, not the crop
# box). Importing it bare would shadow that on every function that takes one.
from scripts.onestep_avatar import dataset, mask_video
from scripts.onestep_avatar import geometry as crop_geometry
from scripts.prune.core import ltx_adapter, model_registry, refine_task
from scripts.prune.core.refine_core import WindowGeometry
from scripts.prune.core.session import DTYPE

SCHEMA_VERSION = 1

# Bundle payload version. v2 holds the source's one continuous encode -- the master -- and
# nothing else. Obsolete per-window bundles must be regenerated rather than supported here.
BUNDLE_SCHEMA_VERSION = 2
# Bump whenever the pixels-to-latent contract changes while the bundle's structural schema
# remains compatible. Currency checks require this exact value, so old outputs are regenerated
# rather than accepted merely because their tensor shape happens to match.
ENCODE_CONTRACT_VERSION = 1

# The three per-view products the trainer reads, all beside the source video. There is no
# experiment-output tree any more: ``expr/onestep_avatar/precomputed/`` existed only to hold
# per-window slices, and SS4.4's master latents make it redundant.
# ``dataset`` owns the objective -> filename mapping (SS1.2).
DEFAULT_CORPUS_ROOT = model_registry.WORKSPACE_ROOT / "data" / "AnimatableHuman" / "DNARenderingVideo"
# Provenance only. `expr/onestep_avatar/precomputed/` held the per-window latent tree until
# SS4.4 (2026-09-14); nothing writes latents there any more, and `train.py` does not read it.
DEFAULT_MANIFEST_ROOT = model_registry.WORKSPACE_ROOT / "expr" / "onestep_avatar" / "paired"
# Each worker holds one source's raw-frame batch (~3000x4096 px, ~1-2 GB) plus its
# accumulated resized crops until the whole source returns. `os.cpu_count()` (e.g. 48
# on this workstation) workers at that footprint can spike host RAM by 50-100+ GB on
# top of other users' jobs, which is exactly what killed the first --capture-only run
# on 2026-09-10. Default low; raise explicitly only after checking `free -h` headroom.
DEFAULT_CROP_WORKERS = 6
# Written by --capture-only at the corpus root; the crop box of record for every view.
CAPTURE_MANIFEST_NAME = "capture_latent_manifest.json"
# Written by --capture-only at the corpus root; caches plan_source's per-source output so a
# restart does not re-open and frame-0-decode all 3360 sources before touching a single bundle
# (plan §1.3: ~1.5 h with no bundle written and no log line, on 3 crop workers). Keyed off the
# same rgb_fingerprint (size;mtime_ns) discover_capture_sources already computes, plus the
# geometry/pad-factor that plan_source's output actually depends on -- so a changed source file
# or a changed --pad-factor/--window-frames/--overlap-frames invalidates only that entry.
PLAN_CACHE_NAME = ".capture_plan_cache.json"
LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def rank_slice(items: list[T], rank: int, n_rank: int) -> list[T]:
    """Deterministic disjoint ownership used by both capture and paired passes."""
    return items[rank::n_rank]


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


@dataclass(frozen=True)
class BundleExpectation:
    source: str
    latent_frames: int
    pixel_frames: int
    channels: int
    edge: int
    fps: float
    scale: int
    objective: str
    box_xyxy: tuple[float, float, float, float]
    input_fingerprint: str
    vae_fingerprint: str


def bundle_path(source: CaptureSource, objective: str = dataset.DEFAULT_OBJECTIVE) -> Path:
    """The single consolidated latent file for one view -- the whole clip, one file.

    Persisted beside the source view, not in an experiment-output tree, and written once per
    source (whole-source atomic save). The two objectives (SS1.2) differ in the pixels that
    were encoded -- unmatted capture vs. capture matted to white -- so they are two files,
    never one file reinterpreted.
    """
    return Path(source.rgb).parent / dataset.capture_bundle_name(objective)


def guide_bundle_path(pair: Pair, objective: str = dataset.DEFAULT_OBJECTIVE) -> Path:
    """The guide bundle beside this pair's render; resolve it at every write site."""
    return Path(pair.guide).with_name(dataset.guide_bundle_name(objective))


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


def file_fingerprint(path: Path) -> str:
    """Cheap invalidation fingerprint for large local inputs/checkpoints."""
    stat = path.stat()
    return f"{path.resolve()}:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"


def capture_input_fingerprint(source: CaptureSource, objective: str) -> str:
    fingerprint = source.rgb_fingerprint
    if objective == "white":
        fingerprint += ";mask=" + file_fingerprint(Path(source.rgb).with_name(dataset.CAPTURE_MASK_NAME))
    return fingerprint


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


def discover_pairs(corpus_root: Path, objective: str = dataset.DEFAULT_OBJECTIVE) -> list[Pair]:
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
    render_name = dataset.render_name(objective)
    metadata_name = dataset.render_metadata_name(objective)
    guides = sorted(corpus_root.glob(f"Part_*/*/views/*/{render_name}"))
    pairs: list[Pair] = []
    stale: list[str] = []
    for guide in guides:
        bundle = guide.with_name(dataset.capture_bundle_name(objective))
        if not bundle.is_file():
            continue
        directory = str(guide.parent.relative_to(corpus_root))

        sidecar = guide.with_name(metadata_name)
        if not sidecar.is_file():
            stale.append(f"{directory} (no {metadata_name})")
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


ALPHA_NAME = dataset.ALPHA_NAME
# The stored-mask resolution, matching what build_guidance.py harvests the render's alpha at.
# Both persisted masks live on this grid so they can be compared without a resample.
ALPHA_GRID = 256


def build_capture_mask_video(
    pair: Pair, box_xyxy: tuple[float, float, float, float], *, overwrite: bool
) -> Path:
    """Persist the capture matte at the render-alpha resolution, without a derived grid.

    ``argavatar_alpha.mp4`` and this file are the canonical masks.  Readers pool both to the
    active latent geometry, so changing geometry never requires regenerating a mask artifact.
    """
    alpha_stem = Path(pair.guide).with_name(dataset.ALPHA_STEM)
    if not mask_video.mask_exists(alpha_stem):
        raise SystemExit(
            f"{pair.relative_dir}: {ALPHA_NAME} is missing. Re-run build_guidance.py --force "
            f"for this view; the render's alpha only exists inside its own temp frames."
        )
    crop_stem = Path(pair.guide).with_name(dataset.CAPTURE_MASK_CROP_STEM)
    mask_path = Path(pair.guide).with_name(dataset.CAPTURE_MASK_NAME)
    output = crop_stem.with_suffix(".mp4")
    if overwrite or not mask_video.mask_exists(crop_stem):
        cropped = _read_cropped_masks(mask_path, box_xyxy, ALPHA_GRID, ALPHA_GRID)
        mask_video.write_mask_video(cropped, output)
    return output


def write_capture_mask_qa(
    source: CaptureSource,
    box_xyxy: tuple[float, float, float, float],
    *,
    overwrite: bool,
) -> Path | None:
    """Write view 0 for each part's first five subjects into the subject QA folder."""
    view = Path(source.rgb).parent
    if not view.name.startswith("view00_"):
        return None
    subject = view.parents[1]
    part = subject.parent
    first_five = sorted(path for path in part.iterdir() if (path / "views").is_dir())[:5]
    if subject not in first_five:
        return None
    output = subject / "qa" / dataset.CAPTURE_MASK_CROP_NAME
    if overwrite or not output.is_file():
        cropped = _read_cropped_masks(
            Path(source.rgb).with_name(dataset.CAPTURE_MASK_NAME),
            box_xyxy,
            ALPHA_GRID,
            ALPHA_GRID,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        mask_video.write_mask_video(cropped, output)
    return output


def _read_cropped_masks(
    path: Path, box_xyxy: tuple[float, float, float, float], height: int, width: int
) -> np.ndarray:
    """``mask.mp4`` cropped to the manifest box and pooled to uint8, ONE frame at a time.

    Streaming rather than ``VideoReader.get_batch(range(len(reader)))``: a 3000x4096 mask is
    36 MB a frame, so reading a 150-frame clip in one call costs ~5.5 GB plus another 5.5 GB
    for the stack -- on a box where host-RAM contention is the documented way these jobs hang
    (and where the capture pass is already running). Cropping and pooling each frame as it is
    decoded keeps the whole pass at the size of the output grid.
    """
    x0, y0, x1, y1 = (round(v) for v in box_xyxy)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    pooled = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cropped = frame[y0:y1, x0:x1, 0]
            pooled.append(cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not pooled:
        raise ValueError(f"{path}: decoded no frames")
    return np.stack(pooled)


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


def _capture_box(
    source: CaptureSource, height: int, width: int, pad_factor: float
) -> tuple[float, float, float, float]:
    """This pass is the SINGLE PRODUCER of the crop box (SS1.7), and it computes it with
    ``geometry``'s rule rather than a copy of it.

    Until 2026-09-15 the arithmetic was transcribed here, because ``crop_geometry`` lived in the
    other tree and the two conda envs could not import each other; a test pinned the two
    spellings together. Consolidating the package removed the seam -- and the two were
    verified identical over all 3360 corpus views x 3 canvas shapes before the copy was
    deleted, so no recorded box moves.
    """
    bbox = np.load(source.bbox, allow_pickle=True).item()
    xyxy = np.asarray(bbox["xyxy"], dtype=np.float64)
    try:
        return crop_geometry.canonical_crop_box(xyxy, np.asarray(bbox["valid"]), width, height, pad_factor)
    except ValueError as exc:
        raise ValueError(f"{source.relative_dir}: {exc}") from exc


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
    return [
        CaptureJob(source, index, start, end, fps, box)
        for index, (start, end) in enumerate(geometry.plan(len(reader)))
    ]


def _plan_cache_key(pad_factor: float, geometry: WindowGeometry) -> str:
    return f"{geometry.window_frames}:{geometry.overlap_frames}:{pad_factor}"


def _load_plan_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload.get("entries", {}) if payload.get("schema_version") == SCHEMA_VERSION else {}


def _jobs_from_cache_entry(source: CaptureSource, entry: dict) -> list[CaptureJob]:
    return [
        CaptureJob(
            source, int(record["index"]), int(record["start"]), int(record["end"]), float(record["fps"]),
            tuple(float(v) for v in record["box_xyxy"]),
        )
        for record in entry["jobs"]
    ]


def _cache_entry_from_jobs(source: CaptureSource, jobs: list[CaptureJob], key: str) -> dict:
    return {
        "fingerprint": source.rgb_fingerprint,
        "key": key,
        "jobs": [
            {"index": job.index, "start": job.start, "end": job.end, "fps": job.fps, "box_xyxy": list(job.box_xyxy)}
            for job in jobs
        ],
    }


def enumerate_capture_jobs(
    sources: list[CaptureSource],
    geometry: WindowGeometry,
    pad_factor: float,
    *,
    max_workers: int | None = None,
    cache_path: Path | None = None,
) -> list[CaptureJob]:
    """Plan target windows for every source, in parallel, reusing a fresh on-disk plan cache.

    Each source only needs one frame decoded to plan its windows, but there can be
    hundreds of sources; running them one at a time serializes hundreds of small
    ``cv2.VideoCapture`` opens for no reason, since sources are independent.

    ``cache_path``, when given, is read for entries whose ``fingerprint`` (the source's own
    ``size;mtime_ns``, from ``discover_capture_sources``) and ``key`` (pad factor + window
    geometry -- everything ``plan_source`` actually depends on) still match, and only the
    remaining sources are opened and planned. This is what makes a restart of a killed
    ``--capture-only`` run cheap: without it, every restart re-opens and frame-0-decodes all
    selected sources before a single (already-complete) bundle is skipped.
    """
    key = _plan_cache_key(pad_factor, geometry)
    entries = _load_plan_cache(cache_path) if cache_path is not None else {}

    by_source: dict[str, list[CaptureJob]] = {}
    to_plan: list[CaptureSource] = []
    for source in sources:
        entry = entries.get(source.relative_dir)
        if entry is not None and entry.get("fingerprint") == source.rgb_fingerprint and entry.get("key") == key:
            by_source[source.relative_dir] = _jobs_from_cache_entry(source, entry)
        else:
            to_plan.append(source)

    if to_plan:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
            planned = pool.map(plan_source, to_plan, itertools.repeat(geometry), itertools.repeat(pad_factor))
            for source, source_jobs in zip(to_plan, planned, strict=True):
                by_source[source.relative_dir] = source_jobs
                entries[source.relative_dir] = _cache_entry_from_jobs(source, source_jobs, key)
        if cache_path is not None:
            atomic_json_save({"schema_version": SCHEMA_VERSION, "entries": entries}, cache_path)

    jobs: list[CaptureJob] = []
    for source in sources:
        jobs.extend(by_source[source.relative_dir])
    return jobs


def crop_source(
    source: CaptureSource, last_needed: int, box_xyxy: tuple[float, float, float, float], edge: int,
    objectives: tuple[str, ...] = (dataset.DEFAULT_OBJECTIVE,),
) -> dict[str, np.ndarray]:
    """Decode, crop, and resize a source's needed pixel range in one sequential pass.

    Returns ONE array ``(last_needed, edge, edge, 3)`` uint8 **per requested objective** --
    every frame decoded, cropped, and resized exactly once, never duplicated across windows.
    This is what makes the single-encode construction in ``encode_capture_jobs`` possible
    (revised 2026-09-11, see the module docstring and plan SS1.6): the whole array is
    VAE-encoded once, and every block's latent is *sliced* from that one encode rather than
    re-derived from a re-cropped, re-encoded pixel range.

    Asking for both objectives costs ONE decode of ``rgb.mp4`` (plus one of ``mask.mp4``),
    not two: the expensive part is the sequential h264 decode of a 4096x3000 source, and the
    matte is a per-frame blend over pixels that are already in hand. What it does cost is a
    second uint8 array of the same size resident in this worker (~0.5 GB at 150 frames), so
    ``--crop-workers`` is the knob if host RAM is tight.

    One sequential decode pass, not one ``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, ...)``
    seek per window. These sources are h264 with extremely sparse keyframes (often a
    single I-frame at frame 0, all P/B after) -- OpenCV/FFmpeg can only start decoding
    from a keyframe, so seeking to a later start silently re-decodes (and discards)
    every frame from 0 up to that point. It is also more correct: CAP_PROP_POS_FRAMES
    seeking on B-frame content (this stream has B-frames) is a known source of
    off-by-a-few-frames errors in OpenCV; true sequential ``.read()`` decode has no such
    ambiguity.

    For the ``white`` objective (SS1.2) the capture is **matted to white** here, in the same
    pass and the same crop: ``frame * a + 255 * (1 - a)``, with ``a`` the view's own
    ``mask.mp4``, decoded in lockstep and cropped identically. It is done at full resolution,
    before the resize, so the matte and the pixels are resampled together -- the same reason
    the guide's composite is built while the full-resolution alpha is still live.
    The matte is used CONTINUOUS, not re-thresholded: the stored mask is already a threshold
    off lossy video (risk 8), and hardening it a second time would quantise the silhouette
    edge that this objective makes the whole task.

    Runs in a worker process: pure CPU/numpy, no torch device involved, so many
    sources can be cropped concurrently while the GPU VAE encoder works through
    whichever source finished cropping first.
    """
    unknown = set(objectives) - set(dataset.OBJECTIVES)
    if unknown:
        raise ValueError(f"unknown objectives {sorted(unknown)}")
    x0, y0, x1, y1 = (round(value) for value in box_xyxy)
    capture = cv2.VideoCapture(source.rgb)
    if not capture.isOpened():
        raise ValueError(f"cannot open video {source.rgb}")
    matte = None
    if "white" in objectives:
        mask_path = Path(source.rgb).with_name(dataset.CAPTURE_MASK_NAME)
        matte = cv2.VideoCapture(str(mask_path))
        if not matte.isOpened():
            raise ValueError(f"cannot open matte {mask_path} (needed by objective 'white')")
    out: dict[str, list[np.ndarray]] = {objective: [] for objective in objectives}
    try:
        for index in range(last_needed):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"{source.rgb}: could not decode frame {index}")
            cropped = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
            if cropped.shape[:2] != (y1 - y0, x1 - x0):
                raise RuntimeError(f"{source.relative_dir} frame {index}: crop escaped source canvas")
            if "bg" in out:
                out["bg"].append(cv2.resize(cropped, (edge, edge), interpolation=cv2.INTER_AREA))
            if matte is not None:
                ok_mask, mask_frame = matte.read()
                if not ok_mask:
                    raise ValueError(f"{source.relative_dir}: mask.mp4 ended at frame {index}")
                alpha = (mask_frame[y0:y1, x0:x1, 0].astype(np.float32) / 255.0)[..., None]
                matted = (cropped.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).round()
                matted = matted.clip(0, 255).astype(np.uint8)
                out["white"].append(cv2.resize(matted, (edge, edge), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        if matte is not None:
            matte.release()

    return {objective: np.stack(frames) for objective, frames in out.items()}


def _video_info(path: Path) -> tuple[int, float, int, int]:
    reader = VideoReader(path)
    if len(reader) == 0:
        raise ValueError(f"{path}: no video frames")
    frame = reader.get_batch([0])
    _, height, width, channels = frame.shape
    if channels != 3:
        raise ValueError(f"{path}: expected RGB video, got {channels} channels")
    return len(reader), float(reader.get_avg_fps()), int(height), int(width)



def load_capture_master(pair: Pair) -> dict:
    """Read the capture master, requiring regeneration of obsolete per-window bundles."""
    bundle = torch.load(pair.bundle, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict):
        raise ValueError(f"{pair.relative_dir}: {pair.bundle} is not a capture latent bundle")
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION or "master" not in bundle:
        raise ValueError(
            f"{pair.relative_dir}: bundle schema_version={bundle.get('schema_version')}, expected "
            f"{BUNDLE_SCHEMA_VERSION} with a master latent. Re-run --capture-only for this view"
        )
    return bundle


def master_record(
    master: torch.Tensor,
    *,
    source: str,
    fps: float,
    pixel_frames: int,
    box_xyxy: tuple[float, float, float, float] | None,
    edge: int | None,
    objective: str = dataset.DEFAULT_OBJECTIVE,
    input_fingerprint: str | None = None,
    vae_fingerprint: str | None = None,
) -> dict[str, object]:
    """Serialize one source's ONE continuous encode -- the whole bundle, not a window of it.

    ``objective`` is recorded inside the bundle as well as in its filename (SS1.2), so a
    bundle that has been moved or renamed still says which pixels it encodes.
    """
    if master.ndim != 5 or master.shape[0] != 1:
        raise ValueError(f"expected VAE latent [1, C, F, H, W], got {tuple(master.shape)}")
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "encode_contract_version": ENCODE_CONTRACT_VERSION,
        "source": source,
        "objective": objective,
        "input_fingerprint": input_fingerprint,
        "vae_fingerprint": vae_fingerprint,
        "master": master.squeeze(0).detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "fps": float(fps),
        "pixel_frames": int(pixel_frames),
        "box_xyxy": None if box_xyxy is None else [float(v) for v in box_xyxy],
        "edge": edge,
    }


def check_pair_alignment(pair: Pair, geometry: WindowGeometry) -> dict[str, object]:
    """Check that a guide render and its capture master cover the same frames, and say how many.

    The window-by-window plan comparison this replaces existed to catch a render that had
    lost or gained frames against its capture. That check survives -- it is now a direct
    comparison of the two frame counts, which is the thing the old one was proving.
    """
    frames, fps, height, width = _video_info(Path(pair.guide))
    if height % geometry.scale_factors.height or width % geometry.scale_factors.width:
        raise ValueError(
            f"{pair.relative_dir}: {width}x{height} is not divisible by the VAE spatial factors "
            f"{geometry.scale_factors.width}x{geometry.scale_factors.height}; rebuild guidance at a valid edge"
        )
    capture = load_capture_master(pair)
    capture_frames = int(capture["pixel_frames"])
    if frames < capture_frames:
        raise ValueError(
            f"unaligned pair {pair.relative_dir}: the guide has {frames} frames but the capture "
            f"master was encoded from {capture_frames}; the render and the capture cover "
            f"different frame ranges"
        )
    if capture["fps"] != fps:
        raise ValueError(
            f"{pair.relative_dir}: guide fps {fps} disagrees with the capture bundle's {capture['fps']}"
        )
    return {
        "pixel_frames": capture_frames,
        "fps": fps,
        "height": height,
        "width": width,
        "latent_frames": int(capture["master"].shape[1]),
        "box_xyxy": capture.get("box_xyxy"),
    }


def manifest(
    model: model_registry.RefinerModel, geometry: WindowGeometry, pairs: list[Pair]
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "one_step_argavatar_master_latents",
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "model": {
            "key": model.key,
            "video_vae": model.paths.video_vae(),
            "video_vae_sha256": sha256(Path(model.paths.video_vae())),
            "scale_factors": list(model.scale_factors),
            "scale_factors_source": model.scale_factors_source,
        },
        "geometry": geometry.as_dict(),
        "pairs": [asdict(pair) for pair in pairs],
    }


def _master_bundle_is_current(
    path: Path,
    expected: BundleExpectation,
) -> bool:
    if not path.is_file():
        return False
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    if not isinstance(bundle, dict):
        return False
    master = bundle.get("master")
    stored_box = bundle.get("box_xyxy")
    return (
        bundle.get("schema_version") == BUNDLE_SCHEMA_VERSION
        and bundle.get("encode_contract_version") == ENCODE_CONTRACT_VERSION
        and bundle.get("source") == expected.source
        and bundle.get("objective") == expected.objective
        and bundle.get("pixel_frames") == expected.pixel_frames
        and bundle.get("edge") == expected.edge
        and bundle.get("input_fingerprint") == expected.input_fingerprint
        and bundle.get("vae_fingerprint") == expected.vae_fingerprint
        and isinstance(stored_box, list)
        and len(stored_box) == 4
        and all(
            abs(float(a) - float(b)) <= 1e-3
            for a, b in zip(stored_box, expected.box_xyxy, strict=True)
        )
        and isinstance(master, torch.Tensor)
        and tuple(master.shape)
        == (
            expected.channels,
            expected.latent_frames,
            expected.edge // expected.scale,
            expected.edge // expected.scale,
        )
        and bundle.get("fps") == expected.fps
    )


def encode_pairs(
    model: model_registry.RefinerModel,
    pairs: list[Pair],
    *,
    gpu_id: int,
    overwrite: bool,
    geometry: WindowGeometry,
    boxes: dict[str, tuple[float, float, float, float]],
    objective: str = dataset.DEFAULT_OBJECTIVE,
) -> tuple[int, int, list[str]]:
    """Write each view's guide master latent and its loss-mask grids, beside the render.

    Two rules, both SS4.4's continuous-encode rule applied here:

    * ``z_g`` comes from ONE continuous VAE encode of the whole guide render -- never a
      per-window encode. Re-encoding each window separately manufactures a fresh causal
      keyframe at every window's local frame 0, which the deployed AR rollout never produces
      past a clip's first window (measured: ~24 % off on a mid-clip window).
    * ``z_y`` is not touched at all. The capture pass is its only producer and the trainer
      reads that bundle directly; copying it into a second tree is what the old
      ``target_latents/`` did, and a copy is free to disagree with its original.
    """
    device = torch.device(f"cuda:{gpu_id}")
    completed = skipped = 0
    failed: list[str] = []

    pending: list[tuple[Pair, dict[str, object]]] = []
    vae_fingerprint = file_fingerprint(Path(model.paths.video_vae()))
    for pair in pairs:
        info = check_pair_alignment(pair, geometry)
        guide_bundle = guide_bundle_path(pair, objective)
        current = _master_bundle_is_current(
            guide_bundle,
            BundleExpectation(
                source=pair.relative_dir,
                latent_frames=int(info["latent_frames"]),
                pixel_frames=int(info["pixel_frames"]),
                channels=model.caps.latent_channels,
                edge=int(info["width"]),
                fps=float(info["fps"]),
                scale=model.scale_factors.width,
                objective=objective,
                box_xyxy=boxes[pair.relative_dir],
                input_fingerprint=pair.guide_sha256,
                vae_fingerprint=vae_fingerprint,
            ),
        )
        masks_current = mask_video.mask_exists(Path(pair.guide).with_name(dataset.CAPTURE_MASK_CROP_STEM))
        if current and masks_current and not overwrite:
            skipped += 1
            continue
        pending.append((pair, info))

    if not pending:
        return completed, skipped, failed

    with ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder:
        for pair, info in pending:
            try:
                guide_bundle = guide_bundle_path(pair, objective)
                reader = VideoReader(pair.guide)
                frames = reader.get_batch(range(int(info["pixel_frames"])))  # [F, H, W, C] uint8
                video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=DTYPE)
                pixels = video / 127.5 - 1.0
                with torch.no_grad():
                    master = encoder.tiled_encode(pixels, None)  # ONE encode for the whole guide
                if master.shape[2] != info["latent_frames"]:
                    raise RuntimeError(
                        f"{pair.relative_dir}: guide encoded to {master.shape[2]} latent frames, "
                        f"the capture master has {info['latent_frames']}"
                    )
                atomic_torch_save(
                    master_record(
                        master,
                        source=pair.relative_dir,
                        fps=float(info["fps"]),
                        pixel_frames=int(info["pixel_frames"]),
                        box_xyxy=boxes[pair.relative_dir],
                        edge=int(info["width"]),
                        objective=objective,
                        input_fingerprint=pair.guide_sha256,
                        vae_fingerprint=vae_fingerprint,
                    ),
                    guide_bundle,
                )
                del pixels, video, master

                mask_path = build_capture_mask_video(
                    pair, boxes[pair.relative_dir], overwrite=overwrite
                )
                completed += 1
                LOGGER.info(
                    "encoded guide master source=%s latent_frames=%d (1 VAE call), mask=%s",
                    pair.relative_dir,
                    int(info["latent_frames"]),
                    mask_path,
                )
            except Exception:
                LOGGER.exception("FAILED guide source=%s -- skipping, not aborting the run", pair.relative_dir)
                failed.append(pair.relative_dir)
    return completed, skipped, failed


def encode_capture_jobs(
    model: model_registry.RefinerModel,
    jobs: list[CaptureJob],
    *,
    gpu_id: int,
    edge: int,
    overwrite: bool,
    max_crop_workers: int | None = None,
    objectives: tuple[str, ...] = (dataset.DEFAULT_OBJECTIVE,),
) -> tuple[int, int, list[str]]:
    """Write each source's ``z_y`` as ONE continuous per-source VAE encode -- the master.

    Revised 2026-09-11 (plan SS4.4): a genuine causal keyframe only ever exists at latent
    frame 0 of a truly continuous encode, and nothing re-keys mid-rollout, so encoding each
    window independently was manufacturing data the deployed AR rollout never produces
    (measured: a mid-clip window differed ~24 % from its sliced counterpart). Revised again
    2026-09-14: the slices are not stored either. The bundle IS the master, and the trainer
    slices the blocks it wants out of it -- which is also what lets the block geometry change
    without re-encoding a single source.

    ``jobs`` still carries the window plan because that is what says how many pixel frames a
    source needs decoded and which crop box it uses; it no longer says anything about how the
    latents are stored.

    Cropping (decode + crop + resize, pure CPU/numpy) runs for many sources concurrently in a
    worker pool, each returning ONE array per requested objective for its whole source. The
    single GPU VAE encoder in this process encodes each of those arrays once. The pool is
    entered before the VAE encoder so worker processes never fork after this process has
    touched CUDA.

    **Both objectives in one pass, and resumable per objective** (SS1.6). ``objectives`` is a
    set, not a choice: asking for both pays ONE sequential h264 decode of the source and
    writes two bundles beside it. Currency is then checked **per (source, objective)** and a
    source is only decoded for the objectives it is actually missing -- so re-running after
    adding ``white`` to a corpus already encoded as ``bg`` re-encodes only ``white``, and a
    run killed halfway resumes at whole-bundle granularity with no partial state to repair
    (every bundle is written by one atomic save). Nothing here is objective-ordered: the two
    are independent artifacts of the same decode.
    """
    device = torch.device(f"cuda:{gpu_id}")
    time_scale = model.scale_factors.time
    vae_fingerprint = file_fingerprint(Path(model.paths.video_vae()))
    completed = skipped = 0

    by_source_dir: dict[str, list[CaptureJob]] = {}
    for job in jobs:
        by_source_dir.setdefault(job.source.relative_dir, []).append(job)

    pending: list[tuple[CaptureSource, list[CaptureJob], tuple[str, ...]]] = []
    for source_jobs in by_source_dir.values():
        source = source_jobs[0].source
        if any(job.box_xyxy != source_jobs[0].box_xyxy for job in source_jobs):
            raise ValueError(f"{source.relative_dir}: blocks of one source must share one fixed crop box")
        last_needed = max(job.end for job in source_jobs)
        # Per-objective currency: this is the whole resume story, and it is deliberately a
        # property of what is on disk rather than of a progress file. A bundle is written by
        # one atomic save, so it is either current or absent -- there is no half-done state a
        # restart could inherit.
        needed = tuple(
            objective
            for objective in objectives
            if overwrite
            or not _master_bundle_is_current(
                bundle_path(source, objective),
                BundleExpectation(
                    source=source.relative_dir,
                    latent_frames=(last_needed - 1) // time_scale + 1,
                    pixel_frames=last_needed,
                    channels=model.caps.latent_channels,
                    edge=edge,
                    fps=source_jobs[0].fps,
                    scale=model.scale_factors.width,
                    objective=objective,
                    box_xyxy=source_jobs[0].box_xyxy,
                    input_fingerprint=capture_input_fingerprint(source, objective),
                    vae_fingerprint=vae_fingerprint,
                ),
            )
        )
        skipped += len(source_jobs) * (len(objectives) - len(needed))
        if needed:
            pending.append((source, source_jobs, needed))

    failed: list[str] = []
    if not pending:
        return completed, skipped, failed

    # Bounded, not "submit all ~3000 up front": `concurrent.futures.as_completed(fs)` makes
    # its OWN internal `set(fs)` and holds it for the generator's entire lifetime (CPython's
    # `_base.as_completed`), so even after this loop drops ITS OWN reference to a finished
    # `Future`, `as_completed`'s internal set keeps it (and the ~0.5-0.7 GB raw-frame array
    # cached on it) alive until every future passed to that ONE call has been yielded.
    # Chunking is the actual fix: each chunk gets its own `as_completed` call, fully exhausted
    # (and therefore fully collectible) before the next chunk's futures are submitted.
    chunk_size = max(1, (max_crop_workers or DEFAULT_CROP_WORKERS) * 4)
    context = multiprocessing.get_context("spawn")
    with (
        # max_tasks_per_child=1: recycle the worker after every source. Each source's
        # raw-frame batch (~1-2 GB) is bounded per task, but numpy/cv2 do not reliably
        # return freed heap to the OS between tasks in a long-lived worker.
        concurrent.futures.ProcessPoolExecutor(
            max_workers=max_crop_workers, mp_context=context, max_tasks_per_child=1
        ) as pool,
        ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder,
    ):
        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start : chunk_start + chunk_size]
            futures = {
                pool.submit(
                    crop_source, source, max(job.end for job in source_jobs),
                    source_jobs[0].box_xyxy, edge, needed
                ): (source, source_jobs, needed)
                for source, source_jobs, needed in chunk
            }
            for future in concurrent.futures.as_completed(futures):
                source, source_jobs, needed = futures.pop(future)
                # A single bad source (corrupt video, a degenerate crop box) must not sink a
                # run over 3360 sources that takes days -- and without this, the
                # `run_b2a.sh` supervisor's "restarting in 15s" would loop forever on the
                # exact same source, burning restart cycles while never making progress.
                try:
                    last_needed = max(job.end for job in source_jobs)
                    # One dict entry per objective this source still owes, from ONE decode.
                    by_objective = future.result()  # objective -> (last_needed, edge, edge, 3)
                    for objective in needed:
                        frames = by_objective[objective]
                        video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=DTYPE)
                        pixels = video / 127.5 - 1.0
                        with torch.no_grad():
                            master = encoder.tiled_encode(pixels, None)  # ONE encode per objective
                        expected_frames = (last_needed - 1) // time_scale + 1
                        if master.shape[2] != expected_frames:
                            raise RuntimeError(
                                f"{source.relative_dir}: encoded {master.shape[2]} latent frames, "
                                f"expected {expected_frames} from {last_needed} pixel frames"
                            )
                        record = master_record(
                            master,
                            source=source.relative_dir,
                            fps=source_jobs[0].fps,
                            pixel_frames=last_needed,
                            box_xyxy=source_jobs[0].box_xyxy,
                            edge=edge,
                            objective=objective,
                            input_fingerprint=capture_input_fingerprint(source, objective),
                            vae_fingerprint=vae_fingerprint,
                        )
                        del pixels, video, master
                        # One atomic save per (source, objective): a bundle is all-or-nothing,
                        # which is what makes the currency check above a complete resume rule.
                        atomic_torch_save(record, bundle_path(source, objective))
                        completed += len(source_jobs)
                        LOGGER.info(
                            "encoded capture master source=%s objective=%s latent_frames=%d -> %s",
                            source.relative_dir,
                            objective,
                            (last_needed - 1) // time_scale + 1,
                            bundle_path(source, objective),
                        )
                    del by_objective
                except Exception:
                    LOGGER.exception(
                        "FAILED capture target source=%s -- skipping, not aborting the run", source.relative_dir
                    )
                    failed.append(source.relative_dir)
    return completed, skipped, failed


def main() -> int:  # noqa: PLR0912, PLR0915 -- two explicit CLI modes share parser/provenance.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Zero-based worker rank. Sources/pairs are assigned by items[rank::n_rank].",
    )
    parser.add_argument(
        "--n-rank",
        "--n_rank",
        dest="n_rank",
        type=int,
        default=1,
        help="Number of independent GPU workers sharing this corpus (default: 1).",
    )
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="Where the paired-run provenance manifest is written. NOT a latent tree any more: "
        "since SS4.4 every latent and mask lives beside its source video, and "
        "expr/onestep_avatar/precomputed/ is no longer produced or read by anything.",
    )
    parser.add_argument(
        "--objective",
        nargs="+",
        choices=dataset.OBJECTIVES,
        default=[dataset.DEFAULT_OBJECTIVE],
        help="SS1.2, and a SET rather than a choice -- pass both to build both in one pass. "
        "bg (default): the product -- z_y is the unmatted capture, z_g the composite guide; "
        "the UNSUFFIXED bundle names every artifact on disk already uses. white: z_y is the "
        "capture matted to white and z_g the render on white, in *_white.pt bundles beside "
        "them. `--objective bg white` pays ONE decode per source and writes both; currency "
        "is tracked per (source, objective), so re-running only encodes what is missing and "
        "adding an objective later never re-encodes the one already on disk.",
    )
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
    parser.add_argument(
        "--mask-qa-only",
        action="store_true",
        help="Write the sampled capture-mask QA gallery and exit without loading the VAE. "
        "Requires --capture-only and view 0 in --views.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Encode at most this many whole sources/views after rank sharding. Intended for smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode every selected window.")
    parser.add_argument(
        "--crop-workers",
        type=int,
        default=DEFAULT_CROP_WORKERS,
        help="Parallel worker processes for window planning/cropping (--capture-only). "
        f"Default: {DEFAULT_CROP_WORKERS} (NOT os.cpu_count() -- see DEFAULT_CROP_WORKERS docstring).",
    )
    args = parser.parse_args()
    # Deduplicated and put in a fixed order so a run's provenance does not depend on the
    # order the flags were typed in.
    objectives = tuple(o for o in dataset.OBJECTIVES if o in set(args.objective))
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.n_rank <= 0:
        parser.error("--n-rank must be positive")
    if args.rank < 0 or args.rank >= args.n_rank:
        parser.error("--rank must satisfy 0 <= rank < n_rank")
    if args.edge <= 0 or args.edge % 32:
        parser.error("--edge must be a positive multiple of 32")
    if args.pad_factor < 1:
        parser.error("--pad-factor must be at least 1")
    if any(view < 0 or view > 7 for view in args.views):
        parser.error("--views must be in [0, 7]")
    if args.mask_qa_only and (not args.capture_only or 0 not in args.views):
        parser.error("--mask-qa-only requires --capture-only and view 0 in --views")

    model = model_registry.resolve(args.model)
    geometry = WindowGeometry(args.window_frames, args.overlap_frames, model.scale_factors)
    if args.capture_only:
        sources = discover_capture_sources(args.corpus_root, set(args.views))
        if not sources:
            raise SystemExit(f"No selected raw rgb.mp4 + bbox.npy views under {args.corpus_root}")
        all_jobs = enumerate_capture_jobs(
            sources, geometry, args.pad_factor, max_workers=args.crop_workers,
            cache_path=args.corpus_root / PLAN_CACHE_NAME,
        )
        qa_paths: list[Path] = []
        if args.rank == 0 and not args.dry_run:
            first_job = {job.source.relative_dir: job for job in all_jobs if job.index == 0}
            qa_paths = [
                path
                for source in sources
                if (path := write_capture_mask_qa(
                    source, first_job[source.relative_dir].box_xyxy, overwrite=args.overwrite
                )) is not None
            ]
        if args.mask_qa_only:
            print(  # noqa: T201
                json.dumps(
                    {"capture_mask_qa": [str(path) for path in qa_paths], "dry_run": args.dry_run},
                    indent=2,
                )
            )
            return 0
        # Shard SOURCES, not windows: every window and requested objective for one source must
        # stay on one rank so two GPUs can never target the same atomic bundle.
        rank_sources = rank_slice(sources, args.rank, args.n_rank)
        if args.limit is not None:
            rank_sources = rank_sources[: args.limit]
        rank_dirs = {source.relative_dir for source in rank_sources}
        jobs = [job for job in all_jobs if job.source.relative_dir in rank_dirs]
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
                    "bundle": str(Path(job.source.relative_dir) / dataset.capture_bundle_name(objectives[0])),
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
            print(  # noqa: T201
                json.dumps(
                    {
                        "rank": args.rank,
                        "n_rank": args.n_rank,
                        "total_sources": len(sources),
                        "rank_sources": len(rank_sources),
                        "rank_windows": len(jobs),
                        "manifest": str(manifest_path),
                    },
                    indent=2,
                )
            )
            return 0
        # Every rank writes the same full-corpus manifest through a PID-unique temp file.
        # Identical atomic replaces are safe, while a rank-local manifest would lose the boxes
        # owned by every other GPU and make downstream rendering incomplete.
        atomic_json_save(capture_manifest, manifest_path)
        completed, skipped, failed = encode_capture_jobs(
            model,
            jobs,
            gpu_id=args.gpu_id,
            edge=args.edge,
            overwrite=args.overwrite,
            max_crop_workers=args.crop_workers,
            objectives=objectives,
        )
        print(  # noqa: T201 -- CLI completion summary.
            f"Capture target VAE precompute complete: rank={args.rank}/{args.n_rank}, "
            f"encoded={completed}, skipped={skipped}, failed={len(failed)}, manifest={manifest_path}"
        )
        if failed:
            # A nonzero exit here WOULD make run_b2a.sh's supervisor restart -- and it would
            # hit these exact sources again, forever, since retrying does not fix a corrupt
            # video or a degenerate crop. Exit 0: failures are already logged per-source above
            # (LOGGER.exception) and summarized here; a human decides whether to exclude or fix
            # them, not an automatic retry loop.
            print("failed sources: " + ", ".join(failed[:20]) + (" ..." if len(failed) > 20 else ""))  # noqa: T201
        return 0
    # The paired pass runs once per objective, over that objective's own (guide, capture)
    # views. It is a loop and not a mode: a guide render on white and a composite guide are
    # different videos beside the same view, so there is nothing to share but the code.
    # Per-objective resume comes from encode_pairs' own bundle currency check, exactly as it
    # does for the capture pass.
    totals = {"completed": 0, "skipped": 0}
    failed: list[str] = []
    for objective in objectives:
        all_pairs = discover_pairs(args.corpus_root, objective)
        if not all_pairs:
            raise SystemExit(
                f"No views with both {dataset.render_name(objective)} and "
                f"{dataset.capture_bundle_name(objective)} under {args.corpus_root}. Run "
                f"--capture-only --objective {objective} first, then build_guidance.py "
                f"--objective {objective} and its visual review gate."
            )
        pairs = rank_slice(all_pairs, args.rank, args.n_rank)
        if args.limit is not None:
            pairs = pairs[: args.limit]
        # The freeze covers the complete corpus even when this invocation encodes just a
        # review shard. A later larger --limit can then safely resume the same root.
        frozen_manifest = manifest(model, geometry, all_pairs)
        suffix = "" if objective == dataset.DEFAULT_OBJECTIVE else f".{objective}"
        manifest_path = args.manifest_root / f"manifest{suffix}.json"
        if manifest_path.exists() and not args.overwrite:
            current = json.loads(manifest_path.read_text())
            if current != frozen_manifest:
                raise SystemExit(
                    f"{manifest_path} differs from this input set; use a new --manifest-root or --overwrite."
                )
        if args.dry_run:
            print(  # noqa: T201 -- CLI's requested machine-readable plan.
                json.dumps(
                    {"objective": objective, "pairs": len(pairs), "manifest": str(manifest_path)}, indent=2
                )
            )
            continue
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_save(frozen_manifest, manifest_path)
        completed, skipped, view_failures = encode_pairs(
            model,
            pairs,
            gpu_id=args.gpu_id,
            overwrite=args.overwrite,
            geometry=geometry,
            boxes=manifest_boxes(args.corpus_root),
            objective=objective,
        )
        totals["completed"] += completed
        totals["skipped"] += skipped
        failed.extend(f"{objective}:{view}" for view in view_failures)
    if args.dry_run:
        return 0
    completed, skipped = totals["completed"], totals["skipped"]
    print(  # noqa: T201 -- CLI completion summary.
        f"Guide master VAE precompute complete: encoded={completed}, skipped={skipped}, "
        f"failed={len(failed)}, objectives={','.join(objectives)}, rank={args.rank}/{args.n_rank}"
    )
    if failed:
        print("failed views: " + ", ".join(failed[:20]) + (" ..." if len(failed) > 20 else ""))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
