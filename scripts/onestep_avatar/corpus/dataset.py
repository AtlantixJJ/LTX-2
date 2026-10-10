"""The DNARenderingVideo corpus layout, artifact names and checked readers.

See doc/corpus/dataset.md. ``data/AnimatableHuman/DNARenderingVideo/`` is a curated,
identically-laid-out copy of ``data/DNARendering/Processed/`` -- same per-clip files
(``cameras.npy``, ``meta.json``, ``views/*/{rgb,mask}.mp4`` +
``bbox/keypoints2d/pose3d.npy``). The corpus root is one constant here, so another corpus
with the same layout is a ``root=`` argument, not a second code path.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch

from scripts.onestep_avatar import WORKSPACE_ROOT

T = TypeVar("T")

DEFAULT_CORPUS_ROOT = WORKSPACE_ROOT / "data" / "AnimatableHuman" / "DNARenderingVideo"

# Written by ``LTX-2/scripts/onestep_avatar/corpus/precompute.py --process_gt_latent`` at the corpus
# root. It is the **single producer** of the crop box: the box it records is the one
# the capture target latents were actually encoded with, so every other stage reads it rather
# than recomputing a box of its own.
CAPTURE_MANIFEST_NAME = "capture_latent_manifest.json"

# -- The two objectives, and the per-objective artifact names ---------------------------
#
# This module is the only spelling of these names. The task splits in two, and the split
# reaches the corpus as exactly two artifacts, not as two pipelines:
#
#   * ``bg``    -- the product. Guide = render composited over the clip's real first frame;
#                  target = the capture, unmatted, with its real background.
#   * ``white`` -- both sides on white. Guide = the render on the renderer's own white
#                  background (no compositing at all); target = the capture matted to white.
#                  Isolates the subject-texture gap from the background question entirely.
#
# Everything else is objective-independent and is deliberately NOT suffixed: the render's
# alpha and the mask coverage grids are the same tensors either way (the render is the
# same render; only what sits behind it differs), so suffixing them would manufacture two
# copies of one thing -- two producers of one artifact.
#
# ``bg`` maps to the UNSUFFIXED names, so existing ``bg`` artifacts keep their names and the
# ``white`` objective adds files beside them.
OBJECTIVES = ("bg", "white")
DEFAULT_OBJECTIVE = "bg"

# The RGB/alpha contract ``build_guidance.composite_guide_frame`` implements, stamped into
# every render sidecar's ``compositing_version`` and checked by ``_render_is_complete`` the
# same way ``objective`` is -- a render built under a different contract describes a
# different artifact and must be rebuilt, not silently trained against.
#
# Version 2 is background REPLACEMENT: ``R_white + (1 - alpha) * (B - white)``, which
# algebraically undoes the renderer's white compositing instead of compositing a second time
# (a second blend would apply alpha twice and leave a white fringe on every soft edge).
# Identity for ``white`` within rounding. A sidecar without this exact version is rebuilt.
GUIDE_COMPOSITING_VERSION = 2

# The side (pixels) both persisted coverage grids are stored/read at. One spelling for the
# producer (precompute.py's alpha/mask crop) and the consumer (build_guidance.py's alpha
# render) to agree on.
ALPHA_GRID = 256

# Both persisted coverage masks are LOSSLESS grayscale MP4 (see mask_video.py): ~42x smaller
# than raw arrays, bit-exact, and readable by the same OpenCV path as every
# other video here. All three are objective-independent -- the render is the same render and
# the matte the same matte; only what sits behind the subject differs between objectives.
ALPHA_STEM = "argavatar_alpha"  # suffix-less name for required MP4
ALPHA_NAME = f"{ALPHA_STEM}.mp4"  # the render's own alpha, ALPHA_GRID**2
CAPTURE_MASK_CROP_STEM = "capture_mask_crop"  # mask.mp4 cropped to the box, ALPHA_GRID**2
CAPTURE_MASK_CROP_NAME = f"{CAPTURE_MASK_CROP_STEM}.mp4"
CAPTURE_MASK_NAME = "mask.mp4"  # the dataset's own, full resolution


def _suffix(objective: str) -> str:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    return "" if objective == DEFAULT_OBJECTIVE else f"_{objective}"


def render_name(objective: str = DEFAULT_OBJECTIVE) -> str:
    """The guide video for one objective, written beside its driving view."""
    return f"argavatar_render{_suffix(objective)}.mp4"


def render_metadata_name(objective: str = DEFAULT_OBJECTIVE) -> str:
    return f"argavatar_render{_suffix(objective)}.json"


def guide_bundle_name(objective: str = DEFAULT_OBJECTIVE) -> str:
    """``z_g`` -- the guide's continuous VAE encode."""
    return f"argavatar_ltx_vae_latent{_suffix(objective)}.pt"


def load_master(path: Path, *, bundle: dict | None = None) -> torch.Tensor:
    """Return the clip's continuous encode from a schema-two bundle; this is the one reader.

    ``bundle`` lets a caller that already loaded the record for other fields (fps,
    pixel_frames, ...) skip reading the file twice; by default this loads it from ``path``.
    """
    if bundle is None:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    version = bundle.get("schema_version")
    if version != 2 or "master" not in bundle:
        raise SystemExit(
            f"{path}: schema_version={version} is not a schema-two master bundle. "
            "Re-run precompute.py --process_gt_latent for this view"
        )
    # [C, F, H, W] -- the frame axis is 1.
    return bundle["master"]


def load_training_master(path: Path, *, bundle: dict | None = None) -> tuple[torch.Tensor, float]:
    """Check the tensor and timebase used by both startup and lazy chain loading."""
    record = torch.load(path, map_location="cpu", weights_only=True) if bundle is None else bundle
    master = load_master(path, bundle=record)
    if (
        not isinstance(master, torch.Tensor)
        or master.ndim != 4
        or not master.is_floating_point()
        or any(size <= 0 for size in master.shape)
    ):
        raise SystemExit(f"{path}: master must be a nonempty floating [C, F, H, W] tensor")
    fps = record.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise SystemExit(f"{path}: fps must be a finite positive number, got {fps!r}")
    return master, float(fps)


def capture_bundle_name(objective: str = DEFAULT_OBJECTIVE) -> str:
    """``z_y`` -- the capture's continuous VAE encode, matted or not per objective."""
    return f"ltx_vae_latent{_suffix(objective)}.pt"


@dataclass(frozen=True)
class ClipRef:
    root: Path
    part: str
    clip_id: str

    @property
    def dir(self) -> Path:
        return self.root / self.part / self.clip_id

    @property
    def name(self) -> str:
        return f"{self.part}_{self.clip_id}"

    def meta(self) -> dict:
        return json.loads((self.dir / "meta.json").read_text())

    def actor_id(self) -> str:
        """The bare actor id, never the (part, id) pair.

        Actor ids are not globally unique across ``Part_*`` (actor 31 is two different
        people across parts, actor 8 is presumably the same one) -- splitting a
        held-out set by the bare id is the conservative choice that makes leakage impossible
        under either interpretation.
        """
        return str(self.meta()["actor"]["id"])

    def fps(self) -> float:
        """Never defaulted: fps scales the temporal RoPE axis (``VideoLatentTools``)."""
        return float(self.meta()["fps"])

    def n_frames(self) -> int:
        return int(self.meta()["n_frames"])

    def is_done(self) -> bool:
        return self.meta().get("_pipeline", {}).get("state") == "done"

    def view_meta(self, view_idx: int) -> dict:
        for view in self.meta()["views"]:
            if view["view_idx"] == view_idx:
                return view
        raise KeyError(f"{self.name}: no view_idx={view_idx} in meta.json")

    def view_dir(self, view_idx: int) -> Path:
        return self.dir / Path(self.view_meta(view_idx)["video"]).parent

    def rgb_path(self, view_idx: int) -> Path:
        return self.dir / self.view_meta(view_idx)["video"]

    def mask_path(self, view_idx: int) -> Path:
        return self.view_dir(view_idx) / "mask.mp4"

    def bbox_path(self, view_idx: int) -> Path:
        return self.view_dir(view_idx) / "bbox.npy"

    def pose3d_path(self, view_idx: int) -> Path:
        return self.view_dir(view_idx) / "pose3d.npy"

    def refined_pose3d_path(self) -> Path:
        """Clip-level multiview pose refinement, shared by every camera view.

        It contains only the body parameters which the multiview solve refines; the
        camera-dependent entries remain in each view's ``pose3d.npy``.
        """
        return self.dir / "refined_pose3d.npy"


def list_clips(root: Path = DEFAULT_CORPUS_ROOT, *, done_only: bool = True) -> list[ClipRef]:
    """Every clip under ``root``, across every ``Part_*`` directory, sorted for determinism."""
    clips = []
    for part_dir in sorted(p for p in root.glob("Part_*") if p.is_dir()):
        for clip_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            ref = ClipRef(root=root, part=part_dir.name, clip_id=clip_dir.name)
            if not done_only or ref.is_done():
                clips.append(ref)
    return clips


@dataclass(frozen=True)
class CaptureManifest:
    """``capture_latent_manifest.json`` -- the crop box of record, per (clip, view).

    ``precompute.py --process_gt_latent`` computes each source's square box, encodes that exact
    pixel region, and writes the box here alongside the window plan. Nothing downstream may
    recompute a box and hope it matches: ``build_guidance.py`` renders the guide into
    ``box_for(...)``, so the guide and the capture target are the same pixel region *by
    construction* rather than by two implementations agreeing.

    A view that is absent has not been encoded yet -- that is an error at the call site, not
    a reason to fall back to a locally computed box (falling back is exactly how the two
    stages desynced before).
    """

    root: Path
    pad_factor: float
    edge: int
    boxes: dict[str, tuple[float, float, float, float]]

    @classmethod
    def load(cls, root: Path) -> CaptureManifest:
        path = root / CAPTURE_MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist -- run `precompute.py --process_gt_latent` over this corpus "
                f"before rendering guides; it produces the crop box this stage renders into"
            )
        record = json.loads(path.read_text())
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for window in record["windows"]:
            # Every window of a source shares one box (fixed for the whole clip), so
            # the first one seen settles it; window 0 is the natural representative.
            relative_dir = window["bundle"].rsplit("/", 1)[0]
            if relative_dir not in boxes:
                boxes[relative_dir] = tuple(float(v) for v in window["box_xyxy"])
        return cls(
            root=root,
            pad_factor=float(record["pad_factor"]),
            edge=int(record["edge"]),
            boxes=boxes,
        )

    def box_for(self, view_dir: Path) -> tuple[float, float, float, float]:
        """The recorded box for a view directory, by its path relative to the corpus root."""
        key = str(Path(view_dir).resolve().relative_to(self.root.resolve()))
        try:
            return self.boxes[key]
        except KeyError:
            raise KeyError(
                f"{key} has no crop box in {CAPTURE_MANIFEST_NAME}: "
                f"`precompute.py --process_gt_latent` has not encoded this view yet. Wait for it -- "
                f"do not render against a locally computed box"
            ) from None

    def has(self, view_dir: Path) -> bool:
        key = str(Path(view_dir).resolve().relative_to(self.root.resolve()))
        return key in self.boxes


def atomic_write(destination: Path, write_to: Callable[[Path], T], *, replace_existing: bool = True) -> T:
    """Publish complete same-directory temporary content, replacing only when requested.

    The hidden temp name retains the real extension so format-dependent writers work.
    ``write_to`` writes complete content and may return a value. On a writer failure,
    remove its temp file and preserve the destination. With ``replace_existing=False``,
    atomic hard-link publication refuses a destination created by another writer.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.stem}.tmp.{os.getpid()}{destination.suffix}")
    try:
        result = write_to(temp)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if replace_existing:
        temp.replace(destination)
    else:
        try:
            os.link(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
    return result


@dataclass(frozen=True)
class EncodedVideo:
    """One checked video; selection of frames belongs to the chosen model mode."""

    source: str
    actor: str
    split: str
    z_y: torch.Tensor
    z_g: torch.Tensor | None
    fps: float
    encode_records: dict
    hashes: dict[str, str]


class ClipStore:
    """Read fixed-video membership without block chains, cache settings, or frame selection."""

    def __init__(self, membership: dict, corpus_root: Path | None = None) -> None:
        from scripts.onestep_avatar.corpus.subset import (  # noqa: PLC0415 -- filenames stay import-light
            validate_membership,
        )

        validate_membership(membership)
        self.membership = membership
        self.root = Path(corpus_root or membership["corpus_root"])
        self.objective = membership["objective"]
        self.sources = {record["relative_dir"]: record for record in membership["sources"]}

    def __len__(self) -> int:
        return len(self.sources)

    def _read(self, source: dict, *, guide: bool) -> tuple[torch.Tensor, dict, str]:
        from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- avoid import-time model dependencies

        role = "guide" if guide else "capture"
        name = guide_bundle_name(self.objective) if guide else capture_bundle_name(self.objective)
        path = self.root / source["relative_dir"] / name
        expected = source.get(f"{role}_latent_sha256")
        if not expected:
            raise ValueError(f"{path}: {role} content hash is not recorded in membership")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: encoded content changed since the fixed video list")
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        # Use the public reader for the exact tensor/timebase checks shared with pruning.
        master, fps = load_training_master(path, bundle=bundle)
        if list(master.shape) != source["shape"] or fps != source["fps"]:
            raise ValueError(f"{path}: shape or frame rate differs from membership")
        if bundle.get("objective") != self.objective:
            raise ValueError(f"{path}: encoded background differs from membership or is not recorded")
        encoding = source.get(f"{role}_encode_record")
        if not isinstance(encoding, dict) or not encoding:
            raise ValueError(f"{path}: {role} encoding record is missing")
        if any(bundle.get(key) != value for key, value in encoding.items()):
            raise ValueError(f"{path}: encoding record differs from membership")
        return master, encoding, actual

    def load(self, source_id: str, require_guide: bool = False) -> EncodedVideo:
        from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415

        source = self.sources[source_id]
        capture, capture_record, capture_hash = self._read(source, guide=False)
        records = {"capture": capture_record}
        hashes = {"capture": capture_hash}
        guide = None
        if require_guide:
            guide, guide_record, guide_hash = self._read(source, guide=True)
            for key in ("source", "objective", "box_xyxy", "vae_fingerprint", "edge", "encode_contract_version"):
                if key not in guide_record or guide_record[key] != capture_record.get(key):
                    raise ValueError(f"{source_id}: capture and guide {key} differ or are not recorded")
            view = self.root / source_id
            sidecar = view / render_metadata_name(self.objective)
            sidecar_hash = sha256(sidecar)
            if sidecar_hash != source.get("guide_sidecar_sha256"):
                raise ValueError(f"{sidecar}: sidecar content changed or is not recorded")
            metadata = json.loads(sidecar.read_text())
            if metadata.get("compositing_version") != GUIDE_COMPOSITING_VERSION:
                raise ValueError(f"{sidecar}: outdated guide compositing")
            if metadata.get("objective") != self.objective:
                raise ValueError(f"{sidecar}: guide background differs or is not recorded")
            render_hash = sha256(view / render_name(self.objective))
            if guide_record.get("input_fingerprint") != render_hash:
                raise ValueError(f"{source_id}: guide master encodes different render bytes")
            records["guide"] = guide_record
            hashes.update(guide=guide_hash, guide_sidecar=sidecar_hash, render=render_hash)
        return EncodedVideo(
            source_id, source["actor"], source["split"], capture, guide, float(source["fps"]), records, hashes
        )

    def verify(self, require_guide: bool = False) -> None:
        """Fail on any selected input before model setup; never skip a selected video."""
        for source_id in self.sources:
            self.load(source_id, require_guide=require_guide)
