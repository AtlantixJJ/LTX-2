"""The DNARenderingVideo/Processed corpus layout that ``build_guidance.py`` reads.

Per ``plans/2026-09-10-ltx25-one-step-argavatar-lora.md``: ``data/AnimatableHuman/
DNARenderingVideo/`` is a curated, identically-laid-out copy of ``data/DNARendering/
Processed/`` -- same per-clip files (``cameras.npy``, ``meta.json``,
``views/*/{rgb,mask}.mp4`` + ``bbox/keypoints2d/pose3d.npy``). The corpus root is one
constant here so T4's scale-out to ``Processed/`` (SS5.0) is a ``root=`` argument, not a
second code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# LTX-2/scripts/onestep_avatar/dataset.py -> parents[3] is the workspace root, one level
# ABOVE the LTX-2 repo. It was parents[2] until the 2026-09-15 consolidation, when this module
# moved from the workspace's own scripts/ tree into LTX-2's; the corpus, expr/ and checkpoints
# did not move with it, so the depth had to change with the file.
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_ROOT = WORKSPACE_ROOT / "data" / "AnimatableHuman" / "DNARenderingVideo"

# Written by ``LTX-2/scripts/onestep_avatar/precompute.py --capture-only`` at the corpus
# root. It is the **single producer** of the crop box (SS4.5): the box it records is the one
# the capture target latents were actually encoded with, so every other stage reads it rather
# than recomputing a box of its own.
CAPTURE_MANIFEST_NAME = "capture_latent_manifest.json"

# -- The two objectives, and the per-objective artifact names ---------------------------
#
# Single-sourced here since 2026-09-15. This mapping used to be transcribed into a second
# module (``corpus_names.py``) because the corpus half and the model half lived in different
# trees and different conda envs and could not import each other; consolidating the package
# removed the seam, so the transcription and the cross-tree test that pinned it are both gone.
#
# The task splits in two (plan SS1.2), and the split reaches the corpus as exactly two
# artifacts, not as two pipelines:
#
#   * ``bg``    -- the product. Guide = render composited over the clip's real first frame;
#                  target = the capture, unmatted, with its real background.
#   * ``white`` -- both sides on white. Guide = the render on the renderer's own white
#                  background (no compositing at all); target = the capture matted to white.
#                  Isolates the subject-texture gap from the background question entirely.
#
# Everything else is objective-independent and is deliberately NOT suffixed: the render's
# alpha and the loss-mask coverage grids are the same tensors either way (the render is the
# same render; only what sits behind it differs), so suffixing them would manufacture two
# copies of one thing -- the two-producer shape SS2 names as the source of every past bug.
#
# ``bg`` maps to the UNSUFFIXED names on purpose: it is what the 2034 capture bundles and 19
# renders already on disk were built as, so adding the second objective does not invalidate
# a single existing artifact.
OBJECTIVES = ("bg", "white")
DEFAULT_OBJECTIVE = "bg"

# Both persisted coverage masks are LOSSLESS grayscale MP4 (see mask_video.py): ~42x smaller
# than the raw arrays they replaced, bit-exact, and readable by the same OpenCV path as every
# other video here. All three are objective-independent -- the render is the same render and
# the matte the same matte; only what sits behind the subject differs between objectives.
ALPHA_STEM = "argavatar_alpha"                     # .mp4 now; .npy still read if present
ALPHA_NAME = f"{ALPHA_STEM}.mp4"                   # the render's own alpha, 256**2
CAPTURE_MASK_CROP_STEM = "capture_mask_crop"       # mask.mp4 cropped to the box, 256**2
CAPTURE_MASK_CROP_NAME = f"{CAPTURE_MASK_CROP_STEM}.mp4"
LOSS_MASK_GRIDS_NAME = "loss_mask_grids.pt"        # pooled to LATENT resolution, tiny
CAPTURE_MASK_NAME = "mask.mp4"                     # the dataset's own, full resolution


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


def capture_bundle_name(objective: str = DEFAULT_OBJECTIVE) -> str:
    """``z_y`` -- the capture's continuous VAE encode, matted or not per objective."""
    return f"ltx_vae_latent{_suffix(objective)}.pt"


# view_idx -> direction, fixed by the capture rig (8 x 45 degree azimuth bins).
VIEW_DIRECTIONS = (
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
)


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

        Actor ids are not globally unique across ``Part_*`` (09-05 plan SS2: actor 31 is two
        different people across parts, actor 8 is presumably the same one) -- splitting a
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

    ``precompute.py --capture-only`` computes each source's square box, encodes that exact
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
                f"{path} does not exist -- run `precompute.py --capture-only` over this corpus "
                f"before rendering guides; it produces the crop box this stage renders into"
            )
        record = json.loads(path.read_text())
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for window in record["windows"]:
            # Every window of a source shares one box (fixed for the whole clip, SS4.5), so
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
                f"`precompute.py --capture-only` has not encoded this view yet. Wait for it, "
                f"or add the view to its --views selection -- do not render against a locally "
                f"computed box"
            ) from None

    def has(self, view_dir: Path) -> bool:
        key = str(Path(view_dir).resolve().relative_to(self.root.resolve()))
        return key in self.boxes
