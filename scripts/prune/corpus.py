"""The frozen sam3dgs refine corpus: sources, subjects, split, and file fps."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import decord

from scripts.prune import artifacts
from scripts.prune.model_registry import WORKSPACE_ROOT

CORPUS_DIR = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"


def subject_of(clip: str) -> str:
    """Return the subject prefix of a clip directory name."""
    return clip.split("__", 1)[0]


@lru_cache(maxsize=None)
def frame_count(source: Path) -> int:
    return len(decord.VideoReader(str(source)))


@lru_cache(maxsize=None)
def fps(source: Path) -> float:
    return float(decord.VideoReader(str(source)).get_avg_fps())


def sources() -> list[Path]:
    return sorted(CORPUS_DIR.glob("*/source.mp4"))


def split(key: str) -> dict[str, str]:
    """Map each frozen-manifest clip name to its calibration/held-out split."""
    manifest = json.loads(artifacts.manifest(key).read_text())
    return {
        **{clip: "calibration" for clip in manifest["split"]["calibration"]},
        **{clip: "held_out" for clip in manifest["split"]["held_out"]},
    }


def pick_clip(geometry, windows: int = 1, *, name: str | None = None, key: str | None = None,
              prefer: str | None = None, longest: bool = False) -> Path:
    """Pick one source with enough frames for whole windows of *geometry*."""
    need = geometry.window_frames + (windows - 1) * geometry.stride_frames
    candidates = [source for source in sources() if frame_count(source) >= need]
    if name:
        candidates = [source for source in candidates if source.parent.name == name] or candidates
    if prefer and key:
        wanted = split(key)
        candidates = [source for source in candidates if wanted.get(source.parent.name) == prefer] or candidates
    if not candidates:
        raise SystemExit(
            f"No clip under {CORPUS_DIR} has the >= {need} frames needed for {windows} window(s) of {geometry.as_dict()}."
        )
    return max(candidates, key=frame_count) if longest else candidates[0]


def pick_one_per_subject(count: int, min_frames: int) -> list[Path]:
    """Return the first usable clip for each distinct subject."""
    by_subject: dict[str, Path] = {}
    for source in sources():
        subject = subject_of(source.parent.name)
        if subject not in by_subject and frame_count(source) >= min_frames:
            by_subject[subject] = source
        if len(by_subject) >= count:
            break
    if len(by_subject) < count:
        raise SystemExit(
            f"Only {len(by_subject)} subjects have a >= {min_frames}-frame clip under {CORPUS_DIR}; need {count}."
        )
    return list(by_subject.values())
