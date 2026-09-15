from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.precompute import (
    BUNDLE_SCHEMA_VERSION,
    CAPTURE_MANIFEST_NAME,
    CaptureSource,
    VideoReader,
    check_pair_alignment,
    discover_pairs,
    enumerate_capture_jobs,
    master_from_windows,
    master_record,
)
from scripts.prune.core.refine_core import WindowGeometry


def test_master_record_stores_the_whole_clip_not_a_window() -> None:
    record = master_record(
        torch.zeros(1, 128, 18, 32, 32), source="v", fps=30.0, pixel_frames=137, box_xyxy=None, edge=1024
    )
    assert record["master"].shape == (128, 18, 32, 32)
    assert record["master"].dtype == torch.bfloat16
    assert (record["schema_version"], record["fps"], record["pixel_frames"]) == (BUNDLE_SCHEMA_VERSION, 30.0, 137)


def test_master_from_windows_reassembles_a_v1_bundle_losslessly() -> None:
    """The migration that makes SS4.4's master-latent rule a reader change, not days of re-encoding.

    Every v1 window was itself sliced from one continuous encode, so the tiling is exact and
    the master reassembles bit-for-bit -- which is what this asserts by slicing every window
    back out of the result.
    """
    # The deployed 25-frame / 16-frame-stride tiling: 4 latent frames per window, overlapping
    # by 2, so two windows span 41 pixel frames and 6 latent frames.
    master = torch.randn(1, 8, 6, 2, 2, dtype=torch.bfloat16)
    windows = {
        index: {
            "latents": master[0][:, start // 8 : start // 8 + 4],
            "fps": 30.0,
            "start": start,
            "end": start + 25,
        }
        for index, start in enumerate((0, 16))
    }
    rebuilt, fps, pixel_frames = master_from_windows({"windows": windows}, 8)
    assert (fps, pixel_frames) == (30.0, 41)
    assert torch.equal(rebuilt, master)
    for record in windows.values():
        first = record["start"] // 8
        assert torch.equal(rebuilt[0][:, first : first + 4], record["latents"])


def test_master_from_windows_refuses_independently_encoded_windows() -> None:
    """Disagreeing overlaps mean the bundle predates the continuous-encode revision.

    Stitching those would splice together windows that each carry their own fabricated causal
    keyframe -- exactly the data SS4.4 removed. Re-encoding is the only fix, so this raises.
    """
    windows = {
        0: {"latents": torch.zeros(8, 4, 2, 2), "fps": 30.0, "start": 0, "end": 25},
        1: {"latents": torch.ones(8, 4, 2, 2), "fps": 30.0, "start": 16, "end": 41},
    }
    with pytest.raises(ValueError, match="continuous encode"):
        master_from_windows({"windows": windows}, 8)


BOX = [0.0, 100.0, 900.0, 1000.0]


def _write_render(view: Path, frames: int = 25, *, box: list[float] | None = None, sidecar: bool = True) -> None:
    """A guide render plus the sidecar recording which box it was rendered into."""
    writer = cv2.VideoWriter(
        str(view / "argavatar_render.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32)
    )
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((32, 32, 3), (0, 255, 0), dtype=np.uint8))
    writer.release()
    if sidecar:
        (view / "argavatar_render.json").write_text(
            json.dumps({"crop_box_xyxy": BOX if box is None else box})
        )


def _write_manifest(root: Path, views: list[Path], *, box: list[float] | None = None) -> None:
    """A capture manifest in --capture-only's own shape, covering ``views``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / CAPTURE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "bundle": f"{view.relative_to(root)}/ltx_vae_latent.pt",
                        "index": 0,
                        "box_xyxy": BOX if box is None else box,
                    }
                    for view in views
                ]
            }
        )
    )


def _write_bundle(view: Path, *, pixel_frames: int = 25, fps: float = 30.0) -> None:
    """A minimal v2 capture bundle in ``--capture-only``'s own format."""
    latent_frames = (pixel_frames - 1) // 8 + 1
    torch.save(
        master_record(
            torch.zeros(1, 128, latent_frames, 1, 1),
            source=str(view),
            fps=fps,
            pixel_frames=pixel_frames,
            box_xyxy=BOX,
            edge=32,
        ),
        view / "ltx_vae_latent.pt",
    )


def test_discover_pairs_requires_a_render_and_a_capture_bundle(tmp_path: Path) -> None:
    """A render without the capture pass's bundle is not a pair -- there is nothing to pair
    it with, and the bundle is the only place ``z_y`` and the crop box come from. A bundle
    without a render is simply not rendered yet. Neither is an error: this stage chases a
    capture run that takes days."""
    complete = tmp_path / "Part_1" / "0008_01" / "views" / "view00"
    render_only = tmp_path / "Part_1" / "0008_01" / "views" / "view01"
    bundle_only = tmp_path / "Part_1" / "0008_01" / "views" / "view02"
    for directory in (complete, render_only, bundle_only):
        directory.mkdir(parents=True)
    _write_render(complete)
    (complete / "ltx_vae_latent.pt").write_bytes(b"bundle")
    _write_render(render_only)
    (bundle_only / "ltx_vae_latent.pt").write_bytes(b"bundle")
    _write_manifest(tmp_path, [complete, bundle_only])

    pairs = discover_pairs(tmp_path)
    assert [pair.relative_dir for pair in pairs] == ["Part_1/0008_01/views/view00"]
    assert pairs[0].guide == str(complete / "argavatar_render.mp4")
    assert pairs[0].bundle == str(complete / "ltx_vae_latent.pt")


def test_discover_pairs_rejects_a_render_built_at_a_different_box(tmp_path: Path) -> None:
    """The failure this whole contract exists to prevent: a guide rendered into one pixel
    region paired with a capture latent encoded from another. Frame counts still line up, so
    nothing downstream would catch it -- it would just train a spatial shift into the model."""
    view = tmp_path / "Part_1" / "0008_01" / "views" / "view00"
    view.mkdir(parents=True)
    _write_render(view, box=[0.0, 100.0, 950.0, 1050.0])  # stale: wider than the manifest's
    (view / "ltx_vae_latent.pt").write_bytes(b"bundle")
    _write_manifest(tmp_path, [view])

    with pytest.raises(SystemExit, match="do not match the crop box"):
        discover_pairs(tmp_path)


def test_video_reader_and_pair_alignment_agree_on_the_frame_range(tmp_path: Path) -> None:
    view = tmp_path / "Part_1" / "0008_01" / "views" / "view00"
    view.mkdir(parents=True)
    _write_render(view)
    _write_bundle(view, pixel_frames=25)
    _write_manifest(tmp_path, [view])

    pair = discover_pairs(tmp_path)[0]
    reader = VideoReader(pair.guide)
    assert len(reader) == 25
    assert reader.get_batch(range(2)).shape == (2, 32, 32, 3)
    info = check_pair_alignment(pair, WindowGeometry(25, 9, SpatioTemporalScaleFactors.default()))
    assert (info["pixel_frames"], info["latent_frames"], info["fps"]) == (25, 4, 30.0)


def test_pair_alignment_rejects_a_guide_shorter_than_its_capture(tmp_path: Path) -> None:
    """The guide and the capture must describe the same moments; a render that lost frames
    against its capture is caught before any encode, not discovered as a quality problem."""
    view = tmp_path / "Part_1" / "0008_01" / "views" / "view00"
    view.mkdir(parents=True)
    _write_render(view, frames=17)
    _write_bundle(view, pixel_frames=25)
    _write_manifest(tmp_path, [view])

    pair = discover_pairs(tmp_path)[0]
    with pytest.raises(ValueError, match="different frame ranges"):
        check_pair_alignment(pair, WindowGeometry(25, 9, SpatioTemporalScaleFactors.default()))


def _write_capture_source(view: Path, *, frames: int = 25, fps: float = 30.0) -> CaptureSource:
    """A raw rgb.mp4 + bbox.npy pair, real enough for plan_source to open and decode."""
    view.mkdir(parents=True, exist_ok=True)
    rgb = view / "rgb.mp4"
    writer = cv2.VideoWriter(str(rgb), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((48, 64, 3), (0, 255, 0), dtype=np.uint8))
    writer.release()
    bbox = view / "bbox.npy"
    np.save(
        bbox,
        {"xyxy": np.tile(np.array([10.0, 5.0, 40.0, 35.0]), (frames, 1)), "valid": np.ones(frames, dtype=bool)},
    )
    stat = rgb.stat()
    return CaptureSource(
        relative_dir=str(view.name), rgb=str(rgb), bbox=str(bbox),
        rgb_fingerprint=f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}",
    )


_GEOMETRY = WindowGeometry(25, 9, SpatioTemporalScaleFactors.default())


def test_enumerate_capture_jobs_reuses_a_fresh_plan_cache(tmp_path: Path) -> None:
    """A restart must not re-open and frame-0-decode a source whose plan is already cached
    and still valid -- that reopening, across thousands of sources, is the ~1.5 h §1.3
    documents as looking like a hang with nothing logged and no bundle written."""
    source = _write_capture_source(tmp_path / "view00")
    cache_path = tmp_path / "plan_cache.json"

    first = enumerate_capture_jobs([source], _GEOMETRY, 1.2, max_workers=1, cache_path=cache_path)
    assert cache_path.is_file()

    # A source no longer openable would fail plan_source; reuse must not touch the file.
    Path(source.rgb).unlink()
    second = enumerate_capture_jobs([source], _GEOMETRY, 1.2, max_workers=1, cache_path=cache_path)
    assert [(job.index, job.start, job.end, job.box_xyxy) for job in second] == [
        (job.index, job.start, job.end, job.box_xyxy) for job in first
    ]


def test_enumerate_capture_jobs_replans_a_changed_source(tmp_path: Path) -> None:
    """A stale fingerprint (the source file changed) or a changed pad factor must invalidate
    the cache entry rather than silently reusing a plan for different pixels."""
    view = tmp_path / "view00"
    source = _write_capture_source(view)
    cache_path = tmp_path / "plan_cache.json"
    enumerate_capture_jobs([source], _GEOMETRY, 1.2, max_workers=1, cache_path=cache_path)

    changed = _write_capture_source(view, frames=41)  # rewrites rgb.mp4 -> new fingerprint, a 2nd window
    assert changed.rgb_fingerprint != source.rgb_fingerprint
    jobs = enumerate_capture_jobs([changed], _GEOMETRY, 1.2, max_workers=1, cache_path=cache_path)
    # A cache hit on the stale (25-frame) entry would yield exactly 1 window, not 2.
    assert [(job.start, job.end) for job in jobs] == [(0, 25), (16, 41)]

    # A pad-factor change must also miss the cache even with the same source file.
    repadded = enumerate_capture_jobs([changed], _GEOMETRY, 1.5, max_workers=1, cache_path=cache_path)
    assert repadded[0].box_xyxy != jobs[0].box_xyxy
