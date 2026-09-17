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
    ENCODE_CONTRACT_VERSION,
    BundleExpectation,
    CaptureSource,
    Pair,
    VideoReader,
    _master_bundle_is_current,
    check_pair_alignment,
    discover_pairs,
    enumerate_capture_jobs,
    guide_bundle_path,
    master_record,
    rank_slice,
    write_capture_mask_qa,
)
from scripts.prune.core.refine_core import WindowGeometry


def test_master_record_stores_the_whole_clip_not_a_window() -> None:
    record = master_record(
        torch.zeros(1, 128, 18, 32, 32), source="v", fps=30.0, pixel_frames=137, box_xyxy=None, edge=1024
    )
    assert record["master"].shape == (128, 18, 32, 32)
    assert record["master"].dtype == torch.bfloat16
    assert (record["schema_version"], record["fps"], record["pixel_frames"]) == (BUNDLE_SCHEMA_VERSION, 30.0, 137)
    assert record["encode_contract_version"] == ENCODE_CONTRACT_VERSION


def test_rank_slice_is_disjoint_and_covers_the_sorted_input() -> None:
    items = list(range(17))
    shards = [rank_slice(items, rank, 4) for rank in range(4)]
    assert [item for shard in shards for item in shard] != items  # round-robin, not contiguous
    assert sorted(item for shard in shards for item in shard) == items
    assert all(set(shards[a]).isdisjoint(shards[b]) for a in range(4) for b in range(a + 1, 4))


def test_bundle_currency_requires_the_current_contract_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "ltx_vae_latent.pt"
    record = master_record(
        torch.zeros(1, 128, 4, 2, 2),
        source="view",
        fps=30.0,
        pixel_frames=25,
        box_xyxy=(1.0, 2.0, 65.0, 66.0),
        edge=64,
        objective="white",
        input_fingerprint="rgb+mask-v1",
        vae_fingerprint="vae-v1",
    )
    torch.save(record, path)
    expected = BundleExpectation(
        source="view",
        latent_frames=4,
        pixel_frames=25,
        channels=128,
        edge=64,
        fps=30.0,
        scale=32,
        objective="white",
        box_xyxy=(1.0, 2.0, 65.0, 66.0),
        input_fingerprint="rgb+mask-v1",
        vae_fingerprint="vae-v1",
    )
    assert _master_bundle_is_current(path, expected)

    record["encode_contract_version"] = ENCODE_CONTRACT_VERSION - 1
    torch.save(record, path)
    assert not _master_bundle_is_current(path, expected)

    record["encode_contract_version"] = ENCODE_CONTRACT_VERSION
    torch.save(record, path)
    assert not _master_bundle_is_current(
        path,
        BundleExpectation(**{**expected.__dict__, "input_fingerprint": "changed"}),
    )


BOX = [0.0, 100.0, 900.0, 1000.0]


def test_guide_bundle_path_is_bound_to_each_pair() -> None:
    first = Pair("a", "/corpus/a/argavatar_render.mp4", "a.pt", "g", "b")
    second = Pair("b", "/corpus/b/argavatar_render.mp4", "b.pt", "g", "b")
    assert guide_bundle_path(first) == Path("/corpus/a/argavatar_ltx_vae_latent.pt")
    assert guide_bundle_path(second) == Path("/corpus/b/argavatar_ltx_vae_latent.pt")
    assert guide_bundle_path(first, "white") == Path(
        "/corpus/a/argavatar_ltx_vae_latent_white.pt"
    )


def test_capture_mask_qa_is_limited_to_view0_of_each_parts_first_five_subjects(tmp_path: Path) -> None:
    part = tmp_path / "Part_1"
    subjects = [part / f"{index:04d}_01" for index in range(6)]
    for subject in subjects:
        (subject / "views" / "view00_cam51").mkdir(parents=True)
    def source(subject: Path, view: str = "view00_cam51") -> CaptureSource:
        view_dir = subject / "views" / view
        mask = view_dir / "mask.mp4"
        writer = cv2.VideoWriter(str(mask), cv2.VideoWriter_fourcc(*"mp4v"), 30, (16, 16))
        assert writer.isOpened()
        writer.write(np.full((16, 16, 3), 255, dtype=np.uint8))
        writer.release()
        return CaptureSource(
            relative_dir=str(view_dir.relative_to(tmp_path)),
            rgb=str(view_dir / "rgb.mp4"),
            bbox=str(view_dir / "bbox.npy"),
            rgb_fingerprint="fixture",
        )

    written = write_capture_mask_qa(source(subjects[0]), (0, 0, 16, 16), overwrite=False)
    assert written == subjects[0] / "qa" / "capture_mask_crop.mp4"
    assert written.is_file()
    assert write_capture_mask_qa(source(subjects[5]), (0, 0, 16, 16), overwrite=False) is None

    (subjects[0] / "views" / "view01_cam52").mkdir()
    assert write_capture_mask_qa(
        source(subjects[0], "view01_cam52"), (0, 0, 16, 16), overwrite=False
    ) is None


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
