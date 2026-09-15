from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.onestep_avatar import dataset, mask_video
from scripts.onestep_avatar.build_guidance import (
    ALPHA_GRID,
    ALPHA_NAME,
    _render_is_complete,
    composite_guide_frame,
    guide_background,
)


def test_composite_guide_frame_is_render_where_opaque_and_background_where_transparent() -> None:
    """SS1.2: guide_t = render_t * alpha_t + background * (1 - alpha_t), continuous alpha.

    One blend serves both objectives; only ``background`` differs.
    """
    render = np.full((2, 2, 3), 200, dtype=np.uint8)
    background = np.full((2, 2, 3), 50, dtype=np.uint8)
    alpha = np.array([[255, 0], [128, 64]], dtype=np.uint8)

    out = composite_guide_frame(render, alpha, background)

    assert out.dtype == np.uint8
    assert tuple(out[0, 0]) == (200, 200, 200)  # alpha=255 -> pure render
    assert tuple(out[0, 1]) == (50, 50, 50)  # alpha=0 -> pure background
    # A mid alpha must land strictly between the two, not saturate to either end -- the
    # continuous-blend property SS1.2 relies on to avoid a hard-edged silhouette.
    mid = int(out[1, 0, 0])
    assert 50 < mid < 200


def test_the_white_objective_blends_a_white_background_and_reads_no_video() -> None:
    """O-white's background is a constant, so it needs no capture frame at all -- which is
    also why ``guide_background`` is where the objectives diverge and the blend is not."""
    background = guide_background("white", clip=None, driving_view=0, box=None, out_size=4)
    assert background.shape == (4, 4, 3)
    assert (background == 255).all()

    render = np.full((4, 4, 3), 200, dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.uint8)
    assert (composite_guide_frame(render, alpha, background) == 255).all()


def _write_video(path: Path, frames: int, size: tuple[int, int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def _complete(tmp_path: Path, objective: str, metadata: dict) -> bool:
    output = tmp_path / dataset.render_name(objective)
    metadata_path = tmp_path / dataset.render_metadata_name(objective)
    _write_video(output, frames=5, size=(32, 32))
    mask_video.write_mask_video(
        np.zeros((5, ALPHA_GRID, ALPHA_GRID), dtype=np.uint8), tmp_path / ALPHA_NAME
    )
    metadata_path.write_text(json.dumps({"n_frames": 5, "alpha_grid": ALPHA_GRID, **metadata}))
    return _render_is_complete(output, metadata_path, 5, 32, objective)


def test_render_is_complete_rejects_a_render_built_for_the_other_objective(tmp_path: Path) -> None:
    """Which background sits behind the render is what an objective IS (SS1.2), so a sidecar
    naming the other one describes a different artifact and must be rebuilt."""
    assert _complete(tmp_path, "bg", {"objective": "bg"}) is True
    assert _complete(tmp_path, "bg", {"objective": "white"}) is False


def test_a_legacy_npy_alpha_still_counts_as_complete(tmp_path: Path) -> None:
    """A render built before the MP4 format must not be rebuilt for the format alone --
    re-rendering 19 views to change a file extension would cost a GPU-day."""
    output = tmp_path / dataset.render_name("bg")
    metadata_path = tmp_path / dataset.render_metadata_name("bg")
    _write_video(output, frames=5, size=(32, 32))
    np.save(tmp_path / f"{dataset.ALPHA_STEM}.npy", np.zeros((5, 8, 8), dtype=np.uint8))
    metadata_path.write_text(
        json.dumps({"n_frames": 5, "alpha_grid": ALPHA_GRID, "objective": "bg"})
    )
    assert _render_is_complete(output, metadata_path, 5, 32, "bg") is True


def test_a_pre_objective_sidecar_is_read_as_bg_not_re_rendered(tmp_path: Path) -> None:
    """The 19 renders already on disk carry ``composited: true`` and no ``objective``. That
    IS the bg render, so it is accepted as one -- re-rendering them would cost a GPU-day for
    a field name. A sidecar with neither flag predates the composite and is still stale."""
    assert _complete(tmp_path, "bg", {"composited": True}) is True
    assert _complete(tmp_path, "bg", {}) is False
