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
    migrate_alpha,
)


def _renderer_white_composite(foreground: float, alpha_frac: float) -> float:
    """Reproduce ARG-Avatar's own rasterizer convention (``forward.cu``: ``C[ch] + T *
    bg_color[ch]`` with ``bg_color=torch.ones(3)``): the renderer's RGB output is already
    alpha-composited over WHITE, never straight foreground. Every fixture below builds its
    ``render_bgr`` input this way, per F1 of the 2026-09-18 audit -- the retired v1 formula
    was only ever validated against invented straight-RGB fixtures, which is how it passed
    review while producing the wrong pixels against the renderer's real contract.
    """
    return alpha_frac * foreground + (1.0 - alpha_frac) * 255.0


def test_composite_guide_frame_replaces_the_renderers_white_background_not_straight_rgb() -> None:
    """Guide contract v2: ``guide = R_white + (1 - alpha) * (B - white)``.

    Numbers reproduce the audit's CPU repro exactly: foreground 50, alpha 128/255,
    background 20 -> renderer RGB 152 (not 50) must become 35 (not the v1 bug's 86).
    """
    foreground, alpha_frac, background_value = 50.0, 128 / 255, 20.0
    render_value = _renderer_white_composite(foreground, alpha_frac)
    assert round(render_value) == 152  # confirms the fixture matches the audit's repro

    render = np.full((1, 1, 3), round(render_value), dtype=np.uint8)
    background = np.full((1, 1, 3), background_value, dtype=np.uint8)
    alpha = np.array([[round(alpha_frac * 255)]], dtype=np.uint8)

    out = composite_guide_frame(render, alpha, background)

    assert out.dtype == np.uint8
    assert abs(int(out[0, 0, 0]) - 35) <= 1  # rounding tolerance only


def test_composite_guide_frame_is_render_where_opaque_and_background_where_transparent() -> None:
    """Opaque (alpha=255): R_white already equals the foreground exactly (T=0), so
    background replacement is a no-op. Transparent (alpha=0): R_white is pure white, and the
    renderer's white contribution is fully replaced by ``B``."""
    fully_opaque_render = _renderer_white_composite(200.0, 1.0)
    fully_transparent_render = _renderer_white_composite(200.0, 0.0)
    partial_render = _renderer_white_composite(200.0, 128 / 255)
    render = np.array(
        [[[fully_opaque_render] * 3, [fully_transparent_render] * 3],
         [[partial_render] * 3, [partial_render] * 3]],
        dtype=np.uint8,
    )
    background = np.full((2, 2, 3), 50, dtype=np.uint8)
    alpha = np.array([[255, 0], [128, 64]], dtype=np.uint8)

    out = composite_guide_frame(render, alpha, background)

    assert out.dtype == np.uint8
    assert tuple(out[0, 0]) == (200, 200, 200)  # alpha=255 -> pure foreground, unchanged
    assert tuple(out[0, 1]) == (50, 50, 50)  # alpha=0 -> pure background
    # A mid alpha must land strictly between the two, not saturate to either end -- the
    # continuous-blend property this relies on to avoid a hard-edged silhouette.
    mid = int(out[1, 0, 0])
    assert 50 < mid < 200


def test_the_white_objective_blend_is_the_identity_on_the_renderers_own_output() -> None:
    """O-white's background is a constant, so it needs no capture frame at all -- which is
    also why ``guide_background`` is where the objectives diverge and the blend is not.

    For ``white``, ``B == white``, so guide contract v2 must reproduce the renderer's RGB
    EXACTLY (within rounding) rather than the v1 bug's second white blend (152 -> 203)."""
    background = guide_background("white", clip=None, driving_view=0, box=None, out_size=4)
    assert background.shape == (4, 4, 3)
    assert (background == 255).all()

    render_value = _renderer_white_composite(50.0, 128 / 255)
    render = np.full((4, 4, 3), round(render_value), dtype=np.uint8)
    alpha = np.full((4, 4), round(128 / 255 * 255), dtype=np.uint8)

    out = composite_guide_frame(render, alpha, background)
    assert (np.abs(out.astype(np.int16) - render.astype(np.int16)) <= 1).all()

    # Fully transparent still reproduces the renderer's own pure-white pixel, not 255 twice
    # over (both formulas agree here, but pin it as the identity's edge case).
    transparent_render = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert (composite_guide_frame(transparent_render, np.zeros((4, 4), dtype=np.uint8), background) == 255).all()


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


_V2 = {"compositing_version": dataset.GUIDE_COMPOSITING_VERSION}


def test_render_is_complete_rejects_a_render_built_for_the_other_objective(tmp_path: Path) -> None:
    """Which background sits behind the render is what an objective IS (SS1.2), so a sidecar
    naming the other one describes a different artifact and must be rebuilt."""
    assert _complete(tmp_path, "bg", {"objective": "bg", **_V2}) is True
    assert _complete(tmp_path, "bg", {"objective": "white", **_V2}) is False


def test_render_is_complete_rejects_a_render_built_under_the_retired_compositing_contract(
    tmp_path: Path,
) -> None:
    """F1: guide contract v1 applied alpha twice against the renderer's actual (already
    white-composited) RGB. Unlike ``objective``, a missing/mismatched compositing_version is
    never grandfathered in -- every render built before the fix used the wrong formula, so
    it must be rebuilt rather than accepted as current."""
    assert _complete(tmp_path, "bg", {"objective": "bg", "compositing_version": 1}) is False
    assert _complete(tmp_path, "bg", {"objective": "bg"}) is False  # no field at all


def test_render_is_complete_rejects_a_render_from_a_different_motion_input(tmp_path: Path) -> None:
    """A pose refinement changes the guide even when geometry and frame count do not."""
    motion_sha256 = "a" * 64
    assert _complete(tmp_path, "bg", {"objective": "bg", "motion_sha256": motion_sha256, **_V2})
    output = tmp_path / dataset.render_name("bg")
    metadata = tmp_path / dataset.render_metadata_name("bg")
    assert _render_is_complete(output, metadata, 5, 32, "bg", motion_sha256) is True
    assert _render_is_complete(output, metadata, 5, 32, "bg", "b" * 64) is False


def test_a_legacy_npy_alpha_still_counts_as_complete(tmp_path: Path) -> None:
    """A render built before the MP4 format must not be rebuilt for the format alone --
    re-rendering 19 views to change a file extension would cost a GPU-day."""
    output = tmp_path / dataset.render_name("bg")
    metadata_path = tmp_path / dataset.render_metadata_name("bg")
    _write_video(output, frames=5, size=(32, 32))
    np.save(tmp_path / f"{dataset.ALPHA_STEM}.npy", np.zeros((5, 8, 8), dtype=np.uint8))
    metadata_path.write_text(
        json.dumps({"n_frames": 5, "alpha_grid": ALPHA_GRID, "objective": "bg", **_V2})
    )
    assert _render_is_complete(output, metadata_path, 5, 32, "bg") is True


def test_a_pre_objective_sidecar_infers_bg_but_still_needs_a_current_compositing_version(
    tmp_path: Path,
) -> None:
    """The 19 renders already on disk carry ``composited: true`` and no ``objective`` --
    that alone still infers as the bg render (a field-name-only change would not justify a
    GPU-day re-render). But they also predate guide contract v2 (F1): every pre-fix render
    used the retired double-alpha formula, so a missing compositing_version is NOT
    grandfathered in the way ``objective`` is -- the sidecar remains stale until rebuilt."""
    assert _complete(tmp_path, "bg", {"composited": True, **_V2}) is True
    assert _complete(tmp_path, "bg", {"composited": True}) is False
    assert _complete(tmp_path, "bg", {}) is False


def _legacy_view(corpus: Path) -> Path:
    view = corpus / "Part_1" / "0001_01" / "views" / "view01_cam01"
    view.mkdir(parents=True)
    np.save(view / f"{dataset.ALPHA_STEM}.npy", np.zeros((3, 8, 8), dtype=np.uint8))
    return view


def test_migrate_alpha_dry_run_describes_the_run_it_previews(tmp_path: Path) -> None:
    """A dry run that called an already-migrated view a pending conversion would not be a
    preview of anything -- and 'how much is left' is the only question it is asked."""
    view = _legacy_view(tmp_path)
    assert migrate_alpha(tmp_path, prune=False, dry_run=True)["converted"] == 1

    mask_video.write_mask_video(
        np.zeros((3, 8, 8), dtype=np.uint8), view / f"{dataset.ALPHA_STEM}.mp4"
    )
    preview = migrate_alpha(tmp_path, prune=False, dry_run=True)
    assert preview == {"converted": 0, "already_mp4": 1, "failed": 0, "removed_npy": 0}
    # And a dry run never touches the disk, whatever --prune-npy says.
    assert migrate_alpha(tmp_path, prune=True, dry_run=True)["removed_npy"] == 0
    assert (view / f"{dataset.ALPHA_STEM}.npy").is_file()
