"""The mask codec: lossless, or it is not fit for purpose.

Masks here are already one generation of lossy video away from the truth (the capture matte
is a hard threshold off h264 -- the plan's risk 8). The storage format must not add a second.
These tests pin that, plus the legacy fallback that keeps pre-MP4 renders readable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.onestep_avatar import mask_video


def _soft_mask(frames: int = 7, size: int = 64) -> np.ndarray:
    """A disc with an anti-aliased edge -- the 1.4 % of pixels that actually matter."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    out = []
    for i in range(frames):
        d = np.sqrt((xx - size / 2 - i) ** 2 + (yy - size / 2) ** 2)
        out.append((np.clip((size / 4 - d) / 2.0 + 0.5, 0, 1) * 255).astype(np.uint8))
    return np.stack(out)


def test_the_round_trip_is_bit_exact(tmp_path: Path) -> None:
    """Not 'close' -- equal. A lossy codec would blur exactly the soft silhouette edge that
    the composite's smooth boundary and the latent coverage both come from."""
    grid = _soft_mask()
    path = mask_video.write_mask_video(grid, tmp_path / "alpha.mp4")
    assert np.array_equal(mask_video.read_mask_video(path), grid)


def test_soft_edge_values_survive_exactly(tmp_path: Path) -> None:
    """The edge is the point: a mask that only round-trips 0 and 255 would pass a naive
    equality check on a binary fixture and still destroy every partial-coverage cell."""
    grid = _soft_mask()
    soft = (grid > 0) & (grid < 255)
    assert soft.any(), "fixture has no anti-aliased edge; the test would prove nothing"
    decoded = mask_video.read_mask_video(mask_video.write_mask_video(grid, tmp_path / "a.mp4"))
    assert np.array_equal(decoded[soft], grid[soft])


def test_frame_count_round_trips(tmp_path: Path) -> None:
    """A dropped or duplicated frame would silently shift every mask against its latent."""
    for frames in (1, 2, 9, 30):
        grid = _soft_mask(frames=frames)
        decoded = mask_video.read_mask_video(
            mask_video.write_mask_video(grid, tmp_path / f"m{frames}.mp4")
        )
        assert decoded.shape == grid.shape


def test_the_encode_stays_lossless() -> None:
    """Pin the setting itself, not just its effect: a drift to a lossy crf would be invisible
    in every downstream number, and cheap to introduce while 'tuning storage'."""
    assert "-crf" in mask_video.MASK_ENCODE_ARGS
    assert mask_video.MASK_ENCODE_ARGS[mask_video.MASK_ENCODE_ARGS.index("-crf") + 1] == "0"
    assert "gray" in mask_video.MASK_ENCODE_ARGS


def test_it_actually_compresses(tmp_path: Path) -> None:
    """The whole reason for the format. Real corpus alphas hit ~42x; a synthetic fixture is
    easier still, so a 5x floor only catches the format silently reverting to something raw."""
    grid = _soft_mask(frames=30, size=256)
    path = mask_video.write_mask_video(grid, tmp_path / "big.mp4")
    assert grid.nbytes / path.stat().st_size > 5


def test_a_legacy_npy_is_still_read(tmp_path: Path) -> None:
    """Renders that predate the format keep working untouched -- a .npy is the same array,
    just 42x larger. Rebuilding one would cost a full re-render."""
    grid = _soft_mask()
    np.save(tmp_path / "alpha.npy", grid)
    stem = tmp_path / "alpha"
    assert mask_video.mask_exists(stem)
    assert np.array_equal(mask_video.read_mask(stem), grid)


def test_the_mp4_wins_when_both_are_present(tmp_path: Path) -> None:
    """After a migration both forms exist until --prune-npy runs; the new one is canonical."""
    stem = tmp_path / "alpha"
    np.save(stem.with_suffix(".npy"), np.zeros((4, 8, 8), dtype=np.uint8))
    mask_video.write_mask_video(np.full((4, 8, 8), 255, dtype=np.uint8), stem.with_suffix(".mp4"))
    assert (mask_video.read_mask(stem) == 255).all()


def test_a_missing_mask_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mask_video.read_mask(tmp_path / "absent")


def test_a_wrong_shaped_or_typed_grid_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected"):
        mask_video.write_mask_video(np.zeros((4, 8), dtype=np.uint8), tmp_path / "a.mp4")
    with pytest.raises(ValueError, match="expected"):
        mask_video.write_mask_video(np.zeros((4, 8, 8), dtype=np.float32), tmp_path / "b.mp4")
    with pytest.raises(ValueError, match="empty"):
        mask_video.write_mask_video(np.zeros((0, 8, 8), dtype=np.uint8), tmp_path / "c.mp4")


def test_masks_are_pooled_to_the_causal_latent_timeline_on_read() -> None:
    """Frame 0 stands alone; each later latent frame averages the next eight pixels."""
    grid = np.stack([np.full((4, 4), value, dtype=np.uint8) for value in range(17)])
    pooled = mask_video.pool_to_latent_grid(
        grid, latent_frames=3, latent_height=1, latent_width=1, time_scale=8
    )
    expected = np.array([0.0, np.mean(range(1, 9)) / 255.0, np.mean(range(9, 17)) / 255.0])
    assert np.allclose(pooled[:, 0, 0].float().numpy(), expected, atol=5e-5)


def test_latent_pooling_refuses_a_short_mask() -> None:
    with pytest.raises(ValueError, match="too few"):
        mask_video.pool_to_latent_grid(
            np.zeros((16, 4, 4), dtype=np.uint8),
            latent_frames=3,
            latent_height=1,
            latent_width=1,
            time_scale=8,
        )
