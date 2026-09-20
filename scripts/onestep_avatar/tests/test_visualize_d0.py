"""CPU tests for the probe's span selection and its burned-in frame captions.

Neither needs a model. What they pin is the arithmetic a reader of the MP4 trusts without
being able to check it: that the video covers the whole clip rather than the training chain's
``K`` blocks, and that "latent 5 · rollout step 2" on a frame is actually true of that frame.
The VAE's frame mapping is not a ratio -- latent frame 0 is one pixel frame and every later
one is eight -- so an off-by-one here mislabels every frame after the first.
"""

from __future__ import annotations

import torch

from scripts.onestep_avatar import causal_core, visualize_d0
from scripts.onestep_avatar.causal_core import CausalGeometry
from scripts.onestep_avatar.train import Chain

SCALE = causal_core.SpatioTemporalScaleFactors(8, 32, 32)


class _Grid:
    """Only ``latent_frames`` is read by the functions under test."""

    def __init__(self, latent_frames: int) -> None:
        self.latent_frames = latent_frames


def _chain(blocks: list[int]) -> Chain:
    return Chain(
        source="part/clip/view00", split="train", actor="a", seed_is_clip_start=True,
        blocks=blocks, z_g=None, z_y=torch.zeros(1, 1, 1, 1), fps=30.0, z0_base=None,
    )


def _geometry() -> CausalGeometry:
    return CausalGeometry(scale_factors=SCALE, block_latent_frames=2)


def test_span_clip_rolls_the_whole_clip_not_the_training_chain() -> None:
    """The default span is the full inference sequence, whatever K the subset froze."""
    geometry, grid = _geometry(), _Grid(18)
    chain = _chain([0, 1, 2])

    whole = visualize_d0._plan_for(chain, geometry, grid, "clip")
    chain_only = visualize_d0._plan_for(chain, geometry, grid, "chain")

    assert whole == geometry.plan(18)
    assert whole[0] == (0, 3)
    assert whole[-1][1] == 17
    assert chain_only == [(0, 3), (3, 5), (5, 7)]
    assert len(whole) > len(chain_only)


def test_span_chain_selects_the_chain_s_own_blocks_mid_clip() -> None:
    geometry, grid = _geometry(), _Grid(18)
    assert visualize_d0._plan_for(_chain([2, 3]), geometry, grid, "chain") == [(5, 7), (7, 9)]


def test_frame_labels_follow_the_vae_s_own_frame_mapping() -> None:
    """Latent frame 0 is ONE pixel frame; every later latent frame is ``time_scale``."""
    labels = visualize_d0._frame_labels([(0, 3), (3, 5)], 8)

    assert len(labels) == causal_core.pixel_frames_for(5, 8) == 33
    assert labels[0].startswith("latent 0")
    assert labels[1].startswith("latent 1")
    assert labels[8].startswith("latent 1")   # last pixel frame of latent 1
    assert labels[9].startswith("latent 2")   # first of latent 2
    assert "rollout step 0" in labels[16]     # latent 2 is still block 0
    assert "rollout step 1" in labels[17]     # latent 3 opens block 1


def test_frame_labels_cover_exactly_the_decoded_frames() -> None:
    """One caption per decoded frame: a short list would leave the tail unstamped."""
    plan = _geometry().plan(18)
    labels = visualize_d0._frame_labels(plan, 8)
    assert len(labels) == causal_core.pixel_frames_for(plan[-1][1], 8)


def test_stamp_writes_only_the_caption_band() -> None:
    pixels = torch.full((3, 3, 64, 64), 0.5)
    stamped = visualize_d0._stamp(pixels, visualize_d0._frame_labels([(0, 3)], 8)[:3])

    assert stamped.shape == pixels.shape
    assert not torch.equal(stamped[0], pixels[0])            # the band was drawn
    assert torch.allclose(stamped[:, :, 40:, :], pixels[:, :, 40:, :])  # the image below is untouched
    assert float(stamped.min()) >= 0.0
    assert float(stamped.max()) <= 1.0
