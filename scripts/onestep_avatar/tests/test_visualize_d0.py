"""CPU tests for the probe's span selection and its burned-in frame captions.

Neither needs a model. What they pin is the arithmetic a reader of the MP4 trusts without
being able to check it: that the video covers the whole clip rather than the training chain's
``K`` blocks, and that "latent 5 · rollout step 2" on a frame is actually true of that frame.
The VAE's frame mapping is not a ratio -- latent frame 0 is one pixel frame and every later
one is eight -- so an off-by-one here mislabels every frame after the first.
"""

from __future__ import annotations

import pytest
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
        source="part/clip/view00",
        split="train",
        actor="a",
        seed_is_clip_start=True,
        blocks=blocks,
        z_g=None,
        z_y=torch.zeros(1, 1, 1, 1),
        fps=30.0,
        z0_base=None,
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
    assert labels[8].startswith("latent 1")  # last pixel frame of latent 1
    assert labels[9].startswith("latent 2")  # first of latent 2
    assert "rollout step 0" in labels[16]  # latent 2 is still block 0
    assert "rollout step 1" in labels[17]  # latent 3 opens block 1


def test_frame_labels_cover_exactly_the_decoded_frames() -> None:
    """One caption per decoded frame: a short list would leave the tail unstamped."""
    plan = _geometry().plan(18)
    labels = visualize_d0._frame_labels(plan, 8)
    assert len(labels) == causal_core.pixel_frames_for(plan[-1][1], 8)


def test_stamp_writes_only_the_caption_band() -> None:
    pixels = torch.full((3, 3, 64, 64), 0.5)
    stamped = visualize_d0._stamp(pixels, visualize_d0._frame_labels([(0, 3)], 8)[:3])

    assert stamped.shape == pixels.shape
    assert not torch.equal(stamped[0], pixels[0])  # the band was drawn
    assert torch.allclose(stamped[:, :, 40:, :], pixels[:, :, 40:, :])  # the image below is untouched
    assert float(stamped.min()) >= 0.0
    assert float(stamped.max()) <= 1.0


def test_probe_sigmas_are_explicit_unique_schedule_members() -> None:
    schedule = [1.0, 0.909375, 0.725, 0.0]
    assert visualize_d0._probe_sigmas([0.909375, 1.0], schedule) == (0.909375, 1.0)
    with pytest.raises(SystemExit, match="duplicate"):
        visualize_d0._probe_sigmas([1.0, 1.0], schedule)
    with pytest.raises(SystemExit, match="nonzero"):
        visualize_d0._probe_sigmas([0.0], schedule)
    with pytest.raises(SystemExit, match="not on model schedule"):
        visualize_d0._probe_sigmas([0.8], schedule)


def test_explicit_block_epsilon_reuses_noise_across_sigma_mixtures() -> None:
    clean = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    epsilon = causal_core.epsilon_block(clean, 42)
    low = causal_core.mix_block_noise(clean, epsilon, 0.25)
    high = causal_core.mix_block_noise(clean, epsilon, 1.0)

    assert torch.equal(high, epsilon)
    assert torch.allclose(low, torch.lerp(clean, epsilon, 0.25))


def test_decoder_output_is_normalized_to_frame_major_video() -> None:
    bcthw = torch.zeros(1, 3, 5, 8, 9)
    assert visualize_d0._as_fchw(bcthw).shape == (5, 3, 8, 9)
    fhwc = torch.zeros(5, 8, 9, 3)
    assert visualize_d0._as_fchw(fhwc).shape == (5, 3, 8, 9)


def test_source_master_is_the_one_line_the_arm_changes() -> None:
    """D0 noises the capture, D1 the guide -- and nothing else about the probe moves.

    The point of ``_source_master`` is that the arm is a *tensor choice*, not a second rollout.
    ``z_y`` stays the target reference and ``c0`` in both, so a test that only checked "d1 does
    something different" would pass on an implementation that also swapped the first-frame
    condition to the guide's composited frame 0 -- which is exactly the thing the conditioning
    contract forbids.
    """
    chain = _chain([0, 1])
    guide = torch.ones(1, 1, 1, 1)
    d1_chain = Chain(**{**chain.__dict__, "z_g": guide})

    assert visualize_d0._source_master(chain, "d0") is chain.z_y
    assert visualize_d0._source_master(d1_chain, "d0") is d1_chain.z_y
    assert visualize_d0._source_master(d1_chain, "d1") is guide


def test_d1_without_a_guide_master_raises_rather_than_falling_back() -> None:
    """A missing ``z_g`` must not silently degrade D1 into D0.

    Falling back to the capture would produce a plausible video labelled as the deployable arm,
    which is the failure this package keeps hitting: a wrong conclusion off a working artifact.
    The error names ``GUIDE_COMPOSITING_VERSION`` because the usual cause is not "no file" but
    "a stale v1 guide the freezer skipped" (G6).
    """
    with pytest.raises(SystemExit, match="GUIDE_COMPOSITING_VERSION"):
        visualize_d0._source_master(_chain([0]), "d1")
