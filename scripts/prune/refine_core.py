"""The ONE implementation of "refine one sliding window".

``scripts/vae_refine_sliding_window.py`` produced every result under
``expr/sam3dgs_vae_refine/`` -- including the reference
``4D-Dress_00129_0__woman_dance_2_crop/k2_longform_v3_carryover/decode_full.mp4``
that the pruning work is judged against. Until this module existed,
``scripts/prune/`` re-implemented that window step three times (``teacher.py``'s
calibration builder, ``chunk_states.make_state``, ``phase1_gates._rollout``),
and the copies had drifted in four ways that each change the transformer's input:

1. **fps.** ``VideoLatentTools`` divides the temporal position axis by ``fps``
   (``ltx_core/tools.py``: ``positions[:, 0, ...] /= self.fps``), so fps is part
   of RoPE, not metadata. The run script passes the clip's own fps; the prune
   copies hardcoded ``24.0``. 41 of the 44 corpus clips are 30 fps.
2. **The index-0 keyframe.** A causal VAE's latent frame 0 encodes ONE pixel
   frame; every later latent frame encodes ``scale_factors.time`` of them, and
   ``VideoLatentTools._first_frame_keyframes_mask`` marks slot 0 accordingly.
   The run script re-encodes each window from pixels, so slot 0 is always a
   genuine keyframe. ``phase1_gates._rollout`` instead spliced a *regular*
   latent frame off the rollout stream into slot 0.
3. **Carryover width.** The run script freezes ``(overlap_frames - 1) //
   time`` latent frames -- 1 frame for the reference 25/9 window -- and denoises
   the rest. The prune copies froze 4 and denoised 1.
4. **Seed.** One fixed seed per run (the run script) vs. ``seed + chunk_index``.

Everything here is expressed in the run script's terms so the two cannot drift
again: ``scripts/prune/method_parity.py`` is the gate that proves a
``refine_core``-driven rollout reproduces the run script's cached latents
bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_pipelines.utils.types import ModalitySpec
from scripts.prune import ltx_adapter

# The carried-over latent is injected at index 1, never 0: this window's own latent
# frame 0 is the causal VAE's single-pixel keyframe, which has no counterpart in the
# previous window's tail (whose last regular latent frame covers `time` DIFFERENT,
# earlier pixels). Frame 0 is left fresh; only the regular latent frames from index 1
# on -- which line up pixel-for-pixel with the previous window's tail -- are frozen.
CARRYOVER_LATENT_IDX = 1


@dataclass(frozen=True)
class WindowGeometry:
    """A pixel-space sliding window, resolved against the VAE's probed scale factors.

    The two rules (``F % time == 1`` and ``(overlap - 1) % time == 0``) are the run
    script's own; ``scripts/prune/geometry.check_window_rules`` raises on violations
    with the full explanation. They are re-asserted here because every derived count
    below is integer division that would silently truncate otherwise.
    """

    window_frames: int
    overlap_frames: int
    scale_factors: SpatioTemporalScaleFactors

    def __post_init__(self) -> None:
        t = self.scale_factors.time
        if self.window_frames % t != 1:
            raise ValueError(f"window_frames {self.window_frames} must satisfy F % {t} == 1")
        if (self.overlap_frames - 1) % t != 0:
            raise ValueError(f"overlap_frames {self.overlap_frames} must satisfy (overlap - 1) % {t} == 0")
        if self.chunk_latent_frames < 1:
            raise ValueError(
                f"a {self.window_frames}-frame window with a {self.overlap_frames}-frame overlap has no fresh "
                "latent frames left to denoise once the keyframe and the carryover are accounted for"
            )

    @classmethod
    def from_latent_frames(
        cls, *, context_latent_frames: int, chunk_latent_frames: int, scale_factors: SpatioTemporalScaleFactors
    ) -> "WindowGeometry":
        """Build the window that freezes ``context`` latent frames and denoises ``chunk`` of them.

        The window also carries the index-0 keyframe, so it spans
        ``1 + context + chunk`` latent frames in total.
        """
        t = scale_factors.time
        return cls(
            window_frames=t * (context_latent_frames + chunk_latent_frames) + 1,
            overlap_frames=t * context_latent_frames + 1,
            scale_factors=scale_factors,
        )

    @property
    def latent_frames(self) -> int:
        return (self.window_frames - 1) // self.scale_factors.time + 1

    @property
    def context_latent_frames(self) -> int:
        """Latent frames carried over from the previous window and frozen."""
        return (self.overlap_frames - 1) // self.scale_factors.time

    @property
    def chunk_latent_frames(self) -> int:
        """Latent frames this window newly finalizes: everything but the keyframe and the context."""
        return self.latent_frames - 1 - self.context_latent_frames

    @property
    def stride_frames(self) -> int:
        return self.window_frames - self.overlap_frames

    def as_dict(self) -> dict:
        return {
            "window_frames": self.window_frames,
            "overlap_frames": self.overlap_frames,
            "latent_frames": self.latent_frames,
            "context_latent_frames": self.context_latent_frames,
            "chunk_latent_frames": self.chunk_latent_frames,
            "stride_frames": self.stride_frames,
            "scale_factors": list(self.scale_factors),
        }

    def plan(self, total_frames: int) -> list[tuple[int, int]]:
        """Fixed-stride window starts -- no clipped or redistributed tail window.

        Every pairwise overlap is then EXACTLY ``overlap_frames``, which the latent
        carryover depends on: it freezes a fixed-size latent slice at a fixed
        destination index, and a tail window that shifts the actual overlap by a
        frame misaligns it against the destination grid. Up to ``stride - 1``
        trailing source frames are therefore dropped rather than force-fitted.
        """
        if total_frames < self.window_frames:
            raise ValueError(f"{total_frames} frames is shorter than one {self.window_frames}-frame window")
        # A *planning*-only rule, deliberately not asserted in __post_init__: a single
        # calibration window (e.g. 17/9, the chunk=1 geometry) is a perfectly valid state
        # to score, it just cannot tile a clip -- with stride <= overlap a frame would fall
        # in three windows at once, which the blend/flush bookkeeping and the
        # single-predecessor carryover chain do not model.
        if 2 * self.overlap_frames >= self.window_frames:
            raise ValueError(
                f"overlap_frames {self.overlap_frames} must be < window_frames/2 ({self.window_frames / 2}) "
                f"to tile a clip: stride = {self.stride_frames} would be <= overlap"
            )
        return [
            (s, s + self.window_frames)
            for s in range(0, total_frames - self.window_frames + 1, self.stride_frames)
        ]


def read_pixel_window(vr, start: int, end: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Read source frames ``[start, end)`` as ``(norm_video, pixel_video)``.

    ``norm_video`` is ``[1, C, F, H, W]`` in ``[-1, 1]`` on *device* (VAE encoder input);
    ``pixel_video`` is ``[F, H, W, C]`` in ``[0, 1]`` on CPU (metric/reference side).

    Frames are CENTER-cropped to a multiple of 32. The crop anchor is part of the method:
    a top-left crop of the same clip is different content, so a score or a PSNR computed
    against it is not comparable with anything under ``expr/sam3dgs_vae_refine/``.
    """
    frames = vr.get_batch(range(start, end))  # [F, H, W, C]
    _, h, w, _ = frames.shape
    if h % 32 != 0:
        crop_h = (h // 32) * 32
        top = (h - crop_h) // 2
        frames = frames[:, top : top + crop_h, :, :]
    if w % 32 != 0:
        crop_w = (w // 32) * 32
        left = (w - crop_w) // 2
        frames = frames[:, :, left : left + crop_w, :]
    pixel_video = (frames.float() / 255.0).clamp(0.0, 1.0)
    norm_video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(dtype=dtype, device=device)
    norm_video = (norm_video / 127.5) - 1.0
    return norm_video, pixel_video


def build_tools(latent: torch.Tensor, fps: float, scale_factors: SpatioTemporalScaleFactors) -> VideoLatentTools:
    """Tools for a window whose encoded latent is ``latent``.

    ``fps`` is required and never defaulted: it scales the temporal RoPE axis, so a
    wrong value silently changes every position the transformer sees. Pass the
    clip's own ``decord.VideoReader.get_avg_fps()``.
    """
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1),
        VideoLatentShape.from_torch_shape(latent.shape),
        float(fps),
        scale_factors=scale_factors,
    )


def tools_for_window(
    geometry: WindowGeometry,
    height: int,
    width: int,
    fps: float,
    latent_channels: int = 128,
) -> VideoLatentTools:
    """``build_tools`` for a window known by its pixel size rather than its encode.

    Identical to ``build_tools`` on that window's encoded latent -- the encoder's
    output shape is exactly ``VideoLatentShape.from_pixel_shape`` of the window --
    but callable before the VAE has run.
    """
    from ltx_core.types import VideoPixelShape

    shape = VideoLatentShape.from_pixel_shape(
        VideoPixelShape(batch=1, frames=geometry.window_frames, height=height, width=width, fps=float(fps)),
        latent_channels=latent_channels,
        scale_factors=geometry.scale_factors,
    )
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1), shape, float(fps), scale_factors=geometry.scale_factors
    )


def make_window_state(
    l_init: torch.Tensor,
    carry_latent: torch.Tensor | None,
    sigma: float,
    tools: VideoLatentTools,
    seed: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> LatentState:
    """Noise ``l_init`` to ``sigma``, freezing ``carry_latent`` at index 1.

    ``l_init`` is the window's FULL VAE encode, context slots included -- the
    conditioning overwrites those slots afterwards, so passing the encode
    (rather than a spliced tensor) is what keeps the index-0 keyframe genuine.
    """
    conditionings = []
    if carry_latent is not None:
        expected = tools.target_shape.frames - 1
        if not 1 <= carry_latent.shape[2] <= expected:
            raise ValueError(
                f"carry_latent has {carry_latent.shape[2]} latent frames; the window holds "
                f"{tools.target_shape.frames} of which at most {expected} can be frozen (index 0 is the keyframe)"
            )
        conditionings = [
            VideoConditionByLatentIndex(latent=carry_latent, strength=1.0, latent_idx=CARRYOVER_LATENT_IDX)
        ]
    return ltx_adapter.build_state(
        ModalitySpec(
            context=None,  # ltx_adapter.build_state ignores spec.context; the denoiser carries it
            conditionings=conditionings,
            noise_scale=float(sigma),
            initial_latent=l_init,
        ),
        tools,
        GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed)),
        dtype if dtype is not None else l_init.dtype,
        device,
    )


def run_schedule(transformer, denoiser, state: LatentState, sigmas: torch.Tensor, stepper=None) -> LatentState:
    """Walk the k-step schedule, returning the still-patchified final state."""
    stepper = stepper if stepper is not None else EulerDiffusionStep()
    for step_idx in range(len(sigmas) - 1):
        result, _ = denoiser(transformer, state, None, sigmas, step_idx)
        if result is None:
            raise RuntimeError("video denoiser unexpectedly returned no result")
        state = ltx_adapter.step_state(state, result.denoised, stepper, sigmas, step_idx)
    return state


def finalize(state: LatentState, tools: VideoLatentTools) -> torch.Tensor:
    """Drop conditioning tokens and unpatchify to a ``(B, C, F, H, W)`` latent."""
    return tools.unpatchify(tools.clear_conditioning(state)).latent


def refine_window(
    transformer,
    denoiser,
    l_init: torch.Tensor,
    carry_latent: torch.Tensor | None,
    sigmas: torch.Tensor,
    tools: VideoLatentTools,
    seed: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    stepper=None,
) -> torch.Tensor:
    """One window, end to end: noise -> k-step denoise -> unpatchified latent."""
    state = make_window_state(l_init, carry_latent, float(sigmas[0].item()), tools, seed, device, dtype)
    return finalize(run_schedule(transformer, denoiser, state, sigmas, stepper), tools)


def carry_from(refined_latent: torch.Tensor, geometry: WindowGeometry) -> torch.Tensor:
    """The slice of a refined window that is frozen into the next window."""
    n = geometry.context_latent_frames
    if n == 0:
        raise ValueError("this geometry has no carryover (overlap_frames == 1)")
    return refined_latent[:, :, -n:, :, :].contiguous()
