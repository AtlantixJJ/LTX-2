"""Shared token grids, noise, clean first-image inputs, prediction adapters and loss.

Read doc/model/common.md for shapes and cross-module rules. This module assembles
modalities from existing cache views but never allocates or updates a causal cache.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.kv_cache import LayerKVCache
from ltx_core.model.transformer.modality import Modality
from ltx_core.tools import VideoLatentTools
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
from ltx_core.utils import to_denoised

MAX_ROPE_SECONDS = 20.0
FULL_FRAME_X0_MSE = "full_frame_x0_mse"


class ScaleGeometry(Protocol):
    """Only VAE scale factors are needed to build a token grid."""

    scale_factors: SpatioTemporalScaleFactors


class CacheView(Protocol):
    """Existing cache contents read by modality construction; no cache operations."""

    caches: list[LayerKVCache]

    @property
    def start(self) -> int: ...


def pixel_frames_for(latent_frames: int, time_scale: int) -> int:
    """Pixel frames a continuous encode of ``latent_frames`` latent frames covers."""
    return (latent_frames - 1) * time_scale + 1


@dataclass(frozen=True)
class ClipGrid:
    """A clip's token grid: the tools, RoPE positions and keyframe marks, built ONCE.

    Everything a block forward needs is a slice of this, which is what makes the positions
    global and the keyframe mark truthful. The old per-window ``tools_for_window`` restarted
    the time axis at 0 for every window *and* had ``VideoLatentTools._first_frame_keyframes_mask``
    mark every window's own first latent frame as a single-pixel keyframe -- which is false
    for every window past a clip's first once windows are sliced out of one continuous encode.
    Building the grid over the master fixes both at once.
    """

    tools: VideoLatentTools
    positions: torch.Tensor  # (1, 3, T, 2) global patch bounds
    keyframes_mask: torch.Tensor  # (1, T, 1), non-zero only on latent frame 0
    denoise_mask: torch.Tensor  # (1, T, 1) all ones; sliced and scaled into per-token timesteps
    latent_frames: int
    tokens_per_latent_frame: int
    fps: float

    @classmethod
    def build(
        cls,
        latent_frames: int,
        height: int,
        width: int,
        fps: float,
        geometry: ScaleGeometry,
        *,
        device: torch.device,
        dtype: torch.dtype,
        latent_channels: int = 128,
    ) -> ClipGrid:
        """``height``/``width`` are PIXEL dimensions of the crop, as ``tools_for_window`` takes them."""
        pixel_frames = pixel_frames_for(latent_frames, geometry.scale_factors.time)
        duration = pixel_frames / float(fps)
        if duration > MAX_ROPE_SECONDS:
            raise ValueError(
                f"a {duration:.1f}s clip exceeds the model's temporal RoPE range "
                f"({MAX_ROPE_SECONDS}s at positional_embedding_max_pos[0]); a causal rollout uses "
                f"GLOBAL positions, so this would extrapolate rather than wrap. Split the clip or "
                f"implement position re-basing before going past it"
            )
        shape = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(batch=1, frames=pixel_frames, height=height, width=width, fps=float(fps)),
            latent_channels=latent_channels,
            scale_factors=geometry.scale_factors,
        )
        if shape.frames != latent_frames:
            raise ValueError(f"pixel shape implies {shape.frames} latent frames, expected {latent_frames}")
        tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1), shape, float(fps), scale_factors=geometry.scale_factors
        )
        state = tools.create_initial_state(device=device, dtype=dtype)
        tokens = state.latent.shape[1]
        return cls(
            tools=tools,
            positions=state.positions,
            keyframes_mask=state.keyframes_mask,
            denoise_mask=state.denoise_mask,
            latent_frames=latent_frames,
            tokens_per_latent_frame=tokens // latent_frames,
            fps=float(fps),
        )

    def token_span(self, latent_start: int, latent_end: int) -> tuple[int, int]:
        return latent_start * self.tokens_per_latent_frame, latent_end * self.tokens_per_latent_frame

    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        """``(1, C, F, H, W)`` master latent -> ``(1, T, C)`` tokens on the grid's own layout."""
        return self.tools.patchifier.patchify(latent)

    def unpatchify_block(self, tokens: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """``(1, L, C)`` block tokens -> ``(1, C, F, H, W)``, using the grid's spatial shape."""
        shape = self.tools.target_shape
        channels = tokens.shape[-1]
        return tokens.transpose(1, 2).reshape(1, channels, latent_frames, shape.height, shape.width).contiguous()


def noise_block(clean_tokens: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """``lerp(clean, eps, sigma)`` -- ``GaussianNoiser``'s own formula, on one block's tokens.

    Reimplemented here rather than routed through ``create_noised_state`` because there is no
    conditioning item left to apply: the carryover that needed one is now the cache. The
    arithmetic is pinned against ``GaussianNoiser`` by ``tests/test_causal_core.py`` so the
    two cannot drift.
    """
    return mix_block_noise(clean_tokens, epsilon_block(clean_tokens, seed), sigma)


def epsilon_block(clean_tokens: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw the epsilon used by :func:`noise_block`, without mixing it with the source.

    Probe code uses this to persist and reuse an identical noise realization across sigma
    arms. Keeping the draw here prevents its seed/device/dtype convention from drifting from
    the rollout's established ``seed + block_index`` convention.
    """
    generator = torch.Generator(device=clean_tokens.device).manual_seed(seed)
    return torch.randn(*clean_tokens.shape, device=clean_tokens.device, dtype=clean_tokens.dtype, generator=generator)


def mix_block_noise(clean_tokens: torch.Tensor, epsilon: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply the rollout's noise mixture to an explicit epsilon tensor."""
    if epsilon.shape != clean_tokens.shape:
        raise ValueError(f"epsilon shape {tuple(epsilon.shape)} does not match block shape {tuple(clean_tokens.shape)}")
    return torch.lerp(clean_tokens.float(), epsilon.float(), sigma).to(clean_tokens.dtype)


def with_clean_prefix(tokens: torch.Tensor, clean_prefix: torch.Tensor | None) -> torch.Tensor:
    """Replace a leading token span with a supplied clean condition.

    ``c0`` is the sole caller today.  Keeping this replacement beside the noiser makes the
    contract explicit: the condition is part of the model input, not an output-only clamp.
    """
    if clean_prefix is None:
        return tokens
    if (
        clean_prefix.ndim != tokens.ndim
        or clean_prefix.shape[0] != tokens.shape[0]
        or clean_prefix.shape[2:] != tokens.shape[2:]
    ):
        raise ValueError("clean prefix must match token batch and channel dimensions")
    if not 0 < clean_prefix.shape[1] <= tokens.shape[1]:
        raise ValueError("clean prefix must contain between one and all block tokens")
    return torch.cat((clean_prefix, tokens[:, clean_prefix.shape[1] :]), dim=1)


def block_modality(
    grid: ClipGrid,
    tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    *,
    token_slices: list[tuple[int, int]],
    cache: CacheView | None = None,
    kv_write: bool = False,
    attention_mask: torch.Tensor | None = None,
    clean_prefix_tokens: int = 0,
    sigma_dtype: torch.dtype | None = None,
) -> Modality:
    """One forward's ``Modality``, assembled from slices of the clip grid.

    ``token_slices`` are half-open token ranges of the clip, concatenated in order -- one
    range for an ordinary block, several for the priming call that seeds a cache from the
    pinned frame 0 plus a non-adjacent stretch of context.
    """
    device = tokens.device
    positions = torch.cat([grid.positions[:, :, lo:hi] for lo, hi in token_slices], dim=2)
    keyframes = torch.cat([grid.keyframes_mask[:, lo:hi] for lo, hi in token_slices], dim=1)
    denoise = torch.cat([grid.denoise_mask[:, lo:hi] for lo, hi in token_slices], dim=1)
    if not 0 <= clean_prefix_tokens <= tokens.shape[1]:
        raise ValueError("clean_prefix_tokens must be within this modality's token span")
    # Whole-clip probes match stock pipelines, which retain float32 schedule
    # precision even when weights and noisy latents are BF16.
    timesteps = denoise.to(sigma_dtype or denoise.dtype) * sigma
    if clean_prefix_tokens:
        timesteps = timesteps.clone()
        timesteps[:, :clean_prefix_tokens] = 0
    sigma_tensor = torch.tensor([sigma], device=device, dtype=sigma_dtype or tokens.dtype)
    return Modality(
        latent=tokens,
        sigma=sigma_tensor,
        timesteps=timesteps,
        positions=positions,
        context=context,
        context_mask=None,
        attention_mask=attention_mask,
        keyframes_mask=keyframes,
        kv_caches=None if cache is None else cache.caches,
        kv_start=0 if cache is None else cache.start,
        kv_write=kv_write,
    )


def base_model(module: torch.nn.Module) -> torch.nn.Module:
    """Peel a wrapper off to reach the ``LTXModel`` underneath -- whichever of the two wrapper
    families this package actually produces.

    * **Training-time**: FSDP/DDP and PEFT wrap the velocity model itself, reachable through
      ``.module`` / ``.base_model`` / ``.model``.
    * **Deploy-time**: the inference session hands back a bare
      ``X0Model(velocity_model=<LTXModel>)`` -- no FSDP, no PEFT (the LoRA is fused at load,
      not applied as an adapter) -- reachable through ``.velocity_model`` alone.

    One implementation since S1(7) of the 2026-09-17 cleanup plan: ``train.py`` had this exact
    depth-capped walk as ``_base_model``, and ``onestep_core.rollout``, ``visualize_d0`` and
    ``bench_forward`` each carried their own copy of an unbounded
    ``while not hasattr(base, "transformer_blocks") and hasattr(base, "velocity_model")`` loop
    that only ever handled the second shape. The two algorithms agreed at every existing call
    site only because each site's wrapper matches exactly one of the two families -- checked
    directly (``tests/test_causal_core.py::test_base_model_agrees_with_both_retired_walks``)
    rather than assumed, since a wrapper matching a *different* attribute here would silently
    size the K/V cache from the wrong module.
    """
    model = module
    for _ in range(8):
        if hasattr(model, "transformer_blocks"):
            return model
        for attribute in ("module", "base_model", "model", "velocity_model"):
            inner = getattr(model, attribute, None)
            if isinstance(inner, torch.nn.Module):
                model = inner
                break
        else:
            break
    raise TypeError(f"cannot find the LTXModel inside {type(module).__name__}")


def denoised_from_velocity_model(model):  # noqa: ANN001, ANN201 -- a PEFT-wrapped LTXModel
    """Adapter for the training side, where the transformer emits velocity."""

    def call(modality: Modality) -> torch.Tensor:
        velocity, _ = model(video=modality, audio=None, perturbations=None)
        return to_denoised(modality.latent, velocity, modality.timesteps)

    return call


def denoised_from_x0_model(model):  # noqa: ANN001, ANN201 -- the X0Model the session yields
    """Adapter for the deployment side, where the session hands back an ``X0Model``."""

    def call(modality: Modality) -> torch.Tensor:
        denoised, _ = model(video=modality, audio=None, perturbations=None)
        return denoised

    return call


def guided_denoised_from_x0_model(model, guider, negative_context: torch.Tensor | None = None):  # noqa: ANN001, ANN201
    """``denoised_from_x0_model`` with CFG / STG / rescale from a pipeline ``MultiModalGuider``.

    The guider's ``calculate`` is the pipelines' own combination rule, so there is no second
    guidance formula here -- only the pass bookkeeping. Passes run **sequentially**, not as a
    batch, so the peak memory is one forward's: conditional, then the negative-prompt pass when
    CFG is on, then the STG pass (video self-attention skipped on ``stg_blocks``) when STG is on.
    Every pass sees the identical modality (tokens, timesteps, positions, masks, clean ``c0``);
    only the text context or the perturbation differs. A ``cfg=1, stg=0`` guider is exactly one
    conditional forward, the same as ``denoised_from_x0_model``.
    """
    if guider.do_unconditional_generation() and negative_context is None:
        raise ValueError("CFG needs a negative-prompt context")

    def call(modality: Modality) -> torch.Tensor:
        cond, _ = model(video=modality, audio=None, perturbations=None)
        uncond = perturbed = 0.0
        if guider.do_unconditional_generation():
            uncond, _ = model(video=replace(modality, context=negative_context), audio=None, perturbations=None)
        if guider.do_perturbed_generation():
            skip = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=list(guider.params.stg_blocks))
            config = BatchedPerturbationConfig(
                [PerturbationConfig([skip])] * cond.shape[0],
                num_blocks=model.num_blocks,
                device=cond.device,
                dtype=cond.dtype,
            )
            perturbed, _ = model(video=modality, audio=None, perturbations=config)
        return guider.calculate(cond, uncond, perturbed, 1.0)

    return call


def full_frame_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain mean squared error over every predicted token and channel.

    Training deliberately has no silhouette, alpha, or disagreement weighting: every pixel
    of the objective's continuous capture encode is part of the target. This is
    ``FULL_FRAME_X0_MSE``.
    """
    return (pred.float() - target.float()).pow(2).mean()


def source_for(capture: torch.Tensor, guide: torch.Tensor | None, guide_mode: str) -> torch.Tensor:
    """Select the noising source; the capture remains the target for both arms."""
    if guide_mode == "d0":
        return capture
    if guide_mode != "d1":
        raise ValueError("guide_mode must be d0 or d1")
    if guide is None:
        raise ValueError("D1 requires guide data")
    if guide.shape != capture.shape:
        raise ValueError("guide shape must match capture shape")
    return guide
