"""One-step avatar deployment through the same causal_core rollout used by training.

The capture first frame stays clean, blocks attend to cached history, and generated
blocks refresh the cache. LoRA weights are fused by the shared model session."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from scripts.onestep_avatar import causal_core, sampling
from scripts.onestep_avatar.causal_core import BlockCache, CausalGeometry, ClipGrid


def guide_conditionings(z_g: torch.Tensor, guide_mode: str) -> tuple:  # noqa: ARG001
    """The extra conditioning a guide arm adds -- ``()`` for D1, the only deployable arm.

    Kept as a function, and kept shared with ``train.py``, because the arm has to match
    between training and deployment exactly: a checkpoint trained with extra guide tokens and
    run without them is being asked to work from half its input, and nothing would raise.

    The old ``d2`` hybrid (the guide again, as clean reference tokens at timestep 0) is gone
    twice over: it was dropped as a live arm 2026-09-13 for costing 1.05x `k2`, and under
    block-causal attention it is no longer even expressible -- reference tokens appended after
    the target are, by construction, future context.
    """
    if guide_mode == "d1":
        return ()
    if guide_mode == "d0":
        raise ValueError(
            "guide_mode 'd0' is a training-only sanity arm: it noises the capture z_y, which "
            "does not exist at inference"
        )
    raise ValueError(f"unknown guide mode {guide_mode!r}; expected 'd1'")


@dataclass(frozen=True)
class RolloutResult:
    """The rolled-out latent plus what it cost, in the units §6 reports.

    ``forwards`` counts **both** passes per block -- the denoise and the cache refresh -- so
    the total includes the cache maintenance needed for later blocks. Reporting only
    denoising omits part of the deployed compute cost.
    """

    latent: torch.Tensor
    forwards: int
    blocks: int
    denoise_forwards: int
    refresh_forwards: int


def one_step_sigma(model_sigmas: list[float], sigma0: float = sampling.ONE_STEP_SIGMA0) -> float:
    """``sampling.one_step_schedule``'s single non-zero sigma, with its on-grid check.

    The schedule is still built and checked through ``sampling`` -- the causal loop has no
    stepper, so it consumes the sigma rather than the pair, but the guard that rejects an
    off-grid sigma0 or a multi-step schedule must not be bypassed. Called from :func:`rollout`
    against the distilled model's fixed 9-point grid, so an off-grid sigma0 raises there instead
    of deploying silently.
    """
    schedule = sampling.one_step_schedule(model_sigmas, sigma0)
    if len(schedule) != 2 or schedule[-1] != 0.0:
        raise ValueError(f"one_step_schedule returned {schedule}, expected [sigma0, 0.0]")
    return float(schedule[0])


def rollout(  # noqa: PLR0913 -- a rollout is defined by its geometry, schedule, arm and device
    transformer,  # noqa: ANN001 -- the X0Model the session yields
    context: torch.Tensor,
    master: torch.Tensor,
    geometry: CausalGeometry,
    sigma0: float,
    fps: float,
    *,
    first_frame_latent: torch.Tensor,
    device: torch.device,
    seed: int = 42,
    latent_channels: int = 128,
    guide_mode: str = "d1",
    dtype: torch.dtype | None = None,
    num_layers: int | None = None,
    inner_dim: int | None = None,
    model_sigmas: list[float] | None = None,
) -> RolloutResult:
    """Roll a clip forward block by block over its ONE continuous guide encode.

    ``master`` is ``[1, C, F_latent, H, W]`` -- the whole guide render encoded in a single
    pass, the same tensor training slices its inits out of. Blocks are token ranges of it;
    nothing is re-encoded and nothing is re-forwarded that the cache already holds.

    ``context`` is passed in rather than defaulted because the refiner runs on ONE constant
    prompt (``DEFAULT_PROMPT``): a rollout that silently conditioned on something
    else would change every number in §8 without changing a call site. Build it with
    ``scripts.prune.data.prompt_cache.get_or_build``, the same call ``train.py`` makes.

    ``sigma0`` is validated against ``model_sigmas`` (default: the distilled checkpoint's fixed
    9-point grid) through :func:`one_step_sigma` before anything is denoised -- an off-grid
    sigma0 raises here rather than deploying a point the model was never trained at.
    """
    if first_frame_latent.shape[2] != 1:
        raise ValueError("first_frame_latent must contain exactly the supplied latent frame 0")
    guide_conditionings(master, guide_mode)  # validates the arm; D1 adds nothing
    sigma0 = one_step_sigma(model_sigmas if model_sigmas is not None else DISTILLED_SIGMA_VALUES, sigma0)
    dtype = dtype if dtype is not None else master.dtype
    latent_frames = master.shape[2]
    grid = ClipGrid.build(
        latent_frames,
        master.shape[-2] * geometry.scale_factors.height,
        master.shape[-1] * geometry.scale_factors.width,
        fps,
        geometry,
        device=device,
        dtype=dtype,
        latent_channels=latent_channels,
    )
    base = causal_core.base_model(transformer)
    cache = BlockCache.allocate(
        grid,
        geometry,
        num_layers=num_layers if num_layers is not None else len(base.transformer_blocks),
        inner_dim=inner_dim if inner_dim is not None else base.inner_dim,
        device=device,
        dtype=dtype,
    )
    guide_tokens = grid.patchify(master.to(device=device, dtype=dtype))
    c0 = grid.patchify(first_frame_latent.to(device=device, dtype=dtype))
    denoise_fn = causal_core.denoised_from_x0_model(transformer)
    plan = geometry.plan(latent_frames)
    with torch.no_grad():
        tokens, forwards = causal_core.rollout(
            denoise_fn,
            grid,
            geometry,
            cache,
            guide_tokens,
            context,
            sigma0,
            seed=seed,
            blocks=plan,
            first_frame_condition=c0,
        )
    covered = plan[-1][1] if plan else 0
    latent = grid.unpatchify_block(tokens[:, : covered * grid.tokens_per_latent_frame], covered)
    return RolloutResult(
        latent=latent,
        forwards=forwards,
        blocks=len(plan),
        denoise_forwards=len(plan),
        refresh_forwards=len(plan),
    )
