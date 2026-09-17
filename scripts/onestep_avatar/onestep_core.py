"""The one-step causal AR rollout -- the deployment counterpart of ``train.py``'s loop.

Built **on** ``causal_core``, never a copy of it (plan 2026-09-10 §7.2, and
``scripts/prune/CLAUDE.md`` rule 1: one implementation of the rollout). What is different
here is only the schedule -- one forward instead of `k2`'s two -- and the guide, which enters
as the init rather than the block being re-encoded from its own pixels.

**Revised 2026-09-14 (§4.4).** The sliding window with a frozen carryover at latent index 1 is
gone, and with it ``refine_core.make_window_state``/``run_schedule`` on this path. Deployment
is now block-causal attention plus a clean-latent K/V cache, exactly as training is:

* a block's queries attend over ``[cached clean context | this block]`` and nothing later;
* after a block is denoised, one clean no-grad ``refresh`` forward puts its keys and values in
  the cache, so no later block ever forwards that content again;
* the pinned frame-0 sink -- the causal keyframe, which under §2.0 is the product's *given*
  real first frame -- stays in the cache for the whole rollout.

`k2` is untouched: it still runs ``refine_core``'s window step, which is what every frozen
number under ``expr/refiner_prune/2.5/`` was measured with. The two schemes coexist rather
than one replacing the other in place, because the baseline has to stay reproducible.

:func:`rollout` takes the clip's **master** latent and slices, so it does not inherit
``K_STEP``'s per-window re-encode -- the §4.4 rule, and the one thing here that would
silently mismatch training.

The LoRA is fused at load rather than applied as an adapter: pass
``session.transformer(loras=...)``. There is no adapter left at inference, which is why
``checks/method_parity.py`` is unaffected by an empty tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.onestep_avatar import causal_core
from scripts.onestep_avatar.causal_core import BlockCache, CausalGeometry, ClipGrid
from scripts.prune.core import refine_task


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
    it is directly comparable with `k2`'s two forwards per window. Reporting only the denoise
    pass would flatter the compute claim by exactly the factor the refresh costs.
    """

    latent: torch.Tensor
    forwards: int
    blocks: int
    denoise_forwards: int
    refresh_forwards: int


def one_step_sigma(model_sigmas: list[float], sigma0: float = refine_task.ONE_STEP_SIGMA0) -> float:
    """``refine_task.one_step_schedule``'s single non-zero sigma, with its on-grid check.

    The schedule is still built and checked through ``refine_task`` -- the causal loop has no
    stepper, so it consumes the sigma rather than the pair, but the guard that rejects an
    off-grid sigma0 or a multi-step schedule must not be bypassed.

    **Nothing calls this today.** ``rollout`` takes ``sigma0`` directly, so the guard is
    currently unreached and an off-grid sigma0 deploys silently. Wiring it in needs the
    model's sigma grid at the call site; see ``doc/onestep_core.md``.
    """
    schedule = refine_task.one_step_schedule(model_sigmas, sigma0)
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
    device: torch.device,
    seed: int = 42,
    latent_channels: int = 128,
    guide_mode: str = "d1",
    dtype: torch.dtype | None = None,
    num_layers: int | None = None,
    inner_dim: int | None = None,
) -> RolloutResult:
    """Roll a clip forward block by block over its ONE continuous guide encode.

    ``master`` is ``[1, C, F_latent, H, W]`` -- the whole guide render encoded in a single
    pass, the same tensor training slices its inits out of. Blocks are token ranges of it;
    nothing is re-encoded and nothing is re-forwarded that the cache already holds.

    ``context`` is passed in rather than defaulted because the refiner runs on ONE constant
    prompt (``refine_task.REFINE_PROMPT``): a rollout that silently conditioned on something
    else would change every number in §8 without changing a call site. Build it with
    ``scripts.prune.data.prompt_cache.get_or_build``, the same call ``train.py`` makes.
    """
    guide_conditionings(master, guide_mode)  # validates the arm; D1 adds nothing
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
    base = transformer
    while not hasattr(base, "transformer_blocks") and hasattr(base, "velocity_model"):
        base = base.velocity_model
    cache = BlockCache.allocate(
        grid,
        geometry,
        num_layers=num_layers if num_layers is not None else len(base.transformer_blocks),
        inner_dim=inner_dim if inner_dim is not None else base.inner_dim,
        device=device,
        dtype=dtype,
    )
    guide_tokens = grid.patchify(master.to(device=device, dtype=dtype))
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
