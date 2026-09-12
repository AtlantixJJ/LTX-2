"""The one-step AR rollout -- ``refine_core.run_schedule``'s counterpart at ``[sigma_0, 0]``.

Built **on** ``refine_core``, never a copy of it (plan 2026-09-10 §7.2, and
``scripts/prune/CLAUDE.md`` rule 1: one implementation of "refine one sliding window"). The
window state, the carryover index, the geometry and the step are all that module's; what is
different here is only the schedule -- one forward instead of ``k2``'s two -- and the guide,
which enters as the init rather than the window being re-encoded from its own pixels.

**The one thing this must NOT inherit from `K_STEP`'s rollout** (§4.4's 2026-09-11 revision):
``refine_core``'s own inference windowing re-encodes every window from pixels, which gives
each one a fresh causal keyframe at its local frame 0. That is fine for `k2`, which was
measured that way, and wrong here: `precompute.py` builds the training targets by encoding
each source **once, continuously** and slicing, so a window past the clip's first has a
regular multi-frame block in slot 0, not a re-keyed one. A rollout that re-keyed per window
would be evaluating a model on inputs it was never trained on -- a train/deploy mismatch that
shows up as a quality number, not an error. So :func:`rollout` takes the clip's **master**
latent and slices, exactly as training does.

The LoRA is fused at load rather than applied as an adapter: pass
``session.transformer(loras=...)``. There is no adapter left at inference, which is why
``checks/method_parity.py`` is unaffected by an empty tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core.tools import VideoLatentTools
from scripts.prune.core import refine_core, refine_task
from scripts.prune.core.refine_core import WindowGeometry


def guide_conditionings(z_g: torch.Tensor, guide_mode: str) -> tuple:
    """The extra conditioning D2 adds, or ``()`` for D1 -- the SAME construction `train.py` uses.

    Shared rather than re-derived because the arm has to match between training and
    deployment exactly: a checkpoint trained with clean reference tokens and run without them
    is being asked to work from half its input, and nothing would raise.
    """
    if guide_mode == "d1":
        return ()
    if guide_mode != "d2":
        raise ValueError(f"unknown guide mode {guide_mode!r}; expected 'd1' or 'd2'")
    return (
        VideoConditionByReferenceLatent(latent=z_g, downscale_factor=1, temporal_scale_factor=1, strength=1.0),
    )


def one_step_window(
    transformer,  # noqa: ANN001 -- the X0Model the session yields
    denoiser,  # noqa: ANN001 -- SimpleDenoiser; the distilled path needs no guider
    z_g: torch.Tensor,
    carry: torch.Tensor | None,
    sigmas: torch.Tensor,
    tools: VideoLatentTools,
    seed: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    guide_mode: str = "d1",
) -> torch.Tensor:
    """One window, one forward: noise the guide to ``sigmas[0]``, denoise, unpatchify.

    ``refine_core.refine_window`` with a two-point schedule would be the same thing; this
    exists to carry ``guide_mode`` through and to make the call site say "one step" rather
    than leaving the reader to count a sigma list.
    """
    if len(sigmas) != 2:
        raise ValueError(
            f"one_step_window needs a 2-point schedule (one forward), got {len(sigmas)} sigmas. "
            f"Use refine_task.one_step_schedule()"
        )
    state = refine_core.make_window_state(
        z_g,
        carry,
        float(sigmas[0].item()),
        tools,
        seed,
        device,
        dtype,
        extra_conditionings=guide_conditionings(z_g, guide_mode),
    )
    return refine_core.finalize(refine_core.run_schedule(transformer, denoiser, state, sigmas), tools)


@dataclass(frozen=True)
class RolloutResult:
    """The rolled-out latent plus what it cost, in the units §6 reports."""

    latents: list[torch.Tensor]
    forwards: int
    windows: int


def rollout(  # noqa: PLR0913 -- a rollout is defined by its geometry, schedule, arm and device
    transformer,  # noqa: ANN001
    denoiser,  # noqa: ANN001
    master: torch.Tensor,
    geometry: WindowGeometry,
    sigmas: torch.Tensor,
    fps: float,
    *,
    device: torch.device,
    seed: int = 42,
    latent_channels: int = 128,
    guide_mode: str = "d1",
    dtype: torch.dtype | None = None,
) -> RolloutResult:
    """Slide over a clip's ONE continuous guide encode, carrying the model's own output.

    ``master`` is ``[1, C, F_latent, H, W]`` -- the whole guide render encoded in a single
    pass, the same tensor `precompute.py` slices the training inits out of. Windows are
    sliced from it; nothing is re-encoded. Past window 0 there is no fresh keyframe, which is
    what the deployed rollout actually has and what the model was trained against.

    The carryover is the model's own previous output at ``CARRYOVER_LATENT_IDX``, seeded with
    nothing at the clip's first window -- deployment has no predecessor there either.
    """
    latent_frames = geometry.latent_frames
    time_scale = geometry.scale_factors.time
    height = master.shape[-2] * geometry.scale_factors.height
    width = master.shape[-1] * geometry.scale_factors.width

    total_pixel_frames = (master.shape[2] - 1) * time_scale + 1
    plan = geometry.plan(total_pixel_frames)
    tools = refine_core.tools_for_window(geometry, height, width, fps, latent_channels=latent_channels)

    outputs: list[torch.Tensor] = []
    carry: torch.Tensor | None = None
    for index, (start, _end) in enumerate(plan):
        first = start // time_scale
        z_g = master[:, :, first : first + latent_frames]
        if z_g.shape[2] != latent_frames:
            break  # the master ran out; a partial window is not a window
        refined = one_step_window(
            transformer, denoiser, z_g, carry, sigmas, tools, seed + index, device, dtype, guide_mode
        )
        outputs.append(refined)
        carry = refine_core.carry_from(refined, geometry).detach()

    return RolloutResult(latents=outputs, forwards=len(outputs) * (len(sigmas) - 1), windows=len(outputs))


def schedule(model_sigmas: list[float], sigma0: float = refine_task.ONE_STEP_SIGMA0, *, device=None) -> torch.Tensor:  # noqa: ANN001
    """``refine_task.one_step_schedule`` as a tensor, with its on-grid check."""
    return torch.tensor(
        refine_task.one_step_schedule(model_sigmas, sigma0), dtype=torch.float32, device=device
    )
