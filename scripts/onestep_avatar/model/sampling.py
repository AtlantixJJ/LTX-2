"""Mode-independent exact denoising schedules and deterministic Euler steps.

See doc/model/sampling.md. No cache allocation, model calls, or adapter checks.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.schedulers import LTX2Scheduler

ONE_STEP = "one_step"
ONE_STEP_SIGMA0 = 0.725


def one_step_schedule(sigmas: list[float], sigma0: float = ONE_STEP_SIGMA0) -> list[float]:
    """``[sigma0, 0.0]`` -- exactly one transformer forward, from a point ON the distilled grid.

    ``sigma0`` is checked against the grid rather than taken on trust. Per the plan's SS2.3(2)
    the distilled checkpoint is a deterministic map defined at nine sigmas, not on a continuum,
    so an off-grid sigma0 is not "slightly different conditions" -- it is a point the model was
    never trained at, and it would produce plausible-looking output with no baseline to compare
    against.
    """
    if not any(abs(sigma0 - value) < 1e-9 for value in sigmas):
        raise ValueError(
            f"sigma0 {sigma0} is not on the distilled sigma grid {sigmas}; the distilled model "
            f"is a map defined at those nine points, not a continuum"
        )
    return [float(sigma0), 0.0]


def truncated_schedule(sigma_start: float, steps: int) -> tuple[float, ...]:
    """The stock ``LTX2Scheduler().execute(steps=N)`` curve, entered at ``sigma_start``.

    Start exactly at ``sigma_start``, then take every stock level strictly below it, down to 0.
    This is what the stock pipeline would do from a partially noised input: a lower start noise
    walks fewer of the N levels, so step count falls with the start sigma (at N 30: 4 steps
    from .421875, 9 from .725, 18 from .909375, 26 from .975, 30 from 1). ``sigma_start == 1``
    is the stock schedule exactly. Compare :func:`rescaled_schedule`, which keeps N steps at
    every start.
    """
    if not 0.0 < sigma_start <= 1.0:
        raise ValueError(f"sigma_start must be in (0, 1], got {sigma_start}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    stock = [float(v) for v in LTX2Scheduler().execute(steps=steps)] if steps > 1 else [1.0, 0.0]
    below = [v for v in stock[1:-1] if v < sigma_start - 1e-9]
    return validate_schedule([float(sigma_start), *below, 0.0])


def thinned_truncated_schedule(sigma_start: float, stock_steps: int, denoising_steps: int) -> tuple[float, ...]:
    """Use fewer calls from one fixed stock tail, retaining both endpoints.

    Select evenly spaced indices, rather than generating a new stock curve with
    different spacing. The complete tail is the sigma-specific upper bound.
    """
    tail = truncated_schedule(sigma_start, stock_steps)
    maximum = len(tail) - 1
    if not 1 <= denoising_steps <= maximum:
        raise ValueError(f"denoising_steps must be within [1, {maximum}] at sigma={sigma_start}")
    indices = [round(i * maximum / denoising_steps) for i in range(denoising_steps + 1)]
    return validate_schedule([tail[i] for i in indices])


def rescaled_schedule(sigma_start: float, steps: int) -> tuple[float, ...]:
    """The stock ``LTX2Scheduler`` curve for ``steps`` steps, scaled to start at ``sigma_start``.

    "Stock" means what the pipelines actually run: ``LTX2Scheduler().execute(steps=N)`` with
    **no latent**, i.e. the shift at the 4096-token anchor. Passing the real latent (~18k tokens
    at 18x32x32) shifts far harder -- 15 of 30 levels above 0.97, then a jump -- and the dev model
    sampled on that schedule degenerates (background texture, hard cuts) where the stock pipeline
    is clean; that was a real bug here, found by comparing against ``ti2vid_one_stage``.

    Truncating the stock schedule would tie step count to the start sigma (a 0.725 start keeps 8
    of 30 levels). Multiplying every level by ``sigma_start`` keeps the curve's shape and makes
    ``steps`` an independent axis; at ``sigma_start == 1`` it is the stock schedule exactly.
    """
    if not 0.0 < sigma_start <= 1.0:
        raise ValueError(f"sigma_start must be in (0, 1], got {sigma_start}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if steps == 1:
        # The terminal stretch divides by (1 - last nonzero level), which is 0 for [1.0, 0].
        return (float(sigma_start), 0.0)
    stock = LTX2Scheduler().execute(steps=steps)
    # The first level is set exactly: float32 stock[0] * sigma is not bit-equal to sigma, and
    # rollout requires the schedule to start at exactly the sigma the source was noised to.
    return validate_schedule([float(sigma_start)] + [float(level) * sigma_start for level in stock[1:-1]] + [0.0])


def euler_to(sample: torch.Tensor, denoised: torch.Tensor, sigma_from: float, sigma_to: float) -> torch.Tensor:
    """One deterministic Euler step along the straight flow path, ``sigma_from -> sigma_to``.

    Positive next levels use the native stepper's dtype and operation order.
    Equivalent bf16 interpolation rounds differently. The terminal step returns
    the prediction exactly, as required by the direct training/generation contract;
    the native stepper's reconstructed bf16 endpoint can differ.

    This is the sampler a multi-step causal teacher needs and the one-step student does not.
    It lives here rather than in a caller because ``rollout`` is the one place a block's state
    advances, and a second stepper somewhere else is exactly the "two producers of one thing"
    shape this package's bugs have taken.

    No noise is injected. A stochastic sampler would make the teacher's endpoint depend on
    random choices the student cannot reproduce from its own inputs, which is the coupling the
    plan's endpoint-distillation construction is built to preserve.
    """
    if not math.isfinite(sigma_from) or not math.isfinite(sigma_to):
        raise ValueError("Euler levels must be finite")
    if not 0.0 <= sigma_to < sigma_from <= 1.0:
        raise ValueError(f"expected 0 <= sigma_to < sigma_from, got {sigma_to} and {sigma_from}")
    if sigma_from == 0.0:
        raise ValueError("sigma_from must be nonzero: there is no step to take from a clean state")
    if sigma_to == 0.0:
        return denoised
    sigmas = torch.tensor((sigma_from, sigma_to), dtype=torch.float32, device=sample.device)
    return EulerDiffusionStep().step(sample, denoised, sigmas, 0)


def validate_schedule(
    schedule: list[float] | tuple[float, ...], model_sigmas: list[float] | None = None
) -> tuple[float, ...]:
    """A per-block denoising schedule: strictly decreasing, ending at exactly 0.

    ``[sigma0, 0.0]`` is the one-step student -- one denoise call, its output taken as the
    endpoint -- so the multi-step path is the same code with a longer list, not a branch.
    ``model_sigmas``, when given, is the selected checkpoint's own grid: every nonzero level
    must sit on it, because an off-grid level is an untrained operating point for a distilled
    model and produces a plausible video with no standing.
    """
    values = tuple(float(v) for v in schedule)
    if len(values) < 2:
        raise ValueError("a schedule needs at least a start sigma and a terminal 0.0")
    if values[-1] != 0.0:
        raise ValueError(f"a schedule must end at exactly 0.0, got {values[-1]}")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("schedule levels must be finite and in [0, 1]")
    if any(b >= a for a, b in pairwise(values)):
        raise ValueError(f"a schedule must be strictly decreasing, got {values}")
    if model_sigmas is not None:
        grid = [float(v) for v in model_sigmas]
        for level in values[:-1]:
            if not any(abs(level - g) < 1e-9 for g in grid):
                raise ValueError(f"schedule level {level} is not on the model grid {grid}")
    return values
