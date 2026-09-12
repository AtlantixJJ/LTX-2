"""The refiner's deployment conditions -- the single source of truth every other
scripts/prune/* module imports instead of re-specifying "the task".

See plans/2026-08-26-refiner-head-ffn-pruning.md §1. Every importance statistic
downstream of Phase 0 must be collected at exactly these conditions: the short
k2 tail, this one constant prompt, video-only, AR-chunk geometry. Calibrating on
anything else (e.g. full bidirectional 16-latent-frame windows) systematically
over-values long-range temporal heads the deployed refiner never exercises.

Do not import from vae_refine_sliding_window.py -- that is a run script, not a
library. The *behaviour* it shares with this package lives in
``scripts/prune/refine_core.py``, which both sides import; the constants that
select which window it runs live here. ``scripts/prune/method_parity.py`` is the
gate that proves a refine_core-driven rollout reproduces that run script's
cached latents bit-for-bit, so these two files cannot drift apart silently.
"""

from __future__ import annotations

# Constant text conditioning for every calibration/deployment call. Deliberately the
# same string as vae_refine_sliding_window.py's DEFAULT_PROMPT -- the refiner is
# scene-agnostic and every result in expr/sam3dgs_vae_refine/ was produced with this
# text, so calibrating against anything else would score the model on conditioning it
# is not deployed under. It is duplicated rather than imported because that script is a
# run script, not a library; scripts/prune/parity_check.py is what keeps the two honest.
# Changing this string changes the prompt-context cache key (scripts/prune/
# prompt_cache.py hashes it), so pin it here rather than letting each script default
# its own text.
REFINE_PROMPT = "a high quality, sharp, detailed video with fine texture and natural lighting"

# The deployed student schedule (2 forwards: sigma 0.725 -> 0.422 -> 0.0). Values come from
# vae_refine_sliding_window.refinement_schedule's own k-step table (DISTILLED_SIGMA_VALUES
# slicing), reproduced here as the *names* Phase 1+ scripts key off rather than the
# literal float lists, which live in ltx_pipelines.utils.constants.
K_STEP = "k2"

# The deployed sliding window: exactly the one that produced
# expr/sam3dgs_vae_refine/*/k2_longform_v3_carryover/decode_full.mp4 -- 25 pixel frames
# (4 latent frames: the index-0 causal keyframe, 1 frozen carryover frame, 2 fresh) with a
# 9-frame overlap, i.e. a 16-frame stride. These are the numbers `--window-frames 25
# --overlap-frames 9` puts in that run's window_plan.json, and scripts/prune/method_parity.py
# is the gate that keeps the two in step.
#
# This USED to be CTX_LATENT_FRAMES = 4 / one fresh frame -- a geometry the refine script
# has never run. Calibrating on it over-weighted long frozen-context attention and
# under-weighted exactly the tokens the deployed refiner emits, and the resulting T1/T2
# rollout was visibly softer than decode_full.mp4. Do not "generalize" it back without
# re-running method_parity.py.
WINDOW_FRAMES = 25
OVERLAP_FRAMES = 9
CTX_LATENT_FRAMES = 1
DEPLOY_CHUNK_LATENT_FRAMES = 2

# Calibration sweeps chunk width around the deployed value so importance scores are not
# fit to one window length; every entry keeps CTX_LATENT_FRAMES and therefore corresponds
# to a real `--window-frames {17,25,33} --overlap-frames 9` run of the refine script.
CHUNK_LATENT_FRAMES = (1, 2, 3)


def deployed_geometry(scale_factors):
    """The deployed window as a ``refine_core.WindowGeometry``.

    Imported lazily so this module stays a pure constants module that any script can
    import without pulling in torch/ltx_core.
    """
    from scripts.prune.core.refine_core import WindowGeometry

    return WindowGeometry(
        window_frames=WINDOW_FRAMES, overlap_frames=OVERLAP_FRAMES, scale_factors=scale_factors
    )


def calibration_geometry(chunk_latent_frames: int, scale_factors):
    """The window that freezes ``CTX_LATENT_FRAMES`` and denoises ``chunk_latent_frames``."""
    from scripts.prune.core.refine_core import WindowGeometry

    return WindowGeometry.from_latent_frames(
        context_latent_frames=CTX_LATENT_FRAMES,
        chunk_latent_frames=chunk_latent_frames,
        scale_factors=scale_factors,
    )


# The one-step schedule the guided-init LoRA is trained for (plan 2026-09-10 SS7.3).
# NOT a k-step tail: the tail from 0.725 is [0.725, 0.421875, 0.0], which is k2 -- two
# forwards. One step means jumping straight to 0, which no slice of the distilled grid
# expresses, so it needs its own constructor. Added ALONGSIDE K_STEP and never in place of
# it: every frozen number under expr/refiner_prune/2.5/ is a k2 baseline.
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


def assert_one_step_conditions(metadata: dict, sigma0: float, schedule: list[float]) -> None:
    """Refuse a fixed-sigma adapter that is being run off-condition (SS7.3, SS9 risk 13).

    Two failures the harness must make loud, because both otherwise produce output rather than
    an error:

    * **More than one step.** For a model distilled onto a fixed grid, extra steps are
      extrapolation, not refinement (SS2.3(1)) -- a fixed-sigma0 LoRA is not a drop-in
      multi-step model.
    * **A sigma0 that disagrees with the checkpoint.** The adapter learned one operating
      point; run at another, it is applying a correction calibrated for a different noise
      level to an input that does not have it.
    """
    if len(schedule) != 2:
        raise ValueError(
            f"a one-step checkpoint was given a {len(schedule) - 1}-forward schedule {schedule}. "
            f"Extra steps are extrapolation for a fixed-sigma adapter, not refinement"
        )
    recorded = metadata.get("onestep_avatar_sigma0")
    if recorded is None:
        return
    if abs(float(recorded) - sigma0) > 1e-9:
        raise ValueError(
            f"checkpoint was trained at sigma0={recorded} but is being run at {sigma0}; "
            f"a fixed-sigma adapter is only defined at its own operating point"
        )


_K_STEP_TAIL_LENGTH = {"k1": 2, "k2": 3, "k3": 4, "k4": 5, "k8": 9}


def schedule_for(sigmas: list[float], k_step: str) -> list[float]:
    """Slice a full distilled sigma schedule down to a k-step tail.
    Same rule as vae_refine_sliding_window.refinement_schedule (reimplemented,
    not imported -- that script is a run script, not a library): ``k2`` is the
    last 3 sigma values (2 forward passes), ``k8`` is the full 9-value schedule.
    """
    if k_step not in _K_STEP_TAIL_LENGTH:
        raise ValueError(f"Unknown k_step {k_step!r}; expected one of {list(_K_STEP_TAIL_LENGTH)}")
    n = _K_STEP_TAIL_LENGTH[k_step]
    return list(sigmas[-n:])
