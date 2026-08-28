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
