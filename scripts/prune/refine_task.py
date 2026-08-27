"""The refiner's deployment conditions -- the single source of truth every other
scripts/prune/* module imports instead of re-specifying "the task".

See plans/2026-08-26-refiner-head-ffn-pruning.md §1. Every importance statistic
downstream of Phase 0 must be collected at exactly these conditions: the short
k2 tail, this one constant prompt, video-only, AR-chunk geometry. Calibrating on
anything else (e.g. full bidirectional 16-latent-frame windows) systematically
over-values long-range temporal heads the deployed refiner never exercises.

Do not import from vae_refine_sliding_window.py -- that is a run script, not a
library. Reuse the *patterns* in it, import constants from here instead.
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

# The deployed student schedule (2 forwards: sigma 0.725 -> 0.422 -> 0.0) and the
# deeper schedule Phase 1's teacher.py builds x0* targets from. Values come from
# vae_refine_sliding_window.refinement_schedule's own k-step table (DISTILLED_SIGMA_VALUES
# slicing), reproduced here as the *names* Phase 1+ scripts key off rather than the
# literal float lists, which live in ltx_pipelines.utils.constants.
K_STEP = "k2"
TEACHER_K_STEP = "k8"

# AR-chunk calibration geometry (Phase 1's chunk_states.py). The upcoming autoregressive
# refiner predicts this many fresh latent frames per chunk, with CTX_LATENT_FRAMES of
# already-refined latent frozen ahead of them via VideoConditionByLatentIndex(strength=1.0)
# -- mirroring vae_refine_sliding_window.py's window-to-window carryover, just at chunk
# instead of window granularity.
CHUNK_LATENT_FRAMES = (1, 2, 3)
CTX_LATENT_FRAMES = 4

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
