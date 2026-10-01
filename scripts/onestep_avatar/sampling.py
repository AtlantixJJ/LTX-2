"""One-step avatar operating-point and checkpoint-condition validation."""

from __future__ import annotations

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
