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


def read_adapter_metadata(path) -> dict[str, str]:  # noqa: ANN001 -- str | Path
    """The safetensors metadata ``train.checkpoint_metadata`` stamped, without loading tensors."""
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def adapter_condition_problems(  # noqa: PLR0912, PLR0913 -- one checklist, one place
    metadata: dict[str, str],
    *,
    base: dict[str, str],
    objective: str,
    guide_mode: str,
    schedule: list[float],
    geometry: dict,
    teacher_forcing: bool,
) -> list[str]:
    """Every way a LoRA would be run off the conditions it was trained under (G3/G5).

    The one package-owned checkpoint-condition reader: the probe calls it before loading the
    22B base. ``base`` is :func:`backbone.identity` of the transformer about to be loaded, so a
    dev adapter is refused on distilled weights (and vice versa) by fingerprint, not by name.

    The allowed sigma is the recorded fixed ``sigma0`` -- or, for a ``mixed`` adapter, one of
    its recorded levels -- and the schedule must be the recorded one (``ONE_STEP`` means
    exactly ``[sigma, 0]``). This is a *calibration* check, separate from whether the base
    model supports a sigma: dev accepts any start in ``(0, 1]``, the distilled grid nine.
    """
    problems: list[str] = []

    def want(key: str, expected: str, label: str) -> None:
        recorded = metadata.get(key)
        if recorded is None:
            problems.append(f"{label}: not recorded in the adapter ({key})")
        elif recorded != expected:
            problems.append(f"{label}: adapter {recorded!r}, requested {expected!r}")

    want("onestep_avatar_base_variant", base["base_variant"], "base variant")
    want("onestep_avatar_base_transformer_fingerprint", base["base_transformer_fingerprint"], "base weights")
    want("model_key", base["model_key"], "model version")
    want("onestep_avatar_objective", objective, "objective")
    want("onestep_avatar_guide_mode", guide_mode, "arm")
    want("onestep_avatar_first_frame_conditioning", "clean_c0_v1", "first-frame conditioning")
    want("onestep_avatar_loss", "full_frame_x0_mse", "loss")
    want("onestep_avatar_attention", "block_causal", "attention")
    want("onestep_avatar_history_computation", "cached_refresh_global_sigma0", "history computation")
    for name in ("block_latent_frames", "context_latent_frames", "sink_latent_frames"):
        want(f"onestep_avatar_{name}", str(geometry[name]), name.replace("_", " "))
    want("onestep_avatar_teacher_forcing", str(bool(teacher_forcing)), "training history policy")
    if metadata.get("lora_rank") != metadata.get("lora_alpha"):
        problems.append(
            f"LoRA alpha/rank {metadata.get('lora_alpha')}/{metadata.get('lora_rank')} != 1 is "
            "stamped but not applied at fusion"
        )

    sigma = float(schedule[0])
    recorded_sigma = metadata.get("onestep_avatar_sigma0")
    levels = [float(v) for v in metadata.get("onestep_avatar_sigma_levels", "").split(",") if v]
    if recorded_sigma is None:
        problems.append("sigma: not recorded in the adapter")
    elif recorded_sigma == "mixed":
        if not any(abs(sigma - level) < 1e-9 for level in levels):
            problems.append(f"sigma: mixed adapter levels {levels}, requested {sigma}")
    elif abs(float(recorded_sigma) - sigma) > 1e-9:
        problems.append(f"sigma: adapter calibrated at {recorded_sigma}, requested {sigma}")

    recorded_schedule = metadata.get("onestep_avatar_schedule")
    if recorded_schedule == "ONE_STEP":
        if len(schedule) != 2:
            problems.append(f"schedule: one-step adapter run with {len(schedule) - 1} denoising calls {schedule}")
    elif recorded_schedule is None:
        problems.append("schedule: not recorded in the adapter")
    else:
        executed = ",".join(repr(float(level)) for level in schedule)
        if recorded_schedule != executed and not (recorded_sigma == "mixed" and len(schedule) == 2):
            problems.append(f"schedule: adapter {recorded_schedule}, requested {executed}")
    return problems


def check_adapter_conditions(metadata: dict[str, str], *, override: bool = False, **conditions) -> list[str]:  # noqa: ANN003
    """Raise on any off-condition use unless ``override``; return the problems either way.

    ``override`` is the explicit research escape hatch for deliberate off-condition
    diagnostics. Callers must record the returned list beside the outputs, so an
    off-condition result can never pass for a calibrated one.
    """
    problems = adapter_condition_problems(metadata, **conditions)
    if problems and not override:
        raise SystemExit(
            "refusing to run this adapter off-condition:\n  "
            + "\n  ".join(problems)
            + "\nPass the probe's --off-condition-override to run it anyway as a recorded diagnostic."
        )
    return problems
