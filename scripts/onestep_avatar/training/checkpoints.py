"""Save, read and check avatar adapters without importing CLI execution.

See doc/training/checkpoints.md. Version-two contract integration is in progress.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817 -- conventional native name

from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model.adapters import LORA_TARGETS
from scripts.onestep_avatar.model.common import SIGMA_PRECISION
from scripts.onestep_avatar.training.config import RunSettings

LOGGER = logging.getLogger("onestep_avatar.checkpoints")


def read_adapter_metadata(path) -> dict[str, str]:  # noqa: ANN001 -- str | Path
    """The safetensors metadata ``training.engine.checkpoint_metadata`` stamped, without loading tensors."""
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def adapter_condition_problems(
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


def check_adapter_conditions(metadata: dict[str, str], *, override: bool = False, **conditions) -> list[str]:
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


def load_stage_init(transformer: torch.nn.Module, path: Path) -> None:
    """Initialize a NEW training stage from a parent adapter's weights -- not a resume.

    Only the LoRA tensors are loaded; the optimizer, scheduler, step counter and RNG start
    fresh, and the parent is recorded in this run's metadata. An exact resume would need all
    of those and is not implemented. Keys are the exported ComfyUI layout ``save_lora`` writes.
    """
    from scripts.onestep_avatar.model.adapters import load_weights  # noqa: PLC0415

    load_weights(transformer, path)
    LOGGER.info("initialized LoRA from parent adapter %s", path)


def _fsdp_adapter_state(transformer: torch.nn.Module, accelerator: Accelerator) -> dict:
    """Gather only separately wrapped trainable matrices, with no frozen-weight copy."""
    if not isinstance(transformer, FSDP) or not transformer.check_is_root():
        raise ValueError('adapter export requires the initialized outer FSDP root')

    def canonical(name: str) -> str:
        return '.'.join(part for part in name.split('.') if part != '_fsdp_wrapped_module')

    expected = {canonical(name) for name, _parameter in transformer.named_parameters()
                if '.lora_A.' in name or '.lora_B.' in name}
    observed, state = set(), {}
    for name, module in transformer.named_modules():
        if not isinstance(module, FSDP) or not ('.lora_A.' in name or '.lora_B.' in name):
            continue
        with FSDP.summon_full_params(module, recurse=False, writeback=False):
            for suffix, parameter in module.module.named_parameters(recurse=False):
                key = canonical(name + '.' + suffix)
                if key in observed or not parameter.requires_grad:
                    raise ValueError('adapter export found duplicate or frozen matrix ownership')
                observed.add(key)
                if accelerator.is_main_process:
                    state[key] = parameter.detach().cpu().clone()
    if not expected or observed != expected:
        raise ValueError('FSDP adapter export has incomplete separately wrapped matrix ownership')
    return state


def save_lora(
    transformer: torch.nn.Module,
    accelerator: Accelerator,
    out_dir: Path,
    step: int,
    metadata: dict[str, str],
    *,
    verify_noop: bool = False,
) -> Path | None:
    """Gather and write the adapter in the trainer's own ComfyUI-compatible layout.

    ``verify_noop`` is only for the pre-optimizer step-0 checkpoint. PEFT's default
    LoRA initialization makes A random and B exactly zero, so the product B @ A --
    and therefore the adapter delta -- must be exactly zero. Verify the *exported*
    state rather than a module attribute: that covers the actual tensors handed to
    inference, including FSDP's gathered representation.
    """
    accelerator.wait_for_everyone()
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    state_dict = _fsdp_adapter_state(transformer, accelerator) if is_fsdp else None
    if not accelerator.is_main_process:
        return None
    unwrapped = accelerator.unwrap_model(transformer, keep_torch_compile=False)
    state_dict = get_peft_model_state_dict(unwrapped, state_dict=state_dict if is_fsdp else None)
    state_dict = {f"diffusion_model.{k.replace('base_model.model.', '', 1)}": v for k, v in state_dict.items()}
    state_dict = {k: v.to(torch.bfloat16).contiguous() for k, v in state_dict.items()}
    if verify_noop:
        assert_exported_lora_is_noop(state_dict)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lora_weights_step_{step:05d}.safetensors"
    if CONTRACT_KEY in metadata:
        record = json.loads(metadata[CONTRACT_KEY])
        record["adapter"]["tensor_shapes"] = {key: list(value.shape) for key, value in state_dict.items()}
        validate_contract(record)
        metadata = {CONTRACT_KEY: json.dumps(record, sort_keys=True)}
    atomic_write(path, lambda temporary: save_file(state_dict, temporary, metadata=metadata))
    return path


def assert_exported_lora_is_noop(state_dict: dict[str, torch.Tensor]) -> None:
    """Raise unless an exported, newly-created LoRA has an exactly-zero B projection.

    A zero B is the standard LoRA no-op initialization: A may be random, but B @ A
    is zero. Checking only B catches changed PEFT initialization without rejecting
    the intended random A initialization.
    """
    b_weights = {name: value for name, value in state_dict.items() if ".lora_B" in name}
    if not b_weights:
        raise RuntimeError("step-0 LoRA export has no lora_B weights; cannot prove it is a no-op")
    nonzero = [name for name, value in b_weights.items() if torch.count_nonzero(value).item()]
    if nonzero:
        raise RuntimeError(
            "refusing to write a purported step-0 checkpoint with a non-zero LoRA delta: " + ", ".join(nonzero[:5])
        )


CONTRACT_KEY = "onestep_avatar_contract"


def make_contract(settings: RunSettings, membership: dict, frame_plan: dict, step: int) -> dict:
    """Record actual training conditions without borrowing fields from another mode."""
    from scripts.onestep_avatar.training.config import SIGMA_SAMPLING, training_sigmas  # noqa: PLC0415

    selected = [sample for sample in frame_plan["samples"] if sample["split"] == settings.split]
    source = next(s for s in membership["sources"] if s["relative_dir"] == selected[0]["source"])
    channels, _, height, width = source["shape"]
    levels = list(training_sigmas(settings))
    mode = settings.mode_settings
    base = settings.base_identity
    record = {
        "schema_version": 2,
        "mode": settings.mode,
        "attention": mode.attention,
        "model": {
            "version": settings.model,
            "variant": settings.variant,
            "base_file": base["base_transformer_file"],
            "base_sha256": base["base_transformer_sha256"],
        },
        "task": {
            "guide_mode": settings.guide_mode,
            "objective": settings.objective,
            "first_frame_conditioning": "clean_c0_v1",
            "loss": "full_frame_x0_mse",
        },
        "shape": {
            "channels": channels,
            "height": height,
            "width": width,
            "frame_counts": sorted({sum(end - start for start, end in sample["ranges"]) for sample in selected}),
        },
        "training": {
            "sigma_levels": levels,
            "schedules": [[level, 0.0] for level in levels],
            "sigma_sampling": SIGMA_SAMPLING,
            "global_sigma_dtype": SIGMA_PRECISION,
            "noise_policy": settings.noise_policy,
            "seeds": {"init": settings.init_seed, "data": settings.data_seed, "noise": settings.noise_seed},
        },
        "data": {
            "membership_sha256": membership["sha256"],
            "frame_plan_sha256": frame_plan["sha256"],
            "coverage": [{"source": sample["source"], "ranges": sample["ranges"]} for sample in selected],
        },
        "adapter": {
            "rank": settings.lora_rank,
            "alpha": settings.lora_alpha,
            "target": settings.lora_target,
            "target_modules": LORA_TARGETS[settings.lora_target],
            "step": step,
            "parent": settings.parent_contract,
            "application_method": "peft_unmerged_fp32",
            "tensor_shapes": {},
        },
        "mode_settings": asdict(mode),
    }
    if mode.start_policy == "random":
        from scripts.onestep_avatar.training.config import window_start_draw  # noqa: PLC0415

        if frame_plan.get("start_draw") != window_start_draw(settings.noise_seed):
            raise ValueError("adapter frame-plan random start draw differs")
        frames = {s["relative_dir"]: s["n_latent_frames"] for s in membership["sources"]}
        record["data"]["segment_selection"] = random_segment_selection(
            record["data"]["coverage"], frames, mode.span_latent_frames, settings.noise_seed
        )
    if settings.mode == "causal":
        record["causal"] = {
            "block_latent_frames": mode.block_latent_frames,
            "blocks_per_sample": mode.blocks_per_sample,
            "context_latent_frames": mode.context_latent_frames,
            "sink_latent_frames": 1,
            "history_policy": "capture" if mode.teacher_forcing else "generated",
            "priming": "capture_prefix_global_sigma0",
            "refresh": "cached_refresh_global_sigma0",
        }
    validate_contract(record, require_tensor_shapes=False)
    return record


def validate_contract(record: dict, *, require_tensor_shapes: bool = True) -> None:
    """Reject incomplete settings; calibration is not inferred from filenames or K."""
    if not isinstance(record, dict) or record.get("schema_version") != 2:
        raise ValueError("adapter requires a version-two onestep_avatar_contract record")
    for key in ("model", "task", "shape", "training", "data", "adapter", "mode_settings"):
        if not isinstance(record.get(key), dict):
            raise ValueError(f"adapter contract is missing {key}")
    mode = record.get("mode")
    if not isinstance(mode, str) or mode not in {"bidirectional", "causal"}:
        raise ValueError("adapter contract mode is missing or unclassified")
    attention = "full_bidirectional" if mode == "bidirectional" else "block_causal"
    if record.get("attention") != attention:
        raise ValueError("adapter attention does not match its explicit mode")
    _validate_model_task_shape(record)
    _validate_training_data(record)
    _validate_adapter(record, require_tensor_shapes)
    _validate_mode(record)


def require_digest(value: object, label: str) -> None:
    """Require the shared adapter digest syntax before checking file identity."""
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"adapter requires a full {label} SHA-256")


def _validate_model_task_shape(record: dict) -> None:
    model, task, shape = record["model"], record["task"], record["shape"]
    if (
        not isinstance(model.get("variant"), str)
        or model["variant"] not in {"dev", "distilled"}
        or not isinstance(model.get("version"), str)
        or not model["version"]
        or not isinstance(model.get("base_file"), str)
        or not model["base_file"]
    ):
        raise ValueError("adapter base version, variant and filename must be recorded")
    require_digest(model.get("base_sha256"), "base")
    if (
        not isinstance(task.get("guide_mode"), str)
        or task["guide_mode"] not in {"d0", "d1"}
        or not isinstance(task.get("objective"), str)
        or task["objective"] not in {"bg", "white"}
    ):
        raise ValueError("adapter task and background must be recorded")
    if task.get("first_frame_conditioning") != "clean_c0_v1" or task.get("loss") != "full_frame_x0_mse":
        raise ValueError("adapter first-image/loss rules are unsupported")
    for key in ("channels", "height", "width"):
        if type(shape.get(key)) is not int or shape[key] < 1:
            raise ValueError(f"adapter shape {key} must be a positive integer")
    counts = shape.get("frame_counts")
    if not isinstance(counts, list) or not counts or any(type(v) is not int or v < 1 for v in counts):
        raise ValueError("adapter selected encoded frame counts are missing")


def _validate_training_data(record: dict) -> None:
    training = record["training"]
    if "global_sigma_dtype" in training and training["global_sigma_dtype"] not in ("float32", "bfloat16"):
        raise ValueError("adapter global_sigma_dtype is unsupported")
    levels = training.get("sigma_levels")
    if (
        not isinstance(levels, list)
        or not levels
        or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in levels)
    ):
        raise ValueError("adapter sigma levels must be finite and in (0, 1]")
    if len(set(levels)) != len(levels) or training.get("schedules") != [[v, 0.0] for v in levels]:
        raise ValueError("adapter requires unique exact levels and direct schedules")
    if training.get("sigma_sampling") != "iid_uniform_v1" or training.get("noise_policy") not in (
        "fresh",
        "fixed_per_chain",
    ):
        raise ValueError("adapter sigma/noise draw rules are missing or unsupported")
    if not isinstance(training.get("seeds"), dict) or any(
        type(training["seeds"].get(k)) is not int for k in ("init", "data", "noise")
    ):
        raise ValueError("adapter init/data/noise seeds must be recorded")
    if any(not record["data"].get(k) for k in ("membership_sha256", "frame_plan_sha256", "coverage")):
        raise ValueError("adapter list/plan hashes and selected coverage must be recorded")
    for key in ("membership_sha256", "frame_plan_sha256"):
        require_digest(record["data"][key], key)
    _validate_coverage(record)


def _validate_coverage(record: dict) -> None:
    coverage = record["data"]["coverage"]
    if not isinstance(coverage, list) or not coverage:
        raise ValueError("adapter selected coverage must be a nonempty list")
    counts = set()
    for sample in coverage:
        if not isinstance(sample, dict) or not isinstance(sample.get("source"), str) or not sample["source"]:
            raise ValueError("adapter selected video identity is missing")
        ranges = sample.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError("adapter selected frame ranges are missing")
        for bounds in ranges:
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(type(v) is not int for v in bounds)
                or not 0 <= bounds[0] < bounds[1]
            ):
                raise ValueError("adapter selected frame ranges are invalid")
        counts.add(sum(end - start for start, end in ranges))
    if counts != set(record["shape"]["frame_counts"]):
        raise ValueError("adapter frame counts differ from selected coverage")
    if record["mode_settings"].get("start_policy") == "random":
        selection = record["data"].get("segment_selection")
        if not isinstance(selection, dict):
            raise ValueError("adapter random segment selection is missing")
        expected = random_segment_selection(
            coverage, selection.get("master_frames"), record["mode_settings"].get("span_latent_frames"),
            record["training"]["seeds"]["noise"],
        )
        if selection != expected:
            raise ValueError("adapter random segment selection differs from its mode, seed or templates")
    elif record["data"].get("segment_selection") is not None:
        raise ValueError("clip-start adapter cannot carry a random segment selection")


def random_segment_selection(coverage: list[dict], frames: dict, window: int | None, seed: int) -> dict:
    """Record allowed random windows, not invented executed sample ranges."""
    from scripts.onestep_avatar.training.config import window_start_draw  # noqa: PLC0415

    if (window is not None and (type(window) is not int or window < 2)) or not isinstance(frames, dict):
        raise ValueError("random segment selection needs a window and master frame inventory")
    sources = dict.fromkeys(sample["source"] for sample in coverage)
    selected = {source: frames.get(source) for source in sources}
    if any(type(count) is not int or count < (2 if window is None else window) for count in selected.values()):
        raise ValueError("random segment window does not fit its masters")
    if any(sample["ranges"] != [[0, window or selected[sample["source"]]]] for sample in coverage):
        raise ValueError("random segment coverage must contain window templates")
    return {
        "coverage_role": "window_templates",
        "window_latent_frames": window,
        "start_draw": window_start_draw(seed),
        "master_frames": selected,
        "start_bounds_inclusive": {source: [0, count - (window or count)] for source, count in selected.items()},
        "first_image": "selected_capture_master_frame",
        "positions": "restart_at_zero",
        "known_gap": "G9",
    }


def _validate_adapter(record: dict, require_tensor_shapes: bool) -> None:
    adapter = record["adapter"]
    if (
        type(adapter.get("rank")) is not int
        or adapter["rank"] < 1
        or type(adapter.get("alpha")) is not int
        or adapter["alpha"] != adapter["rank"]
    ):
        raise ValueError("adapter requires a positive rank and supported alpha = rank")
    if (
        not isinstance(adapter.get("target"), str)
        or adapter["target"] not in LORA_TARGETS
        or adapter.get("target_modules") != LORA_TARGETS.get(adapter.get("target"))
    ):
        raise ValueError("adapter targets must be recorded")
    if (
        type(adapter.get("step")) is not int
        or adapter["step"] < 0
        or adapter.get("application_method") != "peft_unmerged_fp32"
        or "parent" not in adapter
    ):
        raise ValueError("adapter update and actual training application must be recorded")
    if require_tensor_shapes and not adapter.get("tensor_shapes"):
        raise ValueError("adapter exported tensor shapes are missing")
    if require_tensor_shapes:
        _validate_tensor_shapes(adapter)


def _validate_mode(record: dict) -> None:  # noqa: PLR0912 -- compare declared mode fields and causal rules together
    mode = record["mode"]
    if mode == "bidirectional":
        if "causal" in record or any(
            k in record["mode_settings"] for k in ("block_latent_frames", "context_latent_frames", "teacher_forcing")
        ):
            raise ValueError("bidirectional adapter cannot contain causal history settings")
    else:
        causal = record.get("causal")
        if not isinstance(causal, dict) or causal.get("history_policy") not in ("capture", "generated"):
            raise ValueError("causal adapter history must be recorded")
        if any(type(causal.get(k)) is not int or causal[k] < 1 for k in ("block_latent_frames", "blocks_per_sample")):
            raise ValueError("causal adapter block geometry is missing")
        if type(causal.get("context_latent_frames")) is not int or not 0 <= causal["context_latent_frames"] <= 16:
            raise ValueError("causal adapter context depth is invalid")
        if (
            causal.get("sink_latent_frames") != 1
            or causal.get("priming") != "capture_prefix_global_sigma0"
            or causal.get("refresh") != "cached_refresh_global_sigma0"
        ):
            raise ValueError("causal adapter first-image/priming/refresh rules are unsupported")
    settings = record["mode_settings"]
    keys = {"span_latent_frames", "start_policy", "attention"}
    if mode == "causal":
        keys.update({"block_latent_frames", "blocks_per_sample", "context_latent_frames", "teacher_forcing"})
    if set(settings) != keys:
        raise ValueError("adapter mode settings are missing fields or contain unrelated fields")
    if settings.get("attention") != record["attention"] or settings.get("start_policy") not in ("clip_start", "random"):
        raise ValueError("adapter mode attention/start policy is missing or inconsistent")
    span = settings.get("span_latent_frames")
    if span is not None and (type(span) is not int or span < 2):
        raise ValueError("adapter segment length is invalid")
    if mode == "causal":
        if settings["start_policy"] == "random" and (
            settings["blocks_per_sample"] != 1 or span != settings["block_latent_frames"] + 1
        ):
            raise ValueError("causal random segments require one block and window length B+1")
        for key in ("block_latent_frames", "blocks_per_sample", "context_latent_frames"):
            if settings.get(key) != record["causal"][key]:
                raise ValueError(f"adapter mode/causal {key} are inconsistent")
        if type(settings.get("teacher_forcing")) is not bool or settings["teacher_forcing"] != (
            record["causal"]["history_policy"] == "capture"
        ):
            raise ValueError("adapter capture/generated history settings are inconsistent")


def _validate_tensor_shapes(adapter: dict) -> None:
    shapes = adapter["tensor_shapes"]
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("adapter exported tensor inventory is missing")
    rank = adapter["rank"]
    for key, shape in shapes.items():
        if not isinstance(key, str) or not key.startswith("diffusion_model."):
            raise ValueError("adapter tensor keys must use the exported ComfyUI format")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError(f"adapter matrix shape is invalid: {key}")
        if ".lora_A." in key:
            peer = key.replace(".lora_A.", ".lora_B.")
            peer_shape = shapes.get(peer)
            if shape[0] != rank or not isinstance(peer_shape, list) or len(peer_shape) != 2 or peer_shape[1] != rank:
                raise ValueError(f"adapter A/B rank differs: {key}")
        elif ".lora_B." in key:
            if shape[1] != rank or key.replace(".lora_B.", ".lora_A.") not in shapes:
                raise ValueError(f"adapter B/A rank differs: {key}")
        else:
            raise ValueError(f"unsupported non-LoRA tensor: {key}")


def read_contract(path: Path) -> dict:
    """Read and check the single metadata record before opening a transformer."""
    metadata = read_adapter_metadata(path)
    if CONTRACT_KEY not in metadata:
        raise ValueError("adapter has no version-two contract; explicitly convert the original with its run records")
    try:
        record = json.loads(metadata[CONTRACT_KEY])
    except (ValueError, TypeError) as error:
        raise ValueError("adapter contract is not valid JSON") from error
    validate_contract(record)
    return record


def recheck_adapter(path: Path, contract: dict, expected_sha256: str | None) -> None:
    """Repeat the one metadata/tensor validator against unchanged preflight bytes."""
    require_digest(expected_sha256, "adapter")
    if sha256(path) != expected_sha256:
        raise ValueError("adapter content changed since preflight")
    current = read_contract(path)
    if current != contract:
        raise ValueError("adapter contract changed since preflight")
    validate_adapter_tensors(path, current)
    if sha256(path) != expected_sha256:
        raise ValueError("adapter content changed during revalidation")


def contract_condition_problems(record: dict, requested: dict) -> list[str]:  # noqa: PLR0912, PLR0915 -- report each condition
    """Compare complete execution conditions, excluding evaluation people and data hashes."""
    from scripts.onestep_avatar.model.sampling import validate_schedule  # noqa: PLC0415

    validate_contract(record)
    required = ("mode", "model", "task", "shape", "schedule", "mode_settings")
    if any(key not in requested for key in required):
        raise ValueError("requested adapter conditions are incomplete")
    if any(not isinstance(requested[k], dict) for k in ("model", "task", "shape", "mode_settings")):
        raise ValueError("requested model/task/shape/mode settings must be records")
    if any(
        type(requested["shape"].get(k)) is not int or requested["shape"][k] < 1
        for k in ("channels", "height", "width", "frames")
    ):
        raise ValueError("requested encoded shape must contain positive integers")
    schedule = list(validate_schedule(requested["schedule"]))
    problems = []
    precision = requested.get("global_sigma_dtype")
    trained_precision = record["training"].get("global_sigma_dtype")
    if precision not in ("float32", "bfloat16"):
        raise ValueError("requested global_sigma_dtype is missing or unsupported")
    if trained_precision not in ("float32", "bfloat16"):
        raise ValueError("adapter global_sigma_dtype is unknown; historical calibration needs producer evidence")
    if precision != trained_precision:
        problems.append(f"global_sigma_dtype: adapter {trained_precision!r}, requested {precision!r}")
    method = requested.get("application_method")
    if method not in ("peft_unmerged_fp32", "fused_bf16"):
        raise ValueError("requested application_method is missing or unsupported")
    if method != record["adapter"]["application_method"]:
        problems.append(f"application_method: adapter {record['adapter']['application_method']!r}, "
                        f"requested {method!r}; changed adapter function")
    if requested["mode"] == "causal":
        if requested.get("history_mode") not in ("cache", "recompute", "joint"):
            raise ValueError("requested causal history_mode is missing or unsupported")
        if requested.get("kv_source") not in ("refresh", "denoise"):
            raise ValueError("requested causal kv_source is missing or unsupported")
        if requested["kv_source"] == "denoise" and (
            requested["history_mode"] != "cache" or requested["mode_settings"].get("teacher_forcing") is not False
        ):
            raise ValueError("denoise K/V requires cached generated history")
        if record["mode"] == "causal":
            for field, trained in (("history_mode", "cache"), ("kv_source", "refresh")):
                if requested[field] != trained:
                    problems.append(
                        f"{field}: adapter {trained!r} ({record['causal']['refresh']}), "
                        f"requested {requested[field]!r}; changed continuation computation"
                    )
    elif any(field in requested for field in ("history_mode", "kv_source")):
        raise ValueError("bidirectional requests cannot contain causal history fields")
    for section in ("mode", "model", "task", "shape", "mode_settings"):
        actual = requested[section]
        if section in {"model", "task"}:
            for key, value in record[section].items():
                if key == "base_file":
                    continue
                if actual.get(key) != value:
                    problems.append(f"{section}.{key}: adapter {value!r}, requested {actual.get(key)!r}")
        elif section == "shape":
            for key in ("channels", "height", "width"):
                if actual.get(key) != record[section][key]:
                    problems.append(f"shape.{key}: adapter {record[section][key]}, requested {actual.get(key)}")
            if actual.get("frames") not in record[section]["frame_counts"]:
                problems.append(
                    f"shape.frames: adapter {record[section]['frame_counts']}, requested {actual.get('frames')}"
                )
        elif actual != record[section]:
            problems.append(f"{section}: adapter {record[section]!r}, requested {actual!r}")
    if schedule not in record["training"]["schedules"]:
        problems.append(f"schedule: adapter {record['training']['schedules']}, requested {schedule}")
    return problems


def check_contract(record: dict, requested: dict, *, override: bool = False, product: bool = False) -> list[str]:
    """Return recorded research differences; product calls always reject differences."""
    problems = contract_condition_problems(record, requested)
    if problems and (product or not override):
        raise ValueError("incompatible adapter conditions: " + "; ".join(problems))
    return problems


def validate_adapter_tensors(path: Path, record: dict) -> None:
    """Check exported key/shape/rank evidence before loading any transformer weights."""
    from safetensors import safe_open  # noqa: PLC0415

    validate_contract(record)
    expected = record["adapter"]["tensor_shapes"]
    rank = record["adapter"]["rank"]
    with safe_open(str(path), framework="pt") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("adapter tensor inventory differs from its contract")
        for key, shape in expected.items():
            actual = list(handle.get_slice(key).get_shape())
            if actual != shape or len(actual) != 2:
                raise ValueError(f"adapter tensor shape differs: {key}")
            if ".lora_A." in key:
                peer = key.replace(".lora_A.", ".lora_B.")
                if actual[0] != rank or peer not in expected or expected[peer][1] != rank:
                    raise ValueError(f"adapter A/B rank differs: {key}")
            elif ".lora_B." in key:
                if actual[1] != rank or key.replace(".lora_B.", ".lora_A.") not in expected:
                    raise ValueError(f"adapter B/A rank differs: {key}")
            else:
                raise ValueError(f"unsupported non-LoRA tensor: {key}")
