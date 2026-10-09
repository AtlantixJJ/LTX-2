"""The one typed training engine: explicit-mode settings, checked inputs and LoRA updates.

See doc/training/engine.md. ``run_settings`` is the only training entry point; the
mode-less chain runtime and its subset readers retired in refactor group G11.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft.utils.other import fsdp_auto_wrap_policy

from ltx_trainer.model_loader import load_transformer
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.corpus import subset as video_lists
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, backbone, bidirectional, common
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model.common import FULL_FRAME_X0_MSE
from scripts.onestep_avatar.training import checkpoints as sampling
from scripts.onestep_avatar.training import numerics, resources, runtime
from scripts.onestep_avatar.training.checkpoints import load_stage_init, save_lora
from scripts.onestep_avatar.training.config import (
    BidirectionalSettings,
    CausalSettings,
    RunSettings,
    select_frame_plan,
    sigma_for_rank,
    training_noise_seed,
    training_sigmas,
)
from scripts.prune.core import model_registry
from scripts.prune.core.session import DEFAULT_PROMPT
from scripts.prune.data import prompt_cache

LOGGER = logging.getLogger("onestep_avatar.train")


def _launch_evidence(settings: RunSettings, job_digest: str | None) -> tuple[dict | None, Path | None]:
    """Read dispatch authority before model setup; never infer a missing launch."""
    from scripts.onestep_avatar.execution import queue  # noqa: PLC0415 -- public canonical launch authority
    from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_ENV  # noqa: PLC0415 -- model-free protocol
    from scripts.onestep_avatar.training.config import parse_settings  # noqa: PLC0415

    path = os.environ.get(LAUNCH_ENV)
    if path is None:
        if job_digest is not None:
            raise ValueError("queued training requires original dispatch launch evidence")
        return None, None
    launch = queue.read_training_launch(Path(path))
    if launch["schema_version"] != 2:
        raise ValueError("current typed training requires schema-two numerical launch evidence")
    numerics.require_environment(launch["numerical_environment"])
    if launch["job"]["sha256"] != job_digest:
        raise ValueError("dispatch launch identity differs from queued attempt")
    requested = parse_settings(launch["job"]["arguments"])
    ignored = {"base_identity", "parent_contract", "world_size", "preview_record"}
    def normalized(value: object) -> object:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items() if key not in ignored}
        return value
    if normalized(requested.as_dict()) != normalized(settings.as_dict()):
        raise ValueError("dispatch training arguments differ from actual settings")
    return launch, Path(path).resolve()


def _check_launch_current(launch: dict | None, path: Path | None) -> None:
    if launch is not None:
        from scripts.onestep_avatar.execution import queue  # noqa: PLC0415 -- same preflight/publication authority
        if queue.read_training_launch(path) != launch:
            raise ValueError("original dispatch launch changed during training")


def _launch_precision(launch: dict) -> str:
    import base64  # noqa: PLC0415 -- original bytes, never today's requested precision

    import yaml  # noqa: PLC0415 -- installed Accelerate YAML dependency
    return yaml.safe_load(base64.b64decode(launch["accelerate_config_bytes_base64"]))["mixed_precision"]


@contextlib.contextmanager
def timed(label: str) -> Iterator[None]:
    """Bracket a startup phase with a begin/end line carrying its wall duration.

    Startup here is minutes of silent 42 GB checkpoint I/O followed by a collective
    (``prepare``), and both look identical to a hang from outside. Worse, the failure mode
    that actually happens is ONE straggling or dead rank, not a uniformly slow run -- so
    this logs on every rank rather than the main process, and ``ltx_trainer``'s ``[rank N]``
    prefix is the whole point of it. The lines are unconditional because they are a few per
    run; the per-step breakdown, which is per-block, is behind ``--timing``.

    **The prefix is ``timing |``, not ``[timing]``, on purpose.** ``ltx_trainer`` installs a
    ``RichHandler``, which reads ``[...]`` as console markup and silently DROPS an unknown
    tag -- the bracketed form vanished from the log entirely and no grep for it ever matched.
    (``ltx_trainer``'s own ``[rank N]`` survives only because its format string escapes it.)
    """
    LOGGER.info("timing | %s: begin", label)
    started = time.time()
    yield
    LOGGER.info("timing | %s: done in %.1fs", label, time.time() - started)


DTYPE = torch.bfloat16


def assert_rank_lockstep(accelerator: Accelerator, planned_forwards: int, source: str) -> None:
    """Refuse a step whose ranks would issue different numbers of transformer forwards.

    Under FSDP FULL_SHARD a forward is a round of all-gathers, so ranks that run a different
    NUMBER of them desynchronise the collective stream and the job deadlocks -- silently, at
    100 % GPU utilisation, until a watchdog fires minutes later blaming something unrelated.
    That cost a long debugging session on 2026-09-16, when ``prime_cache`` skipped its forward
    for clip-start chains (fixed there; see its empty-spans branch).

    This runs BEFORE the step's forwards, while the ranks are still in step from the previous
    optimizer update, so the gather here is itself safely matched. Checking afterwards cannot
    work: by then the mismatched collective has already been enqueued and this gather would
    join the pile-up rather than report it.

    The count is ``1 prime + K denoise + (K - 1) refresh``. The last refresh is skipped.
    Every rank must compute the same count.
    """
    if accelerator.num_processes == 1:
        return
    counts = accelerator.gather(torch.tensor([planned_forwards], device=accelerator.device, dtype=torch.long)).tolist()
    if len(set(counts)) > 1:
        raise SystemExit(
            f"rank forward counts disagree for this step: {counts} (this rank: "
            f"{planned_forwards}, chain from {source}). Every rank must run the same number of "
            f"transformer forwards per step or FSDP's collectives desynchronise and the job "
            f"hangs instead of failing. Something made a forward conditional on the data -- "
            f"chain length, priming, or an early return on a short clip."
        )


def build_transformer(
    model: model_registry.RefinerModel, args: RunSettings, accelerator: Accelerator
) -> torch.nn.Module:
    """Load the frozen bf16 backbone and inject LoRA -- ``ltx_trainer``'s own plumbing."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Loading straight onto the GPU, not via host RAM: the full bf16 checkpoint is 42 GB and
    # three ranks staging it on the host would need 126 GB of a machine that has ~139 GB free
    # here. FSDP shards in place afterwards, so the 42 GB is transient and fits a 49 GB card.
    init_device = args.init_device if args.init_device != "cuda" else f"cuda:{local_rank}"
    transformer = load_transformer(
        checkpoint_path=model.paths.transformer(), device=init_device, dtype=DTYPE, video_only=True
    )
    transformer.requires_grad_(False)
    # PEFT draws lora_A from the global RNG. Seeding it identically on every rank makes the
    # initialization one reproducible function of --init-seed instead of a per-rank mixture.
    transformer = adapters.attach(transformer, rank=args.lora_rank, alpha=args.lora_alpha,
                                  target=args.lora_target, init_seed=args.init_seed)
    if args.init_adapter is not None:
        load_stage_init(transformer, args.init_adapter)
    if accelerator.distributed_type == DistributedType.FSDP:
        # FSDP needs one dtype per flat parameter, and PEFT makes the adapters fp32 against a
        # bf16 base. This policy wraps the trainable leaves separately, which is what lets the
        # base stay bf16 instead of being promoted to a full fp32 host copy before sharding.
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(transformer)
        plugin = accelerator.state.fsdp_plugin
        from torch.distributed.fsdp.wrap import CustomPolicy  # noqa: PLC0415 -- FSDP launches only

        peft_policy = plugin.auto_wrap_policy

        def typed_policy(module: torch.nn.Module) -> bool | dict:
            if not peft_policy(module=module, recurse=False, nonwrapped_numel=0):
                return False
            parameters = list(module.parameters())
            if parameters and all(parameter.requires_grad for parameter in parameters):
                return {"mixed_precision": None}
            return True

        plugin.auto_wrap_policy = CustomPolicy(typed_policy)
        if plugin.mixed_precision_policy is not None:
            # FSDP otherwise recursively rounds sigma/timesteps/positions in Modality.
            plugin.mixed_precision_policy = replace(
                plugin.mixed_precision_policy, cast_root_forward_inputs=False
            )
    transformer.get_base_model().set_gradient_checkpointing(not args.no_gradient_checkpointing)
    return transformer


def wandb_is_enabled(args: argparse.Namespace) -> bool:
    """True if W&B logging is enabled for this run."""
    if getattr(args, "no_wandb", False):
        return False
    if not args.wandb_project or str(args.wandb_project).strip().lower() in ("", "none"):
        return False
    return getattr(args, "wandb_mode", "online") != "disabled"


def init_wandb(args: argparse.Namespace, *, config: dict) -> object | None:
    """Create one online W&B run on rank 0; all ranks still participate in metric gathers."""
    if not wandb_is_enabled(args):
        return None
    try:
        import wandb  # noqa: PLC0415 -- optional dependency, imported only when requested.
    except ImportError as exc:  # pragma: no cover - environment/setup error
        raise SystemExit(
            f"W&B logging is enabled (project: {args.wandb_project!r}), but the wandb package "
            "is not installed in the active environment. Install wandb or pass --no-wandb."
        ) from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.output.name,
        mode=args.wandb_mode,
        config=config,
    )


def rank_mean(accelerator: Accelerator, values: list[float]) -> list[float]:
    """All-rank mean for W&B, keeping one chart point per optimiser step."""
    local = torch.tensor(values, device=accelerator.device, dtype=torch.float32)
    gathered = accelerator.gather(local).reshape(accelerator.num_processes, -1)
    return gathered.mean(dim=0).cpu().tolist()


def archive_existing_run(output: Path) -> Path | None:
    """Move everything already in ``output`` aside before ``--overwrite`` reuses the directory.

    The old behaviour deleted only ``metrics_rank*.jsonl``, so a relaunch's fresh checkpoints
    and probes landed beside a prior run's under the same directory name with nothing to say
    which run produced which (F6). Archiving the whole directory into one timestamped
    subdirectory keeps that provenance intact and recoverable instead of silently mixed or
    discarded. Returns ``None`` (and does nothing) if there is nothing to move.
    """
    entries = [entry for entry in output.iterdir() if not entry.name.startswith("archived_")]
    if not entries:
        return None
    archive_dir = output / f"archived_{time.strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir()
    for entry in entries:
        entry.rename(archive_dir / entry.name)
    return archive_dir


def read_preview_inputs(path: Path, settings: RunSettings) -> dict:  # noqa: PLR0912 -- ordered pinned input gates
    """Pin all preview inputs before model setup; no execution occurs here."""
    record = json.loads(path.read_text())
    if record.get("schema_version") != 2 or record.get("kind") != "onestep_avatar.preview_inputs":
        raise ValueError("preview inputs require a version-two fixed record")
    if record.get("mode") != settings.mode:
        raise ValueError("preview mode differs from the training mode")
    files = record.get("input_files")
    required = {"subset", "capture", "first_image", "text", "noise"}
    if settings.guide_mode == "d1":
        required.add("guide")
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("preview record lacks fixed capture/guide/image/text/noise inputs")
    for role, identity in files.items():
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError(f"preview {role} identity is malformed")
        source = Path(identity["path"])
        if not source.is_absolute() or sha256(source) != identity.get("sha256"):
            raise ValueError(f"preview {role} file changed or lacks an absolute pinned path")
        if role in required - {"subset"}:
            tensor_hash = identity.get("tensor_sha256")
            if not isinstance(tensor_hash, str) or len(tensor_hash) != 64:
                raise ValueError(f"preview {role} requires the actual selected tensor identity")
    arguments = record.get("evaluation_arguments")
    if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
        raise ValueError("preview evaluation arguments must be a list of strings")
    if any(value.split("=", 1)[0] in {"--output", "--checkpoint", "--dry-run"} for value in arguments):
        raise ValueError("preview executor owns checkpoint/output arguments")
    from scripts.onestep_avatar.evaluate import parse_args as parse_evaluation  # noqa: PLC0415 -- no model session

    args = parse_evaluation([*arguments, "--output", str(settings.output / "previews")])
    if args.cfg != 1.0:
        negative = files.get('negative_text')
        if not isinstance(negative, dict) or not isinstance(negative.get('tensor_sha256'), str):
            raise ValueError('guided preview requires pinned negative text')
    if args.mode != settings.mode or args.guide_mode != settings.guide_mode:
        raise ValueError("preview arguments differ from the recorded training task")
    if args.subset.resolve() != Path(files["subset"]["path"]).resolve():
        raise ValueError("preview arguments use an unpinned video list")
    if args.noise_file is None or args.noise_file.resolve() != Path(files["noise"]["path"]).resolve():
        raise ValueError("preview arguments must consume their pinned noise file")
    if args.schedule != record.get("schedule"):
        raise ValueError("preview arguments differ from the exact recorded schedule")
    from scripts.onestep_avatar.previews import check_preview_reference_bundle  # noqa: PLC0415 -- saved pixels only

    check_preview_reference_bundle(record)
    record["sha256"] = video_lists.record_hash(record)
    return record


def enqueue_preview(checkpoint: Path, record: dict, output: Path) -> Path:
    """Publish an idempotent pending job only for a completed checkpoint."""
    marker = json.loads(checkpoint.with_suffix(".complete.json").read_text())
    if marker.get("state") != "complete" or marker.get("sha256") != sha256(checkpoint):
        raise ValueError("preview checkpoint is incomplete or changed")
    identity = hashlib.sha256((marker["sha256"] + record["sha256"]).encode()).hexdigest()
    job = {
        "schema_version": 2,
        "kind": "onestep_avatar.preview_job",
        "id": identity,
        "state": "pending",
        "checkpoint": {**marker, "path": str(checkpoint.resolve())},
        "fixed_inputs": record,
        "output": str((output / "previews" / identity).resolve()),
    }
    destination = output / "preview_jobs" / f"{identity}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get("fixed_inputs") != record or existing.get("checkpoint") != job["checkpoint"]:
            raise ValueError("preview job identity collides with different pinned inputs")
        return destination
    dataset.atomic_write(destination, lambda temporary: temporary.write_text(json.dumps(job, indent=2) + "\n"))
    return destination


def prepare_run(  # noqa: PLR0912 -- fail all input/parent conditions before runtime setup
    settings: RunSettings,
    *, require_fresh_output: bool = True,
) -> tuple[dataset.ClipStore, dict, model_registry.RefinerModel, bool]:
    """Check the new runtime's inputs and parent before model or output mutation."""
    expected_type = BidirectionalSettings if settings.mode == "bidirectional" else CausalSettings
    if settings.mode not in {"bidirectional", "causal"} or not isinstance(settings.mode_settings, expected_type):
        raise ValueError("run mode and typed mode settings disagree")
    if settings.preview_inputs is not None:
        settings.preview_record = read_preview_inputs(settings.preview_inputs, settings)
    if settings.output.exists() and not settings.output.is_dir():
        raise ValueError("output must be a directory")
    used = settings.output.is_dir() and any(settings.output.iterdir())
    if require_fresh_output and used and not settings.overwrite:
        raise ValueError("output already has a run; explicit overwrite archives it after all checks and setup succeed")
    membership = json.loads(settings.subset.read_text())
    video_lists.validate_membership(membership)
    if membership["objective"] != settings.objective:
        raise ValueError("requested background differs from the fixed video list")
    store = dataset.ClipStore(membership, settings.corpus_root)
    store.verify(require_guide=settings.guide_mode == "d1")
    specification = backbone.resolve(settings.model, settings.variant)
    from scripts.onestep_avatar.corpus.precompute import file_fingerprint  # noqa: PLC0415 -- producer identity

    vae_fingerprint = file_fingerprint(Path(specification.paths.video_vae()))
    for source_id, source in store.sources.items():
        for role in (("capture", "guide") if settings.guide_mode == "d1" else ("capture",)):
            if source.get(f"{role}_encode_record", {}).get("vae_fingerprint") != vae_fingerprint:
                raise ValueError(f"{source_id}: {role} encoding VAE differs from the selected base VAE")
    levels = training_sigmas(settings)
    if settings.variant == "distilled" and any(
        not any(abs(level - v) < 1e-9 for v in specification.sigmas) for level in levels
    ):
        raise ValueError("requested training levels are not on the selected distilled base grid")
    plan = select_frame_plan(settings, membership, specification.scale_factors)
    for sample in plan["samples"]:
        if sample["split"] != settings.split:
            continue
        source = store.sources[sample["source"]]
        if source["shape"][0] != specification.caps.latent_channels:
            raise ValueError(f"{sample['source']}: encoded channel count differs from the selected base")
        mode = settings.mode_settings
        frames = (
            sum(end - start for start, end in sample["ranges"])
            if settings.mode == "bidirectional" or mode.start_policy == "random"
            else sample["ranges"][-1][1]
        )
        seconds = common.pixel_frames_for(frames, specification.scale_factors.time) / float(source["fps"])
        if seconds > common.MAX_ROPE_SECONDS:
            raise ValueError(f"{sample['source']}: selected model positions exceed the 20-second range")
    settings.base_identity = backbone.identity(
        specification.paths.transformer(), settings.variant, settings.model, full_hash=True
    )
    if settings.init_adapter is not None:
        parent = sampling.read_contract(settings.init_adapter)
        sampling.validate_adapter_tensors(settings.init_adapter, parent)
        if parent["model"]["base_sha256"] != settings.base_identity["base_transformer_sha256"]:
            raise ValueError("parent adapter uses different base weights")
        if parent["model"]["version"] != settings.model or parent["model"]["variant"] != settings.variant:
            raise ValueError("parent adapter uses a different base version or variant")
        adapter = parent["adapter"]
        if (adapter["rank"], adapter["alpha"], adapter["target"]) != (
            settings.lora_rank,
            settings.lora_alpha,
            settings.lora_target,
        ):
            raise ValueError("parent adapter has different LoRA rank, alpha or targets")
        if parent["mode"] != settings.mode and not settings.allow_cross_mode_init:
            raise ValueError("cross-mode parent initialization requires explicit --allow-cross-mode-init")
        settings.parent_contract = {
            "path": str(settings.init_adapter),
            "sha256": sha256(settings.init_adapter),
            "original_mode": parent["mode"],
            "base_sha256": parent["model"]["base_sha256"],
            "rank": adapter["rank"],
            "alpha": adapter["alpha"],
            "target": adapter["target"],
            "initialization": "fresh_optimizer_and_random_state",
            "calibration_transferred": False,
        }
    return store, plan, specification, bool(used)


def verify_training_conditions(job: dict, checkpoint: Path, contract: dict) -> None:  # noqa: PLR0915 -- sequential evidence gates
    """Recheck queued scientific evidence without models, updates or output mutation."""
    from scripts.onestep_avatar.training.config import parse_settings  # noqa: PLC0415 -- typed command owner

    settings = parse_settings(job["arguments"])
    store, plan, _specification, _used = prepare_run(settings, require_fresh_output=False)
    settings.world_size = job["processes"]
    expected_checkpoint = settings.output / "checkpoints" / f"lora_weights_step_{settings.steps:05d}.safetensors"
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise ValueError("queue training final checkpoint path differs from requested run")
    expected_contract = sampling.make_contract(settings, store.membership, plan, settings.steps)
    expected_contract["adapter"]["tensor_shapes"] = contract["adapter"]["tensor_shapes"]
    if contract != expected_contract:
        raise ValueError("queue training adapter contract differs from requested scientific settings")
    config_path, plan_path = settings.output / "config.json", settings.output / "frame_plan.json"
    resolved = json.loads(config_path.read_text())
    software.check_current(resolved.get("software"))
    from scripts.onestep_avatar.execution import queue  # noqa: PLC0415 -- current launch authority
    launch = resolved.get("queue_launch")
    queue.verify_training_launch(launch, job)
    runtime.validate(resolved.get("runtime"), settings.world_size, _launch_precision(launch), native=True,
                     numerical_policy=True)
    budget = job.get("effective_budget", resources.read_budget(settings.resource_budget))
    resources.check_budget(budget)
    if resolved.get("resource_budget") != budget:
        raise ValueError("training resource budget differs from original run")
    resource_evidence = {}
    if budget is not None:
        measurements, _journal_identities = resources.read_records(settings.output, settings.world_size)
        snapshot, resource_evidence = resources.read_records(settings.output, settings.world_size, step=settings.steps)
        if measurements != snapshot:
            raise ValueError("final checkpoint resource snapshot differs from rank journals")
        phases = resources.training_phases(settings)
        resources.validate_records(measurements, settings.world_size, phases, budget)
    saved_plan = json.loads(plan_path.read_text())
    sample_count = sum(s["split"] == settings.split for s in plan["samples"])
    tensor_shapes = contract["adapter"]["tensor_shapes"]
    modules = {name.rsplit(".lora_", 1)[0] for name in tensor_shapes}
    expected = {
        **settings.as_dict(), "membership_sha256": store.membership["sha256"],
        "frame_plan_sha256": plan["sha256"], "samples": sample_count,
        "loss": FULL_FRAME_X0_MSE, "samples_per_update": settings.world_size * settings.chains_per_rank,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        "queue_job_sha256": job["sha256"], "producer_source_sha256": sha256(Path(__file__)),
        "software": software.capture("training", settings.mode),
        "resource_budget": budget,
        "sample_tiling": max(1, math.ceil(settings.world_size * settings.chains_per_rank / sample_count)),
        "trainable_params": sum(math.prod(shape) for shape in tensor_shapes.values()),
        "lora_modules": len(modules),
        "lora_target_counts": {
            target: sum(name.endswith(target) for name in modules)
                for target in adapters.LORA_TARGETS[settings.lora_target]
        },
    }
    # Resolve the same JSON representation used by the producer (paths and tuples).
    expected = json.loads(json.dumps(expected, default=str))
    if saved_plan != plan or any(key not in resolved or resolved[key] != value for key, value in expected.items()):
        raise ValueError("queue training saved configuration or frame plan differs from requested run")
    marker = queue.read_training_marker(checkpoint, settings.steps)
    software.check_current(marker.get("software"))
    identities = {
        "queue_job_sha256": job["sha256"], "producer_source_sha256": expected["producer_source_sha256"],
        "software": expected["software"],
        "queue_launch": launch, "runtime": resolved["runtime"],
        "resource_budget": budget, "resource_evidence": resource_evidence,
        "training_record": {"config_sha256": sha256(config_path), "frame_plan_sha256": sha256(plan_path)},
    }
    if settings.consumer_trace:
        identities["consumer_trace_evidence"] = _read_consumer_evidence(
            settings.output, settings.steps, settings.world_size,
            None if resolved.get("queue_launch_path") is None else Path(resolved["queue_launch_path"]),
            job["sha256"], resolved.get("queue_attempt_token"), settings.chains_per_rank,
        )
    if any(marker.get(key) != value for key, value in identities.items()):
        raise ValueError("queue training completed marker has different run provenance")
    if settings.save_update_state:
        text = resolved.get("update_text", {})
        text_path = settings.output / "update_states/text.pt"
        if text.get("path") != str(text_path.resolve()) or text.get("sha256") != sha256(text_path):
            raise ValueError("training update text evidence differs")
        for step in range(1, settings.steps + 1):
            path = settings.output / "update_states" / f"step_{step:05d}.pt"
            state = json.loads(path.with_suffix(".json").read_text())
            if (state.get("step") != step or state.get("sha256") != sha256(path)
                    or state.get("shapes") != tensor_shapes or state.get("software") != expected["software"]
                    or state.get("world_size") != settings.world_size
                    or state.get("accumulation") != settings.chains_per_rank
                    or state.get("optimizer") != expected["optimizer"]
                    or not isinstance(state.get("grad_norm"), (int, float))
                    or not math.isfinite(state["grad_norm"])):
                raise ValueError("training Adam update evidence differs")


def tokens_for_sample(
    video: dataset.EncodedVideo,
    sample: dict,
    settings: RunSettings,
    specification: model_registry.RefinerModel,
    device: torch.device,
    *,
    step: int,
    rank: int,
    slot: int,
) -> tuple:
    """Build one mode's independent segment or original-position causal prefix."""
    mode = settings.mode_settings
    start = 0
    end = sample["ranges"][-1][1]
    if settings.mode == "bidirectional" or mode.start_policy == "random":
        start, end = bidirectional.plan_samples(
            video.z_y.shape[1],
            span_latent_frames=mode.span_latent_frames,
            start_policy=mode.start_policy,
            seed=settings.noise_seed,
            step=step,
            rank=rank,
            slot=slot,
        )
    capture = video.z_y[:, start:end].unsqueeze(0).to(device=device, dtype=DTYPE)
    guide = None if video.z_g is None else video.z_g[:, start:end].unsqueeze(0).to(device=device, dtype=DTYPE)
    grid = common.ClipGrid.build(
        capture.shape[2],
        capture.shape[3] * specification.scale_factors.height,
        capture.shape[4] * specification.scale_factors.width,
        video.fps,
        specification,
        device=device,
        dtype=DTYPE,
        latent_channels=specification.caps.latent_channels,
    )
    return grid, grid.patchify(capture), None if guide is None else grid.patchify(guide), start, end


def _read_consumer_evidence(
    output: Path, step: int, world: int, launch_path: Path | None,
    job: str | None, token: str | None, accumulation: int | None = None,
) -> dict[str, str]:
    """Require actual complete observations, immutable bytes and the original launch."""
    from scripts.onestep_avatar.training.consumer_trace import validate  # noqa: PLC0415 -- selected diagnostic
    evidence = {}
    for rank in range(world):
        path = output / "consumer_traces" / f"step_{step:05d}" / f"rank{rank}.json"
        data = path.read_bytes()
        record = json.loads(data)
        validate(record, {"rank": rank, "world_size": world, "queue_job_sha256": job,
                          "queue_attempt_token": token,
                          "launch_sha256": None if launch_path is None else sha256(launch_path)})
        if accumulation is not None:
            expected_visits = [(update, slot) for update in range(1, step + 1) for slot in range(accumulation)]
            if [(sample["step"], sample["slot"]) for sample in record["samples"]] != expected_visits:
                raise ValueError("consumer trace sample coverage differs from completed updates")
        digest = hashlib.sha256(data).hexdigest()
        if sha256(path) != digest:
            raise ValueError("consumer trace changed during verification")
        evidence[str(path.resolve())] = digest
    return evidence


def run_settings(settings: RunSettings) -> int:
    """Record queued startup failures around the typed runtime, never retry here."""
    from scripts.onestep_avatar.training.startup import StartupEvents  # noqa: PLC0415 -- queue protocol only

    with StartupEvents() as events:
        return _run_settings(settings, events)


def _run_settings(settings: RunSettings, events) -> int:  # noqa: ANN001 -- startup event owner
    with contextlib.ExitStack() as lifecycle:
        return _run_settings_body(settings, events, lifecycle)


def _run_settings_body(settings: RunSettings, events, lifecycle: contextlib.ExitStack) -> int:  # noqa: ANN001, PLR0912, PLR0915 -- linear runtime and startup boundary
    """Run the typed two-mode engine; previews execute in their own model sessions."""
    launch, launch_path = _launch_evidence(settings, events.job)
    budget = resources.budget_from_environment() or resources.read_budget(settings.resource_budget)
    store, plan, specification, needs_archive = prepare_run(settings)
    producer_software = software.capture("training", settings.mode)
    samples = [sample for sample in plan["samples"] if sample["split"] == settings.split]
    resolved = {
        **settings.as_dict(),
        "membership_sha256": store.membership["sha256"],
        "frame_plan_sha256": plan["sha256"],
        "samples": len(samples),
        "loss": FULL_FRAME_X0_MSE,
        "queue_job_sha256": events.job,
        "queue_attempt_token": events.token,
        "producer_source_sha256": sha256(Path(__file__)),
        "software": producer_software,
        "queue_launch": launch,
        "queue_launch_path": None if launch_path is None else str(launch_path),
        "resource_budget": budget,
    }
    if settings.dry_run:
        print(json.dumps(resolved, indent=2, default=str))  # noqa: T201 -- requested CLI plan
        return 0
    applied_numerics = numerics.apply(environment_required=launch is not None)
    if settings.reserve_gpu_gib > 0:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        reserve = torch.empty(int(settings.reserve_gpu_gib * 2**30), dtype=torch.uint8, device=f"cuda:{local}")
        del reserve
    with timed("Accelerator() / process group"):
        accelerator = Accelerator()
    settings.world_size = accelerator.num_processes
    device = accelerator.device
    rank = accelerator.process_index
    if launch is not None:
        runtime.check_accelerator(accelerator, launch["job"]["processes"], _launch_precision(launch),
                                  distributed_type=DistributedType.FSDP)
    _check_launch_current(launch, launch_path)
    measurements = []
    def preserve_measurement(record: dict) -> None:
        measurements.append(record)
        LOGGER.info("resource | %s", json.dumps(record, sort_keys=True))
        if settings.output.is_dir() and not needs_archive:
            dataset.atomic_write(settings.output / f"resources_rank{rank}.jsonl",
                                 lambda temporary: temporary.write_text(
                                     "".join(json.dumps(item) + "\n" for item in measurements)))
        if record["state"] != "passed":
            raise ValueError(record["error"])

    @contextlib.contextmanager
    def measured(phase: str) -> Iterator[None]:
        if budget is None:
            yield
            return
        observation = resources.Phase(device, phase, rank, budget)
        try:
            observation.start()
            yield
        except BaseException as error:
            if observation.started is None:
                raise
            record = observation.finish(error=f"{type(error).__name__}: {error}")
            measurements.append(record)
            LOGGER.error("resource | %s", json.dumps(record, sort_keys=True))
            if settings.output.is_dir() and not needs_archive:
                dataset.atomic_write(settings.output / f"resources_rank{rank}.jsonl",
                                     lambda temporary: temporary.write_text(
                                         "".join(json.dumps(item) + "\n" for item in measurements)))
            raise
        else:
            preserve_measurement(observation.finish())
    with measured("load"):
        software.check_current(producer_software)
        with timed("prompt cache (text encoder)"):
            context = prompt_cache.get_or_build(specification, DEFAULT_PROMPT, DTYPE, device)
        with timed("transformer load + LoRA injection"):
            software.check_current(producer_software)
            transformer = build_transformer(specification, settings, accelerator)
        trainable = [p for p in transformer.parameters() if p.requires_grad]
        lora_modules = sorted(
            {name.rsplit(".lora_", 1)[0] for name, p in transformer.named_parameters() if p.requires_grad}
        )
        target_counts = {
            target: sum(name.endswith(target) for name in lora_modules)
                for target in adapters.LORA_TARGETS[settings.lora_target]
        }
        trainable_count = sum(p.numel() for p in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=settings.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
        with timed("accelerator.prepare (FSDP shard)"):
            transformer, optimizer = accelerator.prepare(transformer, optimizer)
    applied_runtime = runtime.gather(runtime.capture(transformer, accelerator, common.SIGMA_PRECISION), accelerator)
    runtime.validate(applied_runtime, settings.world_size, accelerator.mixed_precision,
                     numerical_policy=applied_numerics)
    if launch is not None:
        runtime.validate(applied_runtime, settings.world_size, _launch_precision(launch), native=True)
    resolved["runtime"] = applied_runtime
    trace = None
    if settings.consumer_trace:
        from scripts.onestep_avatar.training.consumer_trace import Trace  # noqa: PLC0415 -- selected diagnostic
        trace = lifecycle.enter_context(Trace(transformer, {"rank": rank, "world_size": settings.world_size,
                                   "queue_job_sha256": events.job, "queue_attempt_token": events.token,
                                   "launch_sha256": None if launch_path is None else sha256(launch_path)},
                                   failure_path=settings.output / f"consumer_trace_failed_rank{rank}.json"))
    world = settings.world_size
    tiling = max(1, math.ceil(world * settings.chains_per_rank / len(samples)))
    per_rank = len(samples) * tiling // world
    per_rank -= per_rank % settings.chains_per_rank
    if per_rank == 0:
        raise ValueError("sample plan cannot supply the requested ranks and accumulation")
    order = list(range(len(samples))) * tiling
    software.check_current(producer_software)
    _check_launch_current(launch, launch_path)
    resources.check_budget(budget)
    if needs_archive:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            archive_existing_run(settings.output)
        accelerator.wait_for_everyone()
        needs_archive = False
    if accelerator.is_main_process:
        settings.output.mkdir(parents=True, exist_ok=True)
        software.check_current(producer_software)
        _check_launch_current(launch, launch_path)
        resources.check_budget(budget)
        if settings.save_update_state:
            text_path = settings.output / "update_states" / "text.pt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_cpu = context.detach().cpu().contiguous()
            dataset.atomic_write(text_path, lambda temporary: torch.save(text_cpu, temporary))
            text_digest = hashlib.sha256(text_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            resolved["update_text"] = {"path": str(text_path.resolve()), "sha256": sha256(text_path),
                                       "tensor_sha256": text_digest,
                                       "shape": list(text_cpu.shape), "dtype": str(text_cpu.dtype)}
        resolved.update(
            world_size=world,
            trainable_params=trainable_count,
            lora_modules=len(lora_modules),
            lora_target_counts=target_counts,
            samples_per_update=world * settings.chains_per_rank,
            sample_tiling=tiling,
            optimizer={"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        )
        dataset.atomic_write(
            settings.output / "config.json",
            lambda temporary: temporary.write_text(json.dumps(resolved, indent=2, default=str) + "\n"),
        )
        dataset.atomic_write(
            settings.output / "frame_plan.json",
            lambda temporary: temporary.write_text(json.dumps(plan, indent=2) + "\n"),
        )
    accelerator.wait_for_everyone()
    if budget is not None:
        dataset.atomic_write(settings.output / f"resources_rank{rank}.jsonl",
                             lambda temporary: temporary.write_text(
                                 "".join(json.dumps(item) + "\n" for item in measurements)))
    accelerator.wait_for_everyone()
    wandb_run = init_wandb(settings, config=resolved) if accelerator.is_main_process else None

    def save(step: int, *, verify_noop: bool = False) -> None:
        numerics.validate(numerics.capture(), required=True, expected=applied_numerics)
        software.check_current(producer_software)
        _check_launch_current(launch, launch_path)
        resources.check_budget(budget)
        contract = sampling.make_contract(settings, store.membership, plan, step)
        with measured(f"export:{step}"):
            path = save_lora(
                transformer,
                accelerator,
                settings.output / "checkpoints",
                step,
                {sampling.CONTRACT_KEY: json.dumps(contract, sort_keys=True)},
                verify_noop=verify_noop,
            )
            if trace is not None:
                trace.write(settings.output / "consumer_traces" / f"step_{step:05d}" / f"rank{rank}.json")
        if budget is not None:
            resources.save_snapshot(settings.output, rank, step)
        accelerator.wait_for_everyone()
        if path is not None:
            software.check_current(producer_software)
            _check_launch_current(launch, launch_path)
            resources.check_budget(budget)
            resource_evidence = {}
            consumer_evidence = (_read_consumer_evidence(settings.output, step, world, launch_path,
                                 events.job, events.token, settings.chains_per_rank)
                                 if settings.consumer_trace else {})
            if budget is not None:
                records, resource_evidence = resources.read_records(settings.output, world, step=step)
                phases = resources.training_phases(settings, step=step)
                resources.validate_records(records, world, phases, budget)
            dataset.atomic_write(
                path.with_suffix(".complete.json"),
                lambda temporary: temporary.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "step": step,
                            "path": str(path),
                            "sha256": sha256(path),
                            "state": "complete",
                            "queue_job_sha256": events.job,
                            "producer_source_sha256": resolved["producer_source_sha256"],
                            "software": producer_software,
                            "queue_launch": launch,
                            "runtime": applied_runtime,
                            "resource_budget": budget,
                            "resource_evidence": resource_evidence,
                            "consumer_trace_evidence": consumer_evidence,
                            "training_record": {
                                "config_sha256": sha256(settings.output / "config.json"),
                                "frame_plan_sha256": sha256(settings.output / "frame_plan.json"),
                            },
                        },
                        indent=2,
                    )
                    + "\n"
                ),
            )
            if settings.preview_record is not None:
                try:
                    enqueue_preview(path, settings.preview_record, settings.output)
                except Exception:  # Optional preview failures cannot interrupt distributed training.
                    LOGGER.exception("preview enqueue failed; completed checkpoint remains valid: %s", path)
        accelerator.wait_for_everyone()

    if settings.save_initial:
        save(0, verify_noop=settings.init_adapter is None)
    mode = settings.mode_settings
    geometry = (
        causal_core.CausalGeometry(specification.scale_factors, mode.block_latent_frames, mode.context_latent_frames)
        if isinstance(mode, CausalSettings)
        else None
    )
    longest = max(s["ranges"][-1][1] for s in samples)
    cache = None
    generator = torch.Generator().manual_seed(settings.data_seed)
    levels = training_sigmas(settings)
    step = 0
    started = time.time()
    events.begin_updates()
    with (settings.output / f"metrics_rank{rank}.jsonl").open("a") as log:
        while step < settings.steps:
            epoch_order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
            shard = epoch_order[rank * per_rank : (rank + 1) * per_rank]
            for begin in range(0, len(shard), settings.chains_per_rank):
                if step >= settings.steps:
                    break
                indices = shard[begin : begin + settings.chains_per_rank]
                numerics.validate(numerics.capture(), required=True, expected=applied_numerics)
                lr = settings.lr * min(1.0, (step + 1) / max(settings.warmup_steps, 1))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                sigma = sigma_for_rank(levels, rank, step, settings.noise_seed)
                with measured(f"update:{step + 1}"):
                    update_started = time.time()
                    metrics = {"loss": 0.0, "mse": 0.0}
                    details = []
                    counts = {"prime": 0, "denoise": 0, "backward": 0, "refresh": 0}
                    for slot, index in enumerate(indices):
                        sample = samples[index]
                        video = store.load(sample["source"], require_guide=settings.guide_mode == "d1")
                        grid, capture, guide, start, end = tokens_for_sample(
                            video, sample, settings, specification, device, step=step, rank=rank, slot=slot
                        )
                        expected = 1 if settings.mode == "bidirectional" else 2 * len(sample["blocks"])
                        assert_rank_lockstep(accelerator, expected, sample["source"])
                        noise_seed = training_noise_seed(settings, step=step, rank=rank, slot=slot, chain_index=index)
                        trace_scope = (contextlib.nullcontext() if trace is None else
                                       trace.sample(mode=settings.mode, step=step + 1, slot=slot, index=index))
                        with trace_scope:
                            if settings.mode == "bidirectional":
                                result = bidirectional.train_sample(
                                    transformer,
                                    context,
                                    grid,
                                    capture,
                                    guide,
                                    accelerator.backward,
                                    sigma=sigma,
                                    seed=noise_seed,
                                    guide_mode=settings.guide_mode,
                                    accumulation=len(indices),
                                )
                            else:
                                result = causal_core.train_sample(
                                    transformer,
                                    context,
                                    grid,
                                    capture,
                                    guide,
                                    geometry,
                                    sample["blocks"],
                                    accelerator.backward,
                                    sigma=sigma,
                                    seed=noise_seed,
                                    cache=cache,
                                    guide_mode=settings.guide_mode,
                                    teacher_forcing=mode.teacher_forcing,
                                    accumulation=len(indices),
                                    timing=settings.timing,
                                    capacity_latent_frames=longest,
                                )
                                cache = result.pop("cache")
                        for key in metrics:
                            metrics[key] += result[key] / len(indices)
                        for key in counts:
                            counts[key] += result.get(key + "_calls", 0)
                        details.append(
                            {
                                "source": sample["source"],
                                "start": start,
                                "end": end,
                                "ranges": [[start, end]]
                                if settings.mode == "bidirectional" or mode.start_policy == "random"
                                else sample["ranges"],
                                "noise_seed": noise_seed,
                                "per_block": result.get("per_block", []),
                            }
                        )
                    grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), settings.max_grad_norm)
                    optimizer.step()
                    if settings.save_update_state:
                        from scripts.onestep_avatar.training.update_state import save_adam_state  # noqa: PLC0415
                        software.check_current(producer_software)
                        state_path = settings.output / "update_states" / f"step_{step + 1:05d}.pt"
                        shapes = save_adam_state(transformer, optimizer, accelerator, state_path, step + 1)
                        if shapes is not None:
                            software.check_current(producer_software)
                            state_json = json.dumps({"step": step + 1, "sha256": sha256(state_path), "shapes": shapes,
                                                     "grad_norm": float(grad_norm), "world_size": world,
                                                     "accumulation": settings.chains_per_rank,
                                                     "optimizer": resolved["optimizer"],
                                                     "software": producer_software}, indent=2) + "\n"
                            dataset.atomic_write(state_path.with_suffix(".json"),
                                                 lambda temporary, payload=state_json: temporary.write_text(payload))
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {
                    "schema_version": 2,
                    "step": step,
                    "rank": rank,
                    "mode": settings.mode,
                    "lr": lr,
                    "sigma0": sigma,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "elapsed_s": time.time() - started,
                    "update_s": time.time() - update_started,
                    **metrics,
                    "samples": details,
                    "call_counts": counts,
                }
                if settings.save_update_state:
                    record["text_tensor_sha256"] = hashlib.sha256(
                        context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step % settings.log_every == 0 and wandb_is_enabled(settings):
                    mean = rank_mean(
                        accelerator,
                        [metrics["loss"], metrics["mse"], float(grad_norm) if grad_norm is not None else 0.0],
                    )
                    if accelerator.is_main_process and wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": mean[0],
                                "train/mse": mean[1],
                                "train/grad_norm": mean[2],
                                "train/lr": lr,
                                "train/sigma0": sigma,
                                "train/elapsed_s": record["elapsed_s"],
                            },
                            step=step,
                        )
                if step % settings.save_every == 0 or step == settings.steps or (settings.save_initial and step == 1):
                    save(step)
    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()
    accelerator.wait_for_everyone()
    return 0
