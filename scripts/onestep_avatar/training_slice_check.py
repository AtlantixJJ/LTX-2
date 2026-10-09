"""Localize one native update; see doc/training_slice_check.md."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from safetensors.torch import load_file

from scripts.onestep_avatar import evaluate, training_update_check
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import queue, software
from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import bidirectional, causal, common
from scripts.onestep_avatar.training import checkpoints, config, engine, resources, runtime, update_state

ENTRY = "scripts/onestep_avatar/training_slice_check.py"
SUPPORT_CHANGES = {
    "scripts/onestep_avatar/execution/supervision.py", "scripts/onestep_avatar/training/consumer_trace.py",
    "scripts/onestep_avatar/training/checkpoints.py",
}


def _original_context(settings: config.RunSettings, saved: dict) -> torch.Tensor:
    """Read the original context only after checking its producer's exact byte facts."""
    path = settings.output / "update_states/text.pt"
    record = saved.get("update_text", {})
    if (record.get("path") != str(path.resolve()) or record.get("sha256") != sha256(path)):
        raise ValueError("original saved text file differs from its producer")
    context = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(context, torch.Tensor) or not torch.isfinite(context).all()
            or record.get("tensor_sha256") != hashlib.sha256(
                context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            or record.get("shape") != list(context.shape) or record.get("dtype") != str(context.dtype)):
        raise ValueError("original saved text tensor differs from its producer")
    return context


def _original_artifacts(  # noqa: PLR0912 -- sequential original artifact and producer gates
    original: dict, settings: config.RunSettings, saved: dict, store: dataset.ClipStore, plan: dict, budget: dict
) -> list[Path]:
    """Check original scientific and producer bindings without restamping old source profiles."""
    root = settings.output
    config_path, plan_path = root / "config.json", root / "frame_plan.json"
    if (saved.get("base_identity") != settings.base_identity or json.loads(plan_path.read_text()) != plan
            or saved.get("membership_sha256") != store.membership["sha256"]
            or saved.get("frame_plan_sha256") != plan["sha256"]):
        raise ValueError("original frame plan, membership or base identity differs")
    producer = saved.get("producer_source_sha256")
    if producer != saved["software"]["sources"].get("scripts/onestep_avatar/training/engine.py"):
        raise ValueError("original training producer differs from its source profile")
    identities = {"queue_job_sha256": original["sha256"], "producer_source_sha256": producer,
                  "software": saved["software"], "queue_launch": saved["queue_launch"],
                  "runtime": saved["runtime"], "resource_budget": budget,
                  "training_record": {"config_sha256": sha256(config_path), "frame_plan_sha256": sha256(plan_path)}}
    files = [config_path, plan_path, Path(original["accelerate_config"])]
    for step in (0, 1):
        checkpoint = root / "checkpoints" / f"lora_weights_step_{step:05d}.safetensors"
        marker = queue.read_training_marker(checkpoint, step)
        contract = checkpoints.read_contract(checkpoint)
        checkpoints.validate_adapter_tensors(checkpoint, contract)
        expected = checkpoints.make_contract(settings, store.membership, plan, step)
        expected["adapter"]["tensor_shapes"] = contract["adapter"]["tensor_shapes"]
        if contract != expected or any(marker.get(key) != value for key, value in identities.items()):
            raise ValueError("original checkpoint contract or producer binding differs")
        if step == 0:
            checkpoints.assert_exported_lora_is_noop(load_file(checkpoint))
        for field in ("resource_evidence", "consumer_trace_evidence"):
            for path, digest in marker.get(field, {}).items():
                evidence = Path(path)
                if not evidence.resolve().is_relative_to(root.resolve()) or sha256(evidence) != digest:
                    raise ValueError("original checkpoint evidence differs")
                files.append(evidence)
        files.extend((checkpoint, checkpoint.with_suffix(".complete.json")))
    launch_path = saved.get("queue_launch_path")
    if launch_path is not None:
        launch_path = Path(launch_path)
        if queue.read_training_launch(launch_path) != saved["queue_launch"]:
            raise ValueError("original dispatch file differs from saved launch")
        files.append(launch_path)
    samples = [sample for sample in plan["samples"] if sample["split"] == settings.split]
    visits = training_update_check.first_update_visits(settings, samples, original["processes"])
    logs = []
    for rank in range(original["processes"]):
        path = root / f"metrics_rank{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 1:
            raise ValueError("original slice requires exactly one completed update per rank")
        logs.append(rows[0])
        files.append(path)
    training_update_check.check_visits(visits, logs, settings)
    for sample in samples:
        directory = store.root / sample["source"]
        files.append(directory / dataset.capture_bundle_name(settings.objective))
        if settings.guide_mode == "d1":
            files.append(directory / dataset.guide_bundle_name(settings.objective))
    return files


def prepare(job_path: Path, rank: int, slots: int) -> tuple:
    """Bind original inputs while keeping one-rank diagnostic scope explicit."""
    original = queue.prepare_job(json.loads(job_path.read_text()), job_path.resolve().parent)
    settings = config.parse_settings(original["arguments"])
    if (settings.steps != 1 or not settings.save_initial or not settings.save_update_state
            or settings.init_adapter is not None or settings.preview_inputs is not None
            or settings.no_gradient_checkpointing or settings.mode_settings.start_policy != "clip_start"):
        raise ValueError("original slice requires fresh checkpointed one-update evidence with a zero export")
    saved = json.loads((settings.output / "config.json").read_text())
    queue.verify_training_launch(saved["queue_launch"], original)
    runtime.validate(saved["runtime"], original["processes"], "bf16", native=True)
    marker = queue.read_training_marker(settings.output / "checkpoints/lora_weights_step_00001.safetensors", 1)
    if marker["queue_job_sha256"] != original["sha256"]:
        raise ValueError("original completed update belongs to a different job")
    software.validate(saved["software"])
    current = software.capture("training", settings.mode)
    changed = {name for name, digest in saved["software"]["sources"].items()
               if current["sources"].get(name) != digest}
    if changed - SUPPORT_CHANGES or current["runtime"] != saved["software"]["runtime"]:
        raise ValueError("original computation owners or runtime changed")
    budget = resources.read_budget(settings.resource_budget)
    if budget != saved["resource_budget"] or budget["tolerance"] != training_update_check.TOLERANCE:
        raise ValueError("original resource limits or tolerance changed")
    store, plan, spec, _used = engine.prepare_run(settings, require_fresh_output=False)
    settings.world_size = original["processes"]
    files = _original_artifacts(original, settings, saved, store, plan, budget)
    samples = [sample for sample in plan["samples"] if sample["split"] == settings.split]
    visits = [row for row in training_update_check.first_update_visits(settings, samples, original["processes"])
              if row["rank"] == rank][:slots]
    if len(visits) != slots:
        raise ValueError("requested original visit coverage is unavailable")
    text = settings.output / "update_states/text.pt"
    context = _original_context(settings, saved)
    pinned = {str(path.resolve()): sha256(path) for path in
              (job_path, settings.subset, text, settings.resource_budget, *files)}
    pinned[str(Path(spec.paths.transformer()).resolve())] = settings.base_identity["base_transformer_sha256"]
    return original, settings, store, plan, spec, visits, context, budget, pinned, changed


def execute(args: argparse.Namespace) -> dict:  # noqa: PLR0915 -- one bounded diagnostic update
    if args.output.exists():
        raise ValueError("native slice requires a fresh output")
    original, settings, store, plan, spec, visits, text, budget, pinned, changed = prepare(
        args.job, args.rank_key, args.slots)
    if args.deterministic:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('deterministic control requires the declared cuBLAS workspace at launch')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    accelerator = Accelerator(mixed_precision="bf16")
    expected = DistributedType.FSDP if args.execution == "fsdp" else DistributedType.NO
    runtime.check_accelerator(accelerator, 1, "bf16", distributed_type=expected)
    producer = software.capture("training", settings.mode, extra_sources=(
        ENTRY, "scripts/onestep_avatar/training_update_check.py", "scripts/onestep_avatar/evaluate.py"))
    device = accelerator.device
    if device.type != "cuda":
        raise ValueError("native slice requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"scope": "one native rank; not the original four-rank acceptance", "execution": args.execution,
                "original_job": original, "original_rank_key": args.rank_key, "selected_slots": args.slots,
                "visits": visits, "world_size": 1, "averaging_denominator": args.slots,
                "budget": budget, "tolerance": training_update_check.TOLERANCE, "input_sha256": pinned,
                "original_support_source_changes": sorted(changed), "software": producer,
                "kernel_control": {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                                   'cudnn_deterministic': torch.backends.cudnn.deterministic,
                                   'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                   'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                   'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    measurements = []
    phase = training_update_check.measured_phase
    with phase(args.output, device, "load", budget, measurements):
        transformer = engine.build_transformer(spec, settings, accelerator)
        initial_values = {name: evaluate.tensor_sha256(parameter) for name, parameter in transformer.named_parameters()
                          if ".lora_" in name}
        (args.output / "initial_fp32.json").write_text(json.dumps(initial_values, indent=2) + "\n")
        parameters = [p for p in transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=settings.lr / max(settings.warmup_steps, 1),
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        transformer, optimizer = accelerator.prepare(transformer, optimizer)
        applied = runtime.gather(runtime.capture(transformer, accelerator, common.SIGMA_PRECISION), accelerator)
        runtime.validate(applied, 1, "bf16", native=args.execution == "fsdp")
        metadata = {
            checkpoints.CONTRACT_KEY: json.dumps(checkpoints.make_contract(settings, store.membership, plan, 0))}
        zero = checkpoints.save_lora(transformer, accelerator, args.output / "checkpoints", 0, metadata,
                                     verify_noop=True)
        native_zero = load_file(settings.output / "checkpoints/lora_weights_step_00000.safetensors")
        actual_zero = load_file(zero)
        if (set(native_zero) != set(actual_zero)
                or any(not torch.equal(native_zero[k], actual_zero[k]) for k in native_zero)):
            raise ValueError("native slice zero export differs from original initialization")
    from scripts.onestep_avatar.training.consumer_trace import Trace  # noqa: PLC0415 -- diagnostic only

    cache = None
    geometry = None if settings.mode == "bidirectional" else causal.CausalGeometry(
        spec.scale_factors, settings.mode_settings.block_latent_frames, settings.mode_settings.context_latent_frames)
    longest = max(sample["ranges"][-1][1] for sample in plan["samples"] if sample["split"] == settings.split)
    losses = []
    with contextlib.ExitStack() as lifecycle:
        trace = lifecycle.enter_context(Trace(transformer, {"rank": 0, "world_size": 1,
            "queue_job_sha256": os.environ.get(JOB_ENV), "queue_attempt_token": os.environ.get(TOKEN_ENV),
            "launch_sha256": sha256(args.output / "protocol.json")}, max_events=131072,
            failure_path=args.output / "consumer_trace_failed.json"))
        with phase(args.output, device, "update", budget, measurements):
            text = text.to(device)
            for visit in visits:
                sample = visit["sample"]
                video = store.load(sample["source"], require_guide=settings.guide_mode == "d1")
                grid, capture, guide, _, _ = engine.tokens_for_sample(
                    video, sample, settings, spec, device, step=0, rank=visit["rank"], slot=visit["slot"])
                options = {"sigma": visit["sigma"], "seed": visit["noise_seed"],
                           "guide_mode": settings.guide_mode, "accumulation": args.slots}
                with trace.sample(mode=settings.mode, step=1, slot=visit["slot"], index=visit["index"]):
                    if settings.mode == "bidirectional":
                        result = bidirectional.train_sample(transformer, text, grid, capture, guide,
                                                            accelerator.backward, **options)
                    else:
                        result = causal.train_sample(transformer, text, grid, capture, guide, geometry,
                            sample["blocks"], accelerator.backward, cache=cache,
                            teacher_forcing=settings.mode_settings.teacher_forcing,
                            capacity_latent_frames=longest, **options)
                        cache = result.pop("cache")
                losses.append(result["loss"])
            norm = float(accelerator.clip_grad_norm_(transformer.parameters(), settings.max_grad_norm))
            optimizer.step()
        with phase(args.output, device, "export", budget, measurements):
            state = update_state.collect_adam_state(transformer, optimizer, accelerator, 1)
            dataset.atomic_write(args.output / "adam.pt", lambda target: torch.save(state, target))
            metadata = {
                checkpoints.CONTRACT_KEY: json.dumps(checkpoints.make_contract(settings, store.membership, plan, 1))}
            final = checkpoints.save_lora(transformer, accelerator, args.output / "checkpoints", 1, metadata)
            evidence = trace.write(args.output / "consumer_trace.json")
        resources.validate_records(measurements, 1, ["load", "update", "export"], budget)
        software.check_current(producer)
        if any(sha256(Path(path)) != digest for path, digest in pinned.items()):
            raise ValueError("original slice inputs changed")
        store.verify(require_guide=settings.guide_mode == "d1")
        result = {"state": "complete", "execution": args.execution, "protocol": protocol,
                  "runtime": applied, "initial_fp32": initial_values, "loss": sum(losses) / len(losses),
                  "grad_norm": norm, "resources": measurements, "consumer_trace_evidence": evidence,
                  "checkpoint": str(final.resolve()), "checkpoint_sha256": sha256(final),
                  "output_files": {str(path.relative_to(args.output)): sha256(path)
                                   for path in args.output.rglob('*') if path.is_file()}}
        dataset.atomic_write(args.output / "result.json",
                             lambda target: target.write_text(json.dumps(result, indent=2) + "\n"))
    return result


def _saved_record(directory: Path) -> tuple[dict, Path]:
    """Bind comparison to complete local artifacts, before reading moment/adapter tensors."""
    directory = directory.resolve()
    record = json.loads((directory / "result.json").read_text())
    inventory = record.get("output_files")
    required = {"protocol.json", "initial_fp32.json", "resources_rank0.jsonl", "adam.pt", "consumer_trace.json",
                "checkpoints/lora_weights_step_00000.safetensors", "checkpoints/lora_weights_step_00001.safetensors"}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and path != directory / "result.json"}
    if (record.get("state") != "complete" or not isinstance(inventory, dict)
            or not required.issubset(inventory) or set(inventory) != actual):
        raise ValueError("saved slice output inventory is incomplete or changed")
    for name, digest in inventory.items():
        relative = Path(name)
        path = directory / relative
        if (relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
                or not path.resolve().is_relative_to(directory) or sha256(path) != digest):
            raise ValueError("saved slice artifact escapes its output or changed")
    protocol = json.loads((directory / "protocol.json").read_text())
    initial = json.loads((directory / "initial_fp32.json").read_text())
    measurements, _evidence = resources.read_records(directory, 1)
    if (record.get("protocol") != protocol or record.get("initial_fp32") != initial
            or record.get("resources") != measurements):
        raise ValueError("saved slice embedded records differ from their actual files")
    execution = protocol.get("execution")
    if (execution not in {"fsdp", "serial"} or record.get("execution") != execution
            or protocol.get("world_size") != 1
            or protocol.get("selected_slots") not in (1, 2)
            or protocol.get("averaging_denominator") != protocol["selected_slots"]
            or protocol.get("tolerance") != training_update_check.TOLERANCE):
        raise ValueError("saved slice execution or numerical protocol differs")
    software.validate(protocol["software"])
    runtime.validate(record["runtime"], 1, "bf16", native=execution == "fsdp")
    resources.validate_records(measurements, 1, ["load", "update", "export"], protocol["budget"])
    checkpoint = directory / "checkpoints/lora_weights_step_00001.safetensors"
    if (Path(record["checkpoint"]).resolve() != checkpoint or record.get("checkpoint_sha256") != sha256(checkpoint)
            or record["checkpoint_sha256"] != inventory[str(checkpoint.relative_to(directory))]):
        raise ValueError("saved slice checkpoint path or hash differs from its local export")
    trace_path = directory / "consumer_trace.json"
    trace = json.loads(trace_path.read_text())
    trace_evidence = record.get("consumer_trace_evidence", {})
    if (Path(trace_evidence.get("path", "")).resolve() != trace_path
            or trace_evidence.get("sha256") != inventory["consumer_trace.json"]
            or trace_evidence.get("events") != len(trace.get("events", []))
            or trace_evidence.get("complete") is not True or trace.get("complete") is not True
            or trace.get("binding", {}).get("launch_sha256") != inventory["protocol.json"]):
        raise ValueError("saved slice consumer trace differs from its local evidence")
    return record, checkpoint


def compare(left: Path, right: Path, output: Path) -> dict:
    """Compare saved diagnostic arms without loading models or changing tolerances."""
    if output.exists():
        raise ValueError("slice comparison requires a fresh output file")
    (a, left_checkpoint), (b, right_checkpoint) = [_saved_record(path) for path in (left, right)]
    for key in ("original_job", "original_rank_key", "selected_slots", "visits", "budget", "tolerance", "input_sha256",
                "kernel_control", "software"):
        if a["protocol"][key] != b["protocol"][key]:
            raise ValueError("saved slice scientific settings differ")
    if a["initial_fp32"] != b["initial_fp32"]:
        raise ValueError("saved slice fp32 adapter initialization differs")
    result = training_update_check.compare_update(
        torch.load(left / "adam.pt", weights_only=True), torch.load(right / "adam.pt", weights_only=True),
        load_file(left_checkpoint), load_file(right_checkpoint), beta1=0.9,
        norms=(a["grad_norm"], b["grad_norm"]), losses=(a["loss"], b["loss"]))
    output.write_text(json.dumps({"scope": "saved native slice only", "comparison": result,
        "inputs": {str(path.resolve()): sha256(path / "result.json") for path in (left, right)}}, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=("fsdp", "serial"))
    parser.add_argument("--rank-key", type=int, default=0, choices=range(4))
    parser.add_argument("--slots", type=int, default=1, choices=(1, 2))
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--deterministic", action='store_true')
    args = parser.parse_args(argv)
    if args.compare:
        result = compare(*args.compare, args.output)
        return 0 if result["passed"] else 2
    if args.job is None or args.execution is None:
        parser.error("execution requires --job and --execution")
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
