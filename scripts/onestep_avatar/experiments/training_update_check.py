"""Bounded first-update serial reference; see doc/experiments/training_update_check.md."""
# CLI environment bootstrap must precede native imports.

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from scripts.onestep_avatar.training import numerics

if __name__ == "__main__":
    numerics.configure_environment()

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedType
from safetensors.torch import load_file

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import queue, software
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, bidirectional, causal, common
from scripts.onestep_avatar.training import checkpoints, config, engine, resources, runtime, update_state

TOLERANCE = {"relative_l2": 0.02, "near_zero_rms": 1e-8, "absolute_rms": 1e-8}
ENTRY = "scripts/onestep_avatar/experiments/training_update_check.py"
EXTRA_SOURCES = (ENTRY, 'scripts/onestep_avatar/experiments/__init__.py', spec_protocol.ENTRY)


def check_launch(job_path: Path, world: int) -> tuple[dict, dict, str]:
    """Check actual original queue launch before opening scientific inputs or models."""
    job = queue.prepare_job(json.loads(job_path.read_text()), job_path.resolve().parent)
    settings = config.parse_settings(job["arguments"])
    saved = json.loads((settings.output / "config.json").read_text())
    checkpoint = settings.output / "checkpoints/lora_weights_step_00001.safetensors"
    marker = queue.read_training_marker(checkpoint, 1)
    if (
        type(world) is not int
        or world != job["processes"]
        or saved.get("world_size") != world
        or saved.get("queue_job_sha256") != job["sha256"]
        or marker.get("queue_job_sha256") != job["sha256"]
    ):
        raise ValueError("replay world or reconstructed queue launch identity differs from the original run")
    original = saved.get("queue_launch")
    queue.verify_training_launch(original, job)
    if original["schema_version"] != 2:
        raise ValueError("current native replay requires schema-two numerical launch evidence")
    if marker.get("queue_launch") != original:
        raise ValueError("completed checkpoint original launch binding differs")
    precision = yaml.safe_load(base64.b64decode(original["accelerate_config_bytes_base64"]))["mixed_precision"]
    if precision not in ("no", "bf16"):
        raise ValueError("reference requires explicit no/bf16 original launch precision")
    runtime.validate(saved.get("runtime"), world, precision, native=True, numerical_policy=True)
    if any(rank["conditioning_precision"] != common.SIGMA_PRECISION for rank in saved["runtime"]["ranks"]):
        raise ValueError("applied conditioning precision differs from the shared intended policy")
    if marker.get("runtime") != saved["runtime"]:
        raise ValueError("completed checkpoint applied runtime evidence differs")
    budget = resources.read_budget(settings.resource_budget)
    if budget is None or saved.get("resource_budget") != budget or marker.get("resource_budget") != budget:
        raise ValueError("current native replay requires the original unchanged resource budget")
    if budget.get("tolerance") != TOLERANCE:
        raise ValueError("original frozen resource budget tolerance differs from the E4 protocol")
    measurements, _journal_evidence = resources.read_records(settings.output, world)
    snapshot, evidence = resources.read_records(settings.output, world, step=1)
    if measurements != snapshot:
        raise ValueError("original final resource snapshot differs from rank journals")
    resources.validate_records(measurements, world, ["load", "export:0", "update:1", "export:1"], budget)
    if marker.get("resource_evidence") != evidence:
        raise ValueError("completed checkpoint allocated-memory evidence differs")
    return job, saved, precision


@contextmanager
def measured_phase(
    output: Path, device: torch.device, name: str, budget: dict | None, records: list[dict]
) -> Iterator[None]:
    """Preserve measured phase failures before refusing further native acceptance."""
    phase = resources.Phase(device, name, 0, budget)
    error = None
    try:
        phase.start()
        yield
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        if phase.started is not None:
            record = phase.finish(error)
            records.append(record)
            path = output / "resources_rank0.jsonl"
            with path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if error is None and record["state"] != "passed":
                raise ValueError(record["error"])


def gap(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Compare complete finite arrays, with an explicit near-zero denominator rule."""
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("update comparison requires matching finite arrays")
    delta = actual.double() - expected.double()
    rms = float(delta.square().mean().sqrt())
    reference = float(expected.double().square().mean().sqrt())
    near_zero = reference < TOLERANCE["near_zero_rms"]
    relative = None if near_zero else rms / reference
    return {
        "rms": rms,
        "maximum": float(delta.abs().max()),
        "reference_rms": reference,
        "near_zero": near_zero,
        "relative_l2": relative,
        "passed": rms <= TOLERANCE["absolute_rms"] if near_zero else relative <= TOLERANCE["relative_l2"],
    }


def compare_update(
    distributed: dict,
    serial: dict,
    distributed_weights: dict,
    serial_weights: dict,
    *,
    beta1: float,
    norms: tuple[float, float],
    losses: tuple[float, float],
) -> dict:
    """Assess actual first moments and exported correction, not only Adam weight signs."""
    inventories = [set(value) for value in (distributed, serial, distributed_weights, serial_weights)]
    if any(names != inventories[0] for names in inventories) or not inventories[0]:
        raise ValueError("update/adapter parameter inventories differ")
    gradients, weights, second_moments = {}, {}, {}
    for name in sorted(distributed):
        left, right = distributed[name], serial[name]
        if left["step"] != 1 or right["step"] != 1 or not 0 < beta1 < 1:
            raise ValueError("first-update reference requires Adam step one")
        gradients[name] = gap(left["exp_avg"] / (1 - beta1), right["exp_avg"] / (1 - beta1))
        second_moments[name] = gap(left["exp_avg_sq"], right["exp_avg_sq"])
        # B is zero initially, so its value is the actual first-update correction.
        weights[name] = gap(distributed_weights[name], serial_weights[name])
    result = {
        "second_moments": second_moments,
        "clipped_gradients": gradients,
        "exported_matrices": weights,
        "gradient_norm": gap(torch.tensor(norms[0]), torch.tensor(norms[1])),
        "loss": gap(torch.tensor(losses[0]), torch.tensor(losses[1])),
    }
    result["passed"] = all(x["passed"] for group in (gradients, weights, second_moments) for x in group.values())
    result["passed"] &= result["gradient_norm"]["passed"] and result["loss"]["passed"]
    if not any(torch.count_nonzero(value) for name, value in distributed_weights.items() if ".lora_B." in name):
        raise ValueError("distributed step-one B matrices did not change")
    return result


def first_update_visits(settings: config.RunSettings, samples: list[dict], world: int) -> list[dict]:
    """Independently reconstruct the trainer's first seeded rank/accumulation group."""
    if world < 1 or not samples or settings.chains_per_rank < 2:
        raise ValueError("reference requires positive world and accumulation greater than one")
    tiling = max(1, math.ceil(world * settings.chains_per_rank / len(samples)))
    order = list(range(len(samples))) * tiling
    per_rank = len(order) // world
    per_rank -= per_rank % settings.chains_per_rank
    generator = torch.Generator().manual_seed(settings.data_seed)
    permutation = torch.randperm(len(order), generator=generator).tolist()
    shuffled = [order[index] for index in permutation]
    visits = []
    for rank in range(world):
        indices = shuffled[rank * per_rank : rank * per_rank + settings.chains_per_rank]
        for slot, index in enumerate(indices):
            visits.append(
                {
                    "rank": rank,
                    "slot": slot,
                    "index": index,
                    "sample": samples[index],
                    "sigma": config.sigma_for_rank(config.training_sigmas(settings), rank, 0, settings.noise_seed),
                    "noise_seed": config.training_noise_seed(settings, step=0, rank=rank, slot=slot, chain_index=index),
                }
            )
    return visits


def check_visits(visits: list[dict], logs: list[dict], settings: config.RunSettings) -> None:
    """Fail changed visits before replay; logs cannot silently choose the reference inputs."""
    for rank, log in enumerate(logs):
        expected = [visit for visit in visits if visit["rank"] == rank]
        if log.get("rank") != rank or log.get("step") != 1 or len(log["samples"]) != len(expected):
            raise ValueError("rank update log has different coverage")
        for visit, saved in zip(expected, log["samples"], strict=True):
            sample = visit["sample"]
            ranges = sample["ranges"]
            if (
                saved["source"] != sample["source"]
                or saved["ranges"] != ranges
                or saved["noise_seed"] != visit["noise_seed"]
                or log["sigma0"] != visit["sigma"]
            ):
                raise ValueError("rank source/ranges/noise/sigma differs from fixed first-update visit")
        count = settings.chains_per_rank
        k = settings.mode_settings.blocks_per_sample if settings.mode == "causal" else 1
        counts = {
            "prime": count if settings.mode == "causal" else 0,
            "denoise": count * k,
            "backward": count * k,
            "refresh": count * (k - 1) if settings.mode == "causal" else 0,
        }
        if log["call_counts"] != counts:
            raise ValueError("rank explicit forward/backward/refresh counts differ")


def prepare(job_path: Path, world: int) -> tuple:  # noqa: PLR0912, PLR0915 -- original scientific evidence gates
    """Verify original distributed evidence and scientific inputs without model loading."""
    job = json.loads(job_path.read_text())
    settings = config.parse_settings(job["arguments"])
    if (
        settings.steps != 1
        or not settings.save_initial
        or not settings.save_update_state
        or settings.init_adapter is not None
        or settings.preview_inputs is not None
        or settings.no_gradient_checkpointing
        or settings.mode_settings.start_policy != "clip_start"
    ):
        raise ValueError(
            "reference requires fresh one-update, zero export, Adam evidence and checkpointing, without previews"
        )
    job, saved, _precision = check_launch(job_path, world)
    settings = config.parse_settings(job["arguments"])
    accelerate_config = Path(job["accelerate_config"])
    store, plan, specification, _ = engine.prepare_run(settings, require_fresh_output=False)
    checkpoint = settings.output / "checkpoints/lora_weights_step_00001.safetensors"
    contract = checkpoints.read_contract(checkpoint)
    checkpoints.validate_adapter_tensors(checkpoint, contract)
    engine.verify_training_conditions(job, checkpoint, contract)
    settings.world_size = world
    initial = settings.output / "checkpoints/lora_weights_step_00000.safetensors"
    queue.read_training_marker(initial, 0)
    checkpoints.validate_adapter_tensors(initial, checkpoints.read_contract(initial))
    initial_weights = load_file(initial)
    checkpoints.assert_exported_lora_is_noop(initial_weights)
    samples = [sample for sample in plan["samples"] if sample["split"] == settings.split]
    visits = first_update_visits(settings, samples, world)
    paths = [settings.output / f"metrics_rank{rank}.jsonl" for rank in range(world)]
    rows = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    if any(len(row) != 1 for row in rows):
        raise ValueError("reference requires exactly one completed update per rank")
    logs = [row[0] for row in rows]
    check_visits(visits, logs, settings)
    state_path = settings.output / "update_states/step_00001.pt"
    state_record = json.loads(state_path.with_suffix(".json").read_text())
    software.check_current(state_record["software"])
    if (
        state_record["sha256"] != sha256(state_path)
        or state_record["step"] != 1
        or state_record["world_size"] != world
        or state_record["accumulation"] != settings.chains_per_rank
        or state_record["optimizer"] != saved["optimizer"]
    ):
        raise ValueError("saved Adam evidence differs from the distributed update")
    states = torch.load(state_path, map_location="cpu", weights_only=True)
    if set(states) != set(initial_weights) or state_record["shapes"] != contract["adapter"]["tensor_shapes"]:
        raise ValueError("saved Adam evidence has different adapter coverage")
    for name, state in states.items():
        if (
            set(state) != {"step", "exp_avg", "exp_avg_sq"}
            or state["step"] != 1
            or any(
                value.dtype != torch.float32
                or list(value.shape) != state_record["shapes"][name]
                or not torch.isfinite(value).all()
                for value in (state["exp_avg"], state["exp_avg_sq"])
            )
            or torch.any(state["exp_avg_sq"] < 0)
        ):
            raise ValueError("saved Adam evidence is incomplete/nonfinite")
    if any(not math.isfinite(log[key]) for log in logs for key in ("loss", "grad_norm")):
        raise ValueError("distributed update has nonfinite loss/gradient norm")
    if any(log["grad_norm"] != state_record["grad_norm"] for log in logs):
        raise ValueError("distributed ranks disagree on the full preclip gradient norm")
    text = saved["update_text"]
    text_path = settings.output / "update_states/text.pt"
    context = torch.load(text_path, map_location="cpu", weights_only=True)
    tensor_digest = hashlib.sha256(context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if (
        Path(text["path"]).resolve() != text_path.resolve()
        or text["sha256"] != sha256(text_path)
        or text["tensor_sha256"] != tensor_digest
        or text["shape"] != list(context.shape)
        or text["dtype"] != str(context.dtype)
        or not torch.isfinite(context).all()
        or any(log["text_tensor_sha256"] != tensor_digest for log in logs)
    ):
        raise ValueError("distributed text tensors differ from the saved actual training context")
    files = [
        job_path,
        accelerate_config,
        initial,
        checkpoint,
        state_path,
        state_path.with_suffix(".json"),
        settings.output / "config.json",
        settings.output / "frame_plan.json",
        *paths,
        initial.with_suffix(".complete.json"),
        checkpoint.with_suffix(".complete.json"),
        text_path,
        settings.subset,
        settings.resource_budget,
    ]
    _measurements, resource_files = resources.read_records(settings.output, world)
    files.extend(Path(path) for path in resource_files)
    _snapshot, snapshot_files = resources.read_records(settings.output, world, step=1)
    files.extend(Path(path) for path in snapshot_files)
    if saved.get("queue_launch_path"):
        launch_path = Path(saved["queue_launch_path"])
        if queue.read_training_launch(launch_path) != saved["queue_launch"]:
            raise ValueError("original dispatch file differs from saved launch binding")
        files.append(launch_path)
    for sample in samples:
        directory = store.root / sample["source"]
        files.append(directory / dataset.capture_bundle_name(settings.objective))
        if settings.guide_mode == "d1":
            files.append(directory / dataset.guide_bundle_name(settings.objective))
    identities = {str(path.resolve()): sha256(path) for path in files}
    base_path = Path(specification.paths.transformer()).resolve()
    identities[str(base_path)] = settings.base_identity["base_transformer_sha256"]
    return settings, store, plan, specification, visits, logs, states, initial_weights, identities, context


def check_current(identities: dict, producer: dict, *, job_path: Path | None = None, world: int = 4) -> None:
    software.check_current(producer)
    if any(sha256(Path(path)) != value for path, value in identities.items()):
        raise ValueError("distributed reference input changed during replay")
    if job_path is not None:
        check_launch(job_path, world)


def execute(job_path: Path, output: Path, world: int, *, dry_run: bool = False, consumer_trace: bool = False) -> dict:
    with contextlib.ExitStack() as lifecycle:
        return _execute(job_path, output, world, lifecycle, dry_run=dry_run, consumer_trace=consumer_trace)


def _execute(  # noqa: PLR0915 -- bounded replay
    job_path: Path, output: Path, world: int, lifecycle: contextlib.ExitStack, *,
    dry_run: bool = False, consumer_trace: bool = False,
) -> dict:
    if output.exists():
        raise ValueError("serial reference requires a fresh output directory")
    settings, store, plan, spec, visits, logs, states, initial, identities, context = prepare(job_path, world)
    launch, saved, precision = check_launch(job_path, world)
    original_numerics = runtime.numerical_policy(saved["runtime"])
    producer = software.capture("training", settings.mode, extra_sources=EXTRA_SOURCES)
    protocol = {
        "tolerance": TOLERANCE,
        "world_size": world,
        "serial_accumulation": len(visits),
        "mode": settings.mode,
        "mixed_precision": precision,
        "queue_job_sha256": launch["sha256"],
        "queue_launch": saved["queue_launch"],
        "distributed_runtime": saved["runtime"],
        "resource_budget": saved["resource_budget"],
        "visits": visits,
        "input_files": identities,
        "software": producer,
    }
    if dry_run:
        return protocol
    numerics.apply(expected=original_numerics)
    accelerator = Accelerator(mixed_precision=precision)
    runtime.check_accelerator(accelerator, 1, precision, distributed_type=DistributedType.NO)
    check_current(identities, producer, job_path=job_path, world=world)
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    measurements = []
    budget = saved["resource_budget"]
    measured_device = accelerator.device
    if measured_device.type == "cuda" and measured_device.index is None:
        measured_device = torch.device("cuda", torch.cuda.current_device())
    with measured_phase(output, measured_device, "load", budget, measurements):
        context = context.to(accelerator.device)
        transformer = engine.build_transformer(spec, settings, accelerator)
        parameters = [p for p in transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=settings.lr / max(settings.warmup_steps, 1), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        # Save and compare initial rounded exports; replay retains fresh fp32 A.
        metadata = {
            checkpoints.CONTRACT_KEY: json.dumps(checkpoints.make_contract(settings, store.membership, plan, 0))
        }
        zero_path = checkpoints.save_lora(
            transformer, accelerator, output / "checkpoints", 0, metadata, verify_noop=True
        )
        fresh = load_file(zero_path)
        if set(fresh) != set(initial) or any(not torch.equal(fresh[name], initial[name]) for name in initial):
            raise ValueError("fresh serial fp32 initialization differs from distributed zero export")
        transformer, optimizer = accelerator.prepare(transformer, optimizer)
        serial_runtime = runtime.gather(runtime.capture(transformer, accelerator, common.SIGMA_PRECISION), accelerator)
        runtime.validate(serial_runtime, 1, precision, numerical_policy=original_numerics)
        protocol["serial_runtime"] = serial_runtime
    dataset.atomic_write(output / "protocol.json",
                         lambda path: path.write_text(json.dumps(protocol, indent=2) + "\n"))
    trace = None
    if consumer_trace or settings.consumer_trace:
        from scripts.onestep_avatar.execution.queue_protocol import (  # noqa: PLC0415 -- serial attempt identity
            TOKEN_ENV,
        )
        from scripts.onestep_avatar.training.consumer_trace import Trace  # noqa: PLC0415 -- selected diagnostic
        trace = lifecycle.enter_context(Trace(transformer, {"rank": 0, "world_size": 1,
                                   "queue_job_sha256": launch["sha256"],
                                   "queue_attempt_token": os.environ.get(TOKEN_ENV),
                                   "launch_sha256": None if not saved.get("queue_launch_path") else
                                   sha256(Path(saved["queue_launch_path"]))}, max_events=262144,
                                   failure_path=output / "consumer_trace_failed.json"))
    cache = None
    geometry = (
        causal.CausalGeometry(
            spec.scale_factors, settings.mode_settings.block_latent_frames, settings.mode_settings.context_latent_frames
        )
        if settings.mode == "causal"
        else None
    )
    longest = max(sample["ranges"][-1][1] for sample in plan["samples"] if sample["split"] == settings.split)
    losses = []
    numerics.validate(numerics.capture(), required=True, expected=original_numerics)
    with measured_phase(output, measured_device, "update", budget, measurements):
        for visit in visits:
            sample = visit["sample"]
            video = store.load(sample["source"], require_guide=settings.guide_mode == "d1")
            grid, capture, guide, _, _ = engine.tokens_for_sample(
                video, sample, settings, spec, accelerator.device, step=0, rank=visit["rank"], slot=visit["slot"]
            )
            options = {
                "sigma": visit["sigma"],
                "seed": visit["noise_seed"],
                "guide_mode": settings.guide_mode,
                "accumulation": len(visits),
            }
            trace_scope = (contextlib.nullcontext() if trace is None else
                           trace.sample(mode=settings.mode, step=1, slot=visit["slot"], index=visit["index"]))
            with trace_scope:
                if settings.mode == "bidirectional":
                    result = bidirectional.train_sample(
                        transformer, context, grid, capture, guide, accelerator.backward, **options
                    )
                else:
                    result = causal.train_sample(
                        transformer,
                        context,
                        grid,
                        capture,
                        guide,
                        geometry,
                        sample["blocks"],
                        accelerator.backward,
                        cache=cache,
                        teacher_forcing=settings.mode_settings.teacher_forcing,
                        capacity_latent_frames=longest,
                        **options,
                    )
                    cache = result.pop("cache")
            losses.append(result["loss"])
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings.max_grad_norm))
        optimizer.step()
    trace_evidence = {}
    numerics.validate(numerics.capture(), required=True, expected=original_numerics)
    with measured_phase(output, measured_device, "export", budget, measurements):
        serial = update_state.collect_adam_state(transformer, optimizer, accelerator, 1)
        dataset.atomic_write(output / "adam.pt", lambda path: torch.save(serial, path))
        metadata = {
            checkpoints.CONTRACT_KEY: json.dumps(checkpoints.make_contract(settings, store.membership, plan, 1))
        }
        final_path = checkpoints.save_lora(transformer, accelerator, output / "checkpoints", 1, metadata)
        final = load_file(final_path)
        measured = compare_update(
            states,
            serial,
            load_file(settings.output / "checkpoints/lora_weights_step_00001.safetensors"),
            final,
            beta1=0.9,
            norms=(logs[0]["grad_norm"], norm),
            losses=(sum(log["loss"] for log in logs) / world, sum(losses) / len(losses)),
        )
        # Reload the actual distributed export, not a replacement serial result.
        distributed_path = settings.output / "checkpoints/lora_weights_step_00001.safetensors"
        adapters.load_weights(transformer, distributed_path)
        reloaded = checkpoints.save_lora(transformer, accelerator, output / "reload", 1, metadata)
        reload_weights = load_file(reloaded)
        distributed_weights = load_file(distributed_path)
        measured["actual_step_one_reload_exact"] = all(
            torch.equal(distributed_weights[name], reload_weights[name]) for name in distributed_weights
        )
        measured["passed"] &= measured["actual_step_one_reload_exact"]
        if trace is not None:
            trace_evidence = trace.write(output / "consumer_trace.json")
    resources.validate_records(measurements, 1, ["load", "update", "export"], budget)
    store.verify(require_guide=settings.guide_mode == "d1")
    numerics.validate(numerics.capture(), required=True, expected=original_numerics)
    check_current(identities, producer, job_path=job_path, world=world)
    record = {
        "protocol": protocol,
        "comparison": measured,
        "serial_loss": sum(losses) / len(losses),
        "serial_grad_norm": norm,
        "resource_measurements": measurements,
        "consumer_trace_evidence": trace_evidence,
        "serial_runtime": serial_runtime,
        "state": "passed" if measured["passed"] else "failed",
        "scope": "one native update; previews/product/quality remain separate acceptance",
        "output_files": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
    }
    dataset.atomic_write(output / "result.json", lambda path: path.write_text(json.dumps(record, indent=2) + "\n"))
    return record



def scientific_args(argv: list[str]) -> argparse.Namespace:
    """Keep the original scientific replay choices; the queue owns execution."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--consumer-trace", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args(argv)
    if args.world_size != 4:
        parser.error("E4 requires the original four-rank update")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return spec_protocol.outer_args(argv, __doc__)


def read_spec(spec: Path, root: Path) -> argparse.Namespace:
    args = spec_protocol.read_spec(parse_args(["--spec", str(spec), "--output", str(root)]),
                                   "training_update_check", scientific_args,
                                   operations=("reference", "compare"))
    if (args.operation == "compare") != (args.reference is not None):
        raise ValueError("compare requires a separate reference; reference cannot select another reference")
    return args


def resource_phases(spec: Path) -> list[str]:
    args = read_spec(spec, spec.parent / "unused-output")
    return ["runtime"] if args.operation == "compare" else ["load", "update", "export"]


def resource_budget(spec: Path) -> Path | None:
    args = read_spec(spec, spec.parent / "unused-output")
    if args.operation == "compare":
        return None
    path = config.parse_settings(json.loads(args.job.read_text())["arguments"]).resource_budget
    if path is None or not path.is_absolute():
        raise ValueError("native replay requires its original absolute resource budget")
    return path


def dependency_outputs(spec: Path) -> list[Path]:
    args = read_spec(spec, spec.parent / "unused-output")
    return [args.reference] if args.operation == "compare" else []


def launch_requirements(spec: Path) -> dict:
    read_spec(spec, spec.parent / "unused-output")
    return {"processes": 1, "launcher": "python"}


def _saved_reference(root: Path, job: Path, world: int) -> tuple[dict, dict]:
    """Recompute E4 from bound CPU artifacts, using no native model or decoder."""
    record = json.loads((root / "result.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    actual = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
              if path.is_file() and path not in {root / "result.json", root / "manifest.json"}}
    required = {"protocol.json", "adam.pt", "resources_rank0.jsonl",
                "checkpoints/lora_weights_step_00000.safetensors",
                "checkpoints/lora_weights_step_00001.safetensors",
                "reload/lora_weights_step_00001.safetensors"}
    if (record.get("state") != "passed" or record.get("protocol") != protocol
            or record.get("output_files") != actual or not required.issubset(actual)
            or protocol.get("tolerance") != TOLERANCE or protocol.get("world_size") != world):
        raise ValueError("saved E4 reference inventory, state or protocol differs")
    settings, _store, _plan, _spec, visits, logs, states, initial, identities, _text = prepare(job, world)
    launch, saved, precision = check_launch(job, world)
    if (protocol["input_files"] != identities or protocol["visits"] != visits
            or protocol["serial_accumulation"] != len(visits) or protocol["mode"] != settings.mode
            or protocol["queue_job_sha256"] != launch["sha256"] or protocol["queue_launch"] != saved["queue_launch"]
            or protocol["distributed_runtime"] != saved["runtime"] or protocol["mixed_precision"] != precision
            or protocol["resource_budget"] != saved["resource_budget"]):
        raise ValueError("saved E4 reference differs from original scientific inputs")
    check_current(identities, protocol["software"], job_path=job, world=world)
    serial_runtime = record["serial_runtime"]
    if protocol.get("serial_runtime") != serial_runtime:
        raise ValueError("saved E4 serial runtime differs")
    runtime.validate(serial_runtime, 1, precision, numerical_policy=runtime.numerical_policy(saved["runtime"]))
    measurements, _ = resources.read_records(root, 1)
    if record["resource_measurements"] != measurements:
        raise ValueError("saved E4 resource records differ")
    resources.validate_records(measurements, 1, ["load", "update", "export"], saved["resource_budget"])
    trace_evidence = record.get("consumer_trace_evidence")
    if settings.consumer_trace and not trace_evidence:
        raise ValueError("original E4 job requires saved consumer trace evidence")
    if trace_evidence:
        trace_path = root / "consumer_trace.json"
        trace = json.loads(trace_path.read_text())
        if (Path(trace_evidence.get("path", "")).resolve() != trace_path.resolve()
                or trace_evidence.get("sha256") != sha256(trace_path)
                or trace_evidence.get("events") != len(trace.get("events", []))
                or trace_evidence.get("complete") is not True or trace.get("complete") is not True
                or trace.get("binding", {}).get("queue_job_sha256") != launch["sha256"]
                or trace.get("binding", {}).get("world_size") != 1):
            raise ValueError("saved E4 consumer trace differs from actual original-bound evidence")
    zero = load_file(root / "checkpoints/lora_weights_step_00000.safetensors")
    if set(zero) != set(initial) or any(not torch.equal(zero[name], initial[name]) for name in initial):
        raise ValueError("saved E4 initial export differs")
    serial = torch.load(root / "adam.pt", map_location="cpu", weights_only=True)
    if set(serial) != set(states):
        raise ValueError("saved E4 Adam inventory differs")
    for name, state in serial.items():
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"} or state["step"] != 1
                or any(value.dtype != torch.float32 or value.shape != states[name][key].shape
                       or not torch.isfinite(value).all() for key, value in state.items() if key != "step")
                or torch.any(state["exp_avg_sq"] < 0)):
            raise ValueError("saved E4 Adam moments are invalid")
    final = load_file(root / "checkpoints/lora_weights_step_00001.safetensors")
    distributed = load_file(settings.output / "checkpoints/lora_weights_step_00001.safetensors")
    measured = compare_update(states, serial, distributed, final, beta1=0.9,
                             norms=(logs[0]["grad_norm"], record["serial_grad_norm"]),
                             losses=(sum(log["loss"] for log in logs) / world, record["serial_loss"]))
    reloaded = load_file(root / "reload/lora_weights_step_00001.safetensors")
    measured["actual_step_one_reload_exact"] = (set(reloaded) == set(distributed)
        and all(torch.equal(distributed[name], reloaded[name]) for name in distributed))
    measured["passed"] &= measured["actual_step_one_reload_exact"]
    if record["comparison"] != measured or not measured["passed"]:
        raise ValueError("saved E4 numerical comparison fails or differs")
    return record, measured


def _reference_comparison(args: argparse.Namespace) -> dict:
    manifest = json.loads((args.reference / "manifest.json").read_text())
    reference_spec = Path(manifest["spec"]["path"])
    reference_args = read_spec(reference_spec, args.reference)
    if (reference_args.operation != "reference" or reference_args.job != args.job
            or reference_args.world_size != args.world_size):
        raise ValueError("E4 comparison requires a matching separate reference job")
    verify_completion(reference_spec, args.reference)
    _record, measured = _saved_reference(args.reference, args.job, args.world_size)
    return {"scope": "one native update; saved comparison only", "comparison": measured,
            "reference": {"path": str(args.reference.resolve()),
                          "manifest_sha256": sha256(args.reference / "manifest.json")}}


def verify_completion(spec: Path, root: Path) -> None:
    args = read_spec(spec, root)
    spec_protocol.inspect(args, "training_update_check")
    if args.operation == "reference":
        record, _ = _saved_reference(root, args.job, args.world_size)
        if args.consumer_trace and not record.get("consumer_trace_evidence"):
            raise ValueError("requested E4 consumer trace is missing")
    elif json.loads((root / "comparison.json").read_text()) != _reference_comparison(args):
        raise ValueError("saved E4 comparison differs from actual reference")


def evidence_paths(spec: Path, root: Path) -> list[Path]:
    verify_completion(spec, root)
    return spec_protocol.evidence(spec_protocol.inspect(read_spec(spec, root), "training_update_check"), root)


def main(argv: list[str] | None = None) -> int:
    outer = parse_args(argv)
    args = read_spec(outer.spec, outer.output)
    args.gpu_id, args.dry_run = outer.gpu_id, outer.dry_run
    if args.operation == "reference":
        result = execute(args.job, args.output, args.world_size, dry_run=args.dry_run,
                         consumer_trace=args.consumer_trace)
        if args.dry_run:
            return 0
        if not result["comparison"]["passed"]:
            return 2
    else:
        result = _reference_comparison(args)
        if args.dry_run:
            return 0
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    spec_protocol.publish(args, "training_update_check")
    verify_completion(args.spec, args.output)
    print(json.dumps(result, indent=2))  # noqa: T201 -- diagnostic CLI
    return 0


if __name__ == "__main__":
    from scripts.onestep_avatar.execution.runtime import run
    raise SystemExit(run(main))
