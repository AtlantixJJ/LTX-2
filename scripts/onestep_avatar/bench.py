"""Measure explicit-mode generation, or instrument causal operations separately.

Run from LTX-2 in the ltx environment on a free GPU."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from scripts.onestep_avatar import hashing
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.model.causal import BlockCache, CausalGeometry
from scripts.onestep_avatar.model.common import ClipGrid
from scripts.prune.core import model_registry
from scripts.prune.core.session import DEFAULT_PROMPT
from scripts.prune.data import prompt_cache

DTYPE = torch.bfloat16
EDGE = 1024  # §4.5's only geometry


def _time(call, *, reps: int, warmup: int, device: torch.device) -> list[float]:  # noqa: ANN001
    times: list[float] = []
    for i in range(warmup + reps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            call()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    return times


def _stats(times: list[float], tokens: int) -> dict[str, float]:
    return {
        "query_tokens": tokens,
        "median_s": statistics.median(times),
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "reps": len(times),
    }


def measure_generation(
    transformer: torch.nn.Module, *inputs, device: torch.device, repetitions: int = 3, warmup: int = 1, **settings
) -> dict:
    """Measure ordinary mode sampling with fixed inputs and fresh per-call cache."""
    from scripts.onestep_avatar.evaluate import sample_case  # noqa: PLC0415 -- shared execution owner
    from scripts.onestep_avatar.hashing import tensor_sha256  # noqa: PLC0415 -- shared execution owner

    if repetitions < 1 or warmup < 0:
        raise ValueError("benchmark needs positive repetitions and nonnegative warmup")
    gpu = device.type == "cuda"
    rows = []
    for index in range(warmup + repetitions):
        if gpu:
            torch.cuda.synchronize(device)
        baseline = (
            None
            if not gpu
            else {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device)}
        )
        if gpu:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output, record = sample_case(transformer, *inputs, **settings)
        if gpu:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if index >= warmup:
            peaks = (
                None
                if not gpu
                else {
                    "allocated": torch.cuda.max_memory_allocated(device),
                    "reserved": torch.cuda.max_memory_reserved(device),
                }
            )
            pixel_frames = common.pixel_frames_for(record["frames"], inputs[1].tools.scale_factors.time)
            rows.append(
                {
                    "elapsed_s": elapsed,
                    "call_counts": record["call_counts"],
                    "output_sha256": tensor_sha256(output),
                    "encoded_frames": record["frames"],
                    "covered_rgb_frames": pixel_frames,
                    "generated_rgb_frames": pixel_frames - 1,
                    "covered_rgb_frames_per_s": pixel_frames / elapsed,
                    "generated_rgb_frames_per_s": (pixel_frames - 1) / elapsed,
                    "memory_baseline_bytes": baseline,
                    "memory_peak_bytes": peaks,
                    "memory_extra_peak_bytes": None if not gpu else {key: peaks[key] - baseline[key] for key in peaks},
                }
            )
        del output, record
    times = [row["elapsed_s"] for row in rows]
    return {
        "mode": settings["mode"],
        "device": str(device),
        "torch": torch.__version__,
        "warmup": warmup,
        "repetitions": rows,
        "elapsed_s": {"minimum": min(times), "median": statistics.median(times), "maximum": max(times)},
        "boundary": "sample_case through complete CPU encoding; excludes model/text loading, decoding and writes",
    }


def _operation_args(argv: list[str]) -> argparse.Namespace:
    """Parse causal operation timing without reading inputs or opening weights."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument("--operation-timing", action="store_true", required=True)
    p.add_argument("--mode", choices=("causal",), required=True)
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--sigma0", type=float, default=0.725)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--context-latent-frames",
        type=int,
        nargs="+",
        default=[causal_core.CONTEXT_LATENT_FRAMES],
        help="Sweep the cache depth: it is the compute/quality knob; "
        "measure denoising and cache refresh separately at each depth.",
    )
    p.add_argument("--latent-frames", type=int, default=18, help="Clip length; the corpus's 150-frame tier.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.reps < 1 or args.warmup < 0:
        p.error("benchmark needs positive repetitions and nonnegative warmup")
    if not math.isfinite(args.sigma0) or not 0 < args.sigma0 <= 1:
        p.error("operation sigma must be finite and in (0, 1]")
    if len(set(args.context_latent_frames)) != len(args.context_latent_frames) or any(
        not 0 <= depth <= causal_core.MAX_CONTEXT_LATENT_FRAMES for depth in args.context_latent_frames
    ):
        p.error("operation cache depths must be unique and supported")
    if args.latent_frames < 1 + causal_core.BLOCK_LATENT_FRAMES:
        p.error("operation timing needs at least one complete causal block")
    args.out = args.output / "operation_timing.json"
    args.completion = completion_descriptor(args)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Expose ordinary or operation-timing arguments to queue preflight."""
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- ordinary parser owner

    arguments = sys.argv[1:] if argv is None else argv
    if "--operation-timing" in arguments:
        return _operation_args(arguments)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    measured, remaining = parser.parse_known_args(arguments)
    if measured.repetitions < 1 or measured.warmup < 0:
        parser.error("benchmark needs positive repetitions and nonnegative warmup")
    args = evaluate.parse_args(remaining)
    args.operation_timing = False
    args.repetitions, args.warmup = measured.repetitions, measured.warmup
    args.evaluation_arguments = remaining
    args.completion = completion_descriptor(args)
    return args


def completion_descriptor(args: argparse.Namespace) -> dict | None:
    """Describe operation output; ordinary case inventory requires input preflight."""
    if args.operation_timing:
        return {"records": [str(args.out.resolve())]}
    return None


def _operation_settings(args: argparse.Namespace) -> dict:
    return {
        field: getattr(args, field)
        for field in (
            "mode",
            "model",
            "gpu_id",
            "sigma0",
            "reps",
            "warmup",
            "seed",
            "context_latent_frames",
            "latent_frames",
        )
    }


def operation_main(argv: list[str]) -> int:
    args = _operation_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("operation timing output is already used")
    producer_software = software.capture("evaluation", "causal", extra_sources=("scripts/onestep_avatar/bench.py",))

    torch.manual_seed(args.seed)
    model = model_registry.resolve(args.model)
    device = torch.device(f"cuda:{args.gpu_id}")
    context = prompt_cache.get_or_build(model, DEFAULT_PROMPT, DTYPE, device)
    latent_channels = model.caps.latent_channels

    from scripts.prune.core.session import Session  # noqa: PLC0415 -- torch-heavy, imported late

    session = Session(
        model=model,
        device=device,
        script="onestep_avatar.bench",
        context=context,
    )

    results: dict[str, dict[str, float]] = {}
    measurements, cache_details = {}, {}
    with session.transformer() as transformer:
        base = common.base_model(transformer)
        denoise_fn = common.denoised_from_x0_model(transformer)

        # --- The causal path, per cache depth.
        for depth in args.context_latent_frames:
            geometry = CausalGeometry(
                scale_factors=model.scale_factors,
                block_latent_frames=causal_core.BLOCK_LATENT_FRAMES,
                context_latent_frames=depth,
            )
            grid = ClipGrid.build(
                args.latent_frames,
                EDGE,
                EDGE,
                25.0,
                geometry,
                device=device,
                dtype=DTYPE,
                latent_channels=latent_channels,
            )
            cache = BlockCache.allocate(
                grid,
                geometry,
                num_layers=len(base.transformer_blocks),
                inner_dim=base.inner_dim,
                device=device,
                dtype=DTYPE,
            )
            plan = geometry.plan(grid.latent_frames)
            # Measure a STEADY-STATE block, not block 0: block 0 has an empty cache and would
            # flatter the causal path by exactly the attention the cache adds.
            span = plan[-1]
            lo, hi = grid.token_span(*span)
            tokens = torch.randn(1, hi - lo, latent_channels, dtype=DTYPE, device=device)
            for earlier in plan[:-1]:
                e_lo, e_hi = grid.token_span(*earlier)
                causal_core.refresh_block(
                    denoise_fn,
                    grid,
                    cache,
                    torch.randn(1, e_hi - e_lo, latent_channels, dtype=DTYPE, device=device),
                    context,
                    earlier,
                )
            cached_start = cache.start

            def denoise(grid=grid, cache=cache, tokens=tokens, span=span) -> None:  # noqa: ANN001
                causal_core.denoise_block(denoise_fn, grid, cache, tokens, context, args.sigma0, span)

            def refresh(grid=grid, cache=cache, tokens=tokens, span=span, pin=cached_start) -> None:  # noqa: ANN001
                # Re-pin the cache length so repeated refreshes measure the same state rather
                # than a cache that grows (and evicts) under the timer.
                for layer in cache.caches:
                    layer.length = pin
                causal_core.refresh_block(denoise_fn, grid, cache, tokens, context, span)
                for layer in cache.caches:
                    layer.length = pin

            denoise_times = _time(denoise, reps=args.reps, warmup=args.warmup, device=device)
            refresh_times = _time(refresh, reps=args.reps, warmup=args.warmup, device=device)
            measurements[str(depth)] = {"denoise_s": denoise_times, "refresh_s": refresh_times}
            cache_details[str(depth)] = {
                "layers": len(base.transformer_blocks),
                "inner_dim": base.inner_dim,
                "capacity": cache.caches[0].capacity,
                "element_bytes": 2,
            }
            per_chunk = statistics.median(denoise_times) + statistics.median(refresh_times)
            results[f"causal_denoise_ctx{depth}"] = _stats(denoise_times, int(tokens.shape[1]))
            results[f"causal_refresh_ctx{depth}"] = _stats(refresh_times, int(tokens.shape[1]))
            results[f"causal_total_ctx{depth}"] = {
                "per_chunk_s": per_chunk,
                "cached_key_tokens": cached_start,
                "cache_gib": 2 * len(base.transformer_blocks) * cache.caches[0].capacity * base.inner_dim * 2 / 2**30,
            }
            print(  # noqa: T201 -- CLI progress.
                f"ctx={depth}: denoise {statistics.median(denoise_times) * 1000:.0f}ms + refresh "
                f"{statistics.median(refresh_times) * 1000:.0f}ms = {per_chunk * 1000:.0f}ms/chunk, "
            )

    summary = {
        "schema_version": 1,
        "kind": "onestep_avatar.operation_timing",
        "state": "complete",
        "settings": _operation_settings(args),
        "software": producer_software,
        "producer_source_sha256": hashing.sha256(Path(__file__)),
        "transformer": {
            "path": str(Path(model.paths.transformer()).resolve()),
            "sha256": hashing.sha256(Path(model.paths.transformer())),
        },
        "measurements": measurements,
        "cache_details": cache_details,
        "geometry": {
            "edge": EDGE,
            "latent_frames": args.latent_frames,
            "block_latent_frames": causal_core.BLOCK_LATENT_FRAMES,
        },
        **results,
    }
    print(json.dumps(summary, indent=2))  # noqa: T201 -- CLI completion summary.
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        software.check_current(producer_software)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def _positive_times(values: object, count: int) -> list[float]:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values)
    ):
        raise ValueError("benchmark requires the exact finite positive timing inventory")
    return values


def _verify_memory(row: dict, *, gpu: bool) -> None:
    fields = ("memory_baseline_bytes", "memory_peak_bytes", "memory_extra_peak_bytes")
    values = [row.get(field) for field in fields]
    if not gpu:
        if any(value is not None for value in values):
            raise ValueError("CPU benchmark must keep memory measurements unavailable")
        return
    if any(
        not isinstance(value, dict)
        or set(value) != {"allocated", "reserved"}
        or any(type(number) is not int or number < 0 for number in value.values())
        for value in values
    ):
        raise ValueError("benchmark memory inventory is invalid")
    baseline, peak, extra = values
    if (
        baseline["reserved"] < baseline["allocated"]
        or peak["reserved"] < peak["allocated"]
        or any(peak[key] < baseline[key] or extra[key] != peak[key] - baseline[key] for key in peak)
    ):
        raise ValueError("benchmark memory peaks differ from their baseline and extra values")


def _verify_operation(args: argparse.Namespace) -> dict:
    """Recompute retained operation summaries from saved times and cache geometry."""
    record = json.loads(args.out.read_text())
    model = model_registry.resolve(args.model)
    expected = {
        "schema_version": 1,
        "kind": "onestep_avatar.operation_timing",
        "state": "complete",
        "settings": _operation_settings(args),
        "software": software.capture("evaluation", "causal", extra_sources=("scripts/onestep_avatar/bench.py",)),
        "producer_source_sha256": hashing.sha256(Path(__file__)),
        "transformer": {
            "path": str(Path(model.paths.transformer()).resolve()),
            "sha256": hashing.sha256(Path(model.paths.transformer())),
        },
        "geometry": {
            "edge": EDGE,
            "latent_frames": args.latent_frames,
            "block_latent_frames": causal_core.BLOCK_LATENT_FRAMES,
        },
    }
    depths = {str(depth) for depth in args.context_latent_frames}
    measurements, details = record.get("measurements", {}), record.get("cache_details", {})
    if (
        any(record.get(key) != value for key, value in expected.items())
        or not isinstance(measurements, dict)
        or set(measurements) != depths
        or not isinstance(details, dict)
        or set(details) != depths
    ):
        raise ValueError("operation timing settings or depth inventory differs")
    software.check_current(record["software"])
    tokens_per_frame = (EDGE // model.scale_factors.height) * (EDGE // model.scale_factors.width)
    for depth in args.context_latent_frames:
        geometry = CausalGeometry(model.scale_factors, causal_core.BLOCK_LATENT_FRAMES, depth)
        span = geometry.plan(args.latent_frames)[-1]
        tokens = (span[1] - span[0]) * tokens_per_frame
        cached = min(span[0], geometry.sink_latent_frames + depth) * tokens_per_frame
        measurements_row, cache = measurements[str(depth)], details[str(depth)]
        if (
            not isinstance(measurements_row, dict)
            or set(measurements_row) != {"denoise_s", "refresh_s"}
            or not isinstance(cache, dict)
            or set(cache) != {"layers", "inner_dim", "capacity", "element_bytes"}
            or any(type(number) is not int or number < 1 for number in cache.values())
            or cache["element_bytes"] != 2
            or cache["capacity"] != geometry.cache_latent_frames_for(args.latent_frames) * tokens_per_frame
        ):
            raise ValueError("operation timing cache geometry or measurement inventory differs")
        denoise, refresh = (_positive_times(measurements_row[key], args.reps) for key in ("denoise_s", "refresh_s"))
        expected[f"causal_denoise_ctx{depth}"] = _stats(denoise, tokens)
        expected[f"causal_refresh_ctx{depth}"] = _stats(refresh, tokens)
        expected[f"causal_total_ctx{depth}"] = {
            "per_chunk_s": statistics.median(denoise) + statistics.median(refresh),
            "cached_key_tokens": cached,
            "cache_gib": 2 * cache["layers"] * cache["capacity"] * cache["inner_dim"] * 2 / 2**30,
        }
    expected.update(measurements=measurements, cache_details=details)
    if record != expected:
        raise ValueError("operation timing summary differs from its saved measurements")
    return record


def _ordinary_records(args: argparse.Namespace) -> list[Path]:
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- canonical inventory

    _, variants, cases, _ = evaluate.prepare_evaluation(args, require_fresh_output=False)
    return [
        args.output / f"case_{case:04d}" / f"variant_{variant:03d}" / "result.json"
        for case in range(len(cases))
        for variant in range(len(variants))
    ]


def _verify_generation_timing(args: argparse.Namespace, record: dict) -> None:
    timing = record.get("benchmark", {})
    if (
        not isinstance(timing, dict)
        or set(timing)
        != {"mode", "device", "torch", "warmup", "repetitions", "elapsed_s", "boundary", "untimed_artifact_calls"}
        or timing.get("mode") != args.mode
        or timing.get("torch") != torch.__version__
        or timing.get("warmup") != args.warmup
        or type(timing.get("warmup")) is not int
        or timing.get("untimed_artifact_calls") != 1
        or type(timing.get("untimed_artifact_calls")) is not int
        or timing.get("device") not in ("cpu", f"cuda:{args.gpu_id}")
        or timing.get("boundary")
        != "sample_case through complete CPU encoding; excludes model/text loading, decoding and writes"
    ):
        raise ValueError("benchmark warmup, device or measurement boundary differs")
    rows = timing["repetitions"]
    if not isinstance(rows, list) or len(rows) != args.repetitions or any(not isinstance(row, dict) for row in rows):
        raise ValueError("benchmark repetition inventory differs")
    times = _positive_times([row.get("elapsed_s") for row in rows], args.repetitions)
    frame_count = record["frames"]
    specification = model_registry.resolve(args.model)
    covered = common.pixel_frames_for(frame_count, specification.scale_factors.time)
    blocks, refresh = 1, 0
    if args.mode == "causal":
        geometry = CausalGeometry(
            specification.scale_factors,
            args.mode_settings.block_latent_frames,
            args.mode_settings.context_latent_frames,
        )
        blocks = len(geometry.plan(frame_count))
        refresh = blocks if args.history_mode == "cache" and args.kv_source == "refresh" else 0
    denoises = blocks * (len(args.schedule) - 1)
    passes = 1 + int(not math.isclose(args.cfg, 1.0)) + int(not math.isclose(args.stg, 0.0))
    counts = {
        "model_calls": (denoises + refresh) * passes,
        "denoise_calls": denoises,
        "prime_calls": 0,
        "refresh_calls": refresh,
    }
    if record.get("call_counts") != counts:
        raise ValueError("benchmark artifact call inventory differs from requested sampling")
    output = torch.load(Path(record["output"]["path"]), map_location="cpu", weights_only=True)
    for row, elapsed in zip(rows, times, strict=True):
        expected = {
            "call_counts": counts,
            "output_sha256": hashing.tensor_sha256(output),
            "encoded_frames": frame_count,
            "covered_rgb_frames": covered,
            "generated_rgb_frames": covered - 1,
            "covered_rgb_frames_per_s": covered / elapsed,
            "generated_rgb_frames_per_s": (covered - 1) / elapsed,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("benchmark repetition output, calls or coverage differs")
        _verify_memory(row, gpu=timing["device"].startswith("cuda:"))
    if timing["elapsed_s"] != {"minimum": min(times), "median": statistics.median(times), "maximum": max(times)}:
        raise ValueError("benchmark elapsed summary differs from its repetitions")


def verify_completion(arguments: list[str], root: Path) -> dict:
    """Verify saved benchmark outputs and exact measurements without native sessions."""
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- canonical scientific verifier

    args = parse_args(arguments)
    if args.output is None or args.output.resolve() != root.resolve():
        raise ValueError("benchmark completion output differs from the requested execution")
    if args.operation_timing:
        if not args.out.resolve().is_relative_to(root.resolve()):
            raise ValueError("operation timing evidence escapes its output")
        return _verify_operation(args)
    if args.dry_run:
        raise ValueError("benchmark dry run has no complete measurements")
    paths = _ordinary_records(args)
    evaluate.verify_evaluation_conditions(
        args.evaluation_arguments, paths, extra_sources=("scripts/onestep_avatar/bench.py",)
    )
    for path in paths:
        _verify_generation_timing(args, json.loads(path.read_text()))
    return {"records": [str(path.resolve()) for path in paths]}


def evidence_paths(arguments: list[str], root: Path) -> list[Path]:
    """Return every verified measured result and its ordinary saved input/output."""
    record = verify_completion(arguments, root)
    args = parse_args(arguments)
    if args.operation_timing:
        return [args.out]
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- shared byte inventory

    paths = evaluate.evaluation_evidence_paths(args.evaluation_arguments, [Path(path) for path in record["records"]])
    if any(not path.resolve().is_relative_to(root.resolve()) for path in paths):
        raise ValueError("benchmark evidence escapes its output")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Measure explicit-mode generation using ordinary checked evaluation inputs."""
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- reuse the evaluation CLI/session owner

    arguments = sys.argv[1:] if argv is None else argv
    args = parse_args(arguments)
    if args.operation_timing:
        return operation_main(arguments)

    def sample_runner(transformer, *inputs, **settings):  # noqa: ANN001, ANN202 -- ordinary sampler interface
        timings = measure_generation(
            transformer,
            *inputs,
            device=inputs[2].device,
            repetitions=args.repetitions,
            warmup=args.warmup,
            **settings,
        )
        output, record = evaluate.sample_case(transformer, *inputs, **settings)
        identity = hashing.tensor_sha256(output)
        if any(row["output_sha256"] != identity for row in timings["repetitions"]):
            raise ValueError("benchmark measured outputs differ from the saved artifact")
        timings["untimed_artifact_calls"] = 1
        record["benchmark"] = timings
        return output, record

    return evaluate.execute_evaluation(
        args, sample_runner=sample_runner, extra_sources=("scripts/onestep_avatar/bench.py",)
    )


if __name__ == "__main__":
    from scripts.onestep_avatar.execution.runtime import run

    raise SystemExit(run(main))
