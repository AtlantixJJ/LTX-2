"""Measure explicit-mode generation, or instrument causal operations separately.

Run from LTX-2 in the ltx environment on a free GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from scripts.onestep_avatar import hashing
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


def operation_main(argv: list[str]) -> int:  # noqa: PLR0915 -- retained instrumented operation diagnostic
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

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
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Measure explicit-mode generation using ordinary checked evaluation inputs."""
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- reuse the evaluation CLI/session owner

    arguments = sys.argv[1:] if argv is None else argv
    if "--operation-timing" in arguments:
        return operation_main(arguments)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    measured, remaining = parser.parse_known_args(arguments)
    if measured.repetitions < 1 or measured.warmup < 0:
        parser.error("benchmark needs positive repetitions and nonnegative warmup")
    args = evaluate.parse_args(remaining)

    def sample_runner(transformer, *inputs, **settings):  # noqa: ANN001, ANN202 -- ordinary sampler interface
        timings = measure_generation(
            transformer,
            *inputs,
            device=inputs[2].device,
            repetitions=measured.repetitions,
            warmup=measured.warmup,
            **settings,
        )
        output, record = evaluate.sample_case(transformer, *inputs, **settings)
        identity = hashing.tensor_sha256(output)
        if any(row["output_sha256"] != identity for row in timings["repetitions"]):
            raise ValueError("benchmark measured outputs differ from the saved artifact")
        timings["untimed_artifact_calls"] = 1
        record["benchmark"] = timings
        return output, record

    return evaluate.execute_evaluation(args, sample_runner=sample_runner)


if __name__ == "__main__":
    raise SystemExit(main())
