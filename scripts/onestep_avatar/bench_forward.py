"""Wall-clock cost per finalized chunk: the causal one-step rollout against the `k2` window.

Plan 2026-09-10 §2 claims "one forward instead of two, at half the transformer compute". §4.4's
causal scheme (2026-09-14) changes what has to be measured for that claim to mean anything, in
two ways that push in opposite directions:

* the denoising forward now covers only the **block** (2 latent frames), not a whole 4-latent-
  frame window -- half the query tokens;
* but the cache has to be refreshed with the block's clean latents, which is a **second**
  forward, and every block's queries attend over a longer key sequence (the pinned frame-0
  sink plus the retained context) than the old self-contained window did.

So the honest unit is **cost per finalized chunk of 16 pixel frames**: `k2`'s two window
forwards against the causal path's denoise + refresh. A FLOP count will not settle it --
attention-backend selection, kernel occupancy at these shapes, and memory-bandwidth-bound
layers do not scale with FLOPs -- and the predecessor of this script already caught one
FLOP-plausible arm (the extra-token hybrid at 2T) being *slower* than the baseline it was
meant to replace. Content is synthetic; only the geometry and the checkpoint's attention
layers are real.

Run from LTX-2, ltx env, ONE free GPU (~28 GB for the video-only transformer, no LoRA/FSDP):

    conda run -n ltx python -m scripts.onestep_avatar.bench_forward --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from scripts.onestep_avatar import causal_core
from scripts.onestep_avatar.causal_core import BlockCache, CausalGeometry, ClipGrid
from scripts.prune.core import model_registry, refine_core, refine_task
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


def main() -> int:  # noqa: PLR0915 -- one linear benchmark script.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--sigma0", type=float, default=0.725)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--context-latent-frames", type=int, nargs="+", default=[causal_core.CONTEXT_LATENT_FRAMES],
        help="Sweep the cache depth: it is the compute/quality knob of §4.4's scheme, and the "
        "whole point of measuring is to price each setting against k2.",
    )
    p.add_argument("--latent-frames", type=int, default=18, help="Clip length; the corpus's 150-frame tier.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    model = model_registry.resolve(args.model)
    device = torch.device(f"cuda:{args.gpu_id}")
    window = refine_task.deployed_geometry(model.scale_factors)
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    latent_channels = model.caps.latent_channels

    from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: PLC0415 -- torch-heavy, imported late
    from scripts.prune.core.session import Session  # noqa: PLC0415 -- torch-heavy, imported late

    sigmas = torch.tensor([args.sigma0, 0.0], dtype=torch.float32, device=device)
    session = Session(
        model=model, device=device, script="onestep_avatar.bench_forward",
        context=context, denoiser=SimpleDenoiser(context, None), sigmas=sigmas,
    )

    results: dict[str, dict[str, float]] = {}
    with session.transformer() as transformer:
        base = transformer
        while not hasattr(base, "transformer_blocks") and hasattr(base, "velocity_model"):
            base = base.velocity_model
        denoise_fn = causal_core.denoised_from_x0_model(transformer)

        # --- The k2 baseline: one bidirectional forward over a whole 4-latent-frame window.
        # Timed through refine_core, the module that produced every frozen k2 number, so the
        # comparison is against the deployed method rather than a re-implementation of it.
        window_tools = refine_core.tools_for_window(
            window, EDGE, EDGE, 25.0, latent_channels=latent_channels
        )
        z_window = torch.randn(
            1, latent_channels, window.latent_frames, EDGE // 32, EDGE // 32, dtype=DTYPE, device=device
        )
        window_state = refine_core.make_window_state(
            z_window, None, args.sigma0, window_tools, args.seed, device, DTYPE
        )
        window_times = _time(
            lambda: refine_core.run_schedule(transformer, session.denoiser, window_state, sigmas),
            reps=args.reps, warmup=args.warmup, device=device,
        )
        # run_schedule at a 2-point sigma list is ONE forward; k2 is two.
        results["k2_window_single_forward"] = _stats(window_times, int(window_state.latent.shape[1]))
        k2_per_chunk = 2 * statistics.median(window_times)

        # --- The causal path, per cache depth.
        for depth in args.context_latent_frames:
            geometry = CausalGeometry(
                scale_factors=model.scale_factors,
                block_latent_frames=causal_core.BLOCK_LATENT_FRAMES,
                context_latent_frames=depth,
            )
            grid = ClipGrid.build(
                args.latent_frames, EDGE, EDGE, 25.0, geometry,
                device=device, dtype=DTYPE, latent_channels=latent_channels,
            )
            cache = BlockCache.allocate(
                grid, geometry, num_layers=len(base.transformer_blocks), inner_dim=base.inner_dim,
                device=device, dtype=DTYPE,
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
                    denoise_fn, grid, cache,
                    torch.randn(1, e_hi - e_lo, latent_channels, dtype=DTYPE, device=device),
                    context, earlier,
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
                "over_k2_ratio": per_chunk / k2_per_chunk,
                "cached_key_tokens": cached_start,
                "cache_gib": 2 * len(base.transformer_blocks) * cache.caches[0].capacity * base.inner_dim * 2 / 2**30,
            }
            print(  # noqa: T201 -- CLI progress.
                f"ctx={depth}: denoise {statistics.median(denoise_times) * 1000:.0f}ms + refresh "
                f"{statistics.median(refresh_times) * 1000:.0f}ms = {per_chunk * 1000:.0f}ms/chunk, "
                f"{per_chunk / k2_per_chunk:.2f}x k2"
            )

    summary = {
        "geometry": {
            "edge": EDGE,
            "latent_frames": args.latent_frames,
            "block_latent_frames": causal_core.BLOCK_LATENT_FRAMES,
        },
        "k2_per_chunk_s": k2_per_chunk,
        **results,
    }
    print(json.dumps(summary, indent=2))  # noqa: T201 -- CLI completion summary.
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
