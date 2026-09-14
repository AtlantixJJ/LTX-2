"""One-off benchmark: wall-clock cost of one transformer forward at D1's T tokens vs D2's
2T tokens (plan 2026-09-10 SS4.1's "measure it before building anything on the FLOP table").

The FLOP estimate in SS4.1 puts D2 at ~1.09x `k2` (two D1 forwards) once the token
doubling is priced in against the 22B model's real linear/attention split -- which, if
right, is the whole reason D2c (additive guide embedding, still T tokens) is proposed as
the lead arm instead of D2 (extra reference tokens, 2T). A FLOP count is not a wall clock:
attention-backend selection, kernel occupancy at these exact shapes, and memory-bandwidth-
bound layers do not necessarily scale with FLOPs. This script reuses the REAL training
forward path (``train.one_window_forward``, guide_mode d1 vs d2) rather than reimplementing
the token-count math, so what is timed is exactly what a training step or `onestep_core`
rollout would run -- content is synthetic (a random latent of the deployed window shape);
only the token count depends on real geometry and the real checkpoint's attention layers.

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

from scripts.onestep_avatar.train import DTYPE, Window, one_window_forward
from scripts.prune.core import model_registry, refine_core, refine_task
from scripts.prune.data import prompt_cache


def _bench(
    transformer: torch.nn.Module,
    context: torch.Tensor,
    window: Window,
    geometry: refine_core.WindowGeometry,
    *,
    sigma0: float,
    seed: int,
    device: torch.device,
    latent_channels: int,
    guide_mode: str,
    reps: int,
    warmup: int,
) -> tuple[list[float], int]:
    times: list[float] = []
    n_tokens: int | None = None
    for i in range(warmup + reps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            z0_tokens, target_tokens, weights, state, tools = one_window_forward(
                transformer,
                context,
                window,
                None,
                geometry,
                sigma0=sigma0,
                seed=seed + i,
                device=device,
                latent_channels=latent_channels,
                guide_mode=guide_mode,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
            if n_tokens is None:
                n_tokens = int(state.latent.shape[1])
        del z0_tokens, target_tokens, weights, state, tools
    assert n_tokens is not None
    return times, n_tokens


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--sigma0", type=float, default=0.725)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    model = model_registry.resolve(args.model)
    device = torch.device(f"cuda:{args.gpu_id}")
    geometry = refine_task.deployed_geometry(model.scale_factors)
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)

    latent_channels = model.caps.latent_channels
    latent_frames = geometry.context_latent_frames + geometry.chunk_latent_frames + 1
    z = torch.randn(latent_channels, latent_frames, 32, 32, dtype=DTYPE)
    window = Window(z_g=z, z_y=z, fps=25.0, index=0, source="bench", loss_mask=None, z0_base=None)

    from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: PLC0415 -- torch-heavy, imported late
    from scripts.prune.core.session import Session  # noqa: PLC0415 -- torch-heavy, imported late

    sigmas = torch.tensor([args.sigma0, 0.0], dtype=torch.float32, device=device)
    denoiser = SimpleDenoiser(context, None)
    session = Session(
        model=model, device=device, script="onestep_avatar.bench_forward",
        context=context, denoiser=denoiser, sigmas=sigmas,
    )

    results: dict[str, dict[str, float]] = {}
    with session.transformer() as transformer:
        for guide_mode in ("d1", "d2"):
            times, n_tokens = _bench(
                transformer, context, window, geometry,
                sigma0=args.sigma0, seed=args.seed, device=device,
                latent_channels=latent_channels, guide_mode=guide_mode,
                reps=args.reps, warmup=args.warmup,
            )
            results[guide_mode] = {
                "n_tokens": n_tokens,
                "median_s": statistics.median(times),
                "mean_s": statistics.mean(times),
                "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                "reps": args.reps,
            }
            print(  # noqa: T201 -- CLI progress.
                f"{guide_mode}: tokens={n_tokens} median={statistics.median(times) * 1000:.1f}ms "
                f"+/- {results[guide_mode]['stdev_s'] * 1000:.1f}ms (n={args.reps})"
            )

    d1_median = results["d1"]["median_s"]
    d2_median = results["d2"]["median_s"]
    k2_cost = 2 * d1_median  # k2 = two D1-shaped forwards at T tokens, no guide conditioning
    summary = {
        **results,
        "d2_over_d1_measured_ratio": d2_median / d1_median,
        "d2_over_k2_measured_ratio": d2_median / k2_cost,
        "d1_over_k2_measured_ratio": d1_median / k2_cost,
    }
    print(json.dumps(summary, indent=2))  # noqa: T201 -- CLI completion summary.
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
