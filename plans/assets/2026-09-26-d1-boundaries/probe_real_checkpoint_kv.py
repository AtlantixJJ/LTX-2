"""Compare clean block-0 K/V at refresh sigma 0 and an active prompt sigma.

This uses the real frozen model, with identical clean tokens and zero per-token timesteps.
Only the global sigma supplied to prompt AdaLN changes. Run from LTX-2 in the ltx env on a
free GPU; output is a small JSON summary, never a copy of the model or K/V tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.onestep_avatar import causal_core, visualize_d1
from scripts.onestep_avatar.train import clip_grid_for
from scripts.prune.core.session import DTYPE, open_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--objective", choices=("white", "bg"), default="white")
    parser.add_argument("--sigma", type=float, default=0.909375)
    parser.add_argument("--model", default="2.5")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chain = visualize_d1._load_chain(args.view, args.objective)
    session = open_session(args, script="onestep_avatar.probe_real_checkpoint_kv")
    if args.sigma not in session.model.sigmas:
        raise SystemExit(f"sigma {args.sigma} is not on the selected model schedule")
    geometry = causal_core.deployed_geometry(session.model.scale_factors)
    grid = clip_grid_for(chain, geometry, device=session.device, latent_channels=session.model.caps.latent_channels)
    span = geometry.plan(grid.latent_frames)[0]
    lo, hi = grid.token_span(*span)
    tokens = grid.patchify(chain.z_y.unsqueeze(0).to(device=session.device, dtype=DTYPE))[:, lo:hi]

    with session.transformer() as transformer:
        base = causal_core.base_model(transformer)
        cache = causal_core.BlockCache.allocate(
            grid,
            geometry,
            num_layers=len(base.transformer_blocks),
            inner_dim=base.inner_dim,
            device=session.device,
            dtype=DTYPE,
            capacity_latent_frames=span[1],
        )
        denoise = causal_core.denoised_from_x0_model(transformer)

        def write(global_sigma: float) -> None:
            cache.reset()
            modality = causal_core.block_modality(
                grid,
                tokens,
                session.context,
                global_sigma,
                token_slices=[(lo, hi)],
                cache=cache,
                kv_write=True,
                clean_prefix_tokens=tokens.shape[1],
            )
            assert torch.count_nonzero(modality.timesteps) == 0
            with torch.no_grad():
                denoise(modality)

        write(0.0)
        baseline = [
            (layer.k[:, : cache.start].cpu().clone(), layer.v[:, : cache.start].cpu().clone()) for layer in cache.caches
        ]
        write(args.sigma)
        layers = []
        for index, (layer, (old_k, old_v)) in enumerate(zip(cache.caches, baseline, strict=True)):
            comparisons = {}
            for name, current, previous in (
                ("k", layer.k[:, : cache.start], old_k),
                ("v", layer.v[:, : cache.start], old_v),
            ):
                difference = current.float() - previous.to(current.device).float()
                comparisons[name] = {
                    "max_abs": difference.abs().max().item(),
                    "relative_l2": (difference.norm() / previous.float().norm()).item(),
                }
            layers.append({"layer": index, **comparisons})

    result = {
        "model": session.stamp(dtype=str(DTYPE)),
        "view": str(args.view.resolve()),
        "objective": args.objective,
        "span": list(span),
        "token_count": tokens.shape[1],
        "baseline_global_sigma": 0.0,
        "active_global_sigma": args.sigma,
        "history_token_timestep": 0.0,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
