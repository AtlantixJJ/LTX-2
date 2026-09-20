"""Decode the D0 GT-renoise sanity probe at the distilled refiner's three levels.

D0 is deliberately not deployable: it noises the capture latent itself.  That makes it a
useful capacity control, but only if its review artifact uses that exact state rather than a
guide-noised approximation.  This script rolls a fixed chain of causal blocks at each probe
sigma and writes one portable MP4 per level:

    ground-truth capture | frozen base | D0 LoRA checkpoint

The fixed chain and seeds make a sequence of checkpoints directly comparable.  It is an
offline checkpoint probe, so no VAE is resident while FSDP training is stepping.

**Revised 2026-09-14 (SS4.4).** The probe rolls out through ``causal_core`` -- block-causal
attention plus the clean-latent K/V cache -- exactly as training and deployment do, so the
probe cannot silently diverge from either. What keeps it D0-specific is only that it noises
``z_y`` rather than ``z_g``; it still does not call :mod:`onestep_core`, which refuses D0 to
protect deployment from accepting an arm that needs the unavailable capture latent.

Run from ``LTX-2`` in the ``ltx`` environment::

python -m scripts.onestep_avatar.visualize_d0 \
    --subset ../expr/onestep_avatar/windows/t2r2.json \
    --checkpoint ../expr/onestep_avatar/runs/test/checkpoints/lora_weights_step_00100.safetensors \
    --output ../expr/onestep_avatar/runs/test/probes/step_00100 --gpu-id 5

To visualize several checkpoints from one run in a single call (one shared frozen-base decode
and GT panel, reused across every step), pass ``--run`` with ``--steps`` instead of spelling out
each ``--checkpoint`` path::

python -m scripts.onestep_avatar.visualize_d0 \
    --subset ../expr/onestep_avatar/windows/t2r2.json \
    --run ../expr/onestep_avatar/runs/test --steps 100 500 1000 \
    --output ../expr/onestep_avatar/runs/test/probes/multi --gpu-id 5

Pass ``--teacher-forcing`` when the run itself was trained with ``train.py --teacher-forcing``
(check the run's ``config.json``): it refreshes the rollout's cache from the ground-truth
capture instead of the checkpoint's own denoised output, the same ablation ``train.py`` makes,
so the checkpoint is probed under the input distribution it was actually trained on rather than
the self-forced one a real deployment (and the default here) has to use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from scripts.onestep_avatar import causal_core, dataset
from scripts.onestep_avatar.train import Chain, ChainStore, clip_grid_for
from scripts.prune.core.session import DTYPE, add_model_args, open_session
from scripts.prune.evaluate.decode import decode_latent
from scripts.prune.evaluate.metrics import t3_video

# This is the distilled *refiner* schedule's non-trivial levels, not the first entries of the
# full nine-value generation table.  The latter are all nearly pure noise and made the prior
# D0 probe uninformative.  Keep training and checkpoint review on these exact levels.
#
# sigma=0.0 is deliberately excluded. It is not an informative probe point: sigma=0.0 adds no
# noise, so the "denoised" state IS the input and there is nothing to measure -- the same reason
# train.py's `training_sigmas` refuses to train that level. Historically it also *crashed* here,
# in a silent retry loop for hours, because the old probe went through
# `refine_core.run_schedule`, whose `to_velocity(sample, sigma, denoised)` = `(sample -
# denoised)/sigma` raises "Sigma can't be 0.0". The causal rollout has no stepper and no such
# conversion, so that particular crash is gone -- but the level still says nothing.
PROBE_SIGMAS = (0.909375, 0.725, 0.421875)


def _chain(store: ChainStore, index: int) -> Chain:
    if index < 0 or index >= len(store):
        raise SystemExit(f"--chain-index {index} is outside [0, {len(store) - 1}]")
    chain = store[index]
    if not chain.seed_is_clip_start:
        raise SystemExit(
            "the D0 visual probe must start at a clip boundary so its cache is not primed from "
            "a teacher-forced GT prefix; choose a chain whose seed_is_clip_start is true"
        )
    return chain


def _run_d0_chain(transformer, context, chain: Chain, geometry, sigma: float, *, device, latent_channels: int, seed: int, teacher_forcing: bool = False):  # noqa: ANN001
    """Exact D0 AR rollout: ``z_y`` is both the noising source and the target reference.

    Goes through ``causal_core.rollout``, the one implementation training and deployment both
    use, so the cached context, the block-causal attention, the pinned frame-0 sink and the
    RoPE positions are the deployed ones by construction. The single D0-specific line is that
    the guide handed to the rollout is the capture's own master latent -- which for D0 is also
    the ground truth, so ``teacher_forcing`` needs no separate target tensor: refreshing from
    ``guide_tokens`` already means refreshing from ``z_y``.

    It does not call :mod:`onestep_core`: that module refuses D0 on purpose, to protect
    deployment from accepting an arm that needs the unavailable capture latent.
    """
    grid = clip_grid_for(chain, geometry, device=device, latent_channels=latent_channels)
    base = causal_core.base_model(transformer)
    cache = causal_core.BlockCache.allocate(
        grid, geometry, num_layers=len(base.transformer_blocks), inner_dim=base.inner_dim,
        device=device, dtype=DTYPE,
    )
    z_y = grid.patchify(chain.z_y.unsqueeze(0).to(device=device, dtype=DTYPE))
    plan = [geometry.plan(grid.latent_frames)[index] for index in chain.blocks]
    tokens, _ = causal_core.rollout(
        causal_core.denoised_from_x0_model(transformer),
        grid, geometry, cache, z_y, context, sigma, seed=seed, blocks=plan, teacher_forcing=teacher_forcing,
        first_frame_condition=z_y[:, : grid.tokens_per_latent_frame],
    )
    covered = plan[-1][1]
    return grid, grid.unpatchify_block(tokens[:, : covered * grid.tokens_per_latent_frame], covered)


def _target_latent(chain: Chain, grid, geometry, device: torch.device) -> torch.Tensor:  # noqa: ANN001
    """The GT capture over exactly the frames the rollout covered, for a frame-aligned panel."""
    plan = [geometry.plan(grid.latent_frames)[index] for index in chain.blocks]
    return chain.z_y.unsqueeze(0)[:, :, : plan[-1][1]].to(device=device, dtype=DTYPE)


def _checkpoint_name(path: Path) -> str:
    return path.stem.replace("lora_weights_", "")


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    """Combine explicit ``--checkpoint`` paths with ``--run``/``--steps`` shorthand.

    ``--steps`` resolves against ``train.py``'s fixed naming convention
    (``lora_weights_step_{step:05d}.safetensors`` under ``<run>/checkpoints/``) so a multi-step
    call reads as step numbers, not repeated full paths.
    """
    checkpoints = list(args.checkpoint)
    if args.steps:
        if args.run is None:
            raise SystemExit("--steps requires --run")
        checkpoints.extend(
            args.run / "checkpoints" / f"lora_weights_step_{step:05d}.safetensors" for step in args.steps
        )
    if not checkpoints:
        raise SystemExit("no checkpoints requested: pass --checkpoint and/or --run with --steps")
    return checkpoints


def generate_checkpoint(args: argparse.Namespace, checkpoint: Path | None, chain: Chain) -> tuple[object, dict[float, list[torch.Tensor]]]:
    session = open_session(args, script="onestep_avatar.visualize_d0")
    geometry = causal_core.deployed_geometry(session.model.scale_factors)
    loras = ()
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")
        loras = (LoraPathStrengthAndSDOps(str(checkpoint), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)

    # The VAE is held only after all four model outputs have been calculated, avoiding the
    # transformer+decoder coexistence that §7.4 explicitly excludes from the train loop.
    outputs: dict[float, torch.Tensor] = {}
    with session.transformer(loras=loras) as transformer:
        for sigma in PROBE_SIGMAS:
            _, latent = _run_d0_chain(
                transformer, session.context, chain, geometry, sigma,
                device=session.device, latent_channels=session.model.caps.latent_channels, seed=args.seed,
                teacher_forcing=args.teacher_forcing,
            )
            outputs[sigma] = latent

    # CPU tensors keep this result cheap while the next checkpoint's transformer is loaded.
    return session, {sigma: latent.cpu() for sigma, latent in outputs.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument(
        "--corpus-root", type=Path, default=None,
        help="Defaults to the subset's own corpus_root, where the master latents live.",
    )
    parser.add_argument("--checkpoint", type=Path, action="append", default=[], help="Explicit checkpoint path; repeat for multiple.")
    parser.add_argument(
        "--run", type=Path, default=None,
        help="Run directory whose checkpoints/lora_weights_step_NNNNN.safetensors --steps resolves against.",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[],
        help="Step numbers to visualize, e.g. --steps 100 500 1000; resolved under --run. Combines with --checkpoint.",
    )
    parser.add_argument(
        "--objective",
        choices=dataset.OBJECTIVES,
        default=None,
        help="Defaults to the subset's own objective (bg for subsets frozen before this field existed).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "held_out"), default="train")
    parser.add_argument("--chain-index", type=int, default=0, help="Index within --split; must begin at clip start.")
    parser.add_argument(
        "--teacher-forcing", action="store_true",
        help=(
            "Refresh the cache from the ground-truth capture instead of the model's own "
            "denoised output, matching train.py's --teacher-forcing ablation. Use this to probe "
            "a checkpoint trained with --teacher-forcing the way it was actually trained; the "
            "default (off) is the self-forced regime a real deployment has to use."
        ),
    )
    add_model_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subset = json.loads(args.subset.read_text())
    corpus_root = args.corpus_root or Path(subset["corpus_root"])
    objective = args.objective or subset.get("objective", dataset.DEFAULT_OBJECTIVE)
    # D0-only probe: _run_d0_chain never reads chain.z_g, so don't require the guide bundle to
    # exist on disk (see ChainStore.with_guide).
    store = ChainStore(
        subset, corpus_root, split=args.split, objective=objective, with_anchor=False, with_guide=False,
    )
    chain = _chain(store, args.chain_index)
    checkpoints = _resolve_checkpoints(args)
    args.output.mkdir(parents=True, exist_ok=True)
    sessions_and_outputs = [generate_checkpoint(args, checkpoint, chain) for checkpoint in (None, *checkpoints)]
    session, base = sessions_and_outputs[0]
    geometry = causal_core.deployed_geometry(session.model.scale_factors)
    grid = clip_grid_for(chain, geometry, device=session.device, latent_channels=session.model.caps.latent_channels)
    written = []
    # Decode once, after all transformer passes.  Every checkpoint video therefore gives the
    # requested before/after comparison in one frame-aligned artifact: GT | frozen base | LoRA.
    # No stitching any more: a causal rollout writes one latent covering the whole chain, so
    # there is no per-window overlap to drop and no seam to get wrong.
    with session.decoder() as decoder:
        target = decode_latent(session, _target_latent(chain, grid, geometry, session.device), decoder)
        base_pixels = {sigma: decode_latent(session, latent.to(session.device), decoder) for sigma, latent in base.items()}
        # Skip the frozen-base entry of sessions_and_outputs here: its own video would be
        # GT | frozen base | frozen base, which is redundant with base_pixels already being
        # one of the three panels in every LoRA checkpoint's video below.
        for checkpoint, (_unused_session, outputs) in zip(checkpoints, sessions_and_outputs[1:], strict=True):
            for sigma, latent in outputs.items():
                candidate = decode_latent(session, latent.to(session.device), decoder)
                path = args.output / f"{_checkpoint_name(checkpoint)}_sigma_{sigma:.6f}.mp4"
                t3_video(target, base_pixels[sigma], candidate, path, fps=chain.fps)
                written.append(path)
    (args.output / "manifest.json").write_text(json.dumps({
        "kind": "d0_gt_renoise_probe",
        "attention": "block_causal",
        "source": chain.source,
        "blocks": chain.blocks,
        "geometry": geometry.as_dict(),
        "fps": chain.fps,
        "seed": args.seed,
        "sigmas": list(PROBE_SIGMAS),
        "teacher_forcing": args.teacher_forcing,
        "layout": "ground_truth_capture | frozen_base_generated | checkpoint_generated",
        "videos": [str(path.name) for path in written],
    }, indent=2) + "\n")
    for path in written:
        print(path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
