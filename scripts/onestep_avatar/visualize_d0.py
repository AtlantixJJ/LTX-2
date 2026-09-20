"""Decode the D0 GT-renoise sanity probe at the distilled refiner's three levels.

D0 is deliberately not deployable: it noises the capture latent itself.  That makes it a
useful capacity control, but only if its review artifact uses that exact state rather than a
guide-noised approximation.

It runs **one rollout per probe sigma** -- three in total -- and writes one portable MP4 per
level:

    ground-truth capture | frozen base | D0 LoRA checkpoint

Each rollout covers the **whole clip** by default (``--span clip``): block 0 through the last
full block, the same sequence an inference run produces, so error accumulated across the AR
rollout is visible rather than truncated at the training chain's ``K`` blocks. ``--span chain``
restores the old behaviour of covering only the subset chain's blocks.

The fixed clip and seeds make a sequence of checkpoints directly comparable.  It is an
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

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


def _chain(store: ChainStore, index: int, span: str) -> Chain:
    if index < 0 or index >= len(store):
        raise SystemExit(f"--chain-index {index} is outside [0, {len(store) - 1}]")
    chain = store[index]
    # At ``--span clip`` the rollout starts at block 0 regardless of where the chain does, so
    # the cache is never primed and this property is not needed. At ``--span chain`` it is: a
    # mid-clip chain would otherwise need a teacher-forced GT prefix in its cache, which is a
    # different condition from the one the checkpoint is being judged on.
    if span == "chain" and not chain.seed_is_clip_start:
        raise SystemExit(
            "--span chain must start at a clip boundary so its cache is not primed from "
            "a teacher-forced GT prefix; choose a chain whose seed_is_clip_start is true, "
            "or use the default --span clip"
        )
    return chain


def _plan_for(chain: Chain, geometry, grid, span: str) -> list[tuple[int, int]]:  # noqa: ANN001
    """The blocks this probe rolls out.

    ``clip`` (the default) is the whole clip from block 0 to the last full block -- the full
    sequence an inference run produces, so drift across the rollout is visible rather than
    truncated at the training chain's ``K``. ``chain`` reproduces the historical behaviour of
    covering only the subset chain's own blocks.
    """
    plan = geometry.plan(grid.latent_frames)
    if span == "clip":
        return plan
    return [plan[index] for index in chain.blocks]


def _run_d0_chain(  # noqa: ANN202
    transformer,  # noqa: ANN001
    context,  # noqa: ANN001
    chain: Chain,
    geometry,  # noqa: ANN001
    sigma: float,
    *,
    device,  # noqa: ANN001
    latent_channels: int,
    seed: int,
    teacher_forcing: bool = False,
    span: str = "clip",
):
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
    plan = _plan_for(chain, geometry, grid, span)
    tokens, _ = causal_core.rollout(
        causal_core.denoised_from_x0_model(transformer),
        grid, geometry, cache, z_y, context, sigma, seed=seed, blocks=plan, teacher_forcing=teacher_forcing,
        first_frame_condition=z_y[:, : grid.tokens_per_latent_frame],
    )
    covered = plan[-1][1]
    return grid, grid.unpatchify_block(tokens[:, : covered * grid.tokens_per_latent_frame], covered)


def _target_latent(chain: Chain, grid, geometry, device: torch.device, span: str) -> torch.Tensor:  # noqa: ANN001
    """The GT capture over exactly the frames the rollout covered, for a frame-aligned panel."""
    plan = _plan_for(chain, geometry, grid, span)
    return chain.z_y.unsqueeze(0)[:, :, : plan[-1][1]].to(device=device, dtype=DTYPE)


def _frame_labels(plan: list[tuple[int, int]], time_scale: int) -> list[str]:
    """One caption per decoded pixel frame: which latent frame and which rollout step.

    The mapping is the causal VAE's, not a ratio: latent frame 0 is a single pixel frame (the
    keyframe) and every later latent frame is ``time_scale`` of them, so pixel frame ``p``
    belongs to latent frame ``0 if p == 0 else ceil(p / time_scale)``. Reading a drift back to
    the block that produced it is the whole point of the probe, and counting frames by hand
    off a 137-frame video is how that gets got wrong.
    """
    labels: list[str] = []
    for step, (start, end) in enumerate(plan):
        for latent in range(start, end):
            first_pixel = 0 if latent == 0 else (latent - 1) * time_scale + 1
            last_pixel = 0 if latent == 0 else latent * time_scale
            for _ in range(last_pixel - first_pixel + 1):
                labels.append(f"latent {latent}  ·  rollout step {step}  (block {start}-{end - 1})")
    return labels


def _stamp(pixels: torch.Tensor, labels: list[str]) -> torch.Tensor:
    """Burn ``labels[i]`` into the top-left of frame ``i`` of a ``[T, C, H, W]`` 0-1 tensor.

    Burned in rather than written to a sidecar because the artifact that gets looked at, and
    forwarded, is the MP4 itself; a caption that lives anywhere else is not there when someone
    scrubs to the frame where the identity slips.

    Only the caption band is written. Rendering the whole frame through PIL would round-trip
    every pixel through uint8 and quantize the image being reviewed -- a probe must not alter
    the thing it is showing, even by half a level.
    """
    frames, _, height, width = pixels.shape
    size = max(14, height // 36)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 has no size argument
        font = ImageFont.load_default()
    pad = max(2, size // 4)
    bands: dict[str, torch.Tensor] = {}  # a caption repeats for every pixel frame of its latent frame
    out = pixels.clone()
    for index in range(min(frames, len(labels))):
        text = labels[index]
        band = bands.get(text)
        if band is None:
            box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
            image = Image.new("RGB", (min(box[2] + 2 * pad, width), min(box[3] + 2 * pad, height)), (0, 0, 0))
            ImageDraw.Draw(image).text((pad, pad), text, fill=(255, 255, 255), font=font)
            band = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255)
            bands[text] = band.to(dtype=out.dtype, device=out.device)
            band = bands[text]
        out[index, :3, : band.shape[1], : band.shape[2]] = band
    return out


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
                teacher_forcing=args.teacher_forcing, span=args.span,
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
    parser.add_argument("--chain-index", type=int, default=0, help="Index within --split; selects the clip (and, at --span chain, the blocks).")
    parser.add_argument(
        "--span", choices=("clip", "chain"), default="clip",
        help=(
            "clip (default): roll the WHOLE clip from block 0, so each video is the full "
            "inference sequence and drift across it is visible. chain: cover only the subset "
            "chain's own K blocks, the historical behaviour; requires a clip-start chain."
        ),
    )
    parser.add_argument(
        "--teacher-forcing", action="store_true",
        help=(
            "Refresh the cache from the ground-truth capture instead of the model's own "
            "denoised output, matching train.py's --teacher-forcing ablation. Use this to probe "
            "a checkpoint trained with --teacher-forcing the way it was actually trained; the "
            "default (off) is the self-forced regime a real deployment has to use."
        ),
    )
    parser.add_argument(
        "--no-frame-labels", action="store_true",
        help="Do not burn the per-frame 'latent N · rollout step M' caption into the video.",
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
    chain = _chain(store, args.chain_index, args.span)
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
    plan = _plan_for(chain, geometry, grid, args.span)
    labels = [] if args.no_frame_labels else _frame_labels(plan, geometry.scale_factors.time)
    with session.decoder() as decoder:
        target = decode_latent(session, _target_latent(chain, grid, geometry, session.device, args.span), decoder)
        base_pixels = {sigma: decode_latent(session, latent.to(session.device), decoder) for sigma, latent in base.items()}
        if labels:
            # Stamped on every panel, so a frame stays readable however the video is cropped
            # or which panel someone is looking at.
            target = _stamp(target, labels)
            base_pixels = {sigma: _stamp(pixels, labels) for sigma, pixels in base_pixels.items()}
        # Skip the frozen-base entry of sessions_and_outputs here: its own video would be
        # GT | frozen base | frozen base, which is redundant with base_pixels already being
        # one of the three panels in every LoRA checkpoint's video below.
        for checkpoint, (_unused_session, outputs) in zip(checkpoints, sessions_and_outputs[1:], strict=True):
            for sigma, latent in outputs.items():
                candidate = decode_latent(session, latent.to(session.device), decoder)
                if labels:
                    candidate = _stamp(candidate, labels)
                path = args.output / f"{_checkpoint_name(checkpoint)}_sigma_{sigma:.6f}.mp4"
                t3_video(target, base_pixels[sigma], candidate, path, fps=chain.fps)
                written.append(path)
    (args.output / "manifest.json").write_text(json.dumps({
        "kind": "d0_gt_renoise_probe",
        "attention": "block_causal",
        "source": chain.source,
        "span": args.span,
        "blocks": chain.blocks if args.span == "chain" else "whole_clip",
        "geometry": geometry.as_dict(),
        "fps": chain.fps,
        "seed": args.seed,
        "sigmas": list(PROBE_SIGMAS),
        "teacher_forcing": args.teacher_forcing,
        "layout": "ground_truth_capture | frozen_base_generated | checkpoint_generated",
        "frame_labels": not args.no_frame_labels,
        "blocks_rolled_out": [list(span) for span in plan],
        "latent_frames_covered": plan[-1][1] if plan else 0,
        "videos": [str(path.name) for path in written],
    }, indent=2) + "\n")
    for path in written:
        print(path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
