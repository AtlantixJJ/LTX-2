"""Decode a configurable renoise probe of either arm at distilled schedule levels.

``--guide-mode d0`` (the default) noises the capture latent itself.  D0 is deliberately not
deployable, which makes it a useful capacity control, but only if its review artifact uses
that exact state rather than a guide-noised approximation.  ``--guide-mode d1`` noises the
ARGAvatar guide ``z_g`` instead -- the deployable arm, whose checkpoints had no probe at all
until this flag existed (``doc/known_gaps.md`` G4).  The arms differ in that **one tensor**;
the target reference and the clean first-frame condition ``c0`` are the capture in both, and
both go through the same ``causal_core.rollout``.

It runs one rollout per ``--probe-sigmas`` value and writes portable MP4s and raw latents.
Checkpoint mode writes one comparison per level:

    ground-truth capture | frozen base | LoRA checkpoint

Each rollout covers the **whole clip** by default (``--span clip``): block 0 through the last
full block, the same sequence an inference run produces, so error accumulated across the AR
rollout is visible rather than truncated at the training chain's ``K`` blocks. ``--span chain``
restores the old behaviour of covering only the subset chain's blocks.

The fixed clip and seeds make a sequence of checkpoints directly comparable.  It is an
offline checkpoint probe, so no VAE is resident while FSDP training is stepping.

For a matched frozen-base experiment, ``--base-only`` removes the checkpoint requirement.
The probe materializes one epsilon tensor per block, reuses it for every sigma arm, and saves
the tensors beside the outputs. ``--block-latent-frames`` and ``--context-latent-frames``
make the rollout geometry explicit.

**Revised 2026-09-14 (SS4.4).** The probe rolls out through ``causal_core`` -- block-causal
attention plus the clean-latent K/V cache -- exactly as training and deployment do, so the
probe cannot silently diverge from either. It does not call :mod:`onestep_core`, which refuses
D0 on purpose to protect deployment from accepting an arm that needs the unavailable capture
latent -- and this tool has to be able to run both arms through one code path.

**Revised 2026-09-21.** ``--guide-mode d1`` added, closing the "the arm that deploys cannot be
inspected" half of G4.  Teacher forcing now passes the capture target explicitly rather than
letting the refresh read whatever the block was noised from, which was right for D0 and silently
wrong for D1 (G2).

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

python -m scripts.onestep_avatar.visualize_d0 \
    --subset ../expr/onestep_avatar/windows/t2r2.json \
    --run ../expr/onestep_avatar/runs/white-d0-tf-c0-debug \
    --steps 0 1 \
    --output ../expr/onestep_avatar/runs/white-d0-tf-c0-debug/probes/init \
    --teacher-forcing \
    --gpu-id 1

Pass ``--teacher-forcing`` when the run itself was trained with ``train.py --teacher-forcing``
(check the run's ``config.json``): it refreshes the rollout's cache from the ground-truth
capture instead of the checkpoint's own denoised output, the same ablation ``train.py`` makes,
so the checkpoint is probed under the input distribution it was actually trained on rather than
the self-forced one a real deployment (and the default here) has to use.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_trainer.video_utils import save_video
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


def _source_master(chain: Chain, guide_mode: str) -> torch.Tensor:
    """The master latent the block input is noised from -- the one line the arm changes.

    D0 noises the capture itself (``z_y``), so the correspondence gap is zero and the probe
    measures capacity. D1 noises the render (``z_g``) -- the deployable arm's actual input.
    The *target* is ``z_y`` in both, and so is ``c0``: the guide's frame 0 is a render
    composite, never the supplied real first frame (``experiments.md`` §1).
    """
    if guide_mode == "d0":
        return chain.z_y
    if chain.z_g is None:
        raise SystemExit(
            "--guide-mode d1 needs the guide master z_g, which this chain did not load. "
            "Freeze the subset with --require-guide and check the guide latents are current "
            f"under GUIDE_COMPOSITING_VERSION={dataset.GUIDE_COMPOSITING_VERSION}."
        )
    return chain.z_g


def _run_chain(  # noqa: ANN202, PLR0913
    transformer,  # noqa: ANN001
    context,  # noqa: ANN001
    chain: Chain,
    geometry,  # noqa: ANN001
    sigma: float,
    *,
    device,  # noqa: ANN001
    latent_channels: int,
    seed: int,
    guide_mode: str = "d0",
    teacher_forcing: bool = False,
    schedule: list[float] | None = None,
    kv_source: str = "refresh",
    span: str = "clip",
    block_epsilons: list[torch.Tensor] | None = None,
):
    """One AR rollout of the selected arm; ``z_y`` is the target reference in both.

    Goes through ``causal_core.rollout``, the one implementation training and deployment both
    use, so the cached context, the block-causal attention, the pinned frame-0 sink and the
    RoPE positions are the deployed ones by construction. The arm changes exactly one tensor,
    the noising source (``_source_master``); ``c0`` and the teacher target stay ``z_y``.

    Teacher forcing passes ``z_y`` explicitly as ``teacher_tokens`` rather than relying on the
    old implicit refresh-from-the-noising-source, which was right for D0 and wrong for D1
    (``known_gaps.md`` G2).

    It does not call :mod:`onestep_core`: that module refuses D0 on purpose, to protect
    deployment from accepting an arm that needs the unavailable capture latent, and this probe
    has to run both arms through one code path.
    """
    grid = clip_grid_for(chain, geometry, device=device, latent_channels=latent_channels)
    base = causal_core.base_model(transformer)
    cache = causal_core.BlockCache.allocate(
        grid,
        geometry,
        num_layers=len(base.transformer_blocks),
        inner_dim=base.inner_dim,
        device=device,
        dtype=DTYPE,
    )
    z_y = grid.patchify(chain.z_y.unsqueeze(0).to(device=device, dtype=DTYPE))
    source = (
        z_y
        if guide_mode == "d0"
        else grid.patchify(_source_master(chain, guide_mode).unsqueeze(0).to(device=device, dtype=DTYPE))
    )
    plan = _plan_for(chain, geometry, grid, span)
    tokens, _ = causal_core.rollout(
        causal_core.denoised_from_x0_model(transformer),
        grid,
        geometry,
        cache,
        source,
        context,
        sigma,
        seed=seed,
        blocks=plan,
        teacher_forcing=teacher_forcing,
        teacher_tokens=z_y if teacher_forcing else None,
        schedule=schedule,
        kv_source=kv_source,
        first_frame_condition=z_y[:, : grid.tokens_per_latent_frame],
        block_epsilons=block_epsilons,
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


def _as_fchw(pixels: torch.Tensor) -> torch.Tensor:
    """Normalize decoder output to the frame-major layout used by stamping and video I/O."""
    if pixels.ndim == 5:  # B,C,T,H,W
        return pixels.permute(0, 2, 1, 3, 4).flatten(0, 1)
    if pixels.ndim == 4 and pixels.shape[-1] in (1, 3, 4):  # F,H,W,C
        return pixels.permute(0, 3, 1, 2)
    if pixels.ndim != 4:
        raise ValueError(f"expected decoded BCTHW, FCHW or FHWC video, got {tuple(pixels.shape)}")
    return pixels


def _decode(session, latent: torch.Tensor, decoder, seed: int) -> torch.Tensor:  # noqa: ANN001
    """Decode an arm with an identical fresh diffusion-decoder noise stream."""
    generator = torch.Generator(device=session.device).manual_seed(seed)
    return _as_fchw(decode_latent(session, latent.to(session.device), decoder, generator=generator))


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
    if args.base_only and checkpoints:
        raise SystemExit("--base-only cannot be combined with --checkpoint or --steps")
    if not checkpoints and not args.base_only:
        raise SystemExit("no checkpoints requested: pass --checkpoint and/or --run with --steps")
    return checkpoints


def _probe_sigmas(values: list[float], schedule: list[float]) -> tuple[float, ...]:
    """Validate informative sigma arms against the selected distilled model schedule."""
    if not values:
        raise SystemExit("--probe-sigmas requires at least one value")
    if len(set(values)) != len(values):
        raise SystemExit("--probe-sigmas contains duplicate values")
    for sigma in values:
        if sigma <= 0:
            raise SystemExit(f"probe sigma must be nonzero and positive, got {sigma}")
        if not any(abs(sigma - scheduled) < 1e-9 for scheduled in schedule):
            raise SystemExit(f"probe sigma {sigma} is not on model schedule {schedule}")
    return tuple(values)


def _block_epsilons(tokens: torch.Tensor, grid, plan: list[tuple[int, int]], seed: int) -> list[torch.Tensor]:  # noqa: ANN001
    """Materialize the rollout's established seed+block-index noise stream once."""
    return [
        causal_core.epsilon_block(tokens[:, slice(*grid.token_span(*span))], seed + index)
        for index, span in enumerate(plan)
    ]


def generate_checkpoint(args: argparse.Namespace, checkpoint: Path | None, chain: Chain):  # noqa: ANN201
    ckpt_name = "frozen base" if checkpoint is None else checkpoint.name
    print(f"--> Generating rollouts for {ckpt_name}...", flush=True)  # noqa: T201
    session = open_session(args, script="onestep_avatar.visualize_d0")
    geometry = causal_core.deployed_geometry(
        session.model.scale_factors,
        block_latent_frames=args.block_latent_frames,
        context_latent_frames=args.context_latent_frames,
    )
    sigmas = _probe_sigmas(args.probe_sigmas, session.model.sigmas)
    if args.schedule is not None:
        # Validated against the model's own grid here rather than inside the rollout, so an
        # off-grid teacher arm fails before 42 GB of weights are loaded rather than after.
        levels = causal_core.validate_schedule(args.schedule, list(session.model.sigmas))
        if len(sigmas) != 1 or abs(sigmas[0] - levels[0]) > 1e-9:
            raise SystemExit(
                f"--schedule starts at {levels[0]} but --probe-sigmas is {list(sigmas)}; a "
                "multi-step arm probes exactly the one operating point it starts from"
            )
    loras = ()
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")
        loras = (LoraPathStrengthAndSDOps(str(checkpoint), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)

    # The VAE is held only after all four model outputs have been calculated, avoiding the
    # transformer+decoder coexistence that §7.4 explicitly excludes from the train loop.
    outputs: dict[float, torch.Tensor] = {}
    grid = clip_grid_for(chain, geometry, device=session.device, latent_channels=session.model.caps.latent_channels)
    source = grid.patchify(
        _source_master(chain, args.guide_mode).unsqueeze(0).to(device=session.device, dtype=DTYPE)
    )
    plan = _plan_for(chain, geometry, grid, args.span)
    # The epsilon stream depends only on the source's shape/dtype/device, which the two arms
    # share -- but derive it from the arm's own source anyway, so a future shape divergence
    # fails here rather than silently reusing the other arm's noise.
    epsilons = _block_epsilons(source, grid, plan, args.seed)
    started = time.perf_counter()
    if session.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(session.device)
    with session.transformer(loras=loras) as transformer:
        for sigma in sigmas:
            print(f"    rolling out sigma={sigma:.6f}...", flush=True)  # noqa: T201
            _, latent = _run_chain(
                transformer,
                session.context,
                chain,
                geometry,
                sigma,
                device=session.device,
                latent_channels=session.model.caps.latent_channels,
                seed=args.seed,
                guide_mode=args.guide_mode,
                teacher_forcing=args.teacher_forcing,
                schedule=args.schedule,
                kv_source=args.kv_source,
                span=args.span,
                block_epsilons=epsilons,
            )
            outputs[sigma] = latent

    # CPU tensors keep this result cheap while the next checkpoint's transformer is loaded.
    timing = {
        "transformer_wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(session.device)
        if session.device.type == "cuda"
        else None,
    }
    return session, {sigma: latent.cpu() for sigma, latent in outputs.items()}, [eps.cpu() for eps in epsilons], timing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Defaults to the subset's own corpus_root, where the master latents live.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, action="append", default=[], help="Explicit checkpoint path; repeat for multiple."
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Probe only the frozen base; no LoRA checkpoint is required.",
    )
    parser.add_argument(
        "--probe-sigmas",
        type=float,
        nargs="+",
        default=list(PROBE_SIGMAS),
        help="Explicit nonzero sigma values from the selected model schedule.",
    )
    parser.add_argument("--block-latent-frames", type=int, default=causal_core.BLOCK_LATENT_FRAMES)
    parser.add_argument("--context-latent-frames", type=int, default=causal_core.CONTEXT_LATENT_FRAMES)
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Run directory whose checkpoints/lora_weights_step_NNNNN.safetensors --steps resolves against.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[],
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
    parser.add_argument(
        "--chain-index",
        type=int,
        default=0,
        help="Index within --split; selects the clip (and, at --span chain, the blocks).",
    )
    parser.add_argument(
        "--span",
        choices=("clip", "chain"),
        default="clip",
        help=(
            "clip (default): roll the WHOLE clip from block 0, so each video is the full "
            "inference sequence and drift across it is visible. chain: cover only the subset "
            "chain's own K blocks, the historical behaviour; requires a clip-start chain."
        ),
    )
    parser.add_argument(
        "--kv-source",
        choices=("refresh", "denoise"),
        default="refresh",
        help=(
            "What fills the K/V cache. refresh (default): a second forward on the finished "
            "block at timestep zero -- 2 forwards per block. denoise: cache what the denoising "
            "forward already computed and skip the second forward -- 1 forward per block, half "
            "the steady-state latency, at whatever quality cost the comparison shows. "
            "Incompatible with --teacher-forcing, which is defined as caching the target."
        ),
    )
    parser.add_argument(
        "--schedule",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Denoising levels for the CURRENT block, strictly decreasing and ending at 0. "
            "Default (omitted) is the one-step student: [probe sigma, 0]. Pass e.g. "
            "--schedule 0.725 0.421875 0 for the two-step causal teacher arm -- the same "
            "rollout, the same cached history, one more denoising forward per block. Every "
            "nonzero level must be on the selected model's grid. Requires a single "
            "--probe-sigmas value matching the schedule's first level."
        ),
    )
    parser.add_argument(
        "--guide-mode",
        choices=("d0", "d1"),
        default="d0",
        help=(
            "Which master the block input is noised from. d0 (default): the capture z_y -- the "
            "capacity diagnostic, not deployable. d1: the ARGAvatar guide z_g -- the deployable "
            "arm, and the one whose checkpoints could not be looked at before (known_gaps G4). "
            "The target reference and c0 are the capture in both."
        ),
    )
    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
        help=(
            "Refresh the cache from the ground-truth capture instead of the model's own "
            "denoised output, matching train.py's --teacher-forcing ablation. Use this to probe "
            "a checkpoint trained with --teacher-forcing the way it was actually trained; the "
            "default (off) is the self-forced regime a real deployment has to use."
        ),
    )
    parser.add_argument(
        "--no-frame-labels",
        action="store_true",
        help="Do not burn the per-frame 'latent N · rollout step M' caption into the video.",
    )
    add_model_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    args = parse_args(argv)
    subset = json.loads(args.subset.read_text())
    corpus_root = args.corpus_root or Path(subset["corpus_root"])
    objective = args.objective or subset.get("objective", dataset.DEFAULT_OBJECTIVE)
    # D0 never reads chain.z_g, so don't require the guide bundle to exist on disk for it
    # (see ChainStore.with_guide); D1 is the arm that needs it, and says so.
    store = ChainStore(
        subset,
        corpus_root,
        split=args.split,
        objective=objective,
        with_anchor=False,
        with_guide=args.guide_mode == "d1",
    )
    chain = _chain(store, args.chain_index, args.span)
    checkpoints = _resolve_checkpoints(args)
    args.output.mkdir(parents=True, exist_ok=True)
    sessions_and_outputs = [generate_checkpoint(args, checkpoint, chain) for checkpoint in (None, *checkpoints)]
    session, base, epsilons, base_timing = sessions_and_outputs[0]
    for _candidate_session, _outputs, candidate_epsilons, _timing in sessions_and_outputs[1:]:
        if len(candidate_epsilons) != len(epsilons) or any(
            not torch.equal(expected, actual) for expected, actual in zip(epsilons, candidate_epsilons, strict=True)
        ):
            raise RuntimeError("checkpoint arms did not reuse the frozen base's block epsilon tensors")
    geometry = causal_core.deployed_geometry(
        session.model.scale_factors,
        block_latent_frames=args.block_latent_frames,
        context_latent_frames=args.context_latent_frames,
    )
    grid = clip_grid_for(chain, geometry, device=session.device, latent_channels=session.model.caps.latent_channels)
    written = []
    latent_paths = []
    noise_path = args.output / "block_epsilons.pt"
    torch.save(
        {
            "seed": args.seed,
            "blocks": [list(span) for span in _plan_for(chain, geometry, grid, args.span)],
            "epsilons": epsilons,
        },
        noise_path,
    )
    for sigma, latent in base.items():
        path = args.output / f"frozen_base_sigma_{sigma:.6f}.pt"
        torch.save(latent, path)
        latent_paths.append(path)
    for checkpoint, (_unused_session, outputs, _unused_epsilons, _timing) in zip(
        checkpoints, sessions_and_outputs[1:], strict=True
    ):
        for sigma, latent in outputs.items():
            path = args.output / f"{_checkpoint_name(checkpoint)}_sigma_{sigma:.6f}.pt"
            torch.save(latent, path)
            latent_paths.append(path)
    # Decode once, after all transformer passes.  Every checkpoint video therefore gives the
    # requested before/after comparison in one frame-aligned artifact: GT | frozen base | LoRA.
    # No stitching any more: a causal rollout writes one latent covering the whole chain, so
    # there is no per-window overlap to drop and no seam to get wrong.
    plan = _plan_for(chain, geometry, grid, args.span)
    labels = [] if args.no_frame_labels else _frame_labels(plan, geometry.scale_factors.time)
    with session.decoder() as decoder:
        print("--> Decoding target latent...", flush=True)  # noqa: T201
        target = _decode(session, _target_latent(chain, grid, geometry, session.device, args.span), decoder, args.seed)
        print("--> Decoding base latents...", flush=True)  # noqa: T201
        base_pixels = {sigma: _decode(session, latent, decoder, args.seed) for sigma, latent in base.items()}
        capture_path = args.output / "capture.mp4"
        save_video(target, capture_path, fps=chain.fps, video_format="FCHW")
        written.append(capture_path)
        for sigma, pixels in base_pixels.items():
            path = args.output / f"frozen_base_sigma_{sigma:.6f}.mp4"
            save_video(pixels, path, fps=chain.fps, video_format="FCHW")
            written.append(path)
        if args.base_only and len(base_pixels) >= 2:
            first, second = list(base_pixels)[:2]
            path = args.output / f"capture_sigma_{first:.6f}_vs_{second:.6f}.mp4"
            t3_video(target, base_pixels[first], base_pixels[second], path, fps=chain.fps)
            written.append(path)
        if labels:
            # Stamped on every panel, so a frame stays readable however the video is cropped
            # or which panel someone is looking at.
            target = _stamp(target, labels)
            base_pixels = {sigma: _stamp(pixels, labels) for sigma, pixels in base_pixels.items()}
        # Skip the frozen-base entry of sessions_and_outputs here: its own video would be
        # GT | frozen base | frozen base, which is redundant with base_pixels already being
        # one of the three panels in every LoRA checkpoint's video below.
        for checkpoint, (_unused_session, outputs, _unused_epsilons, _timing) in zip(
            checkpoints, sessions_and_outputs[1:], strict=True
        ):
            print(f"--> Decoding and writing videos for {_checkpoint_name(checkpoint)}...", flush=True)  # noqa: T201
            for sigma, latent in outputs.items():
                candidate = _decode(session, latent, decoder, args.seed)
                if labels:
                    candidate = _stamp(candidate, labels)
                path = args.output / f"{_checkpoint_name(checkpoint)}_sigma_{sigma:.6f}.mp4"
                t3_video(target, base_pixels[sigma], candidate, path, fps=chain.fps)
                written.append(path)
                print(f"    Wrote {path}", flush=True)  # noqa: T201
    timing = {"frozen_base": base_timing}
    timing.update(
        {
            _checkpoint_name(checkpoint): checkpoint_timing
            for checkpoint, (_session, _outputs, _epsilons, checkpoint_timing) in zip(
                checkpoints, sessions_and_outputs[1:], strict=True
            )
        }
    )
    base_layout = (
        "ground_truth_capture | first_sigma | second_sigma"
        if len(base) >= 2
        else "individual ground_truth_capture and frozen_base_generated"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "d0_gt_renoise_probe",
                "attention": "block_causal",
                "guide_mode": args.guide_mode,
                "noising_source": "z_y (capture)" if args.guide_mode == "d0" else "z_g (ARGAvatar guide)",
                "source": chain.source,
                "span": args.span,
                "blocks": chain.blocks if args.span == "chain" else "whole_clip",
                "geometry": geometry.as_dict(),
                "fps": chain.fps,
                "seed": args.seed,
                "objective": objective,
                "sigmas": list(base),
                "schedule": list(args.schedule) if args.schedule else None,
                "kv_source": args.kv_source,
                "denoise_forwards_per_block": (len(args.schedule) - 1) if args.schedule else 1,
                "refresh_forwards_per_block": 1 if args.kv_source == "refresh" else 0,
                "teacher_forcing": args.teacher_forcing,
                "history_policy": "real_capture" if args.teacher_forcing else "generated_output",
                "conditioning": {"first_frame": "clean capture latent frame 0", "text": "session prompt cache"},
                "model": session.stamp(dtype=str(DTYPE)),
                "noise": {
                    "path": noise_path.name,
                    "scheme": "torch.Generator(seed + block_index)",
                    "shared_across_sigmas": True,
                },
                "decode_noise": {
                    "seed": args.seed,
                    "fresh_identical_generator_per_arm": True,
                },
                "timing": timing,
                "layout": "ground_truth_capture | frozen_base_generated | checkpoint_generated"
                if checkpoints
                else base_layout,
                "frame_labels": not args.no_frame_labels,
                "blocks_rolled_out": [list(span) for span in plan],
                "latent_frames_covered": plan[-1][1] if plan else 0,
                "videos": [str(path.name) for path in written],
                "latents": [str(path.name) for path in latent_paths],
            },
            indent=2,
        )
        + "\n"
    )
    for path in written:
        print(path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
