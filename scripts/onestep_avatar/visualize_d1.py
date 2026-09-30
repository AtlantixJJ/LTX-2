"""Compare capture and render sourced causal rollouts with one optional LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.transformer.attention import attention_label
from ltx_core.model.transformer.transformer import DEFAULT_TRANSFORMER_OPS
from scripts.onestep_avatar import causal_core, dataset, visualize_d0
from scripts.onestep_avatar.train import Chain, _load_training_master, clip_grid_for
from scripts.prune.core import provenance, refine_task
from scripts.prune.core.session import DTYPE, add_model_args, add_prompt_args, open_session, resolve_prompt
from scripts.prune.evaluate.metrics import t3_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view", type=Path, action="append", required=True, help="Corpus view directory; repeat for each video."
    )
    parser.add_argument("--checkpoint", type=Path, help="ComfyUI LoRA safetensors; omit for the frozen base.")
    parser.add_argument("--objective", choices=dataset.OBJECTIVES, default="white")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=list(visualize_d0.PROBE_SIGMAS),
        help="Distilled schedule levels; defaults to 0.909375, 0.725, 0.421875.",
    )
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument(
        "--trajectory-only", action="store_true", help="Run the official distilled tail, skipping the one-step case."
    )
    parser.add_argument(
        "--history-mode",
        choices=("cache", "recompute", "joint"),
        default="cache",
        help="Use cached history, an explicit causal prefix, or a bidirectional clean window.",
    )
    parser.add_argument(
        "--whole-clip",
        action="store_true",
        help="One bidirectional block over the whole clip (block = latent frames - 1); no K/V cache is allocated.",
    )
    parser.add_argument("--block-latent-frames", type=int, default=causal_core.BLOCK_LATENT_FRAMES)
    parser.add_argument("--context-latent-frames", type=int, default=causal_core.CONTEXT_LATENT_FRAMES)
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Stop after this many blocks, for a short real-checkpoint diagnostic.",
    )
    parser.add_argument(
        "--variant",
        choices=("distilled", "dev"),
        default="distilled",
        help="dev loads the base transformer and samples each sigma with a rescaled --steps schedule.",
    )
    parser.add_argument("--transformer", type=Path, default=None, help="Explicit transformer checkpoint path.")
    parser.add_argument("--steps", type=int, default=None, help="dev only: steps per start sigma (rescaled schedule).")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale; 1 disables the negative-prompt pass.")
    parser.add_argument("--stg", type=float, default=0.0, help="STG scale; 0 disables the perturbed pass.")
    parser.add_argument("--stg-blocks", type=int, nargs="+", default=[28])
    parser.add_argument("--rescale", type=float, default=0.0, help="Guidance rescale (pipelines use 0.7 with CFG).")
    parser.add_argument("--negative-prompt", default=None, help="Default: ltx_pipelines DEFAULT_NEGATIVE_PROMPT.")
    parser.add_argument("--raw-only", action="store_true", help="Save raw latents and manifest without VAE decoding.")
    parser.add_argument("--output", type=Path, required=True)
    add_model_args(parser)
    add_prompt_args(parser)
    args = parser.parse_args(argv)
    if args.whole_clip and (
        args.history_mode != "cache"
        or args.block_latent_frames != causal_core.BLOCK_LATENT_FRAMES
        or args.max_blocks is not None
    ):
        parser.error("--whole-clip sets the block and history itself; drop --history-mode/--block-latent-frames/--max-blocks")
    if args.variant == "dev":
        if args.steps is None or args.steps < 1:
            parser.error("--variant dev needs --steps N")
        if args.trajectory_only:
            parser.error("--trajectory-only walks the distilled grid; dev uses --steps instead")
    elif args.steps is not None:
        parser.error("--steps is dev only; the distilled model uses its own grid (--trajectory-only)")
    return args


DEV_TRANSFORMER = "ltx-2.5-22b-dev-transformer-bf16.safetensors"


def _suffix(args: argparse.Namespace) -> str:
    if args.variant == "dev":
        return f"dev_n{args.steps}_cfg{args.cfg:g}_stg{args.stg:g}"
    return "official" if args.trajectory_only else "one_step"


def whole_clip_block(latent_frame_counts: list[int]) -> int:
    """The block size that makes block 0 ``[0, T)`` cover every latent frame of every view.

    Block 0 absorbs the keyframe, so it spans ``1 + block`` frames. All views must share ``T``:
    one geometry serves the whole invocation, and a shorter view would silently drop no frames
    but a longer one would lose its tail.
    """
    counts = set(latent_frame_counts)
    if len(counts) != 1:
        raise SystemExit(f"--whole-clip needs equal-length views, got latent frame counts {sorted(counts)}")
    (count,) = counts
    if count < 2:
        raise SystemExit("--whole-clip needs at least two latent frames")
    return count - 1


def _load_chain(view: Path, objective: str) -> Chain:
    capture = view / dataset.capture_bundle_name(objective)
    guide = view / dataset.guide_bundle_name(objective)
    for path in (capture, guide):
        if not path.is_file():
            raise SystemExit(f"missing master latent: {path}")
    z_y, fps = _load_training_master(capture)
    z_g, guide_fps = _load_training_master(guide)
    if z_g.shape != z_y.shape or guide_fps != fps:
        raise SystemExit(f"capture/guide shape or fps mismatch: {view}")
    return Chain(str(view), "probe", view.parent.parent.name, True, [], z_g, z_y, fps, None)


def _global_epsilons(
    source: torch.Tensor, grid: causal_core.ClipGrid, plan: list[tuple[int, int]], seed: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Draw noise once over global latent-frame indices, independent of block geometry."""
    epsilon = causal_core.epsilon_block(source, seed)
    return epsilon, [epsilon[:, slice(*grid.token_span(*span))] for span in plan]


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory must be fresh and empty: {args.output}")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    chains = [(view, _load_chain(view, args.objective)) for view in args.view]
    if args.whole_clip:
        args.block_latent_frames = whole_clip_block([chain.z_y.shape[1] for _, chain in chains])
        # One block has no history: the explicit-history path with an empty prefix is the same
        # computation as the cache path, minus the cache allocation and the refresh pass.
        args.history_mode = "recompute"
    prompt = resolve_prompt(args)
    transformer_path = args.transformer
    if transformer_path is None and args.variant == "dev":
        from scripts.prune.core.model_registry import resolve

        transformer_path = Path(resolve(args.model).paths.transformer()).with_name(DEV_TRANSFORMER)
    if transformer_path is not None and not Path(transformer_path).is_file():
        raise SystemExit(f"transformer checkpoint does not exist: {transformer_path}")
    session = open_session(
        args, script="onestep_avatar.visualize_d1", prompt=prompt, transformer_path=transformer_path
    )
    if args.variant == "dev":
        # The dev model has no distilled grid; any start in (0, 1] is a valid operating point.
        if not args.sigmas or any(not 0.0 < s <= 1.0 for s in args.sigmas) or len(set(args.sigmas)) != len(args.sigmas):
            raise SystemExit(f"dev sigmas must be distinct values in (0, 1], got {args.sigmas}")
        sigmas = tuple(args.sigmas)
    else:
        sigmas = visualize_d0._probe_sigmas(args.sigmas, session.model.sigmas)
    guider, negative_context, negative_prompt = None, None, None
    if args.cfg != 1.0 or args.stg != 0.0 or args.rescale != 0.0:
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
        from scripts.prune.data import prompt_cache

        if args.cfg != 1.0:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT if args.negative_prompt is None else args.negative_prompt
            negative_context = prompt_cache.get_or_build(session.model, negative_prompt, DTYPE, session.device)
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=args.cfg, stg_scale=args.stg, stg_blocks=list(args.stg_blocks), rescale_scale=args.rescale
            ),
            negative_context=negative_context,
        )
    geometry = causal_core.deployed_geometry(
        session.model.scale_factors,
        block_latent_frames=args.block_latent_frames,
        context_latent_frames=args.context_latent_frames,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    context_bytes = session.context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    model_stamp = session.stamp(dtype=str(DTYPE))
    model_stamp["video_vae_fingerprint"] = provenance.checkpoint_fingerprint(session.model.paths.video_vae())
    model_stamp["checkpoint"] = (
        None
        if args.checkpoint is None
        else {
            "path": str(args.checkpoint.resolve()),
            "fingerprint": provenance.checkpoint_fingerprint(args.checkpoint),
        }
    )
    loras = (
        ()
        if args.checkpoint is None
        else (LoraPathStrengthAndSDOps(str(args.checkpoint), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)
    )
    passes_per_step = 1 + (args.cfg != 1.0) + (args.stg != 0.0)
    results = []
    manifest = {
        "kind": "d1_paired_source_probe",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "objective": args.objective,
        "sigmas": list(sigmas),
        "trajectory_only": args.trajectory_only,
        "seed": args.seed,
        "teacher_forcing": args.teacher_forcing,
        # A single whole-clip block has no history, so the teacher-forcing flag cannot act.
        "history_policy": "none_single_block"
        if args.whole_clip
        else "real_capture"
        if args.teacher_forcing
        else "generated_output",
        "history_mode": args.history_mode,
        "geometry": geometry.as_dict(),
        "noise": {
            "scheme": "one global epsilon tensor per view, sliced by latent-frame block",
            "shared_across_arms_sigmas_and_block_geometries": True,
        },
        "text_context": {
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "is_default_prompt": prompt == refine_task.REFINE_PROMPT,
            "sha256": hashlib.sha256(context_bytes).hexdigest(),
            "shape": list(session.context.shape),
            "dtype": str(session.context.dtype),
        },
        "model": model_stamp,
        "capabilities": asdict(session.model.caps),
        "whole_clip": args.whole_clip,
        "model_variant": args.variant,
        "steps": args.steps,
        "schedule_policy": "rescaled_ltx2_scheduler" if args.variant == "dev" else "distilled_grid",
        "guidance": {
            "cfg": args.cfg,
            "stg": args.stg,
            "stg_blocks": list(args.stg_blocks) if args.stg != 0.0 else [],
            "rescale": args.rescale,
            "negative_prompt": negative_prompt,
            "negative_prompt_sha256": None
            if negative_prompt is None
            else hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest(),
            "passes_per_step": passes_per_step,
            "execution": "sequential passes",
        },
        "attention": "full_bidirectional"
        if args.whole_clip
        else "bidirectional_clean_history_window"
        if args.history_mode == "joint"
        else "block_causal",
        "attention_backend": {
            "self": attention_label(DEFAULT_TRANSFORMER_OPS.attention_ops.attention_function),
            "masked": attention_label(DEFAULT_TRANSFORMER_OPS.attention_ops.masked_attention_function),
        },
        "latent_dtype": str(DTYPE),
        "max_blocks": args.max_blocks,
        "raw_only": args.raw_only,
        "videos": [],
    }
    with session.transformer(loras=loras) as transformer:
        for view, chain in chains:
            grid = clip_grid_for(
                chain, geometry, device=session.device, latent_channels=session.model.caps.latent_channels
            )
            plan = visualize_d0._plan_for(chain, geometry, grid, "clip")
            if args.max_blocks is not None:
                if not 1 <= args.max_blocks <= len(plan):
                    raise SystemExit(f"--max-blocks must be within [1, {len(plan)}]")
                plan = plan[: args.max_blocks]
            source = grid.patchify(chain.z_y.unsqueeze(0).to(device=session.device, dtype=DTYPE))
            global_epsilon, epsilons = _global_epsilons(source, grid, plan, args.seed)
            stem = f"{view.parent.parent.name}_{view.name}"
            noise_path = args.output / f"{stem}_epsilon.pt"
            torch.save(global_epsilon.cpu(), noise_path)
            view_record = {
                "view": str(view.resolve()),
                "fps": chain.fps,
                "capture": str((view / dataset.capture_bundle_name(args.objective)).resolve()),
                "guide": str((view / dataset.guide_bundle_name(args.objective)).resolve()),
                "capture_sha256": provenance.file_sha256(view / dataset.capture_bundle_name(args.objective)),
                "guide_sha256": provenance.file_sha256(view / dataset.guide_bundle_name(args.objective)),
                "epsilon": noise_path.name,
                "epsilon_sha256": provenance.file_sha256(noise_path),
                "blocks": [list(block) for block in plan],
            }
            arms = {}
            for sigma in sigmas:
                schedule = None
                if args.variant == "dev":
                    schedule = list(
                        causal_core.rescaled_schedule(sigma, args.steps)
                    )
                elif args.trajectory_only:
                    schedule = [float(level) for level in session.model.sigmas if level <= sigma + 1e-9]
                    causal_core.validate_schedule(schedule, session.model.sigmas)
                    if len(schedule) == 2:
                        continue  # the final grid level already has a one-step trajectory
                arms[sigma] = {}
                for mode in ("d0", "d1"):
                    if session.device.type == "cuda":
                        torch.cuda.synchronize(session.device)
                        torch.cuda.reset_peak_memory_stats(session.device)
                    started = time.perf_counter()
                    _, latent = visualize_d0._run_chain(
                        transformer,
                        session.context,
                        chain,
                        geometry,
                        sigma,
                        device=session.device,
                        latent_channels=session.model.caps.latent_channels,
                        seed=args.seed,
                        guide_mode=mode,
                        teacher_forcing=args.teacher_forcing,
                        block_epsilons=epsilons,
                        schedule=schedule,
                        history_mode=args.history_mode,
                        max_blocks=args.max_blocks,
                        guider=guider,
                        negative_context=negative_context,
                    )
                    if session.device.type == "cuda":
                        torch.cuda.synchronize(session.device)
                    view_record.setdefault("rollout_timing", []).append(
                        {
                            "sigma": sigma,
                            "arm": mode,
                            "wall_seconds": time.perf_counter() - started,
                            "schedule": schedule if schedule is not None else [sigma, 0.0],
                            "forward_passes": (len(schedule) - 1 if schedule else 1) * passes_per_step,
                            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(session.device)
                            if session.device.type == "cuda"
                            else None,
                        }
                    )
                    arms[sigma][mode] = latent.cpu()
                    latent_path = (
                        args.output
                        / f"{stem}_sigma_{sigma:.6f}_{_suffix(args)}_{mode}.pt"
                    )
                    torch.save(arms[sigma][mode], latent_path)
                    view_record.setdefault("latents", []).append(
                        {
                            "sigma": sigma,
                            "arm": mode,
                            "path": latent_path.name,
                            "sha256": provenance.file_sha256(latent_path),
                        }
                    )
            results.append((view, chain, plan, arms, view_record))
            print(f"Rolled out {view}", flush=True)  # noqa: T201
    with nullcontext(None) if args.raw_only else session.decoder() as decoder:
        for view, chain, plan, arms, view_record in results:
            target_pixels = None
            if decoder is not None:
                target = chain.z_y.unsqueeze(0)[:, :, : plan[-1][1]]
                target_pixels = visualize_d0._decode(session, target, decoder, args.seed)
            for sigma in arms:
                suffix = _suffix(args)
                output = None
                if decoder is not None:
                    panels = [target_pixels] + [
                        visualize_d0._decode(session, arms[sigma][mode], decoder, args.seed) for mode in ("d0", "d1")
                    ]
                    output = args.output / f"{view.parent.parent.name}_{view.name}_sigma_{sigma:.6f}_{suffix}.mp4"
                    t3_video(*panels, output, fps=chain.fps)
                manifest["videos"].append(
                    {
                        "view": str(view),
                        "sigma": sigma,
                        "trajectory": suffix,
                        "schedule": next(
                            t["schedule"] for t in view_record["rollout_timing"] if t["sigma"] == sigma
                        ),
                        "output": str(output) if output is not None else None,
                        "panels": ["ground_truth", "gt_latent_rollout", "rgb_render_latent_rollout"],
                        "blocks": [list(block) for block in plan],
                        "artifacts": view_record,
                    }
                )
                if output is not None:
                    print(output, flush=True)  # noqa: T201
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
