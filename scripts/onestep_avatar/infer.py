"""Generate D1 output from a guide and supplied image; see doc/infer.md."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from scripts.onestep_avatar import evaluate, media
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone, bidirectional, causal, common
from scripts.onestep_avatar.model.sampling import validate_schedule
from scripts.onestep_avatar.training import checkpoints
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings


def check_inputs(guide: torch.Tensor, first_image: torch.Tensor, guide_record: dict, image_record: dict) -> None:
    """Reject mismatched producer conditions; the supplied image is a separate input."""
    if guide.ndim != 5 or first_image.ndim != 5 or guide.shape[0] != 1 or first_image.shape[2] != 1:
        raise ValueError("guide and image must be B,C,F,H,W; image contains exactly one encoded frame")
    if guide.shape[:2] != first_image.shape[:2] or guide.shape[3:] != first_image.shape[3:]:
        raise ValueError("guide and supplied-image encoded dimensions differ")
    if not torch.isfinite(guide).all() or not torch.isfinite(first_image).all():
        raise ValueError("guide and supplied image must be finite")
    for field in ("objective", "fps", "box_xyxy", "edge", "vae_fingerprint", "encode_contract_version"):
        if field not in guide_record or field not in image_record or guide_record[field] != image_record[field]:
            raise ValueError(f"guide and supplied-image {field} differs or is missing")
    if image_record.get("input_role") != "supplied_image":
        raise ValueError("c0 must be explicitly encoded from the supplied image")
    if image_record.get("pixel_frames") != 1:
        raise ValueError("supplied-image c0 must encode one RGB image, not a video prefix")
    if guide_record["objective"] not in dataset.OBJECTIVES:
        raise ValueError("unsupported product background objective")


@torch.no_grad()
def generate(  # noqa: PLR0912, PLR0913 -- explicit product inputs and mode dispatch
    transformer: torch.nn.Module,
    context: torch.Tensor,
    grid: common.ClipGrid,
    guide: torch.Tensor,
    first_image: torch.Tensor,
    *,
    mode: str,
    settings: BidirectionalSettings | CausalSettings,
    schedule: list[float],
    seed: int,
    epsilon: torch.Tensor | None = None,
    predict_x0=None,  # noqa: ANN001 -- native or guided predictor
) -> tuple[torch.Tensor, dict]:
    """Sample from guide/c0 tokens without inventing a capture target or quality score."""
    levels = list(validate_schedule(schedule))
    if mode not in ("bidirectional", "causal"):
        raise ValueError("product generation requires an explicit supported mode")
    expected = BidirectionalSettings if mode == "bidirectional" else CausalSettings
    if not isinstance(settings, expected):
        raise ValueError("product mode and settings disagree")
    if isinstance(settings, CausalSettings) and settings.teacher_forcing:
        raise ValueError("product generation has no capture past frames")
    if guide.ndim != 3 or guide.shape != (1, grid.latent_frames * grid.tokens_per_latent_frame, guide.shape[-1]):
        raise ValueError("guide token shape differs from its grid")
    if first_image.shape != (1, grid.tokens_per_latent_frame, guide.shape[-1]):
        raise ValueError("supplied image must contain exactly one frame of tokens")
    if predict_x0 is None:
        predict_x0 = common.denoised_from_x0_model(transformer)
    geometry = None
    if mode == "causal":
        geometry = causal.CausalGeometry(
            grid.tools.scale_factors, settings.block_latent_frames, settings.context_latent_frames
        )
    if epsilon is None:
        if geometry is None:
            epsilon = common.epsilon_block(guide, seed)
        else:
            epsilon = torch.zeros_like(guide)
            for index, span in enumerate(geometry.plan(grid.latent_frames)):
                interval = slice(*grid.token_span(*span))
                epsilon[:, interval] = common.epsilon_block(guide[:, interval], seed + index)
    if epsilon.shape != guide.shape or not torch.isfinite(epsilon).all():
        raise ValueError("product noise must be finite and match guide tokens")
    with evaluate.measure_calls(transformer) as measured:
        if mode == "bidirectional":
            tokens, counts = bidirectional.sample(
                predict_x0, context, grid, guide, first_image, schedule=levels, seed=seed, epsilon=epsilon
            )
            frames = grid.latent_frames
        else:
            tokens, counts = causal.sample(
                predict_x0,
                context,
                grid,
                guide,
                first_image,
                transformer=transformer,
                geometry=geometry,
                schedule=levels,
                seed=seed,
                epsilon=epsilon,
            )
            frames = geometry.plan(grid.latent_frames)[-1][1]
    counts.update(measured)
    output = grid.unpatchify_block(tokens[:, : frames * grid.tokens_per_latent_frame], frames).cpu()
    return output, {
        "schema_version": 2,
        "mode": mode,
        "mode_settings": asdict(settings),
        "guide_mode": "d1",
        "global_sigma_dtype": common.SIGMA_PRECISION,
        "schedule": levels,
        "seed": seed,
        "frames": frames,
        "guide_sha256": evaluate.tensor_sha256(guide),
        "c0_sha256": evaluate.tensor_sha256(first_image),
        "noise_sha256": evaluate.tensor_sha256(epsilon),
        "text_sha256": evaluate.tensor_sha256(context),
        "call_counts": counts,
        "capture_reference": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("bidirectional", "causal"))
    parser.add_argument("--guide-mode", choices=("d1",), default="d1")
    for flag in ("guide", "first-image", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", default="2.5")
    parser.add_argument("--variant", choices=backbone.VARIANTS, default=backbone.DEFAULT_VARIANT)
    parser.add_argument("--schedule", type=float, nargs="+", required=True)
    parser.add_argument("--span-latent-frames", type=int)
    for flag in ("block-latent-frames", "blocks-per-sample", "context-latent-frames"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--review", action="store_true", help="save decoded image/guide/output review panels")
    parser.add_argument("--poster-frame", type=int, default=0)
    args = parser.parse_args(argv)
    if args.review and not args.decode:
        parser.error("--review requires --decode")
    args.schedule = list(validate_schedule(args.schedule))
    if args.span_latent_frames is not None and args.span_latent_frames < 1:
        parser.error("segment length must be positive")
    if args.mode == "bidirectional":
        if any(
            getattr(args, field) is not None
            for field in (
                "block_latent_frames",
                "blocks_per_sample",
                "context_latent_frames",
            )
        ):
            parser.error("bidirectional product mode rejects block/cache options")
        args.mode_settings = BidirectionalSettings(args.span_latent_frames)
    else:
        args.mode_settings = CausalSettings(
            2 if args.block_latent_frames is None else args.block_latent_frames,
            3 if args.blocks_per_sample is None else args.blocks_per_sample,
            8 if args.context_latent_frames is None else args.context_latent_frames,
            False,
            None,
        )
        if args.mode_settings.block_latent_frames < 1 or args.mode_settings.blocks_per_sample < 1:
            parser.error("block length and K must be positive")
        if not 0 <= args.mode_settings.context_latent_frames <= causal.MAX_CONTEXT_LATENT_FRAMES:
            parser.error("history depth is outside the supported range")
    return args


def prepare_product(args: argparse.Namespace) -> tuple:  # noqa: PLR0912 -- ordered input and calibration gates
    """Check files and exact adapter conditions without opening any model session."""
    guide_record = torch.load(args.guide, map_location="cpu", weights_only=True)
    image_record = torch.load(args.first_image, map_location="cpu", weights_only=True)
    guide, fps = dataset.load_training_master(args.guide, bundle=guide_record)
    image, _ = dataset.load_training_master(args.first_image, bundle=image_record)
    guide, image = guide.unsqueeze(0), image.unsqueeze(0)
    check_inputs(guide, image, guide_record, image_record)
    args.input_files = {
        "guide": {"path": str(args.guide), "sha256": sha256(args.guide)},
        "first_image": {"path": str(args.first_image), "sha256": sha256(args.first_image)},
    }
    if args.decode and fps != int(fps):
        raise ValueError("native MP4 writer requires an integer playback rate")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("product output is already used; choose a new directory")
    frames = guide.shape[2] if args.span_latent_frames is None else args.span_latent_frames
    if not 1 <= frames <= guide.shape[2]:
        raise ValueError("product segment does not fit the guide")
    specification = backbone.resolve(args.model, args.variant)
    if guide.shape[1] != specification.caps.latent_channels:
        raise ValueError("product channels differ from base model")
    if args.mode == "causal":
        geometry = causal.CausalGeometry(
            specification.scale_factors,
            args.mode_settings.block_latent_frames,
            args.mode_settings.context_latent_frames,
        )
        plan = geometry.plan(frames)
        if not plan:
            raise ValueError("product input has no complete causal block")
        frames = plan[-1][1]
        if args.checkpoint is not None and args.mode_settings.span_latent_frames is None:
            from scripts.onestep_avatar.training.checkpoints import read_contract  # noqa: PLC0415 -- header preflight

            contract = read_contract(args.checkpoint)
            if contract["mode"] == "causal" and contract["mode_settings"]["span_latent_frames"] == frames:
                # Coverage and training selection are separate; represent an
                # explicit calibration only when the actual input equals it.
                args.mode_settings = replace(args.mode_settings, span_latent_frames=frames)
    if ((frames - 1) * specification.scale_factors.time + 1) / fps > common.MAX_ROPE_SECONDS:
        raise ValueError("product guide exceeds the model position limit")
    if args.variant == "distilled" and any(level not in specification.sigmas for level in args.schedule[:-1]):
        raise ValueError("product schedule is outside the distilled base grid")
    from scripts.onestep_avatar.corpus.precompute import (  # noqa: PLC0415 -- reuse producer VAE identity
        file_fingerprint,
    )

    if guide_record["vae_fingerprint"] != file_fingerprint(Path(specification.paths.video_vae())):
        raise ValueError("product VAE differs from the recorded encoding VAE")
    if args.decode and not 0 <= args.poster_frame < (frames - 1) * specification.scale_factors.time + 1:
        raise ValueError("poster frame is outside the generated range")
    base = backbone.identity(specification.paths.transformer(), args.variant, args.model, full_hash=True)
    requested = {
        "application_method": "peft_unmerged_fp32",
        "global_sigma_dtype": common.SIGMA_PRECISION,
        "mode": args.mode,
        "mode_settings": asdict(args.mode_settings),
        "schedule": args.schedule,
        "model": {"version": args.model, "variant": args.variant, "base_sha256": base["base_transformer_sha256"]},
        "task": {
            "guide_mode": "d1",
            "objective": guide_record["objective"],
            "first_frame_conditioning": "clean_c0_v1",
            "loss": "full_frame_x0_mse",
        },
        "shape": {"channels": guide.shape[1], "height": guide.shape[3], "width": guide.shape[4], "frames": frames},
    }
    if args.mode == "causal":
        requested.update(history_mode="cache", kv_source="refresh")
    checked = evaluate.check_adapter(args.checkpoint, requested, product=True)
    return specification, guide[:, :, :frames], image, fps, requested, checked


def render_review(
    session,  # noqa: ANN001 -- native external session handle
    decoder,  # noqa: ANN001 -- native external decoder handle
    generated: torch.Tensor,
    guide: torch.Tensor,
    image: torch.Tensor,
    *,
    completed: dict,
    args: argparse.Namespace,
    vae_hash: str,
) -> dict:
    """Decode checked product inputs and render review panels without generation."""
    settings = media.native_decoder_settings()
    guide_pixels = media.decode(session, guide, decoder, args.seed)
    image_pixels = media.decode(session, image, decoder, args.seed)
    frames = tuple(range(len(generated)))
    panels = [
        media.Panel("first_image", "Input image", image_pixels, still=True, value="VAE still"),
        media.Panel("guide", "Guide", guide_pixels, frames, value="VAE decoded"),
        media.Panel("generated", "Generated", generated, frames),
    ]
    decoded_inputs = {
        role: media.decode_key(
            evaluate.tensor_sha256(latent), vae_hash, list(latent.shape), "native_decode_video", args.seed, settings
        )
        for role, latent in (("first_image", image), ("guide", guide))
    }
    generated_key = media.decode_key(
        completed["output"]["sha256"],
        vae_hash,
        completed["output"]["shape"],
        "native_decode_video",
        args.seed,
        settings,
    )
    pixels, record = media.render_panels(
        panels,
        question="What does the guide produce?",
        layout="inference",
        fps=completed["fps"],
        poster_frame=args.poster_frame,
        common_settings={
            "capture_reference": None,
            "input_files": completed["input_files"],
            "raw_output": completed["output"],
            "decoder_settings": settings,
            "decoded_inputs": decoded_inputs,
            "generated_decode_key": generated_key,
            "mode": completed["mode"],
            "schedule": completed["schedule"],
        },
    )
    if "software" in completed:
        record["software"] = completed["software"]
    return media.save_render(pixels, record, args.output / "review")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.onestep_avatar.execution import software  # noqa: PLC0415

    producer_software = software.capture("inference", args.mode, decoder=args.decode)
    specification, guide, image, fps, requested, checked = prepare_product(args)
    if args.dry_run:
        print(json.dumps({"conditions": requested, "adapter": checked}, indent=2))  # noqa: T201 -- requested dry run
        return 0
    if args.checkpoint is not None:
        checkpoints.recheck_adapter(args.checkpoint, checked["contract"], checked["adapter_sha256"])
    from scripts.prune.core import preflight  # noqa: PLC0415
    from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE, Session  # noqa: PLC0415
    from scripts.prune.data import prompt_cache  # noqa: PLC0415

    software.check_current(producer_software)
    preflight.check(args.model, gpu_id=args.gpu_id, transformer_path=specification.paths.transformer())
    device = torch.device(f"cuda:{args.gpu_id}")
    context = prompt_cache.get_or_build(
        specification, DEFAULT_PROMPT if args.prompt is None else args.prompt, DTYPE, device
    )
    session = Session(specification, device, "onestep_avatar.infer", context)
    grid = common.ClipGrid.build(
        guide.shape[2],
        guide.shape[3] * specification.scale_factors.height,
        guide.shape[4] * specification.scale_factors.width,
        fps,
        specification,
        device=device,
        dtype=DTYPE,
        latent_channels=specification.caps.latent_channels,
    )
    guide_tokens = grid.patchify(guide.to(device=device, dtype=DTYPE))
    image_tokens = grid.patchify(image.to(device=device, dtype=DTYPE))
    from scripts.onestep_avatar.model.adapters import inference_transformer  # noqa: PLC0415

    software.check_current(producer_software)
    if args.checkpoint is not None:
        checkpoints.recheck_adapter(args.checkpoint, checked["contract"], checked["adapter_sha256"])
    with inference_transformer(session, args.checkpoint, checked.get("contract"),
                               adapter_sha256=checked.get("adapter_sha256")) as transformer:
        output, record = generate(
            transformer,
            context,
            grid,
            guide_tokens,
            image_tokens,
            mode=args.mode,
            settings=args.mode_settings,
            schedule=args.schedule,
            seed=args.seed,
        )
    del transformer
    record.update(
        conditions=requested,
        fps=fps,
        input_files=args.input_files,
        software=producer_software,
        **checked,
    )
    software.check_current(producer_software)
    completed = evaluate.save_case(output, record, args.output)
    if args.decode:
        from ltx_trainer.video_utils import save_video  # noqa: PLC0415

        vae_path = Path(specification.paths.video_vae())
        vae_hash = sha256(vae_path)
        decoder_settings = media.native_decoder_settings()
        software.check_current(producer_software)
        with session.decoder() as decoder:
            pixels = media.decode(session, output, decoder, args.seed)
            if args.review:
                render_review(
                    session,
                    decoder,
                    pixels,
                    guide,
                    image,
                    completed=completed,
                    args=args,
                    vae_hash=vae_hash,
                )
        video, poster = args.output / "generated.mp4", args.output / "poster.png"
        software.check_current(producer_software)
        save_video(pixels, video, fps=fps, video_format="FCHW")
        media.frame(pixels, args.poster_frame).save(poster)
        rendering = {
            "schema_version": 2,
            "kind": "onestep_avatar.generated_render",
            "fps": fps,
            "frames": len(pixels),
            "poster_frame": args.poster_frame,
            "capture_reference": None,
            "software": producer_software,
            "decode_key": media.decode_key(
                completed["output"]["sha256"],
                vae_hash,
                list(output.shape),
                "native_decode_video",
                args.seed,
                decoder_settings,
            ),
            "decoder": {
                "vae_path": str(vae_path),
                "vae_sha256": vae_hash,
                "seed": args.seed,
                "settings": decoder_settings,
                "torch": torch.__version__,
            },
            "outputs": {
                "video": {"path": str(video), "sha256": sha256(video)},
                "poster": {"path": str(poster), "sha256": sha256(poster)},
            },
        }
        software.check_current(producer_software)
        dataset.atomic_write(
            args.output / "rendering.json",
            lambda temporary: temporary.write_text(json.dumps(rendering, indent=2) + "\n"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
