"""Save native whole-clip one-step D0/D1 rollouts for pruning; see doc/evaluate/whole_clip_baseline.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ltx_core.model.transformer.attention import attention_label
from ltx_core.model.transformer.transformer import DEFAULT_TRANSFORMER_OPS
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.corpus.dataset import load_training_master
from scripts.onestep_avatar.model import backbone, common
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model import sampling as model_sampling
from scripts.prune.core import native_manifest, provenance
from scripts.prune.core.session import (
    DEFAULT_PROMPT,
    DTYPE,
    add_model_args,
    add_prompt_args,
    open_session,
    resolve_prompt,
)
from scripts.prune.evaluate.decode import decode_latent
from scripts.prune.evaluate.metrics import t3_video

SCRIPT = "prune.evaluate.whole_clip_baseline"

# The distilled refiner schedule's informative levels. sigma=0 adds no noise and is excluded.
DEFAULT_SIGMAS = (0.909375, 0.725, 0.421875)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view", type=Path, action="append", required=True, help="Corpus view directory; repeat for each video."
    )
    parser.add_argument("--objective", choices=dataset.OBJECTIVES, default="white")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIGMAS),
        help="Distilled schedule levels; defaults to 0.909375, 0.725, 0.421875.",
    )
    parser.add_argument("--context-latent-frames", type=int, default=causal_core.CONTEXT_LATENT_FRAMES)
    parser.add_argument(
        "--variant",
        choices=("distilled", "dev"),
        default="distilled",
        help="dev loads the base transformer and enters the stock --steps schedule at each sigma.",
    )
    parser.add_argument("--transformer", type=Path, default=None, help="Explicit transformer checkpoint path.")
    parser.add_argument("--steps", type=int, default=None, help="dev only: N of the stock N-step schedule.")
    parser.add_argument(
        "--dev-denoising-steps",
        type=int,
        default=None,
        help="dev truncated only: actual calls selected at evenly spaced indices from the fixed stock tail.",
    )
    parser.add_argument(
        "--dev-schedule",
        choices=("truncated", "rescaled", "direct"),
        default="truncated",
        help="dev only: stock tail, rescaled curve, or explicit direct [sigma, 0] call.",
    )
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale; 1 disables the negative-prompt pass.")
    parser.add_argument("--stg", type=float, default=0.0, help="STG scale; 0 disables the perturbed pass.")
    parser.add_argument("--stg-blocks", type=int, nargs="+", default=[28])
    parser.add_argument("--rescale", type=float, default=0.0, help="Guidance rescale (pipelines use 0.7 with CFG).")
    parser.add_argument("--negative-prompt", default=None, help="Default: ltx_pipelines DEFAULT_NEGATIVE_PROMPT.")
    parser.add_argument("--raw-only", action="store_true", help="Save raw latents and manifest without VAE decoding.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Several rollout seeds in one load.")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("d0", "d1"),
        default=["d0", "d1"],
        help="d0 noises the capture, d1 the guide; c0 and the target are the capture in both.",
    )
    parser.add_argument("--output", type=Path, required=True)
    add_model_args(parser)
    add_prompt_args(parser)
    args = parser.parse_args(argv)
    if sorted(args.arms) != ["d0", "d1"] and not args.raw_only:
        parser.error("a single --arms value needs --raw-only (the decoded panel video is capture | D0 | D1)")
    if args.variant == "dev":
        if args.dev_schedule == "direct":
            if args.steps is not None and args.steps < 1:
                parser.error("--steps must be positive when supplied")
            if not args.sigmas or len(set(args.sigmas)) != len(args.sigmas):
                parser.error("direct dev needs distinct positive sigmas")
            for sigma in args.sigmas:
                try:
                    model_sampling.validate_schedule([sigma, 0.0])
                except ValueError as error:
                    parser.error(str(error))
        if args.dev_schedule != "direct" and (args.steps is None or args.steps < 1):
            parser.error("--variant dev needs --steps N")
        if args.dev_denoising_steps is not None:
            if args.dev_schedule != "truncated":
                parser.error("--dev-denoising-steps requires --dev-schedule truncated")
            for sigma in args.sigmas:
                try:
                    model_sampling.thinned_truncated_schedule(sigma, args.steps, args.dev_denoising_steps)
                except ValueError as error:
                    parser.error(str(error))
    elif args.dev_schedule == "direct":
        parser.error("--dev-schedule direct is dev only")
    elif args.steps is not None:
        parser.error("--steps is dev only; the distilled model uses [sigma, 0]")
    elif args.dev_denoising_steps is not None:
        parser.error("--dev-denoising-steps is dev only")
    return args


def suffix(args: argparse.Namespace) -> str:
    """The latent/MP4 file-name suffix that names the schedule family."""
    if args.variant == "dev":
        if args.dev_schedule == "direct":
            return "dev_direct"
        calls = "" if args.dev_denoising_steps is None else f"_k{args.dev_denoising_steps}"
        return f"dev_n{args.steps}{calls}_cfg{args.cfg:g}_stg{args.stg:g}"
    return "one_step"


def whole_clip_block(latent_frame_counts: list[int]) -> int:
    """The block size that makes block 0 ``[0, T)`` cover every latent frame of every view.

    Block 0 absorbs the keyframe, so it spans ``1 + block`` frames. All views share ``T``:
    one geometry serves the whole invocation, and a longer view would lose its tail.
    """
    counts = set(latent_frame_counts)
    if len(counts) != 1:
        raise SystemExit(f"whole-clip rollouts need equal-length views, got latent frame counts {sorted(counts)}")
    (count,) = counts
    if count < 2:
        raise SystemExit("whole-clip rollouts need at least two latent frames")
    return count - 1


def load_masters(
    view: Path, objective: str, *, require_guide: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """Load capture and an optional guide; D0-only never opens guide files."""
    capture = view / dataset.capture_bundle_name(objective)
    guide = view / dataset.guide_bundle_name(objective)
    for path in (capture, guide) if require_guide else (capture,):
        if not path.is_file():
            raise SystemExit(f"missing master latent: {path}")
    z_y, fps = load_training_master(capture)
    if not require_guide:
        return z_y, None, fps
    z_g, guide_fps = load_training_master(guide)
    if z_g.shape != z_y.shape or guide_fps != fps:
        raise SystemExit(f"capture/guide shape or fps mismatch: {view}")
    return z_y, z_g, fps


def global_epsilon(
    source: torch.Tensor, grid: common.ClipGrid, plan: list[tuple[int, int]], seed: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Draw noise once over global latent-frame indices, independent of block geometry."""
    epsilon = common.epsilon_block(source, seed)
    return epsilon, [epsilon[:, slice(*grid.token_span(*span))] for span in plan]


def checked_sigmas(values: list[float], schedule: list[float]) -> tuple[float, ...]:
    """Distilled levels must be distinct, positive and on the model's own grid."""
    if not values:
        raise SystemExit("--sigmas requires at least one value")
    if len(set(values)) != len(values):
        raise SystemExit("--sigmas contains duplicate values")
    for sigma in values:
        if sigma <= 0:
            raise SystemExit(f"sigma must be nonzero and positive, got {sigma}")
        if not any(abs(sigma - scheduled) < 1e-9 for scheduled in schedule):
            raise SystemExit(f"sigma {sigma} is not on model schedule {schedule}")
    return tuple(values)


def dev_schedule(args: argparse.Namespace, sigma: float) -> list[float]:
    """The stock dev schedule entered at ``sigma``; validated again by the model owner."""
    if args.dev_schedule == "direct":
        return list(model_sampling.validate_schedule([sigma, 0.0]))
    if args.dev_denoising_steps is not None:
        return list(model_sampling.thinned_truncated_schedule(sigma, args.steps, args.dev_denoising_steps))
    make = model_sampling.truncated_schedule if args.dev_schedule == "truncated" else model_sampling.rescaled_schedule
    return list(make(sigma, args.steps))


def clip_grid(
    z_y: torch.Tensor, fps: float, geometry: causal_core.CausalGeometry, *, device: torch.device, latent_channels: int
) -> common.ClipGrid:
    """The capture's token grid -- positions, keyframe marks and tools."""
    _, latent_frames, height, width = z_y.shape
    return common.ClipGrid.build(
        latent_frames,
        height * geometry.scale_factors.height,
        width * geometry.scale_factors.width,
        fps,
        geometry,
        device=device,
        dtype=DTYPE,
        latent_channels=latent_channels,
    )


@dataclass(frozen=True)
class WholeClip:
    """One view's fixed rollout inputs: masters, grid, one-block plan and saved noise slices."""

    z_y: torch.Tensor
    z_g: torch.Tensor | None
    grid: common.ClipGrid
    geometry: causal_core.CausalGeometry
    plan: list[tuple[int, int]]
    epsilons: list[torch.Tensor]
    device: torch.device


def rollout(
    denoise_fn,
    clip: WholeClip,
    context: torch.Tensor,
    sigma: float,
    *,  # noqa: ANN001
    arm: str,
    seed: int,
    schedule: list[float] | None,
) -> torch.Tensor:
    """One whole-clip block through the model owner's rollout; the arm changes only the noised source."""
    if arm not in ("d0", "d1") or (arm == "d1" and clip.z_g is None):
        raise ValueError("D1 requires a guide master; D0 requires only capture")
    grid = clip.grid
    capture = grid.patchify(clip.z_y.unsqueeze(0).to(device=clip.device, dtype=DTYPE))
    source = capture if arm == "d0" else grid.patchify(clip.z_g.unsqueeze(0).to(device=clip.device, dtype=DTYPE))
    tokens, _ = causal_core.rollout(
        denoise_fn,
        grid,
        clip.geometry,
        None,
        source,
        context,
        sigma,
        seed=seed,
        blocks=clip.plan,
        teacher_forcing=False,
        teacher_tokens=None,
        schedule=schedule,
        kv_source="refresh",
        first_frame_condition=capture[:, : grid.tokens_per_latent_frame],
        block_epsilons=clip.epsilons,
        history_mode="recompute",
    )
    covered = clip.plan[-1][1]
    return grid.unpatchify_block(tokens[:, : covered * grid.tokens_per_latent_frame], covered)


def decode_pixels(session, latent: torch.Tensor, decoder, seed: int) -> torch.Tensor:  # noqa: ANN001
    """Decode with a fresh identical generator for every compared panel."""
    generator = torch.Generator(device=session.device).manual_seed(seed)
    device_context = torch.cuda.device(session.device) if session.device.type == "cuda" else nullcontext()
    with device_context:
        pixels = decode_latent(session, latent.to(session.device), decoder, generator=generator)
    expected = (latent.shape[2] - 1) * 8 + 1
    if len(pixels) != expected:
        raise ValueError(f"decoder returned {len(pixels)} frames; expected {expected}")
    return pixels


def _saved_tensor(root: Path, name: str, digest: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Check contained saved bytes before loading a CPU tensor; never reconstruct device RNG."""
    path = root / name
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or provenance.file_sha256(path) != digest
    ):
        raise ValueError("baseline artifact path or hash differs")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != DTYPE or tuple(tensor.shape) != shape:
        raise ValueError("baseline tensor type, dtype or shape differs")
    if not torch.isfinite(tensor).all():
        raise ValueError("baseline tensor is nonfinite")
    return tensor


def _inspect_completion(args: argparse.Namespace, root: Path) -> list[Path]:  # noqa: PLR0912, PLR0915
    """Verify saved direct dev or distilled outputs without model or decoder loading."""
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("baseline root or manifest is a symlink")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        not args.raw_only
        or args.objective != "white"
        or (args.variant != "distilled" and not (args.variant == "dev" and args.dev_schedule == "direct"))
        or (args.cfg, args.stg, args.rescale) != (1.0, 0.0, 0.0)
        or args.negative_prompt is not None
        or not args.sigmas
        or len(set(args.sigmas)) != len(args.sigmas)
        or any(not 0.0 < sigma <= 1.0 for sigma in args.sigmas)
        or args.arms not in (["d0"], ["d0", "d1"])
    ):
        raise ValueError("saved completion supports raw white distilled or direct dev unguided D0, optional D1")
    seeds = [args.seed] if args.seeds is None else list(args.seeds)
    if len(set(seeds)) != len(seeds) or len({view.resolve() for view in args.view}) != len(args.view):
        raise ValueError("duplicate baseline view or seed")
    fixed = {
        **native_manifest.fields(),
        "kind": "d1_paired_source_probe",
        "checkpoint": None,
        "objective": args.objective,
        "sigmas": list(args.sigmas),
        "seed": args.seed,
        "seeds": seeds,
        "teacher_forcing": False,
        "history_policy": "none_single_block",
        "history_mode": "recompute",
        "model_variant": args.variant,
        "steps": args.steps,
        "denoising_steps_requested": None,
        "schedule_policy": ("direct_ltx2_scheduler" if args.variant == "dev" else "distilled_grid"),
        "max_blocks": None,
        "raw_only": True,
        "arms": list(args.arms),
        "adapter_condition_problems": [],
        "off_condition": False,
        "noise": {
            "scheme": "one global epsilon tensor per view, sliced by latent-frame block",
            "shared_across_arms_sigmas_and_block_geometries": True,
        },
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ValueError("baseline scientific settings differ")
    prompt = resolve_prompt(args)
    text = manifest["text_context"]
    if (
        text["prompt"] != prompt
        or text["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()
        or text["is_default_prompt"] != (prompt == DEFAULT_PROMPT)
        or text["dtype"] != str(DTYPE)
        or len(text["shape"]) != 3
        or text["shape"][0] != 1
        or any(type(size) is not int or size < 1 for size in text["shape"])
        or not isinstance(text["sha256"], str)
        or len(text["sha256"]) != 64
    ):
        raise ValueError("baseline prompt or text context identity differs")
    if manifest["model"].get("model_key") != args.model or manifest["model"].get("script") != SCRIPT:
        raise ValueError("baseline model or producer identity differs")
    rows = manifest["videos"]
    if len(rows) != len(args.view) * len(seeds) * len(args.sigmas):
        raise ValueError("baseline row coverage differs")
    evidence = {root / "manifest.json"}
    seen = set()
    records = {}
    for row in rows:
        artifact = row["artifacts"]
        view = Path(artifact["view"])
        seed, sigma = artifact["seed"], row["sigma"]
        key = (str(view.resolve()), seed, sigma)
        if (
            key in seen
            or view not in [item.resolve() for item in args.view]
            or seed not in seeds
            or sigma not in args.sigmas
        ):
            raise ValueError("baseline view, seed or sigma coverage differs")
        seen.add(key)
        if (
            Path(row["view"]).resolve() != view
            or row["schedule"] != [sigma, 0.0]
            or row["trajectory"] != suffix(args)
            or row["output"] is not None
            or row["blocks"] != artifact["blocks"]
            or row["panels"]
            != ["ground_truth"]
            + [{"d0": "gt_latent_rollout", "d1": "rgb_render_latent_rollout"}[arm] for arm in args.arms]
        ):
            raise ValueError("baseline row schedule, panels or output differs")
        record_key = (str(view), seed)
        if record_key in records:
            if records[record_key] != artifact:
                raise ValueError("baseline rows do not share identical input/noise records")
            continue
        records[record_key] = artifact
        capture, guide, fps = load_masters(view, args.objective, require_guide="d1" in args.arms)
        if not torch.isfinite(capture).all() or (guide is not None and not torch.isfinite(guide).all()):
            raise ValueError("baseline source masters are nonfinite")
        if artifact["fps"] != fps:
            raise ValueError("baseline source fps differs")
        for role, bundle in (
            ("capture", dataset.capture_bundle_name(args.objective)),
            ("guide", dataset.guide_bundle_name(args.objective)),
        ):
            if role == "guide" and "d1" not in args.arms and artifact.get("guide") is None:
                if artifact.get("guide_sha256") is not None:
                    raise ValueError("guide hash without a guide source")
                continue
            source = view / bundle
            if Path(artifact[role]) != source or provenance.file_sha256(source) != artifact[role + "_sha256"]:
                raise ValueError("baseline source path or hash differs")
        frames = capture.shape[1]
        geometry = causal_core.deployed_geometry(
            SpatioTemporalScaleFactors(*manifest["geometry"]["scale_factors"]),
            block_latent_frames=whole_clip_block([frames]),
            context_latent_frames=args.context_latent_frames,
        )
        if (
            manifest["geometry"] != geometry.as_dict()
            or artifact["blocks"] != [[0, frames]]
            or row["blocks"] != [[0, frames]]
            or manifest["capabilities"]["latent_channels"] != capture.shape[0]
        ):
            raise ValueError("baseline whole-clip geometry differs")
        grid = clip_grid(capture, fps, geometry, device=torch.device("cpu"), latent_channels=capture.shape[0])
        tokens = grid.patchify(capture.unsqueeze(0))
        _saved_tensor(root, artifact["epsilon"], artifact["epsilon_sha256"], tuple(tokens.shape))
        evidence.add(root / artifact["epsilon"])
        latents = artifact["latents"]
        timings = artifact["rollout_timing"]
        pairs = {(sigma, arm) for sigma in args.sigmas for arm in args.arms}
        if (
            len(latents) != len(pairs)
            or {(item["sigma"], item["arm"]) for item in latents} != pairs
            or len({item["path"] for item in latents}) != len(pairs)
            or artifact["epsilon"] in {item["path"] for item in latents}
            or len(timings) != len(pairs)
            or {(item["sigma"], item["arm"]) for item in timings} != pairs
            or any(item["schedule"] != [item["sigma"], 0.0] or item["forward_passes"] != 1 for item in timings)
        ):
            raise ValueError("baseline latent or forward coverage differs")
        for item in latents:
            latent = _saved_tensor(root, item["path"], item["sha256"], (1, *capture.shape))
            # Native rollout conditions the model with the master's BF16 conversion.
            if not torch.equal(latent[0, :, 0], capture[:, 0].to(dtype=DTYPE)):
                raise ValueError("baseline c0 differs from clean capture")
            evidence.add(root / item["path"])
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("baseline output contains a symlink")
        if path.is_file():
            actual.add(path)
    if actual != evidence:
        raise ValueError("baseline local artifact inventory differs")
    return sorted(evidence)


def verify_completion(args: argparse.Namespace, root: Path) -> bool:
    """Return false for incomplete or altered saved outputs; no native model lifetime."""
    try:
        _inspect_completion(args, root)
        return True
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        AttributeError,
        EOFError,
        pickle.UnpicklingError,
        SystemExit,
    ):
        return False


def evidence_paths(args: argparse.Namespace, root: Path) -> list[Path]:
    """Expose only fully verified local baseline evidence to its queue wrapper."""
    return _inspect_completion(args, root)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915 -- one ordered producer
    args = parse_args(argv)
    seeds = [args.seed] if args.seeds is None else list(args.seeds)
    if args.seeds is not None and len(seeds) > 1 and not args.raw_only:
        raise SystemExit("--seeds with more than one seed needs --raw-only")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory must be fresh and empty: {args.output}")
    masters = [(view, *load_masters(view, args.objective, require_guide="d1" in args.arms)) for view in args.view]
    block_latent_frames = whole_clip_block([z_y.shape[1] for _, z_y, _, _ in masters])
    prompt = resolve_prompt(args)
    transformer_path = args.transformer
    if transformer_path is None and args.variant == "dev":
        transformer_path = backbone.transformer_path(args.model, "dev")
    if transformer_path is not None and not Path(transformer_path).is_file():
        raise SystemExit(f"transformer checkpoint does not exist: {transformer_path}")
    session = open_session(args, script=SCRIPT, prompt=prompt, transformer_path=transformer_path)
    if args.variant == "dev":
        if not args.sigmas or any(not 0.0 < s <= 1.0 for s in args.sigmas) or len(set(args.sigmas)) != len(args.sigmas):
            raise SystemExit(f"dev sigmas must be distinct values in (0, 1], got {args.sigmas}")
        sigmas = tuple(args.sigmas)
    else:
        sigmas = checked_sigmas(args.sigmas, session.model.sigmas)
    guider, negative_context, negative_prompt = None, None, None
    if args.cfg != 1.0 or args.stg != 0.0 or args.rescale != 0.0:
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams  # noqa: PLC0415
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT  # noqa: PLC0415
        from scripts.prune.data import prompt_cache  # noqa: PLC0415

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
        block_latent_frames=block_latent_frames,
        context_latent_frames=args.context_latent_frames,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    context_bytes = session.context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    model_stamp = session.stamp(dtype=str(DTYPE))
    model_stamp["video_vae_fingerprint"] = provenance.checkpoint_fingerprint(session.model.paths.video_vae())
    model_stamp["checkpoint"] = None
    passes_per_step = 1 + (args.cfg != 1.0) + (args.stg != 0.0)
    manifest = {
        **native_manifest.fields(
            cfg=args.cfg,
            stg=args.stg,
            stg_blocks=args.stg_blocks,
            rescale=args.rescale,
            negative_prompt=negative_prompt,
        ),
        "kind": "d1_paired_source_probe",
        "checkpoint": None,
        "objective": args.objective,
        "sigmas": list(sigmas),
        "seed": args.seed,
        "seeds": seeds,
        "teacher_forcing": False,
        "history_policy": "none_single_block",
        "history_mode": "recompute",
        "geometry": geometry.as_dict(),
        "noise": {
            "scheme": "one global epsilon tensor per view, sliced by latent-frame block",
            "shared_across_arms_sigmas_and_block_geometries": True,
        },
        "text_context": {
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "is_default_prompt": prompt == DEFAULT_PROMPT,
            "sha256": hashlib.sha256(context_bytes).hexdigest(),
            "shape": list(session.context.shape),
            "dtype": str(session.context.dtype),
        },
        "model": model_stamp,
        "capabilities": asdict(session.model.caps),
        "model_variant": args.variant,
        "steps": args.steps,
        "denoising_steps_requested": args.dev_denoising_steps,
        "schedule_policy": f"{args.dev_schedule}_ltx2_scheduler" if args.variant == "dev" else "distilled_grid",
        "attention_backend": {
            "self": attention_label(DEFAULT_TRANSFORMER_OPS.attention_ops.attention_function),
            "masked": attention_label(DEFAULT_TRANSFORMER_OPS.attention_ops.masked_attention_function),
        },
        "max_blocks": None,
        "raw_only": args.raw_only,
        "arms": list(args.arms),
        "adapter_condition_problems": [],
        "off_condition": False,
        "videos": [],
    }
    results = []
    with session.transformer(loras=()) as transformer:
        denoise_fn = (
            common.denoised_from_x0_model(transformer)
            if guider is None
            else common.guided_denoised_from_x0_model(transformer, guider, negative_context)
        )
        for seed, (view, z_y, z_g, fps) in [(seed, item) for seed in seeds for item in masters]:
            grid = clip_grid(
                z_y, fps, geometry, device=session.device, latent_channels=session.model.caps.latent_channels
            )
            plan = geometry.plan(grid.latent_frames)
            if plan != [(0, grid.latent_frames)]:
                raise SystemExit(f"whole-clip plan does not cover the capture once: {plan}")
            source = grid.patchify(z_y.unsqueeze(0).to(device=session.device, dtype=DTYPE))
            epsilon, epsilons = global_epsilon(source, grid, plan, seed)
            clip = WholeClip(z_y, z_g, grid, geometry, plan, epsilons, session.device)
            stem = f"{view.parent.parent.name}_{view.name}" + ("" if args.seeds is None else f"_seed{seed}")
            noise_path = args.output / f"{stem}_epsilon.pt"
            torch.save(epsilon.cpu(), noise_path)
            view_record = {
                "view": str(view.resolve()),
                "seed": seed,
                "fps": fps,
                "capture": str((view / dataset.capture_bundle_name(args.objective)).resolve()),
                "guide": str((view / dataset.guide_bundle_name(args.objective)).resolve()) if z_g is not None else None,
                "capture_sha256": provenance.file_sha256(view / dataset.capture_bundle_name(args.objective)),
                "guide_sha256": provenance.file_sha256(view / dataset.guide_bundle_name(args.objective))
                if z_g is not None
                else None,
                "epsilon": noise_path.name,
                "epsilon_sha256": provenance.file_sha256(noise_path),
                "blocks": [list(block) for block in plan],
            }
            arms = {}
            for sigma in sigmas:
                schedule = dev_schedule(args, sigma) if args.variant == "dev" else None
                arms[sigma] = {}
                for arm in args.arms:
                    if session.device.type == "cuda":
                        torch.cuda.synchronize(session.device)
                        torch.cuda.reset_peak_memory_stats(session.device)
                    started = time.perf_counter()
                    latent = rollout(denoise_fn, clip, session.context, sigma, arm=arm, seed=seed, schedule=schedule)
                    if session.device.type == "cuda":
                        torch.cuda.synchronize(session.device)
                    view_record.setdefault("rollout_timing", []).append(
                        {
                            "sigma": sigma,
                            "arm": arm,
                            "wall_seconds": time.perf_counter() - started,
                            "schedule": schedule if schedule is not None else [sigma, 0.0],
                            "forward_passes": (len(schedule) - 1 if schedule else 1) * passes_per_step,
                            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(session.device)
                            if session.device.type == "cuda"
                            else None,
                        }
                    )
                    arms[sigma][arm] = latent.cpu()
                    latent_path = args.output / f"{stem}_sigma_{sigma:.6f}_{suffix(args)}_{arm}.pt"
                    torch.save(arms[sigma][arm], latent_path)
                    view_record.setdefault("latents", []).append(
                        {
                            "sigma": sigma,
                            "arm": arm,
                            "path": latent_path.name,
                            "sha256": provenance.file_sha256(latent_path),
                        }
                    )
            results.append((view, z_y, fps, plan, arms, view_record))
            print(f"Rolled out {view}", flush=True)
    with nullcontext(None) if args.raw_only else session.decoder() as decoder:
        for view, z_y, fps, plan, arms, view_record in results:
            target_pixels = None
            if decoder is not None:
                target_pixels = decode_pixels(session, z_y.unsqueeze(0)[:, :, : plan[-1][1]], decoder, args.seed)
            for sigma in arms:
                output = None
                if decoder is not None:
                    panels = [target_pixels] + [
                        decode_pixels(session, arms[sigma][arm], decoder, args.seed) for arm in args.arms
                    ]
                    output = args.output / f"{view.parent.parent.name}_{view.name}_sigma_{sigma:.6f}_{suffix(args)}.mp4"
                    t3_video(*panels, output, fps=fps)
                manifest["videos"].append(
                    {
                        "view": str(view),
                        "sigma": sigma,
                        "trajectory": suffix(args),
                        "schedule": next(t["schedule"] for t in view_record["rollout_timing"] if t["sigma"] == sigma),
                        "output": str(output) if output is not None else None,
                        "panels": ["ground_truth"]
                        + [{"d0": "gt_latent_rollout", "d1": "rgb_render_latent_rollout"}[arm] for arm in args.arms],
                        "blocks": [list(block) for block in plan],
                        "artifacts": view_record,
                    }
                )
                if output is not None:
                    print(output, flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
