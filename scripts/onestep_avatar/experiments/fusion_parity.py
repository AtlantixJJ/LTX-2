"""Preserve study protocols through public shared owners; see doc/experiments/fusion_parity.md."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone, causal, common
from scripts.onestep_avatar.training import checkpoints

if TYPE_CHECKING:
    from scripts.prune.core.session import Session


def fusion_probe_block(
    transformer: torch.nn.Module,
    kind: str,
    session: Session,
    capture: torch.Tensor,
    guide: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    """Preserve the historical diagnostic's one empty-cache block, input and seed."""
    from scripts.prune.core.session import DTYPE  # noqa: PLC0415 -- native training dtype

    geometry = causal.deployed_geometry(session.model.scale_factors)
    _, frames, height, width = capture.shape
    grid = common.ClipGrid.build(
        frames,
        height * geometry.scale_factors.height,
        width * geometry.scale_factors.width,
        fps,
        geometry,
        device=session.device,
        dtype=DTYPE,
        latent_channels=session.model.caps.latent_channels,
    )
    span = geometry.plan(frames)[0]
    lo, hi = grid.token_span(*span)
    target = grid.patchify(capture.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    source = grid.patchify(guide.unsqueeze(0).to(device=session.device, dtype=DTYPE))
    c0 = target[:, : grid.tokens_per_latent_frame]
    inner = common.base_model(transformer)
    cache = causal.BlockCache.allocate(
        grid,
        geometry,
        num_layers=len(inner.transformer_blocks),
        inner_dim=inner.inner_dim,
        device=session.device,
        dtype=DTYPE,
    )
    noisy = common.with_clean_prefix(common.noise_block(source[:, lo:hi], 0.421875, 42), c0)
    return (
        causal.fusion_parity_block(
            transformer,
            kind,
            grid,
            cache,
            noisy,
            session.context,
            0.421875,
            span,
            clean_prefix_tokens=c0.shape[1],
        )
        .float()
        .cpu()
    )


def fusion_parity_metrics(outputs: dict[str, torch.Tensor]) -> dict:
    """Compare raw outputs and adapter effects without hiding undefined ratios."""
    if set(outputs) != {"bare", "step0", "fused1", "peft0", "peft1"}:
        raise ValueError("fusion parity requires all five model cases")
    reference = outputs["bare"]
    if reference.numel() == 0 or any(
        value.shape != reference.shape or not torch.isfinite(value).all() for value in outputs.values()
    ):
        raise ValueError("fusion parity outputs must have matching shapes and finite values")

    def relative(delta: torch.Tensor, base: torch.Tensor) -> float | None:
        denominator = float(base.norm())
        return None if denominator == 0 else float(delta.norm() / base.norm())

    effect_peft = outputs["peft1"] - outputs["peft0"]
    effect_fused = outputs["fused1"] - reference
    gap = relative(effect_fused - effect_peft, effect_peft)
    return {
        "step0_equals_bare_bitwise": bool(torch.equal(outputs["step0"], reference)),
        "rel_l2_step0_vs_bare": relative(outputs["step0"] - reference, reference),
        "rel_l2_training_path_vs_probe_path_no_adapter": relative(outputs["peft0"] - reference, reference),
        "rel_l2_adapter_effect_peft": relative(effect_peft, outputs["peft0"]),
        "rel_l2_adapter_effect_fused": relative(effect_fused, reference),
        "rel_l2_effect_fused_vs_effect_peft": gap,
        "rel_l2_fused1_vs_peft1": relative(outputs["fused1"] - outputs["peft1"], outputs["peft1"]),
        "tolerance_rel_l2_effect": 0.2,
        "fused_vs_peft_within_tolerance": gap is not None and gap <= 0.2,
        "effect_ratio_status": "undefined_zero_peft_effect" if gap is None else "defined",
    }


def evaluate_fusion_parity(  # noqa: PLR0915 -- ordered preflight, five native cases and publication
    run: Path, view: Path, output: Path, *, gpu_id: int, step: int = 1, output_tensors_path: Path | None = None
) -> dict:
    """Own the five historical fusion cases and publish only completed diagnostic results."""
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: PLC0415
    from ltx_trainer.model_loader import load_transformer  # noqa: PLC0415
    from scripts.onestep_avatar.model.adapters import LORA_TARGETS  # noqa: PLC0415
    from scripts.prune.core.session import DTYPE, open_session  # noqa: PLC0415

    if type(step) is not int or step < 1 or output.exists():
        raise ValueError("fusion diagnostic requires a positive trained step and fresh output")
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    if (
        config.get("lora_target") not in LORA_TARGETS
        or type(config.get("lora_rank")) is not int
        or config["lora_rank"] < 1
    ):
        raise ValueError("fusion diagnostic requires a valid saved LoRA configuration")
    alpha = config.get("lora_alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("fusion diagnostic requires a positive saved LoRA alpha")
    if alpha != config["lora_rank"]:
        raise ValueError("fusion diagnostic requires alpha equal to rank for the fused path")
    dev = backbone.transformer_path("2.5", "dev")
    step0 = run / "checkpoints/lora_weights_step_00000.safetensors"
    trained = run / f"checkpoints/lora_weights_step_{step:05d}.safetensors"
    capture_path, guide_path = view / dataset.capture_bundle_name("white"), view / dataset.guide_bundle_name("white")
    paths = [config_path, dev, step0, trained, capture_path, guide_path]
    if any(not path.is_file() for path in paths):
        raise ValueError("fusion diagnostic input file is missing")
    capture, fps = dataset.load_training_master(capture_path)
    guide, guide_fps = dataset.load_training_master(guide_path)
    if capture.shape != guide.shape or fps != guide_fps or capture.shape[1] < 3:
        raise ValueError("fusion diagnostic requires matching capture/guide geometry, fps and block-zero coverage")
    identities = {str(path.resolve()): sha256(path) for path in paths}
    session = open_session(
        argparse.Namespace(model="2.5", gpu_id=gpu_id, seed=42),
        script="onestep_avatar.experiments.fusion_parity",
        transformer_path=dev,
    )
    outputs = {}
    for name, adapters in (("bare", ()), ("step0", (step0,)), ("fused1", (trained,))):
        loras = tuple(LoraPathStrengthAndSDOps(str(path), 1.0, LTXV_LORA_COMFY_RENAMING_MAP) for path in adapters)
        with session.transformer(dev, loras=loras) as transformer:
            outputs[name] = fusion_probe_block(transformer, "x0", session, capture, guide, fps)
        del transformer
        torch.cuda.empty_cache()
    model = load_transformer(checkpoint_path=str(dev), device=session.device, dtype=DTYPE, video_only=True)
    model.requires_grad_(False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=alpha,
            target_modules=LORA_TARGETS[config["lora_target"]],
            lora_dropout=0.0,
        ),
    )
    model.eval()
    for name, path in (("peft0", step0), ("peft1", trained)):
        checkpoints.load_stage_init(model, path)
        outputs[name] = fusion_probe_block(model, "velocity", session, capture, guide, fps)
    del model
    torch.cuda.empty_cache()
    if any(sha256(path) != identities[str(path.resolve())] for path in paths):
        raise ValueError("fusion diagnostic inputs changed during execution")
    if output_tensors_path is not None:
        output_tensors_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(output_tensors_path, lambda p: torch.save(outputs, p))
    result = {
        "view": str(view),
        "trained_step": step,
        "sigma": 0.421875,
        "block": 0,
        "noise_seed": 42,
        **fusion_parity_metrics(outputs),
        "input_sha256": identities,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"))
    return result


EXTRA_SOURCES = (
    "scripts/onestep_avatar/experiments/__init__.py",
    "scripts/onestep_avatar/experiments/fusion_parity.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Own the direct diagnostic CLI and its pinned queue specification."""
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    if "--spec" not in arguments:
        parser.add_argument("--fusion-parity", action="store_true")
        parser.add_argument("--run", type=Path, required=True)
        parser.add_argument("--view", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--gpu-id", type=int, required=True)
        parser.add_argument("--step", type=int, default=1)
        return parser.parse_args(arguments)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    outer = parser.parse_args(arguments)
    outer.completion = {"manifest": str(outer.output.resolve() / "manifest.json")}
    return outer


def read_spec(outer: argparse.Namespace) -> argparse.Namespace:
    """Read exact scientific fields after queue path normalization."""
    spec = json.loads(outer.spec.read_text())
    if (
        set(spec) != {"schema_version", "protocol", "run", "view", "step"}
        or spec["schema_version"] != 1
        or spec["protocol"] != "fusion_parity"
        or type(spec["step"]) is not int
        or spec["step"] < 1
    ):
        raise ValueError("fusion parity requires a version-one specification and positive trained step")
    outer.run, outer.view, outer.step = Path(spec["run"]), Path(spec["view"]), spec["step"]
    if not outer.run.is_absolute() or not outer.view.is_absolute():
        raise ValueError("scientific specification paths must be absolute")
    return outer


def verify_completion(spec: Path, root: Path) -> dict:
    """Recompute all five raw comparisons with the unchanged historical tolerance."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: PLC0415 -- model-free native layout

    args = read_spec(parse_args(["--spec", str(spec), "--output", str(root)]))
    result_path, raw_path = root / "result.json", root / "raw_outputs.pt"
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads(result_path.read_text())
    outputs = torch.load(raw_path, map_location="cpu", weights_only=True)
    if not isinstance(outputs, dict):
        raise ValueError("fusion parity completion requires all raw case outputs")
    capture, fps = dataset.load_training_master(args.view / dataset.capture_bundle_name("white"))
    guide, guide_fps = dataset.load_training_master(args.view / dataset.guide_bundle_name("white"))
    if capture.shape != guide.shape or fps != guide_fps or fps <= 0 or capture.shape[1] < 3:
        raise ValueError("fusion parity requires matching masters, fps and block-zero coverage")
    target = VideoLatentPatchifier(patch_size=1).patchify(capture[:, :3].unsqueeze(0).to(torch.bfloat16)).float()
    first_frame_tokens = capture.shape[2] * capture.shape[3]
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != target.shape
           or not torch.isfinite(value).all()
           or not torch.equal(value[:, :first_frame_tokens], target[:, :first_frame_tokens])
           for value in outputs.values()):
        raise ValueError("fusion parity raw output shape, precision or clean c0 differs")
    expected_metrics = fusion_parity_metrics(outputs)
    if any(result.get(key) != value for key, value in expected_metrics.items()):
        raise ValueError("fusion parity saved metrics differ from raw outputs")
    expected_fields = {
        "view": str(args.view),
        "trained_step": args.step,
        "sigma": 0.421875,
        "block": 0,
        "noise_seed": 42,
    }
    if any(result.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("fusion parity settings differ from the specification")
    paths = [
        args.run / "config.json",
        backbone.transformer_path("2.5", "dev"),
        args.run / "checkpoints/lora_weights_step_00000.safetensors",
        args.run / f"checkpoints/lora_weights_step_{args.step:05d}.safetensors",
        args.view / dataset.capture_bundle_name("white"),
        args.view / dataset.guide_bundle_name("white"),
    ]
    if result.get("input_sha256") != {str(path.resolve()): sha256(path) for path in paths}:
        raise ValueError("fusion parity scientific input bytes differ")
    expected = {
        "schema_version": 1,
        "kind": "onestep_avatar.fusion_parity",
        "spec_sha256": sha256(spec),
        "software": software.capture("evaluation", "causal", extra_sources=EXTRA_SOURCES),
        "artifacts": {str(path.resolve()): sha256(path) for path in (result_path, raw_path)},
    }
    if manifest != expected:
        raise ValueError("fusion parity completion specification, software or artifacts differ")
    software.check_current(manifest["software"])
    return manifest


def evidence_paths(spec: Path, root: Path) -> list[Path]:
    """Expose the fully checked raw/record byte inventory to the queue."""
    manifest = verify_completion(spec, root)
    return sorted([root / "manifest.json", *map(Path, manifest["artifacts"])])


def main(argv: list[str] | None = None) -> int:
    """Run the preserved five-case protocol and bind queue-mode raw output evidence."""
    args = parse_args(argv)
    if not hasattr(args, "spec"):
        evaluate_fusion_parity(args.run, args.view, args.output, gpu_id=args.gpu_id, step=args.step)
        return 0
    args = read_spec(args)
    if args.output.exists():
        raise ValueError("fusion parity requires a fresh output directory")
    producer_software = software.capture("evaluation", "causal", extra_sources=EXTRA_SOURCES)
    spec_hash = sha256(args.spec)
    evaluate_fusion_parity(
        args.run,
        args.view,
        args.output / "result.json",
        gpu_id=args.gpu_id,
        step=args.step,
        output_tensors_path=args.output / "raw_outputs.pt",
    )
    if sha256(args.spec) != spec_hash:
        raise ValueError("fusion parity specification changed during execution")
    software.check_current(producer_software)
    manifest = {
        "schema_version": 1,
        "kind": "onestep_avatar.fusion_parity",
        "spec_sha256": spec_hash,
        "software": producer_software,
        "artifacts": {
            str(path.resolve()): sha256(path) for path in (args.output / "result.json", args.output / "raw_outputs.pt")
        },
    }
    atomic_write(
        args.output / "manifest.json", lambda p: p.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    )
    verify_completion(args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
