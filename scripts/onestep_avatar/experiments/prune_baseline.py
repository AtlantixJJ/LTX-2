"""Queue the pruning-owned raw baseline; see doc/experiments/prune_baseline.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import runtime, software
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone
from scripts.prune.core import provenance
from scripts.prune.data import prompt_cache
from scripts.prune.evaluate import whole_clip_baseline as producer

SELECTOR = "prune_baseline"
ENTRY = "scripts/onestep_avatar/experiments/prune_baseline.py"
# Explicit diagnostic dependencies supplement, rather than broaden, ordinary profiles.
EXTRA_SOURCES = (
    ENTRY,
    "scripts/onestep_avatar/experiments/__init__.py",
    spec_protocol.ENTRY,
    "scripts/prune/evaluate/whole_clip_baseline.py",
    "scripts/prune/evaluate/decode.py",
    "scripts/prune/evaluate/metrics.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return spec_protocol.outer_args(argv, __doc__)


def scientific_args(argv: list[str]) -> argparse.Namespace:
    args = producer.parse_args(argv)
    if (
        len(args.view) != 1
        or args.seeds is not None
        or not args.raw_only
        or args.objective != "white"
        or args.variant != "distilled"
        or args.model != "2.5"
        or (args.cfg, args.stg, args.rescale) != (1.0, 0.0, 0.0)
        or args.negative_prompt is not None
        or args.arms not in (["d0"], ["d0", "d1"])
        or args.context_latent_frames != 8
        or not args.sigmas
        or len(set(args.sigmas)) != len(args.sigmas)
        or any(not math.isfinite(sigma) or not 0.0 < sigma <= 1.0 for sigma in args.sigmas)
    ):
        raise ValueError("prune_baseline requires one view/seed, raw white distilled unguided D0, optional D1")
    return args


def read_spec(spec: Path, root: Path) -> argparse.Namespace:
    return spec_protocol.read_spec(
        parse_args(["--spec", str(spec), "--output", str(root)]), SELECTOR, scientific_args
    )


def resource_phases(spec: Path) -> list[str]:
    read_spec(spec, spec.parent / "unused_output")
    return ["runtime"]


def runtime_profile(spec: Path) -> dict:
    return {"generic_phase": True, "phases": resource_phases(spec), "processes": 1}


def input_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve metadata and fixed file inputs without opening a native model session."""
    model = backbone.resolve(args.model, args.variant, args.transformer)
    producer.checked_sigmas(args.sigmas, model.sigmas)
    view = args.view[0]
    paths = [
        args.spec,
        view / dataset.capture_bundle_name(args.objective),
        view / dataset.guide_bundle_name(args.objective),
        Path(model.paths.transformer()),
        Path(model.paths.text_encoder()),
        Path(model.paths.video_vae()),
        prompt_cache.cache_path(args.model, producer.resolve_prompt(args)),
    ]
    if args.prompt_file is not None:
        paths.append(args.prompt_file)
    return sorted({path.resolve() for path in paths})


def input_pins(args: argparse.Namespace) -> dict[str, str]:
    """Use full hashes, including large components, at native acceptance boundaries."""
    return {str(path): sha256(path) for path in input_paths(args)}


def acceptance(args: argparse.Namespace) -> dict:
    return {
        "schema_version": 1,
        "protocol": SELECTOR,
        "settings": spec_protocol.settings(args),
        "inputs": input_pins(args),
        "model": model_identity(args),
        "software": software.capture("evaluation", "causal", extra_sources=EXTRA_SOURCES),
    }


def model_identity(args: argparse.Namespace) -> dict:
    """Read registered native coordinates and component identities without loading tensors."""
    model = backbone.resolve(args.model, args.variant, args.transformer)
    return {
        "model_key": model.key,
        "model_version": list(model.version),
        "scale_factors": list(model.scale_factors),
        "capabilities": asdict(model.caps),
        "transformer_path": str(Path(model.paths.transformer()).resolve()),
        "video_vae_path": str(Path(model.paths.video_vae()).resolve()),
        "transformer_fingerprint": provenance.checkpoint_fingerprint(model.paths.transformer()),
        "video_vae_fingerprint": provenance.checkpoint_fingerprint(model.paths.video_vae()),
    }


def check_model_binding(args: argparse.Namespace, root: Path) -> None:
    """Bind the producer's sampled component stamps to the fully pinned current inputs."""
    expected = model_identity(args)
    manifest = json.loads((root / "baseline/manifest.json").read_text())
    stamp = manifest["model"]
    for key in ("transformer_path", "video_vae_path"):
        stamp[key] = str(Path(stamp[key]).resolve())
    if (manifest["capabilities"] != expected["capabilities"]
            or manifest["geometry"]["scale_factors"] != expected["scale_factors"]
            or any(stamp.get(key) != value for key, value in expected.items() if key != "capabilities")):
        raise ValueError("pruning producer component identity differs from pinned inputs")
    context = torch.load(
        prompt_cache.cache_path(args.model, producer.resolve_prompt(args)), map_location="cpu", weights_only=True
    ).to(dtype=producer.DTYPE)
    if not torch.isfinite(context).all():
        raise ValueError("pruning fixed prompt cache is nonfinite")
    digest = hashlib.sha256(context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    saved = manifest["text_context"]
    if saved["sha256"] != digest or saved["shape"] != list(context.shape) or saved["dtype"] != str(context.dtype):
        raise ValueError("pruning text context differs from pinned cache bytes")


def inspect_completion(args: argparse.Namespace, root: Path) -> dict:
    """Recheck every source/input byte and saved scientific artifact, without CUDA."""
    if root.resolve() != args.output.resolve():
        raise ValueError("pruning completion root differs from requested output")
    manifest = spec_protocol.inspect(args, SELECTOR)
    record = json.loads((root / "acceptance.json").read_text())
    if record != acceptance(args):
        raise ValueError("pruning acceptance inputs, settings or software differ")
    software.check_current(record["software"])
    if not producer.verify_completion(args, root / "baseline"):
        raise ValueError("pruning baseline scientific completion is invalid")
    check_model_binding(args, root)
    return manifest


def verify_completion(spec: Path, root: Path) -> dict:
    """Raise on invalid existing completion, as required by the experiment queue contract."""
    return inspect_completion(read_spec(spec, root), root)


def evidence_paths(spec: Path, root: Path, *, verified: bool = False) -> list[Path]:
    record = (spec_protocol.inspect(read_spec(spec, root), SELECTOR) if verified
              else verify_completion(spec, root))
    return spec_protocol.evidence(record, root)


def main(argv: list[str] | None = None) -> int:
    args = spec_protocol.read_spec(parse_args(argv), SELECTOR, scientific_args)
    if args.dry_run:
        print(json.dumps({"protocol": SELECTOR, "settings": spec_protocol.settings(args)}, indent=2))  # noqa: T201
        return 0
    if args.output.exists():
        raise ValueError("prune_baseline output must be fresh")
    before = acceptance(args)
    record = json.loads(args.spec.read_text())
    if sha256(args.spec) != args.spec_sha256:
        raise ValueError("scientific spec changed before pruning production")
    result = producer.main([
        *record["arguments"], "--output", str(args.output / "baseline"), "--gpu-id", str(args.gpu_id)
    ])
    if result != 0 or not producer.verify_completion(args, args.output / "baseline"):
        raise ValueError("pruning producer did not complete its scientific outputs")
    check_model_binding(args, args.output)
    software.check_current(before["software"])
    if before != acceptance(args):
        raise ValueError("pruning inputs or software changed during production")
    with (args.output / "acceptance.json").open("x") as stream:
        stream.write(json.dumps(before, indent=2) + "\n")
    spec_protocol.publish(args, SELECTOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(runtime.run(main))
