"""Pinned A1/B1c measurements; see doc/experiments/stats.md."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import dataset, mask_video
from scripts.onestep_avatar.corpus.precompute import (
    VideoReader,
    atomic_json_save,
)
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, model_registry
from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE
from scripts.prune.data import prompt_cache

ENTRY = "scripts/onestep_avatar/experiments/stats.py"
EXTRA_SOURCES = (ENTRY, "scripts/onestep_avatar/experiments/__init__.py", spec_protocol.ENTRY)

DEFAULT_SIGMA0 = 0.725
DEFAULT_EPS_SAMPLES = 8


def rms_gap(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    """``||a - b|| / sqrt(d)`` -- a per-dimension RMS difference, not a raw norm.

    Dividing by ``sqrt(d)`` is what makes the number comparable across geometries and
    directly usable in the D1 viability arithmetic: the LoRA must move the model's output by
    ``r / (sigma_0 * sqrt(2))``, which at sigma_0 = 0.725 is ``r`` times its natural scale.

    ``weights`` is a ``[F, H, W]`` latent-grid coverage mask, and the task-relevant number
    needs one. Measured on the corpus: the subject occupies only 12.4 % of the 1024**2 crop,
    the ARGAvatar render's background is white (255) and the capture's is a dark dome (51),
    so an unweighted ``r`` is ~88 % a background flip and says nothing about the task. See
    ``measure_pairs``.
    """
    diff = (a.float() - b.float())
    if weights is None:
        flat = diff.flatten()
        return float(flat.norm() / math.sqrt(flat.numel()))
    w = weights.float().to(diff.device)
    while w.dim() < diff.dim():
        w = w.unsqueeze(0)
    w = w.expand_as(diff)
    mass = w.sum()
    if mass <= 0:
        return float("nan")
    return float(torch.sqrt((diff.pow(2) * w).sum() / mass))


def channel_moments(latent: torch.Tensor) -> dict[str, list[float]]:
    """Per-channel mean/std of a ``[C, F, H, W]`` (or ``[1, C, F, H, W]``) latent."""
    channels = latent.shape[-4] if latent.dim() == 4 else latent.shape[1]
    flat = latent.float().reshape(channels, -1)
    return {"mean": flat.mean(dim=1).tolist(), "std": flat.std(dim=1).tolist()}


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(tensor, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {
        "n": len(values),
        "mean": float(tensor.mean()),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "max": float(tensor.max()),
    }


def measure_pairs(
    corpus_root: Path,
    limit: int | None = None,
    mask_kind: str = "union",
    objective: str = dataset.DEFAULT_OBJECTIVE,
) -> dict:
    """(r) and (c) from the corpus's paired master latents -- CPU only, no model of any kind.

    Reads the same per-view files training reads: the capture master, the guide master, and
    the clip's mask grids. Both halves of every comparison come out of the same view directory
    at the same frame index, so a mismatch in indexing cannot make ``r`` look better than it
    is. Each clip is measured once over its whole length, so no latent frame counts twice.

    **Two ``r``s, and only one of them measures the task.** ``r_full`` is over the whole crop and
    ``r_subject`` is weighted by the subject coverage ``precompute.py`` stores. They are very
    different numbers here and the difference is not subtle: the render composites the avatar
    on WHITE and the capture is a dark capture dome, while the subject covers ~12 % of the
    frame. Under ``bg`` therefore ``r_full`` largely measures a background convention while
    ``r_subject`` measures the task; under ``white`` both sides agree on the background by
    construction and the two converge. Quote ``r_subject`` either way and base the D1
    viability decision on it; report ``r_full`` next to it, because the background flip is real.
    """
    guide_name = dataset.guide_bundle_name(objective)
    capture_name = dataset.capture_bundle_name(objective)
    guides = sorted(corpus_root.rglob(guide_name))
    if limit is not None:
        guides = guides[:limit]
    if not guides:
        raise SystemExit(
            f"no {guide_name} under {corpus_root}; run precompute.py's paired mode "
            f"with --objective {objective} first"
        )

    gaps, masked_gaps, coverage, guide_stats, capture_stats, per_source = [], [], [], [], [], {}
    first_pair = None
    for guide_path in guides:
        relative = guide_path.parent.relative_to(corpus_root)
        z_g = dataset.load_master(guide_path)
        z_y = dataset.load_master(guide_path.with_name(capture_name))
        frames = min(z_g.shape[1], z_y.shape[1])
        z_g, z_y = z_g[:, :frames], z_y[:, :frames]
        if first_pair is None:
            first_pair = (z_g, z_y)
        gap = rms_gap(z_y, z_g)
        gaps.append(gap)

        alpha = guide_path.with_name(dataset.ALPHA_NAME)
        capture_crop = guide_path.with_name(dataset.CAPTURE_MASK_CROP_NAME)
        if alpha.is_file() and capture_crop.is_file():
            masks = mask_video.read_latent_masks(
                guide_path.parent,
                latent_frames=frames,
                latent_height=int(z_y.shape[2]),
                latent_width=int(z_y.shape[3]),
                time_scale=SpatioTemporalScaleFactors.default().time,
            )
            render, capture_mask = masks["render_alpha"].float(), masks["capture_mask"].float()
            weights = {
                "render": render,
                "capture": capture_mask,
                "union": torch.maximum(render, capture_mask),
                "intersection": torch.minimum(render, capture_mask),
            }[mask_kind]
            masked_gaps.append(rms_gap(z_y, z_g, weights[:frames]))
            coverage.append(float(weights[:frames].mean()))

        per_source.setdefault(str(relative), []).append(gap)
        guide_stats.append(torch.stack([z_g.float().mean(), z_g.float().std()]))
        capture_stats.append(torch.stack([z_y.float().mean(), z_y.float().std()]))

    guide = torch.stack(guide_stats).mean(dim=0)
    capture = torch.stack(capture_stats).mean(dim=0)
    subject = _summary(masked_gaps) if masked_gaps else None
    return {
        "r_full": _summary(gaps),
        "r_subject": subject,
        "r_subject_mask": mask_kind,
        "subject_coverage": _summary(coverage) if coverage else None,
        # The viability decision, stated rather than left for a reader to derive from a
        # percentile -- and taken on the SUBJECT-weighted r, the only one about the task.
        "d1_viable": None if subject is None else subject["p90"] < 0.6,
        "by_source": {source: _summary(values) for source, values in per_source.items()},
        "moments": {
            "guide_mean": float(guide[0]),
            "guide_std": float(guide[1]),
            "capture_mean": float(capture[0]),
            "capture_std": float(capture[1]),
            # The reference variance: Var(x_sigma) = (1-s)^2 + s^2 by design, NOT 1.
            "expected_var_at_sigma0": (1 - DEFAULT_SIGMA0) ** 2 + DEFAULT_SIGMA0**2,
        },
        # These are ONE clip's channel moments, not the corpus's.
        # Widen the sample before relying on the tail.
        "channel_moments": {
            "guide": channel_moments(first_pair[0]),
            "capture": channel_moments(first_pair[1]),
        },
    }


def measure_map(
    model: model_registry.RefinerModel,
    videos: list[Path],
    *,
    renders_root: Path,
    gpu_id: int,
    sigma0: float,
    eps_samples: int,
    seed: int,
    evidence_root: Path | None = None,
) -> dict:
    """(a) and (b): one step at sigma_0 from a guide, repeated over eps, on unpaired renders.

    Each guide is encoded once over its complete VAE-aligned prefix. One full
    bidirectional forward predicts the clip at [sigma_0, 0], keeping frame 0 clean.
    """
    device = torch.device(f"cuda:{gpu_id}")
    context = prompt_cache.get_or_build(model, DEFAULT_PROMPT, DTYPE, device)

    excursions, spreads, per_video, raw = [], [], {}, {}

    with ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder:
        encoded: list[tuple[Path, torch.Tensor, float, int, int]] = []
        for path in videos:
            reader = VideoReader(path)
            frames = (len(reader) - 1) // model.scale_factors.time * model.scale_factors.time + 1
            if frames < 1:
                continue
            batch = reader.get_batch(range(frames))
            _, height, width, _ = batch.shape
            height, width = (height // 32) * 32, (width // 32) * 32
            video = batch[:, :height, :width].permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=DTYPE)
            with torch.no_grad():
                latent = encoder.tiled_encode(video / 127.5 - 1.0, None)
            encoded.append((path, latent, reader.get_avg_fps(), height, width))

    from scripts.prune.core.session import Session  # noqa: PLC0415 -- torch-heavy, imported late.

    session = Session(
        model=model, device=device, script="onestep_avatar.experiments.stats", context=context
    )
    with session.transformer() as transformer:
        for path, z_g, fps, height, width in encoded:
            grid = common.ClipGrid.build(
                z_g.shape[2], height, width, fps, causal_core.CausalGeometry(model.scale_factors),
                device=device, dtype=DTYPE, latent_channels=model.caps.latent_channels,
            )
            clean = grid.patchify(z_g)
            c0 = clean[:, :grid.tokens_per_latent_frame]
            outputs = []
            for k in range(eps_samples):
                epsilon = torch.randn(clean.shape, generator=torch.Generator(device=device).manual_seed(seed + k),
                                      device=device, dtype=DTYPE)
                initial = torch.lerp(clean, epsilon, sigma0)
                modality = common.block_modality(
                    grid, common.with_clean_prefix(initial, c0), context, sigma0,
                    token_slices=[grid.token_span(0, grid.latent_frames)],
                    clean_prefix_tokens=grid.tokens_per_latent_frame,
                )
                prediction, _ = transformer(video=modality, audio=None, perturbations=None)
                outputs.append(grid.unpatchify_block(common.with_clean_prefix(prediction, c0), grid.latent_frames))
            excursion = sum(rms_gap(out, z_g) for out in outputs) / len(outputs)
            # Spread is the mean pairwise distance between the N outputs, in the SAME units as
            # the excursion -- so "does eps matter" is answerable by comparing two numbers.
            pairwise = [
                rms_gap(outputs[i], outputs[j]) for i in range(len(outputs)) for j in range(i + 1, len(outputs))
            ]
            spread = sum(pairwise) / len(pairwise) if pairwise else 0.0
            excursions.append(excursion)
            spreads.append(spread)
            # Keyed by the path RELATIVE to --renders, not by basename: every corpus guide is
            # named `argavatar_render.mp4`, so a basename key silently collapses the whole
            # corpus into one entry. The summary statistics were right; `by_video` lost rows.
            try:
                key = str(path.relative_to(renders_root))
            except ValueError:
                key = str(path)
            per_video[key] = {"excursion_a": excursion, "eps_spread": spread}
            if evidence_root is not None:
                raw[key] = {"guide": z_g.detach().cpu(),
                            "outputs": [value.detach().cpu() for value in outputs],
                            "source": str(path.resolve()), "source_sha256": sha256(path)}

    if evidence_root is not None:
        torch.save(raw, evidence_root / "map_raw.pt")
    return {
        "sigma0": sigma0,
        "eps_samples": eps_samples,
        "excursion_a": _summary(excursions),
        "eps_spread": _summary(spreads),
        "eps_spread_over_excursion": _summary(
            [s / e if e > 0 else 0.0 for s, e in zip(spreads, excursions, strict=True)]
        ),
        # A1(a)'s decision, written down rather than left to a reader: if the spread is a few
        # percent of the excursion, eps is a nuisance variable and the seed should just be
        # fixed at inference.
        "eps_is_nuisance": _summary(
            [s / e if e > 0 else 0.0 for s, e in zip(spreads, excursions, strict=True)]
        )["p50"] < 0.1,
        "by_video": per_video,
    }


def scientific_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--renders", type=Path, default=None, help="directory of guide mp4s for A1 (a)+(b)")
    p.add_argument(
        "--render-glob",
        default="*.mp4",
        help="which mp4s under --renders count as guides. Point it at 'argavatar_render.mp4' to "
        "measure on the REAL corpus guides at the deployed 1024**2 geometry rather than on "
        "smoke renders at some other size -- `a` is only comparable with `r` at the same token count",
    )
    p.add_argument(
        "--pairs", type=Path, default=None,
        help="Corpus root holding the per-view master latents, for r and the moments",
    )
    p.add_argument(
        "--objective",
        choices=dataset.OBJECTIVES,
        default=dataset.DEFAULT_OBJECTIVE,
        help="Which objective's (z_g, z_y) pair r and the moments are measured over. "
        "The two give genuinely different numbers -- under 'white' both sides agree on the "
        "background by construction, so r_full stops measuring a background convention.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    p.add_argument("--eps-samples", type=int, default=DEFAULT_EPS_SAMPLES)
    p.add_argument("--max-videos", type=int, default=4, help="A1 needs a handful, not all 16")
    p.add_argument("--max-sources", type=int, default=None,
                   help="cap the pair scan (a source is a whole clip)")
    p.add_argument(
        "--subject-mask",
        choices=("render", "capture", "union", "intersection"),
        default="union",
        help="which coverage grid weights r_subject; 'union' is the whole area either side calls subject",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu", action="store_true", help="run only the pair measurements (r, moments)")
    args = p.parse_args(argv)
    if args.renders is None and args.pairs is None:
        p.error("give --renders, --pairs, or both")

    if (not math.isfinite(args.sigma0) or not 0 < args.sigma0 <= 1 or args.eps_samples < 1
            or args.max_videos < 1 or (args.max_sources is not None and args.max_sources < 1)):
        p.error("measurement sigma/counts must be finite and positive")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return spec_protocol.outer_args(argv, __doc__)


def read_spec(spec: Path, root: Path) -> argparse.Namespace:
    args = spec_protocol.read_spec(parse_args(["--spec", str(spec), "--output", str(root)]),
                                   "stats", scientific_args, output_flag="--out")
    args.out = root / "analysis_summary.json"
    return args


def resource_phases(spec: Path) -> list[str]:
    read_spec(spec, spec.parent / "unused-output")
    return ["runtime"]


def dependency_outputs(spec: Path) -> list[Path]:
    read_spec(spec, spec.parent / "unused-output")
    return []


def _input_files(args: argparse.Namespace) -> dict[str, str]:
    paths = []
    if args.pairs is not None:
        guides = sorted(args.pairs.rglob(dataset.guide_bundle_name(args.objective)))
        if args.max_sources is not None:
            guides = guides[:args.max_sources]
        for guide in guides:
            paths += [guide, guide.with_name(dataset.capture_bundle_name(args.objective))]
            paths += [path for name in (dataset.ALPHA_NAME, dataset.CAPTURE_MASK_CROP_NAME)
                      if (path := guide.with_name(name)).is_file()]
    if args.renders is not None and not args.no_gpu:
        paths += sorted(args.renders.rglob(args.render_glob))[:args.max_videos]
    return {str(path.resolve()): sha256(path) for path in paths}


def _saved_map(args: argparse.Namespace, raw: dict) -> dict:
    excursions, spreads, per_video = [], [], {}
    expected = sorted(args.renders.rglob(args.render_glob))[:args.max_videos]
    keys = {str(path.relative_to(args.renders)): path for path in expected}
    if not isinstance(raw, dict) or set(raw) != set(keys) or not raw:
        raise ValueError("saved stats map video inventory differs")
    for key, source in keys.items():
        saved = raw[key]
        guide, outputs = saved["guide"], saved["outputs"]
        if (saved["source"] != str(source.resolve()) or saved["source_sha256"] != sha256(source)
                or not isinstance(guide, torch.Tensor) or guide.ndim != 5 or guide.shape[0] != 1
                or not guide.is_floating_point() or not torch.isfinite(guide).all()
                or len(outputs) != args.eps_samples):
            raise ValueError("saved stats guide or output count differs")
        if any(not isinstance(value, torch.Tensor) or value.shape != guide.shape
               or not value.is_floating_point() or not torch.isfinite(value).all()
               or not torch.equal(value[:, :, :1], guide[:, :, :1]) for value in outputs):
            raise ValueError("saved stats output shape, finiteness or clean first frame differs")
        excursion = sum(rms_gap(value, guide) for value in outputs) / len(outputs)
        pairwise = [rms_gap(outputs[i], outputs[j]) for i in range(len(outputs)) for j in range(i + 1, len(outputs))]
        spread = sum(pairwise) / len(pairwise) if pairwise else 0.0
        excursions.append(excursion)
        spreads.append(spread)
        per_video[key] = {"excursion_a": excursion, "eps_spread": spread}
    ratios = _summary([s / e if e > 0 else 0.0 for s, e in zip(spreads, excursions, strict=True)])
    return {"sigma0": args.sigma0, "eps_samples": args.eps_samples,
            "excursion_a": _summary(excursions), "eps_spread": _summary(spreads),
            "eps_spread_over_excursion": ratios, "eps_is_nuisance": ratios["p50"] < 0.1,
            "by_video": per_video}


def _same_metrics(actual: object, expected: object) -> bool:
    """Allow only the fixed float32 GPU/CPU reduction roundoff when reading saved map metrics."""
    if isinstance(expected, float):
        return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_same_metrics(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_metrics(a, b) for a, b in zip(actual, expected, strict=True)))
    return actual == expected


def verify_completion(spec: Path, root: Path) -> None:
    args = read_spec(spec, root)
    spec_protocol.inspect(args, "stats")
    binding = json.loads((root / "inputs.json").read_text())
    if binding["input_files"] != _input_files(args):
        raise ValueError("stats scientific input inventory differs")
    software.check_current(binding["software"])
    expected = {"model": model_registry.resolve(args.model).key, "sigma0": args.sigma0}
    if args.pairs is not None:
        expected["pairs"] = measure_pairs(args.pairs, args.max_sources, args.subject_mask, args.objective)
    if args.renders is not None and not args.no_gpu:
        expected["map"] = _saved_map(args, torch.load(root / "map_raw.pt", map_location="cpu", weights_only=True))
    if not _same_metrics(json.loads(args.out.read_text()), expected):
        raise ValueError("saved stats metrics differ from actual scientific measurements")
    def finite(value: object) -> bool:
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, list):
            return all(finite(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite(expected):
        raise ValueError("stats has a nonfinite measurement")


def evidence_paths(spec: Path, root: Path, *, verified: bool = False) -> list[Path]:
    if not verified:
        verify_completion(spec, root)
    return spec_protocol.evidence(spec_protocol.inspect(read_spec(spec, root), "stats"), root)


def main(argv: list[str] | None = None) -> int:
    outer = parse_args(argv)
    args = read_spec(outer.spec, outer.output)
    args.gpu_id, args.dry_run = outer.gpu_id, outer.dry_run
    if args.dry_run:
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    binding = {"input_files": _input_files(args),
               "software": software.capture("evaluation", "bidirectional", extra_sources=EXTRA_SOURCES)}
    atomic_json_save(binding, args.output / "inputs.json")
    model = model_registry.resolve(args.model)
    report: dict[str, object] = {"model": model.key, "sigma0": args.sigma0}
    if args.pairs is not None:
        report["pairs"] = measure_pairs(args.pairs, args.max_sources, args.subject_mask, args.objective)
    if args.renders is not None and not args.no_gpu:
        videos = sorted(args.renders.rglob(args.render_glob))[:args.max_videos]
        if not videos:
            raise ValueError(f"no mp4 under {args.renders}")
        report["map"] = measure_map(model, videos, renders_root=args.renders, gpu_id=args.gpu_id,
            sigma0=args.sigma0, eps_samples=args.eps_samples, seed=args.seed, evidence_root=args.output)
    atomic_json_save(report, args.out)
    spec_protocol.publish(args, "stats")
    verify_completion(args.spec, args.output)
    print(json.dumps(report, indent=2)[:4000])  # noqa: T201 -- CLI summary
    return 0


if __name__ == "__main__":
    from scripts.onestep_avatar.execution.runtime import run
    raise SystemExit(run(main))
