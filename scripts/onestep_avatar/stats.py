"""A1 + B1c: characterise the distilled map, and measure the gap the LoRA has to close.

Everything here is measurement, no training. Four numbers, each of which decides something the
plan is currently guessing at:

* **(a) eps-sensitivity.** SS2.1 argues the distilled checkpoint is a deterministic,
  mode-seeking pushforward rather than a sampler, which leaves open whether eps is a sampling
  variable here *at all*. Run one step at sigma_0 from the same ``z_g`` with N different eps and
  measure the spread. Small spread => eps is a nuisance variable: fix the seed at inference and
  drop every "sampling" framing. Large spread => eps is a free quality knob worth a line in
  the eval.
* **(b) the excursion** ``a = ||Phi(x_sigma0) - z_g|| / sqrt(d)`` -- how far the base model moves
  its own input, in the units SS4.2's viability check uses.
* **(c) latent moments** of ``z_g`` against ``z_y``. SS4.2's input-distribution check: the VAE's
  normalisation is calibrated on real video, so a per-channel moment mismatch reaches the
  transformer scaled by ``(1 - sigma_0) = 0.275``. Mind the reference -- under the linear
  interpolant ``Var(x_sigma) = (1-sigma)^2 + sigma^2 = 0.60`` at sigma = 0.725 BY DESIGN, so
  matching to unit variance would be the wrong correction.
* **(r) the gap** ``r = ||z_y - z_g|| / sqrt(d)``, the number the whole D1-vs-D2 decision hangs
  on (SS4.2, SS9 risk 3) and the one B1 never ran. ``r ~ 0.35`` is comfortable for D1;
  ``r_p90 >~ 0.6`` means the adapter is being asked to overpower the prior and the answer is D2.
  Reported next to ``a``, because ``a ~ r`` is what "the task is well matched to what the model
  already does" means quantitatively.

**Lives in the LTX-2 tree, not beside the other analysis modules** (plan SS7.2 lists it under
``scripts/onestep_avatar/``): (a) and (b) need the VAE *and* the 42 GB transformer, which only
exist in the ``ltx`` env. (r) and (c) are pure tensor arithmetic over the corpus master latents and
need no GPU at all -- ``--no-gpu`` runs exactly those.

    conda run -n ltx python -m scripts.onestep_avatar.stats \\
      --renders ../../ARG-Avatar/expr --pairs ../data/AnimatableHuman/DNARenderingVideo \\
      --out ../expr/onestep_avatar/analysis_summary.json --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import dataset, mask_video
from scripts.onestep_avatar.precompute import (
    BUNDLE_SCHEMA_VERSION,
    VideoReader,
    atomic_json_save,
)
from scripts.prune.core import ltx_adapter, model_registry, refine_core, refine_task
from scripts.prune.core.session import DTYPE
from scripts.prune.data import prompt_cache

DEFAULT_SIGMA0 = 0.725
DEFAULT_EPS_SAMPLES = 8


def rms_gap(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    """``||a - b|| / sqrt(d)`` -- a per-dimension RMS difference, not a raw norm.

    Dividing by ``sqrt(d)`` is what makes the number comparable across geometries and
    directly usable in SS4.2's viability arithmetic: the LoRA must move the model's output by
    ``r / (sigma_0 * sqrt(2))``, which at sigma_0 = 0.725 is ``r`` times its natural scale.

    ``weights`` is a ``[F, H, W]`` latent-grid coverage mask, and passing one is not optional
    for the number SS4.2 actually wants. **Measured 2026-09-12:** the subject occupies only
    12.4 % of the 1024**2 crop, the ARGAvatar render's background is white (255) and the
    capture's is a dark dome (51), so an unweighted ``r`` is ~88 % a background flip and says
    nothing about the task. See ``measure_pairs``.
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


def _master(path: Path) -> torch.Tensor:
    """One view's master latent, refusing a pre-SS4.4 per-window bundle by name."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION or "master" not in bundle:
        raise SystemExit(
            f"{path}: not a v{BUNDLE_SCHEMA_VERSION} master bundle. Re-run "
            "precompute.py --capture-only for this view"
        )
    return bundle["master"]


def measure_pairs(
    corpus_root: Path,
    limit: int | None = None,
    mask_kind: str = "union",
    objective: str = dataset.DEFAULT_OBJECTIVE,
) -> dict:
    """(r) and (c) from the corpus's paired master latents -- CPU only, no model of any kind.

    Reads the same three per-view files ``train.py`` reads (SS4.4, 2026-09-14): the capture
    master, the guide master, and the clip's loss-mask grids. Both halves of every comparison
    come out of the same view directory at the same frame index, so a mismatch in indexing
    cannot make ``r`` look better than it is -- and it is now measured over whole clips
    rather than over windows that overlapped, which double-counted every second latent frame.

    **Two ``r``s, and only one of them answers SS4.2.** ``r_full`` is over the whole crop and
    ``r_subject`` is weighted by the subject coverage ``precompute.py`` stores. They are very
    different numbers here and the difference is not subtle: the render composites the avatar
    on WHITE and the capture is a dark capture dome, while the subject covers ~12 % of the
    frame. Under ``bg`` therefore ``r_full`` largely measures a background convention while
    ``r_subject`` measures the task; under ``white`` both sides agree on the background by
    construction and the two converge. Quote ``r_subject`` either way.
    Take the D1-vs-D2 decision on ``r_subject``; report ``r_full`` next to it, because the
    background flip is real and is the reason SS4.3 row 1's masked loss is mandatory rather
    than a refinement.
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
        z_g = _master(guide_path)
        z_y = _master(guide_path.with_name(capture_name))
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
        # SS4.2's decision, stated rather than left for a reader to derive from a percentile --
        # and taken on the SUBJECT-weighted r, which is the only one about the task.
        "d1_viable": None if subject is None else subject["p90"] < 0.6,
        "by_source": {source: _summary(values) for source, values in per_source.items()},
        "moments": {
            "guide_mean": float(guide[0]),
            "guide_std": float(guide[1]),
            "capture_mean": float(capture[0]),
            "capture_std": float(capture[1]),
            # The reference SS4.2 warns about: Var(x_sigma) = (1-s)^2 + s^2 by design, NOT 1.
            "expected_var_at_sigma0": (1 - DEFAULT_SIGMA0) ** 2 + DEFAULT_SIGMA0**2,
        },
        # SS4.2 caveat, still live: these are ONE clip's channel moments, not the corpus's.
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
) -> dict:
    """(a) and (b): one step at sigma_0 from a guide, repeated over eps, on unpaired renders.

    Deliberately runs through ``refine_core.make_window_state`` / ``run_schedule`` -- the same
    calls the deployed refiner makes -- with a two-point schedule ``[sigma_0, 0]``. That is
    what "one step" means operationally, and reproducing it here rather than hand-rolling a
    forward is what makes ``a`` comparable with the numbers ``k2`` is measured at.
    """
    device = torch.device(f"cuda:{gpu_id}")
    geometry = refine_task.deployed_geometry(model.scale_factors)
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    sigmas = torch.tensor([sigma0, 0.0], dtype=torch.float32, device=device)

    from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: PLC0415 -- torch-heavy, imported late

    denoiser = SimpleDenoiser(context, None)
    excursions, spreads, per_video = [], [], {}

    with ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder:
        encoded: list[tuple[Path, torch.Tensor, float, int, int]] = []
        for path in videos:
            reader = VideoReader(path)
            frames = min(len(reader), geometry.window_frames)
            if frames < geometry.window_frames:
                continue
            batch = reader.get_batch(range(geometry.window_frames))
            _, height, width, _ = batch.shape
            height, width = (height // 32) * 32, (width // 32) * 32
            video = batch[:, :height, :width].permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=DTYPE)
            with torch.no_grad():
                latent = encoder.tiled_encode(video / 127.5 - 1.0, None)
            encoded.append((path, latent, reader.get_avg_fps(), height, width))

    from scripts.prune.core.session import Session  # noqa: PLC0415 -- torch-heavy, imported late.

    session = Session(
        model=model, device=device, script="onestep_avatar.stats", context=context, denoiser=denoiser, sigmas=sigmas
    )
    with session.transformer() as transformer:
        for path, z_g, fps, height, width in encoded:
            tools = refine_core.tools_for_window(
                geometry, height, width, fps, latent_channels=model.caps.latent_channels
            )
            outputs = []
            for k in range(eps_samples):
                state = refine_core.make_window_state(z_g, None, sigma0, tools, seed + k, device, DTYPE)
                final = refine_core.run_schedule(transformer, denoiser, state, sigmas)
                outputs.append(refine_core.finalize(final, tools))
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        help="SS1.2. Which objective's (z_g, z_y) pair r and the moments are measured over. "
        "The two give genuinely different numbers -- under 'white' both sides agree on the "
        "background by construction, so r_full stops measuring a background convention.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    p.add_argument("--eps-samples", type=int, default=DEFAULT_EPS_SAMPLES)
    p.add_argument("--max-videos", type=int, default=4, help="A1 needs a handful, not all 16")
    p.add_argument("--max-windows", type=int, default=None, help="cap the pair scan")
    p.add_argument(
        "--subject-mask",
        choices=("render", "capture", "union", "intersection"),
        default="union",
        help="which coverage grid weights r_subject; 'union' is the whole area either side calls subject",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu", action="store_true", help="run only the pair measurements (r, moments)")
    args = p.parse_args()
    if args.renders is None and args.pairs is None:
        p.error("give --renders, --pairs, or both")

    model = model_registry.resolve(args.model)
    report: dict[str, object] = {"model": model.key, "sigma0": args.sigma0}
    if args.pairs is not None:
        report["pairs"] = measure_pairs(
            args.pairs, args.max_windows, args.subject_mask, args.objective
        )
    if args.renders is not None and not args.no_gpu:
        videos = sorted(args.renders.rglob(args.render_glob))[: args.max_videos]
        if not videos:
            raise SystemExit(f"no mp4 under {args.renders}")
        report["map"] = measure_map(
            model,
            videos,
            renders_root=args.renders,
            gpu_id=args.gpu_id,
            sigma0=args.sigma0,
            eps_samples=args.eps_samples,
            seed=args.seed,
        )
    atomic_json_save(report, args.out)
    print(json.dumps(report, indent=2)[:4000])  # noqa: T201 -- CLI's requested summary.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
