"""
Batch version of the VAE bottleneck probe for generated-video datasets.

This is intentionally numeric-only: it loads the encoder once, runs the compact
rank/border diagnostics over many videos, and writes JSON + CSV artifacts that
can be aggregated without producing hundreds of plots.

Example:
    python3 scripts/vae_dataset_bottleneck_probe.py \
      --manifest ../expr/video-eval-baseline-v1/manifests/ltx23_baseline_v1.json \
      --video-root ../expr/video-eval-baseline-v1/raw \
      --metrics-root ../expr/video-eval-baseline-v1/metrics \
      --output results/dataset_bottleneck_expr_vbench.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import decord
import numpy as np
import torch
from einops import rearrange
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))

from ltx_core.model.video_vae.ops import patchify  # noqa: E402
from ltx_trainer.model_loader import load_video_vae_encoder  # noqa: E402


decord.bridge.set_bridge("torch")

DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors"
)
DEFAULT_MANIFEST = WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "manifests" / "ltx23_baseline_v1.json"
DEFAULT_VIDEO_ROOT = WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw"
DEFAULT_METRICS_ROOT = WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "metrics"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "dataset_bottleneck_expr_vbench.json"
DTYPE = torch.bfloat16


def select_free_gpu() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = [tuple(int(x.strip()) for x in line.split(",")) for line in out.strip().splitlines()]
        return torch.device(f"cuda:{min(rows, key=lambda r: (r[1], r[2]))[0]}")
    except Exception:
        return torch.device("cuda:0")


def to_rows(feat: torch.Tensor) -> np.ndarray:
    return rearrange(feat, "b c f h w -> (b f h w) c").detach().cpu().float().numpy()


def interior_border_split(feat: torch.Tensor, ring: int = 1) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, h, w = feat.shape
    hmask = torch.ones(h, dtype=torch.bool, device=feat.device)
    wmask = torch.ones(w, dtype=torch.bool, device=feat.device)
    hmask[:ring] = False
    hmask[h - ring :] = False
    wmask[:ring] = False
    wmask[w - ring :] = False
    interior_grid = hmask[:, None] & wmask[None, :]
    interior = feat[..., interior_grid]
    border = feat[..., ~interior_grid]

    def flatten(x: torch.Tensor) -> np.ndarray:
        return rearrange(x, "b c f n -> (b f n) c").detach().cpu().float().numpy()

    return flatten(interior), flatten(border)


def single_frame_rows(feat: torch.Tensor) -> np.ndarray:
    mid = feat.shape[2] // 2
    return to_rows(feat[:, :, mid : mid + 1])


def loc_norm_map(feat: torch.Tensor) -> np.ndarray:
    return feat.float().pow(2).sum(dim=1).sqrt().mean(dim=1)[0].detach().cpu().numpy()


def border_interior_ratio_from_map(map_hw: np.ndarray, ring: int = 1) -> float:
    h, w = map_hw.shape
    border = np.ones((h, w), dtype=bool)
    border[ring : h - ring, ring : w - ring] = False
    return float(map_hw[border].mean() / (map_hw[~border].mean() + 1e-12))


def channel_stats(x: np.ndarray) -> dict[str, Any]:
    mean = x.mean(axis=0, dtype=np.float64)
    var = x.var(axis=0, dtype=np.float64)
    total_var = float(var.sum())
    order = np.argsort(var)[::-1]
    var_sorted = var[order]
    cumvar = np.cumsum(var_sorted) / (total_var + 1e-12)
    mean_sq = float(np.mean(var + mean**2))
    global_mean = float(mean.mean())
    return {
        "frac_channels_active_1e-2xmax": float((var > 1e-2 * var.max()).mean()) if var.size else 0.0,
        "n_channels_90pct_var": int(np.searchsorted(cumvar, 0.90) + 1) if var.size else 0,
        "top_channel_var_fraction": float(var_sorted[0] / (total_var + 1e-12)) if var.size else 0.0,
        "max_over_median_var": float(var_sorted[0] / (float(np.median(var)) + 1e-12)) if var.size else 0.0,
        "dc_fraction_of_total_energy": float((mean**2).sum() / ((mean**2).sum() + total_var + 1e-12)),
        "global_rms": float(np.sqrt(mean_sq)),
        "global_std": float(np.sqrt(max(mean_sq - global_mean**2, 0.0))),
        "global_min": float(x.min()),
        "global_max": float(x.max()),
    }


def exact_eigvals(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    xc = x.astype(np.float32, copy=False) - mean
    cov = (xc.T @ xc).astype(np.float64) / float(x.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    return np.clip(ev, 0.0, None)


def randomized_eigvals(x: np.ndarray, max_components: int) -> tuple[np.ndarray, float]:
    ncomp = min(max_components, x.shape[0] - 1, x.shape[1])
    if ncomp <= 0:
        return np.zeros((0,), dtype=np.float64), 0.0
    pca = PCA(
        n_components=ncomp,
        svd_solver="randomized",
        random_state=0,
        iterated_power=4,
    )
    pca.fit(x)
    total = float(x.var(axis=0, ddof=1).sum())
    return np.clip(pca.explained_variance_.astype(np.float64), 0.0, None), total


def rank_metrics(
    x: np.ndarray,
    *,
    max_components: int,
    exact: bool,
) -> dict[str, Any]:
    if exact:
        ev = exact_eigvals(x)
        total = float(ev.sum())
        method = "exact_covariance"
    else:
        ev, total = randomized_eigvals(x, max_components=max_components)
        method = f"randomized_top_{len(ev)}"

    ratio = ev / (total + 1e-12)
    cum = np.cumsum(ratio)
    missing_ratio = float(max(0.0, 1.0 - (float(cum[-1]) if cum.size else 0.0)))

    def n_for(threshold: float) -> int | None:
        if not cum.size or cum[-1] < threshold:
            return None
        return int(np.searchsorted(cum, threshold) + 1)

    if exact:
        entropy_ratio = ratio[ratio > 0]
    else:
        # Concentrating the missing tail into one bucket gives a conservative
        # lower bound for entropy effective rank.
        entropy_ratio = np.concatenate([ratio[ratio > 0], np.array([missing_ratio])])
        entropy_ratio = entropy_ratio[entropy_ratio > 0]

    return {
        "n_tokens": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "method": method,
        "missing_variance_ratio": missing_ratio,
        "cum_evr_4": float(cum[min(3, len(cum) - 1)]) if cum.size else 0.0,
        "entropy_effective_rank": float(np.exp(-np.sum(entropy_ratio * np.log(entropy_ratio))))
        if entropy_ratio.size
        else 0.0,
        "participation_ratio": float((total**2) / (float(np.sum(ev**2)) + 1e-12)) if total > 0 else 0.0,
        "n_comp_90": n_for(0.90),
        "n_comp_98": n_for(0.98),
        "evr_top4": ratio[:4].astype(float).tolist(),
    }


def summarize_stage(
    feat: torch.Tensor,
    *,
    max_components: int,
    exact: bool,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    full_rows = to_rows(feat)
    out: dict[str, Any] = {
        "shape_BCFHW": [int(v) for v in feat.shape],
        "full": rank_metrics(full_rows, max_components=max_components, exact=exact),
        "channels": channel_stats(full_rows),
        "border_interior_energy_ratio": border_interior_ratio_from_map(loc_norm_map(feat)),
    }

    if "interior_border" in splits:
        interior, border = interior_border_split(feat)
        out["interior_only"] = rank_metrics(interior, max_components=max_components, exact=exact)
        out["border_only"] = rank_metrics(border, max_components=max_components, exact=exact)
    if "single_frame" in splits:
        out["single_frame"] = rank_metrics(single_frame_rows(feat), max_components=max_components, exact=exact)
    return out


def read_video(path: Path, max_frames: int | None, device: torch.device) -> torch.Tensor:
    vr = decord.VideoReader(str(path))
    count = len(vr) if max_frames is None else min(len(vr), max_frames)
    frames = vr.get_batch(range(count))
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(DTYPE).to(device)
    video = (video / 127.5) - 1.0
    valid_f = ((video.shape[2] - 1) // 8) * 8 + 1
    return video[:, :, :valid_f]


def load_aux_metrics(metrics_root: Path, sample_id: str) -> dict[str, Any]:
    path = metrics_root / f"{sample_id}.json"
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    motion = record.get("metrics", {}).get("motion", {}).get("summary", {})
    vbench = record.get("metrics", {}).get("vbench", {}).get("dimensions", {})
    return {
        "motion_mean_flow": motion.get("mean_flow_magnitude"),
        "motion_max_flow": motion.get("max_flow_magnitude"),
        "motion_mean_luma_diff": motion.get("mean_abs_luma_difference"),
        "vbench_dynamic_degree": vbench.get("dynamic_degree", {}).get("aggregate_score"),
        "vbench_subject_consistency": vbench.get("subject_consistency", {}).get("aggregate_score"),
        "vbench_background_consistency": vbench.get("background_consistency", {}).get("aggregate_score"),
        "vbench_motion_smoothness": vbench.get("motion_smoothness", {}).get("aggregate_score"),
    }


@torch.no_grad()
def analyze_video(
    encoder: torch.nn.Module,
    video_path: Path,
    *,
    max_frames: int | None,
    max_components: int,
    b6_exact: bool,
    device: torch.device,
) -> dict[str, Any]:
    video = read_video(video_path, max_frames, device)
    x = encoder.conv_in(patchify(video, patch_size_hw=4, patch_size_t=1))
    del video

    captured: dict[str, torch.Tensor] = {}
    for idx, block in enumerate(encoder.down_blocks):
        x = block(x)
        if idx in (6, 7, 8):
            captured[f"enc_b{idx}"] = x.detach().clone()

    post_norm = encoder.conv_norm_out(x)
    means = encoder.conv_out(encoder.conv_act(post_norm))[:, : encoder.latent_channels]
    latent = encoder.per_channel_statistics.normalize(means)

    stages = {
        "enc_b6": summarize_stage(
            captured["enc_b6"],
            max_components=max_components,
            exact=b6_exact,
            splits=(),
        ),
        "enc_b7": summarize_stage(
            captured["enc_b7"],
            max_components=max_components,
            exact=True,
            splits=(),
        ),
        "enc_b8": summarize_stage(
            captured["enc_b8"],
            max_components=max_components,
            exact=True,
            splits=("interior_border", "single_frame"),
        ),
        "postPixelNorm": summarize_stage(
            post_norm,
            max_components=max_components,
            exact=True,
            splits=(),
        ),
        "conv_out_means": summarize_stage(
            means,
            max_components=max_components,
            exact=True,
            splits=(),
        ),
        "latent": summarize_stage(
            latent,
            max_components=max_components,
            exact=True,
            splits=("interior_border", "single_frame"),
        ),
    }
    return {
        "video": {
            "path": str(video_path),
            "frames_analyzed": int(captured["enc_b8"].shape[2] * 8 - 7),
            "latent_frames": int(captured["enc_b8"].shape[2]),
            "height": int(captured["enc_b8"].shape[3] * 32),
            "width": int(captured["enc_b8"].shape[4] * 32),
        },
        "stages": stages,
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "sample_id": record["sample_id"],
        "prompt_id": record["prompt_id"],
        "category": record["category"],
        "seed": record["seed"],
        **record.get("aux_metrics", {}),
    }
    for stage_name, stage in record["analysis"]["stages"].items():
        full = stage["full"]
        prefix = stage_name
        row[f"{prefix}_erank"] = full["entropy_effective_rank"]
        row[f"{prefix}_n98"] = full["n_comp_98"]
        row[f"{prefix}_cum4"] = full["cum_evr_4"]
        row[f"{prefix}_missing_var"] = full["missing_variance_ratio"]
        row[f"{prefix}_active_ch"] = stage["channels"]["frac_channels_active_1e-2xmax"]
        row[f"{prefix}_dc_frac"] = stage["channels"]["dc_fraction_of_total_energy"]
        row[f"{prefix}_rms"] = stage["channels"]["global_rms"]
        row[f"{prefix}_border_interior_energy"] = stage["border_interior_energy_ratio"]
    b8 = record["analysis"]["stages"]["enc_b8"]
    row["enc_b8_interior_n98"] = b8["interior_only"]["n_comp_98"]
    row["enc_b8_interior_cum4"] = b8["interior_only"]["cum_evr_4"]
    row["enc_b8_border_n98"] = b8["border_only"]["n_comp_98"]
    row["enc_b8_single_frame_n98"] = b8["single_frame"]["n_comp_98"]
    latent = record["analysis"]["stages"]["latent"]
    row["latent_interior_n98"] = latent["interior_only"]["n_comp_98"]
    row["latent_border_n98"] = latent["border_only"]["n_comp_98"]
    row["latent_single_frame_n98"] = latent["single_frame"]["n_comp_98"]
    return row


def write_outputs(records: list[dict[str, Any]], output: Path, started_at: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "num_records": len(records),
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = output.with_suffix(".csv")
    rows = [flatten_record(r) for r in records]
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-components", type=int, default=256)
    parser.add_argument(
        "--approx-b6",
        action="store_true",
        help="Use randomized top-k PCA for enc_b6. Other key 8x12/latent stages remain exact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_free_gpu()
    print(f"Device: {device}")
    print(f"Manifest: {args.manifest}")
    print(f"Video root: {args.video_root}")
    print(f"Output: {args.output}")

    manifest = json.loads(args.manifest.read_text())
    runs = manifest["runs"]
    if args.sample_id:
        wanted = set(args.sample_id)
        runs = [run for run in runs if run["sample_id"] in wanted]
    if args.limit is not None:
        runs = runs[: args.limit]

    encoder = load_video_vae_encoder(str(args.checkpoint), device=device, dtype=DTYPE)
    records: list[dict[str, Any]] = []
    started_at = time.time()

    for index, run in enumerate(runs, start=1):
        sample_id = run["sample_id"]
        prompt_id = sample_id.rsplit("_seed_", 1)[0]
        video_path = args.video_root / f"{sample_id}.mp4"
        print(f"[{index}/{len(runs)}] {sample_id}")
        analysis = analyze_video(
            encoder,
            video_path,
            max_frames=args.max_frames,
            max_components=args.max_components,
            b6_exact=not args.approx_b6,
            device=device,
        )
        record = {
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "category": run.get("category"),
            "seed": run.get("seed"),
            "prompt_file": run.get("prompt_file"),
            "assertions": run.get("assertions", []),
            "aux_metrics": load_aux_metrics(args.metrics_root, sample_id),
            "analysis": analysis,
        }
        records.append(record)
        write_outputs(records, args.output, started_at)
        summary = flatten_record(record)
        print(
            "  "
            f"b6 n98={summary['enc_b6_n98']} | "
            f"b8 n98={summary['enc_b8_n98']} interior={summary['enc_b8_interior_n98']} "
            f"border={summary['enc_b8_border_n98']} active={summary['enc_b8_active_ch']:.3f} | "
            f"latent n98={summary['latent_n98']} erank={summary['latent_erank']:.1f}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_outputs(records, args.output, started_at)
    print(f"Saved {args.output}")
    print(f"Saved {args.output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
