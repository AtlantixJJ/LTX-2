"""
Focused VAE tail visualization over multiple expr/VBench videos.

The previous version of this script produced a broad single-video sweep over
every encoder/decoder block. This version is intentionally narrower:

  1. visualize five different generated videos from the expr/VBench run;
  2. compare PCA maps at the bottleneck tail:
       enc_b8 raw -> post PixelNorm -> raw conv_out means -> normalized latent.

Outputs are written under results/vae_multivideo_tail_pca and the directory is
cleaned at the start of each run.

Run:
    python3 scripts/visualize_vae.py

Useful overrides:
    MAX_ANALYSIS_FRAMES=145 python3 scripts/visualize_vae.py
    VAE_VIS_SAMPLE_IDS=static_red_bicycle_seed_54321,dolly_blue_ball_seed_12345 python3 scripts/visualize_vae.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import decord
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from sklearn.decomposition import PCA


decord.bridge.set_bridge("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))

from ltx_core.model.video_vae.ops import patchify  # noqa: E402
from ltx_trainer.model_loader import load_video_vae_encoder  # noqa: E402


CHECKPOINT_PATH = Path(
    os.environ.get(
        "CHECKPOINT_PATH",
        str(WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors"),
    )
)
VIDEO_ROOT = Path(
    os.environ.get("VIDEO_ROOT", str(WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw"))
)
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", str(REPO_ROOT / "results" / "vae_multivideo_tail_pca"))
)

DTYPE = torch.bfloat16
MAX_ANALYSIS_FRAMES = int(os.environ.get("MAX_ANALYSIS_FRAMES", "145"))
NUM_HEATMAP_COMPONENTS = int(os.environ.get("NUM_HEATMAP_COMPONENTS", "4"))
NUM_HEATMAP_FRAMES = int(os.environ.get("NUM_HEATMAP_FRAMES", "4"))
PCA_RGB_COMPONENTS = 3

# Chosen to cover static, motion, composition, and stress prompts plus low/high
# interior-rank cases found in the dataset analysis.
DEFAULT_SAMPLE_IDS = [
    "static_green_teapot_seed_12345",
    "static_red_bicycle_seed_54321",
    "dolly_blue_ball_seed_12345",
    "composition_books_seed_12345",
    "cat_behind_box_seed_54321",
]

STAGES = [
    ("enc_b8", "enc_b8 raw"),
    ("post_pixelnorm", "after PixelNorm"),
    ("conv_out_means", "after conv_out means"),
    ("latent_normalized", "normalized latent"),
]
COMPONENT_HEATMAP_STAGES = {"enc_b8"}
VARIANCE_CURVE_STAGES = {"post_pixelnorm", "conv_out_means", "latent_normalized"}


def select_free_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("No CUDA device available, falling back to CPU.")
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
        idx, mem_used, util = min(rows, key=lambda r: (r[1], r[2]))
        print(f"Selected GPU {idx} (memory.used={mem_used} MiB, utilization={util}%)")
        return torch.device(f"cuda:{idx}")
    except Exception as exc:
        print(f"GPU auto-selection failed ({exc}); falling back to cuda:0")
        return torch.device("cuda:0")


def sample_ids() -> list[str]:
    override = os.environ.get("VAE_VIS_SAMPLE_IDS")
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    return DEFAULT_SAMPLE_IDS


def prepare_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_video(path: Path, device: torch.device) -> torch.Tensor:
    vr = decord.VideoReader(str(path))
    count = min(len(vr), MAX_ANALYSIS_FRAMES)
    frames = vr.get_batch(range(count))
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(DTYPE).to(device)
    video = (video / 127.5) - 1.0
    valid_f = ((video.shape[2] - 1) // 8) * 8 + 1
    return video[:, :, :valid_f]


def rows_from_feature(feature: torch.Tensor) -> np.ndarray:
    return rearrange(feature, "b c f h w -> (b f h w) c").detach().cpu().float().numpy()


def exact_rank_metrics(x: np.ndarray) -> dict[str, Any]:
    if x.shape[0] < 2:
        return {
            "entropy_effective_rank": 0.0,
            "participation_ratio": 0.0,
            "n_comp_90": None,
            "n_comp_98": None,
            "top4_cum_evr": 0.0,
            "evr_top8": [],
        }

    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = x.astype(np.float32, copy=False) - mean
    cov = (centered.T @ centered).astype(np.float64) / float(x.shape[0] - 1)
    eigvals = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)
    ratio = eigvals / (float(eigvals.sum()) + 1e-12)
    cum = np.cumsum(ratio)
    nz = ratio[ratio > 0]

    def n_for(threshold: float) -> int | None:
        if not len(cum) or cum[-1] < threshold:
            return None
        return int(np.searchsorted(cum, threshold) + 1)

    return {
        "entropy_effective_rank": float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0,
        "participation_ratio": float((eigvals.sum() ** 2) / (np.sum(eigvals**2) + 1e-12)),
        "n_comp_90": n_for(0.90),
        "n_comp_98": n_for(0.98),
        "top4_cum_evr": float(cum[min(3, len(cum) - 1)]) if len(cum) else 0.0,
        "evr_top8": ratio[:8].astype(float).tolist(),
        "cumulative_evr_top50": cum[:50].astype(float).tolist(),
    }


def channel_stats(x: np.ndarray) -> dict[str, Any]:
    mean = x.mean(axis=0, dtype=np.float64)
    var = x.var(axis=0, dtype=np.float64)
    total_var = float(var.sum())
    dc = float((mean**2).sum())
    return {
        "active_channel_fraction_1e-2xmax": float((var > 1e-2 * var.max()).mean()) if var.size else 0.0,
        "dc_fraction_of_total_energy": float(dc / (dc + total_var + 1e-12)),
        "global_rms": float(np.sqrt(np.mean(var + mean**2))),
        "global_min": float(x.min()),
        "global_max": float(x.max()),
    }


def pca_projection(feature: torch.Tensor, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    b, _, f, h, w = feature.shape
    x = rows_from_feature(feature)
    n = min(n_components, x.shape[0], x.shape[1])
    projected = PCA(n_components=n, svd_solver="full").fit_transform(x)
    comps = rearrange(projected, "(b f h w) c -> b f h w c", b=b, f=f, h=h, w=w)[0]
    return x, comps


def rgb_from_components(comps: np.ndarray) -> np.ndarray:
    if comps.shape[-1] >= 3:
        rgb = comps[..., :3]
    else:
        rgb = np.zeros((*comps.shape[:-1], 3), dtype=comps.dtype)
        rgb[..., : comps.shape[-1]] = comps
    lo = rgb.min(axis=(0, 1, 2), keepdims=True)
    hi = rgb.max(axis=(0, 1, 2), keepdims=True)
    return (rgb - lo) / (hi - lo + 1e-8)


def frame_indices(num_frames: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, num_frames - 1, min(count, num_frames)).astype(int).tolist()))


def plot_rgb_frames(rgb: np.ndarray, title: str, save_path: Path) -> None:
    indices = frame_indices(rgb.shape[0], NUM_HEATMAP_FRAMES)
    fig, axes = plt.subplots(1, len(indices), figsize=(3.2 * len(indices), 3.0), squeeze=False)
    for col, idx in enumerate(indices):
        ax = axes[0, col]
        ax.imshow(rgb[idx])
        ax.set_title(f"F{idx}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_component_heatmaps(comps: np.ndarray, title: str, save_path: Path) -> None:
    ncomp = min(NUM_HEATMAP_COMPONENTS, comps.shape[-1])
    indices = frame_indices(comps.shape[0], NUM_HEATMAP_FRAMES)
    fig, axes = plt.subplots(ncomp, len(indices), figsize=(3.2 * len(indices), 2.8 * ncomp), squeeze=False)
    for row in range(ncomp):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            im = ax.imshow(comps[idx, :, :, row], cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"F{idx}")
            if col == 0:
                ax.set_ylabel(f"PC{row}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_variance_curve(rank: dict[str, Any], title: str, save_path: Path) -> None:
    cumulative = np.asarray(rank["cumulative_evr_top50"], dtype=np.float64)
    plt.figure(figsize=(5.5, 3.8))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", ms=3)
    plt.ylim(0.0, 1.02)
    plt.xlabel("PCA components")
    plt.ylabel("cumulative explained variance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_overview(records: list[dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(
        len(records),
        len(STAGES),
        figsize=(3.4 * len(STAGES), 2.4 * len(records)),
        squeeze=False,
    )
    for row, record in enumerate(records):
        sample_id = record["sample_id"]
        for col, (stage_key, stage_label) in enumerate(STAGES):
            stage = record["stages"][stage_key]
            rgb = stage["rgb"]
            frame = rgb.shape[0] // 2
            ax = axes[row, col]
            ax.imshow(rgb[frame])
            rank = stage["rank"]
            ax.set_title(
                f"{stage_label}\nn98={rank['n_comp_98']} cum4={rank['top4_cum_evr']:.3f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(sample_id, fontsize=8)
    fig.suptitle("VAE tail PCA overview: middle latent frame for five expr/VBench videos")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


@torch.no_grad()
def encode_tail_stages(encoder: torch.nn.Module, video_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    video = load_video(video_path, device)
    x = encoder.conv_in(patchify(video, patch_size_hw=4, patch_size_t=1))
    del video
    for index, block in enumerate(encoder.down_blocks):
        x = block(x)
        if index == 8:
            b8 = x

    post_pixelnorm = encoder.conv_norm_out(b8)
    conv_out_means = encoder.conv_out(encoder.conv_act(post_pixelnorm))[:, : encoder.latent_channels]
    latent = encoder.per_channel_statistics.normalize(conv_out_means)
    return {
        "enc_b8": b8.detach(),
        "post_pixelnorm": post_pixelnorm.detach(),
        "conv_out_means": conv_out_means.detach(),
        "latent_normalized": latent.detach(),
    }


def visualize_stage(sample_dir: Path, sample_id: str, stage_key: str, stage_label: str, feature: torch.Tensor) -> dict[str, Any]:
    n_pca = NUM_HEATMAP_COMPONENTS if stage_key in COMPONENT_HEATMAP_STAGES else PCA_RGB_COMPONENTS
    x, comps = pca_projection(feature, n_pca)
    rgb = rgb_from_components(comps)
    rank = exact_rank_metrics(x)
    stats = channel_stats(x)

    prefix = sample_dir / f"{sample_id}_{stage_key}"
    plot_rgb_frames(rgb, f"{sample_id}: {stage_label} top-3 PCA RGB", prefix.with_name(prefix.name + "_rgb_frames.png"))
    if stage_key in COMPONENT_HEATMAP_STAGES:
        plot_component_heatmaps(
            comps,
            f"{sample_id}: {stage_label} PCA component heatmaps",
            prefix.with_name(prefix.name + "_component_heatmaps.png"),
        )
    if stage_key in VARIANCE_CURVE_STAGES:
        plot_variance_curve(
            rank,
            f"{sample_id}: {stage_label} PCA variance",
            prefix.with_name(prefix.name + "_variance.png"),
        )

    return {
        "shape_BCFHW": [int(v) for v in feature.shape],
        "rank": rank,
        "channels": stats,
        "rgb": rgb,
    }


def main() -> int:
    prepare_output_dir()
    device = select_free_gpu()
    selected = sample_ids()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Video root: {VIDEO_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Videos: {', '.join(selected)}")

    encoder = load_video_vae_encoder(str(CHECKPOINT_PATH), device=device, dtype=DTYPE)
    overview_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checkpoint": str(CHECKPOINT_PATH),
        "video_root": str(VIDEO_ROOT),
        "max_analysis_frames": MAX_ANALYSIS_FRAMES,
        "sample_ids": selected,
        "records": [],
    }

    for sample_id in selected:
        video_path = VIDEO_ROOT / f"{sample_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print(f"Visualizing {sample_id}")
        sample_dir = OUTPUT_DIR / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        stages = encode_tail_stages(encoder, video_path, device)

        overview_stage_records: dict[str, Any] = {}
        sample_summary: dict[str, Any] = {
            "sample_id": sample_id,
            "video_path": str(video_path),
            "stages": {},
        }
        for stage_key, stage_label in STAGES:
            stage_record = visualize_stage(sample_dir, sample_id, stage_key, stage_label, stages[stage_key])
            overview_stage_records[stage_key] = {
                "rgb": stage_record.pop("rgb"),
                "rank": stage_record["rank"],
            }
            sample_summary["stages"][stage_key] = stage_record
            print(
                "  "
                f"{stage_key}: shape={sample_summary['stages'][stage_key]['shape_BCFHW']} "
                f"n98={stage_record['rank']['n_comp_98']} "
                f"cum4={stage_record['rank']['top4_cum_evr']:.3f} "
                f"erank={stage_record['rank']['entropy_effective_rank']:.1f} "
                f"active={stage_record['channels']['active_channel_fraction_1e-2xmax']:.3f}"
            )

        overview_records.append({"sample_id": sample_id, "stages": overview_stage_records})
        summary["records"].append(sample_summary)
        (sample_dir / f"{sample_id}_summary.json").write_text(json.dumps(sample_summary, indent=2) + "\n")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_overview(overview_records, OUTPUT_DIR / "tail_pca_overview.png")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved overview to {OUTPUT_DIR / 'tail_pca_overview.png'}")
    print(f"Saved summary to {OUTPUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
