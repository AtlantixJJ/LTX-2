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
    python3 scripts/visualize_vae.py --max-analysis-frames 145
    python3 scripts/visualize_vae.py --sample-ids static_red_bicycle_seed_54321,dolly_blue_ball_seed_12345
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

# decord must be imported after torch has touched CUDA, otherwise the two
# libraries race to initialize the CUDA driver and the process segfaults.
torch.cuda.init()

import decord  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


decord.bridge.set_bridge("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))

from ltx_core.model.video_vae.ops import patchify  # noqa: E402
from ltx_trainer.model_loader import load_video_vae_decoder, load_video_vae_encoder  # noqa: E402
from ltx_trainer.video_utils import save_video  # noqa: E402


DTYPE = torch.bfloat16
# DNARendering source videos are shot at 2448x2048 (5MP), far larger than the
# ~256x384 expr/VBench clips this script was tuned for; encoding at native
# resolution tries to allocate >100GiB for a single conv3d. Downscale the long
# side to --max-side, rounded to a multiple of 32 for encoder compatibility.
RESIZE_MULTIPLE = 32
PCA_RGB_COMPONENTS = 3

# Chosen to cover static, motion, composition, and stress prompts plus low/high
# interior-rank cases found in the dataset analysis.
DEFAULT_SAMPLE_IDS = [
    "264_00001_Camera_5mp_cam025",
]

TAIL_STAGES = [
    ("post_pixelnorm", "after PixelNorm"),
    ("conv_out_means", "after conv_out means"),
    ("latent_normalized", "normalized latent"),
]
VARIANCE_CURVE_STAGES = {"post_pixelnorm", "conv_out_means", "latent_normalized"}


def build_stage_defs(encoder: torch.nn.Module) -> tuple[list[tuple[str, str]], set[str]]:
    """Enumerate one raw stage per encoder down_block, plus the fixed bottleneck tail."""
    block_stages = [(f"enc_b{i}", f"enc block {i} raw") for i in range(len(encoder.down_blocks))]
    component_heatmap_stages = {key for key, _ in block_stages}
    return block_stages + TAIL_STAGES, component_heatmap_stages


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors",
        help="VAE checkpoint path.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "DNARendering" / "Videos" / "Part_5" / "0264_01",
        help="Directory containing <sample_id>.mp4 videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "human",
        help="Output directory; wiped and recreated at the start of each run.",
    )
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Comma-separated sample ids (video basenames without .mp4). "
        f"Defaults to: {','.join(DEFAULT_SAMPLE_IDS)}",
    )
    parser.add_argument("--max-analysis-frames", type=int, default=145)
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Downscale the longer spatial side to this many pixels (rounded to a multiple of 32).",
    )
    parser.add_argument(
        "--crop-aspect-wh",
        default="1:1.5",
        help="Center-crop target aspect ratio as width:height, e.g. 1:1.5 for a portrait human capture.",
    )
    parser.add_argument("--num-heatmap-components", type=int, default=4)
    parser.add_argument("--num-heatmap-frames", type=int, default=4)
    parser.add_argument(
        "--diff-gain",
        type=float,
        default=4.0,
        help="Multiplier applied to |original - decoded| before clamping to [0, 1] for the diff panel.",
    )
    return parser


def sample_ids(args: argparse.Namespace) -> list[str]:
    if args.sample_ids:
        return [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    return DEFAULT_SAMPLE_IDS


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def center_crop_to_aspect(video: torch.Tensor, aspect_wh: str) -> torch.Tensor:
    w_ratio, h_ratio = (float(v) for v in aspect_wh.split(":"))
    target_w_over_h = w_ratio / h_ratio
    _, _, _, h, w = video.shape
    if w / h > target_w_over_h:
        new_h, new_w = h, max(1, round(h * target_w_over_h))
    else:
        new_h, new_w = max(1, round(w / target_w_over_h)), w
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return video[:, :, :, top : top + new_h, left : left + new_w]


def resize_video(video: torch.Tensor, max_side: int) -> torch.Tensor:
    _, _, _, h, w = video.shape
    scale = min(1.0, max_side / max(h, w))
    new_h = max(RESIZE_MULTIPLE, round(h * scale / RESIZE_MULTIPLE) * RESIZE_MULTIPLE)
    new_w = max(RESIZE_MULTIPLE, round(w * scale / RESIZE_MULTIPLE) * RESIZE_MULTIPLE)
    if new_h == h and new_w == w:
        return video
    frames = rearrange(video, "b c f h w -> (b f) c h w").float()
    frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return rearrange(frames.to(video.dtype), "(b f) c h w -> b c f h w", b=video.shape[0])


def video_fps(path: Path) -> float:
    return float(decord.VideoReader(str(path)).get_avg_fps())


def load_video(path: Path, device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    vr = decord.VideoReader(str(path))
    count = min(len(vr), args.max_analysis_frames)
    frames = vr.get_batch(range(count))
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(DTYPE).to(device)
    video = (video / 127.5) - 1.0
    video = center_crop_to_aspect(video, args.crop_aspect_wh)
    video = resize_video(video, args.max_side)
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


def plot_rgb_frames(rgb: np.ndarray, title: str, save_path: Path, num_heatmap_frames: int) -> None:
    indices = frame_indices(rgb.shape[0], num_heatmap_frames)
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


def plot_component_heatmaps(
    comps: np.ndarray, title: str, save_path: Path, num_heatmap_components: int, num_heatmap_frames: int
) -> None:
    ncomp = min(num_heatmap_components, comps.shape[-1])
    indices = frame_indices(comps.shape[0], num_heatmap_frames)
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


def plot_overview(records: list[dict[str, Any]], stages: list[tuple[str, str]], save_path: Path) -> None:
    fig, axes = plt.subplots(
        len(records),
        len(stages),
        figsize=(3.4 * len(stages), 2.4 * len(records)),
        squeeze=False,
    )
    for row, record in enumerate(records):
        sample_id = record["sample_id"]
        for col, (stage_key, stage_label) in enumerate(stages):
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
def encode_tail_stages(encoder: torch.nn.Module, video: torch.Tensor) -> dict[str, torch.Tensor]:
    x = encoder.conv_in(patchify(video, patch_size_hw=4, patch_size_t=1))
    stages: dict[str, torch.Tensor] = {}
    for index, block in enumerate(encoder.down_blocks):
        x = block(x)
        stages[f"enc_b{index}"] = x.detach()

    post_pixelnorm = encoder.conv_norm_out(x)
    conv_out_means = encoder.conv_out(encoder.conv_act(post_pixelnorm))[:, : encoder.latent_channels]
    latent = encoder.per_channel_statistics.normalize(conv_out_means)
    stages["post_pixelnorm"] = post_pixelnorm.detach()
    stages["conv_out_means"] = conv_out_means.detach()
    stages["latent_normalized"] = latent.detach()
    return stages


@torch.no_grad()
def decode_latent_to_video(decoder: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Decode a normalized latent back to pixel space as [F, C, H, W] in [0, 1]."""
    video = decoder(latent)  # [1, C, F, H, W], approximately in [-1, 1]
    video = rearrange(video, "1 c f h w -> f c h w")
    return ((video + 1) / 2).clamp(0, 1).float()


def prepare_pixel_video(video: torch.Tensor) -> torch.Tensor:
    """Convert the encoder-input video ([1, C, F, H, W] in [-1, 1]) to [F, C, H, W] in [0, 1]."""
    video = rearrange(video, "1 c f h w -> f c h w")
    return ((video + 1) / 2).clamp(0, 1).float()


def save_comparison_video(
    original: torch.Tensor, decoded: torch.Tensor, save_path: Path, fps: float, diff_gain: float
) -> None:
    """Save a side-by-side [original (resized) | decoded | |diff|*gain] comparison video."""
    num_frames = min(original.shape[0], decoded.shape[0])
    original = original[:num_frames]
    decoded = decoded[:num_frames]
    diff = ((original - decoded).abs() * diff_gain).clamp(0, 1)
    combined = torch.cat([original, decoded, diff], dim=-1)  # panels side by side along width
    save_video(combined, save_path, fps=fps, video_format="FCHW")


def visualize_stage(
    sample_dir: Path,
    sample_id: str,
    stage_key: str,
    stage_label: str,
    feature: torch.Tensor,
    args: argparse.Namespace,
    component_heatmap_stages: set[str],
) -> dict[str, Any]:
    n_pca = args.num_heatmap_components if stage_key in component_heatmap_stages else PCA_RGB_COMPONENTS
    x, comps = pca_projection(feature, n_pca)
    rgb = rgb_from_components(comps)
    rank = exact_rank_metrics(x)
    stats = channel_stats(x)

    prefix = sample_dir / f"{sample_id}_{stage_key}"
    plot_rgb_frames(
        rgb,
        f"{sample_id}: {stage_label} top-3 PCA RGB",
        prefix.with_name(prefix.name + "_rgb_frames.png"),
        args.num_heatmap_frames,
    )
    if stage_key in component_heatmap_stages:
        plot_component_heatmaps(
            comps,
            f"{sample_id}: {stage_label} PCA component heatmaps",
            prefix.with_name(prefix.name + "_component_heatmaps.png"),
            args.num_heatmap_components,
            args.num_heatmap_frames,
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
    args = build_arg_parser().parse_args()
    prepare_output_dir(args.output_dir)
    device = torch.device("cuda")
    selected = sample_ids(args)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Video root: {args.video_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Videos: {', '.join(selected)}")

    encoder = load_video_vae_encoder(str(args.checkpoint), device=device, dtype=DTYPE)
    decoder = load_video_vae_decoder(str(args.checkpoint), device=device, dtype=DTYPE)
    stages_def, component_heatmap_stages = build_stage_defs(encoder)
    overview_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "video_root": str(args.video_root),
        "max_analysis_frames": args.max_analysis_frames,
        "sample_ids": selected,
        "records": [],
    }

    for sample_id in selected:
        video_path = args.video_root / f"{sample_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print(f"Visualizing {sample_id}")
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        video = load_video(video_path, device, args)
        stages = encode_tail_stages(encoder, video)

        overview_stage_records: dict[str, Any] = {}
        sample_summary: dict[str, Any] = {
            "sample_id": sample_id,
            "video_path": str(video_path),
            "stages": {},
        }
        for stage_key, stage_label in stages_def:
            stage_record = visualize_stage(
                sample_dir, sample_id, stage_key, stage_label, stages[stage_key], args, component_heatmap_stages
            )
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

        decoded = decode_latent_to_video(decoder, stages["latent_normalized"])
        original = prepare_pixel_video(video)
        comparison_path = sample_dir / f"{sample_id}_comparison.mp4"
        save_comparison_video(original, decoded, comparison_path, video_fps(video_path), args.diff_gain)
        print(f"  saved comparison video (original | decoded | diff) to {comparison_path}")

        overview_records.append({"sample_id": sample_id, "stages": overview_stage_records})
        summary["records"].append(sample_summary)
        (sample_dir / f"{sample_id}_summary.json").write_text(json.dumps(sample_summary, indent=2) + "\n")

        del video, decoded, original, stages
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_overview(overview_records, stages_def, args.output_dir / "tail_pca_overview.png")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved overview to {args.output_dir / 'tail_pca_overview.png'}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
