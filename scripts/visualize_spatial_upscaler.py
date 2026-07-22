"""
Probe the LTX-2.3 spatial latent upscaler (`ltx-2.3-spatial-upscaler-x2-*.safetensors`).

For each input video this script builds a real "does the learned upsampler beat naive
bicubic" test using genuine higher-resolution ground truth (downsampled from the native
video, never upsampled):

  1. picks a `high` resolution (<= --max-high-side, multiple of 64, never exceeding the
     video's native resolution) and a `low` resolution at exactly half of it;
  2. encodes the `low`-resolution video with the video VAE encoder to get a latent;
  3. decodes that latent directly -> "baseline" reconstruction at `low` resolution;
  4. runs the latent through `LatentUpsampler` (un-normalize -> upsample -> re-normalize,
     via `ltx_core.model.upsampler.upsample_video`) then decodes -> "learned" 2x
     reconstruction at `high` resolution;
  5. bicubic-upsamples the baseline reconstruction to `high` resolution -> "naive" 2x
     reconstruction;
  6. compares both "learned" and "naive" against the real `high`-resolution ground truth
     (bicubic-downsampled from the native video) via PSNR, and saves a side-by-side
     [ground truth | learned upsample | naive bicubic | |learned-diff|] comparison video.

Also prints/saves the upsampler's architecture: the config embedded in the safetensors
header, per-submodule parameter counts, and which forward-pass branch it takes.

Run:
    conda run -n ltx python3 scripts/visualize_spatial_upscaler.py

Useful overrides:
    conda run -n ltx python3 scripts/visualize_spatial_upscaler.py --max-high-side 640
    conda run -n ltx python3 scripts/visualize_spatial_upscaler.py --videos /path/a.mp4,/path/b.mp4
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
from safetensors import safe_open  # noqa: E402


decord.bridge.set_bridge("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder  # noqa: E402
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video  # noqa: E402
from ltx_trainer.model_loader import load_video_vae_decoder, load_video_vae_encoder  # noqa: E402
from ltx_trainer.video_utils import save_video  # noqa: E402


DTYPE = torch.bfloat16
RESIZE_MULTIPLE = 64  # so that `high` and `high // 2` are both multiples of 32 (VAE constraint)

DEFAULT_VIDEO_PATHS = [
    WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw" / "static_green_teapot_seed_12345.mp4",
    WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw" / "dolly_blue_ball_seed_12345.mp4",
    WORKSPACE_ROOT / "data" / "DNARendering" / "Videos" / "Part_5" / "0264_01" / "264_00001_Camera_5mp_cam025.mp4",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors",
        help="Checkpoint containing the video VAE encoder/decoder weights.",
    )
    parser.add_argument(
        "--upsampler-checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        help="Spatial latent upscaler checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "spatial_upscaler_probe",
        help="Output directory; wiped and recreated at the start of each run.",
    )
    parser.add_argument(
        "--videos",
        default=None,
        help=f"Comma-separated video file paths. Defaults to {len(DEFAULT_VIDEO_PATHS)} built-in sample videos.",
    )
    parser.add_argument("--max-analysis-frames", type=int, default=49)
    parser.add_argument(
        "--max-high-side",
        type=int,
        default=512,
        help="Cap for the ground-truth/learned-upsample resolution's longer side (multiple of 64, "
        "never exceeds the video's native resolution). The encoder input is exactly half of this.",
    )
    parser.add_argument("--num-heatmap-frames", type=int, default=4)
    parser.add_argument(
        "--diff-gain",
        type=float,
        default=4.0,
        help="Multiplier applied to |ground_truth - learned| before clamping to [0, 1] for the diff panel.",
    )
    return parser


def video_paths(args: argparse.Namespace) -> list[Path]:
    if args.videos:
        return [Path(item.strip()).expanduser() for item in args.videos.split(",") if item.strip()]
    return DEFAULT_VIDEO_PATHS


def unique_sample_id(video_path: Path, used_ids: set[str]) -> str:
    sample_id = video_path.stem
    if sample_id in used_ids:
        sample_id = f"{video_path.parent.name}_{sample_id}"
    base, suffix = sample_id, 2
    while sample_id in used_ids:
        sample_id = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(sample_id)
    return sample_id


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def high_low_dims(h: int, w: int, max_high_side: int) -> tuple[int, int, int, int]:
    """Pick (high_h, high_w, low_h, low_w), all multiples of 32, high never exceeding native size."""
    scale = min(1.0, max_high_side / max(h, w))
    high_h = max(RESIZE_MULTIPLE, (int(h * scale) // RESIZE_MULTIPLE) * RESIZE_MULTIPLE)
    high_w = max(RESIZE_MULTIPLE, (int(w * scale) // RESIZE_MULTIPLE) * RESIZE_MULTIPLE)
    return high_h, high_w, high_h // 2, high_w // 2


def resize_to(video: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bicubic-resize a [1, C, F, H, W] video to exactly (h, w)."""
    frames = rearrange(video, "b c f h w -> (b f) c h w").float()
    frames = F.interpolate(frames, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return rearrange(frames.to(video.dtype), "(b f) c h w -> b c f h w", b=video.shape[0])


def video_fps(path: Path) -> float:
    return float(decord.VideoReader(str(path)).get_avg_fps())


def load_native_video(path: Path, device: torch.device, max_analysis_frames: int) -> torch.Tensor:
    vr = decord.VideoReader(str(path))
    count = min(len(vr), max_analysis_frames)
    frames = vr.get_batch(range(count))
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(DTYPE).to(device)
    video = (video / 127.5) - 1.0
    valid_f = ((video.shape[2] - 1) // 8) * 8 + 1
    return video[:, :, :valid_f]


def to_pixel_uint_range(video: torch.Tensor) -> torch.Tensor:
    """[1, C, F, H, W] in [-1, 1] -> [F, C, H, W] in [0, 1]."""
    video = rearrange(video, "1 c f h w -> f c h w")
    return ((video + 1) / 2).clamp(0, 1).float()


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """PSNR between two [F, C, H, W] tensors in [0, 1]."""
    num_frames = min(a.shape[0], b.shape[0])
    a = a[:num_frames].float()
    b = b[:num_frames].float()
    per_frame_mse = ((a - b) ** 2).flatten(1).mean(dim=1)
    overall_mse = float(per_frame_mse.mean())
    return {
        "overall_psnr_db": float(10 * np.log10(1.0 / max(overall_mse, 1e-12))),
        "overall_mse": overall_mse,
        "per_frame_psnr_db": [float(10 * np.log10(1.0 / max(float(v), 1e-12))) for v in per_frame_mse],
    }


def frame_indices(num_frames: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, num_frames - 1, min(count, num_frames)).astype(int).tolist()))


def save_comparison_video(
    ground_truth: torch.Tensor,
    learned: torch.Tensor,
    naive: torch.Tensor,
    save_path: Path,
    fps: float,
    diff_gain: float,
) -> None:
    """Save [ground truth | learned upsample | naive bicubic | |learned-diff|*gain] side by side."""
    num_frames = min(ground_truth.shape[0], learned.shape[0], naive.shape[0])
    ground_truth = ground_truth[:num_frames]
    learned = learned[:num_frames]
    naive = naive[:num_frames]
    diff = ((ground_truth - learned).abs() * diff_gain).clamp(0, 1)
    combined = torch.cat([ground_truth, learned, naive, diff], dim=-1)
    save_video(combined, save_path, fps=fps, video_format="FCHW")


def plot_frame_grid(
    panels: dict[str, torch.Tensor], title: str, save_path: Path, num_heatmap_frames: int
) -> None:
    names = list(panels.keys())
    num_frames = next(iter(panels.values())).shape[0]
    indices = frame_indices(num_frames, num_heatmap_frames)
    fig, axes = plt.subplots(
        len(names), len(indices), figsize=(3.0 * len(indices), 3.0 * len(names)), squeeze=False
    )
    for row, name in enumerate(names):
        frames = panels[name]
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(frames[idx].permute(1, 2, 0).cpu().numpy())
            if row == 0:
                ax.set_title(f"F{idx}")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(name, fontsize=9)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_psnr_comparison(records: list[dict[str, Any]], save_path: Path) -> None:
    labels = [r["sample_id"] for r in records]
    learned_vals = [r["psnr_learned_vs_gt"]["overall_psnr_db"] for r in records]
    naive_vals = [r["psnr_naive_vs_gt"]["overall_psnr_db"] for r in records]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(max(6.0, 1.6 * len(labels)), 4.5))
    bars_l = plt.bar(x - width / 2, learned_vals, width, label="learned upsampler", color="#4C72B0")
    bars_n = plt.bar(x + width / 2, naive_vals, width, label="naive bicubic", color="#DD8452")
    for bar, value in zip(list(bars_l) + list(bars_n), learned_vals + naive_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("PSNR vs real high-res ground truth (dB)")
    plt.title("Learned spatial upsampler vs naive bicubic 2x")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def read_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    with safe_open(str(checkpoint_path), framework="pt") as f:
        meta = f.metadata() or {}
        config = json.loads(meta["config"]) if "config" in meta else {}
        tensor_shapes = {key: list(f.get_slice(key).get_shape()) for key in f.keys()}
    return {"config": config, "tensor_shapes": tensor_shapes}


def describe_architecture(upsampler: torch.nn.Module, checkpoint_info: dict[str, Any]) -> dict[str, Any]:
    per_module_params = {
        name: sum(p.numel() for p in child.parameters())
        for name, child in upsampler.named_children()
    }
    total_params = sum(per_module_params.values())

    config = checkpoint_info["config"]
    if config.get("spatial_upsample") and config.get("temporal_upsample"):
        branch = "spatiotemporal: Conv3d(mid, 8*mid) + PixelShuffleND(3) over (d,h,w) jointly"
    elif config.get("spatial_upsample") and config.get("rational_resampler"):
        branch = "rational resampler: per-frame Conv2d + PixelShuffleND(2) by num + BlurDownsample by den"
    elif config.get("spatial_upsample"):
        branch = (
            "spatial-only (this checkpoint): dims=3 ResBlocks/convs, but the upsample stage itself "
            "reshapes 'b c f h w -> (b f) c h w' and applies a per-frame Conv2d(mid, 4*mid) + "
            "PixelShuffleND(2), i.e. plain 2D pixel-shuffle upsampling applied independently per frame "
            "(no temporal mixing in the upsample step itself, only in the surrounding Conv3d ResBlocks)."
        )
    elif config.get("temporal_upsample"):
        branch = "temporal-only: Conv3d(mid, 2*mid) + PixelShuffleND(1), drops the first (duplicated) frame"
    else:
        branch = "unknown"

    return {
        "config": config,
        "forward_branch": branch,
        "total_params": total_params,
        "total_params_millions": round(total_params / 1e6, 1),
        "per_module_params": per_module_params,
    }


def print_architecture_summary(summary: dict[str, Any], checkpoint_path: Path) -> None:
    print(f"\n=== LatentUpsampler architecture ({checkpoint_path.name}) ===")
    print("Config (from safetensors metadata):")
    for key, value in summary["config"].items():
        print(f"  {key}: {value}")
    print(f"\nForward-pass branch taken: {summary['forward_branch']}")
    print(f"\nTotal parameters: {summary['total_params']:,} ({summary['total_params_millions']}M)")
    print("Per top-level submodule:")
    for name, count in summary["per_module_params"].items():
        print(f"  {name:<28} {count:>12,}")


@torch.no_grad()
def process_video(
    video_path: Path,
    sample_id: str,
    sample_dir: Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    upsampler: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    native = load_native_video(video_path, device, args.max_analysis_frames)
    _, _, _, native_h, native_w = native.shape
    high_h, high_w, low_h, low_w = high_low_dims(native_h, native_w, args.max_high_side)
    is_real_gt = high_h <= native_h and high_w <= native_w
    print(
        f"  native={native_h}x{native_w} -> encoder input (low)={low_h}x{low_w}, "
        f"high/ground-truth={high_h}x{high_w} (real ground truth: {is_real_gt})"
    )

    ground_truth_video = resize_to(native, high_h, high_w)
    low_video = resize_to(native, low_h, low_w)

    latent = encoder(low_video)  # normalized latent, [1, 128, F', low_h/32, low_w/32]
    baseline_decoded = decoder(latent)  # [1, 3, F, low_h, low_w]

    upsampled_latent = upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)
    learned_decoded = decoder(upsampled_latent)  # [1, 3, F, high_h, high_w]

    naive_decoded = resize_to(((baseline_decoded.float().clamp(-1, 1))), high_h, high_w).to(DTYPE)

    ground_truth_px = to_pixel_uint_range(ground_truth_video)
    learned_px = to_pixel_uint_range(learned_decoded)
    naive_px = to_pixel_uint_range(naive_decoded)
    baseline_px = to_pixel_uint_range(baseline_decoded)

    psnr_learned = compute_psnr(ground_truth_px, learned_px)
    psnr_naive = compute_psnr(ground_truth_px, naive_px)

    comparison_path = sample_dir / f"{sample_id}_comparison.mp4"
    save_comparison_video(ground_truth_px, learned_px, naive_px, comparison_path, video_fps(video_path), args.diff_gain)

    plot_frame_grid(
        {"ground truth (high)": ground_truth_px, "learned upsample": learned_px, "naive bicubic": naive_px},
        f"{sample_id}: {low_h}x{low_w} -> {high_h}x{high_w}",
        sample_dir / f"{sample_id}_frames.png",
        args.num_heatmap_frames,
    )

    print(
        f"  baseline (low-res direct decode) shape={tuple(baseline_px.shape)}\n"
        f"  learned-upsampled decode shape={tuple(learned_px.shape)}\n"
        f"  PSNR vs ground truth: learned={psnr_learned['overall_psnr_db']:.2f} dB, "
        f"naive bicubic={psnr_naive['overall_psnr_db']:.2f} dB "
        f"(delta={psnr_learned['overall_psnr_db'] - psnr_naive['overall_psnr_db']:+.2f} dB)\n"
        f"  saved comparison video to {comparison_path}"
    )

    del native, ground_truth_video, low_video, latent, baseline_decoded, upsampled_latent, learned_decoded
    del naive_decoded, ground_truth_px, learned_px, naive_px, baseline_px
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "sample_id": sample_id,
        "video_path": str(video_path),
        "native_hw": [native_h, native_w],
        "low_hw": [low_h, low_w],
        "high_hw": [high_h, high_w],
        "is_real_ground_truth": is_real_gt,
        "psnr_learned_vs_gt": psnr_learned,
        "psnr_naive_vs_gt": psnr_naive,
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    prepare_output_dir(args.output_dir)
    device = torch.device("cuda")
    selected = video_paths(args)
    for video_path in selected:
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if not args.vae_checkpoint.exists():
            raise FileNotFoundError(args.vae_checkpoint)
        if not args.upsampler_checkpoint.exists():
            raise FileNotFoundError(args.upsampler_checkpoint)

    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Upsampler checkpoint: {args.upsampler_checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print("Videos:\n  " + "\n  ".join(str(p) for p in selected))

    checkpoint_info = read_checkpoint_config(args.upsampler_checkpoint)

    encoder = load_video_vae_encoder(str(args.vae_checkpoint), device=device, dtype=DTYPE)
    decoder = load_video_vae_decoder(str(args.vae_checkpoint), device=device, dtype=DTYPE)
    upsampler = (
        SingleGPUModelBuilder(
            model_path=str(args.upsampler_checkpoint),
            model_class_configurator=LatentUpsamplerConfigurator,
        )
        .build(device=device, dtype=DTYPE)
        .eval()
    )

    arch_summary = describe_architecture(upsampler, checkpoint_info)
    print_architecture_summary(arch_summary, args.upsampler_checkpoint)

    records: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for video_path in selected:
        sample_id = unique_sample_id(video_path, used_ids)
        print(f"\nProcessing {sample_id} ({video_path})")
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        records.append(process_video(video_path, sample_id, sample_dir, encoder, decoder, upsampler, device, args))

    print("\nPSNR comparison (learned vs naive bicubic, worst-learned-delta first):")
    for rec in sorted(
        records, key=lambda r: r["psnr_learned_vs_gt"]["overall_psnr_db"] - r["psnr_naive_vs_gt"]["overall_psnr_db"]
    ):
        delta = rec["psnr_learned_vs_gt"]["overall_psnr_db"] - rec["psnr_naive_vs_gt"]["overall_psnr_db"]
        print(
            f"  {rec['sample_id']:<28} learned={rec['psnr_learned_vs_gt']['overall_psnr_db']:6.2f} dB  "
            f"naive={rec['psnr_naive_vs_gt']['overall_psnr_db']:6.2f} dB  delta={delta:+.2f} dB"
        )

    plot_psnr_comparison(records, args.output_dir / "psnr_comparison.png")

    summary = {
        "vae_checkpoint": str(args.vae_checkpoint),
        "upsampler_checkpoint": str(args.upsampler_checkpoint),
        "architecture": arch_summary,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved PSNR comparison plot to {args.output_dir / 'psnr_comparison.png'}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
