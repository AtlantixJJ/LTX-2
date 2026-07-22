"""
Zero out the spatial border ring of a video's VAE latent and compare decodes.

Encodes a video to its normalized latent, then zeroes a ring of tokens along
the spatial (H, W) border of every latent frame -- setting that ring to 0.0,
i.e. the per-channel mean in normalized latent space, which is the "empty"
value once un-normalized. Decodes both the untouched and the boundary-removed
latent and writes a side-by-side comparison video:

    original | original decode | modified decode

Video loading follows scripts/visualize_vae.py (same default sample/video root).

Run:
    python3 scripts/vae_latent_boundary_removal.py
    python3 scripts/vae_latent_boundary_removal.py --ring-width 4
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

# decord must be imported after torch has touched CUDA, otherwise the two
# libraries race to initialize the CUDA driver and the process segfaults.
torch.cuda.init()

import decord  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402

decord.bridge.set_bridge("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))

from ltx_core.model.video_vae.ops import patchify  # noqa: E402
from ltx_trainer.model_loader import load_video_vae_decoder, load_video_vae_encoder  # noqa: E402
from ltx_trainer.video_utils import save_video  # noqa: E402


DTYPE = torch.bfloat16
RESIZE_MULTIPLE = 32
DEFAULT_SAMPLE_IDS = [
    "264_00001_Camera_5mp_cam025",
]


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
        default=REPO_ROOT / "results" / "vae_latent_boundary_removal",
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
    parser.add_argument(
        "--ring-width",
        type=int,
        default=1,
        help="Thickness in latent tokens of the spatial border ring to zero out on every latent frame.",
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


@torch.no_grad()
def encode_to_latent(encoder: torch.nn.Module, video: torch.Tensor) -> torch.Tensor:
    """Encode a pixel video ([1, C, F, H, W] in [-1, 1]) to its normalized latent."""
    x = encoder.conv_in(patchify(video, patch_size_hw=4, patch_size_t=1))
    for block in encoder.down_blocks:
        x = block(x)
    post_pixelnorm = encoder.conv_norm_out(x)
    conv_out_means = encoder.conv_out(encoder.conv_act(post_pixelnorm))[:, : encoder.latent_channels]
    return encoder.per_channel_statistics.normalize(conv_out_means).detach()


def zero_spatial_boundary(latent: torch.Tensor, ring_width: int) -> torch.Tensor:
    """Zero the outer ring_width tokens of every latent frame along H and W.

    0.0 is the per-channel mean in this normalized space (see PerChannelStatistics),
    i.e. the "empty" fill value once the decoder un-normalizes it.
    """
    modified = latent.clone()
    if ring_width <= 0:
        return modified
    h, w = modified.shape[-2], modified.shape[-1]
    rh = min(ring_width, (h + 1) // 2)
    rw = min(ring_width, (w + 1) // 2)
    modified[..., :rh, :] = 0.0
    modified[..., h - rh :, :] = 0.0
    modified[..., :, :rw] = 0.0
    modified[..., :, w - rw :] = 0.0
    return modified


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
    original: torch.Tensor,
    original_decoded: torch.Tensor,
    modified_decoded: torch.Tensor,
    save_path: Path,
    fps: float,
) -> None:
    """Save a side-by-side [original | original decode | modified decode] video."""
    num_frames = min(original.shape[0], original_decoded.shape[0], modified_decoded.shape[0])
    original = original[:num_frames]
    original_decoded = original_decoded[:num_frames]
    modified_decoded = modified_decoded[:num_frames]
    combined = torch.cat([original, original_decoded, modified_decoded], dim=-1)
    save_video(combined, save_path, fps=fps, video_format="FCHW")


def main() -> int:
    args = build_arg_parser().parse_args()
    prepare_output_dir(args.output_dir)
    device = torch.device("cuda")
    selected = sample_ids(args)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Video root: {args.video_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Ring width: {args.ring_width} latent tokens")
    print(f"Videos: {', '.join(selected)}")

    encoder = load_video_vae_encoder(str(args.checkpoint), device=device, dtype=DTYPE)
    decoder = load_video_vae_decoder(str(args.checkpoint), device=device, dtype=DTYPE)

    for sample_id in selected:
        video_path = args.video_root / f"{sample_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print(f"Processing {sample_id}")
        video = load_video(video_path, device, args)

        latent = encode_to_latent(encoder, video)
        modified_latent = zero_spatial_boundary(latent, args.ring_width)
        print(f"  latent shape (B,C,F,H,W): {tuple(latent.shape)}")

        original_decoded = decode_latent_to_video(decoder, latent)
        modified_decoded = decode_latent_to_video(decoder, modified_latent)
        original = prepare_pixel_video(video)

        comparison_path = args.output_dir / f"{sample_id}_boundary_removed_comparison.mp4"
        save_comparison_video(original, original_decoded, modified_decoded, comparison_path, video_fps(video_path))
        print(f"  saved comparison video (original | original decode | modified decode) to {comparison_path}")

        del video, latent, modified_latent, original_decoded, modified_decoded, original
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
