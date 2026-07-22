"""
Full-res VAE ceiling vs. the production two-stage cheap path, at 1024x768 (HxW).

Four latents/decodes are produced from the same source video and compared:

  1. original   -- tiled_encode + tiled_decode the video directly at 1024x768.
                   This is the VAE's own reconstruction ceiling: the best any
                   path below can hope to match, since it never touches the
                   upsampler or the diffusion transformer.
  2. low        -- encode the video downsampled to 512x384 (exactly half),
                   decode directly at 512x384. This is the stage-1 cost the
                   two-stage pipelines pay to avoid running the transformer at
                   full resolution.
  3. upsampled  -- run the 512x384 latent through `LatentUpsampler` (2x
                   spatial, via `ltx_core.model.upsampler.upsample_video`) to
                   get a 1024x768-shaped latent, decode it directly with *no*
                   diffusion refinement. Isolates what the upsampler alone
                   contributes.
  4. refined    -- feed the upsampled latent as `initial_latent` into a short
                   distilled-diffusion stage-2 pass -- `SimpleDenoiser` +
                   `STAGE_2_DISTILLED_SIGMAS`, video-only (audio omitted) --
                   exactly the stage-2 step `DistilledPipeline`/
                   `TI2VidTwoStagesPipeline` run in production, then decode.
                   Shows what the second diffusion pass actually buys over (3).

All VAE/diffusion model building blocks are reused from
`ltx_pipelines.utils.blocks` so each stage builds its model, runs, and frees
it before the next stage starts (no manual GPU memory bookkeeping needed).

Compares: decoded frames side by side, latent PCA (n=3) false-color maps of a
middle frame, and full cumulative-PCA-variance curves for all four latents.

Run:
    conda run -n ltx python3 scripts/visualize_two_stage_upscale_refine.py

Useful overrides:
    conda run -n ltx python3 scripts/visualize_two_stage_upscale_refine.py \
        --video /path/to/video.mp4 --prompt "a description of the scene"
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
sys.path.append(str(REPO_ROOT / "packages" / "ltx-pipelines" / "src"))

from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.model.video_vae import TilingConfig  # noqa: E402
from ltx_pipelines.utils.blocks import (  # noqa: E402
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS  # noqa: E402
from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: E402
from ltx_pipelines.utils.types import ModalitySpec  # noqa: E402


DTYPE = torch.bfloat16
HIGH_H, HIGH_W = 1024, 768
LOW_H, LOW_W = 512, 384
STAGES = ["original", "low", "upsampled", "refined"]
STAGE_LABELS = {
    "original": "original (1024x768 direct)",
    "low": "low-res decode (512x384, resized for display)",
    "upsampled": "upsampled latent (no refine)",
    "refined": "refined (two-stage, stage 2)",
}

DEFAULT_VIDEO_PATH = (
    WORKSPACE_ROOT / "data" / "DNARendering" / "Videos" / "Part_5" / "0264_01" / "264_00001_Camera_5mp_cam025.mp4"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors",
        help="Distilled checkpoint containing VAE encoder/decoder + transformer weights.",
    )
    parser.add_argument(
        "--upsampler-checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        help="Spatial latent upscaler checkpoint.",
    )
    parser.add_argument(
        "--gemma-root",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints" / "google" / "gemma-3-12b-it-qat-q4_0-unquantized",
        help="Gemma text encoder root (needed for stage-2 prompt conditioning).",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Source video (needs native res >= 1024x768).")
    parser.add_argument(
        "--prompt",
        default="a high quality, sharp, detailed video with fine texture and natural lighting",
        help="Prompt used for the stage-2 refinement pass (no ground-truth caption is computed for the source video).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sigma-start-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Slice STAGE_2_DISTILLED_SIGMAS[start_index:] for the stage-4 refinement schedule. 0 = full "
        "schedule (default, matches production: noise_scale=0.909, 3 steps). 1 = skip the first (noisiest) "
        "step (noise_scale=0.725, 2 steps). 2 = only the last step (noise_scale=0.422, 1 step). Higher "
        "index = less noise injected = more fidelity to the input latent's motion/identity/environment, "
        "at the cost of less regeneration/cleanup of upsampler artifacts.",
    )
    parser.add_argument(
        "--max-analysis-frames",
        type=int,
        default=100_000,
        help="Cap on raw frames to read (before cropping to satisfy frames %% 8 == 1). Defaults to a large "
        "value so the full source video is used; lower it for a quicker/cheaper test, since the stage-4 "
        "diffusion transformer pass scales (quadratically, via attention) with token count = F'*H'*W'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "two_stage_upscale_refine_probe",
    )
    parser.add_argument("--num-heatmap-frames", type=int, default=4)
    parser.add_argument("--diff-gain", type=float, default=4.0)
    return parser


def resize_to(video: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bicubic-resize a [1, C, F, H, W] video to exactly (h, w)."""
    frames = rearrange(video, "b c f h w -> (b f) c h w").float()
    frames = F.interpolate(frames, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return rearrange(frames.to(video.dtype), "(b f) c h w -> b c f h w", b=video.shape[0])


def load_native_video(path: Path, device: torch.device, max_analysis_frames: int) -> tuple[torch.Tensor, float]:
    vr = decord.VideoReader(str(path))
    fps = float(vr.get_avg_fps())
    count = min(len(vr), max_analysis_frames)
    frames = vr.get_batch(range(count))
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(DTYPE).to(device)
    video = (video / 127.5) - 1.0
    valid_f = ((video.shape[2] - 1) // 8) * 8 + 1
    return video[:, :, :valid_f], fps


def concat_video_chunks(chunks) -> torch.Tensor:
    """Concatenate `decode_video`/`VideoDecoder` chunks ([f, h, w, c] in [0, 1]) along frames."""
    return torch.cat(list(chunks), dim=0)


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """PSNR between two [F, H, W, C] tensors in [0, 1]."""
    num_frames = min(a.shape[0], b.shape[0])
    a = a[:num_frames].float()
    b = b[:num_frames].float()
    mse = float(((a - b) ** 2).mean())
    return {"psnr_db": float(10 * np.log10(1.0 / max(mse, 1e-12))), "mse": mse}


def frame_indices(num_frames: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, num_frames - 1, min(count, num_frames)).astype(int).tolist()))


def rows_from_latent(latent: torch.Tensor) -> np.ndarray:
    return rearrange(latent, "b c f h w -> (b f h w) c").detach().cpu().float().numpy()


def pca_rgb_and_variance(latent: torch.Tensor) -> dict[str, Any]:
    """Top-3-component PCA false-color map of the middle frame + cumulative variance curve."""
    _, _, num_frames, h, w = latent.shape
    x = rows_from_latent(latent)
    n_components = min(128, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    projected = pca.fit_transform(x)
    comps = rearrange(projected, "(f h w) c -> f h w c", f=num_frames, h=h, w=w)
    mid = num_frames // 2
    rgb = comps[mid, ..., :3]
    lo = rgb.min(axis=(0, 1), keepdims=True)
    hi = rgb.max(axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return {"rgb": rgb, "cumulative_evr": cumulative.tolist(), "shape_BCFHW": [int(v) for v in latent.shape]}


def plot_decoded_frame_grid(decoded: dict[str, torch.Tensor], save_path: Path, num_heatmap_frames: int) -> None:
    num_frames = decoded["original"].shape[0]
    indices = frame_indices(num_frames, num_heatmap_frames)
    fig, axes = plt.subplots(
        len(STAGES), len(indices), figsize=(3.2 * len(indices), 3.0 * len(STAGES)), squeeze=False
    )
    for row, stage in enumerate(STAGES):
        frames = decoded[stage]
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(frames[min(idx, frames.shape[0] - 1)].float().cpu().numpy())
            if row == 0:
                ax.set_title(f"F{idx}")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(STAGE_LABELS[stage], fontsize=8)
    fig.suptitle("Decoded frames: original ceiling vs. low-res / upsampled / refined cheap path")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_latent_pca_grid(pca_results: dict[str, dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(STAGES), figsize=(4.0 * len(STAGES), 4.2), squeeze=False)
    for col, stage in enumerate(STAGES):
        ax = axes[0, col]
        ax.imshow(pca_results[stage]["rgb"])
        shape = pca_results[stage]["shape_BCFHW"]
        ax.set_title(f"{STAGE_LABELS[stage]}\nshape={shape}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Latent top-3 PCA false color (middle frame)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_pca_variance_curves(pca_results: dict[str, dict[str, Any]], save_path: Path) -> None:
    plt.figure(figsize=(6.5, 4.5))
    for stage in STAGES:
        cumulative = pca_results[stage]["cumulative_evr"][:50]
        plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", ms=3, label=STAGE_LABELS[stage])
    plt.ylim(0.0, 1.02)
    plt.xlabel("PCA components")
    plt.ylabel("cumulative explained variance")
    plt.title("Latent PCA cumulative variance: original vs. low / upsampled / refined")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_comparison_video(decoded: dict[str, torch.Tensor], save_path: Path, fps: float) -> None:
    from ltx_trainer.video_utils import save_video

    num_frames = min(t.shape[0] for t in decoded.values())
    panels = [rearrange(decoded[stage][:num_frames].clamp(0, 1), "f h w c -> f c h w") for stage in STAGES]
    combined = torch.cat(panels, dim=-1)
    save_video(combined, save_path, fps=fps, video_format="FCHW")


def save_single_decode_video(decode: torch.Tensor, save_path: Path, fps: float) -> None:
    """Save one [F, H, W, C] decoded video (in [0, 1]) to its own file."""
    from ltx_trainer.video_utils import save_video

    save_video(rearrange(decode.clamp(0, 1), "f h w c -> f c h w"), save_path, fps=fps, video_format="FCHW")


@torch.no_grad()
def main() -> int:
    args = build_arg_parser().parse_args()
    if args.sigma_start_index != 0:
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_sigma{args.sigma_start_index}"
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    for path in (args.vae_checkpoint, args.upsampler_checkpoint, args.video):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.gemma_root.exists():
        raise FileNotFoundError(args.gemma_root)

    print(f"VAE/transformer checkpoint: {args.vae_checkpoint}")
    print(f"Upsampler checkpoint: {args.upsampler_checkpoint}")
    print(f"Gemma root: {args.gemma_root}")
    print(f"Video: {args.video}")
    print(f"Prompt (stage-2 conditioning, not a real caption): {args.prompt!r}")

    native_video, fps = load_native_video(args.video, device, args.max_analysis_frames)
    valid_f = native_video.shape[2]
    _, _, _, native_h, native_w = native_video.shape
    if native_h < HIGH_H or native_w < HIGH_W:
        raise ValueError(
            f"Video native resolution {native_h}x{native_w} is smaller than the {HIGH_H}x{HIGH_W} target; "
            "pick a higher-resolution source with --video."
        )
    print(f"Native resolution: {native_h}x{native_w}, using {valid_f} frames @ {fps:.2f} fps")

    full_video = resize_to(native_video, HIGH_H, HIGH_W)
    low_video = resize_to(native_video, LOW_H, LOW_W)
    del native_video
    torch.cuda.empty_cache()
    tiling = TilingConfig.default()

    # --- Steps 1 + 2: encode both resolutions (one encoder build, freed on exit) ---
    print("\n[1/4] Encoding original (1024x768, tiled) and low-res (512x384)...")
    image_conditioner = ImageConditioner(str(args.vae_checkpoint), DTYPE, device)
    original_latent, low_latent = image_conditioner(
        lambda enc: (enc.tiled_encode(full_video, tiling), enc(low_video))
    )
    del full_video, low_video
    torch.cuda.empty_cache()
    print(f"  original_latent shape={tuple(original_latent.shape)}, low_latent shape={tuple(low_latent.shape)}")

    video_decoder = VideoDecoder(str(args.vae_checkpoint), DTYPE, device)
    # Long videos at 1024x768 make each decode several hundred MB to a few GB; move finished
    # decodes to CPU immediately so they don't compete with the ~44GB transformer load in step 4.
    original_decode = concat_video_chunks(video_decoder(original_latent, tiling)).cpu()
    low_decode = concat_video_chunks(video_decoder(low_latent, None)).cpu()
    torch.cuda.empty_cache()
    print(f"  original_decode shape={tuple(original_decode.shape)}, low_decode shape={tuple(low_decode.shape)}")

    # --- Step 3: spatial latent upsample (2x), decode directly (no diffusion) ---
    print("\n[2/4] Upsampling low-res latent to 1024x768 (LatentUpsampler, no refinement)...")
    upsampler = VideoUpsampler(str(args.vae_checkpoint), str(args.upsampler_checkpoint), DTYPE, device)
    upsampled_latent = upsampler(low_latent)
    assert upsampled_latent.shape == original_latent.shape, (
        f"upsampled latent shape {tuple(upsampled_latent.shape)} != original latent shape "
        f"{tuple(original_latent.shape)}"
    )
    upsampled_decode = concat_video_chunks(video_decoder(upsampled_latent, tiling)).cpu()
    torch.cuda.empty_cache()
    print(f"  upsampled_latent shape={tuple(upsampled_latent.shape)}")

    # --- Step 4: stage-2 distilled refinement, exactly as DistilledPipeline does ---
    print("\n[3/4] Refining upsampled latent with a distilled stage-2 diffusion pass...")
    prompt_encoder = PromptEncoder(str(args.vae_checkpoint), str(args.gemma_root), DTYPE, device)
    (ctx,) = prompt_encoder([args.prompt])
    video_context = ctx.video_encoding
    torch.cuda.empty_cache()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS[args.sigma_start_index :].to(dtype=torch.float32, device=device)
    print(
        f"  stage_2_sigmas (start_index={args.sigma_start_index}): {stage_2_sigmas.tolist()} "
        f"({len(stage_2_sigmas) - 1} denoising step(s), noise_scale={stage_2_sigmas[0].item():.4f})"
    )

    diffusion_stage = DiffusionStage.from_checkpoint(str(args.vae_checkpoint), DTYPE, device)
    video_state, _ = diffusion_stage(
        denoiser=SimpleDenoiser(video_context, None),
        sigmas=stage_2_sigmas,
        noiser=noiser,
        width=HIGH_W,
        height=HIGH_H,
        frames=valid_f,
        fps=fps,
        video=ModalitySpec(
            context=video_context,
            conditionings=[],
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=upsampled_latent,
        ),
        audio=None,
    )
    refined_latent = video_state.latent
    refined_decode = concat_video_chunks(video_decoder(refined_latent, tiling)).cpu()
    print(f"  refined_latent shape={tuple(refined_latent.shape)}")

    # --- Compare ---
    print("\n[4/4] Comparing decodes, latent PCA maps, and PCA variance curves...")
    low_decode_resized_for_display = rearrange(
        resize_to(rearrange(low_decode, "f h w c -> 1 c f h w"), HIGH_H, HIGH_W), "1 c f h w -> f h w c"
    ).clamp(0, 1)
    decoded = {
        "original": original_decode,
        "low": low_decode_resized_for_display,
        "upsampled": upsampled_decode,
        "refined": refined_decode,
    }
    latents = {
        "original": original_latent.cpu(),
        "low": low_latent.cpu(),
        "upsampled": upsampled_latent.cpu(),
        "refined": refined_latent.cpu(),
    }

    psnr_vs_original = {
        stage: compute_psnr(decoded[stage], original_decode) for stage in STAGES if stage != "original"
    }
    for stage, psnr in psnr_vs_original.items():
        print(f"  {STAGE_LABELS[stage]:<55} PSNR vs original decode: {psnr['psnr_db']:.2f} dB")

    plot_decoded_frame_grid(decoded, args.output_dir / "decoded_frames.png", args.num_heatmap_frames)
    save_comparison_video(decoded, args.output_dir / "comparison.mp4", fps)
    save_single_decode_video(refined_decode, args.output_dir / "refined_decode.mp4", fps)

    pca_results = {stage: pca_rgb_and_variance(latents[stage]) for stage in STAGES}
    plot_latent_pca_grid(pca_results, args.output_dir / "latent_pca.png")
    plot_pca_variance_curves(pca_results, args.output_dir / "latent_pca_variance.png")

    summary = {
        "vae_checkpoint": str(args.vae_checkpoint),
        "upsampler_checkpoint": str(args.upsampler_checkpoint),
        "video": str(args.video),
        "prompt": args.prompt,
        "seed": args.seed,
        "sigma_start_index": args.sigma_start_index,
        "stage_2_sigmas": stage_2_sigmas.tolist(),
        "native_hw": [native_h, native_w],
        "valid_frames": valid_f,
        "fps": fps,
        "latent_shapes": {stage: [int(v) for v in latents[stage].shape] for stage in STAGES},
        "psnr_vs_original_decode": psnr_vs_original,
        "pca_cumulative_variance_top50": {stage: pca_results[stage]["cumulative_evr"][:50] for stage in STAGES},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nSaved decoded-frame grid to {args.output_dir / 'decoded_frames.png'}")
    print(f"Saved side-by-side comparison video to {args.output_dir / 'comparison.mp4'}")
    print(f"Saved latent PCA grid to {args.output_dir / 'latent_pca.png'}")
    print(f"Saved PCA variance curves to {args.output_dir / 'latent_pca_variance.png'}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
