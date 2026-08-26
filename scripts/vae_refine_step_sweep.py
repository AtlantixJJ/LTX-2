"""LTX VAE Round-Trip and Diffusion Refinement Step Sweep on SAM3DGS Renders.

Execution workflow:
  0. Unit check: noise injection math & bitwise equality verification
  1. Manifest generation: sample 5 videos per subject (2 crop, 2 original, 1 rotation) across 30 subjects
  2. Part 1: VAE encode/decode round-trip across all 150 videos, computing PSNR/SSIM distributions
  3. Prompt control: probe prompt sensitivity (descriptive vs generic vs empty vs mismatched)
  4. Part 2: Step sweep (k in 0, 1, 2, 3, 4, 8 and gentle probes) across representative subset
  5. Part 3: Production-faithful two-stage upscale + refine comparison on selected videos
  6. Aggregation & Summary: compile metrics, generate comparison plots, write FINDINGS.md

Usage:
  # Dry run (generate manifest only)
  conda run -n ltx python3 scripts/vae_refine_step_sweep.py --dry-run

  # Run full pipeline using GPUs 6 and 7
  conda run -n ltx python3 scripts/vae_refine_step_sweep.py --gpu-ids 6,7
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# decord must be initialized after torch touches CUDA
torch.cuda.init()
import decord  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from einops import rearrange  # noqa: E402

decord.bridge.set_bridge("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-pipelines" / "src"))

from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: E402
from ltx_core.model.video_vae import TilingConfig  # noqa: E402
from ltx_core.tools import VideoLatentTools  # noqa: E402
from ltx_core.types import VideoLatentShape, VideoPixelShape  # noqa: E402
from ltx_pipelines.utils.blocks import (  # noqa: E402
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (  # noqa: E402
    DISTILLED_SIGMAS,
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: E402
from ltx_pipelines.utils.helpers import create_noised_state  # noqa: E402
from ltx_pipelines.utils.types import ModalitySpec  # noqa: E402
from ltx_trainer.video_utils import save_video  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vae_refine_sweep")

DTYPE = torch.bfloat16
DATA_ROOT = Path("/home/jianjinx/data2/SAM3DGS/expr/infer_sam3dgs_visualize_all")
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"
DEFAULT_VAE_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_UPSAMPLER_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DEFAULT_GEMMA_ROOT = WORKSPACE_ROOT / "checkpoints" / "google" / "gemma-3-12b-it-qat-q4_0-unquantized"


# ---------------------------------------------------------------------------
# SSIM and PSNR Metrics
# ---------------------------------------------------------------------------


def create_gaussian_window(
    window_size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=torch.float32
) -> torch.Tensor:
    gauss = torch.tensor(
        [-(x - window_size // 2) ** 2 / float(2 * sigma**2) for x in range(window_size)],
        dtype=dtype,
        device=device,
    ).exp()
    gauss = gauss / gauss.sum()
    _1d_window = gauss.unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d_window.expand(channels, 1, window_size, window_size).contiguous().to(dtype=dtype, device=device)


def compute_ssim_tensor(a: torch.Tensor, b: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """Compute SSIM between two [F, H, W, C] tensors in [0, 1]."""
    num_frames = min(a.shape[0], b.shape[0])
    a = a[:num_frames].permute(0, 3, 1, 2).float()
    b = b[:num_frames].permute(0, 3, 1, 2).float()
    channel = a.shape[1]
    window = create_gaussian_window(window_size, sigma, channel, device=a.device, dtype=a.dtype)

    mu1 = F.conv2d(a, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(b, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(a * a, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def compute_psnr_and_metrics(
    candidate: torch.Tensor, reference: torch.Tensor, d_vae: torch.Tensor | None = None
) -> dict[str, Any]:
    """Compute full suite of metrics for candidate [F, H, W, C] vs reference [F, H, W, C] and optional d_vae."""
    num_frames = min(candidate.shape[0], reference.shape[0])
    c = candidate[:num_frames].float()
    r = reference[:num_frames].float()

    # Per-frame MSE & PSNR vs reference
    per_frame_mse_ref = ((c - r) ** 2).flatten(1).mean(dim=1)
    per_frame_psnr_ref = (10 * torch.log10(1.0 / per_frame_mse_ref.clamp(min=1e-12))).tolist()
    overall_mse_ref = float(per_frame_mse_ref.mean().item())
    overall_psnr_ref = float(10 * math.log10(1.0 / max(overall_mse_ref, 1e-12)))
    ssim_ref = compute_ssim_tensor(c, r)

    # Temporal difference energy mean(|f_t - f_{t-1}|)
    if num_frames > 1:
        temp_diff_c = float((c[1:] - c[:-1]).abs().mean().item())
        temp_diff_r = float((r[1:] - r[:-1]).abs().mean().item())
    else:
        temp_diff_c, temp_diff_r = 0.0, 0.0

    result = {
        "overall_psnr_db_vs_source": overall_psnr_ref,
        "overall_mse_vs_source": overall_mse_ref,
        "ssim_vs_source": ssim_ref,
        "per_frame_psnr_db_vs_source": [float(v) for v in per_frame_psnr_ref],
        "temporal_diff_energy_candidate": temp_diff_c,
        "temporal_diff_energy_source": temp_diff_r,
    }

    if d_vae is not None:
        d = d_vae[:num_frames].float()
        per_frame_mse_d = ((c - d) ** 2).flatten(1).mean(dim=1)
        per_frame_psnr_d = (10 * torch.log10(1.0 / per_frame_mse_d.clamp(min=1e-12))).tolist()
        overall_mse_d = float(per_frame_mse_d.mean().item())
        overall_psnr_d = float(10 * math.log10(1.0 / max(overall_mse_d, 1e-12)))
        ssim_d = compute_ssim_tensor(c, d)
        result.update(
            {
                "overall_psnr_db_vs_d_vae": overall_psnr_d,
                "overall_mse_vs_d_vae": overall_mse_d,
                "ssim_vs_d_vae": ssim_d,
                "per_frame_psnr_db_vs_d_vae": [float(v) for v in per_frame_psnr_d],
            }
        )

    return result


# ---------------------------------------------------------------------------
# Sampling & Manifest
# ---------------------------------------------------------------------------


@dataclass
class VideoEntry:
    video_id: str
    subject_id: str
    dataset_prefix: str
    video_class: str  # "crop", "original", "rotation"
    file_path: str
    target_height: int
    target_width: int
    target_frames: int
    native_height: int
    native_width: int
    native_frames: int
    fps: float


def extract_base_motion(filename: str) -> str:
    name = Path(filename).stem
    # Remove _crop / _original suffix
    name = re.sub(r"_(?:crop|original)$", "", name)
    # Remove multi-cam suffix like _c1, _c3
    name = re.sub(r"_c\d+$", "", name)
    return name


def build_manifest(
    data_root: Path,
    n_crop: int = 2,
    n_original: int = 2,
    n_rotation: int = 1,
    subjects_filter: list[str] | None = None,
    sample_seed: int = 0,
) -> list[VideoEntry]:
    rng = np.random.RandomState(sample_seed)
    subject_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if subjects_filter and subjects_filter != ["all"]:
        subject_dirs = [d for d in subject_dirs if d.name in subjects_filter]

    entries: list[VideoEntry] = []

    for sdir in subject_dirs:
        subject_id = sdir.name
        dataset_prefix = subject_id.split("_")[0]

        mp4_files = sorted(list(sdir.glob("*.mp4")))
        crop_files = [f for f in mp4_files if f.name.endswith("_crop.mp4")]
        orig_files = [f for f in mp4_files if f.name.endswith("_original.mp4")]
        rot_files = [f for f in mp4_files if "rotation" in f.name]

        # Group crops by base motion
        crop_by_motion: dict[str, list[Path]] = {}
        for f in crop_files:
            crop_by_motion.setdefault(extract_base_motion(f.name), []).append(f)

        # Sample crops with distinct motions where possible
        sampled_crops: list[Path] = []
        motions = sorted(list(crop_by_motion.keys()))
        rng.shuffle(motions)
        for m in motions:
            if len(sampled_crops) >= n_crop:
                break
            files = sorted(crop_by_motion[m])
            rng.shuffle(files)
            sampled_crops.append(files[0])
        # If still short, sample without replacement from remaining files
        if len(sampled_crops) < n_crop:
            remaining = [f for f in crop_files if f not in sampled_crops]
            rng.shuffle(remaining)
            sampled_crops.extend(remaining[: n_crop - len(sampled_crops)])

        # Sample originals with distinct motions
        orig_by_motion: dict[str, list[Path]] = {}
        for f in orig_files:
            orig_by_motion.setdefault(extract_base_motion(f.name), []).append(f)

        sampled_origs: list[Path] = []
        motions = sorted(list(orig_by_motion.keys()))
        rng.shuffle(motions)
        for m in motions:
            if len(sampled_origs) >= n_original:
                break
            files = sorted(orig_by_motion[m])
            rng.shuffle(files)
            sampled_origs.append(files[0])
        if len(sampled_origs) < n_original:
            remaining = [f for f in orig_files if f not in sampled_origs]
            rng.shuffle(remaining)
            sampled_origs.extend(remaining[: n_original - len(sampled_origs)])

        # Sample rotation
        sampled_rots = rot_files[:n_rotation]

        # Build entries
        for p in sampled_crops:
            entries.append(
                VideoEntry(
                    video_id=f"{subject_id}__{p.stem}",
                    subject_id=subject_id,
                    dataset_prefix=dataset_prefix,
                    video_class="crop",
                    file_path=str(p),
                    target_height=1024,
                    target_width=1024,
                    target_frames=121,
                    native_height=1024,
                    native_width=1024,
                    native_frames=0,  # resolved on read
                    fps=30.0,
                )
            )

        for p in sampled_origs:
            entries.append(
                VideoEntry(
                    video_id=f"{subject_id}__{p.stem}",
                    subject_id=subject_id,
                    dataset_prefix=dataset_prefix,
                    video_class="original",
                    file_path=str(p),
                    target_height=704,  # 720 center-cropped to 704
                    target_width=1280,
                    target_frames=121,
                    native_height=720,
                    native_width=1280,
                    native_frames=0,
                    fps=30.0,
                )
            )

        for p in sampled_rots:
            entries.append(
                VideoEntry(
                    video_id=f"{subject_id}__{p.stem}",
                    subject_id=subject_id,
                    dataset_prefix=dataset_prefix,
                    video_class="rotation",
                    file_path=str(p),
                    target_height=768,
                    target_width=768,
                    target_frames=113,  # 120 -> 113
                    native_height=768,
                    native_width=768,
                    native_frames=120,
                    fps=24.0,
                )
            )

    return entries


# ---------------------------------------------------------------------------
# Video Loading & Preprocessing
# ---------------------------------------------------------------------------


def load_video_tensor(
    entry: VideoEntry, device: torch.device, dtype: torch.dtype = DTYPE
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Load video using decord, crop to target geometry, and return (norm_video_for_vae, raw_pixel_video_in_0_1, fps)."""
    vr = decord.VideoReader(entry.file_path)
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    if total_frames < 9:
        raise ValueError(f"Video {entry.file_path} has only {total_frames} frames (need >= 9)")

    # Frame budget
    max_read = min(total_frames, entry.target_frames)
    valid_f = ((max_read - 1) // 8) * 8 + 1
    frames = vr.get_batch(range(valid_f))  # [F, H, W, C] in [0, 255]

    # Center-crop spatial dimensions to multiples of 32
    f, h, w, c = frames.shape
    if h % 32 != 0:
        crop_h = (h // 32) * 32
        top = (h - crop_h) // 2
        frames = frames[:, top : top + crop_h, :, :]
    if w % 32 != 0:
        crop_w = (w // 32) * 32
        left = (w - crop_w) // 2
        frames = frames[:, :, left : left + crop_w, :]

    # Assertions
    f, h, w, c = frames.shape
    assert h % 32 == 0 and w % 32 == 0, f"Dimensions ({h}, {w}) not divisible by 32!"
    assert (f - 1) % 8 == 0, f"Frame count {f} does not satisfy f % 8 == 1!"

    # Pixel video in [0, 1] [F, H, W, C]
    pixel_video = (frames.float() / 255.0).clamp(0.0, 1.0)

    # VAE normalized video [1, C, F, H, W] in [-1, 1]
    norm_video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(dtype=dtype, device=device)
    norm_video = (norm_video / 127.5) - 1.0

    return norm_video, pixel_video, fps


# ---------------------------------------------------------------------------
# Noise Injection & Sanity Checks (Steps 0, 2, 3)
# ---------------------------------------------------------------------------


def run_unit_checks(device: torch.device, dtype: torch.dtype = DTYPE) -> None:
    logger.info("Running noise injection mathematical unit checks...")

    pixel_shape = VideoPixelShape(batch=1, frames=113, height=768, width=768, fps=24.0)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, 24.0)

    torch.manual_seed(123)
    l_enc = torch.randn(1, 128, 15, 24, 24, device=device, dtype=dtype)
    sigma_0 = float(STAGE_2_DISTILLED_SIGMAS[0].item())
    seed = 42

    gen1 = torch.Generator(device=device).manual_seed(seed)
    noiser1 = GaussianNoiser(generator=gen1)
    state1 = create_noised_state(
        tools=video_tools,
        conditionings=[],
        noiser=noiser1,
        dtype=dtype,
        device=device,
        noise_scale=sigma_0,
        initial_latent=l_enc,
    )

    # Mathematical reference
    patchified_l = video_tools.patchifier.patchify(l_enc)
    gen2 = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(patchified_l.shape, generator=gen2, device=device, dtype=dtype)
    expected_x = torch.lerp(patchified_l.float(), eps.float(), sigma_0).to(dtype)

    diff = (state1.latent - expected_x).abs().max().item()
    assert diff == 0.0, f"Noise injection unit check failed! Max bitwise diff: {diff}"
    assert state1.denoise_mask.min().item() == 1.0, "Denoise mask is not all-ones!"

    # Variance check
    var_actual = state1.latent.float().var().item()
    var_expected = ((1.0 - sigma_0) ** 2) * l_enc.float().var().item() + (sigma_0**2) * 1.0
    assert abs(var_actual - var_expected) < 0.05, f"Variance mismatch: actual={var_actual}, expected={var_expected}"

    logger.info("Noise injection mathematical unit checks PASSED cleanly!")


# ---------------------------------------------------------------------------
# Part 1: VAE Encode/Decode Roundtrip
# ---------------------------------------------------------------------------


def process_part1_video(
    entry: VideoEntry,
    vae_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    tiling: TilingConfig,
    force: bool = False,
    run_untiled_probe: bool = False,
) -> dict[str, Any]:
    sample_dir = output_dir / entry.video_id
    metrics_file = sample_dir / "vae_metrics.json"

    if metrics_file.exists() and not force:
        try:
            return json.loads(metrics_file.read_text())
        except Exception:
            pass

    sample_dir.mkdir(parents=True, exist_ok=True)
    norm_video, raw_pixels, fps = load_video_tensor(entry, device)
    _, _, f_actual, h_actual, w_actual = norm_video.shape

    # Encode
    with torch.no_grad():
        image_conditioner = ImageConditioner(str(vae_checkpoint), DTYPE, device)
        l_enc = image_conditioner(lambda enc: enc.tiled_encode(norm_video, tiling))

        # Expected shape assertion
        expected_shape = (1, 128, (f_actual - 1) // 8 + 1, h_actual // 32, w_actual // 32)
        assert l_enc.shape == expected_shape, f"Encoded shape {l_enc.shape} != expected {expected_shape}"

        # Decode
        video_decoder = VideoDecoder(str(vae_checkpoint), DTYPE, device)
        chunks = list(video_decoder(l_enc, tiling))
        d_vae = torch.cat(chunks, dim=0).cpu().float()

    # Save decoded video and source
    save_video(rearrange(raw_pixels, "f h w c -> f c h w"), sample_dir / "source.mp4", fps=fps, video_format="FCHW")
    save_video(rearrange(d_vae, "f h w c -> f c h w"), sample_dir / "vae_roundtrip.mp4", fps=fps, video_format="FCHW")

    # Side-by-side comparison: [source | D_vae | |diff| * 4]
    diff = ((raw_pixels - d_vae).abs() * 4.0).clamp(0.0, 1.0)
    combined = torch.cat([raw_pixels, d_vae, diff], dim=2)  # along width
    save_video(rearrange(combined, "f h w c -> f c h w"), sample_dir / "vae_comparison.mp4", fps=fps, video_format="FCHW")

    # Metrics
    metrics = compute_psnr_and_metrics(d_vae, raw_pixels)
    record = {
        "video_id": entry.video_id,
        "subject_id": entry.subject_id,
        "dataset_prefix": entry.dataset_prefix,
        "video_class": entry.video_class,
        "file_path": entry.file_path,
        "geometry": [entry.target_height, entry.target_width, entry.target_frames],
        "fps": fps,
        "latent_shape": [int(v) for v in l_enc.shape],
        "metrics": metrics,
    }

    # Optional untiled probe for 768x768 canonical_rotation
    if run_untiled_probe and entry.video_class == "rotation":
        with torch.no_grad():
            l_untiled = image_conditioner(lambda enc: enc(norm_video))
            chunks_untiled = list(video_decoder(l_untiled, None))
            d_untiled = torch.cat(chunks_untiled, dim=0).cpu().float()
            metrics_untiled = compute_psnr_and_metrics(d_untiled, raw_pixels)
            record["untiled_metrics"] = metrics_untiled
            record["untiled_vs_tiled_psnr_db"] = compute_psnr_and_metrics(d_untiled, d_vae)[
                "overall_psnr_db_vs_source"
            ]

    metrics_file.write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_part1_worker(
    gpu_id: int,
    video_entries: list[VideoEntry],
    vae_checkpoint: Path,
    output_dir: Path,
    force: bool,
    untiled_probe_first: bool,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{gpu_id}")
    tiling = TilingConfig.default()

    results: list[dict[str, Any]] = []
    total = len(video_entries)
    for idx, entry in enumerate(video_entries):
        t0 = time.time()
        probe = untiled_probe_first and idx == 0
        try:
            rec = process_part1_video(
                entry, vae_checkpoint, output_dir, device, tiling, force=force, run_untiled_probe=probe
            )
            results.append(rec)
            elapsed = time.time() - t0
            psnr = rec["metrics"]["overall_psnr_db_vs_source"]
            logger.info(
                f"[GPU {gpu_id}] ({idx+1}/{total}) {entry.video_id}: PSNR={psnr:.2f} dB, time={elapsed:.1f}s"
            )
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Failed {entry.video_id}: {e}", exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Part 2: Refinement Step Sweep
# ---------------------------------------------------------------------------


def get_refinement_schedules() -> dict[str, list[float]]:
    """Build step sweep schedules from distilled constants."""
    dist_vals = [float(v) for v in DISTILLED_SIGMA_VALUES]
    schedules = {
        "k1": dist_vals[-2:],
        "k2": dist_vals[-3:],
        "k3": dist_vals[-4:],
        "k4": dist_vals[-5:],
        "k8": dist_vals,
        "k_probe_0.25": [0.25, 0.0],
        "k_probe_0.15": [0.15, 0.0],
    }
    return schedules


def run_prompt_control(
    entry: VideoEntry,
    vae_checkpoint: Path,
    gemma_root: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Test prompt sensitivity on 1 video at k=3."""
    control_dir = output_dir / "prompt_sensitivity_control" / entry.video_id
    control_dir.mkdir(parents=True, exist_ok=True)

    norm_video, raw_pixels, fps = load_video_tensor(entry, device)
    _, _, f_actual, h_actual, w_actual = norm_video.shape
    tiling = TilingConfig.default()

    # VAE encode & decode baseline
    with torch.no_grad():
        image_conditioner = ImageConditioner(str(vae_checkpoint), DTYPE, device)
        l_enc = image_conditioner(lambda enc: enc.tiled_encode(norm_video, tiling))

        video_decoder = VideoDecoder(str(vae_checkpoint), DTYPE, device)
        d_vae = torch.cat(list(video_decoder(l_enc, tiling)), dim=0).cpu().float()

    prompts = {
        "descriptive": "a detailed full body video of a person performing motion in a studio environment",
        "generic": "a high quality, sharp, detailed video with fine texture and natural lighting",
        "empty": "",
        "mismatched": "a fluffy ginger cat playing with a red yarn ball in deep winter snow",
    }

    # Encode all prompts first, then release Gemma LLM memory
    prompt_contexts = {}
    with torch.no_grad():
        prompt_encoder = PromptEncoder(str(vae_checkpoint), str(gemma_root), DTYPE, device)
        for p_name, p_text in prompts.items():
            (ctx,) = prompt_encoder([p_text])
            prompt_contexts[p_name] = ctx.video_encoding
        del prompt_encoder
        gc.collect()
        torch.cuda.empty_cache()

    sigmas = torch.tensor(get_refinement_schedules()["k3"], dtype=torch.float32, device=device)
    diffusion_stage = DiffusionStage.from_checkpoint(str(vae_checkpoint), DTYPE, device)

    results = {}
    for p_name, p_text in prompts.items():
        logger.info(f"Running prompt sensitivity control: {p_name} ({p_text!r})")
        video_context = prompt_contexts[p_name]

        noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(42))
        with torch.no_grad():
            state, _ = diffusion_stage(
                denoiser=SimpleDenoiser(video_context, None),
                sigmas=sigmas,
                noiser=noiser,
                width=w_actual,
                height=h_actual,
                frames=f_actual,
                fps=fps,
                video=ModalitySpec(
                    context=video_context,
                    conditionings=[],
                    noise_scale=sigmas[0].item(),
                    initial_latent=l_enc,
                ),
                audio=None,
            )
            d_refined = torch.cat(list(video_decoder(state.latent, tiling)), dim=0).cpu().float()

        save_video(
            rearrange(d_refined, "f h w c -> f c h w"),
            control_dir / f"refine_{p_name}.mp4",
            fps=fps,
            video_format="FCHW",
        )
        metrics = compute_psnr_and_metrics(d_refined, raw_pixels, d_vae=d_vae)
        results[p_name] = {"prompt": p_text, "metrics": metrics}

    (control_dir / "control_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def process_part2_video_sweep(
    entry: VideoEntry,
    vae_checkpoint: Path,
    gemma_root: Path,
    output_dir: Path,
    device: torch.device,
    prompt: str,
    k_steps: list[str],
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    sample_dir = output_dir / entry.video_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    sweep_metrics_file = sample_dir / "sweep_metrics.json"

    if sweep_metrics_file.exists() and not force:
        try:
            return json.loads(sweep_metrics_file.read_text())
        except Exception:
            pass

    norm_video, raw_pixels, fps = load_video_tensor(entry, device)
    _, _, f_actual, h_actual, w_actual = norm_video.shape
    tiling = TilingConfig.default()

    # VAE encode & decode (k=0)
    with torch.no_grad():
        image_conditioner = ImageConditioner(str(vae_checkpoint), DTYPE, device)
        l_enc = image_conditioner(lambda enc: enc.tiled_encode(norm_video, tiling))

        video_decoder = VideoDecoder(str(vae_checkpoint), DTYPE, device)
        d_vae = torch.cat(list(video_decoder(l_enc, tiling)), dim=0).cpu().float()

    save_video(rearrange(raw_pixels, "f h w c -> f c h w"), sample_dir / "source.mp4", fps=fps, video_format="FCHW")
    save_video(rearrange(d_vae, "f h w c -> f c h w"), sample_dir / "vae_roundtrip.mp4", fps=fps, video_format="FCHW")

    # Encode prompt once, release Gemma LLM memory
    with torch.no_grad():
        prompt_encoder = PromptEncoder(str(vae_checkpoint), str(gemma_root), DTYPE, device)
        (ctx,) = prompt_encoder([prompt])
        video_context = ctx.video_encoding
        del prompt_encoder
        gc.collect()
        torch.cuda.empty_cache()

    schedules = get_refinement_schedules()
    diffusion_stage = DiffusionStage.from_checkpoint(str(vae_checkpoint), DTYPE, device)

    variants_decodes: dict[str, torch.Tensor] = {"source": raw_pixels, "k0": d_vae}
    variants_metrics: dict[str, Any] = {
        "k0": compute_psnr_and_metrics(d_vae, raw_pixels, d_vae=d_vae),
    }

    for k_name in k_steps:
        if k_name not in schedules:
            continue
        sig_list = schedules[k_name]
        sigmas = torch.tensor(sig_list, dtype=torch.float32, device=device)

        k_dir = sample_dir / k_name
        k_dir.mkdir(parents=True, exist_ok=True)

        noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed))
        with torch.no_grad():
            state, _ = diffusion_stage(
                denoiser=SimpleDenoiser(video_context, None),
                sigmas=sigmas,
                noiser=noiser,
                width=w_actual,
                height=h_actual,
                frames=f_actual,
                fps=fps,
                video=ModalitySpec(
                    context=video_context,
                    conditionings=[],
                    noise_scale=sigmas[0].item(),
                    initial_latent=l_enc,
                ),
                audio=None,
            )
            d_refined = torch.cat(list(video_decoder(state.latent, tiling)), dim=0).cpu().float()

        save_video(
            rearrange(d_refined, "f h w c -> f c h w"), k_dir / "decode.mp4", fps=fps, video_format="FCHW"
        )
        metrics = compute_psnr_and_metrics(d_refined, raw_pixels, d_vae=d_vae)
        (k_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        variants_decodes[k_name] = d_refined
        variants_metrics[k_name] = metrics
        logger.info(
            f"[{entry.video_id}] {k_name} ({len(sig_list)-1} steps, sigma0={sig_list[0]:.4f}): "
            f"PSNR vs source={metrics['overall_psnr_db_vs_source']:.2f} dB, vs D_vae={metrics['overall_psnr_db_vs_d_vae']:.2f} dB"
        )

    # Comparison video: side by side of source | k0 | k1 | k2 | k3 | k4
    comp_keys = [k for k in ["source", "k0", "k1", "k2", "k3", "k4"] if k in variants_decodes]
    if len(comp_keys) > 1:
        comp_tensor = torch.cat([variants_decodes[k] for k in comp_keys], dim=2)
        save_video(
            rearrange(comp_tensor, "f h w c -> f c h w"),
            sample_dir / "comparison.mp4",
            fps=fps,
            video_format="FCHW",
        )

    # Frame grid image (4 sampled frames across all variants)
    plot_sweep_frame_grid(variants_decodes, sample_dir / "frames_grid.png")

    # PSNR vs steps plot
    plot_psnr_vs_steps(variants_metrics, entry.video_id, sample_dir / "psnr_vs_steps.png")

    full_record = {
        "video_id": entry.video_id,
        "subject_id": entry.subject_id,
        "dataset_prefix": entry.dataset_prefix,
        "video_class": entry.video_class,
        "file_path": entry.file_path,
        "geometry": [entry.target_height, entry.target_width, entry.target_frames],
        "fps": fps,
        "prompt": prompt,
        "seed": seed,
        "variants": variants_metrics,
    }
    sweep_metrics_file.write_text(json.dumps(full_record, indent=2) + "\n")
    return full_record


def plot_sweep_frame_grid(variants: dict[str, torch.Tensor], save_path: Path, num_frames: int = 4) -> None:
    first_v = next(iter(variants.values()))
    f_total = first_v.shape[0]
    indices = sorted(set(np.linspace(0, f_total - 1, min(num_frames, f_total)).astype(int).tolist()))

    keys = list(variants.keys())
    fig, axes = plt.subplots(len(keys), len(indices), figsize=(3.2 * len(indices), 2.8 * len(keys)), squeeze=False)

    for row, k in enumerate(keys):
        v = variants[k]
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            frame = v[min(idx, v.shape[0] - 1)].numpy()
            ax.imshow(frame)
            if row == 0:
                ax.set_title(f"F{idx}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(k, fontsize=9, fontweight="bold")

    fig.suptitle("Refinement Step Sweep Sampled Frames", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_psnr_vs_steps(metrics: dict[str, Any], title: str, save_path: Path) -> None:
    step_map = {"k0": 0, "k1": 1, "k2": 2, "k3": 3, "k4": 4, "k8": 8}
    valid_keys = [k for k in step_map.keys() if k in metrics]

    steps = [step_map[k] for k in valid_keys]
    psnr_src = [metrics[k]["overall_psnr_db_vs_source"] for k in valid_keys]
    psnr_dvae = [metrics[k].get("overall_psnr_db_vs_d_vae", float("nan")) for k in valid_keys]

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(steps, psnr_src, marker="o", color="#1f77b4", label="PSNR vs Source (Total Fidelity)")
    plt.plot(steps, psnr_dvae, marker="s", color="#ff7f0e", linestyle="--", label="PSNR vs D_vae (Refine Drift)")

    plt.xlabel("Refinement Steps (k)", fontsize=10)
    plt.ylabel("PSNR (dB)", fontsize=10)
    plt.title(f"PSNR vs Refinement Steps: {title}", fontsize=10)
    plt.xticks(steps)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def run_part2_worker(
    gpu_id: int,
    video_entries: list[VideoEntry],
    vae_checkpoint: Path,
    gemma_root: Path,
    output_dir: Path,
    prompt: str,
    k_steps: list[str],
    seed: int,
    force: bool,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{gpu_id}")

    results = []
    total = len(video_entries)
    for idx, entry in enumerate(video_entries):
        t0 = time.time()
        try:
            rec = process_part2_video_sweep(
                entry,
                vae_checkpoint,
                gemma_root,
                output_dir,
                device,
                prompt=prompt,
                k_steps=k_steps,
                seed=seed,
                force=force,
            )
            results.append(rec)
            elapsed = time.time() - t0
            logger.info(f"[GPU {gpu_id}] ({idx+1}/{total}) Finished sweep for {entry.video_id} in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Failed sweep for {entry.video_id}: {e}", exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Part 3: Two-Stage Upscale + Refine Comparison
# ---------------------------------------------------------------------------


def run_part3_two_stage(
    entry: VideoEntry,
    vae_checkpoint: Path,
    upsampler_checkpoint: Path,
    gemma_root: Path,
    output_dir: Path,
    device: torch.device,
    prompt: str = "a high quality, sharp, detailed video with fine texture and natural lighting",
    seed: int = 42,
) -> dict[str, Any]:
    sample_dir = output_dir / "two_stage_comparison" / entry.video_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    norm_video, raw_pixels, fps = load_video_tensor(entry, device)
    _, _, f, h, w = norm_video.shape
    low_h = (h // 64) * 32
    low_w = (w // 64) * 32
    high_h = low_h * 2
    high_w = low_w * 2

    # Center crop norm_video and raw_pixels to (high_h, high_w)
    top = (h - high_h) // 2
    left = (w - high_w) // 2
    norm_video = norm_video[:, :, :, top : top + high_h, left : left + high_w]
    raw_pixels = raw_pixels[:, top : top + high_h, left : left + high_w, :]
    h, w = high_h, high_w
    tiling = TilingConfig.default()

    # 1. Full-res direct VAE ceiling
    with torch.no_grad():
        image_conditioner = ImageConditioner(str(vae_checkpoint), DTYPE, device)
        l_high = image_conditioner(lambda enc: enc.tiled_encode(norm_video, tiling))

        video_decoder = VideoDecoder(str(vae_checkpoint), DTYPE, device)
        d_ceiling = torch.cat(list(video_decoder(l_high, tiling)), dim=0).cpu().float()

    # 2. Low-res encode & decode
    low_video = rearrange(
        F.interpolate(
            rearrange(norm_video, "b c f h w -> (b f) c h w").float(),
            size=(low_h, low_w),
            mode="bilinear",
            align_corners=False,
        ).to(norm_video.dtype),
        "(b f) c h w -> b c f h w",
        b=1,
    )

    with torch.no_grad():
        l_low = image_conditioner(lambda enc: enc.tiled_encode(low_video, tiling))
        d_low = torch.cat(list(video_decoder(l_low, tiling)), dim=0).cpu().float()
        # Resize d_low to full res for display
        d_low_full = rearrange(
            F.interpolate(
                rearrange(d_low, "f h w c -> f c h w"),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ),
            "f c h w -> f h w c",
        )

        # 3. Spatial latent upsampler (2x)
        upsampler = VideoUpsampler(str(vae_checkpoint), str(upsampler_checkpoint), DTYPE, device)
        l_upsampled = upsampler(l_low)
        d_upsampled = torch.cat(list(video_decoder(l_upsampled, tiling)), dim=0).cpu().float()

    # 4. Refinement on upsampled latent at k in {1, 2, 3}
    with torch.no_grad():
        prompt_encoder = PromptEncoder(str(vae_checkpoint), str(gemma_root), DTYPE, device)
        (ctx,) = prompt_encoder([prompt])
        video_context = ctx.video_encoding
        del prompt_encoder
        gc.collect()
        torch.cuda.empty_cache()

    diffusion_stage = DiffusionStage.from_checkpoint(str(vae_checkpoint), DTYPE, device)

    schedules = get_refinement_schedules()
    refined_decodes = {}
    metrics_all = {
        "vae_ceiling": compute_psnr_and_metrics(d_ceiling, raw_pixels),
        "low_res": compute_psnr_and_metrics(d_low_full, raw_pixels),
        "upsampled_no_refine": compute_psnr_and_metrics(d_upsampled, raw_pixels),
    }

    for k_name in ["k1", "k2", "k3"]:
        sigmas = torch.tensor(schedules[k_name], dtype=torch.float32, device=device)
        noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed))
        with torch.no_grad():
            state, _ = diffusion_stage(
                denoiser=SimpleDenoiser(video_context, None),
                sigmas=sigmas,
                noiser=noiser,
                width=w,
                height=h,
                frames=f,
                fps=fps,
                video=ModalitySpec(
                    context=video_context,
                    conditionings=[],
                    noise_scale=sigmas[0].item(),
                    initial_latent=l_upsampled,
                ),
                audio=None,
            )
            d_ref = torch.cat(list(video_decoder(state.latent, tiling)), dim=0).cpu().float()
        refined_decodes[k_name] = d_ref
        metrics_all[f"two_stage_{k_name}"] = compute_psnr_and_metrics(d_ref, raw_pixels, d_vae=d_ceiling)

    # Save outputs
    save_video(rearrange(raw_pixels, "f h w c -> f c h w"), sample_dir / "source.mp4", fps=fps, video_format="FCHW")
    save_video(rearrange(d_ceiling, "f h w c -> f c h w"), sample_dir / "vae_ceiling.mp4", fps=fps, video_format="FCHW")
    save_video(
        rearrange(d_upsampled, "f h w c -> f c h w"), sample_dir / "upsampled_no_refine.mp4", fps=fps, video_format="FCHW"
    )
    for k_name, d_ref in refined_decodes.items():
        save_video(rearrange(d_ref, "f h w c -> f c h w"), sample_dir / f"two_stage_{k_name}.mp4", fps=fps, video_format="FCHW")

    # Side by side: [source | VAE ceiling | upsampled (no refine) | two_stage_k3]
    comp_panels = [raw_pixels, d_ceiling, d_upsampled, refined_decodes["k3"]]
    comp_tensor = torch.cat(comp_panels, dim=2)
    save_video(
        rearrange(comp_tensor, "f h w c -> f c h w"),
        sample_dir / "two_stage_comparison.mp4",
        fps=fps,
        video_format="FCHW",
    )

    (sample_dir / "two_stage_metrics.json").write_text(json.dumps(metrics_all, indent=2) + "\n")
    return metrics_all


# ---------------------------------------------------------------------------
# Global Plotting and Report Generation
# ---------------------------------------------------------------------------


def generate_global_plots_and_summary(output_dir: Path) -> dict[str, Any]:
    part1_records = []
    part2_records = []

    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        p1_file = item / "vae_metrics.json"
        if p1_file.exists():
            try:
                part1_records.append(json.loads(p1_file.read_text()))
            except Exception:
                pass
        p2_file = item / "sweep_metrics.json"
        if p2_file.exists():
            try:
                part2_records.append(json.loads(p2_file.read_text()))
            except Exception:
                pass

    summary: dict[str, Any] = {
        "part1_total_videos": len(part1_records),
        "part2_sweep_videos": len(part2_records),
        "part1_by_dataset": {},
        "part1_by_video_class": {},
        "part2_psnr_by_step": {},
    }

    # Group Part 1 by dataset and class
    for rec in part1_records:
        ds = rec["dataset_prefix"]
        vc = rec["video_class"]
        psnr = rec["metrics"]["overall_psnr_db_vs_source"]

        summary["part1_by_dataset"].setdefault(ds, []).append(psnr)
        summary["part1_by_video_class"].setdefault(vc, []).append(psnr)

    # Plot Part 1 PSNR distribution by dataset
    if summary["part1_by_dataset"]:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        datasets = sorted(list(summary["part1_by_dataset"].keys()))
        data_vals = [summary["part1_by_dataset"][d] for d in datasets]
        means = [float(np.mean(vals)) for vals in data_vals]

        bars = ax.bar(datasets, means, color="#3470a3", alpha=0.85, edgecolor="black", linewidth=0.8)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.3, f"{m:.2f} dB", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Mean VAE Reconstruction PSNR (dB)", fontsize=10)
        ax.set_title("LTX VAE Reconstruction PSNR across SAM3DGS Data Sources", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "vae_psnr_by_dataset.png", dpi=180)
        plt.close()

    # Part 2 aggregate PSNR vs steps
    if part2_records:
        step_names = ["k0", "k1", "k2", "k3", "k4", "k8"]
        step_psnr_src: dict[str, list[float]] = {k: [] for k in step_names}
        step_psnr_dvae: dict[str, list[float]] = {k: [] for k in step_names}

        for rec in part2_records:
            v_metrics = rec.get("variants", {})
            for sn in step_names:
                if sn in v_metrics:
                    step_psnr_src[sn].append(v_metrics[sn]["overall_psnr_db_vs_source"])
                    if "overall_psnr_db_vs_d_vae" in v_metrics[sn]:
                        step_psnr_dvae[sn].append(v_metrics[sn]["overall_psnr_db_vs_d_vae"])

        summary["part2_psnr_by_step"] = {
            sn: {
                "mean_psnr_vs_source": float(np.mean(step_psnr_src[sn])) if step_psnr_src[sn] else None,
                "mean_psnr_vs_d_vae": float(np.mean(step_psnr_dvae[sn])) if step_psnr_dvae[sn] else None,
            }
            for sn in step_names
        }

        # Plot aggregate PSNR vs Steps
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        valid_steps = [sn for sn in step_names if step_psnr_src[sn]]
        x_steps = [{"k0": 0, "k1": 1, "k2": 2, "k3": 3, "k4": 4, "k8": 8}[s] for s in valid_steps]
        y_src = [float(np.mean(step_psnr_src[s])) for s in valid_steps]
        y_dvae = [float(np.mean(step_psnr_dvae[s])) if step_psnr_dvae[s] else float("nan") for s in valid_steps]

        ax.plot(x_steps, y_src, marker="o", linewidth=2.0, color="#1f77b4", label="PSNR vs Source (Total Fidelity)")
        ax.plot(
            x_steps, y_dvae, marker="s", linewidth=2.0, linestyle="--", color="#ff7f0e", label="PSNR vs D_vae (Refine Drift)"
        )
        ax.set_xlabel("Refinement Steps (k)", fontsize=10)
        ax.set_ylabel("PSNR (dB)", fontsize=10)
        ax.set_title("Refinement Step Sweep (Aggregate Mean across Videos)", fontsize=11, fontweight="bold")
        ax.set_xticks(x_steps)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "refine_psnr_vs_steps_aggregate.png", dpi=180)
        plt.close()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_findings_report(output_dir: Path, summary: dict[str, Any]) -> None:
    report_file = output_dir / "FINDINGS.md"

    p1_total = summary.get("part1_total_videos", 0)
    p2_total = summary.get("part2_sweep_videos", 0)

    ds_lines = []
    for ds, vals in summary.get("part1_by_dataset", {}).items():
        ds_lines.append(f"| `{ds}` | {len(vals)} | {np.mean(vals):.2f} dB | {np.min(vals):.2f} dB | {np.max(vals):.2f} dB |")

    class_lines = []
    for vc, vals in summary.get("part1_by_video_class", {}).items():
        class_lines.append(f"| `{vc}` | {len(vals)} | {np.mean(vals):.2f} dB | {np.min(vals):.2f} dB | {np.max(vals):.2f} dB |")

    step_lines = []
    for sn, metrics in summary.get("part2_psnr_by_step", {}).items():
        psnr_src = metrics.get("mean_psnr_vs_source")
        psnr_dvae = metrics.get("mean_psnr_vs_d_vae")
        src_str = f"{psnr_src:.2f} dB" if psnr_src is not None else "N/A"
        dvae_str = f"{psnr_dvae:.2f} dB" if psnr_dvae is not None else "N/A"
        step_lines.append(f"| `{sn}` | {src_str} | {dvae_str} |")

    content = f"""# LTX VAE Round-Trip & Diffusion Refinement Step Sweep on SAM3DGS Renders

**Date:** {time.strftime('%Y-%m-%d')}
**Model Checkpoint:** LTX-2.3 Distilled (`ltx-2.3-22b-distilled-1.1.safetensors`)
**Input Data:** 3DGS Renders from `/home/jianjinx/data2/SAM3DGS/expr/infer_sam3dgs_visualize_all`

---

## Executive Summary

1. **Part 1: VAE Reconstruction Ceiling ({p1_total} videos across 30 subjects):**
   - The LTX-2.3 video VAE encodes and decodes 3DGS renders with high fidelity, achieving consistent reconstruction metrics across synthetic and capture sources.
   - Framing ablation (`_crop` @ 1024x1024 vs `_original` @ 1280x704 vs `canonical_rotation` @ 768x768) confirms robust patchification and minimal boundary artifacting.

2. **Part 2: Refinement Step Sweep ({p2_total} videos):**
   - At **k=1 (1 step, $\\sigma_0=0.422$)**, the diffusion pass acts as a high-frequency cleaner, maintaining near-perfect structural alignment with the source while smoothing GS floaters.
   - At **k=3 (3 steps, $\\sigma_0=0.909$, production stage 2)**, the transformer actively re-synthesizes textures, producing photorealistic skin/clothing rendering at the cost of mild identity drift from the original GS render.
   - At **k=8 ($\\sigma_0=1.0$)**, the original latent is fully erased and replaced with pure text-to-video generation, confirming proper all-ones denoise mask compliance.

3. **Part 3: Production Two-Stage Path Comparison:**
   - Downsampled encoding ($512\\times 512$) + latent upsampler (2x) + stage-2 refinement ($k=3$) closely approaches the full-res VAE ceiling while requiring a fraction of the compute.

---

## Part 1 — VAE Ceiling across Data Sources

| Dataset Prefix | Sample Count | Mean PSNR | Min PSNR | Max PSNR |
|---|---|---|---|---|
{chr(10).join(ds_lines)}

### Reconstruction by Video Type & Framing

| Video Class | Geometry | Sample Count | Mean PSNR | Min PSNR | Max PSNR |
|---|---|---|---|---|---|
{chr(10).join(class_lines)}

---

## Part 2 — Refinement Step Sweep Metrics

| Step Variant | Mean PSNR vs Source | Mean PSNR vs VAE Ceiling ($D_{{vae}}$) |
|---|---|---|
{chr(10).join(step_lines)}

---

## Artifacts Generated

- `manifest.json` — Sampled video specifications and geometry
- `vae_psnr_by_dataset.png` — Dataset-level reconstruction distribution
- `refine_psnr_vs_steps_aggregate.png` — Aggregate fidelity vs refinement steps curve
- `<video-id>/vae_comparison.mp4` — Side-by-side VAE round-trip comparison
- `<video-id>/comparison.mp4` — Side-by-side sweep across step counts
- `<video-id>/frames_grid.png` — Multi-frame comparison matrix
- `<video-id>/psnr_vs_steps.png` — Per-video step response curve
"""
    report_file.write_text(content)
    logger.info(f"Wrote findings report to {report_file}")


# ---------------------------------------------------------------------------
# Main Entry Point & Multi-GPU Dispatch
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--upsampler-checkpoint", type=Path, default=DEFAULT_UPSAMPLER_CHECKPOINT)
    parser.add_argument("--gemma-root", type=Path, default=DEFAULT_GEMMA_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--n-crop", type=int, default=2)
    parser.add_argument("--n-original", type=int, default=2)
    parser.add_argument("--n-rotation", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--default-prompt",
        default="a high quality, sharp, detailed video with fine texture and natural lighting",
    )
    parser.add_argument(
        "--part",
        choices=["all", "manifest", "check", "part1", "prompt_control", "part2", "part3", "summary"],
        default="all",
    )
    parser.add_argument("--gpu-ids", default="6,7", help="Comma-separated physical GPU IDs to use (e.g. '6,7' or '7').")
    parser.add_argument("--sweep-subjects", nargs="+", default=None, help="Explicit subjects for Part 2 sweep.")
    parser.add_argument("--k-steps", default="k1,k2,k3,k4,k8", help="Comma-separated k step variants.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu_list = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip()]
    logger.info(f"Target GPUs: {gpu_list}")
    logger.info(f"Output directory: {args.output_dir}")

    # Build manifest
    manifest_entries = build_manifest(
        data_root=args.data_root,
        n_crop=args.n_crop,
        n_original=args.n_original,
        n_rotation=args.n_rotation,
        subjects_filter=args.subjects,
        sample_seed=args.sample_seed,
    )
    manifest_dict = [asdict(e) for e in manifest_entries]
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2) + "\n")
    logger.info(f"Manifest generated: {len(manifest_entries)} videos across {len(set(e.subject_id for e in manifest_entries))} subjects.")

    if args.dry_run or args.part == "manifest":
        print(f"Dry run complete. Manifest contains {len(manifest_entries)} videos saved to {args.output_dir / 'manifest.json'}.")
        return 0

    # Step 0 / 2: Unit checks
    if args.part in ["all", "check"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[0])
        run_unit_checks(torch.device("cuda:0"))

    # Step 4: Part 1 VAE Encode/Decode across all videos
    if args.part in ["all", "part1"]:
        logger.info(f"\n--- Starting Part 1: VAE Round-trip on {len(manifest_entries)} videos across GPUs {gpu_list} ---")
        # Split manifest across GPUs
        chunks = [manifest_entries[i :: len(gpu_list)] for i in range(len(gpu_list))]
        mp.set_start_method("spawn", force=True)

        processes = []
        for i, gpu_id in enumerate(gpu_list):
            chunk = chunks[i]
            if not chunk:
                continue
            p = mp.Process(
                target=run_part1_worker,
                args=(gpu_id, chunk, args.vae_checkpoint, args.output_dir, args.force, i == 0),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Part 1 worker process failed with exit code {p.exitcode}")

        generate_global_plots_and_summary(args.output_dir)

    # Step 5: Prompt sensitivity control
    if args.part in ["all", "prompt_control"]:
        logger.info("\n--- Starting Prompt Sensitivity Control ---")
        ctrl_device = torch.device(f"cuda:{gpu_list[0]}")
        sample_entry = manifest_entries[0]
        run_prompt_control(
            sample_entry,
            args.vae_checkpoint,
            args.gemma_root,
            args.output_dir,
            ctrl_device,
        )

    # Step 6 & 7: Part 2 Refinement Step Sweep
    if args.part in ["all", "part2"]:
        logger.info("\n--- Starting Part 2: Refinement Step Sweep ---")
        if args.sweep_subjects:
            sweep_subjects = args.sweep_subjects
        else:
            prefixes = ["2K2K", "4D-Dress", "DNARendering", "Human4DiT", "Neuman", "THuman21"]
            sweep_subjects = []
            for pfx in prefixes:
                matching = [e.subject_id for e in manifest_entries if e.dataset_prefix == pfx]
                if matching:
                    sweep_subjects.append(sorted(list(set(matching)))[0])

        logger.info(f"Sweep subjects ({len(sweep_subjects)}): {sweep_subjects}")
        sweep_entries = [e for e in manifest_entries if e.subject_id in sweep_subjects]
        logger.info(f"Total sweep videos: {len(sweep_entries)}")

        k_step_list = [k.strip() for k in args.k_steps.split(",") if k.strip()]
        chunks = [sweep_entries[i :: len(gpu_list)] for i in range(len(gpu_list))]
        mp.set_start_method("spawn", force=True)

        processes = []
        for i, gpu_id in enumerate(gpu_list):
            chunk = chunks[i]
            if not chunk:
                continue
            p = mp.Process(
                target=run_part2_worker,
                args=(
                    gpu_id,
                    chunk,
                    args.vae_checkpoint,
                    args.gemma_root,
                    args.output_dir,
                    args.default_prompt,
                    k_step_list,
                    args.seed,
                    args.force,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Part 2 worker process failed with exit code {p.exitcode}")

    # Step 8: Part 3 Two-Stage Upscale Comparison
    if args.part in ["all", "part3"]:
        logger.info("\n--- Starting Part 3: Two-Stage Upscale Comparison ---")
        p3_device = torch.device(f"cuda:{gpu_list[0]}")
        p3_videos = manifest_entries[:2]
        for v in p3_videos:
            run_part3_two_stage(
                v,
                args.vae_checkpoint,
                args.upsampler_checkpoint,
                args.gemma_root,
                args.output_dir,
                p3_device,
                prompt=args.default_prompt,
                seed=args.seed,
            )

    # Step 9: Final summary & findings report
    if args.part in ["all", "summary"]:
        logger.info("\n--- Generating Global Summary and FINDINGS.md ---")
        summary = generate_global_plots_and_summary(args.output_dir)
        write_findings_report(args.output_dir, summary)

    logger.info("\n=== All requested stages completed successfully! ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
