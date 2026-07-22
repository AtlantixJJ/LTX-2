"""
Focused VAE tail visualization + reconstruction comparison over multiple videos.

For each input video this script:
  1. visualizes PCA maps for every encoder down_block plus the bottleneck tail
     (post PixelNorm -> raw conv_out means -> normalized latent);
  2. decodes the normalized latent back to pixels, visualizing PCA maps for every
     decoder up_block plus its tail (post PixelNorm -> pre-unpatchify conv_out),
     and saves a side-by-side [original (resized) | decoded | |diff|] comparison video;
  3. computes reconstruction PSNR and compares it across all input videos.

Videos may live in different folders -- pass full paths via --videos. Outputs
are written under --output-dir, which is wiped and recreated at the start of
each run.

Run:
    python3 scripts/visualize_vae.py

Useful overrides:
    python3 scripts/visualize_vae.py --max-analysis-frames 145
    python3 scripts/visualize_vae.py --videos /path/a/one.mp4,/path/b/two.mp4
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
from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig  # noqa: E402
from ltx_core.model.video_vae.video_vae import (  # noqa: E402
    map_spatial_interval_to_latent,
    map_spatial_slice,
    map_temporal_interval_to_latent,
    map_temporal_slice,
    to_mapping_operation,
)
from ltx_core.tiling import (  # noqa: E402
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    LatentIntervals,
    MappingOperation,
    SplitOperation,
    Tile,
    create_tiles_from_intervals_and_mappers,
    identity_mapping_operation,
)
from ltx_core.tiling import split_by_size as split_in_spatial  # noqa: E402
from ltx_core.tiling import split_temporal  # noqa: E402
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
# interior-rank cases found in the dataset analysis, plus a real (non-generated)
# human capture. Different videos deliberately live in different folders.
DEFAULT_VIDEO_PATHS = [
    WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw" / "static_green_teapot_seed_12345.mp4",
    WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw" / "static_red_bicycle_seed_54321.mp4",
    WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw" / "dolly_blue_ball_seed_12345.mp4",
    WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw" / "composition_books_seed_12345.mp4",
    WORKSPACE_ROOT / "expr" / "video-eval-baseline-v1" / "raw" / "cat_behind_box_seed_54321.mp4",
    WORKSPACE_ROOT / "data" / "DNARendering" / "Videos" / "Part_5" / "0264_01" / "264_00001_Camera_5mp_cam025.mp4",
    REPO_ROOT / "results/test_videos/dnarendering_refined.mp4",
]

TAIL_STAGES = [
    ("post_pixelnorm", "after PixelNorm"),
    ("conv_out_means", "after conv_out means"),
    ("latent_normalized", "normalized latent"),
]

DECODER_TAIL_STAGES = [
    ("dec_post_pixelnorm", "after PixelNorm"),
    ("dec_conv_out", "after conv_out (pre-unpatchify)"),
]


def build_stage_defs(encoder: torch.nn.Module) -> list[tuple[str, str]]:
    """Enumerate one raw stage per encoder down_block, plus the fixed bottleneck tail."""
    block_stages = [(f"enc_b{i}", f"enc block {i} raw") for i in range(len(encoder.down_blocks))]
    return block_stages + TAIL_STAGES


def build_decoder_stage_defs(decoder: torch.nn.Module) -> list[tuple[str, str]]:
    """Enumerate one raw stage per decoder up_block, plus the fixed tail (mirrors build_stage_defs)."""
    block_stages = [(f"dec_b{i}", f"dec block {i} raw") for i in range(len(decoder.up_blocks))]
    return block_stages + DECODER_TAIL_STAGES


# Mirrors the minimums ltx-core's prepare_tiles_for_encoding enforces, so our per-block tiling
# lines up exactly with the tiles the real tiled_encode/tiled_decode would use.
MIN_SPATIAL_TILE_OVERLAP_PX = 64
MIN_TEMPORAL_TILE_OVERLAP_FRAMES = 16


def build_tiling_config(args: argparse.Namespace) -> TilingConfig | None:
    if not args.tiled:
        return None
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=args.tile_size_pixels,
            tile_overlap_in_pixels=args.tile_overlap_pixels,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.tile_size_frames,
            tile_overlap_in_frames=args.tile_overlap_frames,
        ),
    )


def stage_scale_plan(encoder: torch.nn.Module) -> list[tuple[str, int, int]]:
    """(stage_key, spatial_scale, temporal_scale) per stage, cumulative relative to the raw
    input video, in the same order as build_stage_defs. A block contributes to the cumulative
    scale only if it's a SpaceToDepthDownsample (identified by its "stride" attribute, absent
    from the non-downsampling res_x/res_x_y mid-blocks)."""
    spatial_scale = encoder.patch_size
    temporal_scale = 1
    plan: list[tuple[str, int, int]] = []
    for index, block in enumerate(encoder.down_blocks):
        if hasattr(block, "stride"):
            t, h, _w = block.stride
            temporal_scale *= t
            spatial_scale *= h
        plan.append((f"enc_b{index}", spatial_scale, temporal_scale))
    for key, _label in TAIL_STAGES:
        plan.append((key, spatial_scale, temporal_scale))
    return plan


def decoder_stage_scale_plan(decoder: torch.nn.Module) -> list[tuple[str, int, int]]:
    """(stage_key, spatial_scale, temporal_scale) per stage, cumulative relative to the raw pixel-space
    output, in the same order as build_decoder_stage_defs. Mirror image of stage_scale_plan: starts at
    the full latent scale (32 spatial / 8 temporal) and divides down towards patch_size as up_blocks run,
    since -- unlike the encoder -- the decoder's blocks upsample rather than downsample."""
    spatial_scale = decoder.video_downscale_factors.height
    temporal_scale = decoder.video_downscale_factors.time
    plan: list[tuple[str, int, int]] = []
    for index, block in enumerate(decoder.up_blocks):
        if hasattr(block, "stride"):
            t, h, _w = block.stride
            temporal_scale //= t
            spatial_scale //= h
        plan.append((f"dec_b{index}", spatial_scale, temporal_scale))
    for key, _label in DECODER_TAIL_STAGES:
        plan.append((key, spatial_scale, temporal_scale))
    return plan


def video_axis_intervals(video_shape: torch.Size, tiling_config: TilingConfig) -> LatentIntervals:
    """Split the (B, C, F, H, W) video tensor into the same video-pixel-space tile boundaries
    ltx-core's tiled_encode uses (see prepare_tiles_for_encoding), so per-block feature tiles
    line up exactly with the tiles used for the real tiled encode/decode."""
    splitters: list[SplitOperation] = [DEFAULT_SPLIT_OPERATION] * len(video_shape)
    if tiling_config.spatial_config is not None:
        cfg = tiling_config.spatial_config
        overlap_px = max(cfg.tile_overlap_in_pixels, MIN_SPATIAL_TILE_OVERLAP_PX)
        splitters[3] = split_in_spatial(cfg.tile_size_in_pixels, overlap_px)
        splitters[4] = split_in_spatial(cfg.tile_size_in_pixels, overlap_px)
    if tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        overlap_frames = max(cfg.tile_overlap_in_frames, MIN_TEMPORAL_TILE_OVERLAP_FRAMES)
        splitters[2] = split_temporal(cfg.tile_size_in_frames, overlap_frames)
    intervals = [splitter(length) for splitter, length in zip(splitters, video_shape)]
    return LatentIntervals(original_shape=video_shape, dimension_intervals=tuple(intervals))


def tile_in_coords(intervals: LatentIntervals) -> list[tuple[slice, ...]]:
    """Video-pixel-space input slice per tile, in the stable order shared by every stage
    (in_coords come purely from the input intervals, independent of the output mapping).
    Uses identity_mapping_operation (not DEFAULT_MAPPING_OPERATION) since axes can have more
    than one interval here -- the plain default always emits a single output slice and would
    under-count once an axis is actually split into tiles."""
    identity_mappers: list[MappingOperation] = [identity_mapping_operation] * len(intervals.original_shape)
    tiles = create_tiles_from_intervals_and_mappers(intervals, identity_mappers)
    return [tile.in_coords for tile in tiles]


def tiles_at_scale(intervals: LatentIntervals, spatial_scale: int, temporal_scale: int) -> list[Tile]:
    """Out-coords + blend masks for a stage whose grid is downsampled by spatial_scale/
    temporal_scale relative to the raw input video (e.g. scale=32/8 reproduces the final
    latent tiling ltx-core uses)."""
    mappers: list[MappingOperation] = [DEFAULT_MAPPING_OPERATION] * len(intervals.original_shape)
    mappers[2] = to_mapping_operation(map_temporal_interval_to_latent, scale=temporal_scale)
    mappers[3] = to_mapping_operation(map_spatial_interval_to_latent, scale=spatial_scale)
    mappers[4] = to_mapping_operation(map_spatial_interval_to_latent, scale=spatial_scale)
    return create_tiles_from_intervals_and_mappers(intervals, mappers)


def latent_axis_intervals(latent_shape: torch.Size, tiling_config: TilingConfig, decoder: torch.nn.Module) -> LatentIntervals:
    """Split the (B, C, F', H', W') latent tensor into tile boundaries expressed directly in latent
    units, converting the (pixel/frame) --tile-size-* args via decoder.video_downscale_factors so the
    tiles line up with the pixel-space tiles ltx-core's tiled_decode/_prepare_tiles uses."""
    scales = decoder.video_downscale_factors
    splitters: list[SplitOperation] = [DEFAULT_SPLIT_OPERATION] * len(latent_shape)
    if tiling_config.spatial_config is not None:
        cfg = tiling_config.spatial_config
        overlap_px = max(cfg.tile_overlap_in_pixels, MIN_SPATIAL_TILE_OVERLAP_PX)
        tile_size = max(2, cfg.tile_size_in_pixels // scales.height)
        overlap = max(1, overlap_px // scales.height)
        splitters[3] = split_in_spatial(tile_size, overlap)
        splitters[4] = split_in_spatial(tile_size, overlap)
    if tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        overlap_frames = max(cfg.tile_overlap_in_frames, MIN_TEMPORAL_TILE_OVERLAP_FRAMES)
        tile_size = max(2, cfg.tile_size_in_frames // scales.time)
        overlap = max(1, overlap_frames // scales.time)
        splitters[2] = split_temporal(tile_size, overlap)
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape)]
    return LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))


def decoder_tiles_at_scale(
    intervals: LatentIntervals, spatial_scale: int, temporal_scale: int, full_spatial_scale: int, full_temporal_scale: int
) -> list[Tile]:
    """Out-coords + blend masks for a decoder stage whose grid is upsampled by full_scale/scale
    relative to the raw input latent tile boundaries -- the mirror image of tiles_at_scale, using the
    same multiply-based map_spatial_slice/map_temporal_slice decoder._prepare_tiles uses internally."""
    mappers: list[MappingOperation] = [DEFAULT_MAPPING_OPERATION] * len(intervals.original_shape)
    mappers[2] = to_mapping_operation(map_temporal_slice, scale=full_temporal_scale // temporal_scale)
    mappers[3] = to_mapping_operation(map_spatial_slice, scale=full_spatial_scale // spatial_scale)
    mappers[4] = to_mapping_operation(map_spatial_slice, scale=full_spatial_scale // spatial_scale)
    return create_tiles_from_intervals_and_mappers(intervals, mappers)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors",
        help="VAE checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "vae_multivideo_tail_pca",
        help="Output directory; wiped and recreated at the start of each run.",
    )
    parser.add_argument(
        "--videos",
        default=None,
        help="Comma-separated video file paths; each may live in a different folder. "
        f"Defaults to the built-in set of {len(DEFAULT_VIDEO_PATHS)} sample videos.",
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
        default=None,
        help="Optional center-crop target aspect ratio as width:height, e.g. 1:1.5 for a portrait "
        "human capture. Omit to skip cropping (kept as-is before the --max-side resize).",
    )
    parser.add_argument("--num-heatmap-frames", type=int, default=4)
    parser.add_argument(
        "--diff-gain",
        type=float,
        default=4.0,
        help="Multiplier applied to |original - decoded| before clamping to [0, 1] for the diff panel.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Encode/decode using spatial+temporal tiling (ltx-core TilingConfig) instead of a single "
        "monolithic pass, allowing larger resolutions. Per-block feature maps are stitched from the "
        "same tiles used for the real tiled encode/decode. Independent of --max-side.",
    )
    parser.add_argument("--tile-size-pixels", type=int, default=768, help="Spatial tile size in pixels.")
    parser.add_argument("--tile-overlap-pixels", type=int, default=64, help="Spatial tile overlap in pixels.")
    parser.add_argument("--tile-size-frames", type=int, default=80, help="Temporal tile size in frames.")
    parser.add_argument("--tile-overlap-frames", type=int, default=24, help="Temporal tile overlap in frames.")
    return parser


def video_paths(args: argparse.Namespace) -> list[Path]:
    if args.videos:
        return [Path(item.strip()).expanduser() for item in args.videos.split(",") if item.strip()]
    return DEFAULT_VIDEO_PATHS


def unique_sample_id(video_path: Path, used_ids: set[str]) -> str:
    """Derive an output-folder-safe id from a video path, disambiguating stem collisions
    across different source folders (e.g. two "raw/video.mp4" from different roots)."""
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
    #if output_dir.exists():
    #    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def center_crop_to_aspect(video: torch.Tensor, aspect_wh: str | None) -> torch.Tensor:
    if aspect_wh is None:
        return video
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
    fig.suptitle("VAE tail PCA overview: middle latent frame per video")
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
def encode_tail_stages_tiled(
    encoder: torch.nn.Module, video: torch.Tensor, tiling_config: TilingConfig
) -> dict[str, torch.Tensor]:
    """Tiled equivalent of encode_tail_stages: splits the raw video into the same tiles
    VideoEncoder.tiled_encode uses, runs each tile through conv_in + every down_block (capturing
    every stage's tile output along the way), and blends tiles back together per stage using the
    matching scale mapping + trapezoidal/rectangular masks -- exactly mirroring how tiled_encode
    stitches the final latent, just applied at every intermediate resolution too."""
    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype
    _, _, frames, height, width = video.shape

    intervals = video_axis_intervals(video.shape, tiling_config)
    in_coords_list = tile_in_coords(intervals)
    scale_plan = stage_scale_plan(encoder)
    tiles_by_stage = {key: tiles_at_scale(intervals, s_scale, t_scale) for key, s_scale, t_scale in scale_plan}
    scale_by_stage = {key: (s_scale, t_scale) for key, s_scale, t_scale in scale_plan}

    stage_buffers: dict[str, torch.Tensor] = {}
    weight_buffers: dict[str, torch.Tensor] = {}

    def accumulate(stage_key: str, tensor: torch.Tensor, tile_idx: int) -> None:
        if stage_key not in stage_buffers:
            spatial_scale, temporal_scale = scale_by_stage[stage_key]
            full_shape = (
                1,
                tensor.shape[1],
                ((frames - 1) // temporal_scale) + 1,
                height // spatial_scale,
                width // spatial_scale,
            )
            stage_buffers[stage_key] = torch.zeros(full_shape, device=device, dtype=dtype)
            weight_buffers[stage_key] = torch.zeros(full_shape, device=device, dtype=dtype)
        tile = tiles_by_stage[stage_key][tile_idx]
        mask = tile.blend_mask.to(device=device, dtype=dtype)
        stage_buffers[stage_key][tile.out_coords] += tensor.detach() * mask
        weight_buffers[stage_key][tile.out_coords] += mask

    for tile_idx, in_coords in enumerate(in_coords_list):
        video_tile = video[in_coords].to(device=device, dtype=dtype)
        x = encoder.conv_in(patchify(video_tile, patch_size_hw=4, patch_size_t=1))
        for index, block in enumerate(encoder.down_blocks):
            x = block(x)
            accumulate(f"enc_b{index}", x, tile_idx)

        post_pixelnorm = encoder.conv_norm_out(x)
        conv_out_means = encoder.conv_out(encoder.conv_act(post_pixelnorm))[:, : encoder.latent_channels]
        latent = encoder.per_channel_statistics.normalize(conv_out_means)
        accumulate("post_pixelnorm", post_pixelnorm, tile_idx)
        accumulate("conv_out_means", conv_out_means, tile_idx)
        accumulate("latent_normalized", latent, tile_idx)
        del video_tile, x, post_pixelnorm, conv_out_means, latent

    for key, buffer in stage_buffers.items():
        stage_buffers[key] = buffer / weight_buffers[key].clamp(min=1e-8)
    return stage_buffers


def postprocess_decoded_video(video: torch.Tensor) -> torch.Tensor:
    """Convert the decoder's raw output ([1, C, F, H, W], approximately in [-1, 1]) to [F, C, H, W] in [0, 1]."""
    video = rearrange(video, "1 c f h w -> f c h w")
    return ((video + 1) / 2).clamp(0, 1).float()


@torch.no_grad()
def decode_latent_to_video(decoder: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Decode a normalized latent back to pixel space as [F, C, H, W] in [0, 1]."""
    video = decoder(latent)  # [1, C, F, H, W], approximately in [-1, 1]
    return postprocess_decoded_video(video)


@torch.no_grad()
def decode_tail_stages(decoder: torch.nn.Module, latent: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Mirrors encode_tail_stages, but for the decoder: captures every up_block's raw output plus the
    fixed tail (post PixelNorm, pre-unpatchify conv_out) via forward hooks on a single real decoder
    forward pass, so noise injection / timestep-conditioning AdaLN behave exactly as in decode_latent_to_video
    -- reimplementing that logic by hand would be easy to get subtly wrong. Returns the captured stages
    alongside the decoder's raw output so callers don't need a second forward pass for the final video."""
    stages: dict[str, torch.Tensor] = {}

    def make_hook(key: str):
        def hook(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            stages[key] = output.detach()

        return hook

    handles = [block.register_forward_hook(make_hook(f"dec_b{i}")) for i, block in enumerate(decoder.up_blocks)]
    handles.append(decoder.conv_norm_out.register_forward_hook(make_hook("dec_post_pixelnorm")))
    handles.append(decoder.conv_out.register_forward_hook(make_hook("dec_conv_out")))
    try:
        video = decoder(latent)
    finally:
        for handle in handles:
            handle.remove()
    return stages, video


@torch.no_grad()
def decode_tail_stages_tiled(
    decoder: torch.nn.Module, latent: torch.Tensor, tiling_config: TilingConfig
) -> dict[str, torch.Tensor]:
    """Tiled equivalent of decode_tail_stages: splits the latent into the same latent-space tiles
    decoder.tiled_decode effectively uses (see latent_axis_intervals), runs each tile through the full
    decoder (capturing every stage's tile output via decode_tail_stages), and blends tiles back together
    per stage using decoder_tiles_at_scale's matching scale mapping + blend masks -- mirroring
    encode_tail_stages_tiled but with the multiply-based (upsampling) mapping the decoder needs."""
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype
    _, _, frames, height, width = latent.shape
    scales = decoder.video_downscale_factors

    intervals = latent_axis_intervals(latent.shape, tiling_config, decoder)
    in_coords_list = tile_in_coords(intervals)
    scale_plan = decoder_stage_scale_plan(decoder)
    tiles_by_stage = {
        key: decoder_tiles_at_scale(intervals, s_scale, t_scale, scales.height, scales.time)
        for key, s_scale, t_scale in scale_plan
    }
    scale_by_stage = {key: (s_scale, t_scale) for key, s_scale, t_scale in scale_plan}

    stage_buffers: dict[str, torch.Tensor] = {}
    weight_buffers: dict[str, torch.Tensor] = {}

    def accumulate(stage_key: str, tensor: torch.Tensor, tile_idx: int) -> None:
        if stage_key not in stage_buffers:
            spatial_scale, temporal_scale = scale_by_stage[stage_key]
            spatial_mult = scales.height // spatial_scale
            temporal_mult = scales.time // temporal_scale
            full_shape = (
                1,
                tensor.shape[1],
                1 + (frames - 1) * temporal_mult,
                height * spatial_mult,
                width * spatial_mult,
            )
            stage_buffers[stage_key] = torch.zeros(full_shape, device=device, dtype=dtype)
            weight_buffers[stage_key] = torch.zeros(full_shape, device=device, dtype=dtype)
        tile = tiles_by_stage[stage_key][tile_idx]
        mask = tile.blend_mask.to(device=device, dtype=dtype)
        stage_buffers[stage_key][tile.out_coords] += tensor.detach() * mask
        weight_buffers[stage_key][tile.out_coords] += mask

    for tile_idx, in_coords in enumerate(in_coords_list):
        latent_tile = latent[in_coords].to(device=device, dtype=dtype)
        tile_stages, _video = decode_tail_stages(decoder, latent_tile)
        for stage_key, tensor in tile_stages.items():
            accumulate(stage_key, tensor, tile_idx)
        del latent_tile, tile_stages, _video

    for key, buffer in stage_buffers.items():
        stage_buffers[key] = buffer / weight_buffers[key].clamp(min=1e-8)
    return stage_buffers


@torch.no_grad()
def decode_latent_to_video_tiled(
    decoder: torch.nn.Module, latent: torch.Tensor, tiling_config: TilingConfig
) -> torch.Tensor:
    """Tiled decode via ltx-core's VideoDecoder.tiled_decode, same [F, C, H, W] in [0, 1] output."""
    chunks = list(decoder.tiled_decode(latent, tiling_config=tiling_config))
    video = torch.cat(chunks, dim=2)  # [1, C, F, H, W], concat along frames
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


def compute_psnr(original: torch.Tensor, decoded: torch.Tensor) -> dict[str, Any]:
    """PSNR between original (resized) and decoded pixel videos, both [F, C, H, W] in [0, 1]."""
    num_frames = min(original.shape[0], decoded.shape[0])
    original = original[:num_frames].float()
    decoded = decoded[:num_frames].float()
    per_frame_mse = ((original - decoded) ** 2).flatten(1).mean(dim=1)
    per_frame_psnr = (10 * torch.log10(1.0 / per_frame_mse.clamp(min=1e-12))).tolist()
    overall_mse = float(per_frame_mse.mean())
    overall_psnr = float(10 * np.log10(1.0 / max(overall_mse, 1e-12)))
    return {
        "overall_psnr_db": overall_psnr,
        "overall_mse": overall_mse,
        "per_frame_psnr_db": [float(v) for v in per_frame_psnr],
    }


def plot_psnr_comparison(records: list[dict[str, Any]], save_path: Path) -> None:
    labels = [r["sample_id"] for r in records]
    values = [r["psnr"]["overall_psnr_db"] for r in records]
    plt.figure(figsize=(max(6.0, 1.2 * len(labels)), 4.5))
    bars = plt.bar(labels, values, color="#4C72B0")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Reconstruction PSNR (dB)")
    plt.title("VAE reconstruction PSNR across videos")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def visualize_stage(
    sample_dir: Path,
    sample_id: str,
    stage_key: str,
    feature: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x, comps = pca_projection(feature, PCA_RGB_COMPONENTS)
    rgb = rgb_from_components(comps)
    rank = exact_rank_metrics(x)
    stats = channel_stats(x)

    prefix = sample_dir / f"{sample_id}_{stage_key}"
    title = f"{stage_key} {tuple(int(v) for v in feature.shape)}"
    plot_rgb_frames(
        rgb,
        title,
        prefix.with_name(prefix.name + "_rgb_frames.png"),
        args.num_heatmap_frames,
    )
    plot_variance_curve(
        rank,
        title,
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
    selected = video_paths(args)
    for video_path in selected:
        if not video_path.exists():
            raise FileNotFoundError(video_path)

    tiling_config = build_tiling_config(args)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Tiled: {args.tiled}" + (f" (spatial {args.tile_size_pixels}px/{args.tile_overlap_pixels}px overlap, "
          f"temporal {args.tile_size_frames}f/{args.tile_overlap_frames}f overlap)" if args.tiled else ""))
    print(f"Videos:\n  " + "\n  ".join(str(p) for p in selected))

    encoder = load_video_vae_encoder(str(args.checkpoint), device=device, dtype=DTYPE)
    decoder = load_video_vae_decoder(str(args.checkpoint), device=device, dtype=DTYPE)
    stages_def = build_stage_defs(encoder)
    decoder_stages_def = build_decoder_stage_defs(decoder)
    overview_records: list[dict[str, Any]] = []
    decoder_overview_records: list[dict[str, Any]] = []
    psnr_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "max_analysis_frames": args.max_analysis_frames,
        "videos": [str(p) for p in selected],
        "tiled": args.tiled,
        "tiling_config": (
            {
                "tile_size_pixels": args.tile_size_pixels,
                "tile_overlap_pixels": args.tile_overlap_pixels,
                "tile_size_frames": args.tile_size_frames,
                "tile_overlap_frames": args.tile_overlap_frames,
            }
            if args.tiled
            else None
        ),
        "records": [],
    }

    used_ids: set[str] = set()
    for video_path in selected:
        sample_id = unique_sample_id(video_path, used_ids)

        print(f"Visualizing {sample_id} ({video_path})")
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        video = load_video(video_path, device, args)
        if tiling_config is not None:
            stages = encode_tail_stages_tiled(encoder, video, tiling_config)
        else:
            stages = encode_tail_stages(encoder, video)

        overview_stage_records: dict[str, Any] = {}
        sample_summary: dict[str, Any] = {
            "sample_id": sample_id,
            "video_path": str(video_path),
            "stages": {},
        }
        for stage_key, _stage_label in stages_def:
            stage_record = visualize_stage(sample_dir, sample_id, stage_key, stages[stage_key], args)
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

        decoder_overview_stage_records: dict[str, Any] = {}
        if tiling_config is not None:
            decoder_stages = decode_tail_stages_tiled(decoder, stages["latent_normalized"], tiling_config)
            decoded = decode_latent_to_video_tiled(decoder, stages["latent_normalized"], tiling_config)
        else:
            decoder_stages, raw_decoded = decode_tail_stages(decoder, stages["latent_normalized"])
            decoded = postprocess_decoded_video(raw_decoded)
            del raw_decoded
        for stage_key, _stage_label in decoder_stages_def:
            stage_record = visualize_stage(sample_dir, sample_id, stage_key, decoder_stages[stage_key], args)
            decoder_overview_stage_records[stage_key] = {
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
        del decoder_stages

        original = prepare_pixel_video(video)
        comparison_path = sample_dir / f"{sample_id}_comparison.mp4"
        save_comparison_video(original, decoded, comparison_path, video_fps(video_path), args.diff_gain)
        psnr = compute_psnr(original, decoded)
        sample_summary["psnr"] = psnr
        psnr_records.append({"sample_id": sample_id, "psnr": psnr})
        print(
            f"  saved comparison video (original | decoded | diff) to {comparison_path}\n"
            f"  reconstruction PSNR: {psnr['overall_psnr_db']:.2f} dB (MSE={psnr['overall_mse']:.3e})"
        )

        overview_records.append({"sample_id": sample_id, "stages": overview_stage_records})
        decoder_overview_records.append({"sample_id": sample_id, "stages": decoder_overview_stage_records})
        summary["records"].append(sample_summary)
        (sample_dir / f"{sample_id}_summary.json").write_text(json.dumps(sample_summary, indent=2) + "\n")

        del video, decoded, original, stages
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nReconstruction PSNR comparison (worst to best):")
    for rec in sorted(psnr_records, key=lambda r: r["psnr"]["overall_psnr_db"]):
        print(f"  {rec['sample_id']:<32} {rec['psnr']['overall_psnr_db']:6.2f} dB")

    plot_overview(overview_records, stages_def, args.output_dir / "tail_pca_overview.png")
    plot_overview(decoder_overview_records, decoder_stages_def, args.output_dir / "decoder_tail_pca_overview.png")
    plot_psnr_comparison(psnr_records, args.output_dir / "psnr_comparison.png")
    summary["psnr_comparison"] = psnr_records
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved overview to {args.output_dir / 'tail_pca_overview.png'}")
    print(f"Saved decoder overview to {args.output_dir / 'decoder_tail_pca_overview.png'}")
    print(f"Saved PSNR comparison to {args.output_dir / 'psnr_comparison.png'}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
