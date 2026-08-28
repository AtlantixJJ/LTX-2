"""Sliding-window K-step diffusion refinement for very long SAM3DGS videos.

vae_refine_step_sweep.py caps every video at ~121 frames (one VAE-grid-valid chunk),
which is fine for short clips but throws away almost all of a multi-thousand-frame
motion-capture render. This script instead walks a long video in overlapping windows,
VAE-encodes + K-step-refines + VAE-decodes each one, and stitches them into one
continuous output.

Window-to-window continuity: the overlap is NOT just a post-hoc pixel cross-fade.
Each window's last `overlap_frames` worth of *refined latent* is frozen (via
VideoConditionByLatentIndex, strength=1.0) into the start of the next window, so the
next window continues from the previous window's literal output instead of
re-deriving a fresh guess at the same source content. That fresh-guess re-derivation
is what caused a visible appearance shift at every boundary in an earlier version of
this script: each window renoises from its own local frame 0 (different RoPE
position, different point in the same fixed noise draw for "the same" source
frames), so two independently-refined windows can genuinely diverge in generated
appearance -- with only a couple of frames of cross-fade, that divergence showed as
a pop rather than a fade. `--overlap-frames` must be a multiple of 8 (the VAE's
temporal downsampling) so the carried-over span is a whole number of latent frames.
The pixel cross-fade is still applied on top, now mostly covering the VAE decoder's
own small receptive-field boundary effects rather than doing the heavy lifting.

Memory/robustness: only ever holds one window's tensors in memory (not the whole
video), and periodically flushes finalized frames to a segment .mp4 under
<out>/segments/, then rebuilds <out>/decode_full.mp4 via an ffmpeg concat (stream
copy, no re-encode) so there is always a playable, up-to-date file on disk. Each
window's raw decode is also cached to <out>/window_cache/ so an interrupted run
resumes without redoing GPU work for already-finished windows.

Usage:
  conda run -n ltx python3 scripts/vae_refine_sliding_window.py \
      --video /path/to/long_video.mp4 \
      --output-dir expr/sam3dgs_vae_refine/<video-id>/k2_longform \
      --k-step k2 --gpu-id 7
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

torch.cuda.init()
import decord  # noqa: E402
from einops import rearrange  # noqa: E402

decord.bridge.set_bridge("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-pipelines" / "src"))

from ltx_core.batch_split import BatchSplitAdapter  # noqa: E402
from ltx_core.components.diffusion_steps import EulerDiffusionStep  # noqa: E402
from ltx_core.model.transformer import LTXModelConfigurator, LTXVideoOnlyModelConfigurator  # noqa: E402
from ltx_pipelines.utils.blocks import (  # noqa: E402
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, VIDEO_SCALE_FACTORS  # noqa: E402
from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: E402
from ltx_pipelines.utils.gpu_model import gpu_model  # noqa: E402
from ltx_pipelines.utils.samplers import _step_state  # noqa: E402, PLC2701 -- deliberately reusing
                   # the sampler's own step primitive so each diffusion step can be timed
                   # individually; refine_core.run_schedule is the same three lines uninstrumented.
from ltx_trainer.video_utils import save_video  # noqa: E402

from scripts.prune.core import geometry, model_registry, preflight, refine_core  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vae_refine_sliding_window")

DTYPE = torch.bfloat16
DEFAULT_PROMPT = "a high quality, sharp, detailed video with fine texture and natural lighting"

WINDOW_FRAMES = 121  # F % 8 == 1
OVERLAP_FRAMES = 17  # (overlap - 1) % 8 == 0, i.e. 2 whole latent frames of carryover.
                    # 16 fails the validator below -- every real run has passed 17 explicitly.
FLUSH_EVERY_FRAMES = 24 * 30  # per user: update the on-disk video every 720 finalized frames


def refinement_schedule(k_step: str) -> list[float]:
    dist_vals = [float(v) for v in DISTILLED_SIGMA_VALUES]
    schedules = {"k1": dist_vals[-2:], "k2": dist_vals[-3:], "k3": dist_vals[-4:], "k4": dist_vals[-5:], "k8": dist_vals}
    if k_step not in schedules:
        raise ValueError(f"Unknown k_step {k_step!r}; expected one of {list(schedules)}")
    return schedules[k_step]


def plan_windows(
    total_frames: int,
    window: int = WINDOW_FRAMES,
    overlap: int = OVERLAP_FRAMES,
    scale_factors=None,
) -> list[tuple[int, int]]:
    """Fixed-stride window starts. Delegates to ``refine_core.WindowGeometry`` so the
    gates in ``scripts/prune/`` plan the identical grid (see refine_core's docstring)."""
    return refine_core.WindowGeometry(
        window_frames=window,
        overlap_frames=overlap,
        scale_factors=scale_factors if scale_factors is not None else VIDEO_SCALE_FACTORS,
    ).plan(total_frames)


@dataclass
class StageTimer:
    label: str
    device: torch.device
    t0: float = 0.0

    def __enter__(self) -> "StageTimer":
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        torch.cuda.synchronize(self.device)
        self.elapsed_s = time.perf_counter() - self.t0
        self.peak_alloc_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
        self.peak_reserved_gb = torch.cuda.max_memory_reserved(self.device) / 1e9


def read_pixel_window(vr: decord.VideoReader, start: int, end: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (norm_video [1,C,F,H,W] in [-1,1] on device, pixel_video [F,H,W,C] in [0,1] on cpu)."""
    return refine_core.read_pixel_window(vr, start, end, device, DTYPE)


def rebuild_full_concat(segments_dir: Path, out_path: Path) -> None:
    seg_files = sorted(segments_dir.glob("seg_*.mp4"))
    if not seg_files:
        return
    list_path = segments_dir / "_concat_list.txt"
    list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_files))
    tmp_out = out_path.with_suffix(".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(tmp_out)],
        check=True,
        capture_output=True,
    )
    tmp_out.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k-step", default="k2")
    ap.add_argument("--gpu-id", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--model", default="2.5", choices=model_registry.SUPPORTED_MODELS,
        help="Generation to refine with (see scripts/prune/model_registry.py). Per-component "
        "flags below always override this generation's default path for that component.",
    )
    ap.add_argument("--sampler", default="euler", choices=model_registry.SAMPLER_CHOICES)
    ap.add_argument(
        "--video-only", dest="video_only", action="store_true", default=True,
        help="Build the transformer with LTXVideoOnlyModelConfigurator (skips the audio branch "
        "entirely; documented + gated lossless by scripts/prune/video_only_check.py). Default on.",
    )
    ap.add_argument("--no-video-only", dest="video_only", action="store_false")
    ap.add_argument("--transformer-path", type=Path, default=None, help="Override this --model's transformer checkpoint.")
    ap.add_argument(
        "--text-encoder-path", type=Path, default=None,
        help="Override this --model's text encoder (gemma root dir for 2.3, gemma4 file for 2.5).",
    )
    ap.add_argument("--video-vae-path", type=Path, default=None, help="Override this --model's video VAE checkpoint.")
    ap.add_argument("--window-frames", type=int, default=WINDOW_FRAMES)
    ap.add_argument("--overlap-frames", type=int, default=OVERLAP_FRAMES)
    ap.add_argument("--flush-every-frames", type=int, default=FLUSH_EVERY_FRAMES)
    ap.add_argument("--max-windows", type=int, default=None, help="Process at most this many windows (testing).")
    ap.add_argument(
        "--max-total-frames",
        type=int,
        default=None,
        help="Only plan windows within the first N native frames of the source video (testing on a slice).",
    )
    ap.add_argument(
        "--batch-windows",
        type=int,
        default=20,
        help="Process this many windows per 'resident transformer' chunk: the ~44GB transformer is "
        "built once per chunk (not per window) via DiffusionStage's private _transformer_ctx, "
        "amortizing the dominant per-window cost (checkpoint build, not denoise compute -- see "
        "profile.json). Windows are still refined one at a time (DiffusionStage hardcodes batch=1), "
        "so this changes nothing about the result, only how often the transformer reloads. Larger "
        "values amortize more but bound resume granularity: an interruption mid-chunk redoes that "
        "chunk's un-cached windows (cheap once reload is eliminated). Set to 1 for the old fully "
        "per-window rebuild behavior.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model = preflight.check(
        args.model,
        sampler=args.sampler,
        gpu_id=args.gpu_id,
        transformer_path=args.transformer_path,
        text_encoder_path=args.text_encoder_path,
        video_vae_path=args.video_vae_path,
    )
    if model.stepper_kind == "ancestral":
        # EulerAncestralDiffusionStep needs a per-step noise draw (eta=1.0 renoises after every
        # step); this script's manual _step_state() loop below doesn't supply one. Plan §4
        # decision 1 defaults the refiner to Euler on both generations and defers the ancestral
        # A/B to Phase 1 (needs the noise-injecting loop, mirroring
        # ltx_pipelines.utils.samplers._ancestral_euler_denoising_loop).
        raise SystemExit(
            f"--model {args.model} --sampler {args.sampler} resolved to the ancestral stepper, which "
            "isn't wired into this script's step loop yet (Phase 1 item per plan §4 decision 1). "
            "Pass --sampler euler explicitly."
        )
    # The window grid is checked against the VAE's PROBED temporal scale factor rather than a
    # literal 8 (plan §4 decision 2), which is why this runs after the model resolve. The rule
    # itself is unchanged: each window's own latent frame 0 is a single-pixel causal keyframe
    # rather than a full temporal block, so the carried-over overlap needs the same F%t==1 grid
    # for it to land on whole *regular* latent frames of the next window. See run_batch().
    geometry.check_window_rules(args.window_frames, args.overlap_frames, model.scale_factors)
    window_geometry = refine_core.WindowGeometry(
        window_frames=args.window_frames, overlap_frames=args.overlap_frames, scale_factors=model.scale_factors
    )

    configurator = LTXVideoOnlyModelConfigurator if args.video_only else LTXModelConfigurator
    logger.info(
        f"model={model.key} (version {model.version}) sampler={model.stepper_kind} "
        f"video_only={args.video_only} scale_factors={tuple(model.scale_factors)} "
        f"(from {model.scale_factors_source})"
    )

    device = torch.device(f"cuda:{args.gpu_id}")
    out_dir = args.output_dir
    segments_dir = out_dir / "segments"
    cache_dir = out_dir / "window_cache"
    latent_cache_dir = out_dir / "latent_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    overlap_latent_frames = window_geometry.context_latent_frames

    vr = decord.VideoReader(str(args.video))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    if args.max_total_frames is not None:
        total_frames = min(total_frames, args.max_total_frames)
    windows = window_geometry.plan(total_frames)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    logger.info(f"{args.video.name}: {total_frames} native frames -> {len(windows)} windows of {args.window_frames} "
                f"(overlap {args.overlap_frames}, stride {args.window_frames - args.overlap_frames})")

    (out_dir / "window_plan.json").write_text(
        json.dumps(
            {"total_frames": total_frames, "fps": fps, "geometry": window_geometry.as_dict(), "windows": windows},
            indent=2,
        )
    )

    if args.dry_run:
        print(f"Dry run: {len(windows)} windows planned, written to {out_dir / 'window_plan.json'}")
        return 0

    sigmas_list = refinement_schedule(args.k_step)
    sigmas = torch.tensor(sigmas_list, dtype=torch.float32, device=device)

    profile_path = out_dir / "profile.json"
    profile: list[dict] = json.loads(profile_path.read_text()) if profile_path.exists() else []
    done_indices = {p["window_index"] for p in profile}

    # Encode prompt once -- identical text conditioning for every window.
    with torch.no_grad():
        prompt_encoder = PromptEncoder(model.paths, DTYPE, device)
        (ctx,) = prompt_encoder([args.prompt])
        video_context = ctx.video_encoding
        del prompt_encoder
        gc.collect()
        torch.cuda.empty_cache()

    diffusion_stage = DiffusionStage.from_checkpoint(
        model.paths.transformer(), DTYPE, device, model_configurator=configurator,
        scale_factors=model.scale_factors,
    )

    # Geometry is identical for every window (fixed window_frames, one source video), so the
    # tools DiffusionStage.__call__ would normally rebuild per-call can be built once and reused.
    probe_nv, _ = read_pixel_window(vr, windows[0][0], windows[0][1], device)
    win_h, win_w = int(probe_nv.shape[-2]), int(probe_nv.shape[-1])
    del probe_nv
    # fps is the clip's own, never a constant: VideoLatentTools divides the temporal
    # position axis by it, so it is part of RoPE. See refine_core.build_tools.
    video_tools = refine_core.tools_for_window(window_geometry, win_h, win_w, fps)

    pending_tail: torch.Tensor | None = None  # decoded pixels [ov, H, W, C] not yet finalized
    finalized_since_flush: list[torch.Tensor] = []
    finalized_frame_count_since_flush = 0
    finalized_start_frame = 0  # first native-frame index not yet flushed

    # Window i's last `overlap_latent_frames` refined latent frames get frozen into window
    # i+1's first `overlap_latent_frames` slots (VideoConditionByLatentIndex, strength=1.0),
    # instead of window i+1 VAE-encoding that span fresh from source pixels. This is what
    # actually fixes the appearance-shift-at-boundary artifact: the old pixel cross-fade only
    # blended two INDEPENDENTLY refined windows after the fact, so any real generative drift
    # between them (different RoPE position + different noise realization for the "same"
    # source frames -- each window renoises from its own frame 0) still showed as a visible
    # pop once the short cross-fade ran out. Freezing the overlap means window i+1 continues
    # from window i's literal output, not a fresh guess at it. Kept in memory for the whole
    # run (latents are tiny, ~1-4MB/window) and also persisted to disk so a resumed run can
    # recover the carryover from a window that was cached (pixels only) by an earlier session.
    latent_by_window: dict[int, torch.Tensor] = {}

    def load_cached_or_none(idx: int) -> torch.Tensor | None:
        p = cache_dir / f"win_{idx:04d}.pt"
        if p.exists():
            return torch.load(p, map_location="cpu").float() / 255.0
        return None

    def save_cache(idx: int, decoded: torch.Tensor) -> None:
        torch.save((decoded.clamp(0, 1) * 255.0).to(torch.uint8), cache_dir / f"win_{idx:04d}.pt")

    def load_latent_cache_or_none(idx: int) -> torch.Tensor | None:
        p = latent_cache_dir / f"win_{idx:04d}_latent.pt"
        if p.exists():
            return torch.load(p, map_location="cpu")
        return None

    def save_latent_cache(idx: int, latent: torch.Tensor) -> None:
        torch.save(latent.detach().to(torch.bfloat16).cpu(), latent_cache_dir / f"win_{idx:04d}_latent.pt")

    def flush(final: bool = False) -> None:
        nonlocal finalized_since_flush, finalized_frame_count_since_flush, finalized_start_frame
        if not finalized_since_flush:
            return
        chunk = torch.cat(finalized_since_flush, dim=0)
        seg_end = finalized_start_frame + chunk.shape[0]
        seg_path = segments_dir / f"seg_{finalized_start_frame:06d}_{seg_end:06d}.mp4"
        save_video(rearrange(chunk, "f h w c -> f c h w"), seg_path, fps=fps, video_format="FCHW")
        rebuild_full_concat(segments_dir, out_dir / "decode_full.mp4")
        logger.info(f"Flushed frames [{finalized_start_frame}:{seg_end}) -> {seg_path.name}, rebuilt decode_full.mp4")
        finalized_start_frame = seg_end
        finalized_since_flush = []
        finalized_frame_count_since_flush = 0

    def run_batch(todo_idxs: list[int]) -> dict[int, torch.Tensor]:
        """Process `todo_idxs` windows in three memory-separated phases so the transformer
        is built exactly ONCE for the whole chunk instead of once per window:
          A. per-window VAE encode (transformer not yet loaded; unchanged from before)
          B. one _transformer_ctx() covering every window's denoise loop (the actual fix --
             DiffusionStage.__call__ hardcodes batch=1 internally, so windows can't be
             merged into a single forward pass; the saving instead comes from not tearing
             the ~44GB transformer down and rebuilding it between windows)
          C. per-window VAE decode (transformer freed again; unchanged from before)
        Phases can't overlap in time: transformer (~44GB) + VAE activations together would
        exceed the 49GB card, which is exactly why DiffusionStage frees between calls.
        Latents are tiny (~4MB/window at 121 frames) so holding a whole chunk's worth
        between phases is negligible; decoded pixels are NOT held in bulk (~1.5GB/window),
        they're cached+blended immediately per window in phase C as before.
        """
        per_win_profile: dict[int, dict] = {}
        l_enc_map: dict[int, torch.Tensor] = {}

        # Phase A: encode. The VAE encoder is built ONCE for the whole chunk (like the
        # transformer in phase B) so encoder-build time is no longer interleaved into every
        # window's "encode" number -- it's its own line in the profile.
        with torch.no_grad():
            with StageTimer("encoder_build", device) as t_enc_build:
                image_conditioner = ImageConditioner(model.paths.video_vae(), DTYPE, device)
                encoder_ctx = gpu_model(image_conditioner._build_encoder())
                encoder = encoder_ctx.__enter__()
            chunk_encoder_build_s = t_enc_build.elapsed_s
            chunk_encoder_build_peak_gb = t_enc_build.peak_alloc_gb
            try:
                for i in todo_idxs:
                    s, e = windows[i]
                    with StageTimer("read", device) as t_read:
                        nv, _ = read_pixel_window(vr, s, e, device)
                    per_win_profile[i] = {"window_index": i, "start": s, "end": e, "read_s": t_read.elapsed_s}
                    with StageTimer("encode_compute", device) as t_enc:
                        l_enc_map[i] = encoder.tiled_encode(nv, None)
                    per_win_profile[i]["encoder_build_s"] = chunk_encoder_build_s
                    per_win_profile[i]["encoder_build_peak_alloc_gb"] = chunk_encoder_build_peak_gb
                    per_win_profile[i]["encode_compute_s"] = t_enc.elapsed_s
                    per_win_profile[i]["encode_s"] = t_enc.elapsed_s  # kept for older summaries
                    per_win_profile[i]["encode_peak_alloc_gb"] = t_enc.peak_alloc_gb
            finally:
                encoder_ctx.__exit__(None, None, None)

        # Phase B: refine, transformer built once for the whole chunk.
        # The transformer build itself (entering _transformer_ctx) is timed separately from
        # the per-window work, and each window's denoise loop is unrolled by hand (instead of
        # calling euler_denoising_loop) so every individual diffusion step gets its own timing
        # -- split further into the transformer forward (denoiser call) vs. the Euler update
        # (stepper.step + post_process_latent, both inside _step_state).
        refined_latent_map: dict[int, torch.Tensor] = {}
        stepper = EulerDiffusionStep()
        denoiser = SimpleDenoiser(video_context, None)
        with torch.no_grad():
            with StageTimer("transformer_build", device) as t_build:
                transformer_ctx = diffusion_stage._transformer_ctx(video_tools=video_tools)
                transformer = transformer_ctx.__enter__()
            chunk_transformer_build_s = t_build.elapsed_s
            chunk_transformer_build_peak_gb = t_build.peak_alloc_gb
            try:
                wrapped = BatchSplitAdapter(transformer, max_batch_size=1)
                for i in todo_idxs:
                    per_win_profile[i]["transformer_build_s"] = chunk_transformer_build_s
                    per_win_profile[i]["transformer_build_peak_alloc_gb"] = chunk_transformer_build_peak_gb
                    per_win_profile[i]["chunk_size"] = len(todo_idxs)

                    carry = None
                    if i > 0:
                        prev_latent = latent_by_window.get(i - 1)
                        if prev_latent is None:
                            prev_latent = load_latent_cache_or_none(i - 1)
                        if prev_latent is None:
                            raise RuntimeError(
                                f"Window {i-1}'s refined latent is unavailable (neither in memory nor in "
                                f"{latent_cache_dir}) -- can't build window {i}'s frozen-overlap conditioning. "
                                "Windows must be processed in order from a consistent cache."
                            )
                        # refine_core.make_window_state injects this at latent_idx=1, NOT 0 -- see
                        # refine_core.CARRYOVER_LATENT_IDX for why the keyframe slot stays fresh.
                        carry = refine_core.carry_from(
                            prev_latent.to(device=device, dtype=DTYPE), window_geometry
                        )
                    per_win_profile[i]["carryover_latent_frames"] = overlap_latent_frames if i > 0 else 0

                    with StageTimer("build_state", device) as t_bs:
                        video_state = refine_core.make_window_state(
                            l_enc_map[i], carry, sigmas[0].item(), video_tools, args.seed, device, DTYPE
                        )
                    per_win_profile[i]["build_state_s"] = t_bs.elapsed_s

                    denoiser_call_s: list[float] = []
                    step_update_s: list[float] = []
                    for step_idx in range(len(sigmas) - 1):
                        with StageTimer(f"denoise_fwd_{step_idx}", device) as t_fwd:
                            video_result, _ = denoiser(wrapped, video_state, None, sigmas, step_idx)
                        denoiser_call_s.append(t_fwd.elapsed_s)
                        with StageTimer(f"step_update_{step_idx}", device) as t_step:
                            denoised_video = video_result.denoised if video_result is not None else None
                            video_state = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
                        step_update_s.append(t_step.elapsed_s)
                    per_win_profile[i]["denoiser_call_s_per_step"] = denoiser_call_s
                    per_win_profile[i]["step_update_s_per_step"] = step_update_s
                    per_win_profile[i]["refine_s"] = sum(denoiser_call_s) + sum(step_update_s) + t_bs.elapsed_s
                    per_win_profile[i]["refine_peak_alloc_gb"] = max(t_fwd.peak_alloc_gb, t_step.peak_alloc_gb)

                    with StageTimer("postprocess", device) as t_post:
                        refined = refine_core.finalize(video_state, video_tools)
                    per_win_profile[i]["postprocess_s"] = t_post.elapsed_s
                    refined_latent_map[i] = refined
                    latent_by_window[i] = refined.detach().cpu()
                    save_latent_cache(i, refined)
            finally:
                transformer_ctx.__exit__(None, None, None)

        # Phase C: decode. Same treatment -- the decoder is built ONCE for the whole chunk
        # (bypassing VideoDecoder.__call__'s own build-per-call, via its private
        # _decoder_builder) instead of once per window.
        decoded_map: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            with StageTimer("decoder_build", device) as t_dec_build:
                video_decoder = VideoDecoder(model.paths.video_vae(), DTYPE, device)
                decoder_ctx = gpu_model(video_decoder._decoder_builder.build(device=device, dtype=DTYPE).eval())
                decoder = decoder_ctx.__enter__()
            chunk_decoder_build_s = t_dec_build.elapsed_s
            chunk_decoder_build_peak_gb = t_dec_build.peak_alloc_gb
            try:
                for i in todo_idxs:
                    with StageTimer("decode_compute", device) as t_dec:
                        decoded_map[i] = torch.cat(list(decoder.decode_video(refined_latent_map[i], None, None)), dim=0).cpu().float()
                    per_win_profile[i]["decoder_build_s"] = chunk_decoder_build_s
                    per_win_profile[i]["decoder_build_peak_alloc_gb"] = chunk_decoder_build_peak_gb
                    per_win_profile[i]["decode_compute_s"] = t_dec.elapsed_s
                    per_win_profile[i]["decode_s"] = t_dec.elapsed_s  # kept for older summaries
                    per_win_profile[i]["decode_peak_alloc_gb"] = t_dec.peak_alloc_gb
            finally:
                decoder_ctx.__exit__(None, None, None)

        for i in todo_idxs:
            save_cache(i, decoded_map[i])
            profile.append(per_win_profile[i])
            logger.info(
                f"[window {i+1}/{len(windows)}] chunk={len(todo_idxs)} frames {windows[i]} "
                f"read={per_win_profile[i]['read_s']:.1f}s encode={per_win_profile[i]['encode_s']:.1f}s "
                f"refine={per_win_profile[i]['refine_s']:.1f}s decode={per_win_profile[i]['decode_s']:.1f}s"
            )
        profile_path.write_text(json.dumps(profile, indent=2))
        return decoded_map

    batch_size = max(1, args.batch_windows)
    idx = 0
    while idx < len(windows):
        batch_idxs = list(range(idx, min(idx + batch_size, len(windows))))
        decoded_map: dict[int, torch.Tensor] = {}
        todo: list[int] = []
        for i in batch_idxs:
            if i in done_indices:
                cached = load_cached_or_none(i)
                if cached is not None:
                    decoded_map[i] = cached
                    continue
            todo.append(i)
        if todo:
            decoded_map.update(run_batch(todo))

        for i in batch_idxs:
            decoded = decoded_map[i]
            start, end = windows[i]

            # --- blend this window's head into the previous window's pending tail ---
            overlap_with_prev = 0
            if i > 0:
                prev_end = windows[i - 1][1]
                overlap_with_prev = max(0, prev_end - start)

            if overlap_with_prev > 0 and pending_tail is not None:
                ov = min(overlap_with_prev, pending_tail.shape[0], decoded.shape[0])
                w = torch.linspace(0.0, 1.0, ov).view(ov, 1, 1, 1)
                blended = (1.0 - w) * pending_tail[:ov] + w * decoded[:ov]
                finalized_since_flush.append(blended)
                finalized_frame_count_since_flush += ov
            else:
                ov = 0

            # overlap this window shares with the NEXT window stays pending (not yet finalized)
            overlap_with_next = 0
            if i < len(windows) - 1:
                next_start = windows[i + 1][0]
                overlap_with_next = max(0, end - next_start)

            body_end = decoded.shape[0] - overlap_with_next
            if body_end > ov:
                finalized_since_flush.append(decoded[ov:body_end])
                finalized_frame_count_since_flush += body_end - ov

            pending_tail = decoded[body_end:] if overlap_with_next > 0 else None

            if finalized_frame_count_since_flush >= args.flush_every_frames or i == len(windows) - 1:
                flush(final=(i == len(windows) - 1))

        idx += batch_size

    flush(final=True)
    logger.info(f"Done. Full stitched video at {out_dir / 'decode_full.mp4'} ({finalized_start_frame} frames finalized).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
