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
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-pipelines" / "src"))

from ltx_core.batch_split import BatchSplitAdapter  # noqa: E402
from ltx_core.components.diffusion_steps import EulerDiffusionStep  # noqa: E402
from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex  # noqa: E402
from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: E402
from ltx_core.tools import VideoLatentTools  # noqa: E402
from ltx_core.types import VideoLatentShape, VideoPixelShape  # noqa: E402
from ltx_pipelines.utils.blocks import (  # noqa: E402
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    _build_state,  # noqa: PLC2701 -- deliberately reusing DiffusionStage.__call__'s own primitive
                   # to hold the transformer resident across windows instead of rebuilding it
                   # per window; see the --resident-transformer docstring below for the tradeoff.
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES  # noqa: E402
from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: E402
from ltx_pipelines.utils.gpu_model import gpu_model  # noqa: E402
from ltx_pipelines.utils.model_paths import ModelPaths  # noqa: E402
from ltx_pipelines.utils.samplers import _step_state  # noqa: E402, PLC2701 -- see _build_state note above
from ltx_pipelines.utils.types import ModalitySpec  # noqa: E402
from ltx_trainer.video_utils import save_video  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vae_refine_sliding_window")

DTYPE = torch.bfloat16
DEFAULT_VAE_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_GEMMA_ROOT = WORKSPACE_ROOT / "checkpoints" / "google" / "gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_PROMPT = "a high quality, sharp, detailed video with fine texture and natural lighting"

WINDOW_FRAMES = 121  # F % 8 == 1
OVERLAP_FRAMES = 16
FLUSH_EVERY_FRAMES = 24 * 30  # per user: update the on-disk video every 720 finalized frames


def refinement_schedule(k_step: str) -> list[float]:
    dist_vals = [float(v) for v in DISTILLED_SIGMA_VALUES]
    schedules = {"k1": dist_vals[-2:], "k2": dist_vals[-3:], "k3": dist_vals[-4:], "k4": dist_vals[-5:], "k8": dist_vals}
    if k_step not in schedules:
        raise ValueError(f"Unknown k_step {k_step!r}; expected one of {list(schedules)}")
    return schedules[k_step]


def plan_windows(total_frames: int, window: int = WINDOW_FRAMES, overlap: int = OVERLAP_FRAMES) -> list[tuple[int, int]]:
    if total_frames < window:
        raise ValueError(f"Video has {total_frames} frames, shorter than one window ({window})")
    if 2 * overlap >= window:
        raise ValueError(
            f"overlap_frames ({overlap}) must be < window_frames/2 ({window/2}): stride = window - overlap "
            f"= {window - overlap} would be <= overlap, so a frame could fall in 3+ windows at once. The "
            "blend/flush bookkeeping (and the latent-carryover chain) only track immediate-neighbor "
            "overlap -- pick a bigger window or a smaller overlap."
        )
    stride = window - overlap

    # Fixed stride, no clipped/adjusted tail: every pairwise overlap is then EXACTLY `overlap`,
    # which the latent-carryover conditioning depends on (it freezes a fixed-size latent-frame
    # slice computed from the nominal `overlap`, at a fixed destination latent index -- either a
    # clipped tail window or evenly-redistributed spacing, both tried earlier, shift the ACTUAL
    # per-pair overlap by a frame here and there, which is enough to misalign the carried-over
    # latent frame against the destination window's own grid and visibly hurt boundary
    # continuity). The tradeoff: up to `stride - 1` trailing frames of the source video may not
    # be covered by any window and are silently dropped rather than force-fitting a tail window.
    starts = list(range(0, total_frames - window + 1, stride))
    return [(s, s + window) for s in starts]


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
    frames = vr.get_batch(range(start, end))  # [F,H,W,C] uint8-like
    f, h, w, c = frames.shape
    if h % 32 != 0:
        crop_h = (h // 32) * 32
        top = (h - crop_h) // 2
        frames = frames[:, top : top + crop_h, :, :]
    if w % 32 != 0:
        crop_w = (w // 32) * 32
        left = (w - crop_w) // 2
        frames = frames[:, :, left : left + crop_w, :]
    pixel_video = (frames.float() / 255.0).clamp(0.0, 1.0)
    norm_video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(dtype=DTYPE, device=device)
    norm_video = (norm_video / 127.5) - 1.0
    return norm_video, pixel_video


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
    ap.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    ap.add_argument("--gemma-root", type=Path, default=DEFAULT_GEMMA_ROOT)
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

    if (args.overlap_frames - 1) % 8 != 0:
        raise SystemExit(
            f"--overlap-frames must satisfy (overlap_frames - 1) % 8 == 0 (got {args.overlap_frames}), the "
            "same F%8==1 grid every window size already follows. Reason: each window's OWN latent frame 0 "
            "is a special single-pixel causal keyframe (not an 8-pixel block like every later latent "
            "frame), so the overlap region -- if it were its own standalone clip -- needs that same grid "
            "for the carryover to land on whole latent frames of the *next* window's REGULAR (non-frame-0) "
            "latent frames. See the latent-carryover note in run_batch()."
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
    overlap_latent_frames = (args.overlap_frames - 1) // 8

    vr = decord.VideoReader(str(args.video))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    if args.max_total_frames is not None:
        total_frames = min(total_frames, args.max_total_frames)
    windows = plan_windows(total_frames, args.window_frames, args.overlap_frames)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    logger.info(f"{args.video.name}: {total_frames} native frames -> {len(windows)} windows of {args.window_frames} "
                f"(overlap {args.overlap_frames}, stride {args.window_frames - args.overlap_frames})")

    (out_dir / "window_plan.json").write_text(
        json.dumps({"total_frames": total_frames, "fps": fps, "windows": windows}, indent=2)
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
        prompt_encoder = PromptEncoder(ModelPaths.from_monolith(str(args.vae_checkpoint), str(args.gemma_root)), DTYPE, device)
        (ctx,) = prompt_encoder([args.prompt])
        video_context = ctx.video_encoding
        del prompt_encoder
        gc.collect()
        torch.cuda.empty_cache()

    diffusion_stage = DiffusionStage.from_checkpoint(str(args.vae_checkpoint), DTYPE, device)

    # Geometry is identical for every window (fixed window_frames, one source video), so the
    # tools DiffusionStage.__call__ would normally rebuild per-call can be built once and reused.
    probe_nv, _ = read_pixel_window(vr, windows[0][0], windows[0][1], device)
    win_h, win_w = int(probe_nv.shape[-2]), int(probe_nv.shape[-1])
    del probe_nv
    pixel_shape = VideoPixelShape(batch=1, frames=args.window_frames, height=win_h, width=win_w, fps=fps)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=diffusion_stage.video_scale_factors)
    video_tools = VideoLatentTools(
        VideoLatentPatchifier(patch_size=1), v_shape, fps, scale_factors=diffusion_stage.video_scale_factors
    )

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
                image_conditioner = ImageConditioner(str(args.vae_checkpoint), DTYPE, device)
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
                    noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(args.seed))
                    per_win_profile[i]["transformer_build_s"] = chunk_transformer_build_s
                    per_win_profile[i]["transformer_build_peak_alloc_gb"] = chunk_transformer_build_peak_gb
                    per_win_profile[i]["chunk_size"] = len(todo_idxs)

                    conditionings = []
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
                        # Inject at latent_idx=1, NOT 0: this window's own latent frame 0 is the special
                        # single-pixel causal keyframe (native pixel `next_start` alone), which has no
                        # counterpart in window i-1's tail (whose last regular latent frame covers 8 DIFFERENT,
                        # earlier pixels). Frame 0 is left unconditioned/fresh; only the REGULAR latent frames
                        # from index 1 onward -- which line up pixel-for-pixel with window i-1's tail -- get
                        # frozen. The resulting 1-pixel-frame gap at the very start of the overlap is small
                        # enough for the existing pixel cross-fade to smooth over.
                        carry = prev_latent[:, :, -overlap_latent_frames:, :, :].to(device=device, dtype=DTYPE)
                        conditionings = [VideoConditionByLatentIndex(latent=carry, strength=1.0, latent_idx=1)]
                    per_win_profile[i]["carryover_latent_frames"] = overlap_latent_frames if i > 0 else 0

                    with StageTimer("build_state", device) as t_bs:
                        video_state = _build_state(
                            ModalitySpec(
                                context=video_context,
                                conditionings=conditionings,
                                noise_scale=sigmas[0].item(),
                                initial_latent=l_enc_map[i],
                            ),
                            video_tools,
                            noiser,
                            DTYPE,
                            device,
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
                        video_state = video_tools.clear_conditioning(video_state)
                        video_state = video_tools.unpatchify(video_state)
                    per_win_profile[i]["postprocess_s"] = t_post.elapsed_s
                    refined_latent_map[i] = video_state.latent
                    latent_by_window[i] = video_state.latent.detach().cpu()
                    save_latent_cache(i, video_state.latent)
            finally:
                transformer_ctx.__exit__(None, None, None)

        # Phase C: decode. Same treatment -- the decoder is built ONCE for the whole chunk
        # (bypassing VideoDecoder.__call__'s own build-per-call, via its private
        # _decoder_builder) instead of once per window.
        decoded_map: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            with StageTimer("decoder_build", device) as t_dec_build:
                video_decoder = VideoDecoder(str(args.vae_checkpoint), DTYPE, device)
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
