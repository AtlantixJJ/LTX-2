"""Phase 0 gate: LTXVideoOnlyModelConfigurator must match the audio-video build
within bf16 noise, on real clips (plan §5 item 2).

``ltx_trainer/model_loader.py`` documents dropping the audio branch as lossless
for a video-only call (the audio path is skipped by ``run_ax``/``run_a2v``/
``run_v2a`` guards whenever ``audio=None``), but the refiner's whole memory
headroom argument depends on that holding for *this* checkpoint and *this* call
shape, not just being true in general.

Clips are chosen from **distinct subjects**: the corpus contains several framings
(``_crop`` / ``_original``) and motions of the same capture, so taking the first N
directory names alphabetically can compare the model against itself on what is
essentially one video. A configurator difference that only shows on some content
would slip through that.

Both configurators are built **once** and run over every clip, rather than rebuilt
per clip -- ~20 s per build, and the comparison is per clip, not per build.

    conda run -n ltx python -m scripts.prune.video_only_check --model 2.5 --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decord
import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer import LTXModelConfigurator, LTXVideoOnlyModelConfigurator
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, _build_state
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.types import ModalitySpec

from scripts.prune import model_registry, preflight, prompt_cache, provenance, refine_task
from scripts.prune.model_registry import RefinerModel, WORKSPACE_ROOT

decord.bridge.set_bridge("torch")

DTYPE = torch.bfloat16
CORPUS_DIR = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"
WINDOW_FRAMES = 121  # F % 8 == 1, matches vae_refine_sliding_window.py's window grid
TOLERANCE = 1e-2  # bf16 noise floor (plan §5), not exact zero


def _read_pixel_window(path: Path, device: torch.device, frames: int = WINDOW_FRAMES) -> torch.Tensor:
    """Returns norm_video [1,C,F,H,W] in [-1,1] on device. Mirrors
    vae_refine_sliding_window.read_pixel_window's cropping (reimplemented, not
    imported -- that script is a run script, not a library)."""
    vr = decord.VideoReader(str(path))
    if len(vr) < frames:
        raise ValueError(f"{path}: only {len(vr)} frames, need >= {frames}")
    raw = vr.get_batch(range(frames))  # [F,H,W,C]
    _, h, w, _ = raw.shape
    crop_h, crop_w = (h // 32) * 32, (w // 32) * 32
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    raw = raw[:, top : top + crop_h, left : left + crop_w, :]
    norm = raw.permute(3, 0, 1, 2).unsqueeze(0).to(dtype=DTYPE, device=device)
    return (norm / 127.5) - 1.0


def _subject_of(clip_dir: Path) -> str:
    """``2K2K_00052_0__man_dance_2_crop`` -> ``2K2K_00052_0`` (the captured subject)."""
    return clip_dir.name.split("__", 1)[0]


def _pick_clips(num_clips: int) -> list[Path]:
    """First usable clip per distinct subject, up to *num_clips*."""
    by_subject: dict[str, Path] = {}
    for p in sorted(CORPUS_DIR.glob("*/source.mp4")):
        subject = _subject_of(p.parent)
        if subject in by_subject:
            continue
        try:
            if len(decord.VideoReader(str(p))) >= WINDOW_FRAMES:
                by_subject[subject] = p
        except Exception:
            continue
        if len(by_subject) >= num_clips:
            break
    if len(by_subject) < num_clips:
        raise SystemExit(
            f"Only found {len(by_subject)} subjects with a >= {WINDOW_FRAMES}-frame clip under "
            f"{CORPUS_DIR}, need {num_clips}."
        )
    return list(by_subject.values())


def _encode(model: RefinerModel, clips: list[Path], device: torch.device) -> list[dict]:
    """VAE-encode one window per clip; the encoder is built once and freed."""
    out = []
    with torch.no_grad():
        image_conditioner = ImageConditioner(model.paths.video_vae(), DTYPE, device)
        with gpu_model(image_conditioner._build_encoder()) as encoder:
            for clip in clips:
                norm_video = _read_pixel_window(clip, device)
                _, _, frames, height, width = norm_video.shape
                out.append(
                    {
                        "clip": clip.parent.name,
                        "subject": _subject_of(clip.parent),
                        "latent": encoder.tiled_encode(norm_video, None),
                        "frames": frames,
                        "height": height,
                        "width": width,
                    }
                )
                del norm_video
    torch.cuda.empty_cache()
    return out


def _run_all(
    model: RefinerModel,
    configurator: type,
    encoded: list[dict],
    video_context: torch.Tensor,
    sigma0: float,
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], dict]:
    """One build of *configurator*, one k2-tail first forward per encoded clip.

    Also returns the build's memory profile: the *resident* weight footprint is
    the number §5 item 2's "buys the memory headroom" claim actually rests on,
    and it is not the same as the transient build peak.
    """
    stage = DiffusionStage.from_checkpoint(
        model.paths.transformer(), DTYPE, device, model_configurator=configurator, scale_factors=model.scale_factors
    )
    outputs = []
    torch.cuda.reset_peak_memory_stats(device)
    stats: dict = {}
    with torch.no_grad():
        pixel_shape = VideoPixelShape(
            batch=1, frames=encoded[0]["frames"], height=encoded[0]["height"], width=encoded[0]["width"], fps=24.0
        )
        v_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=model.caps.latent_channels, scale_factors=model.scale_factors
        )
        video_tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1), v_shape, 24.0, scale_factors=model.scale_factors
        )
        denoiser = SimpleDenoiser(video_context, None)
        sigmas = torch.tensor([sigma0, 0.0], dtype=torch.float32, device=device)
        with stage._transformer_ctx(video_tools=video_tools) as transformer:
            stats["resident_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
            stats["build_peak_alloc_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            for item in encoded:
                noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed))
                state = _build_state(
                    ModalitySpec(
                        context=video_context, conditionings=[], noise_scale=sigma0, initial_latent=item["latent"]
                    ),
                    video_tools,
                    noiser,
                    DTYPE,
                    device,
                )
                result, _ = denoiser(transformer, state, None, sigmas, 0)
                outputs.append(result.denoised.detach().clone())
    del stage
    torch.cuda.empty_cache()
    return outputs, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--num-clips", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    model = preflight.check(args.model, gpu_id=args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    video_context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    sigma0 = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)[0]

    clips = _pick_clips(args.num_clips)
    print(f"[video_only_check] subjects: {[_subject_of(c.parent) for c in clips]}", flush=True)
    encoded = _encode(model, clips, device)

    print("[video_only_check] audio-video build ...", flush=True)
    av_out, av_mem = _run_all(model, LTXModelConfigurator, encoded, video_context, sigma0, device, args.seed)
    print("[video_only_check] video-only build ...", flush=True)
    vo_out, vo_mem = _run_all(model, LTXVideoOnlyModelConfigurator, encoded, video_context, sigma0, device, args.seed)
    print(
        f"  resident weights: AV {av_mem['resident_alloc_gb']:.1f} GB vs video-only "
        f"{vo_mem['resident_alloc_gb']:.1f} GB "
        f"(saved {av_mem['resident_alloc_gb'] - vo_mem['resident_alloc_gb']:.1f} GB); "
        f"build peak {av_mem['build_peak_alloc_gb']:.1f} / {vo_mem['build_peak_alloc_gb']:.1f} GB"
    )

    rows, all_pass = [], True
    for item, a, b in zip(encoded, av_out, vo_out):
        max_abs_diff = (a.float() - b.float()).abs().max().item()
        passed = max_abs_diff < args.tolerance
        all_pass &= passed
        rows.append(
            {
                "clip": item["clip"],
                "subject": item["subject"],
                "max_abs_diff": max_abs_diff,
                "bit_exact": bool(torch.equal(a, b)),
                "pass": passed,
            }
        )
        print(f"  {item['clip']}: max_abs_diff={max_abs_diff:.4g} ({'PASS' if passed else 'FAIL'})")

    out_dir = model_registry.WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "video_only_check.json"
    out_path.write_text(
        json.dumps(
            {
                "provenance": provenance.stamp(model, device, script="video_only_check"),
                "tolerance": args.tolerance,
                "sigma0": sigma0,
                "memory": {
                    "audio_video": av_mem,
                    "video_only": vo_mem,
                    "resident_saved_gb": av_mem["resident_alloc_gb"] - vo_mem["resident_alloc_gb"],
                },
                "clips": rows,
                "pass": all_pass,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_path}. Overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
