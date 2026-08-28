"""Phase 1 source-target calibration data.

No fine-step teacher is used.  The target is the VAE-encoded source latent:
``||D(z_sigma, sigma) - x_source||²``.  This is ordinary denoising supervision
and supplies the nonzero output residual used by mask-gradient (VJP) scoring.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import decord
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.tools import VideoLatentTools
from ltx_pipelines.utils.blocks import ImageConditioner
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.samplers import _step_state

from scripts.prune import artifacts, chunk_states, corpus as refine_corpus, model_registry, provenance, refine_core, refine_task, session
from scripts.prune.model_registry import RefinerModel
from scripts.prune.session import DTYPE

decord.bridge.set_bridge("torch")


def corpus() -> list[dict]:
    out = []
    for source in refine_corpus.sources():
        directory = source.parent
        out.append({"clip": directory.name, "source": str(source), "subject": refine_corpus.subject_of(directory.name)})
    if not out:
        raise SystemExit(f"No clips with source.mp4 under {refine_corpus.CORPUS_DIR}")
    return out


def freeze(model: RefinerModel) -> Path:
    """CPU-only: read corpus headers and freeze the calibration/held-out split. No device needed."""
    clips = corpus()
    subjects = sorted({c["subject"] for c in clips})
    held = set(random.Random(0).sample(subjects, min(3, len(subjects))))
    for c in clips:
        c["source_sha256"] = provenance.file_sha256(c["source"])
    data = {
        "provenance": provenance.stamp(model, script="source_target.freeze"),
        "target": {
            "kind": "vae_encoded_source_latent",
            "loss": "masked x0 MSE on fresh chunk tokens",
            "rationale": "a real source target supplies the nonzero residual for J^T residual gate scoring",
        },
        "student": {"k_step": refine_task.K_STEP, "sigmas": refine_task.schedule_for(model.sigmas, refine_task.K_STEP)},
        "split": {
            "seed": 0,
            "holdout_subjects": sorted(held),
            "calibration": [c["clip"] for c in clips if c["subject"] not in held],
            "held_out": [c["clip"] for c in clips if c["subject"] in held],
        },
        "corpus": clips,
    }
    path = artifacts.manifest(model.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def _read_chunk(path: Path, frames: int, device: torch.device, *, spatial_scale: tuple[int, int] | None = None) -> torch.Tensor:
    """First ``frames`` source frames as the VAE encoder's ``[-1, 1]`` input.

    Delegates to ``refine_core.read_pixel_window`` so the crop anchor matches
    ``scripts/vae_refine_sliding_window.py`` (centered, not top-left). ``spatial_scale``
    is accepted and ignored -- the shared reader always crops to a multiple of 32, the
    VAE's spatial factor -- and is kept so existing call sites need no edit.
    """
    vr = decord.VideoReader(str(path))
    if len(vr) < frames:
        raise ValueError(f"{path}: {len(vr)} frames, need >= {frames}")
    norm, _ = refine_core.read_pixel_window(vr, 0, frames, device, DTYPE)
    return norm


def _fps(path: Path | str) -> float:
    """The clip's own frame rate. Never default this: VideoLatentTools divides the
    temporal RoPE axis by it (see refine_core.build_tools), and 41 of the 44 corpus
    clips are 30 fps while the three canonical_rotation renders are 24.
    """
    return refine_corpus.fps(Path(path))


def _tools(model: RefinerModel, latent: torch.Tensor, fps: float) -> VideoLatentTools:
    return refine_core.build_tools(latent, fps, model.scale_factors)


def build_calibration(s: session.Session, *, max_clips: int | None, seed: int) -> Path:
    manifest_path = artifacts.manifest(s.key)
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}; run --freeze first.")
    manifest = json.loads(manifest_path.read_text())
    split = {x: "calibration" for x in manifest["split"]["calibration"]}
    split.update({x: "held_out" for x in manifest["split"]["held_out"]})
    clips = [c for c in manifest["corpus"] if c["clip"] in split]
    clips = clips if max_clips is None else clips[:max_clips]

    encoded = []
    with torch.no_grad(), gpu_model(ImageConditioner(s.model.paths.video_vae(), DTYPE, s.device)._build_encoder()) as encoder:
        for clip in clips:
            for n in refine_task.CHUNK_LATENT_FRAMES:
                # The same window `--window-frames {17,25,33} --overlap-frames 9` gives the
                # refine script: keyframe + CTX_LATENT_FRAMES frozen + n fresh latent frames.
                frames = refine_task.calibration_geometry(n, s.model.scale_factors).window_frames
                try:
                    encoded.append((clip, n, encoder.tiled_encode(_read_chunk(Path(clip["source"]), frames, s.device), None).cpu(), _fps(clip["source"])))
                except ValueError:
                    continue
    if not encoded:
        raise SystemExit("No source clip can build an AR calibration state.")

    out = artifacts.calibration(s.key)
    out.mkdir(parents=True, exist_ok=True)
    # Raw Python floats, not `s.sigmas.tolist()`: 0.725 is not exactly representable
    # in float32, so a tensor round-trip would perturb the renoised-branch `lerp`
    # weight below by ~2e-8 and desync every renoised record from the on-disk cache.
    sigmas = refine_task.schedule_for(s.model.sigmas, refine_task.K_STEP)
    records = []
    with s.transformer() as transformer:
        stepper = EulerDiffusionStep()
        for i, (clip, n, cpu, fps) in enumerate(encoded):
            l_init = cpu.to(device=s.device, dtype=DTYPE)
            ctx = l_init[:, :, 1:1 + refine_task.CTX_LATENT_FRAMES].contiguous()
            state = chunk_states.make_state(l_init, ctx, sigmas[0], _tools(s.model, l_init, fps), seed + i, s.device)
            target = state.clean_latent.detach().clone()
            for step in range(len(sigmas) - 1):
                meta = chunk_states.ChunkStateMeta(clip["clip"], split[clip["clip"]], "on_policy", sigmas[step], step, n, refine_task.CTX_LATENT_FRAMES, seed + i, fps)
                records.append(chunk_states.save_record(out / f"{clip['clip']}__n{n}__s{step}__on_policy.pt", state, target, meta))
                result, _ = s.denoiser(transformer, state, None, s.sigmas, step)
                state = _step_state(state, result.denoised, stepper, s.sigmas, step)
            for step, sigma in enumerate(sigmas[:-1]):
                noise = torch.randn(target.shape, device=s.device, dtype=target.dtype, generator=torch.Generator(device=s.device).manual_seed(seed + i + 100 + step))
                latent = torch.lerp(target.float(), noise.float(), sigma)
                latent = torch.lerp(target.float(), latent, state.denoise_mask.float()).to(DTYPE)
                renoised = replace(state, latent=latent, clean_latent=target)
                meta = chunk_states.ChunkStateMeta(clip["clip"], split[clip["clip"]], "renoised", sigma, step, n, refine_task.CTX_LATENT_FRAMES, seed + i + 100 + step, fps)
                records.append(chunk_states.save_record(out / f"{clip['clip']}__n{n}__s{step}__renoised.pt", renoised, target, meta))

    return chunk_states.write_index(
        out, records,
        provenance=provenance.stamp(s.model, s.device, script="source_target.build_calibration"),
        extra={"target": manifest["target"], "student_sigmas": sigmas, "manifest": str(manifest_path)},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    session.add_model_args(ap)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--build-calibration", action="store_true")
    ap.add_argument("--max-clips", type=int)
    args = ap.parse_args()

    if args.freeze:
        # freeze() only reads corpus headers and writes JSON -- no session needed.
        print(f"Wrote {freeze(model_registry.resolve(args.model))}")

    if args.build_calibration:
        s = session.open_session(args, script="source_target.build_calibration")
        print(f"Wrote {build_calibration(s, max_clips=args.max_clips, seed=args.seed)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
