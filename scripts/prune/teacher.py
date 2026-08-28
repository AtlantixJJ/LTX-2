"""Phase 1 source-target calibration data.

No fine-step teacher is used.  The target is the VAE-encoded source latent:
``||D(z_sigma, sigma) - x_source||²``.  This is ordinary denoising supervision
and supplies the nonzero output residual used by mask-gradient (VJP) scoring.
"""
from __future__ import annotations
import argparse, json, random
from dataclasses import replace
from pathlib import Path
import decord, torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_core.tools import VideoLatentTools
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.samplers import _step_state
from scripts.prune import chunk_states, model_registry, preflight, prompt_cache, provenance, refine_core, refine_task
from scripts.prune.model_registry import RefinerModel, WORKSPACE_ROOT

decord.bridge.set_bridge("torch")
DTYPE = torch.bfloat16; CORPUS_DIR = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"

def corpus() -> list[dict]:
    out = []
    for source in sorted(CORPUS_DIR.glob("*/source.mp4")):
        d = source.parent; out.append({"clip": d.name, "source": str(source), "subject": d.name.split("__", 1)[0]})
    if not out: raise SystemExit(f"No clips with source.mp4 under {CORPUS_DIR}")
    return out

def freeze(model: RefinerModel) -> Path:
    clips = corpus(); subjects = sorted({c["subject"] for c in clips}); held = set(random.Random(0).sample(subjects, min(3, len(subjects))))
    for c in clips: c["source_sha256"] = provenance.file_sha256(c["source"])
    data = {"provenance": provenance.stamp(model, script="source_target.freeze"), "target": {"kind": "vae_encoded_source_latent", "loss": "masked x0 MSE on fresh chunk tokens", "rationale": "a real source target supplies the nonzero residual for J^T residual gate scoring"}, "student": {"k_step": refine_task.K_STEP, "sigmas": refine_task.schedule_for(model.sigmas, refine_task.K_STEP)}, "split": {"seed": 0, "holdout_subjects": sorted(held), "calibration": [c["clip"] for c in clips if c["subject"] not in held], "held_out": [c["clip"] for c in clips if c["subject"] in held]}, "corpus": clips}
    root = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "source_target"; root.mkdir(parents=True, exist_ok=True); path = root / "manifest.json"; path.write_text(json.dumps(data, indent=2)); return path

def _read_chunk(path: Path, frames: int, device: torch.device, *, spatial_scale: tuple[int, int] | None = None) -> torch.Tensor:
    """First ``frames`` source frames as the VAE encoder's ``[-1, 1]`` input.

    Delegates to ``refine_core.read_pixel_window`` so the crop anchor matches
    ``scripts/vae_refine_sliding_window.py`` (centered, not top-left). ``spatial_scale``
    is accepted and ignored -- the shared reader always crops to a multiple of 32, the
    VAE's spatial factor -- and is kept so existing call sites need no edit.
    """
    vr = decord.VideoReader(str(path))
    if len(vr) < frames: raise ValueError(f"{path}: {len(vr)} frames, need >= {frames}")
    norm, _ = refine_core.read_pixel_window(vr, 0, frames, device, DTYPE)
    return norm

def _fps(path: Path | str) -> float:
    """The clip's own frame rate. Never default this: VideoLatentTools divides the
    temporal RoPE axis by it (see refine_core.build_tools), and 41 of the 44 corpus
    clips are 30 fps while the three canonical_rotation renders are 24."""
    return float(decord.VideoReader(str(path)).get_avg_fps())

def _tools(model: RefinerModel, latent: torch.Tensor, fps: float) -> VideoLatentTools:
    return refine_core.build_tools(latent, fps, model.scale_factors)

def build_calibration(model: RefinerModel, device: torch.device, *, max_clips: int | None, seed: int) -> Path:
    manifest_path = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "source_target" / "manifest.json"
    if not manifest_path.exists(): raise SystemExit(f"Missing {manifest_path}; run --freeze first.")
    manifest = json.loads(manifest_path.read_text()); split = {x: "calibration" for x in manifest["split"]["calibration"]}; split.update({x: "held_out" for x in manifest["split"]["held_out"]})
    clips = [c for c in manifest["corpus"] if c["clip"] in split]; clips = clips if max_clips is None else clips[:max_clips]
    encoded = []
    with torch.no_grad(), gpu_model(ImageConditioner(model.paths.video_vae(), DTYPE, device)._build_encoder()) as encoder:
        for clip in clips:
            for n in refine_task.CHUNK_LATENT_FRAMES:
                # The same window `--window-frames {17,25,33} --overlap-frames 9` gives the
                # refine script: keyframe + CTX_LATENT_FRAMES frozen + n fresh latent frames.
                frames = refine_task.calibration_geometry(n, model.scale_factors).window_frames
                try: encoded.append((clip, n, encoder.tiled_encode(_read_chunk(Path(clip["source"]), frames, device), None).cpu(), _fps(clip["source"])))
                except ValueError: continue
    if not encoded: raise SystemExit("No source clip can build an AR calibration state.")
    out = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "calibration"; out.mkdir(parents=True, exist_ok=True)
    sigmas = refine_task.schedule_for(model.sigmas, refine_task.K_STEP); context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device); records = []
    stage = DiffusionStage.from_checkpoint(model.paths.transformer(), DTYPE, device, model_configurator=LTXVideoOnlyModelConfigurator, scale_factors=model.scale_factors)
    with torch.no_grad(), stage._transformer_ctx() as transformer:
        denoiser = SimpleDenoiser(context, None); schedule = torch.tensor(sigmas, device=device); stepper = EulerDiffusionStep()
        for i, (clip, n, cpu, fps) in enumerate(encoded):
            l_init = cpu.to(device=device, dtype=DTYPE); ctx = l_init[:, :, 1:1 + refine_task.CTX_LATENT_FRAMES].contiguous(); state = chunk_states.make_state(l_init, ctx, sigmas[0], _tools(model, l_init, fps), seed + i, device); target = state.clean_latent.detach().clone()
            for step in range(len(sigmas) - 1):
                meta = chunk_states.ChunkStateMeta(clip["clip"], split[clip["clip"]], "on_policy", sigmas[step], step, n, refine_task.CTX_LATENT_FRAMES, seed + i, fps); records.append(chunk_states.save_record(out / f"{clip['clip']}__n{n}__s{step}__on_policy.pt", state, target, meta)); result, _ = denoiser(transformer, state, None, schedule, step); state = _step_state(state, result.denoised, stepper, schedule, step)
            for step, sigma in enumerate(sigmas[:-1]):
                noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=torch.Generator(device=device).manual_seed(seed + i + 100 + step)); latent = torch.lerp(target.float(), noise.float(), sigma); latent = torch.lerp(target.float(), latent, state.denoise_mask.float()).to(DTYPE); renoised = replace(state, latent=latent, clean_latent=target); meta = chunk_states.ChunkStateMeta(clip["clip"], split[clip["clip"]], "renoised", sigma, step, n, refine_task.CTX_LATENT_FRAMES, seed + i + 100 + step, fps); records.append(chunk_states.save_record(out / f"{clip['clip']}__n{n}__s{step}__renoised.pt", renoised, target, meta))
    return chunk_states.write_index(out, records, provenance=provenance.stamp(model, device, script="source_target.build_calibration"), extra={"target": manifest["target"], "student_sigmas": sigmas, "manifest": str(manifest_path)})

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS); ap.add_argument("--gpu-id", type=int, default=0); ap.add_argument("--freeze", action="store_true"); ap.add_argument("--build-calibration", action="store_true"); ap.add_argument("--max-clips", type=int); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args(); model = preflight.check(a.model, gpu_id=a.gpu_id if a.build_calibration else None)
    if a.freeze: print(f"Wrote {freeze(model)}")
    if a.build_calibration: print(f"Wrote {build_calibration(model, torch.device(f'cuda:{a.gpu_id}'), max_clips=a.max_clips, seed=a.seed)}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
