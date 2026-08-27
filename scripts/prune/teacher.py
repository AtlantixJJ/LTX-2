"""Phase 0 item 6: define the teacher, record it, and freeze it.

**The plan's teacher premise does not hold, and this module does not use it.**
§6 proposes taking ``x0*`` from "a schedule *deeper than the student runs*
(student ``k2``, teacher ``k8``; the ``k8`` outputs for all 52 clips are already on
disk)". But in this distilled schedule the k-step preset selects a *tail length*,
and a longer tail starts at a **higher sigma**: ``k2`` starts at sigma 0.725, ``k8``
at sigma 1.0. ``expr/sam3dgs_vae_refine/FINDINGS.md`` measured what that does --
k8 scores **4.44 dB PSNR vs source**, and its own summary says the original latent
is "fully erased and replaced with pure text-to-video generation". The on-disk k8
outputs are therefore not converged refinements of the input; they are unrelated
videos of the same prompt. Regressing the student onto them would optimize the
refiner to *ignore* the latent it is supposed to repair.

What "deeper" has to mean here is **more steps at the student's own starting
sigma**, not more steps from a higher one: hold sigma_0 at the student's 0.725 and
subdivide the interval to 0 into ``--teacher-steps`` Euler steps. That is a
converged solution of the *same* probability-flow ODE the student truncates to two
steps, so ``L = ||D_theta(z_sigma, sigma) - x0*||^2`` is nonzero at every step
(the §6 requirement that kills the degenerate self-distillation gradient) while
still describing the task the refiner is actually deployed to do.

``freeze`` records the teacher spec, the corpus and the calibration split to
``expr/refiner_prune/<key>/teacher/teacher_manifest.json``. The split is **by
subject**, not by clip: the corpus holds several framings (``_crop`` /
``_original``) and motions of the same capture, so a per-clip split would leak the
same person across calibration and held-out and make every §10 held-out number
optimistic.

    conda run -n ltx python -m scripts.prune.teacher --model 2.5 --freeze
    conda run -n ltx python -m scripts.prune.teacher --model 2.5 --gpu-id 2 --validate
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import decord
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, VideoDecoder, _build_state
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.samplers import _step_state
from ltx_pipelines.utils.types import ModalitySpec

from scripts.prune import model_registry, preflight, prompt_cache, provenance, refine_task
from scripts.prune.model_registry import RefinerModel, WORKSPACE_ROOT

decord.bridge.set_bridge("torch")

DTYPE = torch.bfloat16
CORPUS_DIR = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"
WINDOW_FRAMES = 121
TEACHER_STEPS = 16
HOLDOUT_SUBJECTS = 3  # of the corpus's distinct subjects; ~40/12 clips (plan §6)
SPLIT_SEED = 0


@dataclass(frozen=True)
class TeacherSpec:
    """Everything needed to reproduce a teacher target, frozen into the manifest."""

    student_k_step: str
    sigma_0: float
    steps: int
    sigmas: list[float]
    sampler: str
    rationale: str


def teacher_sigmas(sigma_0: float, steps: int) -> list[float]:
    """``steps`` Euler steps from *sigma_0* down to 0, linearly spaced.

    Spacing only has to be fine enough that the Euler discretization error is
    negligible against the two-step student; it does not have to imitate the
    distilled schedule's shape, which was chosen for few-step sampling, not for
    accuracy. Linear is the spacing the sanity check in ``validate`` is run at.
    """
    if steps < 2:
        raise ValueError(f"teacher steps must be >= 2, got {steps}")
    return [sigma_0 * (1.0 - i / steps) for i in range(steps)] + [0.0]


def spec_for(model: RefinerModel, k_step: str = refine_task.K_STEP, steps: int = TEACHER_STEPS) -> TeacherSpec:
    sigma_0 = refine_task.schedule_for(model.sigmas, k_step)[0]
    return TeacherSpec(
        student_k_step=k_step,
        sigma_0=sigma_0,
        steps=steps,
        sigmas=teacher_sigmas(sigma_0, steps),
        sampler="euler",
        rationale=(
            "Deeper schedule at the STUDENT's sigma_0, not the plan's k8 preset: k8 starts at "
            "sigma 1.0, which FINDINGS.md measured at 4.44 dB vs source (the latent is erased and "
            "regenerated from the prompt), so k8 outputs are not converged refinements of the input."
        ),
    )


def corpus() -> list[dict]:
    """Every clip directory with a source video, tagged with its subject id."""
    clips = []
    for source in sorted(CORPUS_DIR.glob("*/source.mp4")):
        clip_dir = source.parent
        metrics_path = clip_dir / "sweep_metrics.json"
        entry = {"clip": clip_dir.name, "source": str(source), "subject": clip_dir.name.split("__", 1)[0]}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            entry["subject"] = metrics.get("subject_id", entry["subject"])
            entry["geometry"] = metrics.get("geometry")
            entry["prompt"] = metrics.get("prompt")
            entry["k_psnr_vs_source"] = {
                k: v.get("overall_psnr_db_vs_source") for k, v in (metrics.get("variants") or {}).items()
            }
        clips.append(entry)
    if not clips:
        raise SystemExit(f"No clips with source.mp4 under {CORPUS_DIR}")
    return clips


def split_by_subject(clips: list[dict], holdout_subjects: int = HOLDOUT_SUBJECTS, seed: int = SPLIT_SEED) -> dict:
    subjects = sorted({c["subject"] for c in clips})
    rng = random.Random(seed)
    holdout = set(rng.sample(subjects, min(holdout_subjects, len(subjects))))
    return {
        "seed": seed,
        "holdout_subjects": sorted(holdout),
        "calibration": [c["clip"] for c in clips if c["subject"] not in holdout],
        "held_out": [c["clip"] for c in clips if c["subject"] in holdout],
    }


def freeze(model: RefinerModel, spec: TeacherSpec, *, hash_sources: bool = True) -> Path:
    """Write the frozen teacher manifest for *model*."""
    clips = corpus()
    if hash_sources:
        for c in clips:
            c["source_sha256"] = provenance.file_sha256(c["source"])
    manifest = {
        "provenance": provenance.stamp(model, script="teacher.freeze"),
        "teacher": asdict(spec),
        "student": {
            "k_step": refine_task.K_STEP,
            "sigmas": refine_task.schedule_for(model.sigmas, refine_task.K_STEP),
        },
        "split": split_by_subject(clips),
        "corpus": clips,
        "k8_is_not_a_teacher": {
            "claim": "plan §6 proposes the on-disk k8 outputs as x0*",
            "evidence": "expr/sam3dgs_vae_refine/FINDINGS.md: k8 mean PSNR vs source 4.44 dB",
            "conclusion": "k8 starts at sigma 1.0 and regenerates from the prompt; not a refinement of the input",
        },
    }
    out_dir = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "teacher_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR between two [0,1] float tensors of identical shape."""
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def _read_window(path: Path, device: torch.device) -> torch.Tensor:
    vr = decord.VideoReader(str(path))
    if len(vr) < WINDOW_FRAMES:
        raise ValueError(f"{path}: {len(vr)} frames, need >= {WINDOW_FRAMES} for one F%t==1 window")
    raw = vr.get_batch(range(WINDOW_FRAMES))
    _, h, w, _ = raw.shape
    crop_h, crop_w = (h // 32) * 32, (w // 32) * 32
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    raw = raw[:, top : top + crop_h, left : left + crop_w, :]
    norm = raw.permute(3, 0, 1, 2).unsqueeze(0).to(dtype=DTYPE, device=device)
    return (norm / 127.5) - 1.0


def validate(model: RefinerModel, spec: TeacherSpec, device: torch.device, num_clips: int, seed: int) -> dict:
    """Run student and teacher schedules on real windows and compare against source.

    The point is falsifiable: if the teacher construction is right, ``x0*`` stays
    close to the source (it is a *repair* of the VAE round-trip, a few dB below the
    VAE ceiling) while the k8 numbers already on disk sit near 4 dB. If the teacher
    came out at 4 dB too, the construction would be wrong and Phase 1 would have
    been built on it.
    """
    # One clip per subject, and only clips holding a whole F%t==1 window -- some
    # corpus entries (the `canonical_rotation` renders) are 113 frames.
    by_subject: dict[str, dict] = {}
    for c in corpus():
        if c["subject"] in by_subject:
            continue
        try:
            if len(decord.VideoReader(c["source"])) >= WINDOW_FRAMES:
                by_subject[c["subject"]] = c
        except Exception:
            continue
        if len(by_subject) >= num_clips:
            break
    if len(by_subject) < num_clips:
        raise SystemExit(f"Only {len(by_subject)} subjects have a >= {WINDOW_FRAMES}-frame clip, need {num_clips}.")
    chosen = list(by_subject.values())

    student_sigmas = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)
    video_context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    stepper = EulerDiffusionStep()

    encoded = []
    with torch.no_grad():
        conditioner = ImageConditioner(model.paths.video_vae(), DTYPE, device)
        with gpu_model(conditioner._build_encoder()) as encoder:
            for c in chosen:
                pixels = _read_window(Path(c["source"]), device)
                encoded.append({**c, "pixels": pixels.cpu(), "latent": encoder.tiled_encode(pixels, None)})
                del pixels
    torch.cuda.empty_cache()

    stage = DiffusionStage.from_checkpoint(
        model.paths.transformer(),
        DTYPE,
        device,
        model_configurator=LTXVideoOnlyModelConfigurator,
        scale_factors=model.scale_factors,
    )

    def tools_for(pixels: torch.Tensor) -> VideoLatentTools:
        _, _, frames, height, width = pixels.shape
        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=24.0)
        v_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=model.caps.latent_channels, scale_factors=model.scale_factors
        )
        return VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, 24.0, scale_factors=model.scale_factors)

    def run(video_tools, latent, sigmas_list):
        sigmas = torch.tensor(sigmas_list, dtype=torch.float32, device=device)
        noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed))
        state = _build_state(
            ModalitySpec(context=video_context, conditionings=[], noise_scale=sigmas_list[0], initial_latent=latent),
            video_tools,
            noiser,
            DTYPE,
            device,
        )
        denoiser = SimpleDenoiser(video_context, None)
        for i in range(len(sigmas_list) - 1):
            result, _ = denoiser(transformer, state, None, sigmas, i)
            state = _step_state(state, result.denoised, stepper, sigmas, i)
        return video_tools.unpatchify(video_tools.clear_conditioning(state)).latent

    refined: list[dict] = []
    with torch.no_grad():
        with stage._transformer_ctx(video_tools=tools_for(encoded[0]["pixels"])) as transformer:
            for item in encoded:
                video_tools = tools_for(item["pixels"])
                print(f"[teacher] {item['clip']}: student k2 ...", flush=True)
                student = run(video_tools, item["latent"], student_sigmas)
                print(f"[teacher] {item['clip']}: teacher {spec.steps} steps ...", flush=True)
                target = run(video_tools, item["latent"], spec.sigmas)
                refined.append({**item, "student": student, "teacher": target})
    del stage
    torch.cuda.empty_cache()

    rows = []
    with torch.no_grad():
        decoder_holder = VideoDecoder(model.paths.video_vae(), DTYPE, device)
        with gpu_model(decoder_holder._decoder_builder.build(device=device, dtype=DTYPE).eval()) as decoder:
            for item in refined:
                source = (item["pixels"].float() + 1.0) / 2.0  # [1,C,F,H,W] in [0,1]
                source = source[0].permute(1, 0, 2, 3)  # [F,C,H,W]
                outs = {}
                for name in ("latent", "student", "teacher"):
                    dec = torch.cat(list(decoder.decode_video(item[name], None, None)), dim=0).cpu().float()
                    outs[name] = dec.clamp(0, 1).permute(0, 3, 1, 2) if dec.shape[-1] in (1, 3) else dec.clamp(0, 1)
                rows.append(
                    {
                        "clip": item["clip"],
                        "subject": item["subject"],
                        "vae_roundtrip_psnr_vs_source": _psnr(outs["latent"], source),
                        "student_k2_psnr_vs_source": _psnr(outs["student"], source),
                        "teacher_psnr_vs_source": _psnr(outs["teacher"], source),
                        "teacher_psnr_vs_student": _psnr(outs["teacher"], outs["student"]),
                        "on_disk_k8_psnr_vs_source": (item.get("k_psnr_vs_source") or {}).get("k8"),
                        "on_disk_k2_psnr_vs_source": (item.get("k_psnr_vs_source") or {}).get("k2"),
                    }
                )
                print(json.dumps(rows[-1], indent=2), flush=True)
    return {"clips": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--teacher-steps", type=int, default=TEACHER_STEPS)
    ap.add_argument("--freeze", action="store_true", help="Write the frozen teacher manifest.")
    ap.add_argument("--validate", action="store_true", help="Run student vs teacher on real clips (GPU).")
    ap.add_argument("--num-clips", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = preflight.check(args.model, gpu_id=args.gpu_id if args.validate else None)
    spec = spec_for(model, steps=args.teacher_steps)
    print(f"teacher sigma_0={spec.sigma_0} steps={spec.steps}")
    print(f"teacher sigmas: {[round(s, 4) for s in spec.sigmas]}")

    if args.freeze:
        path = freeze(model, spec)
        manifest = json.loads(path.read_text())
        print(
            f"Froze {len(manifest['corpus'])} clips / {len(set(c['subject'] for c in manifest['corpus']))} subjects "
            f"-> {len(manifest['split']['calibration'])} calib, {len(manifest['split']['held_out'])} held out"
        )
        print(f"Wrote {path}")

    if args.validate:
        device = torch.device(f"cuda:{args.gpu_id}")
        report = validate(model, spec, device, args.num_clips, args.seed)
        out_dir = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "teacher"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "teacher_validation.json"
        out_path.write_text(
            json.dumps(
                {
                    "provenance": provenance.stamp(model, device, script="teacher.validate"),
                    "teacher": asdict(spec),
                    **report,
                },
                indent=2,
            )
        )
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
