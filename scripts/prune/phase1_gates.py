"""Phase 1's gate: evaluate the unpruned student against the source target.

plans/2026-08-26-refiner-head-ffn-pruning.md §6 ends with a gate that nothing else
in ``scripts/prune/`` answers: *"teacher cached; T0/T1/T2 near-zero for the unpruned
student against itself; the unpruned model's own T2 rollout characterizes the
intrinsic drift floor"*. ``teacher.py`` builds the calibration cache and
``metrics.py`` holds the metric functions, but until this module nothing ran them,
so every threshold in §10 had no denominator and the T2 rollout -- which §6 calls
"the gate that matters" -- had never been executed at all.

What this measures, and why each part is here:

* **T0** -- the unpruned student's own ``rel_l2`` against the frozen teacher target
  on the cached states. This is *not* expected to be zero: §6's whole target
  construction exists so that ``L = ||D_theta(z,sigma) - x0*||^2`` is nonzero at
  ``xi = 1``, which is what keeps §7.2b's mask-gradient estimator from being
  identically zero. This module therefore records the number as the **reference
  level** every pruned candidate is compared against, and separately records the
  per-step single-forward loss so the "nonzero at every step" claim is a measured
  fact rather than an assertion.

  Both token sets are reported: ``fresh`` (``state.denoise_mask``) and ``chunk``
  (``chunk_states.chunk_token_mask``). They differ -- see that function -- and the
  AR-relevant one is ``chunk``.

* **T1** -- the same comparison after decoding, so the latent-space number has a
  pixel-space anchor (§10's gate is stated in dB).

* **T2** -- the sequential AR rollout. Each chunk's *own* output becomes the next
  chunk's frozen context, for both the student and a teacher rolled out the same
  way. This is the only thing here that can catch compounding error, and the
  unpruned model's slope is the floor a pruned model has to be judged against.

  The corpus caps this: sources are 89-145 frames, i.e. 12-19 latent frames, so a
  single clip affords ~7-14 chunks, not §6's 200. ``--cycle-source`` extends the
  rollout by cycling the clip's own latent frames back through the fresh slot; the
  *context* chain is never reset, so the drift being measured is still purely the
  refiner feeding on its own output. The realized chunk count and whether cycling
  was used are both recorded, so a 200-chunk claim can never be read off a
  14-chunk run.

* **T3** -- the review pair, grid PNG and MP4, per §6.

    conda run -n ltx python -m scripts.prune.phase1_gates --model 2.5 --gpu-id 6
    conda run -n ltx python -m scripts.prune.phase1_gates --model 2.5 --gpu-id 6 \
        --rollout-chunks 200 --cycle-source
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import decord
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, VideoDecoder
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.samplers import _step_state

from scripts.prune import (
    chunk_states, losses, metrics, model_registry, preflight, prompt_cache, provenance, refine_task, teacher,
)
from scripts.prune.model_registry import RefinerModel, WORKSPACE_ROOT

decord.bridge.set_bridge("torch")

DTYPE = torch.bfloat16
ROLLOUT_CHUNK_LATENT_FRAMES = 1  # the hardest AR geometry: shortest chunk, most steps


def _tools(model: RefinerModel, latent: torch.Tensor) -> VideoLatentTools:
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1),
        VideoLatentShape.from_torch_shape(latent.shape),
        24.0,
        scale_factors=model.scale_factors,
    )


def _run_schedule(transformer, denoiser, state, sigmas: torch.Tensor, stepper) -> torch.Tensor:
    """Run a full schedule from *state* and return the final token-space latent."""
    for i in range(len(sigmas) - 1):
        result, _ = denoiser(transformer, state, None, sigmas, i)
        state = _step_state(state, result.denoised, stepper, sigmas, i)
    return state.latent


# ---------------------------------------------------------------------------
# T0
# ---------------------------------------------------------------------------


def run_t0(model: RefinerModel, transformer, denoiser, device: torch.device, root: Path) -> dict:
    """Student rel_l2 vs the frozen teacher target, on every cached on-policy state."""
    stepper = EulerDiffusionStep()
    sigmas_list = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)
    sigmas = torch.tensor(sigmas_list, dtype=torch.float32, device=device)
    rows: list[dict] = []
    for path in chunk_states.iter_records(root):
        state, target, meta = chunk_states.load_record(path, device)
        if meta.family != "on_policy" or meta.step_index != 0:
            continue
        chunk_mask = chunk_states.chunk_token_mask(state, meta)

        # (a) the deployed 2-step trajectory, scored against the teacher target.
        final = _run_schedule(transformer, denoiser, state, sigmas, stepper)
        row = {
            "record": path.name,
            "clip": meta.clip,
            "split": meta.split,
            "chunk_latent_frames": meta.chunk_latent_frames,
            "trajectory_rel_l2_fresh": float(losses.rel_l2(final, target, state)),
            "trajectory_rel_l2_chunk": float(losses.rel_l2(final, target, state, chunk_mask)),
        }

        # (b) the per-step single-forward loss -- the quantity §7.2b differentiates.
        # Recording it here is what makes "nonzero at xi = 1 at every step" a
        # measurement instead of an argument.
        step_losses = []
        walk = state
        for i in range(len(sigmas_list) - 1):
            result, _ = denoiser(transformer, walk, None, sigmas, i)
            step_losses.append({
                "step": i,
                "sigma": sigmas_list[i],
                "x0_mse_chunk": float(losses.x0_loss(result.denoised, target, walk, chunk_mask)),
                "rel_l2_chunk": float(losses.rel_l2(result.denoised, target, walk, chunk_mask)),
            })
            walk = _step_state(walk, result.denoised, stepper, sigmas, i)
        row["per_step"] = step_losses
        rows.append(row)
        print(f"[t0] {path.name}: chunk rel_l2 {row['trajectory_rel_l2_chunk']:.4f}", flush=True)

    if not rows:
        raise SystemExit(f"No on-policy step-0 records under {root}; run teacher --build-calibration first.")

    def agg(key: str, subset: list[dict]) -> dict | None:
        vals = [r[key] for r in subset]
        return {"count": len(vals), "mean": sum(vals) / len(vals), "max": max(vals)} if vals else None

    summary = {"records": rows}
    for split in ("calibration", "held_out"):
        subset = [r for r in rows if r["split"] == split]
        summary[split] = {
            "rel_l2_chunk": agg("trajectory_rel_l2_chunk", subset),
            "rel_l2_fresh": agg("trajectory_rel_l2_fresh", subset),
        }
    summary["min_per_step_x0_mse_chunk"] = min(s["x0_mse_chunk"] for r in rows for s in r["per_step"])
    summary["loss_nonzero_at_every_step"] = summary["min_per_step_x0_mse_chunk"] > 0.0
    return summary


# ---------------------------------------------------------------------------
# T2 (and the T1/T3 material it produces)
# ---------------------------------------------------------------------------


def _rollout(model: RefinerModel, transformer, denoiser, source_latent: torch.Tensor,
             sigmas_list: list[float], chunks: int, *, seed: int, cycle: bool, device: torch.device) -> tuple[torch.Tensor, int, int]:
    """Sequential AR rollout: each chunk's output is the next chunk's frozen context.

    Returns ``(stream, chunks_run, linear_chunks)``. The stream starts as the raw
    encoded ``1 + CTX`` leading latent frames -- both the student and the teacher
    start from the identical unrefined context, so the per-chunk comparison
    isolates their own accumulated error.

    ``linear_chunks`` is how many chunks consumed source frames in their natural
    order before ``--cycle-source`` began wrapping. Past that point the stream is
    no longer frame-aligned with the source clip, so any comparison *against the
    source* (T1, the T3 panels) must be truncated to ``linear_chunks``. The
    student-vs-teacher T2 comparison stays valid for the whole run: both rollouts
    consume the identical cycled frame order.
    """
    stepper = EulerDiffusionStep()
    sigmas = torch.tensor(sigmas_list, dtype=torch.float32, device=device)
    ctx_n, n_new = refine_task.CTX_LATENT_FRAMES, ROLLOUT_CHUNK_LATENT_FRAMES
    available = source_latent.shape[2]

    stream = source_latent[:, :, : 1 + ctx_n].clone()
    done = linear = 0
    for j in range(chunks):
        start = 1 + ctx_n + j * n_new
        if start + n_new > available:
            if not cycle:
                break
            # Cycle the clip's own latent frames back through the fresh slot. The
            # context chain is deliberately NOT reset, so what is being measured is
            # still the refiner consuming its own output, just for longer than the
            # clip alone allows.
            start = 1 + ctx_n + ((start - (1 + ctx_n)) % max(available - ctx_n - n_new, 1))
        else:
            linear = j + 1
        fresh = source_latent[:, :, start : start + n_new]
        if fresh.shape[2] != n_new:
            break
        ctx = stream[:, :, -ctx_n:].contiguous()
        keyframe = stream[:, :, -(ctx_n + 1) : -ctx_n].contiguous()
        l_init = torch.cat([keyframe, ctx, fresh], dim=2)
        state = chunk_states.make_state(l_init, ctx, sigmas_list[0], _tools(model, l_init), seed + j, device)
        final = _run_schedule(transformer, denoiser, state, sigmas, stepper)
        tools = _tools(model, l_init)
        refined = tools.unpatchify(tools.clear_conditioning(replace(state, latent=final))).latent
        # Append only the chunk frames; the regenerated index-0 keyframe is a
        # by-product of the calibration geometry and is not part of the stream.
        stream = torch.cat([stream, refined[:, :, -n_new:]], dim=2)
        done = j + 1
        if done % 10 == 0:
            print(f"[t2] chunk {done}/{chunks}", flush=True)
    return stream, done, linear


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--states", type=Path, default=None)
    ap.add_argument("--rollout-chunks", type=int, default=12)
    ap.add_argument("--cycle-source", action="store_true",
                    help="Extend the rollout past the clip's own length by cycling its latent frames.")
    ap.add_argument("--t2-clip", default=None, help="Clip directory name; defaults to the longest held-out clip.")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = preflight.check(args.model, sampler="euler", gpu_id=args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    out_root = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key
    states_root = args.states or (out_root / "calibration")
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    denoiser = SimpleDenoiser(context, None)
    student_sigmas = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)

    # --- pick and encode the T2 clip before the transformer is resident ---
    t2_source = None
    if not args.skip_t2:
        manifest = json.loads((out_root / "source_target" / "manifest.json").read_text())
        held = set(manifest["split"]["held_out"])
        pool = [c for c in manifest["corpus"] if c["clip"] in held] or manifest["corpus"]
        if args.t2_clip:
            pool = [c for c in manifest["corpus"] if c["clip"] == args.t2_clip] or pool
        pick = max(pool, key=lambda c: len(decord.VideoReader(c["source"])))
        frames = len(decord.VideoReader(pick["source"]))
        frames = model.scale_factors.time * ((frames - 1) // model.scale_factors.time) + 1
        conditioner = ImageConditioner(model.paths.video_vae(), DTYPE, device)
        with torch.no_grad(), gpu_model(conditioner._build_encoder()) as encoder:
            pixels = teacher._read_chunk(
                Path(pick["source"]), frames, device,
                spatial_scale=(model.scale_factors.height, model.scale_factors.width),
            )
            t2_source = {"clip": pick["clip"], "pixels": pixels.cpu(), "latent": encoder.tiled_encode(pixels, None).cpu()}
            del pixels
        torch.cuda.empty_cache()
        print(f"[t2] clip {pick['clip']}: {t2_source['latent'].shape[2]} latent frames", flush=True)

    # --- transformer-resident phase: T0 and source-aligned student rollout ---
    stage = DiffusionStage.from_checkpoint(
        model.paths.transformer(), DTYPE, device,
        model_configurator=LTXVideoOnlyModelConfigurator, scale_factors=model.scale_factors,
    )
    result: dict = {"provenance": provenance.stamp(model, device, script="phase1_gates"), "student_sigmas": student_sigmas, "target": "vae_encoded_source_latent"}
    latents: dict = {}
    with torch.no_grad(), stage._transformer_ctx() as transformer:
        result["T0"] = run_t0(model, transformer, denoiser, device, states_root)
        if t2_source is not None:
            src = t2_source["latent"].to(device=device, dtype=DTYPE)
            stream, done, linear = _rollout(model, transformer, denoiser, src, student_sigmas, args.rollout_chunks,
                                            seed=args.seed, cycle=args.cycle_source, device=device)
            latents["student"] = stream.cpu()
            result.setdefault("T2", {}).update({"student_chunks": done, "student_linear_chunks": linear})
            print(f"[t2] student: {done} chunks ({linear} source-aligned)", flush=True)
    del stage
    torch.cuda.empty_cache()

    # --- decode-resident phase: T1, T2 pixel metrics, T3 artifacts ---
    if latents:
        figures = out_root / "figures"
        chunks_done = result["T2"]["student_chunks"]
        ctx_n, n_new = refine_task.CTX_LATENT_FRAMES, ROLLOUT_CHUNK_LATENT_FRAMES
        decoder_holder = VideoDecoder(model.paths.video_vae(), DTYPE, device)
        with torch.no_grad(), gpu_model(decoder_holder._decoder_builder.build(device=device, dtype=DTYPE).eval()) as decoder:
            decoded = {}
            for name in ("student",):
                dec = torch.cat(list(decoder.decode_video(latents[name].to(device=device, dtype=DTYPE), None, None)), dim=0).cpu().float()
                decoded[name] = (dec.clamp(0, 1).permute(0, 3, 1, 2) if dec.shape[-1] in (1, 3) else dec.clamp(0, 1))
        source_px = ((t2_source["pixels"].float() + 1.0) / 2.0)[0].permute(1, 0, 2, 3)

        # Chunk j's pixels: latent frame (1 + ctx_n + j) maps to the pixel window
        # after the leading keyframe, `time` pixel frames per latent frame.
        t = model.scale_factors.time
        rollout_rows = []
        for j in range(chunks_done):
            lo = 1 + (ctx_n + j * n_new) * t
            hi = lo + n_new * t
            if hi > min(decoded["student"].shape[0], source_px.shape[0]):
                break
            rollout_rows.append({"chunk": j, "pred": decoded["student"][lo:hi], "teacher": source_px[lo:hi]})
        result["T2"].update(metrics.t2(rollout_rows))
        result["T2"]["clip"] = t2_source["clip"]
        result["T2"]["cycled_source"] = bool(args.cycle_source)
        result["T2"]["requested_chunks"] = args.rollout_chunks

        # Anything compared against the SOURCE has to stop where the rollout stopped
        # consuming source frames in order. Past the first --cycle-source wrap the
        # stream is still a valid student-vs-teacher comparison but is no longer
        # frame-aligned with the clip, and a PSNR-vs-source over that range would be
        # comparing unrelated frames.
        linear = result["T2"]["student_linear_chunks"]
        aligned = 1 + (ctx_n + linear * n_new) * t
        n = min(decoded["student"].shape[0], source_px.shape[0], aligned)
        result["T1"] = metrics.t1(decoded["student"][:n], source_px[:n])
        result["T1"]["frames_compared"] = n
        result["T1"]["source_aligned_chunks"] = linear
        grid = metrics.t3_grid([(t2_source["clip"], source_px[:n], source_px[:n], decoded["student"][:n])],
                               figures / "phase1_rollout_grid.png")
        video = metrics.t3_video(source_px[:n], source_px[:n], decoded["student"][:n],
                                 figures / "phase1_rollout.mp4", fps=24.0)
        result["T3"] = {"grid": str(grid), "video": str(video)}
        (figures / "INDEX.md").write_text(
            "# Phase 1 gate figures\n\n"
            "- `phase1_rollout_grid.png`: source | source target | student AR-rollout frames\n"
            "- `phase1_rollout.mp4`: aligned source | source target | student AR rollout\n"
        )

    path = out_root / "phase1_gates.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "T0"}, indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
