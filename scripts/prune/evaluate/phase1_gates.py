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

* **T2** -- the sequential sliding-window rollout, run through ``refine_core`` at the
  DEPLOYED geometry: 25-frame windows with a 9-frame overlap, each window noised from
  its own VAE encode and continued from the previous window's refined carryover, then
  cross-faded in pixel space. That is precisely what produced
  ``expr/sam3dgs_vae_refine/*/k2_longform_v3_carryover/decode_full.mp4``, and
  ``scripts/prune/method_parity.py`` is the gate that proves the two agree
  bit-for-bit. Chunk index is rollout depth: chunk *j* is native frames
  ``[j*stride, (j+1)*stride)``, so the PSNR slope still measures compounding error.

  An earlier version of this module rolled out a *different* geometry it invented
  (4 frozen latent frames, 1 fresh, a regular latent frame spliced into the causal
  keyframe slot, 24 fps hardcoded against 30 fps clips). Its baseline video was
  visibly softer than ``decode_full.mp4``, which made every pruning delta measured
  against it a delta on a method nobody deploys.

  The corpus caps the rollout length: sources are 89-145 frames, so a clip affords
  ~5-8 windows. ``--rollout-windows`` caps it further; nothing extends it past the
  clip, because a wrapped window is no longer frame-aligned with the source and a
  PSNR against it compares unrelated frames.

* **T3** -- the review pair, grid PNG and MP4, per §6.

    conda run -n ltx python -m scripts.prune.evaluate.phase1_gates --model 2.5 --gpu-id 6
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import decord
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from scripts.prune.core import artifacts, ltx_adapter, model_registry, refine_core, refine_task, session
from scripts.prune.core.model_registry import RefinerModel
from scripts.prune.core.session import DTYPE
from scripts.prune.data import chunk_states, corpus, records
from scripts.prune.evaluate import decode, metrics
from scripts.prune.score import hooks, losses

decord.bridge.set_bridge("torch")


def _load_head_masks(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a runtime head mask from a ``head_scores.json`` iterative-pruning report."""
    data = json.loads(path.read_text())
    masks = data.get("iterative", {}).get("masks", data.get("masks"))
    if masks is None:
        raise SystemExit(f"{path}: no 'iterative.masks' (or top-level 'masks') found")
    return {name: torch.tensor(values, device=device, dtype=torch.float32) for name, values in masks.items()}


def _run_schedule(transformer, denoiser, state, sigmas: torch.Tensor, stepper) -> torch.Tensor:
    """Run a full schedule from *state* and return the final token-space latent."""
    for i in range(len(sigmas) - 1):
        result, _ = denoiser(transformer, state, None, sigmas, i)
        state = ltx_adapter.step_state(state, result.denoised, stepper, sigmas, i)
    return state.latent


# ---------------------------------------------------------------------------
# T0
# ---------------------------------------------------------------------------


def run_t0(model: RefinerModel, transformer, denoiser, device: torch.device, root: Path, max_records: int | None = None) -> dict:
    """Student rel_l2 vs the frozen teacher target, on every cached on-policy state."""
    stepper = EulerDiffusionStep()
    sigmas_list = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)
    sigmas = torch.tensor(sigmas_list, dtype=torch.float32, device=device)
    rows: list[dict] = []
    candidates = records.select(root, family="on_policy", step_index=0, limit=max_records)
    for path in candidates:
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
            walk = ltx_adapter.step_state(walk, result.denoised, stepper, sigmas, i)
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


def _encode_windows(model: RefinerModel, clip_path: Path, windows: list[tuple[int, int]],
                    device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor, float]:
    """VAE-encode every planned window, plus the source pixels the metrics compare against.

    Done in its own phase with the transformer NOT resident -- the 22B video-only
    transformer peaks around 42 GB and the encoder around 4 GB on this geometry, which
    together do not fit a 49 GB A6000. This is the same A/B/C phase split
    ``vae_refine_sliding_window.run_batch`` uses, and it is free: a window's encode
    depends only on source pixels, never on any earlier window's refinement.
    """
    vr = decord.VideoReader(str(clip_path))
    covered = windows[-1][1]
    latents: list[torch.Tensor] = []
    with ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder:
        source_px = refine_core.read_pixel_window(vr, 0, covered, device, DTYPE)[1]
        for start, stop in windows:
            norm, _ = refine_core.read_pixel_window(vr, start, stop, device, DTYPE)
            latents.append(encoder.tiled_encode(norm, None).cpu())
            del norm
    torch.cuda.empty_cache()
    return latents, source_px, float(vr.get_avg_fps())


def _rollout(transformer, denoiser, window_latents: list[torch.Tensor], geometry: refine_core.WindowGeometry,
             sigmas_list: list[float], fps: float, *, seed: int, device: torch.device) -> list[torch.Tensor]:
    """The deployed sliding-window rollout: window i+1 continues from window i's output.

    Exactly ``scripts/vae_refine_sliding_window.py``'s phase B, through the shared
    ``refine_core`` primitives -- each window is noised from its OWN full VAE encode
    (so latent index 0 is a genuine causal keyframe) and the previous window's trailing
    ``geometry.context_latent_frames`` refined latent frames are frozen in at index 1.
    ``scripts/prune/method_parity.py`` asserts this reproduces that script's
    ``latent_cache/*.pt`` bit-for-bit.

    The seed is constant across windows, as in the run script: each window draws the
    same noise realization, and continuity comes from the frozen carryover rather than
    from correlating the draws.
    """
    stepper = EulerDiffusionStep()
    sigmas = torch.tensor(sigmas_list, dtype=torch.float32, device=device)
    refined: list[torch.Tensor] = []
    carry: torch.Tensor | None = None
    for index, encoded in enumerate(window_latents):
        l_init = encoded.to(device=device, dtype=DTYPE)
        tools = refine_core.build_tools(l_init, fps, geometry.scale_factors)
        latent = refine_core.refine_window(
            transformer, denoiser, l_init, carry, sigmas, tools, seed, device, DTYPE, stepper
        )
        carry = refine_core.carry_from(latent, geometry)
        refined.append(latent.cpu())
        if (index + 1) % 5 == 0:
            print(f"[t2] window {index + 1}/{len(window_latents)}", flush=True)
    return refined


def _stitch(decoded: list[torch.Tensor], windows: list[tuple[int, int]]) -> torch.Tensor:
    """Linear cross-fade over each overlap, exactly as the run script finalizes frames.

    ``decoded[i]`` is window i's decoded pixels ``[F, H, W, C]``; the result is the
    contiguous native-frame range ``[0, windows[-1][1])`` -- the same frames that end up
    in ``decode_full.mp4``.
    """
    out: list[torch.Tensor] = []
    pending_tail: torch.Tensor | None = None
    for i, pixels in enumerate(decoded):
        start, end = windows[i]
        overlap_prev = max(0, windows[i - 1][1] - start) if i > 0 else 0
        taken = 0
        if overlap_prev > 0 and pending_tail is not None:
            ov = min(overlap_prev, pending_tail.shape[0], pixels.shape[0])
            w = torch.linspace(0.0, 1.0, ov).view(ov, 1, 1, 1)
            out.append((1.0 - w) * pending_tail[:ov] + w * pixels[:ov])
            taken = ov
        overlap_next = max(0, end - windows[i + 1][0]) if i < len(windows) - 1 else 0
        body_end = pixels.shape[0] - overlap_next
        if body_end > taken:
            out.append(pixels[taken:body_end])
        pending_tail = pixels[body_end:] if overlap_next > 0 else None
    if pending_tail is not None and pending_tail.numel():
        out.append(pending_tail)
    return torch.cat(out, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--states", type=Path, default=None)
    ap.add_argument("--transformer-path", type=Path, default=None,
                    help="Evaluate a freshly exported pruned transformer rather than the registry default.")
    ap.add_argument("--head-masks", type=Path, default=None,
                    help="Apply a runtime head mask (a head_scores.json iterative-pruning report) for the whole run.")
    ap.add_argument("--output", type=Path, default=None,
                    help="JSON destination; defaults to the unpruned Phase-1 baseline path.")
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="Where to write the T3 grid/video; defaults to <out_root>/figures. "
                         "Set this to a distinct directory for concurrent runs to avoid clobbering each other.")
    ap.add_argument("--t0-max-records", type=int, default=None,
                    help="Cap T0 to an evenly-strided sample of on-policy step-0 records (default: all).")
    ap.add_argument("--rollout-windows", "--rollout-chunks", dest="rollout_windows", type=int, default=None,
                    help="Cap the rollout to this many sliding windows (default: as many as the clip affords).")
    ap.add_argument("--window-frames", type=int, default=refine_task.WINDOW_FRAMES,
                    help="Deployed window length; the default is the geometry that produced "
                         "expr/sam3dgs_vae_refine/*/k2_longform_v3_carryover/decode_full.mp4.")
    ap.add_argument("--overlap-frames", type=int, default=refine_task.OVERLAP_FRAMES)
    ap.add_argument("--t2-clip", default=None, help="Clip directory name; defaults to the longest held-out clip.")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s = session.open_session(args, script="phase1_gates", transformer_path=args.transformer_path)
    model, device = s.model, s.device
    out_root = artifacts.root(model.key)
    states_root = args.states or artifacts.calibration(model.key)
    denoiser = s.denoiser
    # Raw Python floats, not s.sigmas.tolist(): 0.725 is not exactly representable in
    # float32, so a tensor round-trip would perturb this JSON's recorded value even
    # though _rollout's own re-tensorization reconverges to the same bits either way.
    student_sigmas = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)
    geometry = refine_core.WindowGeometry(
        window_frames=args.window_frames, overlap_frames=args.overlap_frames, scale_factors=model.scale_factors
    )

    # --- pick and encode the T2 clip before the transformer is resident ---
    t2 = None
    if not args.skip_t2:
        source = corpus.pick_clip(
            geometry, name=args.t2_clip, key=model.key, prefer="held_out", longest=True
        )
        pick = {"clip": source.parent.name, "source": str(source)}
        total = corpus.frame_count(source)
        windows = geometry.plan(total)
        if args.rollout_windows:
            windows = windows[: args.rollout_windows]
        latents, source_px, fps = _encode_windows(model, Path(pick["source"]), windows, device)
        t2 = {"clip": pick["clip"], "windows": windows, "latents": latents, "source_px": source_px, "fps": fps}
        print(f"[t2] clip {pick['clip']}: {total} frames -> {len(windows)} windows of "
              f"{geometry.window_frames} (overlap {geometry.overlap_frames}, stride {geometry.stride_frames}) "
              f"at {fps} fps", flush=True)

    # --- transformer-resident phase: T0 and the sliding-window rollout ---
    result: dict = {
        "provenance": s.stamp(),
        "student_sigmas": student_sigmas,
        "target": "vae_encoded_source_latent",
        "geometry": geometry.as_dict(),
        "seed": args.seed,
    }
    refined: list[torch.Tensor] = []
    with s.transformer(args.transformer_path) as transformer:
        mask_ctx = hooks.attach_head_masks(transformer, _load_head_masks(args.head_masks, device), requires_grad=False) \
            if args.head_masks is not None else nullcontext()
        with mask_ctx as masks:
            if masks is not None:
                dropped = sum(int((v == 0).sum()) for v in masks.values())
                total_heads = sum(v.numel() for v in masks.values())
                result["head_masks"] = {"source": str(args.head_masks), "heads_dropped": dropped, "heads_total": total_heads}
                print(f"[head-masks] {dropped}/{total_heads} heads zeroed from {args.head_masks}", flush=True)
            result["T0"] = run_t0(model, transformer, denoiser, device, states_root, args.t0_max_records)
            if t2 is not None:
                refined = _rollout(transformer, denoiser, t2["latents"], geometry, student_sigmas, t2["fps"],
                                   seed=args.seed, device=device)
                result.setdefault("T2", {}).update({"windows": len(refined), "clip": t2["clip"], "fps": t2["fps"]})
                print(f"[t2] rollout: {len(refined)} windows refined", flush=True)

    # --- decode-resident phase: T1, T2 pixel metrics, T3 artifacts ---
    if refined:
        figures = args.figures_dir or artifacts.figures(model.key)
        figures.mkdir(parents=True, exist_ok=True)
        with s.decoder() as decoder:
            decoded = [decode.decode_latent(s, latent, decoder) for latent in refined]
        stitched = _stitch(decoded, t2["windows"])
        del decoded

        source_px = t2["source_px"]
        n = min(stitched.shape[0], source_px.shape[0])
        pred = stitched[:n].permute(0, 3, 1, 2)
        source = source_px[:n].permute(0, 3, 1, 2)

        # T2 chunk = one stride of finalized frames, so chunk index IS rollout depth:
        # chunk j spans native frames [j*stride, (j+1)*stride), the span window j is the
        # last (and, outside the cross-fade, only) window to touch.
        stride = geometry.stride_frames
        rollout_rows = []
        for j in range(len(refined)):
            lo, hi = j * stride, min((j + 1) * stride, n)
            if lo >= hi:
                break
            rollout_rows.append({"chunk": j, "pred": pred[lo:hi], "teacher": source[lo:hi]})
        result["T2"].update(metrics.t2(rollout_rows))
        result["T2"]["stride_frames"] = stride

        result["T1"] = metrics.t1(pred, source)
        result["T1"]["frames_compared"] = n
        grid = metrics.t3_grid([(t2["clip"], source, source, pred)], figures / "phase1_rollout_grid.png")
        video = metrics.t3_video(source, source, pred, figures / "phase1_rollout.mp4", fps=t2["fps"])
        result["T3"] = {"grid": str(grid), "video": str(video)}
        (figures / "INDEX.md").write_text(
            "# Phase 1 gate figures\n\n"
            "- `phase1_rollout_grid.png`: source | source target | student sliding-window rollout\n"
            "- `phase1_rollout.mp4`: aligned source | source target | student rollout\n"
        )

    path = args.output or artifacts.phase1(model.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "T0"}, indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
