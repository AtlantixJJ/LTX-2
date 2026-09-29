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
  stitched by keeping the earlier window's overlap in pixel space. That is
  precisely what produced
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
import time
from contextlib import nullcontext
from pathlib import Path

import decord
import torch
from safetensors import safe_open

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from scripts.prune.core import artifacts, ltx_adapter, model_registry, provenance, refine_core, refine_task, session
from scripts.prune.core.model_registry import RefinerModel
from scripts.prune.core.session import DTYPE
from scripts.prune.data import chunk_states, corpus, records
from scripts.prune.evaluate import decode, metrics
from scripts.prune.score import hooks, losses

decord.bridge.set_bridge("torch")


def _load_head_masks(path: Path, device: torch.device, transformer, *, model_key: str,
                     fingerprint: str) -> tuple[dict[str, torch.Tensor], str]:
    """Read an attributable, complete head mask for the loaded transformer."""
    widths = {name: attention.heads for name, attention in hooks.iter_video_attention(transformer)}
    masks, digest = hooks.read_mask_artifact(path, model_key=model_key, fingerprint=fingerprint, widths=widths)
    return {name: torch.tensor(values, device=device, dtype=torch.float32) for name, values in masks.items()}, digest


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
        raise SystemExit(f"No on-policy step-0 records under {root}; run "
                         "python -m scripts.prune.data.source_target --build-calibration first.")

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
        # Keep the reference on CPU in bounded reads. A 200-window source can be
        # thousands of frames; one get_batch over the whole video is too large.
        source_px = torch.cat([
            refine_core.read_pixel_window(vr, start, min(start + 64, covered), torch.device("cpu"), DTYPE)[1]
            for start in range(0, covered, 64)
        ])
        for start, stop in windows:
            norm, _ = refine_core.read_pixel_window(vr, start, stop, device, DTYPE)
            latents.append(encoder.tiled_encode(norm, None).cpu())
            del norm
    torch.cuda.empty_cache()
    return latents, source_px, float(vr.get_avg_fps())


def _rollout(transformer, denoiser, window_latents: list[torch.Tensor], geometry: refine_core.WindowGeometry,
             sigmas_list: list[float], fps: float, *, seed: int, device: torch.device,
             timing_rows: list[dict] | None = None) -> list[torch.Tensor]:
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
        if timing_rows is not None:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        latent = refine_core.refine_window(
            transformer, denoiser, l_init, carry, sigmas, tools, seed, device, DTYPE, stepper
        )
        if timing_rows is not None:
            torch.cuda.synchronize(device)
            timing_rows.append({"window_index": index, "refine_s": time.perf_counter() - started})
        carry = refine_core.carry_from(latent, geometry)
        refined.append(latent.cpu())
        if (index + 1) % 5 == 0:
            print(f"[t2] window {index + 1}/{len(window_latents)}", flush=True)
    return refined


def _stitch(decoded: list[torch.Tensor], windows: list[tuple[int, int]]) -> torch.Tensor:
    """Keep the first window's overlap, exactly as the deployed Stitcher does.

    ``decoded[i]`` is window i's decoded pixels ``[F, H, W, C]``; the result is the
    contiguous native-frame range ``[0, windows[-1][1])`` -- the same frames that end up
    in ``decode_full.mp4``.
    """
    if len(decoded) != len(windows) or not decoded:
        raise ValueError("one decoded video is required per planned window")
    kept = [decoded[0]]
    for i in range(1, len(decoded)):
        overlap = windows[i - 1][1] - windows[i][0]
        if not 0 <= overlap < decoded[i].shape[0]:
            raise ValueError(f"window {i} has invalid overlap {overlap}")
        kept.append(decoded[i][overlap:])
    return torch.cat(kept, dim=0)


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
    ap.add_argument("--profile-output", type=Path, default=None,
                    help="Matched per-window timing JSON; defaults beside --output.")
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
    ap.add_argument("--t2-video", type=Path, default=None,
                    help="Frame-aligned external long source video; requires --expected-source-sha256.")
    ap.add_argument("--expected-source-sha256", default=None,
                    help="Pin the exact external source bytes for a long-form comparison.")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.t2_video is not None:
        if args.t2_clip is not None or not args.expected_source_sha256:
            ap.error("--t2-video requires --expected-source-sha256 and cannot be combined with --t2-clip")
        if not args.t2_video.is_file():
            ap.error(f"source video does not exist: {args.t2_video}")
        actual_sha256 = provenance.file_sha256(args.t2_video)
        if actual_sha256 != args.expected_source_sha256:
            ap.error(f"source SHA256 differs: {actual_sha256}")
    elif args.expected_source_sha256 is not None:
        ap.error("--expected-source-sha256 requires --t2-video")

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
        source = args.t2_video or corpus.pick_clip(
            geometry, name=args.t2_clip, key=model.key, prefer="held_out", longest=True
        )
        pick = {"clip": source.parent.name, "source": str(source)}
        total = corpus.frame_count(source)
        windows = geometry.plan(total)
        if args.rollout_windows is not None and args.rollout_windows > len(windows):
            raise SystemExit(f"{source}: only {len(windows)} full windows, requested {args.rollout_windows}")
        if args.rollout_windows:
            windows = windows[: args.rollout_windows]
        latents, source_px, fps = _encode_windows(model, Path(pick["source"]), windows, device)
        t2 = {"clip": pick["clip"], "windows": windows, "latents": latents, "source_px": source_px,
              "fps": fps, "source": str(source), "source_frames": total,
              "source_sha256": actual_sha256 if args.t2_video is not None else provenance.file_sha256(source)}
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
        "method_sources": provenance.method_source_hashes(),
    }
    result["source_transformer_fingerprint"] = result["provenance"]["transformer_fingerprint"]
    if args.transformer_path is not None:
        with safe_open(args.transformer_path, framework="pt", device="cpu") as handle:
            config = json.loads((handle.metadata() or {}).get("config", "{}"))
        result["source_transformer_fingerprint"] = (
            config.get("transformer", {}).get("pruning", {}).get("source_transformer_fingerprint")
            or result["source_transformer_fingerprint"]
        )
    refined: list[torch.Tensor] = []
    timing_rows: list[dict] = []
    with s.transformer(args.transformer_path) as transformer:
        mask_values, mask_digest = _load_head_masks(
            args.head_masks, device, transformer, model_key=model.key,
            fingerprint=result["provenance"]["transformer_fingerprint"]
        ) if args.head_masks is not None else (None, None)
        mask_ctx = hooks.attach_head_masks(transformer, mask_values, requires_grad=False) \
            if mask_values is not None else nullcontext()
        with mask_ctx as masks:
            if masks is not None:
                dropped = sum(int((v == 0).sum()) for v in masks.values())
                total_heads = sum(v.numel() for v in masks.values())
                result["head_masks"] = {"source": str(args.head_masks), "sha256": mask_digest,
                                        "heads_dropped": dropped, "heads_total": total_heads}
                print(f"[head-masks] {dropped}/{total_heads} heads zeroed from {args.head_masks}", flush=True)
            result["T0"] = run_t0(model, transformer, denoiser, device, states_root, args.t0_max_records)
            if t2 is not None:
                refined = _rollout(transformer, denoiser, t2["latents"], geometry, student_sigmas, t2["fps"],
                                   seed=args.seed, device=device, timing_rows=timing_rows)
                result.setdefault("T2", {}).update({
                    "windows": len(refined), "clip": t2["clip"], "fps": t2["fps"],
                    "source": t2["source"], "source_sha256": t2["source_sha256"],
                    "source_frames": t2["source_frames"],
                    "covered_frames": t2["windows"][-1][1],
                    "source_frame_windows": t2["windows"],
                })
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

        # The first window finalizes all its frames; each later window finalizes
        # only its non-overlapping tail, matching the deployed Stitcher.
        stride = geometry.stride_frames
        rollout_rows = []
        finalized_spans = []
        for j in range(len(refined)):
            lo = 0 if j == 0 else t2["windows"][j - 1][1]
            hi = min(t2["windows"][j][1], n)
            if lo >= hi:
                break
            finalized_spans.append([lo, hi])
            rollout_rows.append({"chunk": j, "pred": pred[lo:hi], "teacher": source[lo:hi]})
        result["T2"].update(metrics.t2(rollout_rows))
        result["T2"]["stride_frames"] = stride
        result["T2"]["finalized_frame_spans"] = finalized_spans

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
    if timing_rows:
        profile_path = args.profile_output or path.with_name(path.stem + "_profile.json")
        profile = {"provenance": result["provenance"],
                   "source_transformer_fingerprint": result["source_transformer_fingerprint"],
                   "geometry": result["geometry"], "seed": result["seed"], "clip": result["T2"]["clip"],
                   "gpu_name": torch.cuda.get_device_name(device), "gpu_index": device.index,
                   "rows": timing_rows}
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "T0"}, indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
