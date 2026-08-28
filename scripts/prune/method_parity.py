"""The gate: ``scripts/prune/``'s rollout must reproduce the refine script bit-for-bit.

``scripts/vae_refine_sliding_window.py`` produced every result under
``expr/sam3dgs_vae_refine/`` -- including the reference
``4D-Dress_00129_0__woman_dance_2_crop/k2_longform_v3_carryover/decode_full.mp4``.
Every pruning number is a delta against that method, so if the harness rolls out
*something else*, the deltas describe a model nobody deploys. That is exactly what
had happened: ``phase1_gates`` invented a 4-frozen/1-fresh geometry, spliced a
regular latent frame into the causal keyframe slot, and hardcoded 24 fps against a
corpus that is mostly 30 fps -- three independent changes to the transformer's
input, each invisible in any JSON the harness wrote.

This module makes the agreement checkable instead of asserted. It runs BOTH paths
on one clip at one geometry with one seed and compares the refined latents with
``torch.equal``:

* **reference** -- ``scripts/vae_refine_sliding_window.py`` in a subprocess, as a
  user would run it, writing ``latent_cache/win_*_latent.pt``.
* **harness** -- ``phase1_gates._encode_windows`` + ``phase1_gates._rollout``, i.e.
  the exact functions the T1/T2/T3 gates call.

Latents, not the decoded mp4: the mp4 goes through an x264 encode, so comparing
files would test ffmpeg determinism rather than the model.

Two windows minimum (the default), because a one-window run never exercises the
window-to-window latent carryover -- the stateful path a refactor is most likely to
break.

    conda run -n ltx python -m scripts.prune.method_parity --model 2.5 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import decord
import torch

from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from scripts.prune import artifacts, model_registry, phase1_gates, preflight, prompt_cache, provenance, refine_core, refine_task
from scripts.prune.model_registry import REPO_ROOT, WORKSPACE_ROOT

decord.bridge.set_bridge("torch")

DTYPE = torch.bfloat16
CORPUS_DIR = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"
SCRIPT = REPO_ROOT / "scripts" / "vae_refine_sliding_window.py"


def pick_clip(geometry: refine_core.WindowGeometry, windows: int, name: str | None = None) -> Path:
    """First corpus clip long enough for *windows* whole windows of this geometry.

    Alphabetically the first clip is a 113-frame ``canonical_rotation`` render at 24
    fps; most of the corpus is 121 frames at 30 fps. Both are long enough for the
    default 25/9 geometry, but the check is explicit so a shorter corpus fails here
    rather than inside the subprocess.
    """
    need = geometry.window_frames + (windows - 1) * geometry.stride_frames
    candidates = sorted(CORPUS_DIR.glob("*/source.mp4"))
    if name:
        candidates = [p for p in candidates if p.parent.name == name] or candidates
    for source in candidates:
        try:
            if len(decord.VideoReader(str(source))) >= need:
                return source
        except Exception:
            continue
    raise SystemExit(f"No clip under {CORPUS_DIR} has the >= {need} frames needed for {windows} window(s).")


def run_reference(model, clip: Path, geometry: refine_core.WindowGeometry, windows: int, seed: int,
                  gpu_id: int, out_dir: Path) -> list[torch.Tensor]:
    """Run the refine script as a user would, and read back its cached latents."""
    command = [
        sys.executable, str(SCRIPT),
        "--video", str(clip),
        "--output-dir", str(out_dir),
        "--model", model.key,
        "--sampler", "euler",
        "--k-step", refine_task.K_STEP,
        "--gpu-id", str(gpu_id),
        "--seed", str(seed),
        "--window-frames", str(geometry.window_frames),
        "--overlap-frames", str(geometry.overlap_frames),
        "--max-windows", str(windows),
    ]
    print(f"[parity] reference: {' '.join(command)}", flush=True)
    proc = subprocess.run(command, cwd=str(REPO_ROOT), check=False)  # returncode handled below
    if proc.returncode != 0:
        raise SystemExit(f"reference run failed with exit code {proc.returncode}")
    cached = sorted((out_dir / "latent_cache").glob("win_*_latent.pt"))
    if not cached:
        raise SystemExit(f"No cached latents under {out_dir / 'latent_cache'}; did the reference run produce any?")
    return [torch.load(p, map_location="cpu") for p in cached]


def run_harness(model, clip: Path, geometry: refine_core.WindowGeometry, windows: int, seed: int,
                device: torch.device) -> list[torch.Tensor]:
    """Run the gates' own rollout -- the same two functions ``phase1_gates.main`` calls."""
    plan = geometry.plan(len(decord.VideoReader(str(clip))))[:windows]
    latents, _, fps = phase1_gates._encode_windows(model, clip, plan, device)
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    denoiser = SimpleDenoiser(context, None)
    sigmas = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)
    stage = DiffusionStage.from_checkpoint(
        model.paths.transformer(), DTYPE, device,
        model_configurator=LTXVideoOnlyModelConfigurator, scale_factors=model.scale_factors,
    )
    with torch.no_grad(), stage._transformer_ctx() as transformer:
        refined = phase1_gates._rollout(transformer, denoiser, latents, geometry, sigmas, fps,
                                        seed=seed, device=device)
    del stage
    torch.cuda.empty_cache()
    # The refine script persists its cache as bf16 on CPU; match that before comparing
    # so the gate tests the model rather than a dtype round-trip.
    return [latent.to(torch.bfloat16).cpu() for latent in refined]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--clip", default=None, help="Corpus clip directory name; default: first long enough.")
    ap.add_argument("--windows", type=int, default=2, help="Windows to compare; >= 2 exercises the carryover.")
    ap.add_argument("--window-frames", type=int, default=refine_task.WINDOW_FRAMES)
    ap.add_argument("--overlap-frames", type=int, default=refine_task.OVERLAP_FRAMES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-runs", action="store_true", help="Do not delete the reference run directory on PASS.")
    args = ap.parse_args()

    if args.windows < 2:
        raise SystemExit("--windows must be >= 2: a single window never exercises the latent carryover.")

    model = preflight.check(args.model, sampler="euler", gpu_id=args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    geometry = refine_core.WindowGeometry(
        window_frames=args.window_frames, overlap_frames=args.overlap_frames, scale_factors=model.scale_factors
    )
    clip = pick_clip(geometry, args.windows, args.clip)
    out_root = artifacts.run_dir(model.key, "method-parity", script="method_parity", argv=sys.argv[1:])
    reference_dir = out_root / "reference"
    print(f"[parity] clip {clip.parent.name}, geometry {geometry.as_dict()}", flush=True)

    reference = run_reference(model, clip, geometry, args.windows, args.seed, args.gpu_id, reference_dir)
    harness = run_harness(model, clip, geometry, args.windows, args.seed, device)

    rows, all_pass = [], len(reference) == len(harness) and bool(reference)
    for i, (a, b) in enumerate(zip(reference, harness, strict=False)):
        equal = a.shape == b.shape and torch.equal(a, b)
        all_pass &= equal
        rows.append({
            "window": i,
            "shape": list(a.shape),
            "bit_exact": bool(equal),
            "max_abs_diff": float((a.float() - b.float()).abs().max()) if a.shape == b.shape else None,
        })
        detail = "" if equal else f" (max |d| = {rows[-1]['max_abs_diff']})"
        print(f"  window {i}: {'BIT-EXACT' if equal else 'DIFFERS'}{detail}")
    if len(reference) != len(harness):
        print(f"  window COUNT differs: reference {len(reference)} vs harness {len(harness)}")

    report = {
        "provenance": provenance.stamp(model, device, script="method_parity"),
        "clip": str(clip),
        "geometry": geometry.as_dict(),
        "windows_compared": len(rows),
        "seed": args.seed,
        "k_step": refine_task.K_STEP,
        "reference_script": str(SCRIPT.relative_to(REPO_ROOT)),
        "harness": "scripts.prune.phase1_gates._encode_windows + _rollout",
        "windows": rows,
        "pass": bool(all_pass),
    }
    (out_root / "method_parity.json").write_text(json.dumps(report, indent=2))
    latest = artifacts.gate(model.key, "method_parity")
    latest.write_text(json.dumps(report, indent=2))

    if all_pass and not args.keep_runs:
        import shutil

        shutil.rmtree(reference_dir, ignore_errors=True)

    print(f"Wrote {latest}. Overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
