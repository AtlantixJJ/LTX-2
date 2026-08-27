"""Phase 0 gate: the registry-driven refine script must reproduce the
pre-refactor script bit-for-bit on 2.3 (plan §5).

Porting ``vae_refine_sliding_window.py`` onto the model registry touched how the
transformer, VAE and text encoder are located. That is exactly the kind of change
that is "obviously" behaviour-preserving right up until a different default path,
a different configurator or a re-ordered build changes one bit of the output --
after which every existing ``expr/sam3dgs_vae_refine/`` number silently stops
being comparable to new ones.

So: check out ``HEAD``'s copy of the script, run both on the same clip with the
same seed, and compare the cached **refined latents** (``latent_cache/*.pt``) with
``torch.equal``. Latents, not the decoded mp4 -- the mp4 goes through an x264
encode, so comparing files would test ffmpeg determinism instead of the model.

The baseline copy is written into ``scripts/`` (not a temp dir) because the script
resolves ``REPO_ROOT`` as ``Path(__file__).resolve().parents[1]``; running it from
anywhere else would point its ``sys.path`` and default checkpoint paths at the
wrong tree. It is removed again in a ``finally``.

    conda run -n ltx python -m scripts.prune.parity_check --gpu-id 2

Only meaningful for ``--model 2.3``: 2.5 has no pre-refactor behaviour to match
(the old script could not load a split pack at all).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.prune import model_registry, preflight, provenance
from scripts.prune.model_registry import REPO_ROOT, WORKSPACE_ROOT

CORPUS_DIR = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"
BASELINE_COPY = REPO_ROOT / "scripts" / "_parity_baseline_vae_refine.py"
SCRIPT = REPO_ROOT / "scripts" / "vae_refine_sliding_window.py"


def _write_baseline_copy(rev: str) -> str:
    """Extract ``<rev>:scripts/vae_refine_sliding_window.py`` next to the current one."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{rev}:scripts/vae_refine_sliding_window.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    BASELINE_COPY.write_text(out.stdout)
    return out.stdout


def _run(cmd: list[str], label: str) -> None:
    print(f"[parity] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"{label} run failed with exit code {proc.returncode}")


def _latents(run_dir: Path) -> dict[str, torch.Tensor]:
    return {p.name: torch.load(p, map_location="cpu") for p in sorted((run_dir / "latent_cache").glob("*_latent.pt"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.3", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--rev", default="HEAD", help="Git revision holding the pre-refactor script.")
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--max-windows", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-runs", action="store_true", help="Do not delete the two run directories.")
    args = ap.parse_args()

    if args.model != "2.3":
        raise SystemExit(
            f"--model {args.model} has no pre-refactor baseline: the HEAD script hardcodes the 2.3 "
            "monolith and cannot load a split pack. Run this gate on 2.3."
        )

    model = preflight.check(args.model, gpu_id=args.gpu_id)
    clip = args.clip or next(iter(sorted(CORPUS_DIR.glob("*/source.mp4"))))
    out_root = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / provenance.run_id("parity")
    before_dir, after_dir = out_root / "pre_refactor", out_root / "registry"

    python = sys.executable
    try:
        _write_baseline_copy(args.rev)
        common = [
            "--video", str(clip),
            "--k-step", "k2",
            "--gpu-id", str(args.gpu_id),
            "--seed", str(args.seed),
            "--max-windows", str(args.max_windows),
        ]
        _run([python, str(BASELINE_COPY), *common, "--output-dir", str(before_dir)], f"pre-refactor ({args.rev})")
        # The pre-refactor script had no --model/--video-only: it always built the
        # audio-video configurator against the 2.3 monolith. Match that exactly --
        # video-only is gated separately by video_only_check.py, and folding it in
        # here would make a parity failure ambiguous between the two changes.
        _run(
            [
                python, str(SCRIPT), *common,
                "--model", "2.3", "--sampler", "euler", "--no-video-only",
                "--output-dir", str(after_dir),
            ],
            "registry-driven",
        )
    finally:
        BASELINE_COPY.unlink(missing_ok=True)

    before, after = _latents(before_dir), _latents(after_dir)
    rows, all_pass = [], bool(before) and before.keys() == after.keys()
    if not before:
        raise SystemExit(f"No cached latents under {before_dir / 'latent_cache'}; did the baseline run produce any?")
    for name in sorted(before.keys() & after.keys()):
        a, b = before[name], after[name]
        equal = a.shape == b.shape and torch.equal(a, b)
        all_pass &= equal
        rows.append(
            {
                "window": name,
                "shape": list(a.shape),
                "bit_exact": equal,
                "max_abs_diff": float((a.float() - b.float()).abs().max()) if a.shape == b.shape else None,
            }
        )
        print(f"  {name}: {'BIT-EXACT' if equal else 'DIFFERS'}")

    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": provenance.stamp(model, torch.device(f"cuda:{args.gpu_id}"), script="parity_check"),
        "baseline_rev": args.rev,
        "clip": str(clip),
        "max_windows": args.max_windows,
        "windows": rows,
        "pass": all_pass,
    }
    (out_root / "parity_check.json").write_text(json.dumps(report, indent=2))
    (WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "parity_check.json").write_text(json.dumps(report, indent=2))

    if not args.keep_runs and all_pass:
        import shutil

        shutil.rmtree(before_dir, ignore_errors=True)
        shutil.rmtree(after_dir, ignore_errors=True)

    print(f"Wrote {out_root / 'parity_check.json'}. Overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
