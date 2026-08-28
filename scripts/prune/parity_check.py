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

from scripts.prune import artifacts, corpus, model_registry, preflight, provenance, refine_core
from scripts.prune.model_registry import REPO_ROOT

BASELINE_COPY = REPO_ROOT / "scripts" / "_parity_baseline_vae_refine.py"
SCRIPT = REPO_ROOT / "scripts" / "vae_refine_sliding_window.py"


REL_SCRIPT = "scripts/vae_refine_sliding_window.py"
# The pre-refactor script located its components itself; the ported one imports the
# registry. That import is the marker used to tell the two apart in git history.
REFACTOR_MARKER = "scripts.prune"


def _blob(rev: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{rev}:{REL_SCRIPT}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def find_baseline_rev() -> str:
    """Newest revision of the refine script that predates the registry port.

    Not simply ``HEAD``: this workspace auto-commits, so by the time the gate runs,
    ``HEAD`` can already *contain* the refactor being checked -- which would make the
    comparison trivially pass by comparing the new script against itself. Walk the
    file's history instead and take the newest blob without the registry import.
    """
    revs = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "--format=%H", "--", REL_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for rev in revs:
        if REFACTOR_MARKER not in _blob(rev):
            return rev
    raise SystemExit(
        f"No revision of {REL_SCRIPT} in this history predates the registry port "
        f"(searched {len(revs)} commits for one without `{REFACTOR_MARKER}`). "
        "Pass --rev explicitly."
    )


def _write_baseline_copy(rev: str) -> str:
    """Extract ``<rev>:scripts/vae_refine_sliding_window.py`` next to the current one."""
    content = _blob(rev)
    if REFACTOR_MARKER in content:
        raise SystemExit(
            f"--rev {rev} already contains the registry refactor (`{REFACTOR_MARKER}`), so comparing "
            "against it would compare the new script with itself. Omit --rev to auto-detect the "
            "pre-refactor revision."
        )
    BASELINE_COPY.write_text(content)
    return content


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
    ap.add_argument(
        "--rev", default=None,
        help="Git revision holding the pre-refactor script. Default: auto-detect the newest "
        "revision of the script without the registry import (this workspace auto-commits, so "
        "HEAD is not a safe default).",
    )
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--max-windows", type=int, default=2)
    # The corpus tops out at 145 frames, so the script's default 121-frame window fits
    # only once -- and a single-window run never exercises the window-to-window latent
    # carryover, which is exactly the stateful path a refactor is most likely to break.
    # A 57-frame window (57 % 8 == 1) with the standard 17-frame overlap gives a stride
    # of 40, so two windows fit in 97 frames and the carryover IS compared.
    ap.add_argument("--window-frames", type=int, default=57)
    ap.add_argument("--overlap-frames", type=int, default=17)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-runs", action="store_true", help="Do not delete the two run directories.")
    args = ap.parse_args()

    if args.model != "2.3":
        raise SystemExit(
            f"--model {args.model} has no pre-refactor baseline: the HEAD script hardcodes the 2.3 "
            "monolith and cannot load a split pack. Run this gate on 2.3."
        )

    model = preflight.check(args.model, gpu_id=args.gpu_id)
    rev = args.rev or find_baseline_rev()
    geometry = refine_core.WindowGeometry(
        window_frames=args.window_frames,
        overlap_frames=args.overlap_frames,
        scale_factors=model.scale_factors,
    )
    clip = args.clip or corpus.pick_clip(geometry, args.max_windows)
    print(f"[parity] baseline rev {rev[:12]}, clip {clip.parent.name}", flush=True)
    out_root = artifacts.run_dir(model.key, "parity", script="parity_check", argv=sys.argv[1:])
    before_dir, after_dir = out_root / "pre_refactor", out_root / "registry"

    python = sys.executable
    try:
        _write_baseline_copy(rev)
        common = [
            "--video", str(clip),
            "--k-step", "k2",
            "--gpu-id", str(args.gpu_id),
            "--seed", str(args.seed),
            "--max-windows", str(args.max_windows),
            "--window-frames", str(args.window_frames),
            "--overlap-frames", str(args.overlap_frames),
        ]
        _run([python, str(BASELINE_COPY), *common, "--output-dir", str(before_dir)], f"pre-refactor ({rev[:12]})")
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

    report = {
        "provenance": provenance.stamp(model, torch.device(f"cuda:{args.gpu_id}"), script="parity_check"),
        "baseline_rev": rev,
        "clip": str(clip),
        "max_windows": args.max_windows,
        "window_frames": args.window_frames,
        "overlap_frames": args.overlap_frames,
        "windows": rows,
        "pass": all_pass,
    }
    (out_root / "parity_check.json").write_text(json.dumps(report, indent=2))
    artifacts.gate(model.key, "parity_check").write_text(json.dumps(report, indent=2))

    if not args.keep_runs and all_pass:
        import shutil

        shutil.rmtree(before_dir, ignore_errors=True)
        shutil.rmtree(after_dir, ignore_errors=True)

    print(f"Wrote {out_root / 'parity_check.json'}. Overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
