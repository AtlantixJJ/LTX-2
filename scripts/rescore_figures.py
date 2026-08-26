#!/usr/bin/env python3
"""Figures for the region-split rescore (see rescore_fg_bg.py).

Replaces the whole-frame figures from the original sweep, which plotted PSNR numbers dominated
by the flat white matte rather than by the subject.

    conda run -n ltx python3 scripts/rescore_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESCORE = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine" / "rescore" / "rescore.json"

REGIONS = [
    ("psnr_all", "whole frame", "#9aa0a6", "--"),
    ("psnr_fg", "foreground (subject)", "#d1495b", "-"),
    ("psnr_halo", "halo (floater band)", "#edae49", "-"),
    ("psnr_bg", "background (matte)", "#00798c", "-"),
]


def fig_inflation(data: dict, out: Path) -> None:
    """Whole-frame PSNR minus foreground PSNR, against how much of the frame is subject."""
    pv = data["per_video"]
    frac = np.array([r["fg_fraction"] for r in pv]) * 100
    infl = np.array([r["variants"]["k0"]["psnr_all"] - r["variants"]["k0"]["psnr_fg"] for r in pv])
    corr = float(np.corrcoef(frac, infl)[0, 1])

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    classes = sorted({r["video_class"] for r in pv})
    colors = {"crop": "#d1495b", "original": "#00798c", "rotation": "#edae49"}
    for cls in classes:
        m = np.array([r["video_class"] == cls for r in pv])
        ax.scatter(frac[m], infl[m], s=42, alpha=0.85, label=cls, color=colors.get(cls, "#666"), edgecolor="white")
    z = np.polyfit(frac, infl, 1)
    xs = np.linspace(frac.min(), frac.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#333", lw=1.2, ls="--", zorder=0)
    ax.set_xlabel("subject share of frame (%)")
    ax.set_ylabel("whole-frame PSNR $-$ foreground PSNR  (dB)")
    ax.set_title(f"How much of the headline PSNR is the white matte\nr = {corr:.3f}, n = {len(pv)}")
    ax.grid(alpha=0.25)
    ax.legend(title="video class", frameon=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_steps(data: dict, out: Path) -> None:
    """PSNR vs refinement step count, one line per region."""
    agg = data["aggregate"]["by_variant"]
    ks = [k for k in ("k0", "k1", "k2", "k3", "k4", "k8") if k in agg]
    x = [int(k[1:]) for k in ks]

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for key, label, color, ls in REGIONS:
        y = [agg[k][key]["mean"] for k in ks]
        lo = [agg[k][key]["min"] for k in ks]
        hi = [agg[k][key]["max"] for k in ks]
        ax.plot(x, y, marker="o", color=color, ls=ls, lw=2, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.10, lw=0)

    ax.axvline(3, color="#333", lw=1, ls=":", zorder=0)
    ax.annotate("production\nstage 2", xy=(3, ax.get_ylim()[1]), xytext=(3.15, ax.get_ylim()[1] - 6),
                fontsize=8, color="#333")
    ax.set_xlabel("refinement steps k  (tail of the distilled sigma schedule)")
    ax.set_ylabel("PSNR vs source (dB)")
    ax.set_title("Refinement degrades every region monotonically\n"
                 f"n = {agg['k1']['psnr_fg']['n']} videos; band = min-max")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_by_dataset(data: dict, out: Path) -> None:
    """VAE ceiling per data source, whole-frame vs foreground."""
    pv = data["per_video"]
    ds = sorted({r["dataset_prefix"] for r in pv})
    fg = [np.mean([r["variants"]["k0"]["psnr_fg"] for r in pv if r["dataset_prefix"] == d]) for d in ds]
    allf = [np.mean([r["variants"]["k0"]["psnr_all"] for r in pv if r["dataset_prefix"] == d]) for d in ds]
    n = [sum(1 for r in pv if r["dataset_prefix"] == d) for d in ds]

    idx = np.arange(len(ds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    ax.bar(idx - w / 2, allf, w, label="whole frame (as originally reported)", color="#9aa0a6")
    ax.bar(idx + w / 2, fg, w, label="foreground (subject only)", color="#d1495b")
    for i, (a, f) in enumerate(zip(allf, fg)):
        ax.text(i - w / 2, a + 0.4, f"{a:.1f}", ha="center", fontsize=8, color="#555")
        ax.text(i + w / 2, f + 0.4, f"{f:.1f}", ha="center", fontsize=8, color="#d1495b")
    ax.axhline(30.0, color="#00798c", lw=1.2, ls="--")
    ax.annotate("~30 dB: prior real-human-capture baseline", xy=(len(ds) - 0.5, 30.2),
                ha="right", fontsize=8, color="#00798c")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{d}\n(n={c})" for d, c in zip(ds, n)], fontsize=8)
    ax.set_ylabel("VAE round-trip PSNR (dB)")
    ax.set_title("VAE ceiling by data source: the whole-frame number is the matte")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", type=Path, default=DEFAULT_RESCORE)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.rescore.read_text())
    out_dir = args.out_dir or args.rescore.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_inflation(data, out_dir / "psnr_inflation_vs_subject_share.png")
    fig_steps(data, out_dir / "psnr_vs_steps_by_region.png")
    fig_by_dataset(data, out_dir / "vae_ceiling_by_dataset_region.png")
    print("wrote 3 figures to", out_dir)


if __name__ == "__main__":
    main()
