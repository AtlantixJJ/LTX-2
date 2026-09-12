"""Compare two vae_refine_sliding_window runs that differ ONLY in latent slot 0.

Run A (``--keyframe-source encoded``) gives each window its own genuine single-pixel causal
keyframe. Run B (``--keyframe-source master-slice``) puts the master latent's regular
``time_scale``-pixel frame there instead. The checkpoint sets
``use_keyframes_abs_pos_embedding``, so the model applies a learned KEYFRAME embedding to
slot 0 either way -- under B it is applied to a token that is not one.

Slot 0's own pixels never reach the stitched output: the stitcher drops each window's first
``overlap_frames``. So this measures slot 0 purely as *conditioning* (through denoising, and
through the causal decode of the later slots).

Frames are re-stitched from each run's uint8 ``window_cache/`` decodes, never from
``decode_full.mp4`` -- that file is h264 crf-18 muxed from per-flush segments, and both its
GOP structure and its segment joins land at multiples of the window stride, i.e. a codec
artifact at exactly the phase being measured. Same rule as
``expr/boundary_jump/FINDINGS.md``.

    conda run -n ltx python3 scripts/keyframe_ab_metrics.py \
        --run-a expr/keyframe_ab/encoded --run-b expr/keyframe_ab/master_slice \
        --source expr/boundary_jump/source_30s.mp4 --out expr/keyframe_ab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decord
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

decord.bridge.set_bridge("torch")


def load_plan(run: Path) -> dict:
    return json.loads((run / "window_plan.json").read_text())


def stitch(run: Path, overlap_frames: int) -> torch.Tensor:
    """Re-stitch a run's uint8 per-window decodes exactly as Stitcher.add does."""
    paths = sorted((run / "window_cache").glob("win_*.pt"))
    if not paths:
        raise SystemExit(f"no window_cache/win_*.pt under {run}")
    parts = []
    for index, path in enumerate(paths):
        frames = torch.load(path, map_location="cpu")  # (F, H, W, C) uint8
        parts.append(frames if index == 0 else frames[overlap_frames:])
    return torch.cat(parts, dim=0)


def source_frames(video: Path, count: int) -> torch.Tensor:
    """The source clip, center-cropped to a multiple of 32 exactly as read_pixel_window does."""
    vr = decord.VideoReader(str(video))
    frames = vr.get_batch(range(count))
    _, h, w, _ = frames.shape
    if h % 32:
        top = (h - (h // 32) * 32) // 2
        frames = frames[:, top : top + (h // 32) * 32, :, :]
    if w % 32:
        left = (w - (w // 32) * 32) // 2
        frames = frames[:, :, left : left + (w // 32) * 32, :]
    return frames.to(torch.uint8)


def psnr_per_frame(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-frame PSNR in dB between two uint8 (F, H, W, C) tensors."""
    x = a.float().div(255.0)
    y = b.float().div(255.0)
    mse = ((x - y) ** 2).flatten(1).mean(dim=1).clamp(min=1e-12)
    return (10.0 * torch.log10(1.0 / mse)).numpy()


def phase_profile(values: np.ndarray, stride: int, offset: int) -> np.ndarray:
    """Mean of ``values`` grouped by position within each window's authored block."""
    idx = (np.arange(len(values)) - offset) % stride
    return np.array([values[idx == p].mean() for p in range(stride)])


def strip_figure(
    source: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    frame_indices: list[int],
    out_path: Path,
    amplify: int,
    labels: tuple[str, str],
) -> None:
    """Source / A / B / amplified |A-B| across a few frames."""
    rows = ["source", f"A: {labels[0]}", f"B: {labels[1]}", f"|A-B| x{amplify}"]
    fig, axes = plt.subplots(len(rows), len(frame_indices), figsize=(3.0 * len(frame_indices), 3.0 * len(rows)))
    for col, f in enumerate(frame_indices):
        diff = (a[f].float() - b[f].float()).abs().mul(amplify).clamp(0, 255).to(torch.uint8)
        for row, img in enumerate((source[f], a[f], b[f], diff)):
            ax = axes[row, col]
            ax.imshow(img.numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=11)
            if row == 0:
                ax.set_title(f"frame {f}", fontsize=11)
    fig.suptitle(f"A: {labels[0]}   vs   B: {labels[1]}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def curve_figure(
    psnr_a: np.ndarray, psnr_b: np.ndarray, psnr_ab: np.ndarray, stride: int, offset: int, out_path: Path,
    labels: tuple[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

    axes[0].plot(psnr_a, lw=0.9, label=f"A {labels[0]} ({psnr_a.mean():.3f} dB)")
    axes[0].plot(psnr_b, lw=0.9, label=f"B {labels[1]} ({psnr_b.mean():.3f} dB)")
    axes[0].set_title("PSNR vs source, per frame")
    axes[0].set_xlabel("output frame")
    axes[0].set_ylabel("dB")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(psnr_a - psnr_b, lw=0.9, color="tab:red")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_title(f"A - B, per frame (mean {np.mean(psnr_a - psnr_b):+.4f} dB)")
    axes[1].set_xlabel("output frame")
    axes[1].set_ylabel("dB")
    axes[1].grid(alpha=0.3)

    axes[2].plot(phase_profile(psnr_a, stride, offset), marker="o", ms=3, label="A")
    axes[2].plot(phase_profile(psnr_b, stride, offset), marker="o", ms=3, label="B")
    axes[2].set_title(f"PSNR by position in the authored block (stride {stride})")
    axes[2].set_xlabel("frames since the block start (slot 0 sits earlier)")
    axes[2].set_ylabel("dB")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"A vs B agree with each other at {psnr_ab.mean():.2f} dB", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", type=Path, required=True, help="--keyframe-source encoded run dir")
    ap.add_argument("--run-b", type=Path, required=True, help="--keyframe-source master-slice run dir")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--amplify", type=int, default=10, help="Gain on the |A-B| row of the strip figure.")
    ap.add_argument("--boundary-index", type=int, default=14, help="Which window boundary the strip centres on.")
    ap.add_argument("--label-a", default="encoded keyframe")
    ap.add_argument("--label-b", default="8-frame latent")
    args = ap.parse_args()

    plan_a, plan_b = load_plan(args.run_a), load_plan(args.run_b)
    if plan_a["geometry"] != plan_b["geometry"]:
        raise SystemExit("the two runs used different window geometry; they are not comparable")
    overlap = plan_a["geometry"]["overlap_frames"]
    stride = plan_a["geometry"]["stride_frames"]

    # Both runs must share a byte-identical master latent, or slot 0 is not the only difference.
    ma = torch.load(args.run_a / "latent_cache" / "master_latent.pt", map_location="cpu")
    mb = torch.load(args.run_b / "latent_cache" / "master_latent.pt", map_location="cpu")
    master_identical = torch.equal(ma["latent"], mb["latent"])
    keyframes_identical = torch.equal(ma["keyframes"], mb["keyframes"])

    a, b = stitch(args.run_a, overlap), stitch(args.run_b, overlap)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    source = source_frames(args.source, n)

    psnr_a = psnr_per_frame(a, source)
    psnr_b = psnr_per_frame(b, source)
    psnr_ab = psnr_per_frame(a, b)
    identical = torch.equal(a, b)

    args.out.mkdir(parents=True, exist_ok=True)
    labels = (args.label_a, args.label_b)
    curve_figure(psnr_a, psnr_b, psnr_ab, stride, overlap, args.out / "keyframe_ab_curves.png", labels)

    boundary = overlap + stride * args.boundary_index
    picks = [f for f in (boundary - 1, boundary, boundary + 1, boundary + 8, boundary + 16) if f < n]
    strip_figure(source, a, b, picks, args.out / "keyframe_ab_strip.png", args.amplify, labels)

    report = {
        "labels": {"a": args.label_a, "b": args.label_b},
        "frames": int(n),
        "geometry": plan_a["geometry"],
        "master_latent_identical": bool(master_identical),
        "window_keyframes_identical": bool(keyframes_identical),
        "outputs_identical": bool(identical),
        "psnr_vs_source": {
            "a_encoded_keyframe": float(psnr_a.mean()),
            "b_master_slice": float(psnr_b.mean()),
            "a_minus_b": float(psnr_a.mean() - psnr_b.mean()),
            "a_minus_b_per_frame_std": float((psnr_a - psnr_b).std()),
            "frames_where_a_better": int((psnr_a > psnr_b).sum()),
        },
        "psnr_a_vs_b": {
            "mean": float(psnr_ab.mean()),
            "min": float(psnr_ab.min()),
            "max": float(psnr_ab.max()),
            "argmin_frame": int(psnr_ab.argmin()),
        },
        "phase_profile_a": phase_profile(psnr_a, stride, overlap).tolist(),
        "phase_profile_b": phase_profile(psnr_b, stride, overlap).tolist(),
        "figures": ["keyframe_ab_curves.png", "keyframe_ab_strip.png"],
    }
    (args.out / "keyframe_ab_metrics.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if not k.startswith("phase_profile")}, indent=2))
    print(f"\nwrote {args.out / 'keyframe_ab_metrics.json'} and 2 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
