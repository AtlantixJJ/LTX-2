"""Plot decoded frame transitions and capture error for the D1 visual review.

Both curves are raw pixel comparisons. They mix motion, appearance and alignment;
use the saved boundary crops to identify which effect caused a peak.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames)


def measure(video: np.ndarray, capture: np.ndarray) -> dict:
    if video.shape != capture.shape:
        raise ValueError(f"video shape {video.shape} differs from capture {capture.shape}")
    _, height, width, _ = video.shape
    # Fixed central body region, with white background removed per frame pair.
    roi = np.zeros((height, width), dtype=bool)
    roi[int(.16 * height):int(.90 * height), int(.31 * width):int(.70 * width)] = True
    foreground = np.min(np.minimum(video, capture), axis=-1) < .94
    foreground &= roi[None]
    capture_error = np.sum(np.abs(video - capture).mean(-1) * foreground, axis=(1, 2)) / np.maximum(
        foreground.sum((1, 2)), 1
    )
    pair_foreground = (foreground[1:] | foreground[:-1])
    transition = np.sum(np.abs(video[1:] - video[:-1]).mean(-1) * pair_foreground, axis=(1, 2)) / np.maximum(
        pair_foreground.sum((1, 2)), 1
    )
    return {"capture_rgb_mae": capture_error.tolist(), "adjacent_rgb_mae": [None, *transition.tolist()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--video", action="append", type=Path, required=True)
    parser.add_argument("--decode-manifest", action="append", type=Path, required=True,
                        help="Decode manifest with the actual block boundaries for each video; repeat if needed.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = read_video(args.capture)
    records = {entry["video"]: entry for path in args.decode_manifest for entry in json.loads(path.read_text())}
    if any(path.name not in records for path in args.video):
        raise ValueError("every plotted video must have an entry in --decode-manifest")
    data = {path.stem: measure(read_video(path), capture) for path in args.video}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    figure, axes = plt.subplots(1 + len(args.video), 1, figsize=(13, 3 + 1.8 * len(args.video)), sharex=True)
    for path in args.video:
        label, item = path.stem, data[path.stem]
        axes[0].plot(item["capture_rgb_mae"], label=label, linewidth=1.3)
    for index, path in enumerate(args.video, start=1):
        ax = axes[index]
        ax.plot(data[path.stem]["adjacent_rgb_mae"], linewidth=1.3)
        for boundary in records[path.name]["boundaries"]:
            ax.axvline(boundary, color="tab:red", linewidth=.8)
        ax.set_ylabel(path.stem.replace("0008_01_", ""), fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].set_ylabel("RGB MAE vs capture")
    axes[-1].set_xlabel("Pixel frame; red lines start new blocks in that row")
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle("Foreground pixels: capture error (top), adjacent RGB change (rows below)")
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
