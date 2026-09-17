"""Store a coverage mask as a **losslessly** encoded grayscale MP4, not a raw array.

The masks this pipeline persists are single-channel uint8 coverage grids at 256**2 over a
whole clip. As `.npy` that is 9.8 MB for a 150-frame clip and 14.8 MB for a 225-frame one --
and the corpus has thousands of views, so the raw form costs tens of GB for data that is
almost entirely flat. Measured on real corpus alphas, lossless x264 in grayscale gives
**~42x** compression with a **bit-exact** round trip:

| | 150 frames | 225 frames |
|---|---|---|
| raw `.npy` | 9.83 MB | 14.75 MB |
| lossless gray MP4 | 0.232 MB | 0.361 MB |

The comparison that makes the point: the dataset's OWN `mask.mp4`, at 4096x3000, is 330-570 KB
-- a mask 180x larger in pixels than our 256**2 grid was, in a fraction of the space.

**Lossless, and not negotiable.** Masks in this project are already one generation of lossy
video away from the truth (the capture matte is a hard threshold off h264, the plan's risk 8),
and plan SS1.7's rule is that the pipeline's job is not to add a *second* generation. Lossy
settings were measured and rejected: crf 12 is only 1.7x smaller than lossless but puts a max
error of 58 and a mean error of 3.2 on exactly the soft silhouette edge that the composite's
smooth boundary and the latent coverage both come from. 42x for free beats 72x for a corrupted
edge.

**Why MP4 and not `.npz`/FFV1.** MP4 + x264 reads through the same OpenCV path every other
video in this pipeline uses (no new dependency, no new failure mode), plays in any viewer for
a review pass, and -- unlike a compressed array -- decodes frame by frame without
materializing the whole clip.

Until 2026-09-15 this module was transcribed into both of the package's former trees, because
they lived in different conda envs and could not import each other. Consolidating the package
removed that seam; what the pin test guarded -- that the ENCODE ARGS never drift to a lossy
setting, which would be invisible in every downstream number -- is now guarded directly by
``test_mask_video``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

# Lossless, grayscale, in MP4. `-crf 0` is x264's lossless mode; `-pix_fmt gray` keeps one
# plane rather than padding to 4:2:0 (which would also be lossless for gray input, but three
# times the planes). Changing ANY of this changes what is on disk -- see the module docstring.
MASK_ENCODE_ARGS = ("-c:v", "libx264", "-preset", "slow", "-crf", "0", "-pix_fmt", "gray")
MASK_FPS = 30


def write_mask_video(grid: np.ndarray, output: Path, fps: int = MASK_FPS) -> Path:
    """Encode ``[N, H, W]`` uint8 coverage to a lossless gray MP4, atomically.

    Atomic for the same reason every other artifact here is: a reader must never observe a
    half-written mask, and a killed ffmpeg must not leave one that looks complete.
    """
    if grid.ndim != 3 or grid.dtype != np.uint8:
        raise ValueError(f"expected [N, H, W] uint8 coverage, got {grid.shape} {grid.dtype}")
    frames, height, width = grid.shape
    if frames < 1:
        raise ValueError("cannot encode an empty mask")
    temp_path = output.with_name(f".{output.stem}.tmp.{os.getpid()}{output.suffix}")
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", *MASK_ENCODE_ARGS, str(temp_path),
        ],
        stdin=subprocess.PIPE,
    )
    process.communicate(grid.tobytes())
    if process.returncode != 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed encoding {output} (exit {process.returncode})")
    temp_path.replace(output)
    return output


def read_mask_video(path: Path) -> np.ndarray:
    """Decode a mask MP4 back to ``[N, H, W]`` uint8 -- the array that was written, exactly.

    One plane is taken from the decoded BGR frame rather than converting: the encode was
    grayscale, so all three channels are the same plane and a color conversion would only
    round-trip it through arithmetic.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open mask video {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame[..., 0])
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"{path}: decoded no frames")
    return np.stack(frames)


def read_mask(path_without_suffix: Path) -> np.ndarray:
    """Read a stored mask, preferring the MP4 and falling back to a legacy ``.npy``.

    The fallback is what lets the renders that predate this format keep working untouched
    instead of being rebuilt: a `.npy` on disk is the same array, just 42x larger. New writes
    are always MP4.
    """
    video = path_without_suffix.with_suffix(".mp4")
    if video.is_file():
        return read_mask_video(video)
    legacy = path_without_suffix.with_suffix(".npy")
    if legacy.is_file():
        return np.load(legacy)
    raise FileNotFoundError(f"no mask at {video} or {legacy}")


def mask_exists(path_without_suffix: Path) -> bool:
    """True if either form is present -- the resumability check's question."""
    return path_without_suffix.with_suffix(".mp4").is_file() or path_without_suffix.with_suffix(".npy").is_file()


def _pixel_range(latent_index: int, time_scale: int) -> tuple[int, int]:
    """Pixel frames represented by one frame of a continuous causal-VAE encode."""
    if latent_index == 0:
        return 0, 1
    start = 1 + (latent_index - 1) * time_scale
    return start, start + time_scale


def pool_to_latent_grid(
    mask: np.ndarray,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    time_scale: int,
) -> torch.Tensor:
    """Pool a stored uint8 mask to ``[F, H, W]`` latent coverage on demand.

    The MP4 remains the only persisted representation.  Training and measurement call this
    reader when they need latent-cell coverage; persisting the derived grid duplicated the
    same information and coupled masks to one latent geometry.
    """
    if mask.ndim != 3 or mask.dtype != np.uint8:
        raise ValueError(f"expected [N, H, W] uint8 coverage, got {mask.shape} {mask.dtype}")
    spatial = np.stack(
        [cv2.resize(frame, (latent_width, latent_height), interpolation=cv2.INTER_AREA) for frame in mask]
    ).astype(np.float32) / 255.0
    pooled = []
    for index in range(latent_frames):
        lo, hi = _pixel_range(index, time_scale)
        if hi > len(spatial):
            raise ValueError(
                f"mask has {len(spatial)} pixel frames, too few for latent frame {index} "
                f"of {latent_frames} (needs frames [{lo}, {hi}))"
            )
        pooled.append(spatial[lo:hi].mean(axis=0))
    return torch.from_numpy(np.stack(pooled)).to(torch.float16).contiguous()


def read_latent_masks(
    view: Path,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    time_scale: int,
) -> dict[str, torch.Tensor]:
    """Read the two canonical mask MP4s and derive transient latent-grid coverage."""
    return {
        "render_alpha": pool_to_latent_grid(
            read_mask(view / "argavatar_alpha"),
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            time_scale=time_scale,
        ),
        "capture_mask": pool_to_latent_grid(
            read_mask(view / "capture_mask_crop"),
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            time_scale=time_scale,
        ),
    }
