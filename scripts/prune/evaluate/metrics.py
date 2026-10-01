"""Pixel metrics and synchronized comparison videos shared with avatar probes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    """Normalize BCTHW/FCHW/BT HWC decoder outputs to float BCHW frames."""
    x = x.float()
    if x.ndim == 5:  # B,C,T,H,W
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
    elif x.ndim == 4 and x.shape[-1] in (1, 3, 4):  # F,H,W,C
        x = x.permute(0, 3, 1, 2)
    if x.ndim != 4:
        raise ValueError(f"expected video frames, got {tuple(x.shape)}")
    return x


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    pred, target = _as_bchw(pred), _as_bchw(target)
    if pred.shape != target.shape:
        raise ValueError(f"PSNR shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}")
    mse = (pred - target).square().mean()
    return float("inf") if mse.item() == 0 else float(10 * torch.log10(torch.tensor(data_range**2, device=mse.device) / mse))


def ssim_global(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """A dependency-free global SSIM; LPIPS remains optional due to its weights."""
    x, y = _as_bchw(pred), _as_bchw(target)
    if x.shape != y.shape:
        raise ValueError(f"SSIM shapes differ: {tuple(x.shape)} vs {tuple(y.shape)}")
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mux, muy = x.mean((-1, -2), keepdim=True), y.mean((-1, -2), keepdim=True)
    vx = ((x - mux) ** 2).mean((-1, -2), keepdim=True)
    vy = ((y - muy) ** 2).mean((-1, -2), keepdim=True)
    cov = ((x - mux) * (y - muy)).mean((-1, -2), keepdim=True)
    return float((((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux.square() + muy.square() + c1) * (vx + vy + c2))).mean())


def t3_video(source: torch.Tensor, teacher: torch.Tensor, candidate: torch.Tensor, output: str | Path, *, fps: float = 24.0) -> Path:
    """Save a synchronized source | reference | candidate MP4.

    The three streams are frame-aligned and concatenated horizontally. ffmpeg is
    used directly so output is an ordinary portable H.264 MP4 rather than an
    environment-specific tensor dump.
    """
    streams = [_as_bchw(v) for v in (source, teacher, candidate)]
    frames = min(v.shape[0] for v in streams)
    c, h, w = streams[0].shape[1:]
    if c < 3 or any(v.shape[1:] != (c, h, w) for v in streams):
        raise ValueError("T3 video streams must share an RGB-compatible C,H,W shape")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w * 3}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(frames):
            frame = torch.cat([v[i, :3] for v in streams], dim=-1).clamp(0, 1)
            process.stdin.write((frame.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()).tobytes())
        process.stdin.close()
        stderr = process.stderr.read()
        code = process.wait()
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    if code:
        raise RuntimeError(f"ffmpeg failed writing {path}: {stderr.decode(errors='replace')}")
    return path
