"""Phase 1 evaluation metrics: T0 latent, T1 pixels, T2 rollouts, T3 grids.

The model lifecycle deliberately stays outside this module.  It accepts tensors
or callbacks so both full and hook-masked/pruned models exercise identical
metric code.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image

from scripts.prune.score.losses import rel_l2


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


def t0(records: Iterable[tuple[torch.Tensor, torch.Tensor, object]], masks: Iterable[torch.Tensor] | None = None) -> dict:
    """T0 over a set of (prediction, target, state) triples.

    ``masks`` overrides the scoring token set per record -- pass
    ``chunk_states.chunk_token_mask(...)`` to score the AR chunk alone rather than
    every fresh token (which also includes the index-0 keyframe; see
    ``chunk_states.chunk_token_mask``).
    """
    records = list(records)
    masks = list(masks) if masks is not None else [None] * len(records)
    if len(masks) != len(records):
        raise ValueError(f"{len(masks)} masks for {len(records)} records")
    values = [float(rel_l2(pred, target, state, mask)) for (pred, target, state), mask in zip(records, masks)]
    return {"count": len(values), "rel_l2_mean": sum(values) / max(len(values), 1), "rel_l2_max": max(values, default=0.0)}


def t1(pred_pixels: torch.Tensor, teacher_pixels: torch.Tensor, source_pixels: torch.Tensor | None = None,
       lpips_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None) -> dict:
    """Single-chunk decoded comparison, including source when supplied."""
    out = {"psnr_vs_teacher": psnr(pred_pixels, teacher_pixels), "ssim_vs_teacher": ssim_global(pred_pixels, teacher_pixels)}
    if source_pixels is not None:
        out.update({"psnr_vs_source": psnr(pred_pixels, source_pixels), "ssim_vs_source": ssim_global(pred_pixels, source_pixels)})
    if lpips_fn is not None:
        out["lpips_vs_teacher"] = float(lpips_fn(_as_bchw(pred_pixels), _as_bchw(teacher_pixels)).mean())
    return out


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    return float(((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).square().sum().clamp_min(1e-12))


def t2(rollout: Iterable[dict], output: str | Path | None = None) -> dict:
    """Summarize a sequential AR rollout.

    Every row must contain ``chunk``, ``pred`` and ``teacher`` pixel tensors;
    optional ``brightness``/``saturation`` values allow callers to use their
    preferred colour convention.  This interface makes it impossible to call
    T2 on independent chunks by accident: the caller owns the sequential state.
    """
    rows = []
    for row in rollout:
        p = psnr(row["pred"], row["teacher"])
        pred = _as_bchw(row["pred"])
        rows.append({"chunk": int(row["chunk"]), "psnr_vs_teacher": p,
                     "brightness": float(row.get("brightness", pred.mean())),
                     "saturation": float(row.get("saturation", (pred.max(1).values - pred.min(1).values).mean()))})
    slope = _linear_slope([r["chunk"] for r in rows], [r["psnr_vs_teacher"] for r in rows])
    result = {"chunks": len(rows), "psnr_vs_teacher": rows, "psnr_slope_db_per_100_chunks": slope * 100,
              "brightness_slope_per_100_chunks": _linear_slope([r["chunk"] for r in rows], [r["brightness"] for r in rows]) * 100,
              "saturation_slope_per_100_chunks": _linear_slope([r["chunk"] for r in rows], [r["saturation"] for r in rows]) * 100}
    if output is not None:
        Path(output).write_text(json.dumps(result, indent=2))
    return result


def t3_grid(rows: Iterable[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]], output: str | Path,
            max_frames: int = 4) -> Path:
    """Write a compact source / teacher / candidate grid for human review (T3).

    ``rows`` are ``(clip_name, source, teacher, candidate)``.  The filename is
    intentionally the only label: keeping the image pixels unannotated avoids
    font/platform variability in review artifacts.

    Clips may differ in resolution -- this corpus mixes 1024x1024, 768x768,
    704x1280 and 1056x640 -- so rows are *not* required to share a shape. Each row
    is composed at its own clip's size and top-left-anchored into a canvas sized to
    the widest row; unused area stays black. Requiring one global shape instead
    made every multi-clip call raise, which is how a completed 3-clip GPU
    validation run got thrown away at the very last step.
    """
    grid_rows: list[list[torch.Tensor]] = []
    for _, source, teacher, candidate in rows:
        source, teacher, candidate = _as_bchw(source), _as_bchw(teacher), _as_bchw(candidate)
        count = min(max_frames, source.shape[0], teacher.shape[0], candidate.shape[0])
        for i in range(count):
            triple = [source[i], teacher[i], candidate[i]]
            if any(t.shape[-2:] != triple[0].shape[-2:] for t in triple):
                raise ValueError(
                    "source/teacher/candidate frames of one clip must share H,W; got "
                    f"{[tuple(t.shape) for t in triple]}"
                )
            grid_rows.append(triple)
    if not grid_rows:
        raise ValueError("T3 needs at least one source/teacher/candidate frame")

    heights = [r[0].shape[-2] for r in grid_rows]
    width = max(r[0].shape[-1] for r in grid_rows) * 3
    canvas = torch.zeros(3, sum(heights), width, dtype=torch.uint8)
    y = 0
    for triple, h in zip(grid_rows, heights):
        w = triple[0].shape[-1]
        for c, tile in enumerate(triple):
            canvas[:, y : y + h, c * w : (c + 1) * w] = (tile[:3].clamp(0, 1) * 255).round().byte()
        y += h
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.permute(1, 2, 0).numpy()).save(path)
    return path


def t3_video(source: torch.Tensor, teacher: torch.Tensor, candidate: torch.Tensor, output: str | Path, *, fps: float = 24.0) -> Path:
    """Save a source | teacher | candidate MP4 for Phase 1 visual review.

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
