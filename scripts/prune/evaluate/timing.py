"""Warmed, synchronized transformer timing and ABA/BAB bracket statistics.

`measure_arm` loads one checkpoint, checks one forward against its saved
prediction (maximum absolute difference at most 0.02), then times repeated
synchronized forwards in milliseconds. `timed` times an already resident model
in seconds. `bracket` summarizes three arms in an ABA or BAB order:
speedup is mean A over mean B, and drift is the outer-arm change over the first arm.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, provenance, session


def measure_arm(
    current: session.Session,
    path: Path,
    expected: Path,
    grid: common.ClipGrid,
    modality: Modality,
    c0: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    label: str,
) -> dict:
    times = []
    with current.transformer(path) as transformer:
        resident_parameters = sum(p.numel() for p in transformer.parameters())

        def forward() -> torch.Tensor:
            prediction, _ = transformer(video=modality, audio=None, perturbations=None)
            return common.with_clean_prefix(prediction, c0)

        with torch.no_grad():
            output = forward()
            torch.cuda.synchronize(current.device)
            recorded = torch.load(expected, map_location="cpu", weights_only=True)
            measured = grid.unpatchify_block(output, grid.latent_frames).cpu()
            if not torch.isfinite(recorded).all() or not torch.isfinite(measured).all():
                raise ValueError(f"{label}: nonfinite recorded or measured output")
            difference = (recorded.float() - measured.float()).abs()
            if not torch.isfinite(difference).all():
                raise ValueError(f"{label}: nonfinite forward difference")
            max_abs = float(difference.max())
            rel_l2 = float(
                torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(recorded.float()).clamp_min(1e-8)
            )
            if max_abs > 0.02:
                raise ValueError(f"{label} forward differs from saved rollout: max_abs={max_abs:.5f}")
            del output, recorded, measured, difference
            torch.cuda.reset_peak_memory_stats(current.device)
            for index in range(warmup + repeats):
                torch.cuda.synchronize(current.device)
                start = time.perf_counter()
                output = forward()
                torch.cuda.synchronize(current.device)
                if index >= warmup:
                    times.append((time.perf_counter() - start) * 1000)
                del output
            peak = torch.cuda.max_memory_allocated(current.device)
    return {
        "arm": label,
        "checkpoint": str(path.resolve()),
        "checkpoint_fingerprint": provenance.checkpoint_fingerprint(path),
        "saved_rollout_max_abs": max_abs,
        "saved_rollout_relative_l2": rel_l2,
        "times_ms": times,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_cuda_memory_bytes": peak,
        "resident_video_parameters": resident_parameters,
    }


def timed(model, video, step, device):
    if step:
        step.masks()
    for _ in range(2):
        ltx_adapter.velocity_forward(model, video)
    times = []
    for _ in range(7):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        ltx_adapter.velocity_forward(model, video)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
    return dict(
        seconds=times, median_seconds=statistics.median(times), peak_allocated=torch.cuda.max_memory_allocated(device)
    )


def bracket(rows, order, key):
    """A/B means over an ABA or BAB order; drift compares the two outer arms."""
    if len(rows) != 3 or order not in ("ABA", "BAB"):
        raise ValueError("a timing bracket needs three arms in ABA or BAB order")
    a = statistics.mean(r[key] for r, k in zip(rows, order, strict=True) if k == "A")
    b = statistics.mean(r[key] for r, k in zip(rows, order, strict=True) if k == "B")
    outside = abs(rows[2][key] - rows[0][key])
    return dict(a=a, b=b, speedup=a / b, saving=a - b, outside_drift=outside, drift_fraction=outside / rows[0][key])
