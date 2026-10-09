"""Measure encoded and RGB outputs; see doc/metrics.md.

These arithmetic and saved-mask helpers never open transformer, text or VAE
weights. An explicit caller may supply an LPIPS model for perceptual scores.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import torch


def encoded_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Full-frame fp32 x0 MSE, with unchanged c0 retained in the denominator."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("encoded metrics require equal B,C,F,H,W tensors")
    error = (prediction.float() - target.float()).square()
    return {
        "definition": "mean((prediction-capture)^2) in fp32, including c0",
        "mse": float(error.mean()),
        "per_frame_mse": error.mean(dim=(0, 1, 3, 4)).tolist(),
        "frames": prediction.shape[2],
    }



def rgb_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Aligned unquantized FCHW float RGB measurements, without a synthetic mask."""
    _check_rgb_pair(prediction, target)
    error = (prediction.float() - target.float()).square()
    mse = error.mean(dim=(1, 2, 3)).tolist()
    total_mse = float(error.mean())
    return {
        "definition": "float RGB MSE and PSNR=-10*log10(MSE), before presentation compression",
        "per_frame_mse": mse,
        "per_frame_psnr": [None if x == 0 else -10 * math.log10(x) for x in mse],
        "exact_match": [x == 0 for x in mse],
        "frames": len(mse),
        "mse": total_mse,
        "psnr": None if total_mse == 0 else -10 * math.log10(total_mse),
        "all_exact_match": total_mse == 0,
    }



def _check_rgb_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if (
        prediction.shape != target.shape
        or prediction.ndim != 4
        or prediction.shape[1] != 3
        or any(size < 1 for size in prediction.shape)
    ):
        raise ValueError("RGB measurement requires matching nonempty F,3,H,W tensors")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1
        for value in (prediction, target)
    ):
        raise ValueError("RGB measurement requires finite floating pixels in [0,1]")



def masked_rgb_transition_steps(video, mask):  # noqa: ANN001, ANN201 -- historical NumPy RGB arrays
    """Measure absolute frame changes in the supplied union-foreground mask."""
    import numpy as np  # noqa: PLC0415 -- historical metric arithmetic

    if (not isinstance(video, np.ndarray) or video.ndim != 4 or video.shape[0] < 2
            or video.shape[-1] != 3 or min(video.shape[1:3]) < 1
            or not np.issubdtype(video.dtype, np.floating) or not np.isfinite(video).all()
            or video.min() < 0 or video.max() > 1):
        raise ValueError("transition measurement requires finite nonempty F,H,W,3 RGB in [0,1]")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != video.shape[:3]:
        raise ValueError("transition measurement requires an aligned boolean foreground mask")
    value = video.astype(np.float32, copy=False) if video.dtype.itemsize < 4 else video
    delta = np.abs(value[1:] - value[:-1]).mean(-1)
    selected = mask[1:] | mask[:-1]
    return (delta * selected).sum((1, 2)) / selected.sum((1, 2)).clip(1)



def subject_mask(path: Path, frames: int, height: int, width: int) -> torch.Tensor | None:
    """Replay the study's optional two-cell mask dilation for RGB QA only."""
    from scripts.onestep_avatar.corpus import mask_video  # noqa: PLC0415 -- CPU lossless mask reader

    if any(type(value) is not int or value < 1 for value in (frames, height, width)):
        raise ValueError("subject mask requires positive frame count and dimensions")
    if not path.is_file():
        return None
    raw = torch.from_numpy(mask_video.read_mask_video(path))
    if raw.dtype != torch.uint8 or raw.ndim != 3 or raw.shape[0] < frames or any(size < 1 for size in raw.shape):
        raise ValueError("subject mask does not cover the requested RGB frames")
    grid = raw[:frames].float()[:, None] / 255
    grid = torch.nn.functional.max_pool2d(grid, 5, stride=1, padding=2)
    return torch.nn.functional.interpolate(grid, size=(height, width), mode="nearest")[:, 0] > 0.5



def subject_rgb_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """Measure aligned supplied subject pixels before presentation compression."""
    _check_rgb_pair(prediction, target)
    if mask.dtype != torch.bool or mask.shape != (prediction.shape[0], *prediction.shape[2:]) or not mask.any():
        raise ValueError("subject mask must be nonempty boolean F,H,W aligned with RGB")
    error = (prediction.float() - target.float()).square().mean(dim=1)
    mse = float(error[mask.to(error.device)].mean())
    return {
        "mse": mse,
        "psnr": None if mse == 0 else -10 * math.log10(mse),
        "exact_match": mse == 0,
        "selected_pixels": int(mask.sum()),
    }



def _lpips_batches(
    model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor, device: torch.device, batch: int
) -> Iterator[torch.Tensor]:
    """Share input/model gates while retaining native per-batch score tensors."""
    _check_rgb_pair(prediction, target)
    if type(batch) is not int or batch < 1:
        raise ValueError("perceptual batch size must be a positive integer")
    for start in range(0, len(prediction), batch):
        left = prediction[start : start + batch].float().to(device) * 2 - 1
        right = target[start : start + batch].float().to(device) * 2 - 1
        scores = model(left, right)
        if not isinstance(scores, torch.Tensor) or scores.numel() != len(left) or not torch.isfinite(scores).all():
            raise ValueError("perceptual model must return one finite score per frame")
        yield scores



@torch.no_grad()
def lpips_frame_scores(
    model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor, device: torch.device, batch: int = 16
) -> list[float]:
    """Return checked aligned scores; the caller chooses c0 exclusion and averaging."""
    return [value for scores in _lpips_batches(model, prediction, target, device, batch)
            for value in scores.flatten().tolist()]



@torch.no_grad()
def lpips_distance(
    model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor, device: torch.device, batch: int = 8
) -> float:
    """Preserve the scalar path's per-batch native sums and frame weighting."""
    total = 0.0
    for scores in _lpips_batches(model, prediction, target, device, batch):
        total += float(scores.sum())
    return total / len(prediction)

