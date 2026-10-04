"""Deterministic spatial sampling of native generated-frame tokens."""

from __future__ import annotations

import hashlib
import struct

import torch

SAMPLERS = ("stride", "balanced_2d_midpoint_v1")
HASH_ENCODING = "uint64_le_v1"


def index_sha256(indices: torch.Tensor) -> str:
    """Hash integer values in an explicit platform-independent byte order."""
    if indices.ndim != 1 or indices.dtype != torch.int64 or torch.any(indices < 0):
        raise ValueError("indices must be one-dimensional nonnegative int64 values")
    digest = hashlib.sha256()
    for index in indices.detach().cpu().tolist():
        digest.update(struct.pack("<Q", index))
    return digest.hexdigest()


def _validate(tokens: int, height: int, width: int, stride: int, sampler: str) -> None:
    if any(type(value) is not int or value < 1 for value in (tokens, height, width, stride)):
        raise ValueError("token count, latent dimensions and stride must be positive integers")
    if tokens % (height * width) or tokens <= height * width:
        raise ValueError("token grid must contain a clean frame and complete generated frames")
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown token sampler: {sampler!r}")


def _spatial_indices(height: int, width: int, stride: int, sampler: str) -> list[int]:
    if sampler == "stride":
        return list(range(0, height * width, stride))
    budget = (height * width + stride - 1) // stride
    # Choose an aspect-appropriate row count that can fit every point uniquely.
    minimum, maximum = (budget + width - 1) // width, min(height, budget)
    rows = min(range(minimum, maximum + 1), key=lambda count: (abs(count * count * width - budget * height), count))
    base, remainder = divmod(budget, rows)
    spatial = []
    for band in range(rows):
        row = (2 * band + 1) * height // (2 * rows)
        columns = base + (band < remainder)
        for column_band in range(columns):
            column = (2 * column_band + 1) * width // (2 * columns)
            spatial.append(row * width + column)
    return spatial


def sample_indices(
    tokens: int, height: int, width: int, stride: int, device: torch.device,
    *, sampler: str = "stride",
) -> torch.Tensor:
    """Use the same equal-budget spatial points on every frame after clean c0."""
    _validate(tokens, height, width, stride, sampler)
    spatial = torch.tensor(_spatial_indices(height, width, stride, sampler), device=device, dtype=torch.int64)
    per_frame = height * width
    starts = torch.arange(per_frame, tokens, per_frame, device=device)
    return (starts[:, None] + spatial[None]).reshape(-1)


def sampling_record(
    indices: torch.Tensor, *, tokens: int, height: int, width: int, stride: int, sampler: str,
) -> dict:
    """Record exact geometry, coverage and index identity without implying full-grid coverage."""
    expected = sample_indices(tokens, height, width, stride, torch.device("cpu"), sampler=sampler)
    if indices.dtype != torch.int64 or not torch.equal(indices.detach().cpu(), expected):
        raise ValueError("sample indices differ from the declared deterministic sampler")
    spatial = torch.tensor(_spatial_indices(height, width, stride, sampler), dtype=torch.int64)
    return {
        "sampler": sampler, "algorithm": "flattened_stride_v1" if sampler == "stride" else sampler,
        "latent_frames": tokens // (height * width), "latent_height": height, "latent_width": width,
        "spatial_stride": stride, "tokens_per_generated_frame": spatial.numel(), "sample_tokens": expected.numel(),
        "sampled_rows": sorted({int(index) // width for index in spatial}),
        "sampled_columns": sorted({int(index) % width for index in spatial}),
        "hash_encoding": HASH_ENCODING, "spatial_indices_sha256": index_sha256(spatial),
        "token_indices_sha256": index_sha256(expected), "clean_frame_included": False,
    }
