"""Mergeable per-sigma native gate state, with exact draw coverage and atomic resume.

Histogram counts are exact integers. Moments use FP64. Original head axes stay
(block, head); frames are local F18 positions. Checkpoints bind arrays and draw
receipts in one atomic NPZ so a crash cannot advance a cursor without statistics.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_arrays(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"]))
        arrays = {key: bundle[key] for key in bundle.files if key != "metadata"}
    return arrays, metadata


class Accumulator:
    def __init__(self, sigmas: int = 10, identity: str = ""):
        self.sigmas = sigmas
        self.identity = identity
        self.arrays: dict[str, np.ndarray] = {}
        self.receipts: list[dict] = []
        self.counts = np.zeros(sigmas, dtype=np.int64)
        self.tokens = np.zeros(sigmas, dtype=np.int64)
        self.clean_tokens = np.zeros(sigmas, dtype=np.int64)

    def add(self, index: int, sigma_index: int, values: dict[str, np.ndarray], receipt: dict) -> None:
        if index in {row["draw_index"] for row in self.receipts}:
            raise ValueError("duplicate accumulated draw")
        if (
            not 0 <= sigma_index < self.sigmas
            or receipt["draw_index"] != index
            or receipt["sigma_index"] != sigma_index
        ):
            raise ValueError("draw identity or sigma group differs")
        per_frame, frames = receipt["tokens_per_frame"], receipt["frames"]
        if frames != 18:
            raise ValueError("statistics require the actual F18 training crop")
        if self.receipts and per_frame != self.receipts[0]["tokens_per_frame"]:
            raise ValueError("unequal draw token counts require explicit per-draw normalization")
        histogram = values["histogram"]
        if histogram.dtype != np.int64 or not np.all(histogram.sum(-1) == (frames - 1) * per_frame):
            raise ValueError("per-head histogram count conservation failed")
        if self.arrays and set(values) != set(self.arrays):
            raise ValueError("native observation fields differ")
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("nonfinite native sufficient statistics")
        if self.arrays and any(
            self.arrays[key].shape[1:] != value.shape or self.arrays[key].dtype != value.dtype
            for key, value in values.items()
        ):
            raise ValueError("native statistic geometry changed")
        for key, value in values.items():
            if key not in self.arrays:
                fill = np.inf if key.endswith("minimum") else -np.inf if key.endswith("maximum") else 0
                self.arrays[key] = np.full((self.sigmas, *value.shape), fill, dtype=value.dtype)
            target = self.arrays[key][sigma_index]
            operation = np.minimum if key.endswith("minimum") else np.maximum if key.endswith("maximum") else np.add
            operation(target, value, out=target)
        self.counts[sigma_index] += 1
        self.tokens[sigma_index] += (frames - 1) * per_frame
        self.clean_tokens[sigma_index] += per_frame
        self.receipts.append(receipt)

    def save(self, path: Path) -> None:
        metadata = {"identity": self.identity, "sigmas": self.sigmas, "receipts": self.receipts}
        save_arrays(
            path,
            self.arrays | {"samples": self.counts, "tokens": self.tokens, "clean_tokens": self.clean_tokens},
            metadata,
        )

    @classmethod
    def load(cls, path: Path, identity: str) -> Accumulator:
        arrays, metadata = load_arrays(path)
        if metadata["identity"] != identity:
            raise ValueError("accumulator belongs to another frozen study")
        result = cls(metadata["sigmas"], identity)
        result.counts, result.tokens, result.clean_tokens = (
            arrays.pop(k) for k in ("samples", "tokens", "clean_tokens")
        )
        result.arrays, result.receipts = arrays, metadata["receipts"]
        if len({r["draw_index"] for r in result.receipts}) != len(result.receipts) or result.counts.sum() != len(
            result.receipts
        ):
            raise ValueError("saved accumulator has duplicate or missing counts")
        result.validate_counts()
        return result

    def validate_counts(self) -> None:
        if not self.arrays:
            return
        expected = self.tokens[:, None, None]
        if not np.all(self.arrays["histogram"].sum(-1) == expected):
            raise ValueError("accumulated histogram count conservation failed")
        actual = np.bincount([r["sigma_index"] for r in self.receipts], minlength=self.sigmas)
        if not np.array_equal(actual, self.counts):
            raise ValueError("receipt sigma counts differ from statistics")

    def merge(self, other: Accumulator) -> None:
        if self.identity != other.identity or self.sigmas != other.sigmas:
            raise ValueError("cannot merge different studies")
        if {r["draw_index"] for r in self.receipts} & {r["draw_index"] for r in other.receipts}:
            raise ValueError("duplicate draw in worker merge")
        if not self.arrays:
            self.arrays = {k: v.copy() for k, v in other.arrays.items()}
        else:
            if self.arrays.keys() != other.arrays.keys():
                raise ValueError("worker statistic fields differ")
            for key, value in other.arrays.items():
                target = self.arrays[key]
                if key.endswith("minimum"):
                    np.minimum(target, value, out=target)
                elif key.endswith("maximum"):
                    np.maximum(target, value, out=target)
                else:
                    target += value
        self.counts += other.counts
        self.tokens += other.tokens
        self.clean_tokens += other.clean_tokens
        self.receipts.extend(other.receipts)
        self.receipts.sort(key=lambda r: r["draw_index"])
        self.validate_counts()

    def require_prefix(self, start: int, stop: int) -> None:
        self.require_indices(list(range(start, stop)))

    def require_indices(self, expected: list[int]) -> None:
        if sorted(r["draw_index"] for r in self.receipts) != sorted(expected):
            raise ValueError("milestone lacks the exact complete ordered draw prefix")
        self.validate_counts()


def means(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return arrays["gate_sum"] / arrays["tokens"][:, None, None]


def probabilities(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return arrays["histogram"] / arrays["tokens"][:, None, None, None]


def distribution_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape or first.shape[-1] != 2048:
        raise ValueError("distribution bins differ")
    return np.abs(np.cumsum(first - second, axis=-1)).sum(-1) / 1024
