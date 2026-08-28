"""Run provenance stamped into every scripts/prune/* artifact.

plans/2026-08-26-refiner-head-ffn-pruning.md §4: "stamp the model key **and the
transformer file hash** into provenance, and refuse to load a mask whose key does
not match." Scores, masks and pruned checkpoints are per generation and head index
spaces are not comparable across them, so an artifact that does not say which
checkpoint produced it is not usable evidence.

Why a *fingerprint* and not a full-file sha256: the 2.5 transformer is 42 GB, so
hashing it end-to-end costs ~90 s per script start-up -- enough that it would get
skipped. ``checkpoint_fingerprint`` instead hashes the safetensors header (every
tensor's name, dtype, shape and byte offsets -- so any structural change, including
a pruned export, changes it) together with the file size and three 1 MiB samples
drawn at fixed fractions of the data section (so a same-shape re-train or a
fine-tune changes it too). It is a collision-resistant *identifier*, not an
integrity check against a deliberate forgery, which is all provenance needs here.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import struct
import subprocess
import time
from pathlib import Path

import torch

_SAMPLE_BYTES = 1 << 20  # 1 MiB per sampled region
_SAMPLE_FRACTIONS = (0.25, 0.5, 0.75)

REPO_ROOT = Path(__file__).resolve().parents[2]


def checkpoint_fingerprint(path: str | Path) -> str:
    """Stable short identifier for a safetensors checkpoint (see module docstring)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(struct.pack("<Q", size))
    with p.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        h.update(f.read(header_len))
        data_start = 8 + header_len
        data_len = max(size - data_start, 0)
        for frac in _SAMPLE_FRACTIONS:
            offset = data_start + int(data_len * frac)
            offset = min(offset, max(size - _SAMPLE_BYTES, data_start))
            f.seek(offset)
            h.update(f.read(_SAMPLE_BYTES))
    return h.hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    """Full sha256 -- for small files only (manifests, prompt caches, source clips)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        rev = out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True, timeout=20
        )
        if dirty.stdout.strip():
            rev += "-dirty"
    except Exception:
        pass
    return rev


def run_id(prefix: str = "") -> str:
    """Timestamped, collision-safe run id -- the ``<run-id>`` level of
    ``expr/refiner_prune/<key>/<run-id>/`` (§12).

    The timestamp alone has 1-second resolution, and parallel sweeps call this at the
    *end* of their work, so a per-GPU launch stagger does not separate them: two jobs
    that happen to finish in the same second get the same directory and one silently
    overwrites the other's report. That happened twice -- once in the first published
    sweep (two schedules, one file, both gates loaded the survivor) and again the first
    time this sweep was re-run with a 7-second stagger. The PID suffix removes the
    failure mode instead of narrowing its window; ids stay sortable because the
    timestamp still leads.
    """
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid():06d}"
    return f"{stamp}-{prefix}" if prefix else stamp


def stamp(model, device: torch.device | None = None, **extra: object) -> dict:
    """Provenance block to embed in every artifact JSON.

    ``model`` is a :class:`model_registry.RefinerModel`; passed untyped to keep
    this module import-light (it is imported by the exporter too).
    """
    block: dict = {
        "model_key": model.key,
        "model_version": list(model.version),
        "transformer_path": model.paths.transformer(),
        "transformer_fingerprint": checkpoint_fingerprint(model.paths.transformer()),
        "video_vae_path": model.paths.video_vae_path,
        "sampler": model.stepper_kind,
        "scale_factors": list(model.scale_factors),
        "scale_factors_source": model.scale_factors_source,
        "git_rev": _git_rev(),
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if device is not None and device.type == "cuda":
        block["gpu_index"] = device.index
        block["gpu_name"] = torch.cuda.get_device_name(device)
    block.update(extra)
    return block
