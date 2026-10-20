"""Content digests and atomic publication of immutable evidence records.

`digest` hashes canonical JSON: sorted keys, compact separators, no NaN.
`tensor_hash` binds shape, dtype and the raw bytes of a CPU copy.
`unchanged_state_hash` hashes a saved payload while it ignores the
deliberately mutable `identity` and `lineage` fields.
`publish` creates a file once. A replay with equal bytes passes; a replay
with other bytes fails. `write_json` replaces a mutable status record after
an fsync. `read_json` reads one JSON file. All writers create parent folders.
These encodings are pinned by saved receipts; do not change them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.prune.core.provenance import file_sha256


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return digest(
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    )


def unchanged_state_hash(state):
    """Hash saved payloads without the deliberately changed identity and lineage."""

    def encode(value):
        if isinstance(value, torch.Tensor):
            return {"tensor": tensor_hash(value)}
        if isinstance(value, np.ndarray):
            return {
                "numpy": str(value.dtype),
                "shape": list(value.shape),
                "bytes": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [encode(v) for v in value]}
        return value

    return digest(encode({k: v for k, v in state.items() if k not in ("identity", "lineage")}))


def publish(path, writer):
    """Atomic create. Writers never replace completed evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".publish-", suffix=path.suffix)
    os.close(fd)
    try:
        writer(Path(name))
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if file_sha256(name) != file_sha256(path):
                raise ValueError(f"completed artifact differs: {path}") from None
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def publish_json(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return publish(path, lambda p: p.write_text(raw))


def publish_tensor(path, value):
    value = value.detach().cpu().contiguous()
    if Path(path).exists():
        if tensor_hash(torch.load(path, map_location="cpu", weights_only=True)) != tensor_hash(value):
            raise ValueError(f"completed tensor differs: {path}")
        return Path(path)

    # A stable stream filename avoids archive-name-dependent torch.save bytes.
    def write(p):
        with p.open("wb") as stream:
            torch.save(value, stream)

    return publish(path, write)


def publish_payload(path, payload):
    """Create a torch payload once; an existing file must hold the same unchanged state."""
    path = Path(path)
    if path.exists():
        if unchanged_state_hash(torch.load(path, map_location="cpu", weights_only=False)) != unchanged_state_hash(
            payload
        ):
            raise ValueError(f"replayed artifact differs: {path}")
        return path

    def write(p):
        with p.open("wb") as stream:
            torch.save(payload, stream)

    return publish(path, write)


def write_json(path, value, *, sort_keys=True):
    """Replace a mutable record durably; the temporary name is unique per process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())
