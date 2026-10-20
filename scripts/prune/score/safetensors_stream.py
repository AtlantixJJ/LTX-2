"""Publish transformed safetensors with one tensor and a bounded copy buffer.

The official serializer supplies header ordering from empty tensors. The writer
reserves that header, streams and hashes payloads, then binds an optional envelope
without rereading the source. See doc/score/safetensors_stream.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save

from scripts.prune.core import records


def inspect(source):
    """Return shape-only tensors and native metadata without reading payloads."""
    with safe_open(source, framework="pt", device="cpu"):
        with Path(source).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        metadata = dict(header.get("__metadata__", {}))
        # Empty tensors let the library resolve all supported dtype names.
        from safetensors.torch import _getdtype  # noqa: PLC0415

        tensors = {
            key: torch.empty(entry["shape"], dtype=_getdtype(entry["dtype"]), device="meta")
            for key, entry in header.items()
            if key != "__metadata__"
        }
    return tensors, metadata


def _header(tensors, metadata):
    # This delegates dtype/key ordering and metadata encoding to the installed
    # serializer without allocating any source-sized tensor.
    empty = {key: torch.empty(0, dtype=value.dtype) for key, value in tensors.items()}
    raw = save(empty, metadata=metadata)
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    # Rust metadata maps have randomized iteration order. Keep the declared
    # metadata order so publication and replay are deterministic.
    if "__metadata__" in header:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        value = tensors[key]
        length = value.numel() * value.element_size()
        entry.update(shape=list(value.shape), data_offsets=[offset, offset + length])
        offset += length
    return header


def _encoded(header):
    value = json.dumps(header, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return value + b" " * (-len(value) % 8)


def _stat(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def write(
    source,
    output,
    tensors,
    metadata,
    *,
    transforms=None,
    inventory=False,
    inventory_dtype=True,
    on_inventory=None,
    chunk_bytes=8 << 20,
):
    """Create one checkpoint atomically; never hold its full state in host RAM.

    `tensors` describes final shapes/dtypes on meta. A transform receives one CPU
    source tensor. `on_inventory` can replace metadata after payload hashes exist;
    its encoded header must have the reserved length (for example a 64-byte hash).
    """
    source, output = Path(source), Path(output)
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if source.resolve() == output.resolve() or (output.exists() and output.samefile(source)):
        raise ValueError("refusing to overwrite source checkpoint")
    before = _stat(source)
    with source.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        original = json.loads(stream.read(length))
    start = 8 + length
    transforms = transforms or {}
    if set(tensors) != set(original) - {"__metadata__"} or set(transforms) - set(tensors):
        raise ValueError("streamed tensor inventory differs from source")
    header = _header(tensors, metadata)
    reserved = _encoded(header)
    result = {}

    def publish(temporary):
        with (
            source.open("rb") as reader,
            temporary.open("wb") as writer,
        ):
            writer.write(struct.pack("<Q", len(reserved)))
            writer.write(reserved)
            for key, entry in header.items():
                if key == "__metadata__":
                    continue
                digest = hashlib.sha256()
                if key in transforms:
                    # Closing each mapping releases touched source pages rather
                    # than accumulating their RSS across the complete export.
                    with safe_open(source, framework="pt", device="cpu") as handle:
                        value = transforms[key](handle.get_tensor(key)).detach().cpu().contiguous()
                    if value.shape != tensors[key].shape or value.dtype != tensors[key].dtype:
                        raise ValueError(f"streamed tensor shape/dtype differs: {key}")
                    payload = memoryview(value.reshape(-1).view(torch.uint8).numpy())
                    for offset in range(0, payload.nbytes, chunk_bytes):
                        chunk = payload[offset : offset + chunk_bytes]
                        writer.write(chunk)
                        if inventory:
                            digest.update(chunk)
                    chunk = None
                    del payload, value
                else:
                    old = original[key]
                    if entry["shape"] != old["shape"] or entry["dtype"] != old["dtype"]:
                        raise ValueError(f"missing shape transform: {key}")
                    lo, hi = old["data_offsets"]
                    reader.seek(start + lo)
                    remaining = hi - lo
                    while remaining:
                        chunk = reader.read(min(remaining, chunk_bytes))
                        if not chunk:
                            raise OSError("source ended during streamed tensor copy")
                        writer.write(chunk)
                        if inventory:
                            digest.update(chunk)
                        remaining -= len(chunk)
                if inventory:
                    dtype = str(tensors[key].dtype)
                    result[key] = dict(
                        shape=entry["shape"],
                        **({"dtype": dtype} if inventory_dtype else {}),
                        sha256=records.digest(dict(shape=entry["shape"], dtype=dtype, bytes=digest.hexdigest())),
                    )
            if _stat(source) != before:
                raise ValueError("source checkpoint changed during streaming")
            if on_inventory is not None:
                final = _encoded(_header(tensors, on_inventory(result)))
                if len(final) != len(reserved):
                    raise ValueError("final streamed header changed reserved length")
                writer.seek(8)
                writer.write(final)
            writer.flush()
            os.fsync(writer.fileno())

    records.publish(output, publish)
    return result
