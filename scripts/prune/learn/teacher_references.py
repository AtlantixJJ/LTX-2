"""Single immutable teacher evidence owner with bounded disk and CPU caches."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.core.records import digest, publish_json, publish_tensor, tensor_hash

REQUIRED = {
    "components",
    "input",
    "epsilon",
    "context",
    "geometry",
    "sigma",
    "code",
    "numerics",
    "software",
    "architecture",
}


def reference_key(identity):
    if set(identity) != REQUIRED or any(v is None for v in identity.values()):
        raise ValueError("teacher reference requires complete input/component/runtime identity, without student state")
    return digest(identity)


class TeacherReferences:
    def __init__(self, root, disk_cap=32 * 2**30, hot_cap=2 * 2**30):
        self.root = Path(root)
        self.disk_cap, self.hot_cap = disk_cap, hot_cap
        self.hot, self.hot_bytes = OrderedDict(), 0

    def budget(self, shapes):
        needed = sum(2 * 2 * torch.Size(shape).numel() for shape in shapes)
        if needed > self.disk_cap:
            raise ValueError("required teacher evidence exceeds disk cap; pinned evidence cannot be evicted")
        return needed

    def read(self, identity):
        key = reference_key(identity)
        root = self.root / key
        record = json.loads((root / "manifest.json").read_text())
        if record.get("identity") != identity or record.get("key") != key:
            raise ValueError("teacher reference identity changed")
        values = {}
        for name in ("velocity", "clean"):
            p = root / f"{name}.pt"
            if file_sha256(p) != record[name]["file_sha256"]:
                raise ValueError("teacher reference content changed")
            value = torch.load(p, map_location="cpu", weights_only=True)
            if (
                value.dtype != torch.bfloat16
                or not torch.isfinite(value).all()
                or tensor_hash(value) != record[name]["tensor_sha256"]
            ):
                raise ValueError("invalid teacher tensor")
            values[name] = value
        if values["clean"].shape != values["velocity"].shape:
            raise ValueError("teacher velocity/clean geometry differs")
        self._remember(key, values)
        return values

    def _remember(self, key, values):
        if key in self.hot:
            self.hot_bytes -= sum(v.numel() * v.element_size() for v in self.hot.pop(key).values())
        size = sum(v.numel() * v.element_size() for v in values.values())
        if size <= self.hot_cap:
            while self.hot and self.hot_bytes + size > self.hot_cap:
                _, old = self.hot.popitem(last=False)
                self.hot_bytes -= sum(v.numel() * v.element_size() for v in old.values())
            self.hot[key] = values
            self.hot_bytes += size

    def produce(self, identity, forward):
        key = reference_key(identity)
        root = self.root / key
        if (root / "manifest.json").exists():
            return self.read(identity)
        # Account for orphaned tensors too; never discard scientific evidence.
        used = sum(p.stat().st_size for p in self.root.rglob("*.pt")) if self.root.exists() else 0
        with torch.no_grad():
            velocity, clean = forward()
        if (
            velocity.shape != clean.shape
            or velocity.dtype != torch.bfloat16
            or clean.dtype != torch.bfloat16
            or not torch.isfinite(velocity).all()
            or not torch.isfinite(clean).all()
        ):
            raise ValueError("teacher forward returned invalid native BF16 tensors")
        if used + sum(v.numel() * v.element_size() for v in (velocity, clean)) > self.disk_cap:
            raise ValueError("teacher disk evidence cap exceeded")
        record = {"key": key, "identity": identity}
        for name, value in (("velocity", velocity), ("clean", clean)):
            p = publish_tensor(root / f"{name}.pt", value)
            record[name] = {"file_sha256": file_sha256(p), "tensor_sha256": tensor_hash(value)}
        publish_json(root / "manifest.json", record)
        return self.read(identity)
