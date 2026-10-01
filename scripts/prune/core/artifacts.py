"""Paths and attributable run directories for whole-clip pruning artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.prune.core.model_registry import WORKSPACE_ROOT

OUT_ROOT = WORKSPACE_ROOT / "expr" / "refiner_prune"
GATES = ("caps", "prompt_cache_check")


def root(key: str) -> Path:
    return OUT_ROOT / key


def gate(key: str, name: str) -> Path:
    if name not in GATES:
        raise KeyError(f"unknown gate {name!r}; expected one of {GATES}")
    return root(key) / f"{name}.json"


def run_dir(key: str, prefix: str, *, script: str, argv: list[str]) -> Path:
    """Create an attributable run directory and append it to ``runs/index.jsonl``."""
    from scripts.prune.core import provenance

    path = root(key) / provenance.run_id(prefix)
    suffix = 1
    while path.exists():
        path = root(key) / f"{provenance.run_id(prefix)}-{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=True)
    line = {"run_id": path.name, "script": script, "argv": argv, "git_rev": provenance._git_rev(), "pid": os.getpid()}
    index = root(key) / "runs" / "index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a") as handle:
        handle.write(json.dumps(line) + "\n")
    return path
