"""The single source of truth for paths under ``expr/refiner_prune/<key>``.

The source-target manifest used to be written and read through different string
literals.  Keep names here so a writer and reader cannot silently drift again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.prune.core.model_registry import WORKSPACE_ROOT

OUT_ROOT = WORKSPACE_ROOT / "expr" / "refiner_prune"
GATES = (
    "caps", "prompt_cache_check", "kv_cache_check", "video_only_check",
    "parity_check", "method_parity", "sampler_ab", "analysis_summary",
)


def root(key: str) -> Path:
    return OUT_ROOT / key


def manifest(key: str) -> Path:
    return root(key) / "source_target" / "manifest.json"


def calibration(key: str) -> Path:
    return root(key) / "calibration"


def calibration_index(key: str) -> Path:
    return calibration(key) / "index.json"


def figures(key: str) -> Path:
    return root(key) / "figures"


def prompt_cache(key: str) -> Path:
    return OUT_ROOT / "prompt_cache" / key


def bench(key: str, tag: str) -> Path:
    return root(key) / f"bench_{tag}.json"


def phase1(key: str, tag: str | None = None) -> Path:
    return root(key) / (f"phase1_gates_{tag}.json" if tag else "phase1_gates.json")


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
