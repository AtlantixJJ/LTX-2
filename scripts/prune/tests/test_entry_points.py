from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_entry_module_imports_and_has_main() -> None:
    """Every module with a `__main__` guard imports without CUDA and exposes `main`."""
    names = sorted(
        ".".join(p.relative_to(ROOT).with_suffix("").parts)
        for p in ROOT.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts and "__main__" in p.read_text()
    )
    assert len(names) > 20, "the entry scan found too few modules -- fix the test, not the code"
    for name in names:
        assert callable(importlib.import_module(f"scripts.prune.{name}").main), name
