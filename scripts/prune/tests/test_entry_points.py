from __future__ import annotations

import importlib
import re
from pathlib import Path


def test_every_documented_entry_point_is_importable():
    """Every `python -m scripts.prune.X.Y` in README.md resolves."""
    text = Path("scripts/prune/README.md").read_text()
    names = set(re.findall(r"scripts\.prune\.(\w+\.\w+)", text))
    assert names, "the extraction regex found nothing -- fix the test, not the code"
    for name in sorted(names):
        assert importlib.import_module(f"scripts.prune.{name}").main
