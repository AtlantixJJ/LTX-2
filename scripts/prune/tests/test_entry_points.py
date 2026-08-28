from __future__ import annotations

import importlib
import re
from pathlib import Path

from scripts.prune import artifacts


def test_every_documented_entry_point_is_importable():
    """Every `python -m scripts.prune.X` in README.md and run_head_sweep.sh resolves."""
    text = Path("scripts/prune/README.md").read_text() + Path("scripts/prune/run_head_sweep.sh").read_text()
    names = set(re.findall(r"scripts\.prune\.(\w+)", text))
    assert names, "the extraction regex found nothing -- fix the test, not the code"
    for name in sorted(names):
        assert importlib.import_module(f"scripts.prune.{name}").main


def test_the_sweep_scripts_two_hard_coded_paths_still_exist():
    sh = Path("scripts/prune/run_head_sweep.sh").read_text()
    assert "method_parity.json" in sh and "calibration/index.json" in sh
    assert artifacts.gate("2.5", "method_parity").name == "method_parity.json"
    assert artifacts.calibration_index("2.5").name == "index.json"
