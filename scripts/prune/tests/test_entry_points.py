from __future__ import annotations

import importlib
import re
from pathlib import Path

from scripts.prune.core import artifacts, preflight


def test_every_documented_entry_point_is_importable():
    """Every `python -m scripts.prune.X.Y` in README.md and run_head_sweep.sh resolves."""
    text = Path("scripts/prune/README.md").read_text() + Path("scripts/prune/run_head_sweep.sh").read_text()
    names = set(re.findall(r"scripts\.prune\.(\w+\.\w+)", text))
    assert names, "the extraction regex found nothing -- fix the test, not the code"
    for name in sorted(names):
        assert importlib.import_module(f"scripts.prune.{name}").main


def test_sweep_checks_named_artifacts_via_preflight():
    sh = Path("scripts/prune/run_head_sweep.sh").read_text()
    source = Path(preflight.__file__).read_text()
    assert "--check-sweep-prereqs" in sh
    assert 'artifacts.gate(model.key, "method_parity")' in source
    assert "artifacts.calibration_index(model.key)" in source
    assert artifacts.gate("2.5", "method_parity").name == "method_parity.json"
    assert artifacts.calibration_index("2.5").name == "index.json"
