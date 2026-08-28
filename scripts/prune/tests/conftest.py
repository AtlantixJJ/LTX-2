"""Fixtures backed by the deployed checkpoint and calibration cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prune.model_registry import WORKSPACE_ROOT

CALIB = WORKSPACE_ROOT / "expr" / "refiner_prune" / "2.5" / "calibration"
CORPUS = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"


@pytest.fixture(scope="session")
def model():
    """The real 2.5 registry entry; resolution reads checkpoint metadata only."""
    from scripts.prune import model_registry

    try:
        return model_registry.resolve("2.5")
    except SystemExit as exc:
        pytest.skip(f"2.5 checkpoint not on disk: {exc}")


@pytest.fixture(scope="session")
def calibration_index():
    path = CALIB / "index.json"
    if not path.exists():
        pytest.skip(f"no calibration cache at {CALIB}")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def record_paths(calibration_index):
    """Four real records spanning both families and both chunk widths."""
    want = ["__n1__s0__on_policy", "__n1__s0__renoised", "__n2__s0__on_policy", "__n2__s1__on_policy"]
    out = [next((path for path in sorted(CALIB.glob("*.pt")) if needle in path.name), None) for needle in want]
    if any(path is None for path in out):
        pytest.skip("calibration cache does not span the expected families")
    return out
