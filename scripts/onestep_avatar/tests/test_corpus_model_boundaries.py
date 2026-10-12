"""Corpus and model imports obey their ownership rows, including lazy imports."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT
from scripts.onestep_avatar.tests.support.imports import internal_imports

PREFIX = "scripts.onestep_avatar"
ROOT_NAMES = {"PACKAGE_ROOT", "LTX_ROOT", "WORKSPACE_ROOT"}


def violations(source: str, relative: str) -> list[tuple[str, str | None]]:
    area = relative.split("/", 1)[0]
    allowed = {PREFIX, PREFIX + ".hashing", PREFIX + ".corpus"}
    if area == "model":
        allowed.add(PREFIX + ".model")
    failures = []
    module = PREFIX + "." + relative.removesuffix(".py").replace("/", ".")
    for target, function in internal_imports(source, module):
        if (
            not any(
                target == permitted or target.startswith(permitted + ".")
                for permitted in allowed
                if permitted != PREFIX
            )
            and target != PREFIX
        ):
            failures.append((target, function))
    return failures


@pytest.mark.parametrize("area", ["corpus", "model"])
def test_corpus_and_model_dependency_rows_include_lazy_imports(area: str) -> None:
    bad = {}
    for path in sorted((PACKAGE_ROOT / area).rglob("*.py")):
        relative = str(path.relative_to(PACKAGE_ROOT))
        if failures := violations(path.read_text(), relative):
            bad[relative] = failures
    assert not bad, bad


def test_boundary_finds_lazy_relative_and_root_named_back_edges() -> None:
    assert violations("def f():\n from ..training import checkpoints\n", "model/adapters.py")
    assert violations("def f():\n from scripts.onestep_avatar import evaluate\n", "corpus/dataset.py")
    assert violations("from scripts.onestep_avatar.model.causal import CausalGeometry\n", "corpus/subset.py")
    assert violations("def f():\n from scripts.onestep_avatar.model.causal import CausalGeometry\n", "corpus/subset.py")


def test_ordinary_corpus_imports_do_not_load_model_or_training() -> None:
    command = """import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith(('scripts.onestep_avatar.model', 'scripts.onestep_avatar.training')):
   raise RuntimeError('blocked owner: ' + fullname)
sys.meta_path.insert(0, Block())
from scripts.onestep_avatar.corpus import dataset, subset, geometry, mask_video, motion, qa
assert dataset.CAPTURE_MANIFEST_NAME == 'capture_latent_manifest.json'
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, "-c", command], cwd=LTX_ROOT, env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
