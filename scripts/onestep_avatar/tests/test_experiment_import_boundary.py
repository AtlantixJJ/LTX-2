"""Ordinary execution refuses experiment imports, including inherited children."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT


def blocked_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["ONESTEP_AVATAR_BLOCK_EXPERIMENTS"] = "1"
    hook = str(PACKAGE_ROOT / "tests/blocked_imports")
    previous = environment.get("PYTHONPATH")
    if previous is None or hook not in previous.split(os.pathsep):
        environment["PYTHONPATH"] = hook + ("" if previous is None else os.pathsep + previous)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def test_inherited_child_cannot_import_experiment_package() -> None:
    source = """
import importlib
for name in ('scripts.onestep_avatar.experiments',
             'scripts.onestep_avatar.experiments.stock_parity',
             'scripts.onestep_avatar.experiments.training_update_check'):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        assert 'experiment imports blocked' in str(error)
    else:
        raise AssertionError(name + ' imported')
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=LTX_ROOT,
        env=blocked_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["train", "evaluate", "infer", "decode_saved", "prepare_inputs", "execution.queue"])
def test_blocked_child_ordinary_help_still_works(module: str) -> None:
    arguments = [sys.executable, "-m", "scripts.onestep_avatar." + module, "--help"]
    if module == "train":
        arguments.extend(["--mode", "bidirectional"])
    result = subprocess.run(
        arguments, cwd=LTX_ROOT, env=blocked_environment(), capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_blocked_child_typed_dry_run_imports_no_experiments(mode: str, tmp_path: Path) -> None:
    # This isolates CLI/import boundaries. The ordinary tiny-update matrix checks
    # actual producer input preflight, updates, exports and preview enqueue.
    source = """
import sys
from pathlib import Path
from types import SimpleNamespace
from scripts.onestep_avatar import train
from scripts.onestep_avatar.training import engine
engine.prepare_run = lambda *_args, **_kwargs: (
    SimpleNamespace(membership={'sha256': 'a'*64}), {'sha256': 'b'*64, 'samples': []}, None, False)
engine.build_transformer = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('loaded transformer'))
output = Path(sys.argv[2])
assert train.main(['--mode', sys.argv[1], '--subset', '/boundary-only', '--output', str(output), '--dry-run']) == 0
assert not output.exists()
assert not any(name.startswith('scripts.onestep_avatar.experiments') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", source, mode, str(tmp_path / "unwritten")],
        cwd=LTX_ROOT,
        env=blocked_environment(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("owner", ["evaluate", "infer"])
@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_blocked_child_support_dry_run_imports_no_experiments(owner: str, mode: str, tmp_path: Path) -> None:
    # Input validation has real-data ordinary tests; this child isolates imports
    # and the public CLI's no-write dry-run branch before any native session.
    source = """
import sys
from pathlib import Path
from scripts.onestep_avatar import evaluate, infer
owner, mode, destination = sys.argv[1:]
output = Path(destination)
if owner == 'evaluate':
    evaluate.prepare_evaluation = lambda _args: (
        None, [None], [(None, None, {'mode': mode}, None)], {})
    arguments = ['--mode', mode, '--subset', '/boundary-only', '--output', str(output),
                 '--variant', 'dev', '--guide-mode', 'd0', '--schedule', '0.725', '0', '--dry-run']
    status = evaluate.main(arguments)
else:
    infer.prepare_product = lambda _args: (None, None, None, None, {'mode': mode}, None)
    arguments = ['--mode', mode, '--guide', '/boundary-guide', '--first-image', '/boundary-image',
                 '--output', str(output), '--variant', 'dev', '--schedule', '0.725', '0', '--dry-run']
    status = infer.main(arguments)
assert status == 0
assert not output.exists()
assert not any(name.startswith('scripts.onestep_avatar.experiments') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", source, owner, mode, str(tmp_path / "unwritten")],
        cwd=LTX_ROOT,
        env=blocked_environment(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
