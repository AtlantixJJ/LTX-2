"""Ordinary execution refuses experiment imports, including inherited children."""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT
from scripts.onestep_avatar.tests.support.children import run_dry_run_rows, run_help_rows
from scripts.onestep_avatar.tests.support.dry_run_entry import SOURCE_0, SOURCE_1


def test_production_imports_keep_experiments_out_of_ordinary_owners() -> None:
    prefix = "scripts.onestep_avatar"
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in ("tests", "experiments"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parent = [prefix, *relative.parts[:-1]]
                base = node.module or ""
                if node.level:
                    base = ".".join([*parent[: len(parent) - node.level + 1], *([base] if base else [])])
                targets = [base, *(base + "." + alias.name for alias in node.names)]
            else:
                continue
            if any(
                target == prefix + ".experiments" or target.startswith(prefix + ".experiments.") for target in targets
            ):
                violations.append((str(relative), node.lineno))
    assert not violations, violations


def test_only_execution_owns_claims_children_and_supervision() -> None:
    """B9: imported aliases cannot conceal a second launch or claim owner."""
    protected = {
        "ProcessRegistry",
        "supervise",
        "prepare_request",
        "guard_command",
        "publish_grant",
        "wait_registration",
        "recover_registered_attempt",
    }
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in ("tests", "execution"):
            continue
        tree = ast.parse(path.read_text())
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in protected:
                        aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
            )
            # Model-free ffmpeg encoding in corpus/mask_video is not a model launcher.
            if (
                name in protected
                or aliases.get(name) in protected
                or (relative.parts[0] == "experiments" and name == "Popen")
            ):
                violations.append((str(relative), node.lineno, name))
    assert not violations, violations


def test_experiments_call_public_package_owners() -> None:
    violations = []
    for path in sorted((PACKAGE_ROOT / "experiments").rglob("*.py")):
        tree = ast.parse(path.read_text())
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scripts.onestep_avatar"):
                for name in node.names:
                    aliases.add(name.asname or name.name)
                    if name.name.startswith("_"):
                        violations.append((str(path.relative_to(PACKAGE_ROOT)), node.lineno, name.name))
            elif isinstance(node, ast.Import):
                aliases.update(
                    name.asname or name.name.split(".")[0]
                    for name in node.names
                    if name.name.startswith("scripts.onestep_avatar")
                )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr.startswith("_")
                and node.attr != "__file__"  # Module-byte provenance is public Python metadata.
            ):
                violations.append((str(path.relative_to(PACKAGE_ROOT)), node.lineno, node.attr))
    assert not violations, violations


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


@pytest.mark.subprocess
def test_blocked_child_ordinary_help_still_works(subtests: object) -> None:
    rows = run_help_rows(
        [
            "scripts.onestep_avatar.train",
            "scripts.onestep_avatar.evaluate",
            "scripts.onestep_avatar.infer",
            "scripts.onestep_avatar.decode_saved",
            "scripts.onestep_avatar.prepare_inputs",
            "scripts.onestep_avatar.execution.queue",
        ],
        blocked_environment(),
    )
    for row in rows:
        with subtests.test(module=row["id"]):
            assert row["returncode"] == 0, row["stderr"]
            assert "usage:" in row["stdout"]


@pytest.mark.subprocess
def test_blocked_child_dry_run_rows(tmp_path: object, subtests: object) -> None:
    cases = [
        ("train", "bidirectional", SOURCE_0),
        ("train", "causal", SOURCE_0),
        ("evaluate", "bidirectional", SOURCE_1),
        ("infer", "bidirectional", SOURCE_1),
        ("evaluate", "causal", SOURCE_1),
        ("infer", "causal", SOURCE_1),
    ]
    rows = run_dry_run_rows(
        [
            {
                "id": owner + "_" + mode,
                "source": source,
                "argv": [
                    "probe",
                    *([mode] if owner == "train" else [owner, mode]),
                    str(tmp_path / (owner + "_" + mode)),
                ],
            }
            for owner, mode, source in cases
        ],
        blocked_environment(),
    )
    for row in rows:
        with subtests.test(case=row["id"]):
            assert row["returncode"] == 0, row["stderr"]
