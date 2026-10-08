"""B4: file moves cannot change repository roots or hide local root derivation."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator

import pytest

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT, WORKSPACE_ROOT


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Visit a scope without assigning nested function locals to its parent."""
    pending = [scope]
    while pending:
        node = pending.pop()
        yield node
        if node is not scope and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _root_derivations(source: str) -> list[int]:  # noqa: PLR0912, PLR0915 -- fixed-point filename flow across aliases/helpers
    """Find ancestor operations fed by a module filename, including helper calls.

    The finite sets track filename values, function return values and tainted
    parameters. Values from independent data paths never enter these sets.
    No source is imported or executed while checking the dependency boundary.
    """
    tree = ast.parse(source)
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    scopes = [tree, *functions.values(), *classes.values()]
    nodes = {id(scope): list(_scope_nodes(scope)) for scope in scopes}
    names: dict[int, set[str]] = {id(scope): set() for scope in scopes}
    names[id(tree)].add("__file__")
    returns: set[int] = set()
    dirname = {"dirname"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("os.path", "posixpath", "ntpath"):
            dirname.update(alias.asname or alias.name for alias in node.names if alias.name == "dirname")

    def filename(node: ast.AST | None, scope: ast.AST) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name):
            return node.id in names[id(scope)] or node.id in names[id(tree)]
        if isinstance(node, ast.Attribute):
            if node.attr == "__file__":
                return True
            if isinstance(node.value, ast.Name) and node.value.id in classes:
                return node.attr in names[id(classes[node.value.id])]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
            return id(functions[node.func.id]) in returns
        return any(filename(child, scope) for child in ast.iter_child_nodes(node))

    def bind(target: ast.AST, scope: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names[id(scope)].add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind(item, scope)

    while True:
        before = (sum(len(values) for values in names.values()), len(returns), len(functions))
        for scope in scopes:
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = [*scope.args.posonlyargs, *scope.args.args]
                defaults = list(zip(positional[-len(scope.args.defaults):], scope.args.defaults, strict=False))
                defaults.extend(zip(scope.args.kwonlyargs, scope.args.kw_defaults, strict=False))
                for parameter, default in defaults:
                    if filename(default, tree):
                        names[id(scope)].add(parameter.arg)
            for node in nodes[id(scope)]:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if filename(node.value, scope):
                        for target in targets:
                            bind(target, scope)
                    if isinstance(node.value, ast.Name) and node.value.id in functions:
                        for target in targets:
                            if isinstance(target, ast.Name):
                                functions[target.id] = functions[node.value.id]
                if isinstance(node, ast.Return) and filename(node.value, scope):
                    returns.add(id(scope))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                    function = functions[node.func.id]
                    parameters = [*function.args.posonlyargs, *function.args.args]
                    for parameter, argument in zip(parameters, node.args, strict=False):
                        if filename(argument, scope):
                            names[id(function)].add(parameter.arg)
                    for keyword in node.keywords:
                        if keyword.arg is not None and filename(keyword.value, scope):
                            names[id(function)].add(keyword.arg)
        after = (sum(len(values) for values in names.values()), len(returns), len(functions))
        if before == after:
            break

    violations = set()
    for scope in scopes:
        for node in nodes[id(scope)]:
            if isinstance(node, ast.Attribute) and node.attr in ("parent", "parents") and filename(node.value, scope):
                violations.add(node.lineno)
            if isinstance(node, ast.Call):
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                )
                if function_name in dirname and any(filename(argument, scope) for argument in node.args):
                    violations.add(node.lineno)
    return sorted(violations)


@pytest.mark.parametrize(
    "source",
    [
        "root = Path(__file__).resolve().parents[2]",
        "class Paths:\n    root = Path(__file__).resolve().parents[2]",
        "class Paths:\n    filename = Path(__file__)\nroot = Paths.filename.parent",
        "def root(filename=__file__):\n    return Path(filename).parents[2]",
        "def root(*, filename=__file__):\n    return Path(filename).parent",
        "filename = Path(__file__)\ncopy = filename.resolve()\nroot = copy.parent",
        "def filename():\n    return Path(__file__)\nroot = filename().parents[1]",
        "def ancestor(path):\n    return path.parent\nroot = ancestor(Path(__file__))",
        "def filename():\n    return Path(__file__)\nget_file = filename\nroot = get_file().parent",
        "from os.path import dirname as parent_dir\nroot = parent_dir(__file__)",
        "import os as operating_system\nfilename = __file__\nroot = operating_system.path.dirname(filename)",
    ],
)
def test_b4_detects_filename_aliases_and_helpers(source: str) -> None:
    assert _root_derivations(source)


@pytest.mark.parametrize(
    "source",
    [
        "digest = sha256(Path(__file__))",
        "digest = sha256(Path(__file__).resolve())",
        "sources = {str(Path(module.__file__).resolve()): sha256(Path(module.__file__))}",
        "root = view.parents[1]",
        "from scripts.onestep_avatar import LTX_ROOT\nchild_cwd = LTX_ROOT",
        "def parent(path):\n    return path.parent\nroot = parent(view)",
    ],
)
def test_b4_accepts_file_identity_and_data_ancestors(source: str) -> None:
    assert _root_derivations(source) == []


def test_b4_package_uses_only_canonical_repository_roots() -> None:
    violations = {
        str(path.relative_to(PACKAGE_ROOT)): lines
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if path != PACKAGE_ROOT / "__init__.py" and (lines := _root_derivations(path.read_text()))
    }
    assert violations == {}


def test_canonical_roots_keep_existing_workspace_and_software_paths() -> None:
    from scripts.onestep_avatar import dataset, software  # noqa: PLC0415 -- inspect current public consumers

    assert PACKAGE_ROOT == LTX_ROOT / "scripts/onestep_avatar"
    assert LTX_ROOT.parent == WORKSPACE_ROOT
    assert dataset.WORKSPACE_ROOT == WORKSPACE_ROOT
    assert software.LTX_ROOT == LTX_ROOT
    assert (PACKAGE_ROOT / "train.py").is_file()
    assert (LTX_ROOT / "pyproject.toml").is_file()
    assert "scripts/onestep_avatar/__init__.py" in software.source_files("training", "causal")


def test_repository_roots_import_with_standard_library_only() -> None:
    source = """
import sys
from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT, WORKSPACE_ROOT
assert PACKAGE_ROOT == LTX_ROOT / 'scripts/onestep_avatar'
assert LTX_ROOT.parent == WORKSPACE_ROOT
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", source], cwd=LTX_ROOT, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
