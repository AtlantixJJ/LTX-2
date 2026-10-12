"""Resolve static import targets, including relative and function-local imports.

The caller supplies the full module name. Targets include the base and each
from-import member, so both module and symbol imports can select their owner.
"""

import ast


def targets(source: str, module: str) -> set[str]:
    result = set()
    parent = module.split(".")[:-1]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(parent[: len(parent) - node.level + 1] + ([base] if base else []))
            result.add(base)
            result.update(base + "." + alias.name for alias in node.names)
    return result


PREFIX = "scripts.onestep_avatar"
ROOT_NAMES = {"PACKAGE_ROOT", "LTX_ROOT", "WORKSPACE_ROOT"}


def internal_imports(source: str, module: str) -> list[tuple[str, str | None]]:
    """Resolve each internal import and its enclosing function; no lexical line scan."""
    found = []
    functions = []

    class Imports(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            functions.append(node.name)
            self.generic_visit(node)
            functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815 -- AST visitor method name

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name.startswith(PREFIX):
                    found.append((alias.name, functions[-1] if functions else None))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            base = node.module or ""
            if node.level:
                parent = module.split(".")[: -node.level]
                base = ".".join([*parent, *([base] if base else [])])
            if base == PREFIX:
                for alias in node.names:
                    target = PREFIX if alias.name in ROOT_NAMES else PREFIX + "." + alias.name
                    found.append((target, functions[-1] if functions else None))
            elif base.startswith(PREFIX + "."):
                found.append((base, functions[-1] if functions else None))

    Imports().visit(ast.parse(source))
    return found
