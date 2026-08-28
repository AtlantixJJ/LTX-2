"""Organizational grouping only -- an import convenience, not a relocation.

The modules below still live at ``scripts/prune/<name>.py`` (that is what
keeps ``python -m scripts.prune.<name>`` and every existing internal
``from scripts.prune import <name>`` working unchanged); this package just
re-exports them under ``scripts.prune.core.<name>`` so the directory tree
groups them the way the README's module table does. ``scripts/`` and
``scripts/prune/`` themselves stay ``__init__.py``-free namespace packages.

Bootstrap / geometry / the model registry / the private-API quarantine.
"""

from scripts.prune import artifacts, geometry, ltx_adapter, model_registry, preflight, provenance, refine_core, refine_task, session

__all__ = [
    "artifacts",
    "geometry",
    "ltx_adapter",
    "model_registry",
    "preflight",
    "provenance",
    "refine_core",
    "refine_task",
    "session",
]
