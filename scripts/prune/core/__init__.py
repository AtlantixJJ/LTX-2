"""Bootstrap / geometry / the model registry / the private-API quarantine.

Physically relocated here (2026-08-28 refactor, revised): every module below
lives at ``scripts/prune/core/<name>.py``. There is no flat-file compat shim
and no promise that ``python -m scripts.prune.<name>`` (the old,
pre-relocation path) still resolves -- callers use
``python -m scripts.prune.core.<name>`` or
``from scripts.prune.core import <name>``. ``scripts/`` and
``scripts/prune/`` themselves stay ``__init__.py``-free namespace packages;
only the subpackages introduced by the grouping carry one.

Deliberately no eager ``from . import <name>`` here: `core` and `data`
import each other's submodules (``core.session`` needs ``data.prompt_cache``;
``data.source_target`` needs ``core.session``), and eagerly importing every
submodule at package-init time turns that into an import cycle. Each module
is reachable directly, e.g. ``from scripts.prune.core import session``,
without this package needing to have pre-imported it.

Modules: artifacts, geometry, ltx_adapter, model_registry, preflight,
provenance, refine_core, refine_task, session.
"""
