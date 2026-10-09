"""Test-only import boundary inherited by children through PYTHONPATH."""

import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

PREFIX = "scripts.onestep_avatar.experiments"


class ExperimentImportBlocker(MetaPathFinder):
    """Refuse the experiment package before its marker or a child can execute."""

    def find_spec(
        self, fullname: str, _path: list[str] | None = None, _target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname == PREFIX or fullname.startswith(PREFIX + "."):
            raise ModuleNotFoundError("experiment imports blocked: " + fullname, name=fullname)
        return None


if os.environ.get("ONESTEP_AVATAR_BLOCK_EXPERIMENTS") == "1":
    sys.meta_path.insert(0, ExperimentImportBlocker())
