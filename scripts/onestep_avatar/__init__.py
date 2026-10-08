"""Canonical repository roots for corpus, model and execution owners.

Resolve the package location once, then select its LTX-2 and workspace parents.
Consumers import these constants so moving a module does not change data paths
or child working directories. Import only the standard library here; the parent
scripts directory remains a namespace package for the ARGAvatar runtime.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
LTX_ROOT = PACKAGE_ROOT.parents[1]
WORKSPACE_ROOT = LTX_ROOT.parent
