"""Deprecated alias for scripts.prune.source_target (renamed 2026-08-28).

``teacher.py`` predates the switch from a 16-step teacher to VAE-encoded source
targets; see the README's "Two places the plan is wrong" section for why that
teacher was dropped. Removing this shim is safe once README.md and
plans/2026-08-2{6,7}-* no longer name it.
"""

from __future__ import annotations

import sys

from scripts.prune.source_target import main

if __name__ == "__main__":
    print("scripts.prune.teacher is deprecated; use scripts.prune.source_target", file=sys.stderr)
    raise SystemExit(main())
