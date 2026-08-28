from __future__ import annotations

import re
import subprocess
from pathlib import Path


def test_no_module_but_the_adapter_imports_ltx_privates():
    """Every underscore-prefixed ltx_core/ltx_pipelines symbol lives in exactly
    one place -- when the LTX-2 pin moves, ltx_adapter.py is the one file
    that needs to be re-checked.
    """
    pattern = re.compile(
        r"from ltx_\w+[\w.]* import [^\n]*\b_\w+"
        r"|\._(build_state|step_state|transformer_ctx|decoder_builder|build_encoder)\b"
    )
    offenders = {f.name for f in Path("scripts/prune").glob("*.py") if pattern.search(f.read_text())} - {
        "ltx_adapter.py"
    }
    assert offenders == set()


def test_adapter_docstring_names_a_resolvable_pinned_commit():
    """A pin recorded but never checked is worse than no pin: assert it names a
    real commit in this history, not a stale/typo'd placeholder.

    Deliberately not compared against the CURRENT ``git rev-parse HEAD`` --
    every commit after this file moves HEAD forward, which would make an
    exact-match assertion fail on the very next unrelated commit. The pin is
    a human-maintained marker of "adapter last reviewed against this LTX-2
    state", refreshed when the submodule pin actually moves, not on every
    scripts/prune commit.
    """
    src = Path("scripts/prune/ltx_adapter.py").read_text()
    m = re.search(r"[Pp]inned at(?: LTX-2 commit)? ([0-9a-f]{7,40})", src)
    assert m, "ltx_adapter.py must record its pinned LTX-2 commit, e.g. 'Pinned at LTX-2 commit 9a1c49b'"
    result = subprocess.run(["git", "cat-file", "-e", m.group(1)], capture_output=True, check=False)
    assert result.returncode == 0, f"{m.group(1)} is not a resolvable commit in this repo"
