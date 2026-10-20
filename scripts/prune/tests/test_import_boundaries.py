"""Package areas import downward: core, data, score, learn, then evaluate and checks.

The allow-list records every existing upward edge. A new upward edge fails;
removing one from the code requires removing it here, so the list only shrinks.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = {"core": 0, "data": 1, "score": 2, "learn": 3, "evaluate": 4, "checks": 4}
ALLOWED = {
    # Session owns the selected prompt context.
    ("core/session.py", "data"),
    # Candidate verification defers to the mask reader and export width probe.
    ("data/whole_clip.py", "score"),
    # Study freezing binds learning identities, protocols and gate geometry.
    ("data/learned_head_study.py", "learn"),
    ("data/ffn_study.py", "learn"),
    # Recovered exports read learned states and replay consumed inputs.
    ("score/export_joint.py", "learn"),
    ("score/export_learned_ffn.py", "learn"),
    ("score/export_learned_heads.py", "learn"),
    # Head monitors reuse the paired evaluation metrics and the timing bracket.
    ("learn/heads.py", "evaluate"),
}


def upward_edges():
    edges = set()
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in ORDER or "__pycache__" in rel.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scripts.prune."):
                target = node.module.split(".")[2]
                if target in ORDER and ORDER[target] > ORDER[rel.parts[0]]:
                    edges.add((str(rel), target))
    return edges


def test_no_new_upward_imports():
    assert upward_edges() - ALLOWED == set()


def test_allow_list_has_no_stale_entries():
    assert ALLOWED - upward_edges() == set()
