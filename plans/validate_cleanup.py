"""Check the plan cleanup without importing model code or changing files.

Run from LTX-2: conda run -n ltx python plans/validate_cleanup.py
Read the original-file inventory, reject restored retired paths, check retained
evidence bytes, and verify recoverable Git blobs. Check local links and anchors
in the three replacement documents and the new METHODS/handoff references. Missing
local experiment assets are failures; this tool never recreates them.
"""

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    inventory = root / "plans/cleanup_2026-10-17.tsv"
    with inventory.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 23, "original inventory is incomplete"
    assert len({row["original_path"] for row in rows}) == len(rows), "duplicate source"
    relocated = 0
    for row in rows:
        original = root / row["original_path"]
        if row["action"] == "replace":
            assert original.is_file(), f"missing replacement: {original}"
            assert hashlib.sha256(original.read_bytes()).hexdigest() != row["sha256"], "old README restored"
        else:
            assert row["action"] in {"remove", "relocate"}, "unknown action"
            assert not original.exists(), f"retired path restored: {original}"
            assert not original.is_symlink(), f"retired symlink restored: {original}"
        if row["action"] == "relocate":
            data = (root / row["retained_path"]).read_bytes()
            assert len(data) == int(row["bytes"]), f"retained size changed: {row['retained_path']}"
            assert hashlib.sha256(data).hexdigest() == row["sha256"], f"retained bytes changed: {row['retained_path']}"
            relocated += 1
        if row["git_blob"] != "-":
            data = subprocess.check_output(["git", "cat-file", "blob", row["git_blob"]], cwd=root)
            assert len(data) == int(row["bytes"]), "Git source size changed"
            assert hashlib.sha256(data).hexdigest() == row["sha256"], "Git source differs"
    assert relocated == 4, "retained evidence inventory is incomplete"

    documents = [root / "plans" / name for name in ("README.md", "RESEARCH_BACKLOG.md", "HISTORY.md")]
    references = {
        root / "scripts/prune/doc/METHODS.md": "../../../plans/",
        root.parent / "plans/2026-10-07-onestep-avatar-development-experiment-handoff.md": "../LTX-2/plans/",
    }
    links = 0
    for document in [*documents, *references]:
        source = document.read_text()
        for match in re.finditer(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", source):
            target = match.group(1).strip("<>")
            if document not in documents and not target.startswith(references[document]):
                continue  # Unrelated historical links are outside this cleanup.
            if target.startswith(("https:", "http:", "mailto:")):
                continue
            relative, _, anchor = target.partition("#")
            path = document.parent / unquote(relative) if relative else document
            assert path.exists(), f"broken link in {document}: {target}"
            if anchor and path.suffix == ".md":
                headings = re.findall(r"^#{1,6}\s+(.+)$", path.read_text(), re.MULTILINE)
                slugs = {re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-") for heading in headings}
                assert anchor in slugs, f"broken anchor in {document}: {target}"
            links += 1
    sys.stdout.write(
        f"PASS: {len(rows)} originals, {relocated} byte-identical retained assets, "
        f"Git sources, {links} local links/anchors; retired paths absent.\n"
    )


if __name__ == "__main__":
    main()
