"""Restore files at their original absolute paths between tamper subtests.

Snapshots preserve bytes, symlink targets and empty directories. Restore removes
new paths first, never follows symlinks, then reinstates the original tree.
"""

import shutil
from pathlib import Path


def snapshot(root: object) -> object:
    root = Path(root)
    result = {}
    for path in root.rglob("*"):
        key = str(path.relative_to(root))
        if path.is_symlink():
            result[key] = ("link", str(path.readlink()))
        elif path.is_dir():
            result[key] = ("dir", None)
        else:
            result[key] = ("file", path.read_bytes())
    return result


def restore(root: object, saved: object) -> object:
    root = Path(root)
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        key = str(path.relative_to(root))
        kind = "link" if path.is_symlink() else "dir" if path.is_dir() else "file"
        if key not in saved or saved[key][0] != kind:
            if kind == "dir":
                shutil.rmtree(path)
            else:
                path.unlink()
    for name, (kind, value) in sorted(saved.items(), key=lambda row: len(Path(row[0]).parts)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "dir":
            path.mkdir(exist_ok=True)
        elif kind == "link":
            if path.is_symlink():
                path.unlink()
            path.symlink_to(value)
        else:
            path.write_bytes(value)
