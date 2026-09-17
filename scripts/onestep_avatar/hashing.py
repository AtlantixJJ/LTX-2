"""``sha256(path)`` -- the one file-hash helper the corpus pass and the subset freezer share.

Not geometry (``geometry.py`` is pure crop-box math, no I/O) and not dataset layout
(``dataset.py`` is filenames and paths, no bytes read) -- a hash is its own small concern, so it
gets its own three-line module rather than living in either. This is the one new file S1 of the
2026-09-17 cleanup plan allows: ``precompute.py`` and ``windows.py`` carried byte-identical
1-MiB-chunk implementations of the same function.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """The file's sha256 digest, read in ``chunk``-byte pieces (default 1 MiB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
