"""CPU-only, no data/GPU needed. Run with: python -m pytest scripts/onestep_avatar/tests"""

from __future__ import annotations

import hashlib

from scripts.onestep_avatar.hashing import sha256


def test_sha256_matches_hashlib(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "blob.bin"
    payload = b"x" * (3 * (1 << 20) + 17)  # spans several read chunks
    path.write_bytes(payload)
    assert sha256(path) == hashlib.sha256(payload).hexdigest()


def test_chunk_size_does_not_change_the_digest(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "blob.bin"
    payload = b"y" * 5000
    path.write_bytes(payload)
    assert sha256(path, chunk=17) == sha256(path, chunk=1 << 20)
