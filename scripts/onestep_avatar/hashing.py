"""Hash file contents and tensor identity through one root leaf owner.

File hashing reads fixed-size chunks. Tensor hashing includes JSON shape and
dtype, then contiguous original bytes on CPU. It imports Torch only inside
the tensor function, so corpus guide imports retain the standard-library-only
module boundary. Neither function changes its input or writes an artifact."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import torch


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """The file's sha256 digest, read in ``chunk``-byte pieces (default 1 MiB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()

def tensor_sha256(value: torch.Tensor) -> str:
    """Hash shape, dtype and original tensor bytes, independently of serialization."""
    import torch  # noqa: PLC0415 -- preserve a stdlib-only leaf at module import

    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
