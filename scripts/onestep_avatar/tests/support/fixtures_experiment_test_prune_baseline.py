"""Shared saved-input builders and fixtures; never a collected test module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.onestep_avatar.tests.support.prune import SCALE, _Caps, _Session
from scripts.prune.core import provenance
from scripts.prune.evaluate import whole_clip_baseline as producer


class _BoundSession(_Session):
    def __init__(self, vae: Path, transformer: Path) -> None:
        super().__init__(vae)
        self._transformer = transformer
        self._vae = vae
        self.model.caps = _BoundCaps()

    def stamp(self, *, dtype: str) -> dict:
        return {
            "model_key": "2.5",
            "model_version": [2, 5],
            "transformer_path": str(self._transformer.resolve()),
            "transformer_fingerprint": provenance.checkpoint_fingerprint(self._transformer),
            "video_vae_path": str(self._vae.resolve()),
            "scale_factors": list(SCALE),
            "script": producer.SCRIPT,
            "dtype": dtype,
        }


@dataclass(frozen=True)
class _BoundCaps(_Caps):
    num_layers: int = 2
