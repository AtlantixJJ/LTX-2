"""Exact tiny whole-clip fixtures shared by avatar pruning comparisons."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import dataset
from scripts.prune.core.session import DTYPE
from scripts.prune.evaluate import whole_clip_baseline as producer

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)

CHANNELS = 8

FRAMES = 5

SIGMAS = (0.725, 0.909375)


@dataclass(frozen=True)
class _Caps:
    latent_channels: int = CHANNELS


def _tiny_model() -> X0Model:
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_layers=2,
        cross_attention_dim=8,
        caption_projection=torch.nn.Linear(16, 8),
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return X0Model(model.to(dtype=DTYPE)).eval()


class _Session:
    def __init__(self, vae: Path, *, decoder: object | None = None) -> None:
        self.device = torch.device("cpu")
        self.context = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(1)).to(dtype=DTYPE)
        self.model = type("Model", (), {})()
        self.model.sigmas = [1.0, 0.909375, 0.725, 0.421875, 0.0]
        self.model.scale_factors = SCALE
        self.model.caps = _Caps()
        self.model.paths = type("Paths", (), {"video_vae": staticmethod(lambda: vae)})()
        self._decoder = decoder

    def stamp(self, *, dtype: str) -> dict:
        return {"model_key": "2.5", "transformer_fingerprint": "tiny", "script": producer.SCRIPT, "dtype": dtype}

    @contextmanager
    def transformer(self, *, loras: tuple = ()):  # noqa: ANN202
        assert loras == ()
        with torch.no_grad():  # as core.session.Session.transformer
            yield _tiny_model()

    @contextmanager
    def decoder(self):  # noqa: ANN202
        yield self._decoder


def _vae(root: Path) -> Path:
    path = root / "vae.safetensors"
    save_file({"weight": torch.zeros(1)}, path)
    return path


def _views(root: Path) -> list[Path]:
    views = []
    for index, actor in enumerate(("0008_01", "0012_09")):
        view = root / actor / "views" / "view00"
        view.mkdir(parents=True)
        generator = torch.Generator().manual_seed(10 + index)
        for name in (dataset.capture_bundle_name("white"), dataset.guide_bundle_name("white")):
            master = torch.randn(CHANNELS, FRAMES, 2, 2, generator=generator).to(dtype=DTYPE)
            torch.save({"schema_version": 2, "master": master, "fps": 30.0}, view / name)
        views.append(view)
    return views
