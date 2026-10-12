"""Shared saved-input builders and fixtures; never a collected test module."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.experiments import causality as evaluate
from scripts.onestep_avatar.tests.support.model import CHANNELS


def fixture_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int = 17
) -> tuple[Path, Path, Path, torch.Tensor, torch.Tensor]:
    view = tmp_path / "view"
    view.mkdir()
    dev, adapter = tmp_path / "dev.safetensors", tmp_path / "adapter.safetensors"
    for path in (dev, adapter, view / dataset.capture_bundle_name("white"), view / dataset.guide_bundle_name("white")):
        path.write_bytes(b"controlled scientific input")
    capture, guide = torch.zeros(8, frames, 2, 2), torch.ones(8, frames, 2, 2)
    monkeypatch.setattr(evaluate.backbone, "transformer_path", lambda *_args: dev)
    monkeypatch.setattr(evaluate.backbone, "identity", lambda *_args: {"base_transformer_sha256": "f" * 64})
    caps = SimpleNamespace(caps=SimpleNamespace(latent_channels=CHANNELS))
    monkeypatch.setattr(evaluate.backbone, "resolve", lambda *_args: caps)
    monkeypatch.setattr(
        evaluate.evaluate,
        "check_adapter",
        lambda path, _requested: {"contract": {"fixture": "adapter"}, "adapter_sha256": evaluate.sha256(path)},
    )
    monkeypatch.setattr(evaluate.checkpoints, "recheck_adapter", lambda *_args: None)
    monkeypatch.setattr(
        evaluate.adapters, "inference_transformer", lambda session, *_args, **_kwargs: session.transformer()
    )
    monkeypatch.setattr(
        dataset,
        "load_training_master",
        lambda path: (capture if path.name == dataset.capture_bundle_name("white") else guide, 30),
    )
    return dev, adapter, view, capture, guide
