"""Shared saved-input builders and fixtures; never a collected test module."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import infer
from scripts.onestep_avatar.corpus.precompute import file_fingerprint


@pytest.fixture
def product_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"vae identity fixture")
    common_record = {
        "schema_version": 2,
        "objective": "white",
        "fps": 30,
        "box_xyxy": [0, 0, 64, 64],
        "edge": 64,
        "encode_contract_version": 1,
        "vae_fingerprint": file_fingerprint(vae),
    }
    guide, image = tmp_path / "guide.pt", tmp_path / "image.pt"
    torch.save({**common_record, "master": torch.zeros(128, 7, 2, 2), "pixel_frames": 49}, guide)
    torch.save(
        {**common_record, "master": torch.zeros(128, 1, 2, 2), "pixel_frames": 1, "input_role": "supplied_image"}, image
    )
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=128),
        sigmas=[0.725],
        paths=SimpleNamespace(transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: vae),
    )
    monkeypatch.setattr(infer.backbone, "resolve", lambda *_args: specification)
    monkeypatch.setattr(infer.backbone, "identity", lambda *_args, **_kwargs: {"base_transformer_sha256": "a" * 64})
    args = [
        "--mode",
        "bidirectional",
        "--guide",
        str(guide),
        "--first-image",
        str(image),
        "--output",
        str(tmp_path / "out"),
        "--variant",
        "dev",
        "--schedule",
        "0.725",
        "0",
    ]
    return args, guide, image


build_product_files = product_files.__wrapped__
