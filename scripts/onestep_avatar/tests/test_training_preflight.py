"""New-runtime preflight checks real input records without model or output mutation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import precompute, subset
from scripts.onestep_avatar.tests.test_subset import old_subset
from scripts.onestep_avatar.training import config, engine


@pytest.fixture
def checked_settings(old_subset: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    membership, _ = subset.convert_legacy(old_subset, original_file_sha256="a" * 64)
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(membership))
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(path),
            "--output",
            str(tmp_path / "output"),
            "--variant",
            "dev",
            "--objective",
            "white",
            "--guide-mode",
            "d0",
            "--span-latent-frames",
            "7",
        ]
    )
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=2),
        sigmas=[0.725],
        paths=SimpleNamespace(transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: tmp_path / "vae.safetensors"),
    )
    monkeypatch.setattr(engine.backbone, "resolve", lambda *args: specification)
    monkeypatch.setattr(precompute, "file_fingerprint", lambda path: "saved VAE identity")
    monkeypatch.setattr(
        engine.backbone,
        "identity",
        lambda *args, **kwargs: {
            "base_transformer_sha256": "a" * 64,
            "base_transformer_file": "base.safetensors",
            "base_variant": "dev",
            "model_key": "2.5",
        },
    )
    monkeypatch.setattr(engine, "build_transformer", lambda *args: pytest.fail("preflight loaded a transformer"))
    return settings, membership


def test_preflight_checks_both_inputs_without_creating_output(checked_settings) -> None:
    settings, membership = checked_settings
    store, plan, _, archive = engine.prepare_run(settings)
    assert len(store) == 2
    assert plan["mode"] == "bidirectional"
    assert plan["membership_sha256"] == membership["sha256"]
    assert archive is False
    assert not settings.output.exists()


def test_bad_guide_fails_before_base_resolution_or_output_mutation(
    checked_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = checked_settings
    settings.guide_mode = "d1"
    monkeypatch.setattr(engine.backbone, "resolve", lambda *args: pytest.fail("missing guide reached base resolution"))
    with pytest.raises(ValueError, match="guide content hash is not recorded"):
        engine.prepare_run(settings)
    assert not settings.output.exists()


def test_overwrite_preflight_preserves_used_output(checked_settings) -> None:
    settings, _ = checked_settings
    settings.output.mkdir()
    old = settings.output / "original.txt"
    old.write_text("existing run")
    with pytest.raises(ValueError, match="already has a run"):
        engine.prepare_run(settings)
    settings.overwrite = True
    _, _, _, archive = engine.prepare_run(settings)
    assert archive is True
    assert old.read_text() == "existing run"
    assert not list(settings.output.glob("archived_*"))


def test_wrong_mode_record_fails_before_any_file_read(checked_settings) -> None:
    settings, _ = checked_settings
    settings.mode = "causal"
    settings.subset = Path("/file/that/does/not/exist")
    with pytest.raises(ValueError, match="typed mode settings disagree"):
        engine.prepare_run(settings)


def test_distilled_off_grid_level_is_rejected(checked_settings) -> None:
    settings, _ = checked_settings
    settings.variant = "distilled"
    settings.sigma0 = 0.6
    with pytest.raises(ValueError, match="distilled base grid"):
        engine.prepare_run(settings)
    assert not settings.output.exists()


def test_wrong_vae_refuses_before_weight_hash_or_archive(checked_settings, monkeypatch):
    settings, _ = checked_settings
    settings.output.mkdir()
    sentinel = settings.output / 'preserved.txt'
    sentinel.write_bytes(b'original run')
    settings.overwrite = True
    monkeypatch.setattr(precompute, 'file_fingerprint', lambda path: 'different VAE')
    monkeypatch.setattr(engine.backbone, 'identity', lambda *a, **k: pytest.fail('invalid VAE reached weights'))
    with pytest.raises(ValueError, match='encoding VAE differs'):
        engine.prepare_run(settings)
    assert sentinel.read_bytes() == b'original run'
    assert not list(settings.output.parent.glob('output.archived*'))
