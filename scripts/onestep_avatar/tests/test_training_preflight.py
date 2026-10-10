"""New-runtime preflight checks real input records without model or output mutation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.onestep_avatar.corpus import precompute
from scripts.onestep_avatar.training import engine


def test_preflight_checks_both_inputs_without_creating_output(checked_settings: tuple) -> None:
    settings, membership = checked_settings
    store, plan, _, archive = engine.prepare_run(settings)
    assert len(store) == 2
    assert plan["mode"] == "bidirectional"
    assert plan["membership_sha256"] == membership["sha256"]
    assert archive is False
    assert not settings.output.exists()


def test_bad_guide_fails_before_base_resolution_or_output_mutation(
    checked_settings: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = checked_settings
    settings.guide_mode = "d1"
    monkeypatch.setattr(engine.backbone, "resolve", lambda *_args: pytest.fail("missing guide reached base resolution"))
    with pytest.raises(ValueError, match="guide content hash is not recorded"):
        engine.prepare_run(settings)
    assert not settings.output.exists()


def test_overwrite_preflight_preserves_used_output(checked_settings: tuple) -> None:
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


def test_wrong_mode_record_fails_before_any_file_read(checked_settings: tuple) -> None:
    settings, _ = checked_settings
    settings.mode = "causal"
    settings.subset = Path("/file/that/does/not/exist")
    with pytest.raises(ValueError, match="typed mode settings disagree"):
        engine.prepare_run(settings)


def test_distilled_off_grid_level_is_rejected(checked_settings: tuple) -> None:
    settings, _ = checked_settings
    settings.variant = "distilled"
    settings.sigma0 = 0.6
    with pytest.raises(ValueError, match="distilled base grid"):
        engine.prepare_run(settings)
    assert not settings.output.exists()


def test_wrong_vae_refuses_before_weight_hash_or_archive(
    checked_settings: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = checked_settings
    settings.output.mkdir()
    sentinel = settings.output / "preserved.txt"
    sentinel.write_bytes(b"original run")
    settings.overwrite = True
    monkeypatch.setattr(precompute, "file_fingerprint", lambda _path: "different VAE")
    monkeypatch.setattr(engine.backbone, "identity", lambda *_a, **_k: pytest.fail("invalid VAE reached weights"))
    with pytest.raises(ValueError, match="encoding VAE differs"):
        engine.prepare_run(settings)
    assert sentinel.read_bytes() == b"original run"
    assert not list(settings.output.parent.glob("output.archived*"))
