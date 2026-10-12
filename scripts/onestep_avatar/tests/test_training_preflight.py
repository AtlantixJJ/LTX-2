"""New-runtime preflight checks real input records without model or output mutation."""

from __future__ import annotations

import os
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


def test_encoding_vae_accepts_existing_alias_with_unchanged_metadata(tmp_path: Path) -> None:
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"VAE weights")
    alias = tmp_path / "old_vae.safetensors"
    alias.symlink_to(vae)
    stat = vae.stat()
    expected = f"{vae}:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
    recorded = f"{alias}:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
    assert engine._encoding_vae_matches(recorded, expected)
    assert not engine._encoding_vae_matches(recorded.replace(f"mtime_ns={stat.st_mtime_ns}", "mtime_ns=0"), expected)
    assert not engine._encoding_vae_matches(recorded.replace(f"size={stat.st_size}", "size=0"), expected)
    alias.unlink()
    assert not engine._encoding_vae_matches(recorded, expected)


def test_encoding_vae_rejects_different_file_with_equal_size_and_time(tmp_path: Path) -> None:
    vae = tmp_path / "vae.safetensors"
    other = tmp_path / "different.safetensors"
    vae.write_bytes(b"VAE weights")
    other.write_bytes(b"bad weights")
    stat = vae.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert other.stat().st_size == stat.st_size
    expected = f"{vae}:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
    recorded = f"{other}:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
    assert not engine._encoding_vae_matches(recorded, expected)


@pytest.mark.parametrize("recorded", [None, "", "malformed", "relative:size=1:mtime_ns=2"])
def test_encoding_vae_rejects_absent_or_malformed_identity(recorded: object) -> None:
    assert not engine._encoding_vae_matches(recorded, "/vae:size=1:mtime_ns=2")
