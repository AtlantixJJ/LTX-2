"""A held-out DNARender actor cannot reappear through another clip, part, view or alias."""

import json
from pathlib import Path

import pytest

from scripts.prune.data import whole_clip
from scripts.prune.score import hooks


def _view(root: Path, part: str, clip: str, *, actor: int | str | None = None) -> Path:
    directory = root / part / clip
    view = directory / "views" / "view00_cam51"
    view.mkdir(parents=True)
    if actor is not None:
        (directory / "meta.json").write_text(json.dumps({"actor": {"id": actor}}))
    return view


@pytest.mark.parametrize(("part", "clip"), [("Part_1", "0008_02"), ("Part_2", "8_01"), ("Part_3", "00008_12")])
def test_same_bare_actor_is_rejected_across_clips_and_parts(tmp_path: Path, part: str, clip: str) -> None:
    calibration = _view(tmp_path, "Part_1", "0008_01")
    heldout = _view(tmp_path, part, clip)
    artifact = tmp_path / "mask.json"
    artifact.write_text(json.dumps({"provenance": {"task": whole_clip.TASK,
                                                   "calibration_views": [str(calibration)]}}))
    with pytest.raises(ValueError, match="actor was used to calibrate"):
        hooks.require_native_heldout_scope(artifact, view=str(heldout), sigmas=[0.725])


def test_metadata_bare_actor_overrides_clip_prefix_and_normalizes_zeros(tmp_path: Path) -> None:
    first = _view(tmp_path, "Part_1", "0012_01", actor="0008")
    second = _view(tmp_path, "Part_2", "0009_01", actor=8)
    prefix = _view(tmp_path, "Part_3", "0008_01")
    assert whole_clip.actor_identity(str(first)) == "dna_actor:8"
    assert whole_clip.actor_identity(str(second)) == whole_clip.actor_identity(str(first))
    assert whole_clip.actor_identity(str(prefix)) == whole_clip.actor_identity(str(first))


def test_symlink_alias_and_other_view_share_the_bare_actor(tmp_path: Path) -> None:
    view = _view(tmp_path, "Part_1", "0008_01")
    other = view.parent / "view01_cam52"
    other.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(view.parent.parent, target_is_directory=True)
    assert whole_clip.actor_identity(str(alias / "views" / "view00_cam51")) == "dna_actor:8"
    assert whole_clip.actor_identity(str(other)) == whole_clip.actor_identity(str(view))


def test_different_actor_is_allowed_by_native_holdout_guard(tmp_path: Path) -> None:
    calibration = _view(tmp_path, "Part_1", "0008_01")
    heldout = _view(tmp_path, "Part_2", "0009_01")
    artifact = tmp_path / "mask.json"
    artifact.write_text(json.dumps({"provenance": {"task": whole_clip.TASK,
                                                   "calibration_views": [str(calibration)],
                                                   "calibration_inputs": [{"capture_sha256": "calibration"}],
                                                   "sigmas": [0.725]}}))
    hooks.require_native_heldout_scope(artifact, view=str(heldout), sigmas=[0.725], baseline={"videos": []})
    assert whole_clip.actor_identity(str(calibration)) != whole_clip.actor_identity(str(heldout))


def test_generic_fixture_names_retain_unambiguous_directory_fallback(tmp_path: Path) -> None:
    first = _view(tmp_path, "generic", "subject")
    second = _view(tmp_path, "generic", "other_subject")
    assert whole_clip.actor_identity(str(first)) == str(first.parent.parent.resolve())
    assert whole_clip.actor_identity(str(first)) != whole_clip.actor_identity(str(second))


def test_existing_malformed_actor_metadata_is_rejected(tmp_path: Path) -> None:
    view = _view(tmp_path, "Part_1", "0008_01")
    (view.parent.parent / "meta.json").write_text(json.dumps({"actor": {"id": True}}))
    with pytest.raises(ValueError, match="invalid DNARender actor"):
        whole_clip.actor_identity(str(view))
