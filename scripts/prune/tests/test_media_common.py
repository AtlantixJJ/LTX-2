"""Shared media helpers: preserved source bytes and probed comparison outputs."""

import pytest
import torch

from scripts.onestep_avatar import media
from scripts.prune.core.provenance import file_sha256
from scripts.prune.evaluate.media_common import preserve_source, verify_comparison


def test_preserved_reference_matches_full_bytes_and_refuses_changed_source(tmp_path):
    source = tmp_path / "mask.mp4"
    source.write_bytes(b"full recorded matte source")
    destination = tmp_path / "media/source_mask.mp4"
    reference = preserve_source(source, destination)
    assert reference["sha256"] == file_sha256(source) == file_sha256(destination)
    assert preserve_source(source, destination) == reference
    source.write_bytes(b"changed matte source")
    with pytest.raises(ValueError, match="preserved source reference differs"):
        preserve_source(source, destination)
    assert destination.read_bytes() == b"full recorded matte source"
    assert not list(destination.parent.glob(".source-*"))


def test_audited_source_hash_checked_before_copy(tmp_path):
    source = tmp_path / "rgb.mp4"
    source.write_bytes(b"changed capture RGB")
    destination = tmp_path / "media/source_rgb.mp4"
    with pytest.raises(ValueError, match="audited content"):
        preserve_source(source, destination, "0" * 64)
    assert not destination.exists()


def test_actual_comparison_mp4_and_png_are_probed(tmp_path):
    pixels = torch.full((3, 3, 24, 48), 127, dtype=torch.uint8)
    frames = tuple(range(3))
    panels = [media.Panel(f"p{i}", f"Panel {i}", pixels, frames, value="VAE decoding") for i in range(6)]
    rendered, record = media.render_panels(
        panels,
        question="Does the shared probe verify saved media?",
        layout="comparison",
        fps=30,
        panel_size=(384, 384),
        viewing_width=480,
        poster_frame=2,
        common_settings={"scope": "CPU fixture"},
    )
    checked = verify_comparison(media.save_render(rendered, record, tmp_path))
    assert int(checked["outputs"]["video"]["verification"]["nb_read_frames"]) == 3
    assert checked["outputs"]["video"]["verification"]["avg_frame_rate"] == "30/1"
    checked["source_frames"] = [0, 1]
    with pytest.raises(ValueError, match="frame count or playback rate differs"):
        verify_comparison(checked)
