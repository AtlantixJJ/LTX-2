"""Saved-only media gate, six-panel synchronization and readable rendering controls."""
import json

import pytest
import torch

from scripts.onestep_avatar import media
from scripts.prune.core.provenance import file_sha256
from scripts.prune.evaluate.mobilewan_media import (
    comparison_panels,
    preserve_source,
    read_evaluation,
    render_comparison,
    verify_comparison,
)


def test_six_panels_preserve_roles_and_full_source_frames():
    recorded = torch.full((3, 3, 24, 48), 127, dtype=torch.uint8)
    decoded = torch.full((3, 3, 48, 24), 64, dtype=torch.uint8)
    predictions = [torch.full((3, 3, 32, 32), i*32, dtype=torch.uint8) for i in range(4)]
    panels = comparison_panels(recorded, decoded, predictions)
    pixels, record = render_comparison(panels, 30, common={'sigma': .4})
    assert pixels.dtype == torch.uint8 and len(pixels) == 3
    assert record['source_frames'] == [0, 1, 2]
    assert record['source_times'] == [0., 1/30, 2/30]
    assert [(p['row'], p['column']) for p in record['panels']] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert [p['title'] for p in record['panels']] == ['Capture RGB', 'Capture VAE', 'D0 / 0%', 'D0 / 10%', 'D0 / 23%', 'D0 / 33%']
    assert record['padding'] == 'aspect-preserving contain with neutral padding'
    assert record['font_size'] * record['viewing_width'] / record['display_size'][0] >= 16
    with pytest.raises(ValueError, match='coverage'):
        comparison_panels(recorded, decoded[:2], predictions)


def test_incomplete_evidence_fails_before_decoder(tmp_path):
    (tmp_path / 'evaluation').mkdir()
    (tmp_path / 'evaluation/results.json').write_text(json.dumps({'status': 'partial', 'prediction_count': 95}))
    with pytest.raises(ValueError, match='96-prediction'):
        read_evaluation(tmp_path, {})


def test_preserved_reference_matches_full_bytes_and_refuses_changed_source(tmp_path):
    source = tmp_path / 'mask.mp4'
    source.write_bytes(b'full recorded matte source')
    destination = tmp_path / 'media/source_mask.mp4'
    reference = preserve_source(source, destination)
    assert reference['sha256'] == file_sha256(source) == file_sha256(destination)
    assert preserve_source(source, destination) == reference
    source.write_bytes(b'changed matte source')
    with pytest.raises(ValueError, match='preserved source reference differs'):
        preserve_source(source, destination)
    assert destination.read_bytes() == b'full recorded matte source'
    assert not list(destination.parent.glob('.source-*'))


def test_audited_source_hash_checked_before_copy(tmp_path):
    source = tmp_path / 'rgb.mp4'
    source.write_bytes(b'changed capture RGB')
    destination = tmp_path / 'media/source_rgb.mp4'
    with pytest.raises(ValueError, match='audited content'):
        preserve_source(source, destination, '0'*64)
    assert not destination.exists()


def test_actual_comparison_mp4_and_png_are_probed(tmp_path):
    pixels = torch.full((3, 3, 24, 48), 127, dtype=torch.uint8)
    panels = comparison_panels(pixels, pixels, [pixels]*4)
    rendered, record = render_comparison(panels, 30, common={'scope': 'CPU synthetic media fixture'})
    checked = verify_comparison(media.save_render(rendered, record, tmp_path))
    assert int(checked['outputs']['video']['verification']['nb_read_frames']) == 3
    assert checked['outputs']['video']['verification']['avg_frame_rate'] == '30/1'
    checked['source_frames'] = [0, 1]
    with pytest.raises(ValueError, match='frame count or playback rate differs'):
        verify_comparison(checked)
