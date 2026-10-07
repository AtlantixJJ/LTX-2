"""Historical measurements use saved evidence and never execute a model."""

import json

import pytest
import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.hashing import sha256


@pytest.mark.parametrize('long,frames', [(False, 17), (True, 27)])
def test_identical_capture_metrics_and_distinct_guide(long, frames):
    capture = torch.randn(2, frames, 3, 4, generator=torch.Generator().manual_seed(2))
    result = evaluate.saved_latent_metrics(capture, capture, capture + 1, long=long)
    assert result['c0_exact'] and all(value == 0 for value in result['per_block_mse'])
    if long:
        assert result['latent_frames'] == frames
        assert all(value == pytest.approx(1) for value in result['per_block_guide_mse'])
        assert all(value == 1 for value in result['per_block_detail_ratio'])
    else:
        assert result['capture_mse'] == 0 and result['guide_mse'] == pytest.approx(1)
        assert result['motion_ratio'] == result['detail_ratio'] == 1
        assert result['seam_ratio'] == result['capture_seam_ratio']


def test_saved_probe_cli_checks_hash_and_preserves_previous_metrics(tmp_path):
    capture = torch.randn(2, 17, 3, 4, generator=torch.Generator().manual_seed(5))
    for role in ('capture', 'guide'):
        torch.save({'schema_version': 2, 'master': capture, 'fps': 30}, tmp_path / f'{role}.pt')
    encoded = tmp_path / 'generated.pt'
    torch.save(capture.unsqueeze(0), encoded)
    artifacts = {
        'view': '/corpus/Part_1/actor/views/view00', 'seed': 42,
        'capture': str(tmp_path / 'capture.pt'), 'guide': str(tmp_path / 'guide.pt'),
        'epsilon_sha256': 'a' * 64,
        'latents': [{'path': 'generated.pt', 'sha256': sha256(encoded), 'sigma': 0.725, 'arm': 'd1'}],
    }
    manifest = {'seed': 42, 'videos': [{'artifacts': artifacts, 'schedule': [0.725, 0]}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert evaluate.main(['--saved-metrics', str(tmp_path)]) == 0
    result = json.loads((tmp_path / 'metrics.json').read_text())
    assert result['rows'][0]['view'] == 'Part_1/actor/view00'
    assert result['rows'][0]['capture_mse'] == 0
    previous = (tmp_path / 'metrics.json').read_bytes()
    encoded.write_bytes(b'changed output')
    with pytest.raises(ValueError, match='content changed'):
        evaluate.main(['--saved-metrics', str(tmp_path)])
    assert (tmp_path / 'metrics.json').read_bytes() == previous


def test_zero_denominator_is_not_published_as_a_number():
    capture = torch.zeros(2, 17, 3, 4)
    with pytest.raises(ValueError, match='zero denominator'):
        evaluate.saved_latent_metrics(capture, capture, capture)
