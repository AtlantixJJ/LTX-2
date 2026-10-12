"""Independent GT metrics and tiny native matched-removal/resume checks."""
import hashlib
import json

import pytest
import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.evaluate.mobilewan_removal import evaluate_case, metrics, save_tensor
from scripts.prune.tests.test_whole_clip_baseline import _Session, _vae, _views


def test_mse_excludes_clean_frame_and_preserves_times():
    capture = torch.zeros(1, 2, 4, 2, 3)
    prediction = capture.clone()
    prediction[:, :, 0] = 100
    prediction[:, :, 1:] = torch.tensor([1., 2., 3.]).view(1, 1, 3, 1, 1)
    result = metrics(prediction, capture, 30, 8)
    assert result['per_frame_mse'] == [1., 4., 9.]
    assert result['mse_to_gt'] == pytest.approx(14/3)
    assert result['times_seconds'] == [8/30, 16/30, 24/30]
    assert result['clean_frame_equal'] is False


def test_saved_tensor_refuses_different_value(tmp_path):
    path = tmp_path / 'value.pt'
    save_tensor(path, torch.ones(3))
    save_tensor(path, torch.ones(3))
    with pytest.raises(ValueError, match='differs'):
        save_tensor(path, torch.zeros(3))


def test_all_arms_share_inputs_and_resume_verified_tensors(tmp_path):
    view = _views(tmp_path / 'corpus')[0]
    capture = view / 'ltx_vae_latent_white.pt'
    bundle = torch.load(capture, map_location='cpu', weights_only=True)
    source = {'capture': str(capture), 'capture_latent_sha256': file_sha256(capture),
              'shape': list(bundle['master'].shape), 'fps': bundle['fps'], 'resolved_dir': str(view),
              'capture_encode_record': {'schema_version': 2}, 'actor': '8', 'split': 'held_out',
              'relative_dir': 'Part_1/0008_01/views/view00'}
    current = _Session(_vae(tmp_path))
    budgets = [{'requested_fraction': f, 'realized_fraction': (0 if f == 0 else .25),
                'removed_count': (0 if f == 0 else 1),
                'masks': {'0': [1, 1] if f == 0 else [0, 1], '1': [1, 1]}}
               for f in (0., .1, .23, .33)]
    frozen = {'allocation': {'budgets': budgets}}
    run = tmp_path / 'run'
    with current.transformer() as model:
        calls = []
        handle = model.register_forward_pre_hook(lambda *_: calls.append(1))
        result = evaluate_case(model, current, source, {}, run, frozen, {'pin': 'a'})
        assert len(calls) == 8*5
        assert len(result['sigmas']) == 8
        assert all(row['all_one_equal'] for row in result['sigmas'])
        assert all(arm['clean_frame_equal'] for row in result['sigmas'] for arm in row['arms'])
        calls.clear()
        assert evaluate_case(model, current, source, {}, run, frozen, {'pin': 'a'}) == result
        assert not calls
        path = run / result['sigmas'][0]['arms'][1]['prediction']
        tensor = torch.load(path, weights_only=True)
        torch.save(tensor+1, path)
        with pytest.raises(ValueError, match='prediction changed'):
            evaluate_case(model, current, source, {}, run, frozen, {'pin': 'a'})
        handle.remove()
    key = hashlib.sha256(source['relative_dir'].encode()).hexdigest()
    row = json.loads((run / 'evaluation/records' / key / 'sigma_0.json').read_text())
    assert row['identity']['epsilon_sha256'] == file_sha256(run / 'inputs/views' / key / 'epsilon.pt')
