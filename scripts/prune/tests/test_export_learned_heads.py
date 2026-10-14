import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.prune.score.export_learned_heads import masks_from_tensor, merge
from scripts.prune.score.export_pruned import PREFIX


def test_merge_algebra_inventory_and_reload(tmp_path):
    path, output = tmp_path/'base.safetensors', tmp_path/'merged.safetensors'
    v, o = f'{PREFIX}.0.attn1.to_v.weight', f'{PREFIX}.0.attn1.to_out.0.weight'
    tensors = {v: torch.randn(4, 8).to(torch.bfloat16), o: torch.randn(8, 4).to(torch.bfloat16), 'unchanged': torch.randn(3)}
    save_file(tensors, str(path), metadata={'config': json.dumps({'transformer': {'pruning': {}}})})
    state = {'0.attn1.v.A': torch.randn(2, 8), '0.attn1.v.B': torch.randn(4, 2),
             '0.attn1.out.A': torch.randn(2, 4), '0.attn1.out.B': torch.randn(8, 2)}
    merge(path, state, output)
    with safe_open(output, framework='pt', device='cpu') as handle:
        for key, name in ((v, 'v'), (o, 'out')):
            expected = (tensors[key].float()+state[f'0.attn1.{name}.B']@state[f'0.attn1.{name}.A']).to(torch.bfloat16)
            assert torch.equal(handle.get_tensor(key), expected)
        assert torch.equal(handle.get_tensor('unchanged'), tensors['unchanged'])
    mask = masks_from_tensor(torch.tensor([[1., 0., 1., 0.]]), {'0.attn1': 4, '0.attn2': 4})
    assert mask['0.attn2'] == [1]*4
    with pytest.raises(ValueError, match='mask'):
        masks_from_tensor(torch.full((1, 4), .5), {'0.attn1': 4})


def test_gt_export_supplies_missing_inputs_without_model_or_rescoring(tmp_path, monkeypatch):
    import hashlib

    from scripts.prune.core.provenance import file_sha256
    from scripts.prune.data import whole_clip
    from scripts.prune.learn import heads
    from scripts.prune.score.export_learned_heads import geometry_wrappers, prepare_geometry_inputs
    from scripts.prune.tests.test_whole_clip_baseline import _Session, _vae, _views

    current = _Session(_vae(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError('structural input preparation must not load a model')

    monkeypatch.setattr(current, 'transformer', forbidden)
    monkeypatch.setattr(heads, 'open_current', lambda *args: current)
    sources = []
    for view, frames in zip(_views(tmp_path / 'corpus'), (18, 28), strict=True):
        capture = view / 'ltx_vae_latent_white.pt'
        bundle = torch.load(capture, map_location='cpu', weights_only=True)
        bundle['master'] = torch.zeros(bundle['master'].shape[0], frames, 2, 2, dtype=torch.bfloat16)
        torch.save(bundle, capture)
        sources.append({
            'capture': str(capture), 'capture_latent_sha256': file_sha256(capture),
            'shape': list(bundle['master'].shape), 'fps': bundle['fps'], 'resolved_dir': str(view),
            'relative_dir': view.relative_to(tmp_path / 'corpus').as_posix(), 'n_latent_frames': frames,
            'capture_encode_record': {'schema_version': bundle['schema_version']},
        })
    study = {'root': str(tmp_path / 'study'), 'splits': {'learning': sources}, 'config': {'sigmas': [.9, .4]}}
    first = tmp_path / 'study/inputs' / hashlib.sha256(sources[0]['relative_dir'].encode()).hexdigest() / 'seed_42'
    whole_clip.prepare_score_inputs(first, sources[0], current=current, sigmas=[.9, .4], seed=42)
    original = (first / 'epsilon.pt').read_bytes()
    prepare_geometry_inputs(study, 0)
    assert (first / 'epsilon.pt').read_bytes() == original
    wrappers = geometry_wrappers(study, {'0.attn1': [1, 0]}, tmp_path / 'masks')
    assert set(wrappers) == {'18', '28'}
    for wrapper in wrappers.values():
        record = json.loads(open(wrapper['path']).read())
        assert record['provenance']['seed'] == 42
        whole_clip.validate_native_provenance(record['provenance'])
    monkeypatch.setattr(heads, 'open_current', forbidden)
    prepare_geometry_inputs(study, 0)
