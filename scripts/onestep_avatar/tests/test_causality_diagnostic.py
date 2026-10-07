"""Eight-block diagnostics preserve full-master noise and historical execution."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import dataset, evaluate, visualize_d0, visualize_d1
from scripts.onestep_avatar.model import causal
from scripts.onestep_avatar.training import engine


@pytest.mark.parametrize('frames', [17, 21])
def test_matches_original_two_rollouts_and_actual_call_counts(frames, monkeypatch):
    from ltx_core.model.transformer.model import X0Model
    from scripts.prune.core.session import DTYPE
    from scripts.onestep_avatar.tests.test_causal_core import CHANNELS, SCALE, _context, _model

    model = X0Model(_model().to(dtype=DTYPE))
    capture = torch.randn(CHANNELS, frames, 2, 2, generator=torch.Generator().manual_seed(8))
    guide = torch.randn(CHANNELS, frames, 2, 2, generator=torch.Generator().manual_seed(9))
    session = SimpleNamespace(device=torch.device('cpu'), context=_context().to(dtype=DTYPE),
                              model=SimpleNamespace(scale_factors=SCALE, caps=SimpleNamespace(latent_channels=CHANNELS)))
    chain = SimpleNamespace(z_y=capture, z_g=guide, fps=30)
    geometry = causal.deployed_geometry(SCALE)
    grid = engine.clip_grid_for(chain, geometry, device=session.device, latent_channels=CHANNELS)
    plan = geometry.plan(frames)[:8]
    target = grid.patchify(capture.unsqueeze(0).to(dtype=DTYPE))
    _, eps_a = visualize_d1.global_epsilons(target, grid, plan, 42)
    _, eps_b = visualize_d1.global_epsilons(target, grid, plan, 99)
    expected = []
    with torch.no_grad():
        for epsilon in (eps_a, eps_a[:4] + eps_b[4:]):
            _, output = visualize_d0.run_chain(model, session.context, chain, geometry, 0.421875,
                device=session.device, latent_channels=CHANNELS, seed=42, guide_mode='d1',
                block_epsilons=epsilon, max_blocks=8)
            expected.append(output.float())
    original_sample = causal.sample
    actual = []

    def sample(*args, **kwargs):
        assert not torch.is_grad_enabled()
        tokens, counts = original_sample(*args, **kwargs)
        actual.append(grid.unpatchify_block(tokens[:, :17 * grid.tokens_per_latent_frame], 17).float())
        return tokens, counts

    monkeypatch.setattr(causal, 'sample', sample)
    result = evaluate.causality_probe(model, session, capture, guide, 30, 0.421875)
    assert len(actual) == 2
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert result['latent_frames_compared_equal'] == [0, 9]
    assert result['earlier_blocks_bit_identical']
    assert result['later_blocks_changed'] and result['later_blocks_max_abs_diff'] > 0
    assert all(record['call_counts']['model_calls'] == 16 for record in result['records'])
    assert all(record['call_counts']['denoise_calls'] == 8 for record in result['records'])
    assert all(record['call_counts']['refresh_calls'] == 8 for record in result['records'])
    assert result['records'][0]['noise_sha256'] != result['records'][1]['noise_sha256']


def fixture_inputs(tmp_path, monkeypatch, frames=17):
    view = tmp_path / 'view'
    view.mkdir()
    dev, adapter = tmp_path / 'dev.safetensors', tmp_path / 'adapter.safetensors'
    for path in (dev, adapter, view / dataset.capture_bundle_name('white'), view / dataset.guide_bundle_name('white')):
        path.write_bytes(b'controlled scientific input')
    capture, guide = torch.zeros(8, frames, 2, 2), torch.ones(8, frames, 2, 2)
    monkeypatch.setattr(evaluate.backbone, 'transformer_path', lambda *_args: dev)
    monkeypatch.setattr(evaluate.backbone, 'identity', lambda *_args: {'fixture': 'dev'})
    monkeypatch.setattr(evaluate.checkpoints, 'read_adapter_metadata', lambda _path: {'fixture': 'adapter'})
    monkeypatch.setattr(dataset, 'load_training_master', lambda path: (
        capture if path.name == dataset.capture_bundle_name('white') else guide, 30))
    return dev, adapter, view, capture, guide


def test_checked_orchestration_and_atomic_saved_fields(tmp_path, monkeypatch):
    import scripts.prune.core.session as sessions

    dev, adapter, view, capture, guide = fixture_inputs(tmp_path, monkeypatch)
    events = []

    def check(meta, **conditions):
        assert meta == {'fixture': 'adapter'}
        assert conditions['schedule'] == [0.421875, 0.0]
        assert conditions['geometry'] == {'block_latent_frames': 2, 'context_latent_frames': 8, 'sink_latent_frames': 1}
        assert conditions['objective'] == 'white' and conditions['guide_mode'] == 'd1'
        assert conditions['teacher_forcing'] is False
        events.append('check')

    def open_session(*_args, **_kwargs):
        assert events == ['check']
        events.append('open')
        return SimpleNamespace(transformer=lambda path, loras: nullcontext(object()))

    def probe(_transformer, _session, target, source, fps, sigma):
        assert target is capture and source is guide and fps == 30 and sigma == 0.421875
        return {'shared_noise_blocks': [0, 1, 2, 3], 'changed_noise_blocks': [4, 5, 6, 7],
                'latent_frames_compared_equal': [0, 9], 'earlier_blocks_bit_identical': True,
                'later_blocks_max_abs_diff': 0.5}

    monkeypatch.setattr(evaluate.checkpoints, 'check_adapter_conditions', check)
    monkeypatch.setattr(sessions, 'open_session', open_session)
    monkeypatch.setattr(evaluate, 'causality_probe', probe)
    output = tmp_path / 'diagnostic.json'
    result = evaluate.evaluate_causality(adapter, view, output, gpu_id=0, sigma=0.421875)
    assert result['checkpoint'] == str(adapter) and result['view'] == str(view)
    assert len(result['input_sha256']) == 4
    assert result['earlier_blocks_bit_identical']
    assert json.loads(output.read_text()) == result


def test_short_master_fails_before_session(tmp_path, monkeypatch):
    import scripts.prune.core.session as sessions

    _, adapter, view, _, _ = fixture_inputs(tmp_path, monkeypatch, frames=15)
    monkeypatch.setattr(sessions, 'open_session', lambda *_args, **_kwargs: pytest.fail('opened weights'))
    with pytest.raises(ValueError, match='eight complete blocks'):
        evaluate.evaluate_causality(adapter, view, tmp_path / 'result.json', gpu_id=0, sigma=0.421875)


def test_existing_result_preserved(tmp_path):
    output = tmp_path / 'result.json'
    output.write_bytes(b'original evidence')
    with pytest.raises(ValueError, match='fresh output'):
        evaluate.evaluate_causality(tmp_path, tmp_path, output, gpu_id=0, sigma=0.421875)
    assert output.read_bytes() == b'original evidence'
