"""Package fusion execution preserves all five paths and publishes only complete evidence."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.corpus import dataset


def test_package_block_matches_original_grid_and_noise_on_real_transformer():
    from scripts.onestep_avatar.model import causal, common
    from scripts.onestep_avatar.tests.test_causal_core import CHANNELS, SCALE, _context, _model
    from scripts.onestep_avatar.training import engine
    from scripts.prune.core.session import DTYPE

    model = _model().to(dtype=DTYPE)
    capture = torch.randn(CHANNELS, 7, 2, 2, generator=torch.Generator().manual_seed(9))
    guide = torch.randn(CHANNELS, 7, 2, 2, generator=torch.Generator().manual_seed(10))
    session = SimpleNamespace(device=torch.device('cpu'), context=_context().to(dtype=DTYPE),
                              model=SimpleNamespace(scale_factors=SCALE, caps=SimpleNamespace(latent_channels=CHANNELS)))
    geometry = causal.deployed_geometry(SCALE)
    chain = SimpleNamespace(z_y=capture, fps=30)
    grid = engine.clip_grid_for(chain, geometry, device=session.device, latent_channels=CHANNELS)
    span = geometry.plan(grid.latent_frames)[0]
    lo, hi = grid.token_span(*span)
    target = grid.patchify(capture.unsqueeze(0).to(dtype=DTYPE))
    source = grid.patchify(guide.unsqueeze(0).to(dtype=DTYPE))
    c0 = target[:, :grid.tokens_per_latent_frame]
    cache = causal.BlockCache.allocate(grid, geometry, num_layers=len(model.transformer_blocks),
                                      inner_dim=model.inner_dim, device=session.device, dtype=DTYPE)
    noisy = common.with_clean_prefix(common.noise_block(source[:, lo:hi], 0.421875, 42), c0)
    expected = causal.fusion_parity_block(model, 'velocity', grid, cache, noisy, session.context,
                                        0.421875, span, clean_prefix_tokens=c0.shape[1]).float().cpu()
    actual = evaluate.fusion_probe_block(model, 'velocity', session, capture, guide, 30)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, :c0.shape[1]], c0.float())


def test_fusion_orchestration_preserves_paths_and_result_values(tmp_path, monkeypatch):
    import peft

    import ltx_trainer.model_loader as loaders
    import scripts.prune.core.session as sessions

    run, view = tmp_path / 'run', tmp_path / 'view'
    (run / 'checkpoints').mkdir(parents=True)
    view.mkdir()
    (run / 'config.json').write_text(json.dumps({'lora_rank': 16, 'lora_alpha': 16, 'lora_target': 'attn'}))
    dev = tmp_path / 'dev.safetensors'
    paths = [dev, run / 'checkpoints/lora_weights_step_00000.safetensors',
             run / 'checkpoints/lora_weights_step_00001.safetensors',
             view / dataset.capture_bundle_name('white'), view / dataset.guide_bundle_name('white')]
    for path in paths:
        path.write_bytes(b'controlled fixture')
    capture, guide = torch.zeros(3, 3, 2, 2), torch.ones(3, 3, 2, 2)
    monkeypatch.setattr(evaluate.backbone, 'transformer_path', lambda *_args: dev)
    monkeypatch.setattr(dataset, 'load_training_master', lambda path: (
        capture if path == paths[-2] else guide, 30))
    events = []

    class Model:
        def __init__(self, name):
            self.name = name

        def requires_grad_(self, value):
            assert value is False

        def eval(self):
            return self

    def transformer(path, *, loras):
        assert path == dev
        name = 'bare' if not loras else 'step0' if '00000' in str(loras[0].path) else 'fused1'
        events.append(('fused_load', name))
        return nullcontext(Model(name))

    monkeypatch.setattr(sessions, 'open_session', lambda *_args, **_kwargs: SimpleNamespace(
        device=torch.device('cpu'), transformer=transformer))
    monkeypatch.setattr(loaders, 'load_transformer', lambda **_kwargs: Model('unmerged'))
    monkeypatch.setattr(peft, 'get_peft_model', lambda model, _config: model)

    def load_adapter(model, path):
        model.name = 'peft0' if '00000' in str(path) else 'peft1'
        events.append(('peft_load', model.name))

    monkeypatch.setattr(evaluate.checkpoints, 'load_stage_init', load_adapter)
    values = {'bare': 1., 'step0': 1., 'fused1': 1.18, 'peft0': 1., 'peft1': 1.2}

    def probe(model, kind, session, target, source, fps):
        assert target is capture and source is guide and fps == 30
        events.append(('sample', model.name, kind))
        return torch.full((1, 3, 4), values[model.name])

    monkeypatch.setattr(evaluate, 'fusion_probe_block', probe)
    output = tmp_path / 'diagnostic.json'
    result = evaluate.evaluate_fusion_parity(run, view, output, gpu_id=0)
    assert [event for event in events if event[0] == 'sample'] == [
        ('sample', 'bare', 'x0'), ('sample', 'step0', 'x0'), ('sample', 'fused1', 'x0'),
        ('sample', 'peft0', 'velocity'), ('sample', 'peft1', 'velocity'),
    ]
    assert result['step0_equals_bare_bitwise']
    assert result['rel_l2_adapter_effect_peft'] == pytest.approx(0.2)
    assert result['rel_l2_adapter_effect_fused'] == pytest.approx(0.18)
    assert result['rel_l2_effect_fused_vs_effect_peft'] == pytest.approx(0.1, abs=1e-6)
    assert result['fused_vs_peft_within_tolerance']
    assert len(result['input_sha256']) == 6
    assert json.loads(output.read_text()) == result


def test_zero_adapter_effect_is_undefined_and_cannot_pass():
    outputs = {name: torch.ones(1, 3) for name in ['bare', 'step0', 'fused1', 'peft0', 'peft1']}
    result = evaluate.fusion_parity_metrics(outputs)
    assert result['rel_l2_effect_fused_vs_effect_peft'] is None
    assert result['effect_ratio_status'] == 'undefined_zero_peft_effect'
    assert not result['fused_vs_peft_within_tolerance']
    json.dumps(result, allow_nan=False)


def test_existing_fusion_result_is_preserved_before_loading(tmp_path, monkeypatch):
    import scripts.prune.core.session as sessions

    output = tmp_path / 'result.json'
    output.write_bytes(b'original scientific result')
    monkeypatch.setattr(sessions, 'open_session', lambda *_args, **_kwargs: pytest.fail('opened a session'))
    with pytest.raises(ValueError, match='fresh output'):
        evaluate.evaluate_fusion_parity(tmp_path, tmp_path, output, gpu_id=0)
    assert output.read_bytes() == b'original scientific result'
