"""Actual stock components with controlled x0 predictions; not native weight acceptance."""
import torch
import pytest
import json
from argparse import Namespace

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import stock_parity
from scripts.onestep_avatar.model import common


class Geometry:
    scale_factors = SpatioTemporalScaleFactors(time=8, height=32, width=32)


class ControlledX0(torch.nn.Module):
    num_blocks = 1

    def forward(self, video, audio=None, perturbations=None):
        assert audio is None
        # Depend on current state, text and noise levels so a mismatched call matters.
        denoised = video.latent.float()*0.31 + video.timesteps*0.17 + video.context.float().mean()*0.2
        denoised = torch.where(video.timesteps == 0, video.latent.float(), denoised)
        return denoised.to(video.latent.dtype), None


def inputs():
    grid = common.ClipGrid.build(3, 64, 64, 30.0, Geometry(), device=torch.device('cpu'), dtype=torch.bfloat16,
                                 latent_channels=128)
    image = torch.randn(1, 128, 1, 2, 2, generator=torch.Generator().manual_seed(81)).bfloat16()
    context = torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(82)).bfloat16()
    return grid, image, context


def test_native_components_repeat_and_match_custom_before_endpoint():
    grid, image, context = inputs()
    outputs, traces, noise, timings = stock_parity.sample_paths(
        ControlledX0(), context, grid, image, torch.tensor([1., .725, .421875, 0.]), 91)
    comparison = stock_parity.compare_paths(outputs, traces)
    assert comparison['stock_repeat']['unequal_elements'] == 0
    assert comparison['stock_repeat_calls_exact']
    assert comparison['all_call_inputs_and_predictions_exact']
    assert comparison['terminal_difference_only']
    for index, call in enumerate(comparison['calls']):
        for field in ('latent', 'sigma', 'timesteps', 'positions', 'keyframes_mask'):
            assert call['fields'][field]['unequal_elements'] == 0, (index, field)
    assert len(traces['stock']) == 3
    assert all(item['forward_calls'] == 3 for item in timings.values())
    assert noise.dtype == torch.bfloat16
    for output in outputs.values():
        assert torch.equal(output[:, :4], grid.patchify(image))
    assert comparison['final_outputs']['unequal_elements'] > 0


def test_comparison_exposes_changed_call_and_repeat():
    grid, image, context = inputs()
    outputs, traces, _, _ = stock_parity.sample_paths(
        ControlledX0(), context, grid, image, torch.tensor([1., .5, 0.]), 91)
    traces['bidirectional'][0]['positions'][0, 0, 0, 0] += 1
    outputs['stock_repeat'][0, -1, 0] += 1
    comparison = stock_parity.compare_paths(outputs, traces)
    assert not comparison['terminal_difference_only']
    assert comparison['stock_repeat']['unequal_elements'] > 0
    assert comparison['calls'][0]['fields']['positions']['unequal_elements'] > 0


def test_initial_stock_tokens_use_fixed_noise_and_independent_image():
    grid, image, context = inputs()
    outputs, traces, noise, _ = stock_parity.sample_paths(
        ControlledX0(), context, grid, image, torch.tensor([1., .5, 0.]), 19)
    expected = common.with_clean_prefix(noise, grid.patchify(image))
    for name in outputs:
        assert torch.equal(traces[name][0]['latent'], expected)


def test_differences_reject_nonfinite_and_mismatched_shape():
    with pytest.raises(ValueError, match='finite equal-shaped'):
        stock_parity.differences(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match='finite equal-shaped'):
        stock_parity.differences(torch.tensor([float('nan')]), torch.zeros(1))


def test_current_check_rejects_changed_input_and_software(tmp_path, monkeypatch):
    path = tmp_path/'input.pt'
    path.write_bytes(b'fixed bytes')
    monkeypatch.setattr(stock_parity.software, 'check_current', lambda record: None)
    with pytest.raises(ValueError, match='inputs or weights changed'):
        stock_parity.check_current({str(path): '0'*64}, {})
    monkeypatch.setattr(stock_parity.software, 'check_current',
                        lambda record: (_ for _ in ()).throw(ValueError('changed software')))
    with pytest.raises(ValueError, match='changed software'):
        stock_parity.check_current({}, {})


@pytest.mark.parametrize('defect', ['record_hash', 'file_hash', 'tensor_dtype'])
def test_bad_fixed_text_fails_before_backbone_resolution(tmp_path, monkeypatch, defect):
    image_path, text_path, record_path = (tmp_path/name for name in ('image.pt', 'text.pt', 'preview.json'))
    torch.save({'input_role': 'supplied_image', 'pixel_frames': 1}, image_path)
    monkeypatch.setattr(stock_parity.dataset, 'load_training_master',
                        lambda *args, **kwargs: (torch.zeros(128, 1, 2, 2), 30.0))
    context = torch.ones(1, 2, 8, dtype=torch.float32 if defect == 'tensor_dtype' else torch.bfloat16)
    torch.save(context, text_path)
    fixed = {'kind': 'onestep_avatar.preview_inputs', 'schema_version': 2,
             'software': stock_parity.software.capture('preparation', 'bidirectional'),
             'input_files': {'text': {'path': str(text_path), 'sha256': stock_parity.sha256(text_path),
                                     'tensor_sha256': stock_parity.evaluate.tensor_sha256(context)}}}
    if defect == 'file_hash':
        fixed['input_files']['text']['sha256'] = '0'*64
    fixed['sha256'] = stock_parity.subset.record_hash(fixed)
    if defect == 'record_hash':
        fixed['sha256'] = '0'*64
    record_path.write_text(json.dumps(fixed))
    monkeypatch.setattr(stock_parity.backbone, 'resolve',
                        lambda *args: (_ for _ in ()).throw(AssertionError('backbone reached')))
    args = Namespace(output=tmp_path/'fresh', first_image=image_path, text_record=record_path, steps=4, frames=17)
    with pytest.raises(ValueError, match={'record_hash': 'record hash', 'file_hash': 'text file',
                                         'tensor_dtype': 'text tensor'}[defect]):
        stock_parity.prepare(args)


def test_stock_titles_are_readable_without_browser_upscaling():
    panels = [stock_parity.media.Panel(name, title, torch.zeros(1, 3, 64, 64), (0,))
              for name, title in [('stock', 'Stock video; RGB decoded'), ('stock_repeat', 'Stock repeat; RGB decoded'),
                                  ('bidirectional', 'Bidirectional; RGB decoded')]]
    question, panel_size = 'Does video sampling match stock?', (448, 448)
    layout = stock_parity.media.compact_layout(panels, question=question, layout='comparison', panel_size=panel_size)
    pixels, record = stock_parity.media.render_panels(panels, question=question, layout=layout, fps=30,
                                                     panel_size=panel_size)
    natural_width = record['display_size'][0]
    assert record['font_size']*min(480, natural_width)/natural_width >= 16
    assert len(pixels) == 1
