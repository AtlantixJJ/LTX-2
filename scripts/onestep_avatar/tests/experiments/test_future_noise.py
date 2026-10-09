"""Future-noise protocols retain native CPU controls and complete byte-bound verification."""

import json
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.experiments import causality
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.tests import test_evaluation_completion
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- fixture dependency
from scripts.onestep_avatar.tests.test_training_preflight import checked_settings  # noqa: F401 -- fixture dependency
from scripts.onestep_avatar.training.config import CausalSettings

# Expose the original pytest fixture without shadowing an imported function name.
completed = test_evaluation_completion.completed


def test_future_noise_probe_preserves_completed_causal_blocks():
    model = _model()
    grid = _grid(_geometry())
    capture = torch.zeros(1, 28, 8)
    noise = torch.ones_like(capture)
    changed = noise.clone()
    changed[:, 12:] = -1
    outputs, record = causality.probe_future_noise(
        model, torch.zeros(1, 3, 16), grid, capture, capture, noise, changed,
        change_start_frame=3, mode='causal', mode_settings=CausalSettings(),
        guide_mode='d1', schedule=[0.725, 0], seed=42,
        predict_x0=common.denoised_from_velocity_model(model),
    )
    assert record['earlier_output_bit_identical']
    assert record['earlier_output_max_abs_delta'] == 0
    assert record['later_output_max_abs_delta'] > 0
    assert record['records'][0]['noise_sha256'] != record['records'][1]['noise_sha256']
    assert torch.equal(outputs[0][:, :, :3], outputs[1][:, :, :3])


@pytest.mark.parametrize('invalid', ['nan_later', 'earlier_change', 'identical', 'dtype', 'source_shape', 'inside_block'])
def test_future_noise_invalid_inputs_fail_before_either_sampling(invalid, monkeypatch):
    grid = _grid(_geometry())
    capture = torch.zeros(1, 28, 8)
    noise = torch.ones_like(capture)
    changed = noise.clone()
    changed[:, 12:] = -1
    boundary = 3
    if invalid == 'nan_later':
        changed[:, 12:] = float('nan')
    elif invalid == 'earlier_change':
        changed[:, 0] = 2
    elif invalid == 'identical':
        changed = noise.clone()
    elif invalid == 'dtype':
        changed = changed.double()
    elif invalid == 'source_shape':
        noise, changed = noise[:, :20], changed[:, :20]
    else:
        boundary = 2
    monkeypatch.setattr(
        causality.evaluate, 'sample_case', lambda *a, **k: pytest.fail('invalid probe executed sampling')
    )
    with pytest.raises(ValueError):
        causality.probe_future_noise(
            None, torch.zeros(1, 3, 16), grid, capture, capture, noise, changed,
            change_start_frame=boundary, mode='causal', mode_settings=CausalSettings(), guide_mode='d1',
            schedule=[0.725, 0], seed=42,
        )


@pytest.mark.parametrize('options', [
    ['--future-noise-start', '3'],
    ['--changed-noise-file', 'changed.pt'],
    ['--changed-noise-file', 'changed.pt', '--future-noise-start', '3'],
    ['--noise-file', 'original.pt', '--changed-noise-file', 'changed.pt', '--future-noise-start', '0'],
])
def test_future_noise_cli_requires_complete_saved_inputs(options):
    with pytest.raises(SystemExit):
        causality.parse_future_args([
            '--mode', 'causal', '--subset', 'unused', '--output', 'unused',
            '--schedule', '0.725', '0', *options,
        ])


@pytest.mark.parametrize('bad_prefix', [False, True])
def test_future_noise_preflight_checks_saved_pair(checked_settings, tmp_path, bad_prefix):
    settings, membership = checked_settings
    source = membership['sources'][0]['relative_dir']
    from scripts.onestep_avatar.corpus import dataset
    video = dataset.ClipStore(membership).load(source, require_guide=False)
    original = torch.zeros(1, 7 * video.z_y.shape[2] * video.z_y.shape[3], video.z_y.shape[0], dtype=torch.bfloat16)
    changed = original.clone()
    changed[:, (0 if bad_prefix else 3 * video.z_y.shape[2] * video.z_y.shape[3]):] = 1
    left, right = tmp_path / 'original.pt', tmp_path / 'changed.pt'
    torch.save(original, left)
    torch.save(changed, right)
    args = causality.parse_future_args([
        '--mode', 'causal', '--subset', str(settings.subset), '--output', str(settings.output),
        '--variant', 'dev', '--guide-mode', 'd0', '--source', source,
        '--schedule', '0.725', '0', '--span-latent-frames', '7',
        '--noise-file', str(left), '--changed-noise-file', str(right), '--future-noise-start', '3',
    ])
    if bad_prefix:
        with pytest.raises(ValueError, match='changed earlier noise'):
            causality.prepare_evaluation(args)
    else:
        causality.prepare_evaluation(args)
        assert torch.equal(args.changed_noise, changed)
    assert not settings.output.exists()


def test_future_noise_publication_preserves_distinct_noise_and_shared_provenance(tmp_path):
    outputs = [torch.zeros(1, 2, 7, 1, 1), torch.ones(1, 2, 7, 1, 1)]
    diagnostic = {'records': [{'noise_sha256': 'a' * 64}, {'noise_sha256': 'b' * 64}], 'earlier_output_bit_identical': False}
    provenance = {'source': 'fixed/view', 'adapter_sha256': 'c' * 64, 'fps': 30}
    completed = causality.save_future_noise_probe(outputs, diagnostic, provenance, tmp_path / 'probe')
    assert 'output' not in diagnostic['records'][0]
    for index, record in enumerate(completed['records']):
        assert record['source'] == 'fixed/view' and record['adapter_sha256'] == 'c' * 64
        assert record['noise_sha256'] == ('a' if index == 0 else 'b') * 64
        assert torch.equal(torch.load(record['output']['path'], weights_only=True), outputs[index])
    assert json.loads((tmp_path / 'probe/future_noise.json').read_text()) == completed


def test_future_noise_requires_both_branches_and_original_noise_input(completed):
    execute, _, settings, membership, _ = completed
    noise = torch.randn(1, 28, 2, dtype=torch.bfloat16)
    changed = noise.clone()
    changed[:, 12:] += 1
    root = settings.output.parent
    noise_path, changed_path = root/'noise.pt', root/'changed.pt'
    torch.save(noise, noise_path)
    torch.save(changed, changed_path)
    source = membership['sources'][0]['relative_dir']
    job, paths = execute(
        ['--source', source, '--noise-file', str(noise_path),
         '--changed-noise-file', str(changed_path), '--future-noise-start', '3'],
        execution_owner=causality,
    )
    assert len(paths) == 2
    with pytest.raises(ValueError, match='inventory differs'):
        causality.verify_evaluation_conditions(
            json.loads(Path(job['spec']).read_text())['arguments'] + ['--output', job['output']], [paths[0]])
    torch.save(noise+1, noise_path)
    with pytest.raises(ValueError):
        queue.verify_completion(job)


@pytest.mark.parametrize('field', ['change_start_encoded_frame', 'records', 'earlier_output_bit_identical',
                                  'earlier_output_max_abs_delta', 'later_output_max_abs_delta'])
def test_future_noise_diagnostic_is_bound_to_both_actual_outputs(completed, field):
    execute, _, settings, membership, _ = completed
    noise = torch.randn(1, 28, 2, dtype=torch.bfloat16)
    changed = noise.clone()
    changed[:, 12:] += 1
    noise_path, changed_path = settings.output.parent/'noise.pt', settings.output.parent/'changed.pt'
    torch.save(noise, noise_path)
    torch.save(changed, changed_path)
    job, _ = execute(
        ['--source', membership['sources'][0]['relative_dir'], '--noise-file', str(noise_path),
         '--changed-noise-file', str(changed_path), '--future-noise-start', '3'],
        execution_owner=causality,
    )
    receipt = queue.completion_receipt(job)
    assert len(receipt['evidence']) == 9
    path = settings.output/'case_0000/variant_000/future_noise.json'
    data = json.loads(path.read_text())
    if field == 'records': data[field].reverse()
    elif field == 'earlier_output_bit_identical': data[field] = not data[field]
    else: data[field] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='future-noise diagnostic differs'):
        queue.verify_completion(job)



def test_signed_zero_early_byte_change_fails_before_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = _grid(_geometry())
    capture = torch.zeros(1, 28, 8)
    original = torch.zeros_like(capture)
    changed = original.clone()
    changed[:, 12:] = 1
    changed[:, 0, 0] = -0.0
    assert torch.equal(original[:, :12], changed[:, :12])
    monkeypatch.setattr(causality.evaluate, 'sample_case', lambda *_a, **_k: pytest.fail('opened model'))
    with pytest.raises(ValueError, match='changed earlier noise'):
        causality.probe_future_noise(
            None, torch.zeros(1, 3, 16), grid, capture, capture, original, changed,
            change_start_frame=3, mode='causal', mode_settings=CausalSettings(),
            guide_mode='d1', schedule=[0.725, 0], seed=42,
        )
