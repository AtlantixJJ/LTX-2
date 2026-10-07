"""Legacy conversion changes checked metadata while preserving the original payload."""

import copy
import json
import struct

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.onestep_avatar import dataset, subset, windows
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone
from scripts.onestep_avatar.training import checkpoints
from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B, _contract
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- pytest fixture


def payload(path):
    with path.open('rb') as handle:
        header_length = struct.unpack('<Q', handle.read(8))[0]
        handle.read(header_length)
        return handle.read()


def inputs(tmp_path, old_subset, mode='bidirectional', guide_mode='d0', random_window=False):
    old = copy.deepcopy(old_subset)
    old['chains'] = [chain for chain in old['chains'] if chain['seed_is_clip_start']]
    blocks = [0] if mode == 'bidirectional' or random_window else [0, 1, 2]
    for chain in old['chains']:
        chain['blocks'] = blocks
    old['content_pinned'] = True
    old['chain_length'] = len(blocks)
    old['geometry']['block_latent_frames'] = 6 if mode == 'bidirectional' else 2
    old['geometry']['context_latent_frames'] = 8
    for source in old['sources']:
        view = tmp_path / source['relative_dir']
        capture = view / dataset.capture_bundle_name('white')
        source['capture_latent_sha256'] = sha256(capture)
        if guide_mode == 'd1':
            render = view / dataset.render_name('white')
            render.write_bytes(b'original guide rendering')
            source['guide_sha256'] = sha256(render)
            guide = torch.load(capture, weights_only=True)
            guide['input_fingerprint'] = source['guide_sha256']
            torch.save(guide, view / dataset.guide_bundle_name('white'))
            source['guide_latent_sha256'] = sha256(view / dataset.guide_bundle_name('white'))
            sidecar = view / dataset.render_metadata_name('white')
            sidecar.write_text(json.dumps({'compositing_version': dataset.GUIDE_COMPOSITING_VERSION}))
    subset_path = tmp_path / 'original_subset.json'
    subset_path.write_text(json.dumps(old))
    membership, plan = subset.convert_legacy(old, original_file_sha256=sha256(subset_path))
    membership_path, plan_path = tmp_path / 'membership.json', tmp_path / 'frame_plan.json'
    membership_path.write_text(json.dumps(membership))
    plan_path.write_text(json.dumps(plan))
    base = tmp_path / 'base.safetensors'
    save_file({'base': torch.ones(2)}, base)
    identity = backbone.identity(base, 'dev', '2.5')
    record = _contract(mode)
    record['task']['guide_mode'] = guide_mode
    record['model'] = {'version': '2.5', 'variant': 'dev', 'base_file': base.name, 'base_sha256': sha256(base)}
    record['shape']['channels'] = 2
    if random_window:
        record['mode_settings'].update(start_policy='random', span_latent_frames=old['geometry']['block_latent_frames']+1)
        record['shape']['frame_counts'] = [record['mode_settings']['span_latent_frames']]
        if mode == 'causal':
            record['causal']['blocks_per_sample'] = 1
            record['mode_settings']['blocks_per_sample'] = 1
    record['data'] = {'membership_sha256': membership['sha256'], 'frame_plan_sha256': plan['sha256'],
                      'coverage': [{'source': sample['source'], 'ranges': sample['ranges']}
                                   for sample in plan['samples'] if sample['split'] == 'train']}
    config = {'split': 'train', 'subset_full_sha256': windows.subset_sha256(old), 'skip_subset_check': False,
              'init_adapter': None, 'model': '2.5', 'variant': 'dev', 'base_identity': identity,
              'lora_rank': 2, 'lora_alpha': 2, 'lora_target': 'attn', 'teacher_forcing': False,
              'sigma_levels': record['training']['sigma_levels'], 'anchor_weight': 0.0}
    config.update({key: old['geometry'][key] for key in ('block_latent_frames', 'context_latent_frames')})
    metadata = {'step': '0', 'lora_rank': '2', 'lora_alpha': '2', 'lora_target': 'attn',
                'model_key': '2.5', 'onestep_avatar_base_variant': 'dev', 'onestep_avatar_teacher_forcing': 'False',
                'onestep_avatar_subset_full_sha256': windows.subset_sha256(old),
                'onestep_avatar_chain_length': str(len(blocks)), 'onestep_avatar_anchor_weight': '0.0',
                'onestep_avatar_sigma_levels': ','.join(repr(v) for v in config['sigma_levels'])}
    for key in ('guide_mode', 'objective', 'loss', 'first_frame_conditioning'):
        metadata[f'onestep_avatar_{key}'] = record['task'][key]
        config[key] = record['task'][key]
    config['noise_policy'] = record['training']['noise_policy']
    metadata['onestep_avatar_noise_policy'] = config['noise_policy']
    config['sigma_sampling'] = record['training']['sigma_sampling']
    metadata['onestep_avatar_sigma_sampling'] = config['sigma_sampling']
    for key, value in {'schedule': 'ONE_STEP', 'attention': 'block_causal',
                       'history_computation': 'cached_refresh_global_sigma0',
                       'parent_adapter': '', 'window': 'clip_start'}.items():
        metadata[f'onestep_avatar_{key}'] = value
    for key, value in record['training']['seeds'].items():
        config[f'{key}_seed'] = value
        metadata[f'onestep_avatar_{key}_seed'] = str(value)
    if random_window:
        from scripts.onestep_avatar.training.config import window_start_draw
        window = record['mode_settings']['span_latent_frames']
        config['random_window_latent_frames'] = window
        metadata['onestep_avatar_window'] = f'random_start_v1:{window}'
        plan['start_draw'] = window_start_draw(config['noise_seed'])
        plan['sha256'] = subset.record_hash(plan)
        plan_path.write_text(json.dumps(plan))
        record['data']['frame_plan_sha256'] = plan['sha256']
        record['data']['segment_selection'] = checkpoints.random_segment_selection(
            record['data']['coverage'], {s['relative_dir']: s['n_latent_frames'] for s in membership['sources']},
            window, config['noise_seed'])
    for key in ('base_transformer_file', 'base_transformer_fingerprint'):
        metadata[f'onestep_avatar_{key}'] = identity[key]
    for key in ('block_latent_frames', 'context_latent_frames', 'sink_latent_frames'):
        metadata[f'onestep_avatar_{key}'] = str(old['geometry'][key])
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    source = tmp_path / 'original.safetensors'
    save_file({A: torch.arange(8).reshape(2, 4).bfloat16(), B: torch.arange(8).reshape(4, 2).bfloat16()},
              source, metadata=metadata)
    evidence = dict(config_path=config_path, subset_path=subset_path, membership_path=membership_path,
                    frame_plan_path=plan_path, base_path=base)
    return source, record, evidence


@pytest.mark.parametrize('mode', ['bidirectional', 'causal'])
@pytest.mark.parametrize('guide_mode', ['d0', 'd1'])
def test_checked_conversion_preserves_every_tensor_byte_and_original(tmp_path, old_subset, mode, guide_mode):
    source, record, evidence = inputs(tmp_path, old_subset, mode, guide_mode)
    original_bytes = source.read_bytes()
    output = tmp_path / 'derived.safetensors'
    converted = checkpoints.convert_legacy_adapter(source, output, record, **evidence)
    assert source.read_bytes() == original_bytes
    assert payload(output) == payload(source)
    assert all(torch.equal(value, load_file(output)[key]) for key, value in load_file(source).items())
    assert checkpoints.read_contract(output) == converted
    assert converted['conversion']['source_sha256'] == sha256(source)
    assert converted['conversion']['original_metadata'] == checkpoints.read_adapter_metadata(source)
    assert checkpoints.CONTRACT_KEY not in checkpoints.read_adapter_metadata(source)
    identities = converted['conversion']['original_subset_identity']
    assert identities['file_sha256'] == sha256(evidence['subset_path'])
    assert identities['canonical_sha256'] == windows.subset_sha256(json.loads(evidence['subset_path'].read_text()))
    assert identities['canonical_sha256'] != identities['file_sha256']


@pytest.mark.parametrize('defect', ['sigma', 'subset', 'random', 'parent', 'shape', 'mapping', 'history', 'forcing'])
def test_unproven_or_different_conditions_refuse_conversion(tmp_path, old_subset, defect):
    source, record, evidence = inputs(tmp_path, old_subset)
    if defect == 'sigma':
        record['training']['sigma_levels'] = [0.5]
        record['training']['schedules'] = [[0.5, 0.0]]
    elif defect in ('subset', 'random', 'parent', 'forcing'):
        cfg = json.loads(evidence['config_path'].read_text())
        cfg[{'subset': 'subset_full_sha256', 'random': 'random_window_latent_frames', 'parent': 'init_adapter',
             'forcing': 'teacher_forcing'}[defect]] = {
            'subset': '0' * 64, 'random': 7, 'parent': 'parent.safetensors', 'forcing': True}[defect]
        evidence['config_path'].write_text(json.dumps(cfg))
    elif defect == 'shape':
        record['shape']['channels'] = 128
    else:
        plan = json.loads(evidence['frame_plan_path'].read_text())
        plan['samples'][0]['original_chain_index' if defect == 'mapping' else 'blocks'] = 999 if defect == 'mapping' else [1]
        plan['sha256'] = subset.record_hash(plan)
        evidence['frame_plan_path'].write_text(json.dumps(plan))
        record['data']['frame_plan_sha256'] = plan['sha256']
    original = source.read_bytes()
    with pytest.raises(ValueError):
        checkpoints.convert_legacy_adapter(source, tmp_path / 'derived.safetensors', record, **evidence)
    assert source.read_bytes() == original and not (tmp_path / 'derived.safetensors').exists()


def test_existing_derived_result_is_preserved(tmp_path, old_subset):
    source, record, evidence = inputs(tmp_path, old_subset)
    output = tmp_path / 'derived.safetensors'
    output.write_bytes(b'original derived evidence')
    with pytest.raises(ValueError, match='fresh derived'):
        checkpoints.convert_legacy_adapter(source, output, record, **evidence)
    assert output.read_bytes() == b'original derived evidence'


@pytest.mark.parametrize('mode', ['bidirectional', 'causal'])
@pytest.mark.parametrize('guide_mode', ['d0', 'd1'])
def test_random_conversion_preserves_window_rule_and_g9(tmp_path, old_subset, mode, guide_mode):
    # More master frames than the window prove that [0,W] is only a template.
    for source in old_subset['sources']:
        path = tmp_path / source['relative_dir'] / dataset.capture_bundle_name('white')
        bundle = torch.load(path, weights_only=True)
        bundle['master'] = torch.arange(104).reshape(2,13,2,2).float()
        torch.save(bundle,path)
        source['n_latent_frames'] = 13
    source, record, evidence = inputs(tmp_path, old_subset, mode, guide_mode, random_window=True)
    before = source.read_bytes()
    output = tmp_path / 'derived.safetensors'
    converted = checkpoints.convert_legacy_adapter(source,output,record,**evidence)
    assert source.read_bytes() == before and payload(source) == payload(output)
    selection = converted['data']['segment_selection']
    assert selection['coverage_role'] == 'window_templates'
    assert selection['start_bounds_inclusive'] == {
        sample['source']: [0,13-record['mode_settings']['span_latent_frames']]
        for sample in record['data']['coverage']}
    assert selection['known_gap'] == 'G9' and selection['first_image'] == 'selected_capture_master_frame'
    assert selection['positions'] == 'restart_at_zero'
    assert converted['mode_settings']['start_policy'] == 'random'


@pytest.mark.parametrize('defect', ['missing_draw', 'wrong_seed', 'wrong_key', 'length', 'clip_start', 'master_frames'])
def test_random_conversion_refuses_different_original_draws(tmp_path, old_subset, defect):
    source, record, evidence = inputs(tmp_path, old_subset, random_window=True)
    plan = json.loads(evidence['frame_plan_path'].read_text())
    if defect == 'missing_draw':
        plan.pop('start_draw')
    elif defect in ('wrong_seed','wrong_key'):
        plan['start_draw']['seed' if defect == 'wrong_seed' else 'key'] = 99 if defect == 'wrong_seed' else 'changed'
    elif defect == 'length':
        cfg = json.loads(evidence['config_path'].read_text())
        cfg['random_window_latent_frames'] = 6
        evidence['config_path'].write_text(json.dumps(cfg))
    elif defect == 'clip_start':
        record['mode_settings']['start_policy'] = 'clip_start'
    else:
        selection = record['data']['segment_selection']
        key = next(iter(selection['master_frames']))
        selection['master_frames'][key] += 1
        selection['start_bounds_inclusive'][key][1] += 1
    plan['sha256'] = subset.record_hash(plan)
    evidence['frame_plan_path'].write_text(json.dumps(plan))
    record['data']['frame_plan_sha256'] = plan['sha256']
    with pytest.raises(ValueError):
        checkpoints.convert_legacy_adapter(source,tmp_path/'derived.safetensors',record,**evidence)
    assert not (tmp_path/'derived.safetensors').exists()


@pytest.mark.parametrize('field', ['sigma_sampling', 'schedule', 'attention', 'history_computation',
                                  'parent_adapter', 'window'])
@pytest.mark.parametrize('value', [None, 'different'])
def test_missing_or_changed_execution_stamps_fail_before_base_hash(tmp_path, old_subset, monkeypatch, field, value):
    source, record, evidence = inputs(tmp_path, old_subset)
    metadata = checkpoints.read_adapter_metadata(source)
    if value is None:
        metadata.pop(f'onestep_avatar_{field}')
    else:
        metadata[f'onestep_avatar_{field}'] = value
    save_file(load_file(source), source, metadata=metadata)
    original = source.read_bytes()
    from scripts.onestep_avatar import hashing
    original_sha = hashing.sha256

    def checked_hash(path):
        assert path != evidence['base_path'], 'ambiguous record must fail before base hashing'
        return original_sha(path)

    monkeypatch.setattr(hashing, 'sha256', checked_hash)
    destination = tmp_path / 'fresh' / 'derived.safetensors'
    with pytest.raises(ValueError, match=field.replace('_', ' ' if field == 'sigma_sampling' else '_')):
        checkpoints.convert_legacy_adapter(source, destination, record, **evidence)
    assert source.read_bytes() == original and not destination.parent.exists()


def test_original_config_must_stamp_same_sigma_draw_rule(tmp_path, old_subset):
    source, record, evidence = inputs(tmp_path, old_subset)
    config = json.loads(evidence['config_path'].read_text())
    config.pop('sigma_sampling')
    evidence['config_path'].write_text(json.dumps(config))
    with pytest.raises(ValueError, match='sigma sampling'):
        checkpoints.convert_legacy_adapter(source, tmp_path / 'derived.safetensors', record, **evidence)


def test_changed_clip_start_evidence_refuses_classification(tmp_path, old_subset):
    source, record, evidence = inputs(tmp_path, old_subset)
    plan = json.loads(evidence['frame_plan_path'].read_text())
    plan['samples'][0]['seed_is_clip_start'] = False
    plan['sha256'] = subset.record_hash(plan)
    evidence['frame_plan_path'].write_text(json.dumps(plan))
    record['data']['frame_plan_sha256'] = plan['sha256']
    with pytest.raises(ValueError, match='chain mapping differs'):
        checkpoints.convert_legacy_adapter(source, tmp_path / 'derived.safetensors', record, **evidence)


def test_public_conversion_cli(tmp_path, old_subset):
    source, record, evidence = inputs(tmp_path, old_subset)
    contract = tmp_path / 'reviewed_contract.json'
    contract.write_text(json.dumps(record))
    output = tmp_path / 'derived.safetensors'
    args = ['--source', str(source), '--output', str(output), '--contract', str(contract)]
    for option, key in [('original-config', 'config_path'), ('original-subset', 'subset_path'),
                        ('membership', 'membership_path'), ('frame-plan', 'frame_plan_path'), ('base', 'base_path')]:
        args += [f'--{option}', str(evidence[key])]
    assert checkpoints.main(args) == 0
    assert payload(source) == payload(output)


def test_exclusive_publication_preserves_competing_result(tmp_path, old_subset, monkeypatch):
    import os

    source, record, evidence = inputs(tmp_path, old_subset)
    output = tmp_path / 'derived.safetensors'
    original_link = os.link

    def publish(temporary, destination):
        destination.write_bytes(b'competing result')
        return original_link(temporary, destination)

    monkeypatch.setattr(os, 'link', publish)
    with pytest.raises(FileExistsError):
        checkpoints.convert_legacy_adapter(source, output, record, **evidence)
    assert output.read_bytes() == b'competing result'
    assert sorted(path.name for path in tmp_path.glob('*.safetensors')) == [
        'base.safetensors', 'derived.safetensors', 'original.safetensors']
