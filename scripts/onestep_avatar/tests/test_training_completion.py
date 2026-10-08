"""Training completion binds checked small data and actual adapter artifacts to exact jobs."""
import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.onestep_avatar import queue
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.test_applied_runtime import inventory
from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- fixture dependency
from scripts.onestep_avatar.tests.test_training_preflight import checked_settings  # noqa: F401
from scripts.onestep_avatar.training import checkpoints, config, engine, resources


@pytest.fixture
def completed(checked_settings, monkeypatch):
    initial, _ = checked_settings
    specification = engine.backbone.resolve('2.5', 'dev')
    base = specification.paths.transformer()
    base.write_bytes(b'controlled base bytes')
    monkeypatch.setattr(engine.backbone, 'identity', lambda path, *_a, **_k: {
        'base_transformer_file': Path(path).name, 'base_transformer_sha256': sha256(Path(path))})
    fsdp = initial.output.parent/'fsdp.yaml'
    fsdp.write_text('distributed_type: FSDP\nnum_processes: 4\nmixed_precision: bf16\n')

    def produce(*, mode='bidirectional', extra=()):
        arguments = ['--mode', mode, '--subset', str(initial.subset), '--output', str(initial.output),
                     '--variant', 'dev', '--objective', 'white', '--guide-mode', 'd0', '--steps', '2',
                     '--lora-rank', '2', '--lora-alpha', '2', '--no-wandb', *extra]
        if mode == 'bidirectional': arguments.extend(['--span-latent-frames', '7'])
        settings = config.parse_settings(arguments)
        store, plan, _, _ = engine.prepare_run(settings)
        checkpoint = settings.output/'checkpoints/lora_weights_step_00002.safetensors'
        raw_job = {'id': 'training', 'kind': 'train', 'arguments': arguments, 'output': str(settings.output),
                   'processes': 4, 'port': 29600, 'accelerate_config': str(fsdp),
                   'completion': {'checkpoint': str(checkpoint), 'step': 2}}
        jobs_path = initial.output.parent/'jobs.json'
        jobs_path.write_text(json.dumps({'schema_version': 1, 'jobs': [raw_job]}))
        job = queue.prepare_jobs(jobs_path)[0]
        launch = queue.training_launch_record(job)
        applied_runtime = inventory(numerical=True)
        settings.world_size = 4
        budget = resources.read_budget(settings.resource_budget)
        contract = checkpoints.make_contract(settings, store.membership, plan, settings.steps)
        contract['adapter']['tensor_shapes'] = {A: [2, 4], B: [4, 2]}
        checkpoint.parent.mkdir(parents=True)
        save_file({A: torch.ones(2, 4, dtype=torch.bfloat16), B: torch.zeros(4, 2, dtype=torch.bfloat16)},
                  checkpoint, metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        samples = sum(s['split'] == settings.split for s in plan['samples'])
        resolved = {**settings.as_dict(), 'membership_sha256': store.membership['sha256'],
                    'software': engine.software.capture('training', settings.mode),
                    'frame_plan_sha256': plan['sha256'], 'samples': samples, 'loss': engine.FULL_FRAME_X0_MSE,
                    'queue_job_sha256': job['sha256'], 'producer_source_sha256': sha256(Path(engine.__file__)),
                    'queue_launch': launch, 'runtime': applied_runtime, 'resource_budget': budget,
                    'samples_per_update': 4*settings.chains_per_rank,
                    'sample_tiling': max(1, math.ceil(4*settings.chains_per_rank/samples)),
                    'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0},
                    'trainable_params': 16, 'lora_modules': 1,
                    'lora_target_counts': {target: int(target == 'to_q') for target in config.LORA_TARGETS['attn']}}
        (settings.output/'config.json').write_text(json.dumps(resolved, default=str))
        (settings.output/'frame_plan.json').write_text(json.dumps(plan))
        resource_evidence = {}
        if budget is not None:
            # Controlled metadata verifies completion binding; no CUDA work runs.
            saves = ([0] if settings.save_initial else []) + [
                step for step in range(1, settings.steps + 1)
                if step % settings.save_every == 0 or step == settings.steps or (settings.save_initial and step == 1)]
            phases = ['load', *[f'update:{step}' for step in range(1, settings.steps + 1)],
                      *[f'export:{step}' for step in saves]]
            for rank in range(settings.world_size):
                records = [
                    {'schema_version': 1, 'rank': rank, 'phase': phase, 'device': f'cuda:{rank}',
                     'elapsed_s': 10, 'peak_allocated_bytes': 1000, 'peak_reserved_bytes': 2000,
                     'budget_sha256': budget['sha256'], 'state': 'passed', 'error': None}
                    for phase in phases]
                (settings.output/f'resources_rank{rank}.jsonl').write_text(
                    '\n'.join(json.dumps(record) for record in records) + '\n')
                resources.save_snapshot(settings.output, rank, settings.steps)
            resource_evidence = resources.read_records(settings.output, settings.world_size, step=settings.steps)[1]
        marker = {'schema_version': 2, 'step': 2, 'path': str(checkpoint), 'sha256': sha256(checkpoint),
                  'state': 'complete', 'queue_job_sha256': job['sha256'],
                  'producer_source_sha256': resolved['producer_source_sha256'],
                  'software': resolved['software'],
                  'queue_launch': launch, 'runtime': applied_runtime,
                  'resource_budget': budget, 'resource_evidence': resource_evidence,
                  'training_record': {'config_sha256': sha256(settings.output/'config.json'),
                                      'frame_plan_sha256': sha256(settings.output/'frame_plan.json')}}
        checkpoint.with_suffix('.complete.json').write_text(json.dumps(marker))
        assert queue.verify_completion(job)
        return job, settings, checkpoint

    return produce, initial, fsdp


@pytest.mark.parametrize('mode', ['bidirectional', 'causal'])
def test_both_training_modes_verify_without_models_and_keep_fresh_output_gate(completed, mode):
    produce, _, _ = completed
    job, settings, _ = produce(mode=mode)
    with pytest.raises(ValueError, match='output already has a run'):
        engine.prepare_run(settings)
    assert queue.verify_completion(job)
    resolved = json.loads((settings.output/'config.json').read_text())
    assert resolved['resource_budget'] is None
    assert 'resource_budget_sha256' not in job
    assert not list(settings.output.glob('resources_rank*.jsonl'))
    receipt = queue.completion_receipt(job)
    assert len(receipt['evidence']) == 4
    assert queue.verify_receipt(job, receipt)


@pytest.mark.parametrize('mode', ['bidirectional', 'causal'])
def test_budget_backed_completion_binds_record_and_complete_rank_phase_evidence(completed, mode):
    produce, initial, _ = completed
    budget_path = initial.output.parent/'resource_budget.json'
    budget_path.write_text(json.dumps({'wall_seconds_per_phase': 1800, 'memory_limit_allocated_bytes': 48000000000}))
    job, settings, checkpoint = produce(mode=mode, extra=['--resource-budget', str(budget_path)])
    resolved = json.loads((settings.output/'config.json').read_text())
    marker = json.loads(checkpoint.with_suffix('.complete.json').read_text())
    assert resolved['resource_budget'] == resources.read_budget(budget_path)
    assert marker['resource_budget'] == resolved['resource_budget']
    assert job['resource_budget_sha256'] == sha256(budget_path)
    records, evidence = resources.read_records(settings.output, 4, step=2)
    assert len(records) == 16
    assert {record['phase'] for record in records} == {'load', 'update:1', 'update:2', 'export:2'}
    assert marker['resource_evidence'] == evidence
    assert queue.verify_receipt(job, queue.completion_receipt(job))


@pytest.mark.parametrize('option,value', [
    ('--lr', '0.005'), ('--warmup-steps', '3'), ('--max-grad-norm', '2'), ('--chains-per-rank', '2'),
    ('--seed', '43'), ('--noise-seed', '43'), ('--sigma0', '0.6'), ('--noise-policy', 'fixed_per_chain'),
    ('--lora-target', 'attn_ffn'), ('--lora-rank', '4'), ('--lora-alpha', '4'), ('--split', 'held_out'),
])
def test_checkpoint_cannot_complete_different_scientific_request(completed, option, value):
    produce, _, _ = completed
    job, _, _ = produce()
    if option in job['arguments']:
        job['arguments'][job['arguments'].index(option)+1] = value
    else:
        job['arguments'].extend([option, value])
    if option == '--lora-rank':
        job['arguments'][job['arguments'].index('--lora-alpha')+1] = value
    with pytest.raises((ValueError, SystemExit)):
        queue.verify_completion(job)


@pytest.mark.parametrize('field,value', [
    ('lr', 0.005), ('world_size', 1), ('samples_per_update', 1), ('sample_tiling', 1),
    ('samples', 10), ('loss', 'other loss'), ('membership_sha256', 'b'*64), ('frame_plan_sha256', 'b'*64),
    ('optimizer', {}), ('queue_job_sha256', 'b'*64), ('producer_source_sha256', 'b'*64),
    ('trainable_params', 0), ('lora_modules', 0), ('lora_target_counts', {}),
])
def test_saved_config_mutation_fails_even_after_marker_hash_is_updated(completed, field, value):
    produce, _, _ = completed
    job, settings, checkpoint = produce()
    path = settings.output/'config.json'
    resolved = json.loads(path.read_text())
    resolved[field] = value
    path.write_text(json.dumps(resolved))
    marker_path = checkpoint.with_suffix('.complete.json')
    marker = json.loads(marker_path.read_text())
    marker['training_record']['config_sha256'] = sha256(path)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='saved configuration or frame plan'):
        queue.verify_completion(job)


@pytest.mark.parametrize('field', ['queue_job_sha256', 'producer_source_sha256', 'training_record'])
def test_marker_requires_matching_run_identity(completed, field):
    produce, _, _ = completed
    job, _, checkpoint = produce()
    path = checkpoint.with_suffix('.complete.json')
    marker = json.loads(path.read_text())
    del marker[field]
    path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='different run provenance'):
        queue.verify_completion(job)


@pytest.mark.parametrize('asset', ['config', 'plan', 'base', 'master', 'checkpoint_path'])
def test_changed_or_relocated_training_evidence_is_refused(completed, asset):
    produce, _, _ = completed
    job, settings, checkpoint = produce()
    if asset in ('config', 'plan'):
        path = settings.output/('config.json' if asset == 'config' else 'frame_plan.json')
        # JSON whitespace preserves values but changes the marker's bound bytes.
        path.write_text(path.read_text()+'\n')
    elif asset == 'base':
        engine.backbone.resolve('2.5', 'dev').paths.transformer().write_bytes(b'changed weights')
    elif asset == 'master':
        membership = json.loads(settings.subset.read_text())
        view = Path(membership['corpus_root'])/membership['sources'][0]['relative_dir']
        (view/'ltx_vae_latent_white.pt').write_bytes(b'changed input')
    else:
        other = checkpoint.with_name('unrequested.safetensors')
        other.write_bytes(checkpoint.read_bytes())
        marker = json.loads(checkpoint.with_suffix('.complete.json').read_text())
        marker['path'] = str(other)
        other.with_suffix('.complete.json').write_text(json.dumps(marker))
        job['completion']['checkpoint'] = str(other)
    with pytest.raises(ValueError):
        queue.verify_completion(job)


@pytest.mark.parametrize('asset', ['config', 'plan'])
def test_receipt_preserves_initial_config_and_plan_bytes_even_after_valid_marker_update(completed, asset):
    produce, _, _ = completed
    job, settings, checkpoint = produce()
    receipt = queue.completion_receipt(job)
    path = settings.output/('config.json' if asset == 'config' else 'frame_plan.json')
    path.write_text(path.read_text()+'\n')
    marker_path = checkpoint.with_suffix('.complete.json')
    marker = json.loads(marker_path.read_text())
    marker['training_record']['config_sha256' if asset == 'config' else 'frame_plan_sha256'] = sha256(path)
    marker_path.write_text(json.dumps(marker))
    assert queue.verify_completion(job)
    with pytest.raises(ValueError, match='receipt evidence changed'):
        queue.verify_receipt(job, receipt)


@pytest.mark.parametrize('changed', [None, 'bytes', 'cross_mode_permission'])
def test_parent_bytes_and_cross_mode_fresh_initialization_lineage(completed, changed):
    from scripts.onestep_avatar.tests.test_checkpoint_contract import _contract

    produce, initial, _ = completed
    parent = initial.output.parent/'parent.safetensors'
    contract = _contract()
    contract['model']['base_sha256'] = sha256(engine.backbone.resolve('2.5', 'dev').paths.transformer())
    metadata = {checkpoints.CONTRACT_KEY: json.dumps(contract)}
    save_file({A: torch.ones(2, 4, dtype=torch.bfloat16), B: torch.zeros(4, 2, dtype=torch.bfloat16)},
              parent, metadata=metadata)
    job, _, checkpoint = produce(mode='causal', extra=['--init-adapter', str(parent), '--allow-cross-mode-init'])
    lineage = checkpoints.read_contract(checkpoint)['adapter']['parent']
    assert lineage['sha256'] == sha256(parent)
    assert lineage['original_mode'] == 'bidirectional'
    assert lineage['initialization'] == 'fresh_optimizer_and_random_state'
    assert lineage['calibration_transferred'] is False
    if changed == 'bytes':
        save_file({A: torch.ones(2, 4, dtype=torch.bfloat16), B: torch.ones(4, 2, dtype=torch.bfloat16)},
                  parent, metadata=metadata)
    elif changed == 'cross_mode_permission':
        job['arguments'].remove('--allow-cross-mode-init')
    if changed:
        with pytest.raises(ValueError):
            queue.verify_completion(job)


def test_training_model_owner_change_invalidates_current_completion(completed, monkeypatch):
    produce, _, _ = completed
    job, settings, checkpoint = produce()
    manifest = json.loads((settings.output/'config.json').read_text())['software']
    original = engine.software.sha256
    monkeypatch.setattr(engine.software, 'sha256', lambda p: 'f'*64 if p.name == 'common.py' else original(p))
    engine.software.validate(manifest)
    with pytest.raises(ValueError, match='software.*changed since preflight'):
        queue.verify_completion(job)
