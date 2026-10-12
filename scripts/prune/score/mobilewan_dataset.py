"""Resident, resumable full-membership MobileWan self-attention scoring worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from scripts.prune.core import provenance, session
from scripts.prune.data import whole_clip
from scripts.prune.data.head_score_dataset import SIGMAS, atomic_json
from scripts.prune.score import hooks, mobilewan_records, token_sampling
from scripts.prune.score.mobilewan_heads import projected_energy


def source_identity():
    repo = Path(__file__).resolve().parents[3]
    # Pin worker dependencies; adding saved-report or evaluation code must not
    # invalidate calibration observations that never execute those modules.
    dependencies = (
        'score/mobilewan_dataset.py', 'score/mobilewan_records.py', 'score/mobilewan_heads.py',
        'score/hooks.py', 'score/token_sampling.py', 'data/whole_clip.py', 'data/head_score_dataset.py',
        'data/prompt_cache.py', 'core/session.py', 'core/ltx_adapter.py', 'core/model_registry.py',
        'core/preflight.py', 'core/provenance.py', 'core/artifacts.py', 'core/geometry.py',
    )
    paths = {repo / 'scripts/prune' / name for name in dependencies}
    paths.update(repo.joinpath('packages/ltx-core/src/ltx_core/model/transformer').rglob('*.py'))
    paths.update(repo.joinpath('scripts/onestep_avatar/model').glob('*.py'))
    paths.add(repo / 'scripts/onestep_avatar/corpus/dataset.py')
    files = {str(p.relative_to(repo)): provenance.file_sha256(p) for p in sorted(paths) if 'tests' not in p.parts}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return digest, files


def assigned_views(sources, split, shard, shards, maximum=None):
    if not 0 <= shard < shards or shards < 1 or (maximum is not None and maximum < 1):
        raise ValueError('invalid worker shard or view limit')
    selected = sorted((s for s in sources if s['split'] == split), key=lambda s: s['relative_dir'])
    return selected[shard::shards][:maximum]


def verify_gates(run):
    for frames in (18, 28):
        path = run / f'validation/native_check_{frames}.json'
        record = json.loads(path.read_text())
        rows = record['runs']
        if record.get('score_source_sha256') != provenance.file_sha256(Path(__file__).with_name('mobilewan_heads.py')):
            raise ValueError('native gate scoring source differs; fresh checks are required')
        if (len(rows) != 8 or [r['sigma'] for r in rows] != SIGMAS
                or any(not all(r.get(k) is True for k in ('stock_parity', 'hook_transparent',
                            'all_one_equal', 'saved_baseline_equal', 'clean_frame_equal')) for r in rows)
                or any(r['sampling']['latent_frames'] != frames for r in rows)):
            raise ValueError('native gate is incomplete or failed')


def verify_train_freeze(run, dataset, study):
    frozen = json.loads((run / 'masks/frozen_train.json').read_text())
    train = {s['relative_dir'] for s in dataset['sources'] if s['split'] == 'train'}
    if (frozen.get('schema_version') != 1 or frozen.get('kind') != 'mobilewan_train_masks'
            or frozen.get('selection_cohort') != 'train' or frozen.get('train_view_count') != len(train)
            or frozen.get('train_observation_count') != len(train)*8
            or frozen.get('membership_semantic_sha256') != dataset['membership_semantic_sha256']
            or frozen.get('transformer_sha256') != study['components']['transformer']['file_sha256']
            or set(frozen.get('train_records', {})) != train):
        raise ValueError('train-only mask freeze identity or complete coverage differs')
    for view, digest in frozen['train_records'].items():
        path = mobilewan_records.record_path(run / 'scores/per_view', view)
        if provenance.file_sha256(path) != digest:
            raise ValueError('frozen train scoring records changed')
    if [b['requested_fraction'] for b in frozen['allocation']['budgets']] != study['removal_fractions']:
        raise ValueError('frozen removal budgets differ from study')


def score_view(model, current, run, source, study, code_hash, widths, deadline):
    view = source['relative_dir']
    inputs = run / 'inputs/views' / hashlib.sha256(view.encode()).hexdigest()
    manifest = whole_clip.prepare_score_inputs(inputs, source, current=current, sigmas=SIGMAS, seed=42)
    artifact = manifest['videos'][0]['artifacts']
    identity = {'view': view, 'actor': source['actor'], 'split': source['split'],
                'capture_sha256': source['capture_latent_sha256'], 'epsilon_sha256': artifact['epsilon_sha256'],
                'context_sha256': study['text_context']['tensor_sha256'],
                'transformer_sha256': study['components']['transformer']['file_sha256'],
                'source_sha256': code_hash, 'sampler': source['sampler']}
    output = mobilewan_records.record_path(run / 'scores/per_view', view)
    if output.exists():
        mobilewan_records.read_record(output, identity, widths)
        print(f'resume complete view {view}', flush=True)
        return
    observations = []
    for j, sigma in enumerate(SIGMAS):
        if time.monotonic() >= deadline:
            raise TimeoutError('worker wall-time bound reached; saved observations remain resumable')
        path = mobilewan_records.observation_path(run / 'scores/per_sigma', view, j)
        if path.exists():
            observations.append(mobilewan_records.read_observation(path, identity, widths, j))
            continue
        grid, modality, c0, _ = whole_clip.build_input(inputs, manifest, view=source['resolved_dir'],
                                                      sigma=sigma, current=current)
        shape = grid.tools.target_shape
        indices = token_sampling.sample_indices(modality.latent.shape[1], shape.height, shape.width, 1,
                                                 current.device, sampler='stride')
        sampler = token_sampling.sampling_record(indices, tokens=modality.latent.shape[1], height=shape.height,
                                                  width=shape.width, stride=1, sampler='stride')
        if sampler != source['sampler']:
            raise ValueError('executed token sampling differs from complete audited clip')
        energies = {}

        def observe(name, activation, attention, indices=indices):
            if name.endswith('.attn1'):
                if name in energies:
                    raise ValueError('self-attention executed twice in a direct D0 call')
                energies[name] = projected_energy(activation, attention.to_out[0].weight,
                                                  attention.heads, indices).cpu().tolist()

        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(current.device)
        with hooks.collect_activations(model, 'head', observe):
            prediction, _ = model(video=modality, audio=None, perturbations=None)
        if not torch.isfinite(prediction).all() or set(energies) != set(widths):
            raise ValueError('nonfinite prediction or incomplete executed head coverage')
        if torch.cuda.max_memory_allocated(current.device) > 48_000_000_000:
            raise MemoryError('worker exceeded allocated-memory bound')
        if source_identity()[0] != code_hash:
            raise ValueError('source code changed during resident execution')
        mobilewan_records.write_observation(path, identity, widths, j, energies)
        observations.append(energies)
        print(f'scored {view} sigma_index={j} seconds={time.monotonic()-started:.3f}', flush=True)
        del grid, modality, c0, prediction, indices
    energies = {block: [observation[block] for observation in observations] for block in widths}
    mobilewan_records.write_record(output, identity, widths, energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'held_out'), required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--max-views', type=int)
    parser.add_argument('--wall-seconds', type=float, default=86400)
    args = parser.parse_args()
    started = time.monotonic()
    if args.wall_seconds <= 0:
        raise ValueError('worker wall bound must be positive')
    run = args.study.resolve().parent.parent
    study = json.loads(args.study.read_text())
    if (study['schedule']['sigmas'] != SIGMAS or study['model_variant'] != 'dev' or study['seed'] != 42
            or study['prompt'] != '' or study['guidance'] != {'cfg': 1, 'stg': 0, 'rescale': 0}):
        raise ValueError('study differs from supported exact dev D0 task')
    verify_gates(run)
    path = run / study['dataset']['manifest']
    if provenance.file_sha256(path) != study['dataset']['file_sha256']:
        raise ValueError('audited dataset manifest changed')
    dataset = json.loads(path.read_text())
    if args.split == 'held_out':
        verify_train_freeze(run, dataset, study)
    selected = assigned_views(dataset['sources'], args.split, args.shard, args.shards, args.max_views)
    if not selected:
        raise ValueError('empty worker assignment')
    for component in study['components'].values():
        if provenance.file_sha256(component['path']) != component['file_sha256']:
            raise ValueError('full component identity changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
        raise ValueError('incorrect deterministic cuBLAS workspace')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    current = session.open_session(argparse.Namespace(model='2.5', gpu_id=args.gpu_id, seed=42),
                                   script='prune.score.mobilewan_dataset', prompt='',
                                   transformer_path=Path(study['components']['transformer']['path']))
    for role, actual in (('transformer', current.model.paths.transformer()),
                         ('text_encoder', current.model.paths.text_encoder()),
                         ('video_vae', current.model.paths.video_vae())):
        if Path(actual).resolve() != Path(study['components'][role]['path']).resolve():
            raise ValueError(f'actual {role} differs from pinned study component')
    if torch.cuda.mem_get_info(current.device)[0] < 44 * 2**30:
        raise MemoryError('worker requires at least 44 GiB free before transformer loading')
    context = current.context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(context).hexdigest() != study['text_context']['tensor_sha256']:
        raise ValueError('actual text context differs from study pin')
    code_hash, files = source_identity()
    evidence = run / f'provenance/worker_{args.split}_{args.shard}_pid{os.getpid()}.json'
    atomic_json(evidence, {'schema_version': 1, 'state': 'running', 'pid': os.getpid(), 'arguments': vars(args) | {
                'study': str(args.study)}, 'source_sha256': code_hash, 'sources': files, 'assigned_views': len(selected),
                'torch_version': torch.__version__, 'memory_limit_allocated_bytes': 48_000_000_000})
    record = json.loads(evidence.read_text())
    try:
        with current.transformer() as model:
            model.eval()
            model.requires_grad_(False)
            widths = {name: attention.heads for name, attention in hooks.iter_video_attention(model)
                      if name.endswith('.attn1')}
            for source in selected:
                record['active_view'] = source['relative_dir']
                score_view(model, current, run, source, study, code_hash, widths, started+args.wall_seconds)
    except BaseException as exc:
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}', wall_seconds=time.monotonic()-started)
        atomic_json(evidence, record)
        raise
    record.update(state='complete', wall_seconds=time.monotonic()-started)
    atomic_json(evidence, record)


if __name__ == '__main__':
    main()
