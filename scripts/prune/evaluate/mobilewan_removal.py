"""Matched held-out D0 functional removal, saved full latents and GT frame MSE."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from scripts.prune.core import provenance, session
from scripts.prune.data import whole_clip
from scripts.prune.data.head_score_dataset import SIGMAS, atomic_json
from scripts.prune.score import hooks
from scripts.prune.score.mobilewan_dataset import source_identity, verify_gates, verify_train_freeze


def metrics(prediction, capture, fps, temporal_scale):
    """Exclude conditioned frame zero; time uses the latent coordinate f*scale/fps."""
    if (prediction.shape != capture.shape or prediction.ndim != 5 or prediction.shape[2] < 2
            or not torch.isfinite(prediction).all() or not torch.isfinite(capture).all()
            or fps <= 0 or temporal_scale < 1):
        raise ValueError('invalid paired full-clip latents or time scale')
    error = (prediction[:, :, 1:].double() - capture[:, :, 1:].double()).square()
    per_frame = error.mean(dim=(0, 1, 3, 4))
    return {'mse_to_gt': error.mean().item(), 'per_frame_mse': per_frame.tolist(),
            'latent_frame_indices': list(range(1, capture.shape[2])),
            'times_seconds': [temporal_scale*f/fps for f in range(1, capture.shape[2])],
            'clean_frame_equal': torch.equal(prediction[:, :, 0], capture[:, :, 0])}


def save_tensor(path, tensor):
    """Atomic no-overwrite tensor publication; existing bytes must represent the same tensor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().cpu().contiguous()
    if path.exists():
        if not torch.equal(torch.load(path, map_location='cpu', weights_only=True), tensor):
            raise ValueError('saved evaluation tensor differs')
        return
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.tensor-')
    try:
        with os.fdopen(fd, 'wb') as output:
            torch.save(tensor, output)
            output.flush()
            os.fsync(output.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def mask_families(model, frozen):
    """Keep all cross-attention heads and exact original self-attention identities."""
    widths = {name: attention.heads for name, attention in hooks.iter_video_attention(model)}
    families = []
    for budget in frozen['allocation']['budgets']:
        raw = {f'{block}.attn1': values for block, values in budget['masks'].items()}
        raw.update({name: [1]*count for name, count in widths.items() if name.endswith('.attn2')})
        if set(raw) != set(widths):
            raise ValueError('frozen mask architecture differs')
        for name, values in raw.items():
            if len(values) != widths[name] or not any(values) or any(v not in (0, 1) for v in values):
                raise ValueError('invalid binary frozen mask family')
        eligible = sum(count for name, count in widths.items() if name.endswith('.attn1'))
        removed = sum(values.count(0) for name, values in raw.items() if name.endswith('.attn1'))
        if removed != budget['removed_count'] or removed / eligible != budget['realized_fraction']:
            raise ValueError('frozen removal counts differ from masks')
        families.append((budget, {name: torch.tensor(values) for name, values in raw.items()}))
    if [b['requested_fraction'] for b, _ in families] != [0., .1, .23, .33]:
        raise ValueError('held-out study requires all four frozen removal arms')
    if any(not torch.all(v == 1) for v in families[0][1].values()):
        raise ValueError('zero removal mask is not all-one')
    return families


@torch.no_grad()
def evaluate_case(model, current, source, study, run, frozen, identity):
    view = source['relative_dir']
    key = hashlib.sha256(view.encode()).hexdigest()
    inputs = run / 'inputs/views' / key
    manifest = whole_clip.prepare_score_inputs(inputs, source, current=current, sigmas=SIGMAS, seed=42)
    capture, fps = whole_clip.load_training_master(Path(source['capture']))
    capture = capture.unsqueeze(0).cpu()
    gt_path = run / 'evaluation/latents' / key / 'gt.pt'
    save_tensor(gt_path, capture)
    families = mask_families(model, frozen)
    common = {**identity, 'view': view, 'actor': source['actor'],
              'input_manifest_sha256': provenance.file_sha256(inputs / 'manifest.json'),
              'epsilon_sha256': manifest['videos'][0]['artifacts']['epsilon_sha256'],
              'gt': str(gt_path.relative_to(run)), 'gt_sha256': provenance.file_sha256(gt_path)}
    records = []
    for j, sigma in enumerate(SIGMAS):
        record_path = run / 'evaluation/records' / key / f'sigma_{j}.json'
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            if saved['identity'] != common or saved['sigma'] != sigma or saved['all_one_equal'] is not True:
                raise ValueError('saved removal evaluation identity or control differs')
            if (saved.get('schema_version') != 1 or saved.get('schedule') != [sigma, 0.]
                    or len(saved['arms']) != 4):
                raise ValueError('saved removal arm or schedule coverage differs')
            for row, (budget, _) in zip(saved['arms'], families, strict=True):
                if (any(row[k] != budget[k] for k in ('requested_fraction', 'realized_fraction', 'removed_count'))
                        or row['mse_delta_from_dense'] != row['mse_to_gt'] - saved['arms'][0]['mse_to_gt']):
                    raise ValueError('saved removal budget or paired delta differs')
                path = run / row['prediction']
                if provenance.file_sha256(path) != row['prediction_sha256']:
                    raise ValueError('saved prediction changed')
                value = torch.load(path, map_location='cpu', weights_only=True)
                measured = metrics(value, capture, fps, current.model.scale_factors.time)
                if any(row[k] != v for k, v in measured.items()):
                    raise ValueError('saved GT metric differs from paired tensors')
            records.append(saved)
            continue
        grid, modality, c0, _ = whole_clip.build_input(inputs, manifest, view=source['resolved_dir'],
                                                      sigma=sigma, current=current)
        dense, _ = model(video=modality, audio=None, perturbations=None)
        with hooks.attach_head_masks(model, families[0][1], requires_grad=False):
            all_one, _ = model(video=modality, audio=None, perturbations=None)
        if not torch.equal(dense, all_one):
            raise ValueError('all-one masked result differs from dense')
        rows = []
        for budget, masks in families:
            if budget['requested_fraction'] == 0:
                prediction = dense
            else:
                with hooks.attach_head_masks(model, masks, requires_grad=False):
                    prediction, _ = model(video=modality, audio=None, perturbations=None)
            restored = torch.cat([c0, prediction[:, c0.shape[1]:]], dim=1)
            latent = grid.unpatchify_block(restored, grid.latent_frames).cpu()
            measured = metrics(latent, capture, fps, current.model.scale_factors.time)
            if not measured['clean_frame_equal']:
                raise ValueError('restored clean frame differs from capture')
            path = gt_path.parent / f"sigma_{j}_removal_{round(budget['requested_fraction']*100):03d}.pt"
            save_tensor(path, latent)
            rows.append({**measured, 'requested_fraction': budget['requested_fraction'],
                         'realized_fraction': budget['realized_fraction'], 'removed_count': budget['removed_count'],
                         'prediction': str(path.relative_to(run)), 'prediction_sha256': provenance.file_sha256(path)})
        for row in rows:
            row['mse_delta_from_dense'] = row['mse_to_gt'] - rows[0]['mse_to_gt']
        saved = {'schema_version': 1, 'identity': common, 'sigma': sigma, 'schedule': [sigma, 0.],
                 'all_one_equal': True, 'arms': rows}
        if (identity.get('shared_source_sha256') is not None
                and source_identity()[0] != identity['shared_source_sha256']):
            raise ValueError('shared source changed during removal execution')
        if (identity.get('evaluation_source_sha256') is not None
                and provenance.file_sha256(__file__) != identity['evaluation_source_sha256']):
            raise ValueError('evaluation source changed during execution')
        atomic_json(record_path, saved)
        records.append(saved)
        print(f'evaluated held-out {view} sigma_index={j}', flush=True)
    return {'view': view, 'actor': source['actor'], 'sigmas': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=4)
    args = parser.parse_args()
    run = args.study.resolve().parent.parent
    study = json.loads(args.study.read_text())
    path = run / study['dataset']['manifest']
    if provenance.file_sha256(path) != study['dataset']['file_sha256']:
        raise ValueError('audited membership manifest changed')
    dataset = json.loads(path.read_text())
    verify_gates(run)
    verify_train_freeze(run, dataset, study)
    if study['cases'] != dataset['cases'] or len(study['cases']) != 3:
        raise ValueError('preselected held-out cases changed')
    if (study['schedule']['sigmas'] != SIGMAS or study['model_variant'] != 'dev'
            or study['seed'] != 42 or study['prompt'] != ''
            or study['guidance'] != {'cfg': 1, 'stg': 0, 'rescale': 0}):
        raise ValueError('removal study differs from exact unguided dev D0 task')
    for component in study['components'].values():
        if provenance.file_sha256(component['path']) != component['file_sha256']:
            raise ValueError('pinned component changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
        raise ValueError('incorrect deterministic workspace')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    current = session.open_session(argparse.Namespace(model='2.5', gpu_id=args.gpu_id, seed=42),
                                   script='prune.evaluate.mobilewan_removal', prompt='',
                                   transformer_path=Path(study['components']['transformer']['path']))
    context = current.context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(context).hexdigest() != study['text_context']['tensor_sha256']:
        raise ValueError('actual context differs from study')
    if torch.cuda.mem_get_info(current.device)[0] < 44 * 2**30:
        raise MemoryError('removal evaluation needs 44 GiB free before model loading')
    frozen_path = run / 'masks/frozen_train.json'
    frozen = json.loads(frozen_path.read_text())
    identity = {'study_sha256': provenance.file_sha256(args.study),
                'frozen_train_sha256': provenance.file_sha256(frozen_path),
                'evaluation_source_sha256': provenance.file_sha256(__file__),
                'shared_source_sha256': source_identity()[0]}
    for role, actual in (('transformer', current.model.paths.transformer()),
                         ('text_encoder', current.model.paths.text_encoder()),
                         ('video_vae', current.model.paths.video_vae())):
        if Path(actual).resolve() != Path(study['components'][role]['path']).resolve():
            raise ValueError(f'actual {role} differs from pinned component')
    with current.transformer() as model:
        model.eval().requires_grad_(False)
        cases = [evaluate_case(model, current, source, study, run, frozen, identity) for source in study['cases']]
    if sum(len(row['arms']) for case in cases for row in case['sigmas']) != 96:
        raise ValueError('incomplete three-case removal coverage')
    atomic_json(run / 'evaluation/results.json', {'schema_version': 1, 'kind': 'mobilewan_removal_evaluation',
                'status': 'complete', 'identity': identity, 'cases': cases, 'prediction_count': 96,
                'metric_definition': 'FP64 squared error to recorded capture over generated latent frames; clean frame excluded',
                'limitation': 'three selected actors and one seed; MSE does not measure perceptual quality'})


if __name__ == '__main__':
    main()
