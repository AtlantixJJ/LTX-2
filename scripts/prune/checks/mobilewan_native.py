"""Check exact native dev baseline, stock primitive, score hooks and all-one masks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ltx_core.types import LatentState
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from scripts.prune.core import provenance, session
from scripts.prune.data import whole_clip
from scripts.prune.data.head_score_dataset import SIGMAS, atomic_json
from scripts.prune.score import hooks, token_sampling
from scripts.prune.score.mobilewan_heads import projected_energy


def stock_prediction(model, modality):
    """Use the stock unguided denoising primitive at precisely the saved inputs."""
    mask = torch.ones_like(modality.timesteps, dtype=torch.float32)
    mask[modality.timesteps == 0] = 0
    state = LatentState(latent=modality.latent, denoise_mask=mask, positions=modality.positions,
                        clean_latent=modality.latent, attention_mask=None, keyframes_mask=modality.keyframes_mask)
    schedule = torch.stack([modality.sigma[0], modality.sigma[0] * 0])
    video, audio = SimpleDenoiser(modality.context, None)(model, state, None, schedule, 0)
    if audio is not None or video is None:
        raise ValueError('stock primitive returned unexpected modalities')
    return video.denoised


def check(baseline, current, model):
    manifest = whole_clip.load_manifest(baseline)
    if manifest.get('model_variant') != 'dev' or manifest.get('schedule_policy') != 'direct_ltx2_scheduler':
        raise ValueError('native MobileWan check requires an explicit direct dev baseline')
    if manifest['sigmas'] != SIGMAS:
        raise ValueError('native baseline differs from the eight exact selected sigmas')
    rows = whole_clip.records(manifest)
    if len(rows) != 8 or len({view for view, _ in rows}) != 1:
        raise ValueError('native check requires one complete view and eight sigma observations')
    score_path = Path(projected_energy.__code__.co_filename)
    score_hash = provenance.file_sha256(score_path)
    report = {'schema_version': 1, 'kind': 'mobilewan_native_check', 'runs': [],
              'score_source_sha256': score_hash,
              'stock_scope': 'SimpleDenoiser primitive with saved inputs; no top-level pipeline',
              'baseline_manifest_sha256': provenance.file_sha256(baseline / 'manifest.json')}
    for (view, sigma), row in rows.items():
        grid, modality, c0, _ = whole_clip.build_input(baseline, manifest, view=view, sigma=sigma, current=current)
        shape = grid.tools.target_shape
        indices = token_sampling.sample_indices(modality.latent.shape[1], shape.height, shape.width, 1,
                                                 current.device, sampler='stride')
        energies = {}

        def observe(name, activation, attention):
            if name.endswith('.attn1'):
                if name in energies:
                    raise ValueError('self-attention block executed twice in a direct call')
                energies[name] = projected_energy(activation, attention.to_out[0].weight,
                                                  attention.heads, indices).cpu().tolist()

        stock = stock_prediction(model, modality)
        with hooks.collect_activations(model, 'head', observe):
            observed, _ = model(video=modality, audio=None, perturbations=None)
        if not torch.equal(stock, observed):
            raise ValueError('score hook output differs from stock denoiser primitive')
        expected = {name for name, _ in hooks.iter_video_attention(model) if name.endswith('.attn1')}
        if set(energies) != expected:
            raise ValueError('native score block coverage is incomplete')
        with hooks.attach_head_masks(model, requires_grad=False):
            all_one, _ = model(video=modality, audio=None, perturbations=None)
        if not torch.equal(stock, all_one):
            raise ValueError('all-one functional masks differ from stock denoiser primitive')
        prediction = grid.unpatchify_block(torch.cat([c0, stock[:, c0.shape[1]:]], dim=1), grid.latent_frames)
        recorded = torch.load(whole_clip.latent_path(baseline, row), map_location='cpu', weights_only=True)
        if not torch.isfinite(prediction).all() or not torch.equal(prediction.cpu(), recorded):
            raise ValueError('fresh stock call does not reproduce the saved baseline exactly')
        sampling = token_sampling.sampling_record(indices, tokens=modality.latent.shape[1], height=shape.height,
                                                  width=shape.width, stride=1, sampler='stride')
        report['runs'].append({'view': view, 'sigma': sigma, 'schedule': [sigma, 0.0], 'sampling': sampling,
                               'energies': energies, 'stock_parity': True, 'hook_transparent': True,
                               'all_one_equal': True, 'saved_baseline_equal': True,
                               'clean_frame_equal': torch.equal(prediction[:, :, 0].cpu(), recorded[:, :, 0])})
        print(f'verified native hooks and stock parity: {view} sigma={sigma}', flush=True)
    if provenance.file_sha256(score_path) != score_hash:
        raise ValueError('scoring source changed during native verification')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()
    # Match the queue's declared deterministic policy before opening CUDA.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
        raise ValueError('native check requires the baseline cuBLAS workspace')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = whole_clip.load_manifest(args.baseline)
    checkpoint = Path(manifest['model']['transformer_path'])
    if provenance.checkpoint_fingerprint(checkpoint) != manifest['model']['transformer_fingerprint']:
        raise ValueError('native baseline checkpoint changed')
    current = session.open_session(argparse.Namespace(model='2.5', gpu_id=args.gpu_id, seed=42),
                                   script='prune.checks.mobilewan_native', transformer_path=checkpoint,
                                   prompt=manifest['text_context']['prompt'])
    if torch.backends.cuda.matmul.allow_tf32 or not torch.are_deterministic_algorithms_enabled():
        raise ValueError('native check requires the deterministic baseline numerical policy')
    with current.transformer(checkpoint) as model:
        report = check(args.baseline, current, model)
    atomic_json(args.output, report)


if __name__ == '__main__':
    main()
