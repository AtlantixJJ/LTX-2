"""Complete capture-only membership audit for the MobileWan dev-eight study.

The public ClipStore verifies encoding records. This owner adds full source
hashes, actor metadata, declared coverage and exact official schedule pins.
Failures are written per row; an incomplete audit never publishes a manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_pipelines.utils.constants import detect_params
from scripts.onestep_avatar.corpus.dataset import ClipStore
from scripts.onestep_avatar.corpus.subset import load_membership
from scripts.prune.core.provenance import file_sha256
from scripts.prune.score.token_sampling import sample_indices, sampling_record

CSV_SHA = '9ff5416a8bd1e2c6b9f70e620139f9c99a1d5146ec0bcfb6bb0f1b4a8e4bdbd7'
JSON_SHA = 'ab373ebdad2cd804bb1d30f23046543da27edeeecdc7d4c727abf78b9535617c'
SEMANTIC_SHA = '6585d3ef8b6c177d2c3738cf7d6db38938520880afbea5eed1b3a7ca7ad2ff8e'
SIGMAS = [0.9489720463752747, 0.9208537936210632, 0.8844786882400513, 0.8355841040611267,
          0.7663628458976746, 0.6608133316040039, 0.48016369342803955, 0.10000002384185791]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def official_schedule(checkpoint):
    steps = detect_params(str(checkpoint)).num_inference_steps
    if steps != 30:
        raise ValueError('official checkpoint defaults changed')
    stock = LTX2Scheduler().execute(steps=steps)
    eligible = [i for i, value in enumerate(stock) if 0 < float(value) <= .95]
    indices = list(range(8, 30, 3))
    selected = [float(stock[i]) for i in indices]
    if eligible != list(range(8, 30)) or selected != SIGMAS:
        raise ValueError('official schedule changed; reconcile before execution')
    return {'reference_grid_steps': 30, 'eligible_tail_steps': 22, 'selected_level_count': 8,
            'executed_calls_per_view': 8, 'original_grid': stock.tolist(), 'eligible_tail': stock[8:].tolist(),
            'selected_tail_indices': list(range(0, 22, 3)), 'stock_step_indices': indices, 'sigmas': selected,
            'scheduler_arguments': {'steps': 30, 'latent': None, 'others': 'defaults'},
            'scheduler_source_sha256': file_sha256(Path(__import__('inspect').getfile(LTX2Scheduler))),
            'evaluation_schedules': [[s, 0.] for s in selected]}


def choose_cases(sources):
    held = [s for s in sources if s['split'] == 'held_out']
    def key(value):
        return hashlib.sha256(('seed42:' + value).encode()).hexdigest(), value
    actors = sorted({s['actor'] for s in held}, key=key)[:3]
    if len(actors) != 3:
        raise ValueError('three held-out actors required')
    return [min((s for s in held if s['actor'] == actor), key=lambda s: key(s['relative_dir'])) for actor in actors]


def audit(membership_path, output):
    path, output = Path(membership_path), Path(output)
    if file_sha256(path) != JSON_SHA or file_sha256(path.with_suffix('.csv')) != CSV_SHA:
        raise ValueError('membership file pins changed')
    membership = load_membership(path)
    if membership['sha256'] != SEMANTIC_SHA or membership['objective'] != 'white':
        raise ValueError('membership semantic identity or objective changed')
    sources = membership['sources']
    if Counter(s['split'] for s in sources) != {'train': 3104, 'held_out': 256}:
        raise ValueError('membership coverage changed')
    root = Path(membership['corpus_root']).resolve()
    store = ClipStore(membership)
    entries, failures, seen = [], [], set()
    ledger = output / 'provenance/coverage.jsonl'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('w') as log:
        for i, source in enumerate(sources):
            name = source['relative_dir']
            try:
                view = (root / name).resolve()
                if not view.is_relative_to(root) or view in seen:
                    raise ValueError('source alias duplicates or escapes corpus')
                seen.add(view)
                clip = store.load(name, require_guide=False)
                if clip.z_y.dtype != torch.bfloat16 or not torch.isfinite(clip.z_y).all():
                    raise ValueError('capture must be finite BF16')
                shape = list(clip.z_y.shape)
                if shape not in ([128, 18, 32, 32], [128, 28, 32, 32]) or clip.fps != 30:
                    raise ValueError('unexpected full capture geometry or fps')
                for filename, field in [('rgb.mp4', 'rgb_sha256'), ('../../meta.json', 'clip_metadata_sha256')]:
                    if file_sha256(view / filename) != source[field]:
                        raise ValueError(f'{filename}: content changed')
                actor = str(json.loads((view / '../../meta.json').read_text())['actor']['id'])
                if actor != source['actor']:
                    raise ValueError('actor metadata differs')
                encoding = clip.encode_records['capture']
                if encoding['pixel_frames'] != (shape[1] - 1)*8 + 1:
                    raise ValueError('capture pixel span differs')
                indices = sample_indices(shape[1]*1024, 32, 32, 1, torch.device('cpu'))
                entry = {**source, 'resolved_dir': str(view), 'capture': str(view / 'ltx_vae_latent_white.pt'),
                         'sampler': sampling_record(indices, tokens=shape[1]*1024, height=32, width=32,
                                                    stride=1, sampler='stride')}
                entries.append(entry)
                status = {'view': name, 'status': 'verified'}
            except (OSError, ValueError, KeyError, SystemExit) as exc:
                status = {'view': name, 'status': 'failed', 'error': str(exc)}
                failures.append(status)
            log.write(json.dumps(status) + '\n')
            log.flush()
            if (i + 1) % 100 == 0:
                print(f'audited {i+1}/{len(sources)}, failures={len(failures)}', flush=True)
    atomic_json(output / 'provenance/failures.json', failures)
    if failures or len(entries) != 3360:
        raise ValueError('incomplete audit; inspect failure ledger')
    manifest = {'schema_version': 1, 'kind': 'mobilewan_capture_dataset', 'membership_file_sha256': JSON_SHA,
                'membership_csv_sha256': CSV_SHA, 'membership_semantic_sha256': SEMANTIC_SHA,
                'sources': entries, 'cases': choose_cases(entries), 'requires_guide_latent': False}
    atomic_json(output / 'inputs/dataset_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--membership', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = audit(args.membership, args.output)
    print(f'verified {len(manifest["sources"])} views', flush=True)


if __name__ == '__main__':
    main()
