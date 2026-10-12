"""Complete-coverage train mask freezing and final actor-cluster head statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prune.core import provenance
from scripts.prune.data import whole_clip
from scripts.prune.data.head_score_dataset import SIGMAS, atomic_json
from scripts.prune.score import mobilewan_records
from scripts.prune.score.export_pruned import checkpoint_mask_widths
from scripts.prune.score.mobilewan_dataset import source_identity, verify_train_freeze
from scripts.prune.score.mobilewan_heads import cohort_statistics, global_masks

DEFINITIONS = {
    'raw_energy': 'mean generated-query squared norm of each gated head projected without output bias',
    'pooled': 'equal sigma mean per view followed by equal view mean',
    'normalized_mean': 'raw cohort mean divided by mean head energy of its block',
    'std': 'sample dispersion across views, ddof=1; normalized std divides by fixed cohort block mean',
    'ci95': 'pointwise actor-cluster percentile confidence interval for a mean; not individual ranges',
    'bootstrap': '2000 actor draws with replacement, seed42; all actor views and sigmas retained with multiplicity',
    'bootstrap_normalization': 'recompute equal-view mean and block denominator within each replicate',
    'conditions': 'row zero pooled, rows one through eight follow exact selected sigma order',
    'selection': 'ascending pooled train normalized score, stable original block/head ties, one head per block',
    'scope': 'training-free, unrecovered functional head removal; local energy does not prove perceptual usefulness',
}


def collect(run, dataset, study, widths, split):
    sources = sorted((s for s in dataset['sources'] if split == 'all' or s['split'] == split),
                     key=lambda s: s['relative_dir'])
    if not sources or len({s['relative_dir'] for s in sources}) != len(sources):
        raise ValueError('empty or duplicate membership coverage')
    code_hash = source_identity()[0]
    records, pins = [], {}
    for source in sources:
        view = source['relative_dir']
        path = mobilewan_records.record_path(run / 'scores/per_view', view)
        inputs = run / 'inputs/views' / hashlib.sha256(view.encode()).hexdigest()
        manifest = whole_clip.load_manifest(inputs)
        if (manifest['sigmas'] != SIGMAS or manifest['seed'] != 42
                or manifest['videos'][0]['artifacts']['capture_sha256'] != source['capture_latent_sha256']):
            raise ValueError('saved input manifest differs from membership or task')
        row = manifest['videos'][0]
        epsilon = whole_clip.load_epsilon(inputs, row)
        shape = source['shape']
        if list(epsilon.shape) != [1, shape[1]*shape[2]*shape[3], shape[0]]:
            raise ValueError('saved noise shape differs from complete clip')
        identity = {'view': view, 'actor': source['actor'], 'split': source['split'],
                    'capture_sha256': source['capture_latent_sha256'],
                    'epsilon_sha256': row['artifacts']['epsilon_sha256'],
                    'context_sha256': study['text_context']['tensor_sha256'],
                    'transformer_sha256': study['components']['transformer']['file_sha256'],
                    'source_sha256': code_hash, 'sampler': source['sampler']}
        record = mobilewan_records.read_record(path, identity, widths)
        for j in range(8):
            observation = mobilewan_records.observation_path(run / 'scores/per_sigma', view, j)
            energies = mobilewan_records.read_observation(observation, identity, widths, j)
            if energies != {block: rows[j] for block, rows in record['energies'].items()}:
                raise ValueError('complete view differs from saved sigma observations')
        records.append(record)
        pins[view] = provenance.file_sha256(path)
    return sources, records, pins


def train_allocation(records, fractions):
    blocks = sorted(records[0]['energies'], key=lambda b: int(b.split('.')[0]))
    scores = {}
    for block in blocks:
        values = np.array([r['energies'][block] for r in records], dtype=np.float64)
        mean = values.mean(axis=1).mean(axis=0)
        denominator = mean.mean()
        if not np.isfinite(mean).all() or denominator <= 0:
            raise ValueError('invalid pooled train block mean')
        scores[int(block.split('.')[0])] = (mean / denominator).tolist()
    return scores, global_masks(scores, fractions)


def summarize(records, sources, *, replicates=2000):
    cohorts = {}
    blocks = sorted(records[0]['energies'], key=lambda b: int(b.split('.')[0]))
    for cohort in ('train', 'held_out', 'all'):
        selected = [i for i, s in enumerate(sources) if cohort == 'all' or s['split'] == cohort]
        actors = [sources[i]['actor'] for i in selected]
        layers = {}
        for block in blocks:
            values = np.array([records[i]['energies'][block] for i in selected], dtype=np.float64)
            stats = cohort_statistics(values, actors, replicates=replicates, seed=42)
            stats['head_indices'] = list(range(values.shape[2]))
            layers[block.split('.')[0]] = stats
        ranks = []
        for condition in range(9):
            order = sorted((layer['normalized_mean'][condition][head], int(block), head)
                           for block, layer in layers.items() for head in layer['head_indices'])
            ranks.append([[block, head] for _, block, head in order])
        cohorts[cohort] = {'view_count': len(selected), 'actor_count': len(set(actors)), 'layers': layers,
                           'ranking_ascending': {'pooled': ranks[0], 'per_sigma': ranks[1:]}}
    return cohorts


def validate_final(record, dataset, widths):
    if (record.get('schema_version') != 1 or record.get('kind') != 'mobilewan_head_scores'
            or record.get('status') != 'complete' or record.get('sigmas') != SIGMAS
            or record.get('membership_semantic_sha256') != dataset['membership_semantic_sha256']
            or set(record.get('cohorts', {})) != {'train', 'held_out', 'all'}):
        raise ValueError('invalid final head-score schema or coverage identity')
    expected_blocks = {b.split('.')[0]: h for b, h in widths.items()}
    eligible = {(int(b), h) for b, count in expected_blocks.items() for h in range(count)}
    for cohort, statistics in record['cohorts'].items():
        sources = [s for s in dataset['sources'] if cohort == 'all' or s['split'] == cohort]
        if (statistics['view_count'] != len(sources)
                or statistics['actor_count'] != len({s['actor'] for s in sources})
                or set(statistics['layers']) != set(expected_blocks)):
            raise ValueError('final cohort view/actor/block coverage differs')
        for block, layer in statistics['layers'].items():
            count = expected_blocks[block]
            if layer['head_indices'] != list(range(count)) or layer['view_count'] != len(sources):
                raise ValueError('final original head identity or observation count differs')
            for key in ('raw_mean', 'raw_std', 'raw_min', 'raw_max', 'normalized_mean', 'normalized_std'):
                array = np.asarray(layer[key], dtype=np.float64)
                if array.shape != (9, count) or not np.isfinite(array).all() or np.any(array < 0):
                    raise ValueError('missing or nonfinite per-head statistics')
            for key in ('raw_ci95', 'normalized_ci95'):
                intervals = np.asarray(layer[key], dtype=np.float64)
                if (intervals.shape != (9, count, 2) or not np.isfinite(intervals).all()
                        or np.any(intervals[..., 0] > intervals[..., 1])):
                    raise ValueError('missing or invalid pointwise confidence intervals')
            for key, shape in (('layer_mean', (9,)), ('layer_std', (9,)), ('layer_ci95', (9, 2))):
                array = np.asarray(layer[key], dtype=np.float64)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError('missing or nonfinite layer summaries')
            if (layer['bootstrap']['replicates'] != 2000 or layer['bootstrap']['seed'] != 42
                    or len(layer['bootstrap']['draws_sha256']) != 64):
                raise ValueError('incorrect actor bootstrap settings or identity')
            if not np.allclose(np.array(layer['normalized_mean']).mean(axis=-1), 1., rtol=0, atol=1e-12):
                raise ValueError('normalized block means differ from one')
            denominator = np.array(layer['raw_mean']).mean(axis=-1, keepdims=True)
            if (not np.allclose(layer['normalized_mean'], np.array(layer['raw_mean']) / denominator, rtol=0, atol=1e-12)
                    or not np.allclose(layer['normalized_std'], np.array(layer['raw_std']) / denominator, rtol=0, atol=1e-12)):
                raise ValueError('raw and normalized statistics disagree')
        order = statistics['ranking_ascending']
        rows = [order['pooled'], *order['per_sigma']]
        if len(rows) != 9 or any(len(row) != len(eligible) or set(map(tuple, row)) != eligible for row in rows):
            raise ValueError('incomplete descriptive ranks')
        for condition, row in enumerate(rows):
            expected = sorted(eligible, key=lambda pair: (
                statistics['layers'][str(pair[0])]['normalized_mean'][condition][pair[1]], pair[0], pair[1]))
            if list(map(tuple, row)) != expected:
                raise ValueError('descriptive rank order differs from scores')
    if record['coverage'] != {'views': len(dataset['sources']), 'view_sigma_observations': len(dataset['sources'])*8,
                              'eligible_heads': len(eligible)}:
        raise ValueError('final observation or head coverage differs')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--freeze-train', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Recompute final statistics from saved raw records, without GPU work.')
    args = parser.parse_args()
    if args.freeze_train and args.verify:
        parser.error('--freeze-train and --verify are separate operations')
    run = args.study.resolve().parent.parent
    study = json.loads(args.study.read_text())
    path = run / study['dataset']['manifest']
    if provenance.file_sha256(path) != study['dataset']['file_sha256']:
        raise ValueError('audited dataset manifest changed')
    dataset = json.loads(path.read_text())
    if len(dataset['sources']) != 3360:
        raise ValueError('final study requires all 3360 supplied views')
    architecture = checkpoint_mask_widths(study['components']['transformer']['path'])
    widths = {b: h for b, h in architecture.items() if b.endswith('.attn1')}
    if args.freeze_train:
        destination = run / 'masks/frozen_train.json'
        if destination.exists():
            verify_train_freeze(run, dataset, study)
            print('existing frozen train masks verified', flush=True)
            return
        for source in dataset['sources']:
            held_path = mobilewan_records.record_path(run / 'scores/per_view', source['relative_dir'])
            sigma_paths = [mobilewan_records.observation_path(run / 'scores/per_sigma', source['relative_dir'], j) for j in range(8)]
            if source['split'] == 'held_out' and (held_path.exists() or any(p.exists() for p in sigma_paths)):
                raise ValueError('cannot create new train mask selection after held-out scoring')
        sources, records, pins = collect(run, dataset, study, widths, 'train')
        if len(sources) != 3104:
            raise ValueError('train freeze requires all 3104 supplied train views')
        scores, allocation = train_allocation(records, study['removal_fractions'])
        frozen = {'schema_version': 1, 'kind': 'mobilewan_train_masks', 'selection_cohort': 'train',
                  'train_view_count': len(sources), 'train_observation_count': len(sources)*8,
                  'membership_semantic_sha256': dataset['membership_semantic_sha256'],
                  'transformer_sha256': study['components']['transformer']['file_sha256'],
                  'train_records': pins, 'scores': scores, 'allocation': allocation}
        for budget in allocation['budgets']:
            masks = {f'{block}.attn1': values for block, values in budget['masks'].items()}
            masks.update({b: [1]*h for b, h in architecture.items() if b.endswith('.attn2')})
            atomic_json(run / f"masks/removal_{round(budget['requested_fraction']*100):03d}.json",
                        {'schema_version': 1, 'kind': 'mobilewan_functional_masks', 'selection_cohort': 'train',
                         'membership_semantic_sha256': dataset['membership_semantic_sha256'],
                         'transformer_sha256': frozen['transformer_sha256'], 'budget': budget, 'masks': masks})
        atomic_json(destination, frozen)
        print('complete train coverage verified; train-only masks frozen', flush=True)
        return
    verify_train_freeze(run, dataset, study)
    sources, records, pins = collect(run, dataset, study, widths, 'all')
    result = {'schema_version': 1, 'kind': 'mobilewan_head_scores', 'status': 'complete', 'sigmas': SIGMAS,
              'membership_file_sha256': dataset['membership_file_sha256'],
              'membership_csv_sha256': dataset['membership_csv_sha256'],
              'membership_semantic_sha256': dataset['membership_semantic_sha256'], 'definitions': DEFINITIONS,
              'components': study['components'], 'text_context': study['text_context'], 'schedule': study['schedule'],
              'coverage': {'views': len(sources), 'view_sigma_observations': len(sources)*8,
                           'eligible_heads': sum(widths.values())},
              'architecture': {'self_attention_heads': widths}, 'cohorts': summarize(records, sources),
              'provenance': {'view_records': pins, 'source_sha256': source_identity()[0],
                             'frozen_train_sha256': provenance.file_sha256(run / 'masks/frozen_train.json')},
              'train_selection': json.loads((run / 'masks/frozen_train.json').read_text())}
    validate_final(result, dataset, widths)
    train_records = [r for r in records if r['identity']['split'] == 'train']
    _, expected_allocation = train_allocation(train_records, study['removal_fractions'])
    if json.loads(json.dumps(expected_allocation)) != result['train_selection']['allocation']:
        raise ValueError('frozen masks differ from complete pooled train scores')
    if args.verify:
        saved = json.loads((run / 'head_scores.json').read_text())
        validate_final(saved, dataset, widths)
        if saved != result:
            raise ValueError('saved final statistics differ from independent raw-record rebuild')
        atomic_json(run / 'validation/head_scores_validation.json', {'schema_version': 1, 'passed': True,
                    'head_scores_file_sha256': provenance.file_sha256(run / 'head_scores.json'),
                    'coverage': result['coverage'], 'scope': 'complete raw-record recomputation'})
        print('saved final head scores verified against all raw observations', flush=True)
        return
    atomic_json(run / 'head_scores.json', result)
    print('complete final head scores published', flush=True)


if __name__ == '__main__':
    main()
