"""Complete pooled train selection and final cohort schema controls."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.prune.data.head_score_dataset import SIGMAS
from scripts.prune.score.mobilewan_aggregate import summarize, train_allocation, validate_final


def fixtures():
    sources, records = [], []
    rng = np.random.default_rng(42)
    for i in range(8):
        split = 'train' if i < 4 else 'held_out'
        source = {'actor': str(i//2), 'split': split, 'relative_dir': str(i)}
        sources.append(source)
        records.append({'identity': source, 'energies': {'0.attn1': rng.uniform(.1, 2, (8, 3)).tolist(),
                                                       '1.attn1': rng.uniform(.1, 5, (8, 2)).tolist()}})
    dataset = {'sources': sources, 'membership_semantic_sha256': 'a'*64}
    widths = {'0.attn1': 3, '1.attn1': 2}
    result = {'schema_version': 1, 'kind': 'mobilewan_head_scores', 'status': 'complete', 'sigmas': SIGMAS,
              'membership_semantic_sha256': 'a'*64, 'cohorts': summarize(records, sources),
              'coverage': {'views': 8, 'view_sigma_observations': 64, 'eligible_heads': 5}}
    return sources, records, dataset, widths, result


def test_all_three_cohorts_and_paired_bootstrap_identities():
    sources, records, dataset, widths, result = fixtures()
    assert validate_final(result, dataset, widths) == result
    all_stats = result['cohorts']['all']['layers']['0']
    values = np.array([r['energies']['0.attn1'] for r in records])
    np.testing.assert_allclose(all_stats['raw_mean'][0], values.mean(1).mean(0))
    for cohort in result['cohorts'].values():
        assert cohort['layers']['0']['bootstrap']['draws_sha256'] == cohort['layers']['1']['bootstrap']['draws_sha256']
        assert cohort['layers']['0']['bootstrap']['actors'] == sorted(set(cohort['layers']['0']['bootstrap']['actors']))
    # Held-out values cannot enter the train-only selection helper.
    train = [r for r, s in zip(records, sources) if s['split'] == 'train']
    scores, masks = train_allocation(train, [0, .1, .23, .33])
    before = deepcopy(masks)
    for r in records[4:]:
        r['energies']['0.attn1'] = [[1e10]*3 for _ in range(8)]
    assert train_allocation(train, [0, .1, .23, .33])[1] == before
    assert masks['global_order'] == result['cohorts']['train']['ranking_ascending']['pooled']
    assert sum(len(s) for s in scores.values()) == 5


@pytest.mark.parametrize('defect', ['missing_ci', 'nan', 'count', 'rank', 'head'])
def test_final_schema_rejects_missing_or_corrupt_results(defect):
    _, _, dataset, widths, result = fixtures()
    stats = result['cohorts']['all']['layers']['0']
    if defect == 'missing_ci':
        stats['raw_ci95'] = []
    elif defect == 'nan':
        stats['raw_mean'][0][0] = float('nan')
    elif defect == 'count':
        result['coverage']['view_sigma_observations'] = 8
    elif defect == 'rank':
        result['cohorts']['all']['ranking_ascending']['pooled'].reverse()
    else:
        stats['head_indices'] = [0, 1]
    with pytest.raises(ValueError):
        validate_final(result, dataset, widths)
