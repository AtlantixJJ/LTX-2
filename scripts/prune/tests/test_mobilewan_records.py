"""Reject partial coverage and changed provenance before accepting resume."""
from copy import deepcopy

import pytest

from scripts.prune.score.mobilewan_records import read_record, record_path, write_record


def fixture_identity():
    return {'view': 'Part_1/0008_01/views/view00', 'actor': '8', 'split': 'train',
            **{key: 'a'*64 for key in ('capture_sha256', 'epsilon_sha256', 'context_sha256',
                                     'transformer_sha256', 'source_sha256')},
            'sampler': {'sampler': 'stride', 'spatial_stride': 1, 'clean_frame_included': False,
                        'sample_tokens': 17408}}


def test_atomic_resume_and_changed_identity(tmp_path):
    identity = fixture_identity()
    widths = {'0.attn1': 2, '1.attn1': 3}
    energies = {key: [[float(h+1) for h in range(width)] for _ in range(8)] for key, width in widths.items()}
    path = record_path(tmp_path, identity['view'])
    result = write_record(path, identity, widths, energies)
    assert read_record(path, identity, widths) == result
    assert write_record(path, identity, widths, energies) == result
    other = deepcopy(identity)
    other['epsilon_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='identity changed'):
        read_record(path, other, widths)
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('defect', ['sigma', 'head', 'nan', 'block'])
def test_partial_records_never_published(tmp_path, defect):
    energies = {'0.attn1': [[1., 2.] for _ in range(8)]}
    if defect == 'sigma':
        energies['0.attn1'].pop()
    elif defect == 'head':
        energies['0.attn1'][0].pop()
    elif defect == 'nan':
        energies['0.attn1'][0][0] = float('nan')
    else:
        energies = {}
    path = tmp_path / 'record.json'
    with pytest.raises(ValueError):
        write_record(path, fixture_identity(), {'0.attn1': 2}, energies)
    assert not path.exists()


def test_sigma_observations_resume_independently(tmp_path):
    from scripts.prune.score.mobilewan_records import observation_path, read_observation, write_observation

    identity = fixture_identity()
    widths = {'0.attn1': 2}
    path = observation_path(tmp_path, identity['view'], 3)
    write_observation(path, identity, widths, 3, {'0.attn1': [1., 2.]})
    assert read_observation(path, identity, widths, 3) == {'0.attn1': [1., 2.]}
    with pytest.raises(ValueError, match='sigma identity'):
        read_observation(path, identity, widths, 4)
    with pytest.raises(ValueError):
        write_observation(path, identity, widths, 3, {'0.attn1': [3., 4.]})
