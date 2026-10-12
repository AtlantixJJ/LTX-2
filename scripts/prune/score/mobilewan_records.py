"""Atomic per-view raw score records with strict resume identity checks.

A record contains all eight sigma observations and every discovered self-attention
head. The worker supplies complete provenance, including saved noise, source,
context, model, code and sampler hashes. A changed identity fails on resume;
callers must retain the failed record for diagnosis rather than overwrite it.
Temporary writes are private to one worker and become visible by atomic rename.
This owner validates records; it does not regenerate missing observations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from scripts.prune.data.head_score_dataset import SIGMAS

SCHEMA = 1
IDENTITIES = ('view', 'actor', 'split', 'capture_sha256', 'epsilon_sha256',
              'context_sha256', 'transformer_sha256', 'source_sha256', 'sampler')


def record_path(root, view):
    """Use the full relative view identity, never only its repeated basename."""
    return Path(root) / (hashlib.sha256(view.encode()).hexdigest() + '.json')


def validate_record(record, identity, widths):
    if record.get('schema_version') != SCHEMA or record.get('kind') != 'mobilewan_view_energy':
        raise ValueError('unsupported per-view score schema')
    if set(identity) != set(IDENTITIES) or record.get('identity') != identity:
        raise ValueError('per-view score identity changed or incomplete')
    if identity['split'] not in ('train', 'held_out'):
        raise ValueError('invalid score split')
    for key in IDENTITIES:
        if key.endswith('sha256'):
            value = identity[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'invalid {key}')
    sampler = identity['sampler']
    if (sampler.get('sampler') != 'stride' or sampler.get('spatial_stride') != 1
            or sampler.get('clean_frame_included') is not False or sampler.get('sample_tokens', 0) < 1):
        raise ValueError('per-view record requires full generated-token coverage')
    if record.get('sigmas') != SIGMAS or not widths:
        raise ValueError('missing or changed sigma/head coverage')
    energies = record.get('energies')
    if not isinstance(energies, dict) or set(energies) != set(widths):
        raise ValueError('missing or unexpected self-attention blocks')
    for block, heads in widths.items():
        rows = energies[block]
        if not block.endswith('.attn1') or heads < 1 or not isinstance(rows, list) or len(rows) != 8:
            raise ValueError('invalid self-attention coverage')
        for values in rows:
            if (not isinstance(values, list) or len(values) != heads
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
                raise ValueError('invalid raw head energies')
    return record


def read_record(path, identity, widths):
    return validate_record(json.loads(Path(path).read_text()), identity, widths)


def write_record(path, identity, widths, energies):
    path = Path(path)
    record = validate_record({'schema_version': SCHEMA, 'kind': 'mobilewan_view_energy',
                              'identity': identity, 'sigmas': SIGMAS, 'energies': energies}, identity, widths)
    if path.exists():
        existing = read_record(path, identity, widths)
        if existing != record:
            raise ValueError('attempt to replace different completed scores')
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        with temp.open('x') as stream:
            json.dump(record, stream, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic create prevents competing workers from replacing completed bytes.
        try:
            os.link(temp, path)
        except FileExistsError:
            if read_record(path, identity, widths) != record:
                raise ValueError('competing worker published different completed scores') from None
    finally:
        temp.unlink(missing_ok=True)
    return record


def observation_path(root, view, index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('invalid selected sigma index')
    return record_path(root, view).with_suffix(f'.sigma{index}.json')


def read_observation(path, identity, widths, index):
    record = json.loads(Path(path).read_text())
    if (record.get('schema_version') != SCHEMA or record.get('kind') != 'mobilewan_sigma_energy'
            or record.get('sigma_index') != index or record.get('sigma') != SIGMAS[index]):
        raise ValueError('changed observation sigma identity')
    # Reuse the complete head/identity validator without publishing a pooled record.
    validate_record({'schema_version': SCHEMA, 'kind': 'mobilewan_view_energy', 'identity': record['identity'],
                     'sigmas': SIGMAS, 'energies': {k: [v]*8 for k, v in record['energies'].items()}}, identity, widths)
    return record['energies']


def write_observation(path, identity, widths, index, energies):
    validate_record({'schema_version': SCHEMA, 'kind': 'mobilewan_view_energy', 'identity': identity,
                     'sigmas': SIGMAS, 'energies': {k: [v]*8 for k, v in energies.items()}}, identity, widths)
    path = Path(path)
    if path.exists():
        if read_observation(path, identity, widths, index) != energies:
            raise ValueError('different completed observation already exists')
        return
    record = {'schema_version': SCHEMA, 'kind': 'mobilewan_sigma_energy', 'identity': identity,
              'sigma_index': index, 'sigma': SIGMAS[index], 'energies': energies}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        with temp.open('x') as stream:
            json.dump(record, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if read_observation(path, identity, widths, index) != energies:
                raise ValueError('competing worker published different observation') from None
    finally:
        temp.unlink(missing_ok=True)
