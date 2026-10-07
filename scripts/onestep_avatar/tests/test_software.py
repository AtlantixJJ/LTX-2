"""Dirty owner bytes and runtime changes invalidate current claims, not history."""
from copy import deepcopy

import pytest

from scripts.onestep_avatar import software


@pytest.mark.parametrize('profile', software.PROFILES)
def test_declared_real_owner_inventory_and_runtime(profile):
    mode = None if profile == 'decoding' else 'causal'
    record = software.capture(profile, mode)
    software.check_current(record)
    assert 'scripts/onestep_avatar/model/common.py' in record['sources']
    assert 'scripts/onestep_avatar/model/adapters.py' in record['sources']
    assert 'scripts/onestep_avatar/precompute.py' in record['sources']
    assert 'packages/ltx-core/src/ltx_core/model/transformer/model.py' in record['sources']
    if profile == 'decoding':
        assert 'scripts/prune/evaluate/decode.py' in record['sources']
        assert any('/video_vae/' in name for name in record['sources'])
    else:
        assert 'scripts/onestep_avatar/model/causal.py' in record['sources']
        assert 'scripts/onestep_avatar/model/bidirectional.py' not in record['sources']
    assert record['runtime']['distributions']['torch'] is not None


@pytest.mark.parametrize('owner', ['model/common.py', 'model/causal.py', 'model/sampling.py', 'model/adapters.py', 'precompute.py'])
def test_changed_owner_fails_current_check_and_history_stays_readable(monkeypatch, owner):
    record = software.capture('evaluation', 'causal')
    original = software.sha256
    path = 'scripts/onestep_avatar/' + owner
    monkeypatch.setattr(software, 'sha256', lambda p: 'f'*64 if str(p.relative_to(software.ROOT)) == path else original(p))
    software.validate(record)
    with pytest.raises(ValueError, match='changed since preflight'):
        software.check_current(record)
    assert record['sources']['scripts/onestep_avatar/evaluate.py'] == original(software.ROOT/'scripts/onestep_avatar/evaluate.py')


@pytest.mark.parametrize('phase', ['before_write', 'during_write'])
def test_decoder_owner_change_prevents_render_publication(tmp_path, monkeypatch, phase):
    import torch
    from scripts.onestep_avatar import media
    from ltx_trainer import video_utils

    pixels, record = media.render_panels(
        [media.Panel('p0', 'Output', torch.zeros(1, 3, 16, 16), (0,))],
        question='What is generated?', layout='comparison', fps=30,
    )
    record['software'] = software.capture('decoding')
    original = software.sha256
    owner = software.ROOT/'packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py'

    def change():
        monkeypatch.setattr(software, 'sha256', lambda p: 'f'*64 if p == owner else original(p))

    def writer(_pixels, path, **_kwargs):
        path.write_bytes(b'controlled saved video')
        change()

    monkeypatch.setattr(video_utils, 'save_video', writer)
    if phase == 'before_write':
        change()
    with pytest.raises(ValueError, match='changed since preflight'):
        media.save_render(pixels, record, tmp_path/'render')
    assert not (tmp_path/'render/rendering.json').exists()
    assert (tmp_path/'render/comparison.mp4').exists() == (phase == 'during_write')


def test_runtime_change_fails_current_check(monkeypatch):
    record = software.capture('training', 'bidirectional')
    changed = deepcopy(record['runtime'])
    changed['distributions']['torch'] = 'different-build'
    monkeypatch.setattr(software, 'runtime_versions', lambda: changed)
    software.validate(record)
    with pytest.raises(ValueError, match='changed since preflight'):
        software.check_current(record)


@pytest.mark.parametrize('defect', ['hash', 'path', 'digest', 'runtime', 'empty'])
def test_malformed_manifest_refused(defect):
    record = software.capture('evaluation', 'bidirectional')
    if defect == 'hash':
        record['sha256'] = 'a'*64
    elif defect == 'path':
        record['sources']['../outside.py'] = 'a'*64
    elif defect == 'digest':
        record['sources'][next(iter(record['sources']))] = 'bad'
    elif defect == 'runtime':
        record['runtime'] = {}
    else:
        record['sources'] = {}
    if defect != 'hash':
        record['sha256'] = software._digest(record)
    with pytest.raises(ValueError, match='software manifest'):
        software.validate(record)


@pytest.mark.parametrize('change', ['added', 'removed'])
def test_dependency_group_inventory_changes_are_bound(tmp_path, monkeypatch, change):
    monkeypatch.setattr(software, 'ROOT', tmp_path)
    monkeypatch.setattr(software, 'COMMON', ())
    monkeypatch.setattr(software, 'ENTRIES', {p: () for p in software.PROFILES})
    monkeypatch.setattr(software, 'DEPENDENCIES', ())
    monkeypatch.setattr(software, 'GROUPS', ('dependencies',))
    mode = tmp_path/'scripts/onestep_avatar/model/causal.py'
    mode.parent.mkdir(parents=True)
    mode.write_text('mode source')
    group = tmp_path/'dependencies'
    group.mkdir()
    first = group/'first.py'
    first.write_text('first')
    second = group/'second.py'
    second.write_text('second')
    record = software.capture('evaluation', 'causal')
    if change == 'added':
        (group/'new.py').write_text('new')
    else:
        second.unlink()
    software.validate(record)
    with pytest.raises(ValueError, match='changed since preflight'):
        software.check_current(record)
