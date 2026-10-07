"""A second GPU observation precedes every launch and rolls back owned claims."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.onestep_avatar import queue


def inventory(**usage):
    return SimpleNamespace(stdout='\n'.join(f'{gpu}, {usage.get(str(gpu), 0)}' for gpu in range(8)))


def setup_dispatch(tmp_path, kind):
    job = {'id': 'case', 'kind': kind, 'sha256': 'a' * 64,
           'output': str(tmp_path / 'output'), 'dependencies': []}
    if kind == 'render':
        (tmp_path / 'panel.pt').write_bytes(b'saved input readiness only')
        spec = tmp_path / 'render.json'
        spec.write_text(json.dumps({'comparisons': [{'panels': [{'latent': 'panel.pt'}]}]}))
        job['render_spec'] = str(spec)
    state_path = tmp_path / 'state.json'
    with queue.queue_state(state_path, [job]):
        pass
    claims = tmp_path / 'claims'
    claims.mkdir()
    (claims / '5').write_text('foreign legacy reservation')
    return job, state_path, claims


def assert_no_launch_artifacts(tmp_path, state_path, original, claims):
    assert state_path.read_bytes() == original
    assert not (tmp_path / 'output').exists() and not (tmp_path / 'logs').exists()
    assert sorted(p.name for p in claims.iterdir() if p.name.isdigit()) == ['5']
    assert (claims / '5').read_text() == 'foreign legacy reservation'


@pytest.mark.parametrize('kind', ['evaluate', 'render', 'decode', 'train'])
@pytest.mark.parametrize('memory', [1024, 4096])
def test_contended_claim_set_is_released_before_any_attempt(tmp_path, monkeypatch, kind, memory):
    job, state_path, claims = setup_dispatch(tmp_path, kind)
    original = state_path.read_bytes()
    expected = {0, 1, 2, 3} if kind == 'train' else {4}
    calls = []

    def query(*args, **kwargs):
        assert kwargs['check'] and kwargs['capture_output'] and kwargs['text']
        calls.append(args)
        if len(calls) == 1:
            return inventory()
        assert {int(p.name) for p in claims.iterdir() if p.name.isdigit()} == expected | {5}
        assert all(json.loads((claims / str(gpu)).read_text())['job'] == 'case' for gpu in expected)
        return inventory(**{str(min(expected)): memory})

    monkeypatch.setattr(subprocess, 'run', query)
    monkeypatch.setattr(queue, 'run_child', lambda *_a, **_k: pytest.fail('started contended child'))
    state = queue.read_queue_state(state_path, [job])
    assert not queue.dispatch_ready([job], state, state_path, claims)
    assert len(calls) == 2
    assert_no_launch_artifacts(tmp_path, state_path, original, claims)


@pytest.mark.parametrize('error', ['command', 'malformed', 'incomplete', 'interrupt'])
@pytest.mark.parametrize('kind', ['evaluate', 'train'])
def test_failed_second_observation_releases_claims_and_preserves_journal(tmp_path, monkeypatch, kind, error):
    job, state_path, claims = setup_dispatch(tmp_path, kind)
    original = state_path.read_bytes()
    calls = []

    def query(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return inventory()
        assert (claims / ('0' if kind == 'train' else '4')).is_file()
        if error == 'command':
            raise subprocess.CalledProcessError(1, 'nvidia-smi')
        if error == 'interrupt':
            raise KeyboardInterrupt
        return SimpleNamespace(stdout='0, unavailable' if error == 'malformed' else '0, 0')

    monkeypatch.setattr(subprocess, 'run', query)
    monkeypatch.setattr(queue, 'run_child', lambda *_a, **_k: pytest.fail('started child without inventory'))
    expected = {'command': subprocess.CalledProcessError, 'interrupt': KeyboardInterrupt}.get(error, ValueError)
    with pytest.raises(expected):
        queue.dispatch_ready([job], queue.read_queue_state(state_path, [job]), state_path, claims)
    assert len(calls) == 2
    assert_no_launch_artifacts(tmp_path, state_path, original, claims)


def test_subsequent_selection_uses_changed_inventory_and_rechecks_it(tmp_path, monkeypatch):
    first, state_path, claims = setup_dispatch(tmp_path, 'evaluate')
    second = {**first, 'id': 'second', 'sha256': 'b' * 64, 'output': str(tmp_path / 'second')}
    jobs = [first, second]
    state = queue.read_queue_state(state_path, jobs)
    observations, launches = [], []

    def query(*_args, **_kwargs):
        observations.append({int(p.name) for p in claims.iterdir() if p.name.isdigit()})
        if len(observations) == 1:
            return inventory(**{'3': 1024})
        if len(observations) == 2:
            assert observations[-1] == {4, 5}
            return inventory(**{'4': 1024})
        assert observations[-1] == {3, 5}
        return inventory(**{'4': 1024})

    def launch(job, owned, _log, **_kwargs):
        launches.append((job['id'], set(owned.owned)))
        owned.release()

    monkeypatch.setattr(subprocess, 'run', query)
    monkeypatch.setattr(queue, 'run_child', launch)
    assert queue.dispatch_ready(jobs, state, state_path, claims)
    assert launches == [('second', {3})]
    assert observations == [{5}, {4, 5}, {3, 5}]
    assert state['jobs']['case']['state'] == 'pending'
    assert sorted(p.name for p in claims.iterdir() if p.name.isdigit()) == ['5']
