"""Continuous queue dispatch checks real state, reservations and completion receipts."""

import json
import subprocess
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.execution import process_registry, queue


def controlled_terminal_child(monkeypatch):
    """Immediate fake Popen has no OS lifetime; retain real identity for our owner."""
    original = process_registry._identity
    monkeypatch.setattr(process_registry, '_identity', lambda pid: (
        {'pid': pid, 'start_ticks': 1, 'command': ['controlled child'], 'terminal': True}
        if pid == 99999999 else original(pid)))


def job(name, dependencies=()):
    return {'id': name, 'kind': 'evaluate', 'output': name, 'dependencies': list(dependencies),
            'arguments': ['--mode', 'causal', '--subset', 'membership.json', '--output', name,
                          '--schedule', '0.725', '0'], 'completion': {'records': [f'{name}/result.json']}}


def write_jobs(path, jobs):
    path.write_text(json.dumps({'schema_version': 1, 'jobs': jobs}))


def inventory(busy=False):
    return SimpleNamespace(stdout='\n'.join(f'{gpu}, {1024 if busy and gpu < 6 else 0}' for gpu in range(8)))


def reservations(path):
    return sorted(file.name for file in path.iterdir() if file.name.isdigit()) if path.exists() else []


def test_loop_waits_then_dispatches_appended_dependency_and_verifies_all_receipts(tmp_path, monkeypatch, controlled_queue_launch, controlled_evaluation_conditions):
    jobs_path, state_path, claims_dir = tmp_path / 'jobs.json', tmp_path / 'state.json', tmp_path / 'claims'
    first, second = job('first'), job('second', ['first'])
    write_jobs(jobs_path, [first])
    claims_dir.mkdir()
    (claims_dir / '5').write_text('historical evaluation')
    ledger = tmp_path / 'processes.json'
    controlled_terminal_child(monkeypatch)
    launched, sleeps, queries = [], [], []

    def gpu_query(*_args, **_kwargs):
        queries.append(1)
        return inventory(busy=len(queries) == 1)

    def sleep(seconds):
        assert reservations(claims_dir) == ['5']
        assert not state_path.exists()
        sleeps.append(seconds)

    def launch(command, **kwargs):
        output = tmp_path / command[command.index('--output') + 1]
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '5'
        assert command[-2:] == ['--gpu-id', '0']
        assert (claims_dir / '5').read_text() == 'historical evaluation'
        saved = json.loads(ledger.read_text())['attempts'][kwargs['env']['ONESTEP_AVATAR_QUEUE_TOKEN']]
        assert saved['job'] == output.name and saved['gpus'] == [5] and saved['state'] == 'active'
        if launched:
            state = json.loads(state_path.read_text())
            assert state['jobs']['first']['state'] == 'complete'
        launched.append(output.name)
        evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {'mode': 'causal'}, output)
        if output.name == 'first':
            write_jobs(jobs_path, [first, second])
        return SimpleNamespace(pid=99999999, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(subprocess, 'run', gpu_query)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    monkeypatch.setattr(queue.time, 'sleep', sleep)
    assert queue.main(['--jobs', str(jobs_path), '--state', str(state_path), '--execute', '--loop',
                       '--process-ledger', str(ledger), '--poll-seconds', '0.5']) == 0
    assert launched == ['first', 'second'] and sleeps == [0.5] and len(queries) == 7
    state = json.loads(state_path.read_text())
    prepared = queue.prepare_jobs(jobs_path)
    assert state['job_order'] == ['first', 'second']
    assert all(queue.verify_receipt(item, state['jobs'][item['id']]['receipt']) for item in prepared)
    assert len(list((tmp_path / 'logs').glob('*.log'))) == 2
    assert reservations(claims_dir) == ['5']
    assert all(row['state'] == 'closed' for row in json.loads(ledger.read_text())['attempts'].values())


def test_loop_refuses_edits_while_waiting_before_any_claim_or_state(tmp_path, monkeypatch):
    path = tmp_path / 'jobs.json'
    original = job('first')
    write_jobs(path, [original])
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: inventory(busy=True))

    def edit(_seconds):
        changed = job('first')
        changed['arguments'][changed['arguments'].index('0.725')] = '0.5'
        write_jobs(path, [changed])

    monkeypatch.setattr(queue.time, 'sleep', edit)
    with pytest.raises(ValueError, match='append unchanged jobs'):
        queue.execute_jobs(path, tmp_path / 'state.json', tmp_path / 'claims', once=False)
    assert not (tmp_path / 'state.json').exists()
    assert reservations(tmp_path / 'claims') == []


@pytest.mark.parametrize('status', ['running', 'failed'])
def test_loop_preserves_unrecovered_or_failed_attempts_without_inventory(tmp_path, monkeypatch, status):
    path, state_path = tmp_path / 'jobs.json', tmp_path / 'state.json'
    write_jobs(path, [job('first')])
    with queue.queue_state(state_path, queue.prepare_jobs(path)) as state:
        state['jobs']['first']['state'] = status
    original = state_path.read_bytes()
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('queried inventory'))
    if status == 'running':
        with pytest.raises(ValueError, match='explicit recovery'):
            queue.execute_jobs(path, state_path, tmp_path / 'claims', once=False)
    else:
        assert queue.execute_jobs(path, state_path, tmp_path / 'claims', once=False) == 1
    assert state_path.read_bytes() == original and not (tmp_path / 'claims').exists()


def test_loop_does_not_accept_complete_label_with_missing_evidence(tmp_path, monkeypatch, controlled_evaluation_conditions):
    path, state_path = tmp_path / 'jobs.json', tmp_path / 'state.json'
    write_jobs(path, [job('first')])
    prepared = queue.prepare_jobs(path)
    output = tmp_path / 'first'
    evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {'mode': 'causal'}, output)
    receipt = queue.completion_receipt(prepared[0])
    with queue.queue_state(state_path, prepared) as state:
        state['jobs']['first'].update(state='complete', receipt=receipt)
    (output / 'generated.pt').unlink()
    original = state_path.read_bytes()
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('queried inventory'))
    with pytest.raises(ValueError, match='missing verified evidence'):
        queue.execute_jobs(path, state_path, tmp_path / 'processes.json', once=False)
    assert state_path.read_bytes() == original


def test_empty_loop_is_read_only(tmp_path, monkeypatch):
    path = tmp_path / 'jobs.json'
    write_jobs(path, [])
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('queried inventory'))
    assert queue.execute_jobs(path, tmp_path / 'state.json', tmp_path / 'claims', once=False) == 0
    assert list(tmp_path.iterdir()) == [path]


def test_interrupt_during_wait_leaves_no_claim_or_state(tmp_path, monkeypatch):
    path = tmp_path / 'jobs.json'
    write_jobs(path, [job('first')])
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: inventory(busy=True))

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(queue.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        queue.execute_jobs(path, tmp_path / 'state.json', tmp_path / 'claims', once=False)
    assert not (tmp_path / 'state.json').exists()
    assert reservations(tmp_path / 'claims') == []


def test_child_failure_stops_loop_and_preserves_partial_outputs(tmp_path, monkeypatch, controlled_queue_launch):
    path, state_path = tmp_path / 'jobs.json', tmp_path / 'state.json'
    write_jobs(path, [job('first'), job('second', ['first'])])
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: inventory())
    launched = []
    controlled_terminal_child(monkeypatch)

    def fail(command, **_kwargs):
        output = tmp_path / command[command.index('--output') + 1]
        launched.append(output.name)
        output.mkdir()
        (output / 'partial.txt').write_text('preserved failed scientific output')
        return SimpleNamespace(pid=99999999, poll=lambda: 1, wait=lambda: 1)

    monkeypatch.setattr(subprocess, 'Popen', fail)
    with pytest.raises(ValueError, match='failed with exit 1'):
        queue.execute_jobs(path, state_path, tmp_path / 'processes.json', once=False)
    state = json.loads(state_path.read_text())
    assert launched == ['first']
    assert state['jobs']['first']['state'] == 'failed'
    assert state['jobs']['second']['state'] == 'pending'
    assert (tmp_path / 'first/partial.txt').read_text() == 'preserved failed scientific output'
    assert reservations(tmp_path / 'claims') == []
    assert all(row['state'] == 'closed' for row in json.loads((tmp_path / 'processes.json').read_text())['attempts'].values())


@pytest.mark.parametrize('interval', ['0', '-1', 'nan', 'inf', '61'])
def test_invalid_poll_interval_fails_before_reading_jobs(interval, tmp_path):
    with pytest.raises(SystemExit):
        queue.main(['--jobs', str(tmp_path / 'missing.json'), '--state', str(tmp_path / 'state.json'),
                    '--execute', '--loop', '--claims-dir', str(tmp_path / 'claims'), '--poll-seconds', interval])
    assert not list(tmp_path.iterdir())


def recovery_fixture(tmp_path, *, child_pid=99999999, publish=False):
    path, state_path = tmp_path / 'jobs.json', tmp_path / 'state.json'
    write_jobs(path, [job('first')])
    prepared = queue.prepare_jobs(path)
    if publish:
        evaluate.save_case(torch.zeros(1, 2, 7, 2, 2), {'mode': 'causal'}, tmp_path / 'first')
    with queue.queue_state(state_path, prepared) as state:
        state['jobs']['first'].update(state='running', child_pid=child_pid)
    return path, state_path, prepared


@pytest.mark.parametrize('publish', [False, True])
def test_recovery_cli_verifies_terminal_outputs_without_inventory(tmp_path, monkeypatch, capsys, publish, controlled_evaluation_conditions):
    path, state_path, prepared = recovery_fixture(tmp_path, publish=publish)
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('queried GPU inventory'))
    assert queue.main(['--jobs', str(path), '--state', str(state_path), '--recover']) == 0
    result = json.loads(capsys.readouterr().out)
    expected = 'complete' if publish else 'failed'
    assert result == {'recovered': True, 'jobs': [{'id': 'first', 'state': expected}]}
    state = json.loads(state_path.read_text())
    assert state['jobs']['first']['state'] == expected
    if publish:
        assert queue.verify_receipt(prepared[0], state['jobs']['first']['receipt'])
    assert not (tmp_path / 'claims').exists() and not (tmp_path / 'logs').exists()


@pytest.mark.parametrize('same_owner', [False, True])
def test_recovery_refuses_real_surviving_child_even_for_current_owner(tmp_path, same_owner):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        path, state_path, _ = recovery_fixture(tmp_path, child_pid=child.pid)
        state = json.loads(state_path.read_text())
        state['owner_pid'] = os.getpid() if same_owner else 99999999
        state['jobs']['first']['child_identity'] = queue.process_identity(child.pid)
        state_path.write_text(json.dumps(state))
        original = state_path.read_bytes()
        with pytest.raises(ValueError, match='surviving child'):
            queue.recover_jobs(path, state_path)
        assert child.poll() is None and state_path.read_bytes() == original
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_recovery_refuses_unknown_launch_for_current_owner(tmp_path):
    path, state_path, _ = recovery_fixture(tmp_path, child_pid=None)
    original = state_path.read_bytes()
    with pytest.raises(ValueError, match='no recorded PID'):
        queue.recover_jobs(path, state_path)
    assert state_path.read_bytes() == original


def test_recovery_requires_existing_journal_without_creating_anything(tmp_path):
    with pytest.raises(ValueError, match='existing saved journal'):
        queue.recover_jobs(tmp_path / 'jobs.json', tmp_path / 'state.json')
    assert not list(tmp_path.iterdir())


def test_recovery_summary_without_running_jobs_preserves_journal(tmp_path, monkeypatch):
    path = tmp_path / 'jobs.json'
    write_jobs(path, [job('first')])
    state_path = tmp_path / 'state.json'
    with queue.queue_state(state_path, queue.prepare_jobs(path)):
        pass
    original = state_path.read_bytes()
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('queried GPU inventory'))
    assert queue.recover_jobs(path, state_path) == {'recovered': False, 'jobs': [{'id': 'first', 'state': 'pending'}]}
    assert state_path.read_bytes() == original


def test_recovery_corrupt_evidence_preserves_journal_and_outputs(tmp_path):
    path, state_path, _ = recovery_fixture(tmp_path, publish=True)
    encoded = tmp_path / 'first/generated.pt'
    torch.save(torch.ones(1, 2, 7, 2, 2), encoded)
    original_state, original_output = state_path.read_bytes(), encoded.read_bytes()
    with pytest.raises(ValueError, match='encoding content changed'):
        queue.recover_jobs(path, state_path)
    assert state_path.read_bytes() == original_state and encoded.read_bytes() == original_output
