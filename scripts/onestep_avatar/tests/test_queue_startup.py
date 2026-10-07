"""Startup retries require terminal workers, exact events and preserved attempts."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import queue
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.training import startup


def event(name, token='b' * 32, job='a' * 64, rank=2):
    return startup.PREFIX + json.dumps({'schema_version': 1, 'event': name,
        'token': token, 'job_sha256': job, 'rank': rank, 'reason': 'cuda_oom'}) + '\n'


def failed_attempt(tmp_path):
    output = tmp_path / 'runs' / 'training'
    output.mkdir(parents=True)
    (output / 'config.json').write_text('original startup settings')
    (output / 'partial.bin').write_bytes(b'preserve every original byte')
    log = tmp_path / 'attempt.log'
    log.write_text(event('startup_contended') + 'torch.cuda.OutOfMemoryError: fixture\n')
    job = {'id': 'training', 'kind': 'train', 'sha256': 'a' * 64,
           'output': str(output), 'dependencies': [], 'arguments': ['--mode', 'causal']}
    attempt = {'state': 'failed', 'returncode': 1, 'child_pid': 99999999,
               'child_session': 99999999, 'child_identity': None, 'log': str(log),
               'log_sha256': sha256(log), 'environment_changes': {
                   startup.TOKEN_ENV: 'b' * 32, startup.JOB_ENV: job['sha256']}}
    attempt['attempt_started_ticks'] = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf('SC_CLK_TCK'))
    state_path = tmp_path / 'state.json'
    with queue.queue_state(state_path, [job]) as state:
        state['jobs'][job['id']].update(state='failed', error='QueueChildError: CUDA startup failure',
            returncode=1, child_pid=99999999, child_session=99999999, attempts=[attempt])
        state['jobs'][job['id']].update(environment_changes=attempt['environment_changes'],
            attempt_started_ticks=attempt['attempt_started_ticks'])
    return job, state_path, output, log


@pytest.mark.parametrize("reason", ["cuda_oom", "port_in_use"])
def test_retry_preserves_output_log_error_and_identity_before_pending(tmp_path, reason):
    job, state_path, output, log = failed_attempt(tmp_path)
    if reason == "port_in_use":
        log.write_text(log.read_text().replace("cuda_oom", "port_in_use"))
        s=json.loads(state_path.read_text());s["jobs"][job["id"]]["attempts"][0]["log_sha256"]=sha256(log)
        state_path.write_text(json.dumps(s))
    source_hashes = {p.name: sha256(p) for p in output.iterdir()}
    log_hash = sha256(log)
    assert queue.retry_startup_failure(job, [job], state_path)
    row = json.loads(state_path.read_text())['jobs'][job['id']]
    assert row['state'] == 'pending' and row['child_pid'] is None and row['child_session'] is None
    assert row['startup_retries'] == 1 and len(row['attempts']) == 1
    old = row['attempts'][0]
    assert old['state'] == 'failed' and old['returncode'] == 1
    archive = Path(old['archive'])
    assert not output.exists()
    assert source_hashes == {p.name: sha256(p) for p in (archive / 'output').iterdir()}
    assert sha256(archive / 'child.log') == log_hash == sha256(log)
    assert (archive / 'README.md').is_file()
    preserved = json.loads((archive / 'preservation.json').read_text())
    assert preserved['job_sha256'] == job['sha256'] and preserved['files'] == source_hashes
    assert preserved['error'] == row['error'] and preserved['contended_ranks'] == [2]


@pytest.mark.parametrize('corruption', [
    'evaluation', 'no_event', 'wrong_token', 'wrong_job', 'wrong_rank', 'boolean_rank',
    'wrong_schema', 'extra_field', 'bad_json', 'prefixed_line', 'updates_begin',
    'other_rank_validation', 'metrics', 'zero_exit', 'no_session', 'changed_log', 'retry_limit',
])
def test_ambiguous_or_nonstartup_failure_never_mutates_state_or_output(tmp_path, corruption):
    job, state_path, output, log = failed_attempt(tmp_path)
    s = json.loads(state_path.read_text()); row = s['jobs'][job['id']]; attempt = row['attempts'][0]
    changed = None
    if corruption == 'evaluation': job['kind'] = 'evaluate'
    elif corruption == 'no_event': changed = 'CUDA out of memory: an untrusted traceback\n'
    elif corruption == 'wrong_token': changed = event('startup_contended', token='c' * 32)
    elif corruption == 'wrong_job': changed = event('startup_contended', job='c' * 64)
    elif corruption == 'wrong_rank': changed = event('startup_contended', rank=4)
    elif corruption == 'boolean_rank': changed = event('startup_contended', rank=True)
    elif corruption == 'wrong_schema': changed = event('startup_contended').replace('"schema_version": 1', '"schema_version": true')
    elif corruption == 'extra_field': changed = event('startup_contended').replace('"rank": 2', '"rank": 2, "extra": 0')
    elif corruption == 'bad_json': changed = startup.PREFIX + 'invalid JSON\n'
    elif corruption == 'prefixed_line': changed = '[rank2] ' + event('startup_contended')
    elif corruption == 'updates_begin': changed = event('startup_contended') + event('updates_begin', rank=0)
    elif corruption == 'other_rank_validation': changed = event('startup_contended') + event('startup_failed', rank=0)
    elif corruption == 'metrics': (output / 'metrics_rank0.jsonl').write_text('{"step":1}\n')
    elif corruption == 'zero_exit': attempt['returncode'] = 0
    elif corruption == 'no_session': attempt['child_session'] = None
    elif corruption == 'changed_log': log.write_text('changed after terminal receipt')
    elif corruption == 'retry_limit': row['attempts'] = [dict(attempt) for _ in range(4)]
    if changed is not None:
        log.write_text(changed); attempt['log_sha256'] = sha256(log)
    state_path.write_text(json.dumps(s))
    original = state_path.read_bytes(); inputs = {p.name: sha256(p) for p in output.iterdir()}
    assert not queue.retry_startup_failure(job, [job], state_path)
    assert original == state_path.read_bytes()
    assert inputs == {p.name: sha256(p) for p in output.iterdir()}
    assert not (output.parent / 'superseded_startup_contention').exists()


def test_archive_publication_failure_restores_output_and_preserves_journal(tmp_path, monkeypatch):
    from scripts.onestep_avatar import dataset
    job, state_path, output, log = failed_attempt(tmp_path)
    original = state_path.read_bytes(); hashes = {p.name: sha256(p) for p in output.iterdir()}
    def unavailable(*_args, **_kwargs): raise OSError('state disk failed')
    monkeypatch.setattr(dataset, 'atomic_write', unavailable)
    with pytest.raises(OSError, match='state disk failed'):
        queue.retry_startup_failure(job, [job], state_path)
    assert state_path.read_bytes() == original
    assert hashes == {p.name: sha256(p) for p in output.iterdir()}
    assert log.is_file()


@pytest.mark.parametrize('after_updates', [False, True])
@pytest.mark.parametrize('kind', ['oom', 'validation'])
def test_typed_runtime_reports_only_oom_before_any_training(monkeypatch, capsys, after_updates, kind):
    from scripts.onestep_avatar.training import engine
    monkeypatch.setenv(startup.TOKEN_ENV, 'b' * 32)
    monkeypatch.setenv(startup.JOB_ENV, 'a' * 64)
    monkeypatch.setenv('RANK', '2')
    def fail(_settings, events):
        if after_updates: events.begin_updates()
        raise torch.cuda.OutOfMemoryError('CUDA contention') if kind == 'oom' else ValueError('invalid adapter')
    monkeypatch.setattr(engine, '_run_settings', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError if kind == 'oom' else ValueError):
        engine.run_settings(None)
    records = [json.loads(line[len(startup.PREFIX):]) for line in capsys.readouterr().out.splitlines()]
    assert [r['event'] for r in records] == (
        ['updates_begin', 'startup_failed'] if after_updates else
        ['startup_contended' if kind == 'oom' else 'startup_failed'])
    assert all(r['rank'] == 2 and r['job_sha256'] == 'a' * 64 and r['token'] == 'b' * 32 for r in records)


def test_nonqueued_startup_failure_has_no_protocol(monkeypatch, capsys):
    monkeypatch.delenv(startup.TOKEN_ENV, raising=False); monkeypatch.delenv(startup.JOB_ENV, raising=False)
    with pytest.raises(torch.cuda.OutOfMemoryError), startup.StartupEvents():
        raise torch.cuda.OutOfMemoryError('unrelated direct run')
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('success_after', [None, 2])
def test_loop_retries_at_most_three_times_with_new_claims_and_preserved_attempts(tmp_path, monkeypatch, success_after, controlled_queue_launch):
    job = {'id': 'training', 'kind': 'train', 'sha256': 'a' * 64, 'dependencies': [],
           'output': str(tmp_path / 'runs/training'), 'arguments': ['--mode', 'causal']}
    state_path, claims = tmp_path / 'state.json', tmp_path / 'claims'
    claims.mkdir(); (claims / '5').write_text('legacy foreign claim')
    monkeypatch.setattr(queue, 'prepare_jobs', lambda *_: [job])
    monkeypatch.setattr(queue, 'job_command', lambda *_: (['controlled child'], {'CUDA_VISIBLE_DEVICES': '0,1,2,3'}))
    monkeypatch.setattr(subprocess, 'run', lambda *_a, **_k: SimpleNamespace(stdout='\n'.join(f'{g}, 0' for g in range(8))))
    launches = []
    def launch(command, **kwargs):
        assert kwargs['start_new_session'] and kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
        token = kwargs['env'][startup.TOKEN_ENV]
        assert token not in launches; launches.append(token)
        output = Path(job['output']); output.mkdir(parents=True)
        (output / 'partial.bin').write_bytes(f'original attempt {len(launches)}'.encode())
        code = 0 if success_after == len(launches) else 1
        kwargs['stdout'].write(event('startup_contended', token=token)); kwargs['stdout'].flush()
        return SimpleNamespace(pid=99999999, poll=lambda: code, wait=lambda: code)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    monkeypatch.setattr(queue, 'completion_receipt', lambda *_: {'job_sha256': job['sha256'], 'evidence': []})
    monkeypatch.setattr(queue, 'verify_receipt', lambda *_: True)
    if success_after is None:
        with pytest.raises(queue.QueueChildError, match='failed with exit 1'):
            queue.execute_jobs(tmp_path / 'jobs.json', state_path, claims, once=False)
        assert len(launches) == 4
    else:
        assert queue.execute_jobs(tmp_path / 'jobs.json', state_path, claims, once=False) == 0
        assert len(launches) == success_after
    row = json.loads(state_path.read_text())['jobs']['training']
    assert len(row['attempts']) == len(launches)
    assert row['state'] == ('failed' if success_after is None else 'complete')
    for index, old in enumerate(row['attempts'][:-1], 1):
        archive = Path(old['archive'])
        assert (archive / 'output/partial.bin').read_bytes() == f'original attempt {index}'.encode()
        assert old['error'].startswith('QueueChildError:') and old['returncode'] == 1
    assert sorted(p.name for p in claims.iterdir() if p.name.isdigit()) == ['5']
    assert (claims / '5').read_text() == 'legacy foreign claim'


@pytest.mark.parametrize('new_session', [False, True])
def test_surviving_worker_keeps_claim_and_refuses_adoption_after_leader_exit(tmp_path, monkeypatch, new_session):
    pid_file = tmp_path / 'worker.pid'
    code = (
        'import subprocess,sys; from pathlib import Path; '
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],'
        f'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session={new_session!r}); '
        f'Path({str(pid_file)!r}).write_text(str(p.pid))'
    )
    job = {'id': 'owned', 'kind': 'evaluate', 'sha256': 'a' * 64,
           'output': str(tmp_path / 'output'), 'arguments': ['--mode', 'causal'], 'dependencies': []}
    claims = queue.GPUClaims(tmp_path / 'claims')
    assert claims.acquire((4,), job='owned')
    monkeypatch.setattr(queue, 'job_command', lambda *_: ([sys.executable, '-c', code], {}))
    state_path = tmp_path / 'state.json'
    worker = None
    try:
        with pytest.raises(ValueError, match='still has live workers'):
            queue.run_child(job, claims, tmp_path / 'child.log', state_path=state_path, jobs=[job], poll_seconds=0.01)
        worker = int(pid_file.read_text())
        row = json.loads(state_path.read_text())['jobs']['owned']
        assert row['returncode'] == 0 and row['state'] == 'running'
        assert queue.inspect_child(row) == 'live'
        assert queue.live_session_processes(row['child_session']) == ([] if new_session else [worker])
        assert queue.live_attempt_processes(row['environment_changes'][startup.TOKEN_ENV],
            started_ticks=row['attempt_started_ticks']) == [worker]
        assert claims.owned == {4} and (tmp_path / 'claims/4').is_file()
        original = state_path.read_bytes()
        with pytest.raises(ValueError, match='surviving child'):
            with queue.queue_state(state_path, [job], recover=True):
                pytest.fail('adopted surviving worker')
        assert state_path.read_bytes() == original
        with pytest.raises(ValueError, match='live child'):
            claims.release()
        # A dead launcher/owner cannot cause this session's reservation to expire.
        monkeypatch.setattr(queue, 'process_alive', lambda *_: False)
        os.utime(tmp_path / 'claims/4', (time.time() - 1000, time.time() - 1000))
        memory = dict.fromkeys(range(8), 2048); memory[4] = 0
        assert queue.GPUClaims(tmp_path / 'claims').choose(memory, training=False) is None
    finally:
        if worker is None and pid_file.exists(): worker = int(pid_file.read_text())
        if worker is not None:
            import signal
            try: os.kill(worker, signal.SIGTERM)
            except ProcessLookupError: pass
            for _ in range(100):
                if not queue.owned_workers_live(row): break
                time.sleep(0.01)
        if claims.owned: claims.release()


def test_session_scan_distinguishes_live_terminal_and_unknown_handles(tmp_path):
    def stat(pid, state, session_id):
        p = tmp_path / str(pid); p.mkdir()
        fields = [state, '0', '0', str(session_id), *(['0'] * 15), '123']
        (p / 'stat').write_text(f'{pid} (worker (name)) ' + ' '.join(fields))
    stat(10, 'S', 42); stat(11, 'Z', 42); stat(12, 'S', 99)
    assert queue.live_session_processes(42, proc_root=tmp_path) == [10]
    assert queue.live_session_processes(100, proc_root=tmp_path) == []
    (tmp_path / '12/stat').write_text('bad stat')
    with pytest.raises(ValueError, match='malformed'):
        queue.live_session_processes(42, proc_root=tmp_path)


def test_once_records_retry_pending_without_a_second_launch(tmp_path, monkeypatch):
    job, state_path, output, log = failed_attempt(tmp_path)
    # Reuse the real preservation path after a mocked single failed dispatch.
    monkeypatch.setattr(queue, 'prepare_jobs', lambda *_: [job])
    seen = []
    def dispatch(*_args):
        seen.append(1)
        assert queue.retry_startup_failure(job, [job], state_path)
        return True
    monkeypatch.setattr(queue, 'dispatch_ready', dispatch)
    assert queue.execute_jobs(tmp_path / 'jobs.json', state_path, tmp_path / 'claims', once=True) == 2
    assert seen == [1] and json.loads(state_path.read_text())['jobs']['training']['state'] == 'pending'


@pytest.mark.parametrize('after_updates', [False, True])
@pytest.mark.parametrize('number', [98, 13])
def test_typed_port_contention_is_distinct_from_other_os_errors(monkeypatch, capsys, after_updates, number):
    import errno
    monkeypatch.setenv(startup.TOKEN_ENV, 'b' * 32); monkeypatch.setenv(startup.JOB_ENV, 'a' * 64)
    monkeypatch.setenv('RANK', '0')
    with pytest.raises(OSError), startup.StartupEvents() as reporter:
        if after_updates: reporter.begin_updates()
        raise OSError(errno.EADDRINUSE if number == 98 else errno.EACCES, 'fixture error')
    events = [json.loads(line[len(startup.PREFIX):]) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['event'] == ('startup_contended' if number == 98 and not after_updates else 'startup_failed')
    assert events[-1]['reason'] == ('port_in_use' if number == 98 and not after_updates else None)


def test_claim_free_wait_after_preserved_retry_creates_no_additional_attempt(tmp_path, monkeypatch):
    job, state_path, output, log = failed_attempt(tmp_path)
    assert queue.retry_startup_failure(job, [job], state_path)
    original = state_path.read_bytes()
    monkeypatch.setattr(queue, 'prepare_jobs', lambda *_: [job])
    monkeypatch.setattr(subprocess, 'run', lambda *_a, **_k: SimpleNamespace(stdout='\n'.join(f'{g}, 2048' for g in range(8))))
    monkeypatch.setattr(queue, 'run_child', lambda *_a, **_k: pytest.fail('launched on busy device'))
    assert queue.execute_jobs(tmp_path / 'jobs.json', state_path, tmp_path / 'claims', once=True) == 2
    assert state_path.read_bytes() == original and not output.exists()
    assert not any(p.name.isdigit() for p in (tmp_path / 'claims').iterdir())


def test_attempt_worker_scan_requires_exact_token_and_skips_older_handles(tmp_path):
    def create(pid, tick, environment):
        p = tmp_path / str(pid); p.mkdir()
        (p / 'stat').write_text(f'{pid} (worker) ' + ' '.join(['S', '0', '0', '999', *(['0'] * 15), str(tick)]))
        if environment is not None: (p / 'environ').write_bytes(environment)
    marker = (startup.TOKEN_ENV + '=' + 'b' * 32).encode()
    create(42, 200, marker + b'\0OTHER=x\0')
    create(43, 200, marker + b'x\0')
    create(44, 50, None)  # Older process may have an inaccessible environment.
    assert queue.live_attempt_processes('b' * 32, started_ticks=100, proc_root=tmp_path) == [42]


def test_attempt_worker_scan_refuses_unreadable_or_reused_handles(tmp_path, monkeypatch):
    p = tmp_path / '42'; p.mkdir()
    stat = p / 'stat'
    stat.write_text('42 (worker) ' + ' '.join(['S', '0', '0', '999', *(['0'] * 15), '200']))
    (p / 'environ').write_bytes((startup.TOKEN_ENV + '=' + 'b' * 32).encode())
    original = Path.read_bytes
    def unreadable(path):
        if path == p / 'environ': raise PermissionError('unknown eligible process')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', unreadable)
    with pytest.raises(PermissionError):
        queue.live_attempt_processes('b' * 32, started_ticks=100, proc_root=tmp_path)
    def reused(path):
        result = original(path)
        if path == p / 'environ': stat.write_text(stat.read_text().replace('200', '201'))
        return result
    monkeypatch.setattr(Path, 'read_bytes', reused)
    with pytest.raises(ValueError, match='handle changed'):
        queue.live_attempt_processes('b' * 32, started_ticks=100, proc_root=tmp_path)
