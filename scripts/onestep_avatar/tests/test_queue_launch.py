"""The real child remains inert until durable registration and launch approval."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.onestep_avatar import queue, queue_launch as launch
from scripts.onestep_avatar.queue_protocol import JOB_ENV, LAUNCH_PROTOCOL, TOKEN_ENV


def request_for(tmp_path, *, owner=None):
    executed = tmp_path / 'executed.json'
    command = [sys.executable, '-c',
        'import os,json; from pathlib import Path; '
        'from scripts.onestep_avatar.queue import process_identity; '
        f'Path({str(executed)!r}).write_text(json.dumps(process_identity(os.getpid())))']
    path = launch.prepare_request(tmp_path / 'launch', token='b' * 32, job_sha256='a' * 64,
        owner_identity=queue.process_identity(os.getpid()) if owner is None else owner,
        job_id='case', journal=tmp_path / 'journal.json', command=command)
    return path, executed


def start_guard(path, log, *, timeout=5, environment=None):
    return subprocess.Popen([sys.executable, '-m', 'scripts.onestep_avatar.queue_launch',
        '--request', str(path), '--timeout', str(timeout), '--poll-seconds', '0.01'],
        env={**os.environ, TOKEN_ENV:'b' * 32, JOB_ENV:'a' * 64, **(environment or {})},
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_bootstrap(path, child):
    deadline = time.monotonic() + 5
    while not (path.parent / 'bootstrap.json').is_file():
        if child.poll() is not None: pytest.fail('guard exited before registration')
        if time.monotonic() >= deadline: pytest.fail('guard did not register')
        time.sleep(0.01)
    return json.loads((path.parent / 'bootstrap.json').read_text())


def register(path, bootstrap):
    request = launch.load_request(path)
    row = {'state':'running', 'sha256': request['job_sha256'], 'command': request['command'],
        'launch_protocol': LAUNCH_PROTOCOL, 'launch_request': str(path.resolve()),
        'launch_request_sha256': bootstrap['request_sha256'], 'child_pid': bootstrap['identity']['pid'],
        'child_identity': bootstrap['identity'],
        'environment_changes': {TOKEN_ENV:request['token'], JOB_ENV:request['job_sha256']}}
    state = {'schema_version':1, 'owner_pid':request['owner_identity']['pid'], 'jobs':{'case':row}}
    Path(request['journal']).write_text(json.dumps(state))
    return state


def stop(child):
    if child.poll() is None: child.terminate()
    child.wait(timeout=5)


@pytest.mark.parametrize('approval', [None, 'valid', 'bad_grant', 'bad_journal', 'bad_request'])
def test_grant_published_during_owner_inspection_keeps_all_approval_gates(tmp_path, monkeypatch, approval):
    path, executed = request_for(tmp_path)
    original_identity = launch.process_identity
    injected, commands = [], []
    monkeypatch.setenv(TOKEN_ENV, 'b'*32)
    monkeypatch.setenv(JOB_ENV, 'a'*64)

    def inspect(pid):
        bootstrap_path = path.parent/'bootstrap.json'
        if not injected and bootstrap_path.exists():
            injected.append(True)
            if approval:
                bootstrap = json.loads(bootstrap_path.read_text())
                register(path, bootstrap)
                # The publisher still proves a live original owner and child.
                launch.publish_grant(path)
                changed = {'bad_grant': path.parent/'grant.json',
                           'bad_journal': tmp_path/'journal.json', 'bad_request': path}.get(approval)
                if changed:
                    value = json.loads(changed.read_text())
                    if approval == 'bad_grant': value['token'] = 'c'*32
                    elif approval == 'bad_journal': value['jobs']['case']['command'] = ['other']
                    else: value['command'] = ['other']
                    changed.write_text(json.dumps(value))
            return None  # Observe owner exit after the loop's initial absent-grant check.
        return original_identity(pid)

    monkeypatch.setattr(launch, 'process_identity', inspect)
    monkeypatch.setattr(launch.os, 'execvpe', lambda executable, command, environment:
                        commands.append((executable, command, environment)))
    if approval == 'valid':
        launch.run_guard(path, timeout=1)
        request = launch.load_request(path)
        assert len(commands) == 1 and commands[0][:2] == (request['command'][0], request['command'])
        assert commands[0][2][JOB_ENV] == request['job_sha256']
    else:
        with pytest.raises(ValueError, match='before approval|grant or request changed|journal does not bind'):
            launch.run_guard(path, timeout=1)
        assert not commands
    assert injected == [True] and not executed.exists()


def test_exec_requires_approval_and_keeps_registered_pid_and_start_ticks(tmp_path):
    path, executed = request_for(tmp_path)
    with (tmp_path / 'child.log').open('w') as log:
        child = start_guard(path, log)
        try:
            bootstrap = wait_bootstrap(path, child)
            assert bootstrap['identity'] == queue.process_identity(child.pid)
            assert not executed.exists() and child.poll() is None
            register(path, bootstrap)
            assert not executed.exists()
            assert launch.publish_grant(path) == bootstrap
            assert child.wait(timeout=5) == 0
            actual = json.loads(executed.read_text())
            assert actual['pid'] == child.pid
            assert actual['start_ticks'] == bootstrap['identity']['start_ticks']
            assert actual['command'] == launch.load_request(path)['command']
        finally: stop(child)


@pytest.mark.parametrize('field', ['state','sha256','command','child_pid','child_identity',
    'launch_request','launch_request_sha256','environment_changes','launch_protocol'])
def test_unbound_journal_cannot_publish_grant_or_execute(tmp_path, field):
    path, executed = request_for(tmp_path)
    with (tmp_path / 'child.log').open('w') as log:
        child = start_guard(path, log)
        try:
            bootstrap = wait_bootstrap(path, child); state = register(path, bootstrap)
            state['jobs']['case'][field] = None
            Path(launch.load_request(path)['journal']).write_text(json.dumps(state))
            with pytest.raises((ValueError, AttributeError), match='launch journal|has no attribute'):
                launch.publish_grant(path)
            assert not (path.parent / 'grant.json').exists() and not executed.exists()
            assert child.poll() is None
        finally: stop(child)


def test_missing_pid_registration_and_missing_journal_never_authorize_execution(tmp_path):
    path, executed = request_for(tmp_path)
    with (tmp_path / 'child.log').open('w') as log:
        child = start_guard(path, log)
        try:
            bootstrap = wait_bootstrap(path, child)
            with pytest.raises(FileNotFoundError): launch.publish_grant(path)
            state = register(path, bootstrap); state['jobs']['case']['child_pid'] = None
            Path(launch.load_request(path)['journal']).write_text(json.dumps(state))
            with pytest.raises(ValueError): launch.publish_grant(path)
            assert not executed.exists()
        finally: stop(child)


def test_owner_exit_before_approval_starts_no_consumer(tmp_path):
    owner = subprocess.Popen([sys.executable,'-c','import time; print("ready",flush=True); time.sleep(30)'],
        stdout=subprocess.PIPE, text=True)
    child = None
    try:
        assert owner.stdout.readline().strip() == 'ready'
        path, executed = request_for(tmp_path, owner=queue.process_identity(owner.pid))
        with (tmp_path / 'child.log').open('w') as log:
            child = start_guard(path, log)
            wait_bootstrap(path, child)
            stop(owner)
            assert child.wait(timeout=5) != 0
            assert not executed.exists() and not (path.parent / 'grant.json').exists()
    finally:
        stop(owner)
        if child is not None: stop(child)


def test_timeout_is_a_failure_not_permission_to_execute(tmp_path):
    path, executed = request_for(tmp_path)
    with (tmp_path / 'child.log').open('w') as log:
        child = start_guard(path, log, timeout=0.1)
        try:
            wait_bootstrap(path, child)
            assert child.wait(timeout=5) != 0
            assert not executed.exists()
        finally: stop(child)
    assert 'approval deadline expired' in (tmp_path / 'child.log').read_text()


def test_changed_request_while_waiting_refuses_execution(tmp_path):
    path, executed = request_for(tmp_path)
    with (tmp_path / 'child.log').open('w') as log:
        child = start_guard(path, log)
        try:
            wait_bootstrap(path, child)
            path.write_text(path.read_text()+'\n')
            assert child.wait(timeout=5) != 0
            assert not executed.exists()
        finally: stop(child)


def test_wrong_environment_and_preexisting_grant_refuse_before_bootstrap(tmp_path):
    path, executed = request_for(tmp_path)
    with (tmp_path / 'child.log').open('w') as log:
        child = start_guard(path, log, environment={TOKEN_ENV:'c'*32})
        assert child.wait(timeout=5) != 0
        assert not (path.parent / 'bootstrap.json').exists()
        (path.parent / 'grant.json').write_text('{}')
        child = start_guard(path, log)
        assert child.wait(timeout=5) != 0
    assert not executed.exists() and not (path.parent / 'bootstrap.json').exists()


def test_record_publication_is_exclusive_and_invalid_input_creates_no_directory(tmp_path):
    path, _ = request_for(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError): launch.publish_exclusive(path, {'replacement': True})
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        launch.prepare_request(tmp_path/'invalid',token='bad',job_sha256='a'*64,
            owner_identity=queue.process_identity(os.getpid()),job_id='case',journal=tmp_path/'j',command=['python'])
    assert not (tmp_path/'invalid').exists()


def test_owner_may_exit_after_durable_approval_without_revoking_registered_child(tmp_path):
    executed = tmp_path / 'executed.json'
    script = '''
import json,os,subprocess,sys,time
from pathlib import Path
from scripts.onestep_avatar import queue,queue_launch as launch
from scripts.onestep_avatar.queue_protocol import TOKEN_ENV,JOB_ENV,LAUNCH_PROTOCOL
root=Path(sys.argv[1])
command=[sys.executable,'-c',"import os,json; from pathlib import Path; from scripts.onestep_avatar.queue import process_identity; Path("+repr(str(root/'executed.json'))+").write_text(json.dumps(process_identity(os.getpid())))"]
path=launch.prepare_request(root/'launch',token='b'*32,job_sha256='a'*64,owner_identity=queue.process_identity(os.getpid()),job_id='case',journal=root/'journal.json',command=command)
child=subprocess.Popen([sys.executable,'-m','scripts.onestep_avatar.queue_launch','--request',str(path),'--timeout','5'],env={**os.environ,TOKEN_ENV:'b'*32,JOB_ENV:'a'*64},start_new_session=True)
deadline=time.monotonic()+5
while not (path.parent/'bootstrap.json').exists():
 if time.monotonic()>deadline: raise RuntimeError('missing bootstrap')
 time.sleep(.01)
bootstrap=json.loads((path.parent/'bootstrap.json').read_text())
request=launch.load_request(path)
row={'state':'running','sha256':request['job_sha256'],'command':command,'launch_protocol':LAUNCH_PROTOCOL,'launch_request':str(path.resolve()),'launch_request_sha256':bootstrap['request_sha256'],'child_pid':child.pid,'child_identity':bootstrap['identity'],'environment_changes':{TOKEN_ENV:'b'*32,JOB_ENV:'a'*64}}
(root/'journal.json').write_text(json.dumps({'schema_version':1,'owner_pid':os.getpid(),'jobs':{'case':row}}))
launch.publish_grant(path)
os._exit(0)
'''
    with (tmp_path/'owner.log').open('w') as log:
        owner = subprocess.Popen([sys.executable,'-c',script,str(tmp_path)],stdout=log,stderr=subprocess.STDOUT)
        try:
            assert owner.wait(timeout=5) == 0
            bootstrap=json.loads((tmp_path/'launch/bootstrap.json').read_text())
            deadline = time.monotonic()+5
            while True:
                handle=queue.process_identity(bootstrap['identity']['pid'])
                if handle is None or handle['terminal']: break
                assert handle['start_ticks']==bootstrap['identity']['start_ticks']
                if time.monotonic()>=deadline: pytest.fail('approved child never finished')
                time.sleep(.01)
            actual=json.loads(executed.read_text())
            assert actual['pid']==bootstrap['identity']['pid']
            assert actual['start_ticks']==bootstrap['identity']['start_ticks']
        finally:
            stop(owner)
            bootstrap_path=tmp_path/'launch/bootstrap.json'
            if bootstrap_path.exists():
                saved=json.loads(bootstrap_path.read_text())['identity']
                current=queue.process_identity(saved['pid'])
                if current and not current['terminal'] and current['start_ticks']==saved['start_ticks']:
                    os.kill(saved['pid'],15)


@pytest.mark.parametrize('corruption',['schema_bool','unbound_journal'])
def test_child_rechecks_grant_and_journal_instead_of_trusting_publication(tmp_path,corruption):
    path,executed=request_for(tmp_path)
    with (tmp_path/'child.log').open('w') as log:
        child=start_guard(path,log)
        try:
            bootstrap=wait_bootstrap(path,child);state=register(path,bootstrap)
            approved=dict(bootstrap)
            if corruption=='schema_bool':
                approved['schema_version']=True
            else:
                state['jobs']['case']['child_pid']=None
                Path(launch.load_request(path)['journal']).write_text(json.dumps(state))
            # Bypass the parent validator deliberately; the real child must refuse.
            launch.publish_exclusive(path.parent/'grant.json',approved)
            assert child.wait(timeout=5)!=0
            assert not executed.exists()
        finally:stop(child)


@pytest.mark.parametrize('changed', [None, 'grant', 'bootstrap', 'request', 'command', 'ticks', 'protocol'])
def test_real_dispatch_registers_before_grant_and_observes_approved_transition(tmp_path, monkeypatch, changed):
    executed, release = tmp_path/'executed.json', tmp_path/'release'
    command = [sys.executable, '-c',
        'import os,json,time; from pathlib import Path; '
        'from scripts.onestep_avatar.queue import process_identity; '
        f'Path({str(executed)!r}).write_text(json.dumps(process_identity(os.getpid()))); '
        f'release=Path({str(release)!r}); '
        '\nwhile not release.exists(): time.sleep(.01)']
    job = {'id':'case','kind':'evaluate','sha256':'a'*64,'dependencies':[],
           'arguments':['--mode','causal'],'output':str(tmp_path/'output')}
    monkeypatch.setattr(queue,'job_command',lambda *_: (command,{}))
    monkeypatch.setattr(queue,'completion_receipt',lambda *_: {'job_sha256':job['sha256'],'evidence':[]})
    claims=queue.GPUClaims(tmp_path/'claims')
    assert claims.acquire((4,),job='case')
    state_path=tmp_path/'state.json'
    original=launch.publish_grant
    observed=[]
    def approve(path):
        state=json.loads(state_path.read_text()); row=state['jobs']['case']
        assert row['child_pid']==row['child_identity']['pid']
        assert row['child_identity']['command']==launch.guard_command(path)
        assert row['attempts'][-1]['child_identity']==row['child_identity']
        assert not executed.exists()
        result=original(path)
        deadline=time.monotonic()+5
        while not executed.exists():
            if time.monotonic()>=deadline: pytest.fail('approved consumer did not execute')
            time.sleep(.01)
        assert queue.inspect_child(row)=='live'
        actual=json.loads(executed.read_text())
        assert actual['command']==command and actual['pid']==row['child_pid']
        assert actual['start_ticks']==row['child_identity']['start_ticks']
        if changed in ('grant', 'bootstrap', 'request'):
            changed_path = path if changed == 'request' else path.parent / (changed + '.json')
            original_bytes = changed_path.read_bytes()
            changed_record = json.loads(original_bytes)
            changed_record['token'] = 'c' * 32
            try:
                changed_path.write_text(json.dumps(changed_record))
                with pytest.raises(ValueError): queue.inspect_child(row)
            finally:
                changed_path.write_bytes(original_bytes)
            assert queue.inspect_child(row) == 'live'
        elif changed:
            invalid = json.loads(json.dumps(row))
            if changed == 'command': invalid['command'] = ['other']
            if changed == 'ticks': invalid['child_identity']['start_ticks'] += 1
            if changed == 'protocol': invalid['launch_protocol'] = 'unrecognized'
            with pytest.raises(ValueError): queue.inspect_child(invalid)
        observed.append(row)
        release.write_text('exit')
        return result
    monkeypatch.setattr(launch,'publish_grant',approve)
    try:
        result=queue.run_child(job,claims,tmp_path/'child.log',state_path=state_path,jobs=[job],poll_seconds=.01)
        row=json.loads(state_path.read_text())['jobs']['case']
        assert row['state']=='complete' and result['returncode']==0
        assert row['child_identity']==observed[0]['child_identity']
        assert row['launch_protocol']==LAUNCH_PROTOCOL and not claims.owned
        assert row['command']==command
    finally:
        release.write_text('exit')


def test_real_dispatch_registration_failure_never_grants_and_retains_live_child_claims(tmp_path, monkeypatch):
    path, executed=request_for(tmp_path)
    request=launch.load_request(path)
    job={'id':'case','kind':'evaluate','sha256':'a'*64,'dependencies':[],
         'arguments':['--mode','causal'],'output':str(tmp_path/'output')}
    monkeypatch.setattr(queue,'job_command',lambda *_: (request['command'],{}))
    claims=queue.GPUClaims(tmp_path/'claims'); assert claims.acquire((4,),job='case')
    original=launch.wait_registration
    child_handle=[]
    def fail(path,child,command,refresh,**kwargs):
        child_handle.append(child)
        original(path,child,command,refresh,**kwargs)
        raise ValueError('controlled registration interruption')
    monkeypatch.setattr(launch,'wait_registration',fail)
    state_path=tmp_path/'state.json'
    try:
        with pytest.raises(ValueError,match='controlled registration interruption'):
            queue.run_child(job,claims,tmp_path/'child.log',state_path=state_path,jobs=[job],poll_seconds=.01)
        row=json.loads(state_path.read_text())['jobs']['case']
        assert row['state']=='running' and row['child_pid']==child_handle[0].pid
        assert not (Path(row['launch_request']).parent/'grant.json').exists()
        assert not executed.exists() and claims.owned=={4}
        assert child_handle[0].poll() is None
    finally:
        for child in child_handle: stop(child)
        if claims.owned: claims.release()


@pytest.mark.parametrize('fail_sync', [False, True])
def test_journal_sync_precedes_publication_and_failure_preserves_original(tmp_path, monkeypatch, fail_sync):
    job={'id':'case','sha256':'a'*64}
    path=tmp_path/'state.json'
    with queue.queue_state(path,[job]): pass
    original=path.read_bytes()
    actual=os.fsync; observations=[]
    def sync(fd):
        target=os.readlink(f'/proc/self/fd/{fd}')
        observations.append(target)
        if fail_sync and target.endswith('.json'):
            raise OSError('controlled sync failure')
        actual(fd)
    monkeypatch.setattr(queue.os,'fsync',sync)
    if fail_sync:
        with pytest.raises(OSError,match='controlled sync failure'):
            with queue.queue_state(path,[job]) as state: state['jobs']['case']['error']='changed'
        assert path.read_bytes()==original
        assert len(observations)==1
    else:
        with queue.queue_state(path,[job]) as state: state['jobs']['case']['error']='changed'
        assert observations[0]==str(path.with_name(f'.state.tmp.{os.getpid()}.json'))
        assert observations[1]==str(tmp_path)
        assert json.loads(path.read_text())['jobs']['case']['error']=='changed'
