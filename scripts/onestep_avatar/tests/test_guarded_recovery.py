"""Real launch owners die at specific boundaries; recovery must never grant work."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.onestep_avatar.execution import queue, queue_launch

OWNER = '''import os,sys,subprocess
from pathlib import Path
from scripts.onestep_avatar.execution import queue, queue_launch
root=Path(sys.argv[1]); phase=sys.argv[2]
command=[sys.executable,'-c',"from pathlib import Path; Path("+repr(str(root/'model_executed'))+").write_text('forbidden')"]
job={'id':'case','sha256':'a'*64,'kind':'evaluate','dependencies':[],
     'arguments':['--mode','causal'],'output':str(root/'output')}
queue.job_command=lambda *_: (command,{})
claims=queue.GPUClaims(root/'claims'); assert claims.acquire((4,),job='case')
original_popen=subprocess.Popen
original_wait=queue_launch.wait_registration
def start(*args,**kwargs):
    if phase=='before_fork': os._exit(17)
    child=original_popen(*args,**kwargs)
    (root/'actual_child_pid').write_text(str(child.pid))
    return child
def wait(*args,**kwargs):
    if phase in ('after_bootstrap','paused_guard','interrupted_registration'): original_wait(*args,**kwargs)
    if phase=='paused_guard':
        import signal
        os.kill(args[1].pid,signal.SIGSTOP)
    if phase=='interrupted_registration': raise KeyboardInterrupt
    os._exit(17)
subprocess.Popen=start
queue_launch.wait_registration=wait
try:
    queue.run_child(job,claims,root/'child.log',state_path=root/'state.json',jobs=[job])
except KeyboardInterrupt:
    os._exit(17)
'''


def interrupted_owner(root: Path, phase: str) -> tuple[dict, dict]:
    child = subprocess.Popen([sys.executable, '-c', OWNER, str(root), phase])
    assert child.wait(timeout=10) == 17
    state = json.loads((root/'state.json').read_text())
    job = {'id':'case','sha256':'a'*64,'kind':'evaluate','dependencies':[],
           'arguments':['--mode','causal'],'output':str(root/'output')}
    return job, state


def wait_guard_exit(row: dict) -> None:
    path = Path(row['launch_request']).parent/'bootstrap.json'
    deadline = time.monotonic()+10
    while time.monotonic()<deadline:
        if path.exists():
            identity=json.loads(path.read_text())['identity']
            current=queue.process_identity(identity['pid'])
            if current is None or current['terminal']:
                return
        time.sleep(.01)
    pytest.fail('unapproved guard did not stop after its owner exited')


@pytest.mark.parametrize('phase', ['before_fork','before_bootstrap','after_bootstrap','interrupted_registration'])
def test_explicit_recovery_of_actual_unapproved_launch_preserves_outputs_and_attempts(tmp_path, monkeypatch, phase):
    job,state=interrupted_owner(tmp_path,phase)
    row=state['jobs']['case']
    assert (row['child_pid'] is None)==(phase!='interrupted_registration')
    assert row['child_identity'] is None and row['state']=='running'
    assert not (Path(row['launch_request']).parent/'grant.json').exists()
    if phase!='before_fork': wait_guard_exit(row)
    assert queue.inspect_child(row)=='unapproved'
    output=Path(job['output']); output.mkdir(); (output/'old.bin').write_bytes(b'old evidence')
    original=(tmp_path/'state.json').read_bytes()
    monkeypatch.setattr(queue,'verify_completion',lambda *_: pytest.fail('adopted unapproved old output'))
    with pytest.raises(ValueError,match='require explicit recovery'):
        with queue.queue_state(tmp_path/'state.json',[job]): pytest.fail('automatic takeover')
    assert (tmp_path/'state.json').read_bytes()==original
    with queue.queue_state(tmp_path/'state.json',[job],recover=True) as recovered:
        result=recovered['jobs']['case']
        assert result['state']=='failed' and 'before launch approval' in result['error']
        assert result['launch_recovery']['model_execution_authorized'] is False
        assert result['launch_recovery']['prior_attempt']==row['attempts'][-1]
        assert result['attempts'][-1]['launch_recovery']==result['launch_recovery']
    assert (output/'old.bin').read_bytes()==b'old evidence'
    assert len(result['attempts'])==1
    assert (tmp_path/'claims/4').is_file()  # Recovery never releases reservations.
    assert not (tmp_path/'model_executed').exists()
    assert not (Path(row['launch_request']).parent/'grant.json').exists()


@pytest.mark.parametrize('changed', ['protocol','request','grant','attempt','environment','ticks','job','journal','grant_symlink','bootstrap_symlink','attempt_pid'])
def test_inconsistent_unapproved_launch_refuses_without_mutation(tmp_path, changed):
    job,state=interrupted_owner(tmp_path,'before_fork')
    row=state['jobs']['case']; request=Path(row['launch_request'])
    if changed=='protocol': row['launch_protocol']='old'
    if changed=='request': request.write_text('{}')
    if changed=='grant': (request.parent/'grant.json').write_text('{}')
    if changed=='attempt': row['attempts']=[]
    if changed=='environment': row['environment_changes']={}
    if changed=='ticks': row['attempt_started_ticks']=False
    if changed=='job': row['command']=['other']
    if changed=='journal': state['owner_pid']=123456789
    if changed=='grant_symlink': (request.parent/'grant.json').symlink_to(tmp_path/'missing')
    if changed=='bootstrap_symlink': (request.parent/'bootstrap.json').symlink_to(tmp_path/'missing')
    if changed=='attempt_pid': row['attempts'][-1]['child_pid']=123456789
    (tmp_path/'state.json').write_text(json.dumps(state))
    original=(tmp_path/'state.json').read_bytes()
    with pytest.raises(ValueError):
        with queue.queue_state(tmp_path/'state.json',[job],recover=True): pytest.fail('recovered changed evidence')
    assert (tmp_path/'state.json').read_bytes()==original
    assert not (tmp_path/'model_executed').exists()


def test_live_unapproved_guard_refuses_recovery_until_its_recorded_handle_exits(tmp_path):
    job,state=interrupted_owner(tmp_path,'paused_guard')
    row=state['jobs']['case']; request=Path(row['launch_request'])
    identity=json.loads((request.parent/'bootstrap.json').read_text())['identity']
    original=(tmp_path/'state.json').read_bytes()
    try:
        with pytest.raises(ValueError,match='surviving child'):
            with queue.queue_state(tmp_path/'state.json',[job],recover=True): pytest.fail('adopted live guard')
        assert (tmp_path/'state.json').read_bytes()==original
        assert (tmp_path/'claims/4').exists()
        assert not (request.parent/'grant.json').exists()
    finally:
        current=queue.process_identity(identity['pid'])
        if current and current['start_ticks']==identity['start_ticks'] and not current['terminal']:
            os.kill(identity['pid'],signal.SIGCONT)
        wait_guard_exit(row)
    assert not (tmp_path/'model_executed').exists()
    with queue.queue_state(tmp_path/'state.json',[job],recover=True) as recovered:
        assert recovered['jobs']['case']['state']=='failed'


def test_unapproved_launch_refuses_surviving_token_worker_in_another_session(tmp_path):
    job,state=interrupted_owner(tmp_path,'before_fork')
    row=state['jobs']['case']
    worker=subprocess.Popen([sys.executable,'-c','import time; print("ready",flush=True); time.sleep(60)'],
                            env={**os.environ,**row['environment_changes']},start_new_session=True,
                            stdout=subprocess.PIPE,text=True)
    assert worker.stdout.readline().strip()=='ready'
    identity=queue.process_identity(worker.pid)
    original=(tmp_path/'state.json').read_bytes()
    try:
        with pytest.raises(ValueError,match='surviving child or worker'):
            with queue.queue_state(tmp_path/'state.json',[job],recover=True): pytest.fail('recovered live worker')
        assert (tmp_path/'state.json').read_bytes()==original
        assert (tmp_path/'claims/4').is_file()
    finally:
        current=queue.process_identity(worker.pid)
        assert current['start_ticks']==identity['start_ticks']
        worker.terminate(); worker.wait(timeout=5)
    with queue.queue_state(tmp_path/'state.json',[job],recover=True) as recovered:
        assert recovered['jobs']['case']['state']=='failed'


@pytest.mark.parametrize('observation', ['permission','owner_live','owner_reused'])
def test_unknown_or_live_original_owner_preserves_state(tmp_path,monkeypatch,observation):
    job,state=interrupted_owner(tmp_path,'before_fork')
    row=state['jobs']['case']; request=queue_launch.load_request(Path(row['launch_request']))
    original=(tmp_path/'state.json').read_bytes()
    saved=request['owner_identity']
    actual=queue_launch.process_identity
    def observe(pid):
        if pid!=saved['pid']: return actual(pid)
        if observation=='permission': raise PermissionError('controlled owner observation denial')
        return {**saved,'start_ticks':saved['start_ticks']+(observation=='owner_reused')}
    monkeypatch.setattr(queue_launch,'process_identity',observe)
    with pytest.raises(PermissionError if observation=='permission' else ValueError):
        with queue.queue_state(tmp_path/'state.json',[job],recover=True): pytest.fail('adopted unknown/live owner')
    assert (tmp_path/'state.json').read_bytes()==original
