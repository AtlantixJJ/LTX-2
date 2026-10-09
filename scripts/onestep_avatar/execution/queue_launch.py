"""Register a waiting child before its recorded command can execute; see doc/execution/queue_launch.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.onestep_avatar.execution.queue import process_identity
from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, LAUNCH_PROTOCOL, TOKEN_ENV

if TYPE_CHECKING:
    import subprocess

REGISTERED_RECOVERY_MODULES = frozenset({
    'experiments.adapter_effect_check', 'bench', 'experiments.continuation_check', 'decode_saved', 'evaluate',
    'infer', 'media', 'prepare_inputs', 'experiments.stock_parity',
})


def guard_command(path: Path) -> list[str]:
    """Use the current conda interpreter; no model command runs at process creation."""
    return [sys.executable, '-m', 'scripts.onestep_avatar.execution.queue_launch', '--request', str(path.resolve())]


def wait_registration(path: Path, child: subprocess.Popen, command: list[str], refresh: Callable[[], None],
                      *, timeout: float = 45) -> dict:
    """Observe the exact waiting child before the dispatcher saves its identity."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('registration timeout must be finite and positive')
    request, request_hash = read_request(path)
    deadline = time.monotonic() + timeout
    bootstrap_path = path.parent / 'bootstrap.json'
    while not bootstrap_path.exists():
        if child.poll() is not None:
            raise ValueError('launch guard exited before registration')
        if time.monotonic() >= deadline:
            raise TimeoutError('launch child registration deadline expired')
        refresh()
        time.sleep(0.05)
    bootstrap = json.loads(bootstrap_path.read_text())
    identity = bootstrap.get('identity') if isinstance(bootstrap, dict) else None
    if (type(bootstrap.get('schema_version')) is not int or bootstrap['schema_version'] != 1
            or bootstrap.get('request_sha256') != request_hash or digest(path) != request_hash
            or bootstrap.get('token') != request['token'] or bootstrap.get('job_sha256') != request['job_sha256']
            or not isinstance(identity, dict) or identity.get('pid') != child.pid
            or identity.get('command') != command or not same_handle(process_identity(child.pid), identity)):
        raise ValueError('launch registration differs from the waiting child')
    return bootstrap


def verify_command_transition(row: dict, current: dict) -> None:
    """Accept only the bound grant's exec transition on the registered PID/ticks."""
    if row.get('launch_protocol') != LAUNCH_PROTOCOL:
        raise ValueError('queue child identity differs; recovery is required')
    path = Path(row['launch_request'])
    request, request_hash = read_request(path)
    bootstrap = json.loads((path.parent / 'bootstrap.json').read_text())
    grant = json.loads((path.parent / 'grant.json').read_text())
    equal_records = json.dumps(grant, sort_keys=True, allow_nan=False) == json.dumps(
        bootstrap, sort_keys=True, allow_nan=False
    )
    if (request_hash != row.get('launch_request_sha256')
            or not equal_records
            or bootstrap.get('identity') != row.get('child_identity')
            or bootstrap.get('request_sha256') != request_hash
            or bootstrap.get('token') != request['token'] or bootstrap.get('job_sha256') != request['job_sha256']
            or current.get('command') != request['command'] or request['command'] != row.get('command')
            or any(current.get(key) != bootstrap['identity'].get(key) for key in ('pid', 'start_ticks'))):
        raise ValueError('queue child command transition is not approved')
    verify_journal(path, request, bootstrap)


def inspect_unapproved_launch(row: dict) -> dict:  # noqa: PLR0912, PLR0915 -- ordered ownership and evidence gates
    """Prove that an interrupted guard can never receive authority to run a model."""
    from scripts.onestep_avatar.execution.queue import owned_workers_live  # noqa: PLC0415 -- avoid import cycle

    if row.get('launch_protocol') != LAUNCH_PROTOCOL or row.get('child_identity') is not None:
        raise ValueError('queue child has no recorded PID; launch recovery is required')
    path = Path(row['launch_request'])
    request, request_hash = read_request(path)
    expected = {'launch_protocol': LAUNCH_PROTOCOL, 'launch_request': str(path.resolve()),
                'launch_request_sha256': request_hash, 'sha256': request['job_sha256'],
                'command': request['command'], 'owner_pid': request['owner_identity']['pid']}
    environment = {TOKEN_ENV: request['token'], JOB_ENV: request['job_sha256']}
    attempts = row.get('attempts')
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        raise ValueError('unapproved launch has no bound attempt')
    for record in (row, attempts[-1]):
        # Attempts store the job identity in their environment; the row owns sha256.
        if any(record.get(key) != value for key, value in expected.items() if key != 'sha256'):
            raise ValueError('unapproved launch attempt bindings differ')
        if (not isinstance(record.get('environment_changes'), dict)
                or any(record['environment_changes'].get(key) != value for key, value in environment.items())
                or type(record.get('attempt_started_ticks')) is not int or record['attempt_started_ticks'] < 0):
            raise ValueError('unapproved launch environment or birth boundary differs')
        if record.get('child_pid') not in (None, row.get('child_pid')) or record.get('child_identity') is not None:
            raise ValueError('unapproved launch contains inconsistent child registration')
    if (row.get('sha256') != request['job_sha256']
            or row['attempt_started_ticks'] != attempts[-1]['attempt_started_ticks']):
        raise ValueError('unapproved launch job or birth boundary differs')
    state = json.loads(Path(request['journal']).read_text())
    saved_row = state.get('jobs', {}).get(request['job_id'])
    if (type(state.get('schema_version')) is not int or state['schema_version'] != 1
            or state.get('owner_pid') != expected['owner_pid']
            or saved_row != row or row.get('state') != 'running'):
        raise ValueError('unapproved launch journal differs')
    owner = process_identity(expected['owner_pid'])
    if owner is not None:
        if any(owner.get(key) != request['owner_identity'][key] for key in ('pid', 'start_ticks')):
            raise ValueError('unapproved launch owner handle changed')
        if not owner['terminal']:
            raise ValueError('unapproved launch owner is still live')
        if owner['command'] not in ([], request['owner_identity']['command']):
            raise ValueError('unapproved launch owner command changed')
    grant = path.parent / 'grant.json'
    if grant.exists() or grant.is_symlink():
        raise ValueError('missing-PID launch has approval evidence; recovery is required')
    bootstrap_path = path.parent / 'bootstrap.json'
    bootstrap = None
    if bootstrap_path.is_symlink():
        raise ValueError('unapproved launch bootstrap must not be a symlink')
    if bootstrap_path.exists():
        bootstrap = json.loads(bootstrap_path.read_text())
        identity = bootstrap.get('identity') if isinstance(bootstrap, dict) else None
        if (type(bootstrap.get('schema_version')) is not int or bootstrap['schema_version'] != 1
                or bootstrap.get('token') != request['token'] or bootstrap.get('job_sha256') != request['job_sha256']
                or bootstrap.get('request_sha256') != request_hash or not isinstance(identity, dict)
                or identity.get('command') != guard_command(path) or identity.get('terminal') is not False):
            raise ValueError('unapproved launch bootstrap differs')
        if row.get('child_pid') is not None and identity.get('pid') != row['child_pid']:
            raise ValueError('unapproved launch child PID differs from bootstrap')
        current = process_identity(identity.get('pid'))
        if current is not None:
            if (any(current.get(key) != identity.get(key) for key in ('pid', 'start_ticks'))
                    or current['command'] not in ([], identity['command'])):
                raise ValueError('unapproved launch child handle changed')
            if not current['terminal']:
                raise ValueError('unapproved launch has a surviving child')
    elif row.get('child_pid') is not None and process_identity(row['child_pid']) is not None:
        raise ValueError('unapproved launch has an unidentified surviving child handle')
    if owned_workers_live(row):
        raise ValueError('unapproved launch has a surviving child or worker')
    if digest(path) != request_hash or grant.exists() or grant.is_symlink():
        raise ValueError('unapproved launch evidence changed during inspection')
    return {'reason': 'original owner terminated before launch approval',
            'launch_request': str(path.resolve()), 'request_sha256': request_hash,
            'grant_absent': True, 'bootstrap': bootstrap, 'model_execution_authorized': False,
            'prior_attempt': json.loads(json.dumps(attempts[-1], allow_nan=False))}


def digest(path: Path) -> str:
    """Hash the small launch record itself, never a model weight fingerprint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_exclusive(path: Path, record: dict) -> None:
    """Expose one complete, flushed record; never replace a previous producer."""
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.launch-') as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
        os.link(handle.name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def same_handle(current: dict | None, saved: dict) -> bool:
    return bool(current and saved.get('terminal') is False and not current['terminal'] and all(
        current[key] == saved.get(key) for key in ('pid', 'start_ticks', 'command')
    ))


def load_request(path: Path) -> dict:
    """Require exact identity, journal and command data before waiting or execution."""
    return read_request(path)[0]


def read_request(path: Path) -> tuple[dict, str]:
    """Parse and hash the same observed bytes, not two independently opened versions."""
    raw = path.read_bytes()
    request = json.loads(raw)
    validate_request(request)
    return request, hashlib.sha256(raw).hexdigest()


def validate_request(request: dict) -> None:
    """Validate producer inputs before creating a launch directory or record."""
    if (not isinstance(request, dict) or set(request) != {
        'schema_version', 'token', 'job_sha256', 'owner_identity', 'job_id', 'journal', 'command'
    } or type(request.get('schema_version')) is not int or request['schema_version'] != 1):
        raise ValueError('invalid launch request schema')
    for field, size in [('token', 32), ('job_sha256', 64)]:
        if not isinstance(request[field], str) or re.fullmatch(r'[0-9a-f]{' + str(size) + '}', request[field]) is None:
            raise ValueError('invalid launch request identity')
    owner = request['owner_identity']
    if (not isinstance(owner, dict) or type(owner.get('pid')) is not int or owner['pid'] <= 0
            or type(owner.get('start_ticks')) is not int or owner['start_ticks'] < 0
            or owner.get('terminal') is not False
            or not isinstance(owner.get('command'), list) or not owner['command']):
        raise ValueError('invalid launch owner identity')
    if (not isinstance(request['job_id'], str) or not request['job_id']
            or not isinstance(request['journal'], str) or not Path(request['journal']).is_absolute()
            or not isinstance(request['command'], list) or not request['command']
            or any(not isinstance(arg, str) for arg in request['command']) or not request['command'][0]):
        raise ValueError('invalid launch command or journal')


def prepare_request(directory: Path, *, token: str, job_sha256: str, owner_identity: dict,
                    job_id: str, journal: Path, command: list[str]) -> Path:
    """Publish a fresh request; the queue remains the command and settings owner."""
    record = {'schema_version': 1, 'token': token, 'job_sha256': job_sha256,
              'owner_identity': owner_identity, 'job_id': job_id,
              'journal': str(journal.resolve()), 'command': command}
    validate_request(record)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'request.json'
    publish_exclusive(path, record)
    load_request(path)
    return path


def verify_journal(path: Path, request: dict, bootstrap: dict) -> None:
    """A launch grant needs a durable running row with the registered child PID."""
    state = json.loads(Path(request['journal']).read_text())
    if not isinstance(state, dict) or not isinstance(state.get('jobs'), dict):
        raise ValueError('launch journal must contain job records')
    row = state.get('jobs', {}).get(request['job_id'])
    expected_environment = {TOKEN_ENV: request['token'], JOB_ENV: request['job_sha256']}
    if (type(state.get('schema_version')) is not int or state['schema_version'] != 1
            or state.get('owner_pid') != request['owner_identity']['pid']
            or not isinstance(row, dict) or row.get('state') != 'running'
            or row.get('sha256') != request['job_sha256'] or row.get('command') != request['command']
            or row.get('launch_protocol') != LAUNCH_PROTOCOL or row.get('launch_request') != str(path.resolve())
            or row.get('launch_request_sha256') != bootstrap['request_sha256']
            or row.get('child_pid') != bootstrap['identity']['pid']
            or row.get('child_identity') != bootstrap['identity']
            or not isinstance(row.get('environment_changes'), dict)
            or any(row.get('environment_changes', {}).get(key) != value
                   for key, value in expected_environment.items())):
        raise ValueError('launch journal does not bind the registered child and command')


def publish_grant(path: Path) -> dict:
    """Approve only the currently waiting, durably registered child."""
    request, request_hash = read_request(path)
    if not same_handle(process_identity(os.getpid()), request['owner_identity']):
        raise ValueError('only the recorded launch owner can publish approval')
    bootstrap = json.loads((path.parent / 'bootstrap.json').read_text())
    if (type(bootstrap.get('schema_version')) is not int or bootstrap['schema_version'] != 1
            or bootstrap.get('request_sha256') != request_hash
            or bootstrap.get('token') != request['token'] or bootstrap.get('job_sha256') != request['job_sha256']
            or not isinstance(bootstrap.get('identity'), dict)
            or not same_handle(process_identity(bootstrap['identity'].get('pid')), bootstrap['identity'])):
        raise ValueError('launch bootstrap is changed or its child is not waiting')
    verify_journal(path, request, bootstrap)
    if digest(path) != request_hash:
        raise ValueError('launch request changed before approval')
    publish_exclusive(path.parent / 'grant.json', bootstrap)
    return bootstrap


def run_guard(path: Path, *, timeout: float = 60, poll_seconds: float = 0.1) -> None:
    """Publish identity, wait for approval, then exec exactly the recorded command."""
    if not math.isfinite(timeout) or not math.isfinite(poll_seconds) or timeout <= 0 or not 0 < poll_seconds <= 1:
        raise ValueError('launch wait limits must be finite and positive')
    path = path.resolve()
    request, request_hash = read_request(path)
    if os.environ.get(TOKEN_ENV) != request['token'] or os.environ.get(JOB_ENV) != request['job_sha256']:
        raise ValueError('launch environment identity differs from the request')
    if (path.parent / 'grant.json').exists():
        raise ValueError('launch grant already exists before child registration')
    identity = process_identity(os.getpid())
    if identity is None or identity['terminal']:
        raise ValueError('cannot register a stable live launch child')
    bootstrap = {'schema_version': 1, 'token': request['token'], 'job_sha256': request['job_sha256'],
                 'request_sha256': request_hash, 'identity': identity}
    publish_exclusive(path.parent / 'bootstrap.json', bootstrap)
    deadline = time.monotonic() + timeout
    grant = path.parent / 'grant.json'
    while not grant.exists():
        if digest(path) != bootstrap['request_sha256']:
            raise ValueError('launch request changed while waiting')
        if not same_handle(process_identity(request['owner_identity']['pid']), request['owner_identity']):
            # Approval may publish between the loop's grant check and owner exit.
            # It still must pass every binding check below; absence never approves.
            if not grant.exists():
                raise ValueError('launch owner exited or changed before approval')
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('launch approval deadline expired')
        time.sleep(poll_seconds)
    approved = json.loads(grant.read_text())
    if (json.dumps(approved, sort_keys=True, allow_nan=False) != json.dumps(bootstrap, sort_keys=True, allow_nan=False)
            or digest(path) != bootstrap['request_sha256']):
        raise ValueError('launch grant or request changed')
    verify_journal(path, request, bootstrap)
    os.execvpe(request['command'][0], request['command'], os.environ.copy())


def _bounded_bytes(path: Path, deadline: float, *, limit: int = 65536) -> bytes:
    """Read recovery input without blocking on a FIFO or accepting unbounded bytes."""
    if time.monotonic() >= deadline:
        raise TimeoutError('owned recovery observation deadline exceeded')
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('owned recovery input must be a bounded regular file')
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError('owned recovery input exceeds its byte bound')
    if time.monotonic() >= deadline:
        raise TimeoutError('owned recovery observation deadline exceeded')
    return data


def _saved_notification_bytes(path: Path, deadline: float) -> dict[str, bytes]:
    records = {'contract': _bounded_bytes(path, deadline)}
    total = len(records['contract'])
    events = path.with_name(path.name + '.events')
    for index, target in enumerate(events.iterdir()):
        if index >= 8192:
            raise ValueError('owned recovery notification inventory exceeds 8192 files')
        if target.name.startswith('.supervision-'):
            continue
        data = _bounded_bytes(target, deadline)
        total += len(data)
        if total > 8 * 1024 * 1024:
            raise ValueError('owned recovery notification inventory exceeds 8 MiB')
        records[target.name] = data
    return records


def read_completed_notifications(
    path: Path, *, contract_sha256: str, token: str, job_sha256: str, world: int, timeout: float = 5,
) -> dict:
    """Check saved rank events with existing supervisor rules; infer no live timing."""
    from scripts.onestep_avatar.execution import supervision  # noqa: PLC0415 -- scoped recovery compatibility

    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('owned recovery timeout must be finite and positive')
    if not isinstance(contract_sha256, str) or re.fullmatch('[0-9a-f]{64}', contract_sha256) is None:
        raise ValueError('owned recovery requires the original frozen notification hash')
    deadline = time.monotonic() + timeout
    records = _saved_notification_bytes(path, deadline)
    with tempfile.TemporaryDirectory(prefix='onestep-owned-recovery-') as directory:
        copied = Path(directory) / 'contract.json'
        copied.write_bytes(records['contract'])
        copied.with_name(copied.name + '.events').mkdir()
        for name, data in records.items():
            if name != 'contract':
                (copied.with_name(copied.name + '.events') / name).write_bytes(data)
        contract, contract_hash = supervision._contract(copied, contract_sha256)
        if (type(world) is not int or world < 1 or contract['world'] != world
                or contract['token'] != token or contract['job_sha256'] != job_sha256
                or world * len(contract['phases']) * 2 > 8192):
            raise ValueError('saved notifications differ from the original attempt/rank inventory')
        progress = {rank: {'index': 0, 'active': None, 'identity': None, 'awaiting_since': 0}
                    for rank in range(world)}
        seen = {}
        supervision._consume(copied, contract, contract_hash, progress, seen, 0)
        if any(item['index'] != len(contract['phases']) or item['active'] is not None
               for item in progress.values()):
            raise ValueError('saved notifications lack the complete expected rank/phase inventory')
    if records != _saved_notification_bytes(path, deadline):
        raise ValueError('owned recovery notification bytes changed during inspection')
    return {'schema_version': 1, 'contract_sha256': contract_hash, 'token': token,
            'job_sha256': job_sha256, 'world': world, 'phases': contract['phases'],
            'rank_identities': [{'rank': rank, 'identity': item['identity']} for rank, item in progress.items()],
            'event_sha256': seen, 'continuous_supervision': False}


def _ended_handles(identities: list[dict], deadline: float) -> list[dict]:
    observations = []
    for saved in identities:
        if time.monotonic() >= deadline:
            raise TimeoutError('owned recovery observation deadline exceeded')
        current = process_identity(saved['pid'])
        if current is not None:
            if any(current.get(key) != saved.get(key) for key in ('pid', 'start_ticks')):
                raise ValueError('owned recovery registered PID was reused')
            if not current['terminal']:
                raise ValueError('owned recovery registered handle remains live')
            if current['command'] not in ([], saved['command']):
                raise ValueError('owned recovery terminal command differs')
        observations.append({'saved_identity': saved, 'current_identity': current})
    return observations


def _row_digest(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _recovery_identity(identity: dict) -> None:
    if (not isinstance(identity, dict) or type(identity.get('pid')) is not int or identity['pid'] <= 0
            or type(identity.get('start_ticks')) is not int or identity['start_ticks'] < 0
            or identity.get('terminal') is not False or not isinstance(identity.get('command'), list)
            or not identity['command'] or any(not isinstance(value, str) for value in identity['command'])):
        raise ValueError('registered bookkeeping recovery requires exact original live identities')


def recover_registered_attempt(  # noqa: PLR0912, PLR0913, PLR0915 -- explicit bounded operational closure
    process_ledger: Path, token: str, *, expected_row_sha256: str, launch_evidence: Path,
    expected_launch_sha256: str, timeout: float = 10,
) -> dict:
    """Close ended registered non-training handles; certify no unknown workers or result."""
    from scripts.onestep_avatar.execution import process_registry  # noqa: PLC0415 -- one existing ledger transaction
    from scripts.onestep_avatar.execution.queue import ALLOWED_GPUS  # noqa: PLC0415 -- existing dispatch device scope

    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('registered bookkeeping recovery timeout must be finite and positive')
    for name, value, length in (('token', token, 32), ('row hash', expected_row_sha256, 64),
                                ('launch hash', expected_launch_sha256, 64)):
        if not isinstance(value, str) or re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is None:
            raise ValueError('registered bookkeeping recovery requires exact original ' + name)
    deadline = time.monotonic() + timeout
    if launch_evidence.is_symlink() or process_ledger.is_symlink():
        raise ValueError('registered bookkeeping recovery paths must not be symlinks')
    launch_evidence, process_ledger = launch_evidence.absolute(), process_ledger.absolute()
    launch_bytes = _bounded_bytes(launch_evidence, deadline)
    if hashlib.sha256(launch_bytes).hexdigest() != expected_launch_sha256:
        raise ValueError('registered bookkeeping recovery original launch hash differs')
    launch = json.loads(launch_bytes)
    registry = object.__new__(process_registry.ProcessRegistry)
    registry.path = process_ledger
    with registry._lock(timeout=max(deadline - time.monotonic(), 1e-12)):
        record = registry._read()
        row = record['attempts'].get(token)
        if (not isinstance(row, dict) or _row_digest(row) != expected_row_sha256
                or row.get('state') != 'active' or row.get('containment') != 'linux_subreaper_v1'
                or row.get('reported_rank_identities') != []):
            raise ValueError('registered bookkeeping recovery original row is changed or unsupported')
        _recovery_identity(row.get('owner'))
        processes = row.get('processes')
        if not isinstance(processes, dict) or not processes or len(processes) > 4096:
            raise ValueError('registered bookkeeping recovery descendant inventory is invalid')
        child = processes.get(str(row.get('child_pid')))
        if not isinstance(child, dict) or 'child' not in child.get('roles', []):
            raise ValueError('registered bookkeeping recovery has no exact recorded child')
        identities = []
        for key, process in processes.items():
            if not isinstance(process, dict):
                raise ValueError('registered bookkeeping recovery process entry is invalid')
            identity, roles = process.get('identity'), process.get('roles')
            _recovery_identity(identity)
            if (key != str(identity['pid']) or not isinstance(roles, list) or not roles
                    or any(role not in ('child', 'descendant', 'observed') for role in roles)
                    or ('child' in roles and identity['pid'] != row['child_pid'])
                    or identity['pid'] == row['owner']['pid']):
                raise ValueError('registered bookkeeping recovery has ranks or ambiguous ownership')
            identities.append(identity)
        command = child['identity']['command']
        if (len(command) < 3 or not Path(command[0]).is_absolute()
                or re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(command[0]).name) is None
                or command[1] != '-m'
                or command[2] not in {'scripts.onestep_avatar.' + name for name in REGISTERED_RECOVERY_MODULES}
                or any(flag in command[3:] for flag in ('--supervise', '--execute', '--loop', '--recover'))):
            raise ValueError('registered bookkeeping recovery requires an allowed direct non-training command')
        gpus = row.get('gpus')
        if (not isinstance(gpus, list) or len(gpus) != 1 or type(gpus[0]) is not int
                or not set(gpus).issubset(ALLOWED_GPUS)):
            raise ValueError('registered bookkeeping recovery requires one allowed physical GPU')
        environment = launch.get('environment_changes') if isinstance(launch, dict) else None
        if (not isinstance(launch, dict) or launch.get('command') != command
                or launch.get('physical_gpu') != gpus[0] or type(launch.get('physical_gpu')) is not int
                or type(launch.get('visible_gpu')) is not int or launch['visible_gpu'] != 0
                or not isinstance(environment, dict) or environment.get(TOKEN_ENV) != token
                or not isinstance(environment.get(JOB_ENV), str)
                or re.fullmatch('[0-9a-f]{64}', environment[JOB_ENV]) is None
                or environment.get('CUDA_VISIBLE_DEVICES') != str(gpus[0])):
            raise ValueError('registered bookkeeping recovery launch identity or device mapping differs')
        complete = [observation for observation in row.get('observations', [])
                    if isinstance(observation, dict) and observation.get('complete') is True
                    and observation.get('error') is None]
        if not any(all(any(isinstance(seen, dict) and all(seen.get(key) == identity.get(key)
                       for key in ('pid', 'start_ticks', 'command'))
                       for seen in observation.get('identities', [])) for identity in identities)
                   for observation in complete):
            raise ValueError('registered bookkeeping recovery lacks complete original registered containment')
        handles = [row['owner'], *identities]
        before = _ended_handles(handles, deadline)
        memory = process_registry.gpu_memory(timeout=max(deadline - time.monotonic(), 1e-12))
        if (gpus[0] not in memory or any(type(value) is not int or value < 0 for value in memory.values())
                or memory[gpus[0]] >= 1024):
            raise ValueError('registered bookkeeping recovery requires complete currently idle GPU inventory')
        after = _ended_handles(handles, deadline)
        if (launch_evidence.is_symlink() or process_ledger.is_symlink()
                or launch_bytes != _bounded_bytes(launch_evidence, deadline)
                or _row_digest(registry._read()['attempts'].get(token)) != expected_row_sha256):
            raise ValueError('registered bookkeeping recovery original bytes or paths changed')
        result = {'schema_version': 1, 'basis': 'registered_handle_bookkeeping_recovery',
                  'token': token, 'job': row.get('job'), 'original_row_sha256': expected_row_sha256,
                  'launch_evidence': str(launch_evidence), 'launch_evidence_sha256': expected_launch_sha256,
                  'recovery_source_sha256': digest(Path(__file__)), 'observer': process_identity(os.getpid()),
                  'handle_observations': [before, after],
                  'sampled_gpu_memory_mib': {str(gpus[0]): memory[gpus[0]]},
                  'registered_processes_absent': True, 'continuous_supervision': False,
                  'containment_complete': False, 'unknown_descendants_unproven': True,
                  'original_exitcode': None,
                  'scope_limit': 'Only exact registered handles were observed ended. Unknown descendants '
                                 'after owner loss, original exit and scientific success are unproven. '
                                 'The legacy closed-row containment label adds no evidence.',
                  'recovered_at': time.time()}
        if time.monotonic() >= deadline:
            raise TimeoutError('registered bookkeeping recovery observation deadline exceeded')
        row.update(state='closed', recovery=result)
        registry._write(record)
        return result


def recover_owned_attempt(  # noqa: PLR0912, PLR0915 -- one locked bounded closure, original artifacts retained
    process_ledger: Path, request_path: Path, *, timeout: float = 10,
) -> dict:
    """Close an ended approved registered workload; preserve lost-supervision scope."""
    from scripts.onestep_avatar.execution import process_registry, supervision  # noqa: PLC0415 -- operational owners

    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('owned recovery timeout must be finite and positive')
    deadline = time.monotonic() + timeout
    request_path, process_ledger = request_path.resolve(), process_ledger.resolve()
    original_bytes = {request_path: _bounded_bytes(request_path, deadline)}
    request = json.loads(original_bytes[request_path])
    validate_request(request)
    request_hash = hashlib.sha256(original_bytes[request_path]).hexdigest()
    for name in ('bootstrap.json', 'grant.json'):
        path = request_path.parent / name
        original_bytes[path] = _bounded_bytes(path, deadline)
    bootstrap = json.loads(original_bytes[request_path.parent / 'bootstrap.json'])
    grant = json.loads(original_bytes[request_path.parent / 'grant.json'])
    if (bootstrap != grant or not isinstance(bootstrap, dict)
            or type(bootstrap.get('schema_version')) is not int or bootstrap['schema_version'] != 1
            or bootstrap.get('request_sha256') != request_hash or bootstrap.get('token') != request['token']
            or bootstrap.get('job_sha256') != request['job_sha256']
            or not isinstance(bootstrap.get('identity'), dict)
            or bootstrap['identity'].get('command') != guard_command(request_path)):
        raise ValueError('owned recovery launch approval bindings differ')
    journal = Path(request['journal'])
    original_bytes[journal] = _bounded_bytes(journal, deadline, limit=8 * 1024 * 1024)
    state = json.loads(original_bytes[journal])
    # Validate the exact journal byte snapshot through the existing launch gate.
    with tempfile.TemporaryDirectory(prefix='onestep-owned-journal-') as directory:
        copied = Path(directory) / 'journal.json'
        copied.write_bytes(original_bytes[journal])
        verify_journal(request_path, {**request, 'journal': str(copied)}, bootstrap)
    row = state['jobs'][request['job_id']]
    attempts = row.get('attempts')
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        raise ValueError('owned recovery has no original attempt')
    for field in ('child_pid', 'child_identity', 'command', 'owner_pid', 'gpus', 'process_ledger',
                  'environment_changes', 'supervision_contract', 'launch_protocol',
                  'launch_request', 'launch_request_sha256', 'attempt_started_ticks'):
        if row.get(field) != attempts[-1].get(field):
            raise ValueError('owned recovery journal and original attempt differ')
    if row.get('process_ledger') != str(process_ledger):
        raise ValueError('owned recovery process ledger path differs')
    environment = row['environment_changes']
    notification_path = Path(row['supervision_contract'])
    if environment.get(supervision.SUPERVISION_ENV) != str(notification_path):
        raise ValueError('owned recovery notification path differs')
    # The original canonical launch states the exact process count.
    command = request['command']
    if command.count('--num_processes') != 1:
        raise ValueError('owned recovery requires an explicit original rank count')
    try:
        world = int(command[command.index('--num_processes') + 1])
    except (IndexError, ValueError) as error:
        raise ValueError('owned recovery original rank count is invalid') from error
    remaining = deadline - time.monotonic()
    notifications = read_completed_notifications(
        notification_path, contract_sha256=environment[supervision.SUPERVISION_SHA_ENV],
        token=request['token'], job_sha256=request['job_sha256'], world=world, timeout=remaining,
    )
    registry = object.__new__(process_registry.ProcessRegistry)
    registry.path = process_ledger
    with registry._lock(timeout=max(deadline - time.monotonic(), 1e-12)):
        record = registry._read()
        owned = record['attempts'].get(request['token'])
        if (not isinstance(owned, dict) or owned.get('state') != 'active'
                or owned.get('containment') != 'linux_subreaper_v1'
                or owned.get('owner') != request['owner_identity']
                or owned.get('child_pid') != bootstrap['identity']['pid']
                or owned.get('gpus') != row.get('gpus') or owned.get('job') != request['job_id']
                or owned.get('attempt_started_ticks') != row.get('attempt_started_ticks')):
            raise ValueError('owned recovery ledger differs from the original launch')
        processes = owned.get('processes')
        if not isinstance(processes, dict) or not processes or len(processes) > 4096:
            raise ValueError('owned recovery registered descendant inventory is invalid')
        identities = [item['identity'] for item in processes.values()]
        for item in notifications['rank_identities']:
            identity = item['identity']
            saved = processes.get(str(identity['pid']))
            if saved is None or saved['identity'] != identity:
                raise ValueError('owned recovery rank is not an exact previously contained descendant')
        child = processes.get(str(owned['child_pid']))
        if (child is None or 'child' not in child.get('roles', [])
                or child['identity']['start_ticks'] != bootstrap['identity'].get('start_ticks')
                or child['identity']['command'] != command):
            raise ValueError('owned recovery original launch child differs')
        complete = [observation for observation in owned.get('observations', [])
                    if observation.get('complete') is True and observation.get('error') is None]
        if not any(all(any(all(seen.get(key) == identity.get(key) for key in ('pid', 'start_ticks', 'command'))
                           for seen in observation.get('identities', [])) for identity in identities)
                   for observation in complete):
            raise ValueError('owned recovery lacks original complete registered containment')
        handles = [owned['owner'], *identities]
        before = _ended_handles(handles, deadline)
        memory = process_registry.gpu_memory(timeout=max(deadline - time.monotonic(), 1e-12))
        gpus = owned['gpus']
        if (not gpus or not set(gpus).issubset(memory)
                or any(type(value) is not int or value < 0 for value in memory.values())
                or any(memory[gpu] >= 1024 for gpu in gpus)):
            raise ValueError('owned recovery requires complete currently idle selected GPU inventory')
        after = _ended_handles(handles, deadline)
        for path, data in original_bytes.items():
            if data != _bounded_bytes(path, deadline, limit=8 * 1024 * 1024 if path == journal else 65536):
                raise ValueError('owned recovery original launch or journal bytes changed')
        if notifications != read_completed_notifications(
            notification_path, contract_sha256=notifications['contract_sha256'], token=request['token'],
            job_sha256=request['job_sha256'], world=world, timeout=max(deadline - time.monotonic(), 1e-12),
        ):
            raise ValueError('owned recovery saved notification evidence changed')
        result = {'schema_version': 1, 'basis': 'bounded_dead_owner_recovery',
                  'token': request['token'], 'job_sha256': request['job_sha256'],
                  'request': str(request_path), 'request_sha256': request_hash,
                  'journal': str(journal), 'journal_sha256': hashlib.sha256(original_bytes[journal]).hexdigest(),
                  'original_row_sha256': hashlib.sha256(json.dumps(
                      owned, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest(),
                  'recovery_source_sha256': digest(Path(__file__)), 'observer': process_identity(os.getpid()),
                  'notifications': notifications, 'handle_observations': [before, after],
                  'sampled_gpu_memory_mib': {str(gpu): memory[gpu] for gpu in gpus},
                  'registered_workers_absent': True, 'continuous_supervision': False,
                  'containment_complete': False,
                  'scope_limit': 'Unknown descendants after original owner loss are not certified absent; '
                                 'the legacy closed-row reader basis adds no containment evidence.',
                  'recovered_at': time.time()}
        if time.monotonic() >= deadline:
            raise TimeoutError('owned recovery observation deadline exceeded')
        owned.update(state='closed', recovery=result)
        registry._write(record)
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--recover-owned', action='store_true', help='close an ended approved own attempt')
    recovery.add_argument('--recover-registered-owned', action='store_true', help='close ended direct non-training handles')
    parser.add_argument('--process-ledger', type=Path)
    parser.add_argument('--token')
    parser.add_argument('--expected-row-sha256')
    parser.add_argument('--launch-evidence', type=Path)
    parser.add_argument('--expected-launch-sha256')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--poll-seconds', type=float, default=0.1)
    args = parser.parse_args(argv)
    registered_options = (args.token, args.expected_row_sha256, args.launch_evidence, args.expected_launch_sha256)
    if args.recover_registered_owned:
        if args.request is not None or args.process_ledger is None or any(value is None for value in registered_options):
            parser.error('--recover-registered-owned requires ledger, token, original row hash and launch evidence/hash')
        print(json.dumps(recover_registered_attempt(  # noqa: T201 -- explicit operational evidence
            args.process_ledger, args.token, expected_row_sha256=args.expected_row_sha256,
            launch_evidence=args.launch_evidence, expected_launch_sha256=args.expected_launch_sha256,
            timeout=args.timeout), indent=2))
    elif any(value is not None for value in registered_options):
        parser.error('registered recovery options require --recover-registered-owned')
    elif args.request is None:
        parser.error('ordinary launch and approved recovery require --request')
    elif args.recover_owned:
        if args.process_ledger is None:
            parser.error('--recover-owned requires --process-ledger')
        print(json.dumps(recover_owned_attempt(  # noqa: T201 -- explicit operational evidence
            args.process_ledger, args.request, timeout=args.timeout), indent=2))
    else:
        if args.process_ledger is not None:
            parser.error('--process-ledger requires --recover-owned')
        run_guard(args.request, timeout=args.timeout, poll_seconds=args.poll_seconds)


if __name__ == '__main__':
    main()
