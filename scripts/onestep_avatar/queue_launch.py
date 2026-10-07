"""Register a waiting child before its recorded command can execute; see doc/queue_launch.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.onestep_avatar.queue import process_identity
from scripts.onestep_avatar.queue_protocol import JOB_ENV, LAUNCH_PROTOCOL, TOKEN_ENV

if TYPE_CHECKING:
    import subprocess


def guard_command(path: Path) -> list[str]:
    """Use the current conda interpreter; no model command runs at process creation."""
    return [sys.executable, '-m', 'scripts.onestep_avatar.queue_launch', '--request', str(path.resolve())]


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
    from scripts.onestep_avatar.queue import owned_workers_live  # noqa: PLC0415 -- avoid import cycle

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--poll-seconds', type=float, default=0.1)
    args = parser.parse_args(argv)
    run_guard(args.request, timeout=args.timeout, poll_seconds=args.poll_seconds)


if __name__ == '__main__':
    main()
