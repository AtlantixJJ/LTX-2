"""Bound registered children and retain own process records; see doc/execution/supervision.md."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import re
import select
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV

if TYPE_CHECKING:
    from scripts.onestep_avatar.execution.process_registry import ProcessRegistry

SUPERVISION_ENV = "ONESTEP_AVATAR_SUPERVISION"
SUPERVISION_SHA_ENV = "ONESTEP_AVATAR_SUPERVISION_SHA256"


def _publish(path: Path, record: dict) -> None:
    """Publish complete bytes exclusively; duplicate events must fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".supervision-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _contract(path: Path, expected_sha: str | None = None) -> tuple[dict, str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    record = json.loads(data)
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or (expected_sha is not None and digest != expected_sha)
            or not isinstance(record.get("token"), str) or re.fullmatch("[0-9a-f]{32}", record["token"]) is None
            or not isinstance(record.get("job_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", record["job_sha256"]) is None
            or not isinstance(record.get("budget_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", record["budget_sha256"]) is None
            or type(record.get("world")) is not int or record["world"] < 1
            or not isinstance(record.get("phases"), list) or not record["phases"]
            or any(not isinstance(value, str) or not value for value in record["phases"])
            or len(set(record["phases"])) != len(record["phases"])):
        raise ValueError("supervision notification contract is invalid or changed")
    return record, digest


def prepare_notifications(path: Path, *, token: str, job_sha256: str, world: int,
                          phases: list[str], budget_sha256: str) -> dict[str, str]:
    """Freeze one attempt's expected rank/phase inventory before its launch."""
    path = path.resolve()
    record = {"schema_version": 1, "token": token, "job_sha256": job_sha256,
              "world": world, "phases": phases, "budget_sha256": budget_sha256}
    # Validate before creating even the contract directory.
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "contract.json"
        probe.write_bytes(data)
        _contract(probe)
    _publish(path, record)
    path.with_name(path.name + ".events").mkdir()
    return {SUPERVISION_ENV: str(path), SUPERVISION_SHA_ENV: hashlib.sha256(data).hexdigest()}


def notify_phase(phase: str, event: str, rank: int, *, budget_sha256: str | None = None,
                 environment: dict | None = None) -> None:
    """Emit one bound phase boundary; ordinary unguarded runs emit nothing."""
    environment = os.environ if environment is None else environment
    path, digest = environment.get(SUPERVISION_ENV), environment.get(SUPERVISION_SHA_ENV)
    if path is None and digest is None:
        return
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ValueError("supervision requires paired notification path and hash")
    contract, digest = _contract(Path(path), digest)
    declared_rank = environment.get("RANK")
    if (environment.get(TOKEN_ENV) != contract["token"] or environment.get(JOB_ENV) != contract["job_sha256"]
            or budget_sha256 != contract["budget_sha256"]
            or (declared_rank is not None and declared_rank != str(rank))
            or type(rank) is not int or not 0 <= rank < contract["world"]
            or phase not in contract["phases"] or event not in ("begin", "end")):
        raise ValueError("phase notification differs from its bound attempt/rank/phase")
    from scripts.onestep_avatar.execution.queue import (  # noqa: PLC0415 -- one model-free identity owner
        process_identity,
    )

    index = contract["phases"].index(phase)
    identity = process_identity(os.getpid())
    if identity is None or identity["terminal"]:
        raise ValueError("phase producer identity is unavailable")
    record = {"schema_version": 1, "contract_sha256": digest, "token": contract["token"],
              "job_sha256": contract["job_sha256"], "budget_sha256": contract["budget_sha256"],
              "rank": rank, "phase": phase, "phase_index": index, "event": event, "identity": identity}
    target = Path(path).with_name(Path(path).name + ".events") / f"rank{rank:04d}.phase{index:06d}.{event}.json"
    _publish(target, record)


def _positive(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _observe(identity: dict, command: list[str], worker_record: dict) -> dict | None:
    from scripts.onestep_avatar.execution.queue import process_identity  # noqa: PLC0415 -- existing identity authority

    current = process_identity(identity["pid"])
    if current is None:
        return None
    if any(current[key] != identity[key] for key in ("pid", "start_ticks")):
        raise ValueError("registered child PID/start identity changed")
    if not current["terminal"] and current["command"] != identity["command"]:
        from scripts.onestep_avatar.execution.queue_launch import verify_command_transition  # noqa: PLC0415

        verify_command_transition(worker_record, current)
    if not current["terminal"] and current["command"] not in (identity["command"], command):
        raise ValueError("registered child command differs")
    return current


def _owned_pidfd(identity: dict, command: list[str], worker_record: dict) -> int:
    before = _observe(identity, command, worker_record)
    if before is None or before["terminal"]:
        raise ProcessLookupError("registered child is already terminal or missing")
    descriptor = _pidfd_open(identity["pid"])
    try:
        after = _observe(identity, command, worker_record)
        if after is None or after["terminal"]:
            raise ProcessLookupError("registered child ended during pidfd acquisition")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _pidfd_syscall(number: int, *arguments: object) -> int:
    """Use native Linux pidfds when this old Python/libc lacks named wrappers."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise OSError("exact child signaling requires supported Linux pidfd syscalls")
    # These two numbers match Linux's x86_64 and asm-generic syscall ABI.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _pidfd_open(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def _pidfd_signal(descriptor: int, signum: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(descriptor, signum)
    else:
        _pidfd_syscall(424, ctypes.c_int(descriptor), ctypes.c_int(signum), ctypes.c_void_p(), ctypes.c_uint(0))


def _worker_inventory(registry: ProcessRegistry, timeout: float) -> dict:
    """Observe only the exact live owner's tracked tree; no environment scan."""
    observed = registry.observe_workers(timeout=timeout)
    if (not isinstance(observed, dict) or type(observed.get("workers_live")) is not bool
            or type(observed.get("complete")) is not bool
            or type(observed.get("schema_version")) is not int or observed.get("schema_version") != 1):
        raise ValueError("invalid worker inventory result")
    return observed


def _gpu_inventory(gpus: tuple[int, ...], timeout: float) -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=timeout, check=True,
    )
    from scripts.onestep_avatar.execution.queue import parse_gpu_memory  # noqa: PLC0415 -- shared inventory validation

    rows = parse_gpu_memory(result.stdout)
    if not set(gpus).issubset(rows):
        raise ValueError("sampled total-device inventory omits a selected GPU")
    return {gpu: rows[gpu] * 1024 * 1024 for gpu in gpus}


def _pidfd_terminal(descriptor: int) -> bool:
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    return bool(poller.poll(0))


def _stop_owned(  # noqa: PLR0912, PLR0913, PLR0915 -- bounded aggregate exact-handle shutdown
    child: subprocess.Popen, identity: dict, command: list[str], worker_record: dict,
    registry: ProcessRegistry, leader_descriptor: int | None, *, grace: float,
    inventory_timeout: float, poll_seconds: float, result: dict, started: float,
) -> bool:
    """Stop every proved own descendant; keep ambiguous handles unsignaled."""
    handles = {}
    newly_opened = set()
    terminal = set()
    uncertain = False
    if leader_descriptor is not None:
        handles[(identity["pid"], identity["start_ticks"])] = (identity, leader_descriptor)
    try:
        for signum in (signal.SIGTERM, signal.SIGKILL):
            deadline = time.monotonic() + grace
            signaled = set()
            while True:
                remaining = deadline - time.monotonic()
                candidates = []
                inventory = None
                if remaining > 0:
                    try:
                        budget = min(inventory_timeout, remaining)
                        candidates.extend(registry.registered_identities(timeout=budget))
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            inventory = _worker_inventory(registry, min(inventory_timeout, remaining))
                            candidates.extend(inventory.get("identities", []))
                    except Exception as error:
                        uncertain = True
                        result["events"].append({"event": "shutdown_inventory_failed", "error": str(error)})
                for candidate in candidates:
                    if time.monotonic() >= deadline:
                        break
                    pair = (candidate["pid"], candidate["start_ticks"])
                    if pair in terminal:
                        continue
                    if pair in handles:
                        # The live-owner registry can observe an owned worker's exec.
                        # Retain the original launch proof for the leader's transition.
                        if pair[0] != child.pid:
                            handles[pair] = (candidate, handles[pair][1])
                        continue
                    try:
                        descriptor = _owned_pidfd(candidate, candidate["command"], {})
                    except ProcessLookupError:
                        terminal.add(pair)
                        continue
                    except (OSError, ValueError) as error:
                        uncertain = True
                        result["events"].append({"event": "worker_signal_refused", "identity": candidate,
                                                  "error": str(error)})
                        continue
                    handles[pair] = (candidate, descriptor)
                    newly_opened.add(descriptor)
                for pair, (saved, descriptor) in handles.items():
                    if time.monotonic() >= deadline:
                        break
                    if pair in signaled or _pidfd_terminal(descriptor):
                        continue
                    try:
                        if pair[0] == child.pid:
                            current = _observe(identity, command, worker_record)
                        else:
                            current = _observe(saved, saved["command"], {})
                        if current is None or current["terminal"]:
                            continue
                        _pidfd_signal(descriptor, signum)
                        signaled.add(pair)
                        result["events"].append({"event": "signal", "signal": signal.Signals(signum).name,
                                                  "pid": saved["pid"], "start_ticks": saved["start_ticks"],
                                                  "elapsed_s": time.monotonic() - started})
                    except ProcessLookupError:
                        terminal.add(pair)
                    except (OSError, ValueError) as error:
                        uncertain = True
                        result["events"].append({"event": "worker_signal_refused", "identity": saved,
                                                  "error": str(error)})
                        signaled.add(pair)  # Retry only in the next finite stage.
                child.poll()
                all_terminal = all(_pidfd_terminal(descriptor) for _, descriptor in handles.values())
                if (child.returncode is not None and all_terminal and inventory is not None
                        and inventory["complete"] and not inventory["workers_live"]):
                    return uncertain
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(poll_seconds, remaining))
        return uncertain
    finally:
        for descriptor in newly_opened:
            os.close(descriptor)


def _consume(path: Path, contract: dict, digest: str, progress: dict, seen: dict[str, str], now: float,
             observations: list[dict] | None = None) -> None:
    """Consume complete ordered rank events without trusting filename alone."""
    _contract(path, digest)
    events = path.with_name(path.name + ".events")
    available = {target.name: target for target in events.iterdir() if not target.name.startswith(".supervision-")}
    for name, saved_sha in seen.items():
        if name not in available or hashlib.sha256(available[name].read_bytes()).hexdigest() != saved_sha:
            raise ValueError("consumed phase notification bytes changed or disappeared")
    for rank in range(contract["world"]):
        item = progress[rank]
        while item["index"] < len(contract["phases"]):
            event = "end" if item["active"] is not None else "begin"
            name = f"rank{rank:04d}.phase{item['index']:06d}.{event}.json"
            target = events / name
            if name not in available:
                break
            data = target.read_bytes()
            record = json.loads(data)
            expected = {"schema_version": 1, "contract_sha256": digest, "token": contract["token"],
                        "job_sha256": contract["job_sha256"], "budget_sha256": contract["budget_sha256"],
                        "rank": rank, "phase_index": item["index"], "phase": contract["phases"][item["index"]],
                        "event": event}
            identity = record.get("identity") if isinstance(record, dict) else None
            if (not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items())
                    or type(record.get("schema_version")) is not int or type(record.get("rank")) is not int
                    or type(record.get("phase_index")) is not int or not isinstance(identity, dict)
                    or type(identity.get("pid")) is not int or identity["pid"] <= 0
                    or type(identity.get("start_ticks")) is not int or identity["start_ticks"] < 0
                    or identity.get("terminal") is not False or not isinstance(identity.get("command"), list)
                    or not identity["command"] or any(not isinstance(value, str) for value in identity["command"])
                    or (item["identity"] is not None and identity != item["identity"])
                    or any(other != rank and value["identity"] is not None
                           and all(value["identity"][key] == identity[key] for key in ("pid", "start_ticks"))
                           for other, value in progress.items())):
                raise ValueError("phase notification identity/rank/order/bindings differ")
            item["identity"] = identity
            seen[name] = hashlib.sha256(data).hexdigest()
            if observations is not None:
                observations.append({"rank": rank, "phase": expected["phase"], "event": event,
                                     "producer_identity": identity,
                                     "received_monotonic_s": now,
                                     "observed_phase_elapsed_s": None if event == "begin" else now - item["active"]})
            if event == "begin":
                item["active"] = now
            else:
                item["active"] = None
                item["index"] += 1
                item["awaiting_since"] = now
    for name in available:
        if name not in seen:
            raise ValueError("unexpected or out-of-order phase notification")


def supervise(  # noqa: PLR0912, PLR0913, PLR0915 -- one finite lifecycle, explicit independent bounds
    child: subprocess.Popen, *, identity: dict, command: list[str], worker_record: dict,
    claims: ProcessRegistry, gpus: tuple[int, ...], evidence_path: Path, notifications_path: Path | None,
    startup_seconds: float, phase_seconds: float, shutdown_seconds: float,
    overall_seconds: float | None = None, sampled_total_device_limit_bytes: int | None = None,
    inventory_timeout_seconds: float = 5, poll_seconds: float = 0.1,
    sample_seconds: float = 5,
) -> dict:
    """Return persisted bounded evidence; leave own record closure to the queue."""
    for value, name in ((startup_seconds, "startup_seconds"), (phase_seconds, "phase_seconds"),
                        (shutdown_seconds, "shutdown_seconds"),
                        (inventory_timeout_seconds, "inventory_timeout_seconds"),
                        (poll_seconds, "poll_seconds"), (sample_seconds, "sample_seconds")):
        _positive(value, name)
    if overall_seconds is not None:
        _positive(overall_seconds, "overall_seconds")
    if sampled_total_device_limit_bytes is not None and (
            type(sampled_total_device_limit_bytes) is not int or sampled_total_device_limit_bytes <= 0):
        raise ValueError("sampled_total_device_limit_bytes must be a positive integer")
    changes = worker_record.get("environment_changes", {})
    if notifications_path is None:
        if overall_seconds is None:
            raise ValueError("supervision without phase notifications requires an explicit overall deadline")
        contract = {"token": claims.token, "job_sha256": changes.get(JOB_ENV), "world": 0, "phases": []}
        digest = None
    else:
        contract, digest = _contract(notifications_path)
    if (not isinstance(identity, dict) or identity.get("pid") != child.pid
            or type(identity.get("start_ticks")) is not int or not isinstance(identity.get("command"), list)
            or not identity["command"] or not command or type(identity.get("pid")) is not int
            or set(gpus) != claims.owned or len(set(gpus)) != len(gpus)
            or claims.token != contract["token"]
            or changes.get(TOKEN_ENV) != contract["token"]
            or not isinstance(changes.get(JOB_ENV), str) or re.fullmatch("[0-9a-f]{64}", changes[JOB_ENV]) is None
            or changes.get(JOB_ENV) != contract["job_sha256"]
            or (notifications_path is not None and (
                changes.get(SUPERVISION_ENV) != str(notifications_path.resolve())
                or changes.get(SUPERVISION_SHA_ENV) != digest))):
        raise ValueError("supervision differs from registered child/process registry/attempt")
    started = time.monotonic()
    progress = {rank: {"index": 0, "active": None, "awaiting_since": started, "identity": None}
                for rank in range(contract["world"])}
    result = {"schema_version": 1, "state": "failed", "error": None, "identity": identity,
              "command": command, "notification_contract_sha256": digest, "workers_absent": False,
              "process_records_retained": True, "returncode": None, "events": [], "sampled_total_device_bytes": [],
              "phase_observations": [],
              "limits": {"startup_seconds": startup_seconds if notifications_path is not None else None,
                         "notification_gap_seconds": startup_seconds if notifications_path is not None else None,
                         "phase_seconds": phase_seconds if notifications_path is not None else None,
                         "shutdown_seconds": shutdown_seconds,
                         "overall_seconds": overall_seconds,
                         "sampled_total_device_limit_bytes": sampled_total_device_limit_bytes,
                         "inventory_timeout_seconds": inventory_timeout_seconds}}
    descriptor = None
    identity_uncertain = False
    seen: dict[str, str] = {}
    next_sample = started
    try:
        descriptor = _owned_pidfd(identity, command, worker_record)
        while True:
            now = time.monotonic()
            claims.refresh(child_pid=child.pid)
            previously_observed = len(result["phase_observations"])
            if notifications_path is not None:
                _consume(notifications_path, contract, digest, progress, seen, now, result["phase_observations"])
            for event in result["phase_observations"][previously_observed:]:
                claims.register_identity(event["producer_identity"], role=f"rank:{event['rank']}")
            returncode = child.poll()
            current = None
            if returncode is None:
                try:
                    current = _observe(identity, command, worker_record)
                except (OSError, ValueError):
                    # Linux may clear cmdline just before the terminal stat appears.
                    # Only this exact Popen child's reaped exit can resolve that race.
                    from scripts.onestep_avatar.execution.queue import process_identity  # noqa: PLC0415
                    latest = process_identity(child.pid)
                    empty_owned_command = (latest is not None and not latest["command"]
                                           and all(latest[key] == identity[key] for key in ("pid", "start_ticks")))
                    if _pidfd_terminal(descriptor) or empty_owned_command:
                        returncode = child.wait(timeout=shutdown_seconds)
                    else:
                        time.sleep(poll_seconds)
                        returncode = child.poll()
                    if returncode is None:
                        raise
                if current is not None and current["terminal"] and returncode is None:
                    returncode = child.wait(timeout=shutdown_seconds)
            if returncode is not None:
                result["returncode"] = returncode
                if returncode != 0:
                    raise ValueError(f"registered child exited with status {returncode}")
                if any(item["index"] != len(contract["phases"]) for item in progress.values()):
                    raise ValueError("child exited without complete rank/phase notifications")
                inventory = _worker_inventory(claims, inventory_timeout_seconds)
                if not inventory["complete"] or inventory["workers_live"]:
                    raise ValueError("child exited with live or unproved own descendants")
                break
            if current is None or current["terminal"]:
                identity_uncertain = True
                raise ValueError("registered child disappeared before Popen terminal observation")
            if overall_seconds is not None and now - started > overall_seconds:
                raise TimeoutError("explicit overall command deadline exceeded")
            for rank, item in progress.items():
                if item["active"] is not None and now - item["active"] > phase_seconds:
                    raise TimeoutError(f"rank {rank} phase {contract['phases'][item['index']]} deadline exceeded")
                if (item["active"] is None and item["index"] < len(contract["phases"])
                        and now - item["awaiting_since"] > startup_seconds):
                    phase = "startup" if item["index"] == 0 else "notification gap"
                    raise TimeoutError(f"rank {rank} {phase} deadline exceeded")
            if sampled_total_device_limit_bytes is not None and now >= next_sample:
                memory = _gpu_inventory(gpus, inventory_timeout_seconds)
                result["sampled_total_device_bytes"].append({"elapsed_s": now - started,
                                                            "gpus": {str(gpu): value for gpu, value in memory.items()}})
                if any(value > sampled_total_device_limit_bytes for value in memory.values()):
                    raise ValueError("sampled total-device stop guard exceeded")
                next_sample = time.monotonic() + sample_seconds
            time.sleep(poll_seconds)
        result["state"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["events"].append({"event": "failure", "elapsed_s": time.monotonic() - started,
                                  "error": result["error"]})
        if descriptor is None:
            identity_uncertain = True
        identity_uncertain |= _stop_owned(
            child, identity, command, worker_record, claims, descriptor,
            grace=shutdown_seconds, inventory_timeout=inventory_timeout_seconds,
            poll_seconds=poll_seconds, result=result, started=started,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        result["returncode"] = child.poll()
        try:
            inventory = _worker_inventory(claims, inventory_timeout_seconds)
            result["worker_inventory"] = inventory
            result["workers_absent"] = (
                result["returncode"] is not None and not identity_uncertain and inventory["complete"]
                and not inventory["workers_live"]
            )
            if not result["workers_absent"]:
                result["state"] = "failed"
                result["error"] = result["error"] or "child terminal ownership or worker absence is unproven"
        except Exception as inventory_error:
            result["state"] = "failed"
            result["error"] = result["error"] or f"worker inventory failed: {type(inventory_error).__name__}"
            result["events"].append({"event": "worker_inventory_failed", "error": str(inventory_error)})
        result["elapsed_s"] = time.monotonic() - started
        result["notifications"] = sorted(seen)
        _publish(evidence_path, result)
    return result
