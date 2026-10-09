"""Review or dispatch saved package jobs with shared GPU claims and verified receipts."""

from __future__ import annotations

import fcntl
import importlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_PROTOCOL

ALLOWED_GPUS = frozenset(range(6))
TRAIN_GPUS = (0, 1, 2, 3)
EVALUATION_PREFERENCE = (5, 4, 3, 2, 1, 0)

# Fixed reviewed selectors; ordinary kinds never consult this table.
EXPERIMENTS = {"sigma_sweep": "scripts.onestep_avatar.experiments.sigma_sweep"}


def experiment_entry(selector: str) -> str:
    """Resolve a fixed experiment selector without importing or discovering code."""
    if not isinstance(selector, str) or selector not in EXPERIMENTS:
        raise ValueError("queue experiment selector is unknown")
    return EXPERIMENTS[selector]


def validate_completion_descriptor(descriptor: dict) -> None:
    """Require model-free evidence paths as data, before normalization or imports."""
    if (not isinstance(descriptor, dict) or not descriptor
            or set(descriptor) - {"manifest", "records"}
            or ("manifest" in descriptor and (not isinstance(descriptor["manifest"], str)
                or not descriptor["manifest"]))
            or ("records" in descriptor and (not isinstance(descriptor["records"], list)
                or not descriptor["records"] or any(not isinstance(p, str) or not p
                                                   for p in descriptor["records"])))):
        raise ValueError("queue experiment parser completion descriptor is malformed")


def validate_experiment_job(job: dict) -> None:
    """Check the selector and pinned spec as data before an owner can be imported."""
    experiment_entry(job.get("experiment"))
    validate_completion_descriptor(job.get("completion"))
    spec = job.get("spec")
    digest = job.get("spec_sha256")
    if (not isinstance(spec, str) or not spec
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("queue experiment requires a specification path and SHA-256")
    arguments = job["arguments"]
    if arguments.count("--spec") != 1:
        raise ValueError("queue experiment requires one explicit specification")
    index = arguments.index("--spec")
    if index + 1 >= len(arguments) or arguments[index + 1] != spec:
        raise ValueError("queue experiment specification differs from command")



class QueueChildError(ValueError):
    """A reaped nonzero child; validation and ownership errors use other failures."""


def _read_process_environment(entry: Path, start_ticks: str) -> bytes | None:
    """Resolve brief exit-time denial only by readable or confirmed terminal handles."""
    for attempt in range(10):
        try:
            return (entry / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            return None
        except PermissionError:
            try:
                raw = (entry / "stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                return None
            fields = raw[raw.rfind(")") + 1:].split()
            if len(fields) < 20 or fields[19] != start_ticks:
                raise
            if fields[0] in ("Z", "X", "x"):
                return None
            if attempt == 9:
                raise
            time.sleep(0.01)
    raise AssertionError("bounded environment observation exhausted without a decision")


def live_attempt_processes(  # noqa: PLR0912 -- stable handle and token observation gates
    token: str, *, started_ticks: int | None = None, proc_root: Path = Path("/proc")
) -> list[int]:
    """Find token-bound workers even when Torch Elastic gives ranks new sessions."""
    from scripts.onestep_avatar.execution.queue_protocol import TOKEN_ENV  # noqa: PLC0415

    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("worker inspection requires an exact attempt token")
    if started_ticks is not None and (type(started_ticks) is not int or started_ticks < 0):
        raise ValueError("worker inspection requires valid attempt start ticks")
    marker = os.fsencode(TOKEN_ENV + "=" + token)
    live = set()
    for _ in range(2):
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.geteuid():
                    continue
                raw = (entry / "stat").read_text()
                fields = raw[raw.rfind(")") + 1:].split()
                if len(fields) < 20:
                    raise ValueError("malformed process stat during worker inspection")
                if started_ticks is not None and int(fields[19]) < started_ticks:
                    continue  # Older processes cannot inherit this newly created token.
                if fields[0] in ("Z", "X", "x"):
                    continue
                raw_environment = _read_process_environment(entry, fields[19])
                if raw_environment is None:
                    continue
                environment = raw_environment.split(b"\0")
                if marker not in environment:
                    continue
                after = (entry / "stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            final = after[after.rfind(")") + 1:].split()
            if len(final) < 20 or final[19] != fields[19]:
                raise ValueError("worker handle changed during inspection")
            if final[0] not in ("Z", "X", "x"):
                live.add(int(entry.name))
        if live:
            break
    return sorted(live)


def owned_workers_live(row: dict) -> bool:
    """A session leader is insufficient to prove distributed child termination."""
    from scripts.onestep_avatar.execution.queue_protocol import (  # noqa: PLC0415 -- canonical environment field
        TOKEN_ENV,
    )

    token = row.get("environment_changes", {}).get(TOKEN_ENV)
    if row.get("process_ledger") is not None:
        from scripts.onestep_avatar.execution.process_registry import (  # noqa: PLC0415 -- targeted own tree
            inspect_record,
        )
        observed = inspect_record(Path(row["process_ledger"]), token)
        return not observed["complete"] or observed["workers_live"]
    return bool(
        (row.get("child_session") is not None and live_session_processes(row["child_session"]))
        or (token is not None and live_attempt_processes(token, started_ticks=row.get("attempt_started_ticks")))
    )


def live_session_processes(session_id: int, *, proc_root: Path = Path("/proc")) -> list[int]:
    """Find live members of an owned Linux session; unreadable handles fail closed."""
    if type(session_id) is not int or session_id <= 0:
        raise ValueError("child session requires a positive ID")
    live = set()
    # Reobserve an empty session rather than infer termination from its leader.
    for _ in range(2):
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            fields = raw[raw.rfind(")") + 1:].split()
            if len(fields) < 20:
                raise ValueError("malformed process stat during child-session inspection")
            if int(fields[3]) == session_id and fields[0] not in ("Z", "X", "x"):
                live.add(int(entry.name))
        if live:
            break
    return sorted(live)


def process_identity(pid: int, *, proc_root: Path = Path("/proc")) -> dict | None:
    """Read a stable Linux process handle; I/O denial never establishes death."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("process identity requires a positive PID")
    root = proc_root / str(pid)

    def read_stat() -> tuple[str, int]:
        raw = (root / "stat").read_text()
        closing = raw.rfind(")")
        fields = raw[closing + 1 :].split()
        if closing < 0 or len(fields) < 20:
            raise ValueError("malformed process stat record")
        return fields[0], int(fields[19])

    try:
        before = read_stat()
        command = (root / "cmdline").read_bytes()
        after = read_stat()
    except FileNotFoundError:
        return None
    if before[1] != after[1]:
        raise ValueError("process identity changed during inspection")
    return {
        "pid": pid,
        "start_ticks": after[1],
        "command": [os.fsdecode(value) for value in command.rstrip(b"\0").split(b"\0")] if command else [],
        "terminal": after[0] in ("Z", "X", "x"),
    }


def process_alive(pid: int | None) -> bool:
    """Probe the recorded process; permission denial means it remains live."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        identity = process_identity(pid)
    except (OSError, ValueError):
        return True  # An unreadable handle is not proof of death.
    return not (identity is None or identity["terminal"])


def inspect_child(row: dict) -> str:
    """Classify a saved child without adopting a reused or unidentified live PID."""
    pid = row.get("child_pid")
    if pid is None:
        from scripts.onestep_avatar.execution.queue_launch import inspect_unapproved_launch  # noqa: PLC0415

        inspect_unapproved_launch(row)
        return "unapproved"
    saved = row.get("child_identity")
    if saved is None and row.get("launch_protocol") == LAUNCH_PROTOCOL:
        from scripts.onestep_avatar.execution.queue_launch import inspect_unapproved_launch  # noqa: PLC0415

        inspect_unapproved_launch(row)
        return "unapproved"
    current = process_identity(pid)
    if current is None:
        if owned_workers_live(row):
            return "live"
        return "missing"
    if not isinstance(saved, dict) or any(saved.get(key) != current[key] for key in ("pid", "start_ticks")):
        raise ValueError("queue child identity differs; recovery is required")
    if not isinstance(saved.get("command"), list) or not saved["command"] or any(
        not isinstance(value, str) for value in saved["command"]
    ):
        raise ValueError("queue child identity differs; recovery is required")
    # Linux clears cmdline after exit while an unreaped zombie retains its PID/ticks.
    if current["command"] != saved.get("command") and not (current["terminal"] and current["command"] == []):
        from scripts.onestep_avatar.execution.queue_launch import verify_command_transition  # noqa: PLC0415

        verify_command_transition(row, current)
    if owned_workers_live(row):
        return "live"
    return "terminal" if current["terminal"] else "live"


def parse_gpu_memory(output: str) -> dict[int, int]:
    """Reject incomplete/malformed inventory rather than assuming a free card."""
    result = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 2:
            raise ValueError("malformed GPU memory inventory")
        try:
            gpu, memory = (int(field.strip()) for field in fields)
        except ValueError as error:
            raise ValueError("malformed GPU memory inventory") from error
        if gpu < 0 or memory < 0 or gpu in result:
            raise ValueError("invalid or duplicate GPU memory inventory")
        result[gpu] = memory
    if not ALLOWED_GPUS.issubset(result):
        raise ValueError("GPU inventory omits an allowed device")
    return result


class GPUClaims:
    """Serialize reservations and remove only claims carrying this attempt's token."""

    def __init__(self, directory: Path, *, ttl: float = 600):
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("claim TTL must be positive")
        self.directory = directory
        self.ttl = ttl
        self.token = uuid.uuid4().hex
        self.started_ticks = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))
        self.owned: set[int] = set()

    @contextmanager
    def _lock(self):  # noqa: ANN202 -- private context manager
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".claims.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self, gpu: int) -> tuple[dict | None, float] | None:
        path = self.directory / str(gpu)
        try:
            # Use one open inode for both content and age.
            with path.open() as handle:
                stamp = os.fstat(handle.fileno()).st_mtime
                raw = handle.read()
        except FileNotFoundError:
            return None
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            record = None  # Historical text claims use the established TTL.
        return (record if isinstance(record, dict) else None), stamp

    def _active(self, claim: tuple[dict | None, float]) -> bool:
        record, stamp = claim
        if time.time() - stamp < self.ttl:
            return True
        return (
            record is not None and (
                process_alive(record.get("owner_pid")) or process_alive(record.get("child_pid"))
                or (record.get("child_session") is not None and bool(live_session_processes(record["child_session"])))
                or (record.get("token") is not None and bool(live_attempt_processes(
                    record["token"], started_ticks=record.get("attempt_started_ticks")
                )))
            )
        )

    def choose(self, memory: dict[int, int], *, training: bool) -> tuple[int, ...] | None:
        """Choose only eligible unclaimed cards; acquisition still rechecks claims."""
        if not ALLOWED_GPUS.issubset(memory) or any(value < 0 for value in memory.values()):
            raise ValueError("incomplete or invalid GPU memory inventory")
        free = set()
        with self._lock():
            for gpu in ALLOWED_GPUS:
                claim = self._read(gpu)
                if memory[gpu] < 1024 and (claim is None or not self._active(claim)):
                    free.add(gpu)
        if training:
            return TRAIN_GPUS if set(TRAIN_GPUS).issubset(free) else None
        return next(((gpu,) for gpu in EVALUATION_PREFERENCE if gpu in free), None)

    def acquire(self, gpus: tuple[int, ...], *, job: str) -> bool:
        """Reserve every requested GPU exclusively, or roll back this attempt."""
        import tempfile  # noqa: PLC0415 -- complete exclusive claim publication

        if not gpus or len(set(gpus)) != len(gpus) or not set(gpus).issubset(ALLOWED_GPUS) or self.owned:
            raise ValueError("claim request must name unique allowed GPUs on a fresh attempt")
        record = {"token": self.token, "owner_pid": os.getpid(), "child_pid": None, "job": job,
                  "attempt_started_ticks": self.started_ticks}
        with self._lock():
            try:
                for gpu in gpus:
                    path = self.directory / str(gpu)
                    claim = self._read(gpu)
                    if claim is not None and self._active(claim):
                        self._release_owned()
                        return False
                    if claim is not None:
                        path.unlink(missing_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, prefix=".claim-") as handle:
                        json.dump(record, handle)
                        handle.flush()
                        self.owned.add(gpu)
                        try:
                            os.link(handle.name, path)
                        except FileExistsError:
                            self._release_owned()
                            return False
            except BaseException:
                self._release_owned()
                raise
        return True

    def refresh(self, *, child_pid: int | None = None) -> None:
        """Refresh only token-matched claims; loss of ownership fails visibly."""
        with self._lock():
            for gpu in self.owned:
                claim = self._read(gpu)
                if claim is None or claim[0] is None or claim[0].get("token") != self.token:
                    raise ValueError("GPU reservation ownership changed")
                record = {**claim[0], "child_pid": claim[0].get("child_pid") if child_pid is None else child_pid}
                if child_pid is not None:
                    record["child_session"] = child_pid
                path = self.directory / str(gpu)
                # Update through the open inode; never replace a foreign path.
                with path.open("r+") as handle:
                    current = json.load(handle)
                    if current.get("token") != self.token:
                        raise ValueError("GPU reservation ownership changed")
                    handle.seek(0)
                    json.dump(record, handle)
                    handle.truncate()
                    os.utime(handle.fileno(), None)

    def _release_owned(self) -> None:
        for gpu in self.owned:
            claim = self._read(gpu)
            if claim is not None and claim[0] is not None and claim[0].get("token") == self.token:
                (self.directory / str(gpu)).unlink(missing_ok=True)
        self.owned.clear()

    def release(self) -> None:
        """Release this attempt; preserve reservations replaced by another owner."""
        with self._lock():
            for gpu in self.owned:
                claim = self._read(gpu)
                if (
                    claim is not None
                    and claim[0] is not None
                    and claim[0].get("token") == self.token
                    and (process_alive(claim[0].get("child_pid")) or (
                        claim[0].get("child_session") is not None
                        and live_session_processes(claim[0]["child_session"])
                    ) or (claim[0].get("token") is not None and live_attempt_processes(
                        claim[0]["token"], started_ticks=claim[0].get("attempt_started_ticks")
                    )))
                ):
                    raise ValueError("cannot release a GPU used by a live child")
            self._release_owned()


def validate_job_list(record: dict) -> list[dict]:  # noqa: PLR0912 -- ordered saved-job schema gates
    """Check explicit package jobs and dependency order without touching inputs."""
    if record.get("schema_version") != 1 or not isinstance(record.get("jobs"), list):
        raise ValueError("queue requires a version-one job list")
    jobs, identifiers, outputs = [], set(), set()
    for job in record["jobs"]:
        if not isinstance(job, dict):
            raise ValueError("queue job must be a record")
        identifier = job.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("queue job IDs must be unique nonempty strings")
        if job.get("kind") not in ("train", "evaluate", "decode", "render", "experiment"):
            raise ValueError("queue job must select a supported package owner")
        arguments = job.get("arguments")
        if not isinstance(arguments, list) or any(not isinstance(value, str) for value in arguments):
            raise ValueError("queue arguments must be strings")
        if job["kind"] == "experiment":
            validate_experiment_job(job)
        if job["kind"] in ("train", "evaluate"):
            if arguments.count("--mode") != 1:
                raise ValueError("queue model jobs require one explicit mode")
            index = arguments.index("--mode")
            if index + 1 >= len(arguments) or arguments[index + 1] not in ("bidirectional", "causal"):
                raise ValueError("queue model jobs require a supported explicit mode")
        if job["kind"] == "render" and sum(
            value.split("=", 1)[0] == "--render-saved-comparisons" for value in arguments
        ) != 1:
            raise ValueError("queue render jobs require one explicit specification")
        output = job.get("output")
        if not isinstance(output, str) or not output or output in outputs:
            raise ValueError("queue outputs must be unique nonempty paths")
        if arguments.count("--output") != 1:
            raise ValueError("queue job requires one explicit output argument")
        index = arguments.index("--output")
        if index + 1 >= len(arguments) or arguments[index + 1] != output:
            raise ValueError("queue recorded output differs from command")
        dependencies = job.get("dependencies", [])
        if not isinstance(dependencies, list) or any(not isinstance(value, str) for value in dependencies):
            raise ValueError("queue dependencies must be job IDs")
        if len(set(dependencies)) != len(dependencies) or not set(dependencies).issubset(identifiers):
            raise ValueError("queue dependencies must name unique earlier jobs")
        if not isinstance(job.get("completion"), dict) or not job["completion"]:
            raise ValueError("queue completion evidence is required")
        if job["kind"] == "train":
            if job.get("processes") != 4 or not isinstance(job.get("accelerate_config"), str):
                raise ValueError("queued training requires four processes and a recorded Accelerate config")
            port = job.get("port")
            if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
                raise ValueError("queue training port is invalid")
        identifiers.add(identifier)
        outputs.add(output)
        jobs.append(dict(job))
    return jobs


def prepare_job(raw: dict, root: Path) -> dict:  # noqa: PLR0912, PLR0915 -- shared exact launch normalization
    """Resolve one job through the same parser, defaults and hash used by the queue.

    Dependencies need list-level ordering checks in prepare_jobs. Here their
    strings remain part of the identity, without requiring the complete list.
    A supplied prepared digest is a claim to check, never the hash authority.
    """
    import hashlib  # noqa: PLC0415 -- canonical job identity
    from dataclasses import asdict, is_dataclass  # noqa: PLC0415 -- owner defaults

    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- original launch bytes

    if not isinstance(raw, dict):
        raise ValueError("queue job must be a record")
    dependencies = raw.get("dependencies", [])
    if (not isinstance(dependencies, list) or any(not isinstance(value, str) for value in dependencies)
            or len(set(dependencies)) != len(dependencies) or raw.get("id") in dependencies):
        raise ValueError("queue dependencies must name unique earlier jobs")
    checked = validate_job_list({"schema_version": 1, "jobs": [{**raw, "dependencies": []}]})[0]
    if "dependencies" in raw:
        checked["dependencies"] = dependencies
    else:
        checked.pop("dependencies", None)
    job = checked
    claimed_hash = job.pop("sha256", None)
    root = root.resolve()
    arguments = list(job["arguments"])
    forbidden = {"--gpu-id", "--dry-run", "--help", "--overwrite", "--preview-job", "--saved-metrics", "--verify"}
    if any(value.split("=", 1)[0] in forbidden for value in arguments):
        raise ValueError("queue command contains a forbidden execution override")
    if job["kind"] == "train":
        from scripts.onestep_avatar.training.config import parse_settings as parser  # noqa: PLC0415
    elif job["kind"] == "decode":
        from scripts.onestep_avatar.decode_saved import parse_args as parser  # noqa: PLC0415
    elif job["kind"] == "experiment":
        selected = importlib.import_module(experiment_entry(job["experiment"]))
        parser = selected.parse_args
    else:
        from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- selected ordinary owner
        from scripts.onestep_avatar import comparisons  # noqa: PLC0415 -- render owner is selected lazily

        parser = comparisons.parse_saved_comparison_args if job["kind"] == "render" else evaluate.parse_args
    try:
        parsed = parser(arguments)
    except SystemExit as error:
        raise ValueError(f"invalid package arguments for job {job['id']}") from error
    fields = vars(parsed)
    for field, value in fields.items():
        values = [value] if isinstance(value, Path) else value if isinstance(value, list) else []
        if not values or not all(isinstance(item, Path) for item in values):
            continue
        flag = "--" + field.replace("_", "-")
        for index, argument in enumerate(arguments):
            if argument == flag:
                arguments[index + 1] = str((root / arguments[index + 1]).resolve())
            elif argument.startswith(flag + "="):
                arguments[index] = flag + "=" + str((root / argument.split("=", 1)[1]).resolve())
    parsed = parser(arguments)
    output = str(parsed.output.resolve())
    job.update(arguments=arguments, output=output)
    completion = dict(job["completion"])
    if "records" in completion:
        completion["records"] = [str((root / value).resolve()) for value in completion["records"]]
    if "checkpoint" in completion:
        completion["checkpoint"] = str((root / completion["checkpoint"]).resolve())
    if "manifest" in completion:
        completion["manifest"] = str((root / completion["manifest"]).resolve())
    job["completion"] = completion
    if job["kind"] == "train":
        job["accelerate_config"] = str((root / job["accelerate_config"]).resolve())
        config_path = Path(job["accelerate_config"])
        if not config_path.is_file():
            raise ValueError("queue Accelerate configuration is missing")
        digest = sha256(config_path)
        if job.get("accelerate_config_sha256", digest) != digest:
            raise ValueError("queue Accelerate configuration changed from its claimed hash")
        job["accelerate_config_sha256"] = digest
        if parsed.resource_budget is not None:
            if not parsed.resource_budget.is_file():
                raise ValueError("queue resource budget is missing")
            digest = sha256(parsed.resource_budget)
            if job.get("resource_budget_sha256", digest) != digest:
                raise ValueError("queue resource budget changed from its claimed hash")
            job["resource_budget_sha256"] = digest
        elif "resource_budget_sha256" in job:
            raise ValueError("queue resource budget hash requires an explicit budget")
        if type(completion.get("step")) is not int or completion["step"] != parsed.steps:
            raise ValueError("queue final checkpoint step differs from training settings")
    if job["kind"] == "decode":
        if not parsed.jobs.is_file():
            raise ValueError("queue decoder job list is missing")
        job["decoder_jobs"] = str(parsed.jobs.resolve())
        job["decoder_jobs_sha256"] = sha256(parsed.jobs)
    if job["kind"] == "render":
        if not parsed.render_saved_comparisons.is_file():
            raise ValueError("queue saved-comparison specification is missing")
        job["render_spec"] = str(parsed.render_saved_comparisons.resolve())
        job["render_spec_sha256"] = sha256(parsed.render_saved_comparisons)
        if completion.get("manifest") != str(Path(output) / "render_manifest.json"):
            raise ValueError("queue saved-comparison completion manifest differs from output")
    if job["kind"] == "experiment":
        spec = (root / job["spec"]).resolve()
        if parsed.spec.resolve() != spec:
            raise ValueError("queue experiment specification differs from parsed arguments")
        if not spec.is_file():
            raise ValueError("queue experiment specification is missing")
        if sha256(spec) != job["spec_sha256"]:
            raise ValueError("queue experiment specification changed from its claimed hash")
        job["spec"] = str(spec)
        descriptor = getattr(parsed, "completion", None)
        validate_completion_descriptor(descriptor)
        if completion != descriptor:
            raise ValueError("queue experiment completion manifest differs from parsed output")
        evidence = ([descriptor["manifest"]] if "manifest" in descriptor else []) + descriptor.get("records", [])
        if any(not Path(p).resolve().is_relative_to(Path(output)) for p in evidence):
            raise ValueError("queue experiment completion descriptor escapes its output directory")
    defaults = asdict(parsed) if is_dataclass(parsed) else vars(parsed)

    def encode(value):  # noqa: ANN001, ANN202 -- canonical JSON path/dataclass conversion
        if isinstance(value, Path):
            return str(value.resolve())
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"unsupported canonical job value: {type(value)}")

    canonical = json.dumps(
        {"job": job, "resolved_settings": defaults},
        sort_keys=True,
        separators=(",", ":"),
        default=encode,
        allow_nan=False,
    )
    job["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    if claimed_hash is not None and claimed_hash != job["sha256"]:
        raise ValueError("queue prepared job identity differs from reconstructed launch")
    return job


def prepare_jobs(path: Path) -> list[dict]:
    """Check list topology, then reuse the exact public single-job authority."""
    raw = validate_job_list(json.loads(path.read_text()))
    jobs = [prepare_job(job, path.resolve().parent) for job in raw]
    outputs = [job["output"] for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("queue output paths alias another job")
    return jobs


def training_launch_record(job: dict, gpus: tuple[int, ...] = TRAIN_GPUS, *, schema_version: int = 2) -> dict:
    """Capture original requested launch facts before dispatch, with exact YAML bytes."""
    import base64  # noqa: PLC0415 -- lossless original configuration snapshot
    import hashlib  # noqa: PLC0415 -- validate snapshot against canonical pinned hash

    from scripts.onestep_avatar.training import numerics  # noqa: PLC0415 -- import-light numerical launch owner

    prepared = prepare_job(job, Path.cwd())
    if prepared["kind"] != "train":
        raise ValueError("training launch binding requires a training job")
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError("training launch binding has an unsupported schema")
    command, _environment = job_command(prepared, gpus)
    original = Path(prepared["accelerate_config"]).read_bytes()
    if hashlib.sha256(original).hexdigest() != prepared["accelerate_config_sha256"]:
        raise ValueError("queue Accelerate configuration changed during launch snapshot")
    record = {"schema_version": schema_version, "job": prepared, "command": command, "gpus": list(gpus),
            "accelerate_config_bytes_base64": base64.b64encode(original).decode("ascii"),
            "accelerate_config_sha256": prepared["accelerate_config_sha256"]}
    if schema_version == 2:
        record["numerical_environment"] = dict(numerics.ENVIRONMENT)
    return record


def verify_training_launch(record: dict, job: dict) -> dict:
    """Reconstruct canonical identity and compare exact original launch bytes/command."""
    import base64  # noqa: PLC0415 -- checked original bytes
    import binascii  # noqa: PLC0415 -- malformed original snapshot failure
    import hashlib  # noqa: PLC0415 -- original-byte integrity

    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record.get("schema_version") not in (1, 2)):
        raise ValueError("original training launch evidence is missing or malformed")
    prepared = prepare_job(job, Path.cwd())
    expected = training_launch_record(prepared, TRAIN_GPUS, schema_version=record["schema_version"])
    try:
        original = base64.b64decode(record["accelerate_config_bytes_base64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as error:
        raise ValueError("original Accelerate snapshot is malformed") from error
    if (hashlib.sha256(original).hexdigest() != record.get("accelerate_config_sha256")
            or record != expected):
        raise ValueError("original training launch identity, command or Accelerate bytes differ")
    return prepared


def read_training_launch(path: Path) -> dict:
    """Read and verify dispatch evidence; do not infer missing original launch facts."""
    record = json.loads(path.read_text())
    verify_training_launch(record, record.get("job", {}))
    return record

def job_command(job: dict, gpus: tuple[int, ...]) -> tuple[list[str], dict[str, str]]:  # noqa: PLR0912 -- explicit owner and byte gates
    """Construct package-only child commands and physical-device environment changes."""
    import sys  # noqa: PLC0415 -- current conda Python owns all execution

    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus).issubset(ALLOWED_GPUS):
        raise ValueError("job requires unique allowed physical GPUs")
    kind = job["kind"]
    environment = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    if kind == "train":
        from scripts.onestep_avatar.training import numerics  # noqa: PLC0415 -- import-light launch policy
        environment.update(numerics.ENVIRONMENT)
        if gpus != TRAIN_GPUS or job["processes"] != 4:
            raise ValueError("queued training requires GPUs 0-3 and four processes")
        from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- pinned launch bytes

        config_path = Path(job["accelerate_config"])
        if not config_path.is_file() or sha256(config_path) != job.get("accelerate_config_sha256"):
            raise ValueError("queue Accelerate configuration changed after preparation")
        from scripts.onestep_avatar.training.config import parse_settings  # noqa: PLC0415 -- typed budget path

        budget_path = parse_settings(job["arguments"]).resource_budget
        if budget_path is not None:
            if not budget_path.is_file() or sha256(budget_path) != job.get("resource_budget_sha256"):
                raise ValueError("queue resource budget changed after preparation")
        elif "resource_budget_sha256" in job:
            raise ValueError("queue resource budget hash requires an explicit budget")
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            job["accelerate_config"],
            "--num_processes",
            "4",
            "--main_process_port",
            str(job["port"]),
            "-m",
            "scripts.onestep_avatar.train",
            *job["arguments"],
        ]
    else:
        if kind not in ("evaluate", "decode", "render", "experiment") or len(gpus) != 1:
            raise ValueError("queued evaluation/decoding requires one GPU and a package owner")
        module = "scripts.onestep_avatar.decode_saved" if kind == "decode" else "scripts.onestep_avatar.evaluate"
        if kind == "render":
            module = "scripts.onestep_avatar.comparisons"
        if kind == "experiment":
            module = experiment_entry(job["experiment"])
            from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- pinned experiment specification

            spec_path = Path(job["spec"])
            if not spec_path.is_file() or sha256(spec_path) != job["spec_sha256"]:
                raise ValueError("queue experiment specification changed after preparation")
        if kind == "decode":
            from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- pinned decoder list

            jobs_path = Path(job.get("decoder_jobs", ""))
            if not jobs_path.is_file() or sha256(jobs_path) != job.get("decoder_jobs_sha256"):
                raise ValueError("queue decoder job list changed after preparation")
        if kind == "render":
            from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415

            spec_path = Path(job["render_spec"])
            if not spec_path.is_file() or sha256(spec_path) != job["render_spec_sha256"]:
                raise ValueError("queue saved-comparison specification changed after preparation")
        command = [sys.executable, "-m", module, *job["arguments"], "--gpu-id", "0"]
    return command, environment


def read_training_marker(checkpoint: Path, step: int) -> dict:
    """Validate checkpoint readiness through one shared model-free authority."""
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- actual published bytes

    checkpoint = checkpoint.resolve()
    marker = json.loads(checkpoint.with_suffix(".complete.json").read_text())
    if (
        not isinstance(marker, dict)
        or type(marker.get("schema_version")) is not int
        or marker.get("schema_version") != 2
        or marker.get("state") != "complete"
        or type(step) is not int
        or step < 0
        or type(marker.get("step")) is not int
        or marker.get("step") != step
        or not isinstance(marker.get("path"), str)
        or Path(marker["path"]).resolve() != checkpoint
        or marker.get("sha256") != sha256(checkpoint)
    ):
        raise ValueError("queue checkpoint completion marker is invalid")
    return marker


def verify_completion(job: dict) -> bool:  # noqa: PLR0911, PLR0912, PLR0915 -- artifact-specific completion gates
    """Verify saved training/evaluation evidence; missing artifacts remain pending."""
    import torch  # noqa: PLC0415 -- saved tensor verification only

    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- artifact identities
    from scripts.onestep_avatar.training import checkpoints  # noqa: PLC0415 -- shared adapter checks

    root = Path(job["output"]).resolve()

    def evidence_path(value: str) -> Path:
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("queue completion evidence escapes its output directory")
        return path

    completion = job["completion"]
    if job["kind"] == "train":
        checkpoint = evidence_path(completion["checkpoint"])
        marker_path = checkpoint.with_suffix(".complete.json")
        if not checkpoint.is_file() or not marker_path.is_file():
            return False
        read_training_marker(checkpoint, completion["step"])
        contract = checkpoints.read_contract(checkpoint)
        checkpoints.validate_adapter_tensors(checkpoint, contract)
        mode = job["arguments"][job["arguments"].index("--mode") + 1]
        if contract["mode"] != mode:
            raise ValueError("queue checkpoint contract has the wrong mode")
        if contract["adapter"]["step"] != completion["step"]:
            raise ValueError("queue checkpoint contract has the wrong final step")
        from scripts.onestep_avatar.training import engine  # noqa: PLC0415 -- typed scientific verification owner

        engine.verify_training_conditions(job, checkpoint, contract)
        return True
    if job["kind"] == "decode":
        return verify_decoder_completion(job)
    if job["kind"] == "experiment":
        selected = importlib.import_module(experiment_entry(job["experiment"]))
        spec = Path(job["spec"])
        if not spec.is_file() or sha256(spec) != job["spec_sha256"]:
            raise ValueError("queue experiment specification changed after preparation")
        evidence = ([completion["manifest"]] if "manifest" in completion else []) + completion.get("records", [])
        if any(not evidence_path(value).is_file() for value in evidence):
            return False
        selected.verify_completion(spec, root)
        return True
    if job["kind"] == "render":
        from scripts.onestep_avatar import comparisons  # noqa: PLC0415 -- canonical rendering verifier

        spec = Path(job["render_spec"])
        if not spec.is_file() or sha256(spec) != job["render_spec_sha256"]:
            raise ValueError("queue saved-comparison specification changed after preparation")
        args = comparisons.parse_saved_comparison_args(job["arguments"])
        return comparisons.verify_saved_comparison_completion(spec, root, seed=args.seed)
    if job["kind"] != "evaluate":
        raise ValueError("unsupported completion owner")
    records = completion.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("queue evaluation completion requires result records")
    mode = job["arguments"][job["arguments"].index("--mode") + 1]
    for value in records:
        path = evidence_path(value)
        if not path.is_file():
            return False
        record = json.loads(path.read_text())
        if record.get("state") != "complete" or record.get("mode") != mode:
            raise ValueError("queue evaluation result is incomplete or has the wrong mode")
        output = record["output"]
        encoded = evidence_path(output["path"])
        if not encoded.is_file():
            return False
        if sha256(encoded) != output["sha256"]:
            raise ValueError("queue evaluation encoding content changed")
        tensor = torch.load(encoded, map_location="cpu", weights_only=True)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 5
            or tensor.shape[0] != 1
            or list(tensor.shape) != output["shape"]
            or not tensor.is_floating_point()
            or any(size < 1 for size in tensor.shape)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("queue evaluation encoding shape/content is invalid")
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- canonical scientific evidence owner

    evaluate.verify_evaluation_conditions(job["arguments"], [Path(value) for value in records])
    return True


def read_queue_state(path: Path, jobs: list[dict]) -> dict:  # noqa: PLR0912 -- validate persisted lifecycle fields together
    """Validate append-only job identities without mutating state or claiming ownership."""
    state = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "job_order": [], "jobs": {}}
    if not isinstance(state, dict) or state.get("schema_version") != 1 or not isinstance(state.get("jobs"), dict):
        raise ValueError("invalid queue state schema")
    owner = state.get("owner_pid")
    if owner is not None and (type(owner) is not int or owner <= 0):
        raise ValueError("invalid queue owner PID")
    order = state.get("job_order")
    identifiers = [job["id"] for job in jobs]
    if not isinstance(order, list) or order != identifiers[: len(order)]:
        raise ValueError("queue revisions may only append jobs")
    if set(state["jobs"]) != set(order):
        raise ValueError("queue state job inventory differs from its order")
    for job in jobs:
        prior = state["jobs"].get(job["id"])
        if job["id"] in state["jobs"]:
            if not isinstance(prior, dict):
                raise ValueError("invalid saved queue job record")
            attempts = prior.get("attempts")
            if not isinstance(attempts, list) or any(not isinstance(attempt, dict) for attempt in attempts):
                raise ValueError("invalid saved queue attempts")
            child = prior.get("child_pid")
            if child is not None and (type(child) is not int or child <= 0):
                raise ValueError("invalid queue child PID")
            session = prior.get("child_session")
            if session is not None and (type(session) is not int or session <= 0 or session != child):
                raise ValueError("invalid queue child session")
        if prior is not None and prior.get("sha256") != job["sha256"]:
            raise ValueError("saved queue job settings changed")
        if prior is None:
            state["jobs"][job["id"]] = {"sha256": job["sha256"], "state": "pending", "attempts": []}
        elif prior.get("state") not in ("pending", "running", "complete", "failed"):
            raise ValueError("invalid saved queue job state")
    state["job_order"] = identifiers
    return state


@contextmanager
def queue_state(path: Path, jobs: list[dict], *, recover: bool = False):  # noqa: ANN201, PLR0912 -- locked recovery gates
    """Write state atomically only after a successful, exclusively owned transaction."""
    from scripts.onestep_avatar.corpus.dataset import atomic_write  # noqa: PLC0415 -- shared publication primitive

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            state = read_queue_state(path, jobs)
            owner = state.get("owner_pid")
            if owner != os.getpid() and process_alive(owner):
                raise ValueError("queue state belongs to a live owner")
            if owner != os.getpid() or recover:
                running = [job for job in jobs if state["jobs"][job["id"]]["state"] == "running"]
                unapproved = {}
                for job in running:
                    row = state["jobs"][job["id"]]
                    if process_alive(row.get("child_pid")):
                        raise ValueError("queue has a surviving child; recovery is required")
                    if row.get("child_pid") is None or (
                        row.get("child_identity") is None and row.get("launch_protocol")
                    ):
                        from scripts.onestep_avatar.execution.queue_launch import (  # noqa: PLC0415
                            inspect_unapproved_launch,
                        )

                        unapproved[job["id"]] = inspect_unapproved_launch(row)
                    elif inspect_child(row) == "live":
                        raise ValueError("queue has a surviving child; recovery is required")
                if running and not recover:
                    raise ValueError("queue interrupted attempts require explicit recovery")
                for job in running:
                    row = state["jobs"][job["id"]]
                    if job["id"] in unapproved:
                        result = {"state": "failed", "error": "queue owner interrupted before launch approval",
                                  "recovered_at": time.time(), "launch_recovery": unapproved[job["id"]]}
                        row.update(result)
                        row["attempts"][-1].update(result)
                    elif verify_completion(job):
                        row.update(state="complete", receipt=completion_receipt(job), recovered_at=time.time())
                    else:
                        row.update(
                            state="failed",
                            error="queue owner interrupted before verified completion",
                            recovered_at=time.time(),
                        )
            state["owner_pid"] = os.getpid()
            yield state

            def write_state(temporary: Path) -> None:
                with temporary.open("w") as output:
                    json.dump(state, output, indent=2, allow_nan=False)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())

            atomic_write(path, write_state)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_decoder_completion(job: dict) -> bool:  # noqa: PLR0912, PLR0915 -- exact decoder inventories and artifact gates
    """Check the exact saved decoder inventory and all declared media artifacts."""
    import torch  # noqa: PLC0415 -- saved encoding shape, no model session

    from scripts.onestep_avatar import (  # noqa: PLC0415 -- saved decoder and canonical identity owners
        decode_saved,
        media,
    )
    from scripts.onestep_avatar.execution import (  # noqa: PLC0415 -- saved decoder and canonical identity owners
        software,
    )
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- artifact hashes
    from scripts.prune.core import model_registry  # noqa: PLC0415 -- same saved-decoder model selection

    root = Path(job["output"]).resolve()
    manifest_path = Path(job["completion"]["manifest"]).resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("decoder completion manifest escapes output")
    if not manifest_path.is_file():
        return False
    args = decode_saved.parse_args(job["arguments"])
    if "decoder_jobs_sha256" in job and (
        not args.jobs.is_file()
        or str(args.jobs.resolve()) != job["decoder_jobs"]
        or sha256(args.jobs) != job["decoder_jobs_sha256"]
    ):
        raise ValueError("queue decoder job list changed after preparation")
    model = model_registry.resolve(args.model)
    vae_path = Path(model.paths.video_vae()).resolve()
    vae_hash = sha256(vae_path)
    settings = media.native_decoder_settings()
    source_hash = sha256(Path(decode_saved.__file__))
    requested = json.loads(args.jobs.read_text())
    if not requested.get("jobs"):
        raise ValueError("decoder completion requires at least one requested decode")
    manifest = json.loads(manifest_path.read_text())
    for field in ("jobs", "comparisons"):
        expected = requested.get(field, [])
        actual = manifest.get(field, [])
        if (
            not isinstance(actual, list)
            or len({row["id"] for row in expected}) != len(expected)
            or len({row["id"] for row in actual}) != len(actual)
            or {row["id"] for row in actual} != {row["id"] for row in expected}
        ):
            raise ValueError("decoder manifest job/comparison inventory differs")
        indexed = {row["id"]: row for row in actual}
        for request in expected:
            row = indexed[request["id"]]
            software.check_current(row.get('software'))
            if field == "jobs":
                if row.get("input") != request or not isinstance(row.get("frames"), int) or row["frames"] < 1:
                    raise ValueError("decoder manifest source/coverage differs")
                source = Path(request["latent"])
                if not source.is_file():
                    return False
                if sha256(source) != request["sha256"]:
                    raise ValueError("decoded source content changed")
                latent = torch.load(source, map_location="cpu", weights_only=True)
                if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[0] != 1:
                    raise ValueError("decoded source must contain one B,C,F,H,W tensor")
                if request.get("latent_frames"):
                    if type(request["latent_frames"]) is not int or request["latent_frames"] < 1:
                        raise ValueError("decoder requested encoded range is invalid")
                    latent = latent[:, :, : request["latent_frames"]]
                if (
                    not latent.is_floating_point()
                    or latent.shape[1] != model.caps.latent_channels
                    or any(size < 1 for size in latent.shape)
                    or not torch.isfinite(latent).all()
                ):
                    raise ValueError("decoded source has invalid shape/content")
                seed = request.get("seed", args.seed)
                identity = row.get("decoder", {})
                key = media.decode_key(
                    request["sha256"], vae_hash, list(latent.shape), "native_decode_video", seed, settings
                )
                if (
                    identity.get("model") != args.model
                    or identity.get("vae_sha256") != vae_hash
                    or Path(identity.get("vae_path", "")).resolve() != vae_path
                    or identity.get("settings") != settings
                    or identity.get("seed") != seed
                    or row.get("decode_key") != key
                    or row.get("source_code_sha256") != source_hash
                ):
                    raise ValueError("decoder identity differs from current requested runtime")
                frames = (latent.shape[2] - 1) * model.scale_factors.time + 1
                sample_frames = request.get("sample_frames", [0, 32, 48, 60, 64, 65])
                if not isinstance(sample_frames, list) or any(
                    type(frame) is not int or frame < 0 for frame in sample_frames
                ):
                    raise ValueError("decoder requested sample frames are invalid")
                wanted = [frame for frame in sample_frames if frame < frames]
                actual_frames = [sample.get("frame") for sample in row.get("samples", [])]
                if row["frames"] != frames or actual_frames != wanted:
                    raise ValueError("decoder output/sample coverage differs from source")
                artifacts = [{"file": row["video"], "sha256": row["video_sha256"]}, *row.get("samples", [])]
            else:
                if any(row.get(key) != value for key, value in request.items()):
                    raise ValueError("decoder comparison inputs changed")
                artifacts = row.get("samples", [])
            for artifact in artifacts:
                path = (root / artifact["file"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("decoder artifact escapes output")
                if not path.is_file():
                    return False
                if sha256(path) != artifact["sha256"]:
                    raise ValueError("decoder artifact content changed")
            if field == "jobs":
                video = media.verify_saved_video(root / row["video"], frames, request.get("fps", 30))
                if (int(video["height"]), int(video["width"])) != (
                    latent.shape[3] * model.scale_factors.height,
                    latent.shape[4] * model.scale_factors.width,
                ):
                    raise ValueError("decoder video dimensions differ from selected model geometry")
                for sample in row.get("samples", []):
                    media.verify_saved_png(root / sample["file"], int(video["width"]), int(video["height"]))
    return True


def completion_receipt(job: dict) -> dict:
    """Capture verified evidence identities; child lifecycle enforcement is caller-owned."""
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- immutable receipt identities

    if not verify_completion(job):
        raise ValueError("cannot receipt incomplete queue outputs")
    completion = job["completion"]
    if job["kind"] == "train":
        checkpoint = Path(completion["checkpoint"])
        paths = [checkpoint, checkpoint.with_suffix(".complete.json"),
                 Path(job["output"]) / "config.json", Path(job["output"]) / "frame_plan.json"]
        from scripts.onestep_avatar.training.config import (  # noqa: PLC0415 -- optional evidence inventory
            parse_settings,
        )

        settings = parse_settings(job["arguments"])
        if settings.save_update_state:
            paths.append(settings.output / "update_states/text.pt")
            for step in range(1, settings.steps + 1):
                path = settings.output / "update_states" / f"step_{step:05d}.pt"
                paths.extend((path, path.with_suffix(".json")))
    elif job["kind"] == "experiment":
        paths = ([Path(completion["manifest"])] if "manifest" in completion else [])
        paths.extend(Path(value) for value in completion.get("records", []))
    elif job["kind"] in ("decode", "render"):
        paths = [Path(completion["manifest"])]
    else:
        paths = [Path(value) for value in completion["records"]]
        from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- verified evaluation artifact inventory

        paths = evaluate.evaluation_evidence_paths(job["arguments"], paths)
    return {
        "job_sha256": job["sha256"],
        "evidence": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
    }


def verify_receipt(job: dict, receipt: dict) -> bool:
    """Recheck both the bound job identity and its unchanged evidence/artifacts."""
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- saved receipt verification

    if receipt.get("job_sha256") != job["sha256"]:
        raise ValueError("completion receipt belongs to another job")
    for record in receipt.get("evidence", []):
        path = Path(record["path"])
        if not path.is_file():
            return False
        if sha256(path) != record["sha256"]:
            raise ValueError("completion receipt evidence changed")
    if not verify_completion(job):
        return False
    current = completion_receipt(job)
    if current != receipt:
        raise ValueError("completion receipt evidence inventory differs")
    return True


def ready_jobs(jobs: list[dict], state: dict) -> list[dict]:
    """Select pending work only after immutable completed dependencies verify."""
    verified = set()
    for job in jobs:
        row = state["jobs"][job["id"]]
        if row["sha256"] != job["sha256"]:
            raise ValueError("saved queue job settings changed")
        if row["state"] == "complete":
            receipt = row.get("receipt")
            if not isinstance(receipt, dict):
                raise ValueError("completed queue job has no verified receipt")
            if verify_receipt(job, receipt):
                verified.add(job["id"])
    ready = [
        job
        for job in jobs
        if state["jobs"][job["id"]]["state"] == "pending" and set(job.get("dependencies", [])).issubset(verified)
    ]
    if any(job["kind"] == "render" for job in ready):
        from scripts.onestep_avatar import comparisons  # noqa: PLC0415 -- saved-file readiness, no model session

        ready = [job for job in ready if job["kind"] != "render"
                 or comparisons.saved_comparison_inputs_ready(Path(job["render_spec"]))]
    return sorted(ready, key=lambda job: job["kind"] == "train")


def run_child(  # noqa: PLR0912, PLR0915 -- child lifecycle and persistent journal
    job: dict,
    claims: GPUClaims,
    log_path: Path,
    *,
    poll_seconds: float = 1,
    state_path: Path | None = None,
    jobs: list[dict] | None = None,
) -> dict:
    """Run one claimed package child with optional persistent attempt records."""
    import subprocess  # noqa: PLC0415 -- package-only subprocess execution

    from scripts.onestep_avatar.execution import queue_launch  # noqa: PLC0415 -- model-free registered launch

    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("child polling interval must be finite and positive")
    if not claims.owned:
        raise ValueError("child execution requires owned GPU reservations")
    child = None
    returncode = None
    journal_started = False
    request_path = None
    try:
        output = Path(job["output"])
        if log_path.resolve().is_relative_to(output.resolve()):
            raise ValueError("queue child log must stay outside its output directory")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("queue child requires a fresh output directory")
        command, changes = job_command(job, tuple(sorted(claims.owned)))
        if job.get("sha256") is not None:
            from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV  # noqa: PLC0415

            changes.update({TOKEN_ENV: claims.token, JOB_ENV: job["sha256"]})
        claims.refresh()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        launch_binding = None
        if job["kind"] == "train":
            from scripts.onestep_avatar.execution.queue_protocol import (  # noqa: PLC0415 -- exact launch evidence
                LAUNCH_ENV,
            )

            launch_record = training_launch_record(job, tuple(sorted(claims.owned)))
            if command != launch_record["command"]:
                raise ValueError("queue command changed before launch binding")
            launch_binding = log_path.with_name(f"{log_path.stem}.{claims.token}.launch.json").resolve()
            queue_launch.publish_exclusive(launch_binding, launch_record)
            changes[LAUNCH_ENV] = str(launch_binding)
        notifications = None
        budget = None
        if job["kind"] == "train":
            from scripts.onestep_avatar.training import config, resources  # noqa: PLC0415 -- shared phase authority
            settings = config.parse_settings(job["arguments"])
            budget = resources.read_budget(settings.resource_budget)
            if budget is not None:
                from scripts.onestep_avatar.execution import supervision  # noqa: PLC0415 -- optional bounded observer
                notifications = log_path.with_name(f"{log_path.stem}.{claims.token}.phases.json").resolve()
                changes.update(supervision.prepare_notifications(
                    notifications, token=claims.token, job_sha256=job["sha256"],
                    world=job["processes"], phases=resources.training_phases(settings),
                    budget_sha256=budget["sha256"],
                ))
        registry_path = getattr(claims, "path", None)
        if state_path is not None:
            if jobs is None:
                raise ValueError("persistent child execution requires the full prepared job list")
            if sum(candidate == job for candidate in jobs) != 1:
                raise ValueError("queue child differs from its prepared job list")
            with queue_state(state_path, jobs) as state:
                row = state["jobs"][job["id"]]
                if row["state"] != "pending":
                    raise ValueError("queue child requires a pending saved job")
                if job not in ready_jobs(jobs, state):
                    raise ValueError("queue child dependencies are not verified complete")
                request_path = queue_launch.prepare_request(
                    state_path.parent / "launches" / claims.token,
                    token=claims.token, job_sha256=job["sha256"], owner_identity=process_identity(os.getpid()),
                    job_id=job["id"], journal=state_path, command=command,
                )
                attempt = {
                    "command": command,
                    "log": str(log_path.resolve()),
                    "started_at": time.time(),
                    "gpus": sorted(claims.owned),
                    "owner_pid": os.getpid(),
                    "child_pid": None,
                    "environment_changes": changes,
                    "attempt_started_ticks": claims.started_ticks,
                    "launch_protocol": LAUNCH_PROTOCOL,
                    "launch_request": str(request_path.resolve()),
                    "launch_request_sha256": queue_launch.digest(request_path),
                }
                if registry_path is not None:
                    attempt["process_ledger"] = str(registry_path.resolve())
                if notifications is not None:
                    attempt["supervision_contract"] = str(notifications)
                if launch_binding is not None:
                    attempt.update(training_launch=str(launch_binding),
                                   training_launch_sha256=queue_launch.digest(launch_binding))
                row.update(
                    state="running", error=None, returncode=None, child_identity=None, child_session=None, **attempt
                )
                row["attempts"].append(dict(attempt))
            journal_started = True
        with log_path.open("x") as log:
            launch_command = command if request_path is None else queue_launch.guard_command(request_path)
            child = subprocess.Popen(
                launch_command,
                cwd=LTX_ROOT,
                env={**os.environ, **changes},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            claims.refresh(child_pid=child.pid)
            if state_path is not None:
                bootstrap = queue_launch.wait_registration(
                    request_path, child, launch_command, lambda: claims.refresh(child_pid=child.pid)
                )
                identity = bootstrap["identity"]
                with queue_state(state_path, jobs) as state:
                    state["jobs"][job["id"]]["child_pid"] = child.pid
                    state["jobs"][job["id"]]["child_identity"] = identity
                    state["jobs"][job["id"]]["child_session"] = child.pid
                    state["jobs"][job["id"]]["attempts"][-1]["child_pid"] = child.pid
                    state["jobs"][job["id"]]["attempts"][-1]["child_identity"] = identity
                    state["jobs"][job["id"]]["attempts"][-1]["child_session"] = child.pid
                queue_launch.publish_grant(request_path)
            worker_record = {"child_session": child.pid, "environment_changes": changes,
                             "attempt_started_ticks": claims.started_ticks}
            if registry_path is not None:
                worker_record["process_ledger"] = str(registry_path.resolve())
            if notifications is not None:
                if state_path is not None:
                    worker_record = read_queue_state(state_path, jobs)["jobs"][job["id"]]
                else:
                    identity = process_identity(child.pid)
                result = supervision.supervise(
                    child, identity=identity, command=command, worker_record=worker_record,
                    claims=claims, gpus=tuple(sorted(claims.owned)),
                    evidence_path=log_path.with_name(f"{log_path.stem}.{claims.token}.supervision.json"),
                    notifications_path=notifications, startup_seconds=budget["wall_seconds_per_phase"],
                    phase_seconds=budget["wall_seconds_per_phase"], shutdown_seconds=30,
                    poll_seconds=min(poll_seconds, 1),
                )
                returncode = result["returncode"]
                if result["state"] != "passed":
                    raise QueueChildError(f"bounded package child failed: {result['error']}; log: {log_path}")
            else:
                while child.poll() is None:
                    time.sleep(poll_seconds)
                    claims.refresh(child_pid=child.pid)
                returncode = child.poll()
            if returncode != 0:
                raise QueueChildError(f"queue child failed with exit {returncode}; log: {log_path}")
            if owned_workers_live(worker_record):
                raise ValueError("queue child still has live workers after leader exit")
            receipt = completion_receipt(job)
            result = {
                "child_pid": child.pid,
                "returncode": returncode,
                "command": command,
                "environment_changes": changes,
                "log": str(log_path.resolve()),
                "receipt": receipt,
            }
            if state_path is not None:
                with queue_state(state_path, jobs) as state:
                    state["jobs"][job["id"]].update(state="complete", finished_at=time.time(), **result)
                    state["jobs"][job["id"]]["attempts"][-1].update(state="complete", **result)
            return result
    except BaseException as error:
        if journal_started:
            with queue_state(state_path, jobs) as state:
                row = state["jobs"][job["id"]]
                if row["state"] == "running":
                    live = child is not None and (
                        child.poll() is None or owned_workers_live({
                            "child_session": child.pid, "environment_changes": changes,
                            "attempt_started_ticks": claims.started_ticks,
                            **({"process_ledger": str(registry_path.resolve())} if registry_path is not None else {}),
                        })
                    )
                    row.update(
                        state="running" if live else "failed",
                        error=f"{type(error).__name__}: {error}",
                        child_pid=None if child is None else child.pid,
                        returncode=returncode,
                        finished_at=time.time(),
                    )
                    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- terminal log evidence

                    row["attempts"][-1].update(
                        state=row["state"], error=row["error"], returncode=returncode,
                        finished_at=row["finished_at"], log_sha256=sha256(log_path) if log_path.is_file() else None,
                    )
        raise
    finally:
        if child is None or (child.poll() is not None and not owned_workers_live({
            "child_session": child.pid, "environment_changes": changes,
            "attempt_started_ticks": claims.started_ticks,
            **({"process_ledger": str(registry_path.resolve())} if registry_path is not None else {}),
        })):
            claims.release()


def startup_retry_evidence(job: dict, row: dict) -> dict | None:  # noqa: PLR0911 -- conservative ordered retry gates
    """Authorize proven startup contention with no surviving workers or update evidence."""
    from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, PREFIX, TOKEN_ENV  # noqa: PLC0415
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415

    attempts = row.get("attempts", [])
    if job["kind"] != "train" or row.get("state") != "failed" or not attempts or len(attempts) >= 4:
        return None
    attempt = attempts[-1]
    if (attempt.get("state") != "failed" or type(attempt.get("returncode")) is not int
            or attempt["returncode"] == 0 or not attempt.get("child_session")
            or attempt["returncode"] != row.get("returncode")
            or attempt.get("child_pid") != row.get("child_pid")
            or attempt["child_session"] != row.get("child_session")):
        return None
    if inspect_child(row) == "live":
        return None
    changes = attempt.get("environment_changes", {})
    token = changes.get(TOKEN_ENV)
    if not token or changes.get(JOB_ENV) != job["sha256"]:
        return None
    log = Path(attempt.get("log", ""))
    if not log.is_file() or sha256(log) != attempt.get("log_sha256"):
        return None
    contended = []
    for line in log.read_text().splitlines():
        if PREFIX not in line:
            continue
        if not line.startswith(PREFIX):
            return None
        try:
            event = json.loads(line[len(PREFIX):])
        except json.JSONDecodeError:
            return None
        if (not isinstance(event, dict)
                or set(event) != {"schema_version", "event", "token", "job_sha256", "rank", "reason"}
                or type(event.get("schema_version")) is not int or event["schema_version"] != 1
                or event.get("token") != token
                or event.get("job_sha256") != job["sha256"] or type(event.get("rank")) is not int
                or not 0 <= event["rank"] < 4 or event.get("event") != "startup_contended"
                or event.get("reason") not in ("cuda_oom", "port_in_use")):
            return None  # Includes any rank's updates_begin boundary.
        contended.append(event["rank"])
    output = Path(job["output"])
    if not contended or any(path.stat().st_size for path in output.rglob("metrics_rank*.jsonl")):
        return None
    return {"token": token, "log": str(log.resolve()), "log_sha256": sha256(log),
            "contended_ranks": sorted(set(contended)), "attempt": len(attempts)}


def retry_startup_failure(job: dict, jobs: list[dict], state_path: Path) -> bool:
    """Preserve a proven failed attempt before making that same job pending again."""
    import shutil  # noqa: PLC0415 -- preserve terminal logs without changing originals

    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415

    output, moved = Path(job["output"]), None
    if startup_retry_evidence(job, read_queue_state(state_path, jobs)["jobs"][job["id"]]) is None:
        return False
    try:
        with queue_state(state_path, jobs) as state:
            row = state["jobs"][job["id"]]
            evidence = startup_retry_evidence(job, row)
            if evidence is None:
                return False
            archive = output.parent / "superseded_startup_contention" / f"{output.name}_{evidence['token']}"
            files = {}
            if output.exists():
                if output.is_symlink() or not output.is_dir() or any(path.is_symlink() for path in output.rglob("*")):
                    raise ValueError("startup archive requires ordinary output files")
                files = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()}
            archive.mkdir(parents=True, exist_ok=False)
            if output.exists():
                moved = archive / "output"
                output.rename(moved)
            shutil.copyfile(evidence["log"], archive / "child.log")
            if sha256(archive / "child.log") != evidence["log_sha256"]:
                raise ValueError("startup log changed during preservation")
            if moved is not None and files != {
                str(path.relative_to(moved)): sha256(path) for path in moved.rglob("*") if path.is_file()
            }:
                raise ValueError("startup output changed during preservation")
            record = {"job_id": job["id"], "job_sha256": job["sha256"], **evidence,
                      "original_output": str(output.resolve()), "files": files,
                      "error": row["error"], "returncode": row["returncode"]}
            (archive / "preservation.json").write_text(json.dumps(record, indent=2) + "\n")
            (archive / "README.md").write_text(
                "Startup contention before training began. The replacement is the same saved\n"
                "queue job; it remains pending until another attempt completes. Original\n"
                "output bytes and the failed log are preserved here. Do not cite this attempt\n"
                "as a scientific result. The queue journal records whether preservation\n"
                "and retry publication succeeded.\n"
            )
            row["attempts"][-1].update(archive=str(archive.resolve()), retry_evidence=evidence)
            row.update(state="pending", child_pid=None, child_identity=None, child_session=None,
                       startup_retries=len(row["attempts"]))
        return True
    except BaseException:
        if moved is not None and moved.exists():
            if output.exists():
                raise ValueError("startup archive rollback refused an occupied output path") from None
            moved.rename(output)
        raise


def dispatch_ready(jobs: list[dict], state: dict, state_path: Path, claims_dir: Path) -> bool:
    """Refresh device availability and run at most one eligible reserved child."""
    import subprocess  # noqa: PLC0415 -- GPU inventory for live selection

    ready = ready_jobs(jobs, state)
    if not ready:
        return False
    def inventory() -> dict[int, int]:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        return parse_gpu_memory(raw)

    memory = inventory()
    for job in ready:
        from scripts.onestep_avatar.execution.process_registry import ProcessRegistry  # noqa: PLC0415 -- current policy
        claims = ProcessRegistry(claims_dir)
        gpus = claims.choose(memory, training=job["kind"] == "train")
        if gpus is None or not claims.acquire(gpus, job=job["id"]):
            continue
        # Claims cannot prevent an external process from starting between
        # the first observation and our reservation.
        try:
            memory = inventory()
        except BaseException:
            claims.release()
            raise
        if any(memory[gpu] >= 1024 for gpu in gpus):
            claims.release()
            continue
        log = state_path.parent / "logs" / f"{claims.token}.log"
        try:
            run_child(job, claims, log, state_path=state_path, jobs=jobs)
        except QueueChildError:
            if not retry_startup_failure(job, jobs, state_path):
                raise
            return True
        return True
    return False


def execute_jobs(jobs_path: Path, state_path: Path, claims_dir: Path, *, once: bool, poll_seconds: float = 30) -> int:
    """Reread append-only work, verify dependencies, and wait without reserving devices."""
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("queue polling interval must be finite, positive and at most 60 seconds")
    observed = []
    while True:
        jobs = prepare_jobs(jobs_path)
        identities = [(job["id"], job["sha256"]) for job in jobs]
        if identities[:len(observed)] != observed:
            raise ValueError("queue revisions may only append unchanged jobs")
        observed = identities
        state = read_queue_state(state_path, jobs)
        if not once:
            rows = [state["jobs"][job["id"]] for job in jobs]
            if any(row["state"] == "running" for row in rows):
                raise ValueError("queue running attempts require explicit recovery before loop execution")
            if any(row["state"] == "failed" for row in rows):
                return 1
            if all(row["state"] == "complete" for row in rows):
                for job in jobs:
                    receipt = state["jobs"][job["id"]].get("receipt")
                    if not isinstance(receipt, dict) or not verify_receipt(job, receipt):
                        raise ValueError("completed queue job has missing verified evidence")
                return 0
        if dispatch_ready(jobs, state, state_path, claims_dir):
            if once:
                updated = read_queue_state(state_path, jobs)
                return 2 if any(
                    row["state"] == "pending" and row["attempts"] and row["attempts"][-1].get("retry_evidence")
                    for row in updated["jobs"].values()
                ) else 0
            continue
        if once:
            return 2 if ready_jobs(jobs, state) else 0
        time.sleep(poll_seconds)


def recover_jobs(jobs_path: Path, state_path: Path) -> dict:
    """Recover recorded terminated children without launching work or changing claims."""
    if not state_path.is_file():
        raise ValueError("queue recovery requires an existing saved journal")
    jobs = prepare_jobs(jobs_path)
    state = read_queue_state(state_path, jobs)
    recovery_needed = any(state["jobs"][job["id"]]["state"] == "running" for job in jobs)
    if recovery_needed:
        with queue_state(state_path, jobs, recover=True) as recovered:
            state = recovered
    return {"recovered": recovery_needed,
            "jobs": [{"id": job["id"], "state": state["jobs"][job["id"]]["state"]} for job in jobs]}


def main(argv: list[str] | None = None) -> int:
    """Review, dispatch once, or continuously execute normalized package-owned jobs."""
    import argparse  # noqa: PLC0415 -- lightweight queue CLI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="claim available GPUs and run ready jobs")
    parser.add_argument("--recover", action="store_true", help="inspect terminated children and verify saved outputs")
    parser.add_argument("--process-ledger", type=Path, help="one shared JSON record of package-owned processes")
    parser.add_argument("--claims-dir", type=Path, help=argparse.SUPPRESS)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--once", action="store_true")
    execution.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30, help="loop wait interval, at most 60 seconds")
    args = parser.parse_args(argv)
    if sum((args.dry_run, args.execute, args.recover)) != 1:
        parser.error("choose exactly one of --dry-run, --execute or --recover")
    if args.process_ledger is not None and args.claims_dir is not None:
        parser.error("choose --process-ledger or the deprecated --claims-dir path alias")
    ledger = args.process_ledger if args.process_ledger is not None else (
        None if args.claims_dir is None else args.claims_dir / "processes.json")
    if args.execute and (not (args.once or args.loop) or ledger is None):
        parser.error("--execute requires --once or --loop and --process-ledger")
    if (args.loop or args.once) and not args.execute:
        parser.error("--once and --loop require --execute")
    if not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 60:
        parser.error("--poll-seconds must be finite, positive and at most 60")
    if args.recover:
        print(json.dumps(recover_jobs(args.jobs, args.state), indent=2))  # noqa: T201 -- requested recovery summary
        return 0
    if args.execute:
        return execute_jobs(args.jobs, args.state, ledger, once=args.once, poll_seconds=args.poll_seconds)
    jobs = prepare_jobs(args.jobs)
    state = read_queue_state(args.state, jobs)
    planned = []
    for job in jobs:
        gpus = TRAIN_GPUS if job["kind"] == "train" else (EVALUATION_PREFERENCE[0],)
        command, changes = job_command(job, gpus)
        planned.append(
            {
                "id": job["id"],
                "sha256": job["sha256"],
                "state": state["jobs"][job["id"]]["state"],
                "command": command,
                "environment_changes": changes,
                "planned_physical_gpus": list(gpus),
            }
        )
    print(  # noqa: T201 -- requested dry-run artifact
        json.dumps(
            {"validation": "arguments_and_identity_only", "gpu_availability_checked": False, "jobs": planned}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
