"""Jobs owner for managed execution; see doc/execution/jobs.md."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from scripts.onestep_avatar.execution.devices import GPU_POOL, TRAIN_PROCESSES, validate_selection

EXPERIMENTS = {
    "sigma_sweep": "scripts.onestep_avatar.experiments.sigma_sweep",
    "causality": "scripts.onestep_avatar.experiments.causality",
    "fusion_parity": "scripts.onestep_avatar.experiments.fusion_parity",
    "stock_parity": "scripts.onestep_avatar.experiments.stock_parity",
    "prune_baseline": "scripts.onestep_avatar.experiments.prune_baseline",
    "prune_reproduction": "scripts.onestep_avatar.experiments.prune_reproduction",
    "adapter_effect_check": "scripts.onestep_avatar.experiments.adapter_effect_check",
    "continuation_check": "scripts.onestep_avatar.experiments.continuation_check",
    "training_update_check": "scripts.onestep_avatar.experiments.training_update_check",
    "training_slice_check": "scripts.onestep_avatar.experiments.training_slice_check",
    "conditioning_check": "scripts.onestep_avatar.experiments.conditioning_check",
    "continuation_identity_check": "scripts.onestep_avatar.experiments.continuation_identity_check",
    "stats": "scripts.onestep_avatar.experiments.stats",
}


ORDINARY = {
    "evaluate": "scripts.onestep_avatar.evaluate",
    "decode": "scripts.onestep_avatar.decode_saved",
    "render": "scripts.onestep_avatar.comparisons",
    "preview": "scripts.onestep_avatar.previews",
    "product": "scripts.onestep_avatar.infer",
    "bench": "scripts.onestep_avatar.bench",
}


def experiment_entry(selector: str) -> str:
    """Resolve a fixed experiment selector without importing or discovering code."""
    if not isinstance(selector, str) or selector not in EXPERIMENTS:
        raise ValueError("queue experiment selector is unknown")
    return EXPERIMENTS[selector]


def validate_completion_descriptor(descriptor: dict) -> None:
    """Require model-free evidence paths as data, before normalization or imports."""
    if (
        not isinstance(descriptor, dict)
        or not descriptor
        or set(descriptor) - {"manifest", "records"}
        or ("manifest" in descriptor and (not isinstance(descriptor["manifest"], str) or not descriptor["manifest"]))
        or (
            "records" in descriptor
            and (
                not isinstance(descriptor["records"], list)
                or not descriptor["records"]
                or any(not isinstance(p, str) or not p for p in descriptor["records"])
            )
        )
    ):
        raise ValueError("queue experiment parser completion descriptor is malformed")


def validate_experiment_job(job: dict) -> None:
    """Check the selector and pinned spec as data before an owner can be imported."""
    experiment_entry(job.get("experiment"))
    validate_completion_descriptor(job.get("completion"))
    spec = job.get("spec")
    digest = job.get("spec_sha256")
    if (
        not isinstance(spec, str)
        or not spec
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("queue experiment requires a specification path and SHA-256")
    arguments = job["arguments"]
    if arguments.count("--spec") != 1:
        raise ValueError("queue experiment requires one explicit specification")
    index = arguments.index("--spec")
    if index + 1 >= len(arguments) or arguments[index + 1] != spec:
        raise ValueError("queue experiment specification differs from command")


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
        if job.get("kind") not in ("train", "experiment", *ORDINARY):
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
        if (
            job["kind"] == "render"
            and sum(value.split("=", 1)[0] == "--render-saved-comparisons" for value in arguments) != 1
        ):
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
            if (
                type(job.get("processes")) is not int
                or job["processes"] != TRAIN_PROCESSES
                or not isinstance(job.get("accelerate_config"), str)
            ):
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
    if (
        not isinstance(dependencies, list)
        or any(not isinstance(value, str) for value in dependencies)
        or len(set(dependencies)) != len(dependencies)
        or raw.get("id") in dependencies
    ):
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
    forbidden = {"--gpu-id", "--dry-run", "--help", "--overwrite", "--saved-metrics", "--verify"}
    if job["kind"] != "preview":
        forbidden.add("--preview-job")
    if any(value.split("=", 1)[0] in forbidden for value in arguments):
        raise ValueError("queue command contains a forbidden execution override")
    if job["kind"] == "train":
        from scripts.onestep_avatar.training.config import parse_settings as parser  # noqa: PLC0415
    elif job["kind"] == "decode":
        from scripts.onestep_avatar.decode_saved import parse_args as parser  # noqa: PLC0415
    elif job["kind"] == "experiment":
        selected = importlib.import_module(experiment_entry(job["experiment"]))
        parser = selected.parse_args
    elif job["kind"] in ("preview", "product", "bench"):
        parser = importlib.import_module(ORDINARY[job["kind"]]).parse_args
    else:
        from scripts.onestep_avatar import (  # noqa: PLC0415 -- select ordinary owners lazily
            comparisons,
            evaluate,
        )

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
    if job["kind"] in ("preview", "product", "bench"):
        selected = importlib.import_module(ORDINARY[job["kind"]])
        descriptor = getattr(parsed, "completion", None)
        if job["kind"] == "bench" and descriptor is None:
            descriptor = completion  # The benchmark verifier derives its exact case inventory.
        validate_completion_descriptor(descriptor)
        if completion != descriptor:
            raise ValueError("queue ordinary completion differs from parsed output")
        if job["kind"] == "preview":
            identity = selected.queue_identity(arguments)
            if job.get("preview_identity", identity) != identity:
                raise ValueError("queue immutable preview identity changed")
            job["preview_identity"] = identity
        paths = ([descriptor["manifest"]] if "manifest" in descriptor else []) + descriptor.get("records", [])
        if any(not Path(path).is_relative_to(Path(output)) for path in paths):
            raise ValueError("queue ordinary completion escapes its output directory")
    from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
        resources,
    )

    budget_path = (
        parsed.resource_budget
        if job["kind"] == "train"
        else (None if job.get("resource_budget") is None else (root / job["resource_budget"]).resolve())
    )
    if job["kind"] == "train" and "resource_budget" in job:
        raise ValueError("training resource budget belongs in its typed arguments")
    if budget_path is not None:
        if not budget_path.is_file():
            raise ValueError("queue resource budget is missing")
        digest = sha256(budget_path)
        if job.get("resource_budget_sha256", digest) != digest:
            raise ValueError("queue resource budget changed from its claimed hash")
        job["resource_budget_sha256"] = digest
        if job["kind"] != "train":
            job["resource_budget"] = str(budget_path)
    elif "resource_budget_sha256" in job:
        raise ValueError("queue resource budget hash requires an explicit budget")
    budget = resources.effective_budget(budget_path)
    if job.get("effective_budget", budget) != budget:
        raise ValueError("queue effective resource budget differs from checked limits")
    job["effective_budget"] = budget
    if job["kind"] == "experiment":
        selected = importlib.import_module(experiment_entry(job["experiment"]))
        if hasattr(selected, "resource_budget"):
            required = selected.resource_budget(Path(job["spec"]))
            if required is not None and (budget_path is None or required.resolve() != budget_path.resolve()):
                raise ValueError("queue budget differs from the experiment's scientific budget")
        if hasattr(selected, "runtime_profile"):
            profile = selected.runtime_profile(Path(job["spec"]))
            if profile.get("resource_budget") is not None and profile["resource_budget"] != budget:
                raise ValueError("queue budget differs from the experiment's scientific budget")
        if hasattr(selected, "launch_requirements"):
            requirements = selected.launch_requirements(Path(job["spec"]))
            if requirements.get("launcher") == "accelerate":
                requirements = dict(requirements)
                config_path = Path(requirements["accelerate_config"]).resolve()
                requirements.update(accelerate_config=str(config_path), accelerate_config_sha256=sha256(config_path))
                port = job.get("port")
                if type(port) is not int or not 1 <= port <= 65535:
                    raise ValueError("FSDP experiment requires a recorded rendezvous port")
                if job.get("experiment_launch", requirements) != requirements:
                    raise ValueError("queue experiment launch requirements changed")
                job["experiment_launch"] = requirements
    if job.get("numerical_policy") not in (None, "deterministic"):
        raise ValueError("queue numerical policy is unknown")
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
    by_id = {job["id"]: job for job in jobs}
    for job in jobs:
        if job["kind"] == "experiment":
            selected = importlib.import_module(experiment_entry(job["experiment"]))
            if hasattr(selected, "dependency_outputs"):
                declared = {Path(by_id[name]["output"]).resolve() for name in job.get("dependencies", [])}
                if not set(selected.dependency_outputs(Path(job["spec"]))).issubset(declared):
                    raise ValueError("queue checker reference must be an explicit dependency output")
    return jobs


def training_launch_record(job: dict, gpus: tuple[int, ...] = GPU_POOL[:TRAIN_PROCESSES]) -> dict:
    """Capture the requested launch facts before dispatch, with exact YAML bytes."""
    import base64  # noqa: PLC0415 -- lossless original configuration snapshot
    import hashlib  # noqa: PLC0415 -- validate snapshot against canonical pinned hash

    from scripts.onestep_avatar.training import numerics  # noqa: PLC0415 -- import-light numerical launch owner

    prepared = prepare_job(job, Path.cwd())
    if prepared["kind"] != "train":
        raise ValueError("training launch binding requires a training job")
    command, _environment = job_command(prepared, gpus)
    original = Path(prepared["accelerate_config"]).read_bytes()
    if hashlib.sha256(original).hexdigest() != prepared["accelerate_config_sha256"]:
        raise ValueError("queue Accelerate configuration changed during launch snapshot")
    return {
        "schema_version": 2,
        "job": prepared,
        "command": command,
        "gpus": list(gpus),
        "accelerate_config_bytes_base64": base64.b64encode(original).decode("ascii"),
        "accelerate_config_sha256": prepared["accelerate_config_sha256"],
        "numerical_environment": dict(numerics.ENVIRONMENT),
    }


def verify_training_launch(record: dict, job: dict) -> dict:
    """Reconstruct canonical identity and compare exact original launch bytes/command."""
    import base64  # noqa: PLC0415 -- checked original bytes
    import binascii  # noqa: PLC0415 -- malformed original snapshot failure
    import hashlib  # noqa: PLC0415 -- original-byte integrity

    if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 2:
        raise ValueError("original training launch evidence is missing or malformed")
    prepared = prepare_job(job, Path.cwd())
    gpus = record.get("gpus")
    if not isinstance(gpus, list):
        raise ValueError("original training launch GPU inventory is missing")
    validate_selection(tuple(gpus), training=True)
    expected = training_launch_record(prepared, tuple(gpus))
    try:
        original = base64.b64decode(record["accelerate_config_bytes_base64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as error:
        raise ValueError("original Accelerate snapshot is malformed") from error
    if hashlib.sha256(original).hexdigest() != record.get("accelerate_config_sha256") or record != expected:
        raise ValueError("original training launch identity, command or Accelerate bytes differ")
    return prepared


def read_training_launch(path: Path) -> dict:
    """Read and verify dispatch evidence; do not infer missing original launch facts."""
    record = json.loads(path.read_text())
    verify_training_launch(record, record.get("job", {}))
    return record


def job_command(job: dict, gpus: tuple[int, ...]) -> tuple[list[str], dict[str, str]]:  # noqa: PLR0912, PLR0915 -- explicit owner and byte gates
    """Construct package-only child commands and physical-device environment changes."""
    import sys  # noqa: PLC0415 -- current conda Python owns all execution

    kind = job["kind"]
    validate_selection(gpus, training=kind == "train")
    environment = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    if "effective_budget" in job:
        from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
            resources,
        )

        environment.update(resources.budget_environment(job["effective_budget"]))
    if job.get("numerical_policy") == "deterministic":
        from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
            numerics,
        )

        environment.update(numerics.ENVIRONMENT)
    if kind == "train":
        from scripts.onestep_avatar.training import numerics  # noqa: PLC0415 -- import-light launch policy

        environment.update(numerics.ENVIRONMENT)
        if job["processes"] != TRAIN_PROCESSES:
            raise ValueError("queued training requires four free GPUs and four processes")
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
        if kind not in (*ORDINARY, "experiment") or len(gpus) != 1:
            raise ValueError("queued evaluation/decoding requires one GPU and a package owner")
        module = ORDINARY.get(kind)
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
        if kind == "experiment" and "experiment_launch" in job:
            from scripts.onestep_avatar.hashing import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
                sha256,
            )

            launch = job["experiment_launch"]
            if (
                launch["processes"] != 1
                or launch["mixed_precision"] != "bf16"
                or sha256(Path(launch["accelerate_config"])) != launch["accelerate_config_sha256"]
            ):
                raise ValueError("one-rank FSDP experiment launch facts changed")
            command = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "--config_file",
                launch["accelerate_config"],
                "--num_processes",
                "1",
                "--mixed_precision",
                "bf16",
                "--main_process_port",
                str(job["port"]),
                "-m",
                module,
                *job["arguments"],
                "--gpu-id",
                "0",
            ]
    return command, environment


def execution_profile(job: dict) -> dict:
    """Keep phase inventories with the scientific owners; no model sessions."""
    if job["kind"] == "train":
        from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
            config,
            resources,
        )

        return {
            "world": TRAIN_PROCESSES,
            "phases": resources.training_phases(config.parse_settings(job["arguments"])),
            "generic_phase": False,
        }
    if job["kind"] == "experiment":
        selected = importlib.import_module(experiment_entry(job["experiment"]))
        if hasattr(selected, "runtime_profile"):
            profile = dict(selected.runtime_profile(Path(job["spec"])))
            profile["world"] = profile.pop("processes", 1)
            return profile
        if hasattr(selected, "resource_phases"):
            phases = selected.resource_phases(Path(job["spec"]))
            return {"world": 1, "phases": phases, "generic_phase": phases == ["runtime"]}
    return {"world": 1, "phases": ["runtime"], "generic_phase": True}
