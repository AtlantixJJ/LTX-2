"""Completion owner for managed execution; see doc/execution/completion.md."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

from scripts.onestep_avatar.execution import clock as execution_clock
from scripts.onestep_avatar.execution import jobs as job_contracts


def verify_attempt_evidence(job: dict, row: dict) -> list[Path]:
    """Require saved successful supervision and actual child evidence before receipt."""
    from scripts.onestep_avatar.execution import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
        queue_launch,
        runtime,
    )
    from scripts.onestep_avatar.execution.queue_protocol import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
        TOKEN_ENV,
    )
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- selected owner at use; avoids import cycles

    execution_clock.validate_execution_clock(job, row)
    if not {
        "supervision_result",
        "supervision_result_sha256",
        "supervision_contract",
        "command",
        "child_identity",
        "environment_changes",
        "gpus",
    }.issubset(row):
        raise ValueError("queue success lacks its original supervision evidence")
    command, environment = job_contracts.job_command(job, tuple(row["gpus"]))
    if row["command"] != command:
        raise ValueError("queue supervision command differs from requested owner")
    if any(row["environment_changes"].get(key) != value for key, value in environment.items()):
        raise ValueError("queue supervision environment differs from requested devices, budget or policy")
    path = Path(row["supervision_result"])
    if sha256(path) != row["supervision_result_sha256"]:
        raise ValueError("queue supervision evidence changed")
    result = json.loads(path.read_text())
    if (
        result.get("state") != "passed"
        or result.get("returncode") != 0
        or result.get("workers_absent") is not True
        or result.get("worker_inventory", {}).get("complete") is not True
        or result.get("worker_inventory", {}).get("workers_live") is not False
        or result.get("command") != row["command"]
        or result.get("identity") != row["child_identity"]
    ):
        raise ValueError("queue success lacks complete owned-worker absence evidence")
    contract = Path(row["supervision_contract"])
    if sha256(contract) != result.get("notification_contract_sha256"):
        raise ValueError("queue supervision contract changed")
    paths = [path, contract]
    profile = job_contracts.execution_profile(job)
    notification = json.loads(contract.read_text())
    if notification.get("budget_sha256") != job["effective_budget"]["sha256"]:
        raise ValueError("queue supervision budget differs from requested limits")
    observed = queue_launch.read_completed_notifications(
        contract,
        contract_sha256=sha256(contract),
        token=row["environment_changes"][TOKEN_ENV],
        job_sha256=job["sha256"],
        world=profile["world"],
    )
    if observed["phases"] != profile["phases"]:
        raise ValueError("queue supervision phase inventory differs from its owner")
    paths.extend(contract.with_name(contract.name + ".events") / name for name in observed["event_sha256"])
    if job["kind"] != "train":
        runtime_path = Path(row["runtime_contract"])
        binding = runtime.read_contract(runtime_path, row["runtime_contract_sha256"])
        expected = {
            "token": row["environment_changes"][TOKEN_ENV],
            "job_sha256": job["sha256"],
            "budget": job["effective_budget"],
            "generic_phase": profile["generic_phase"],
            "numerical_policy": job.get("numerical_policy"),
        }
        if any(binding[key] != value for key, value in expected.items()):
            raise ValueError("queue runtime proof differs from requested attempt, policy or budget")
        runtime.verify_result(runtime_path, row["runtime_contract_sha256"])
        paths.extend((runtime_path, runtime.result_path(runtime_path)))
    return paths


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
    """Verify saved training/evaluation evidence; a job with missing artifacts stays pending."""
    import torch  # noqa: PLC0415 -- saved tensor verification only

    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- artifact identities
    from scripts.onestep_avatar.training import checkpoints  # noqa: PLC0415 -- shared adapter checks

    if "effective_budget" in job:
        from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- selected owner at use; avoids import cycles
            resources,
        )

        resources.check_budget(job["effective_budget"])

    root = Path(job["output"]).resolve()

    def evidence_path(value: str) -> Path:
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("queue completion evidence escapes its output directory")
        return path

    completion = job["completion"]
    if job["kind"] in ("preview", "product", "bench"):
        selected = importlib.import_module(job_contracts.ORDINARY[job["kind"]])
        if job["kind"] == "preview" and selected.queue_identity(job["arguments"]) != job["preview_identity"]:
            raise ValueError("queue immutable preview identity changed")
        evidence = ([completion["manifest"]] if "manifest" in completion else []) + completion.get("records", [])
        if any(not evidence_path(value).is_file() for value in evidence):
            return False
        verified = selected.verify_completion(job["arguments"], root)
        if (
            job["kind"] == "bench"
            and isinstance(verified, dict)
            and "records" in verified
            and verified["records"] != completion.get("records")
        ):
            raise ValueError("queue benchmark inventory differs from its scientific owner")
        return True
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
        selected = importlib.import_module(job_contracts.experiment_entry(job["experiment"]))
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
            software.check_current(row.get("software"))
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


def completion_receipt(job: dict, *, attempt: dict | None = None) -> dict:
    """Capture verified evidence identities; child lifecycle enforcement is caller-owned."""
    if not verify_completion(job):
        raise ValueError("cannot receipt incomplete queue outputs")
    return _completion_receipt(job, attempt=attempt)


def _completion_receipt(job: dict, *, attempt: dict | None = None) -> dict:
    """Build the inventory only after the caller has verified completion once."""
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- immutable receipt identities

    completion = job["completion"]
    if job["kind"] == "train":
        checkpoint = Path(completion["checkpoint"])
        paths = [
            checkpoint,
            checkpoint.with_suffix(".complete.json"),
            Path(job["output"]) / "config.json",
            Path(job["output"]) / "frame_plan.json",
        ]
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
        paths = [Path(completion["manifest"])] if "manifest" in completion else []
        paths.extend(Path(value) for value in completion.get("records", []))
        selected = importlib.import_module(job_contracts.experiment_entry(job["experiment"]))
        if hasattr(selected, "evidence_paths"):
            paths = selected.evidence_paths(Path(job["spec"]), Path(job["output"]), verified=True)
            if any(not path.resolve().is_relative_to(Path(job["output"]).resolve()) for path in paths):
                raise ValueError("experiment receipt evidence escapes its output directory")
    elif job["kind"] in ("decode", "render"):
        paths = [Path(completion["manifest"])]
    elif job["kind"] in ("preview", "product", "bench"):
        paths = importlib.import_module(job_contracts.ORDINARY[job["kind"]]).evidence_paths(
            job["arguments"], Path(job["output"]), verified=True
        )
        if any(not path.resolve().is_relative_to(Path(job["output"]).resolve()) for path in paths):
            raise ValueError("ordinary receipt evidence escapes its output directory")
    else:
        paths = [Path(value) for value in completion["records"]]
        from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- verified evaluation artifact inventory

        paths = evaluate.evaluation_evidence_paths(job["arguments"], paths)
    receipt = {
        "job_sha256": job["sha256"],
        "evidence": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
    }
    if attempt is not None:
        execution_paths = verify_attempt_evidence(job, attempt)
        keys = {
            "command",
            "gpus",
            "child_identity",
            "environment_changes",
            "supervision_result",
            "supervision_result_sha256",
            "supervision_contract",
            "runtime_contract",
            "runtime_contract_sha256",
            "first_attempt_started_at",
            "overall_deadline_at",
        }
        binding = {key: value for key, value in attempt.items() if key in keys}
        binding["attempts"] = [
            {
                key: value
                for key, value in row.items()
                if key in {"started_at", "first_attempt_started_at", "overall_deadline_at"}
            }
            for row in attempt.get("attempts", [])
        ]
        receipt["execution"] = {
            "binding": binding,
            "evidence": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in execution_paths],
        }
    return receipt


def verify_receipt(job: dict, receipt: dict) -> bool:
    """Recheck both the bound job identity and its unchanged evidence/artifacts."""
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415 -- saved receipt verification

    if receipt.get("job_sha256") != job["sha256"]:
        raise ValueError("completion receipt belongs to another job")
    if "effective_budget" in job and "execution" not in receipt:
        raise ValueError("current queue receipt lacks its supervised execution proof")
    for record in receipt.get("evidence", []):
        path = Path(record["path"])
        if not path.is_file():
            return False
        if sha256(path) != record["sha256"]:
            raise ValueError("completion receipt evidence changed")
    if not verify_completion(job):
        return False
    current = _completion_receipt(job)
    if current != {key: value for key, value in receipt.items() if key != "execution"}:
        raise ValueError("completion receipt evidence inventory differs")
    if "execution" in receipt:
        execution = receipt["execution"]
        completed_at = execution["binding"].get("completed_at")
        deadline = execution["binding"].get("overall_deadline_at")
        if (
            type(completed_at) not in (int, float)
            or not math.isfinite(completed_at)
            or (
                deadline is not None
                and (completed_at >= deadline or completed_at < execution["binding"]["first_attempt_started_at"])
            )
        ):
            raise ValueError("queue completion was not observed before its original deadline")
        paths = verify_attempt_evidence(job, execution["binding"])
        if execution["evidence"] != [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths]:
            raise ValueError("queue completion execution evidence changed")
    return True


def ready_jobs(jobs: list[dict], state: dict) -> list[dict]:
    """Select pending work only after the completed dependencies it needs verify.

    A receipt check can rehash every pinned input, including model weights, so only
    receipts that gate a pending job are verified here. The loop verifies every
    receipt once more before it reports the whole list complete.
    """
    pending = [job for job in jobs if state["jobs"][job["id"]]["state"] == "pending"]
    needed = {dependency for job in pending for dependency in job.get("dependencies", [])}
    verified = set()
    for job in jobs:
        row = state["jobs"][job["id"]]
        if row["sha256"] != job["sha256"]:
            raise ValueError("saved queue job settings changed")
        if row["state"] == "complete":
            receipt = row.get("receipt")
            if not isinstance(receipt, dict):
                raise ValueError("completed queue job has no verified receipt")
            if job["id"] in needed and verify_receipt(job, receipt):
                verified.add(job["id"])
    ready = [job for job in pending if set(job.get("dependencies", [])).issubset(verified)]
    if any(job["kind"] == "render" for job in ready):
        from scripts.onestep_avatar import comparisons  # noqa: PLC0415 -- saved-file readiness, no model session

        ready = [
            job
            for job in ready
            if job["kind"] != "render" or comparisons.saved_comparison_inputs_ready(Path(job["render_spec"]))
        ]
    return sorted(ready, key=lambda job: job["kind"] == "train")
