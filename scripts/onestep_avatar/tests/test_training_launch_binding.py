"""Original launch controls use the real queue normalization and replay gate."""

import base64
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.training import config, engine, numerics, resources


def original_job(tmp_path: Path, *, tolerance: dict | None = None) -> tuple:
    config = tmp_path / "fsdp.yaml"
    config.write_text("compute_environment: LOCAL_MACHINE\ndistributed_type: FSDP\nmixed_precision: bf16\n")
    output = tmp_path / "distributed"
    budget_path = tmp_path / "native_protocol.json"
    protocol = {"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48000000000}
    if tolerance is not None:
        protocol["tolerance"] = tolerance
    budget_path.write_text(json.dumps(protocol))
    arguments = [
        "--mode",
        "bidirectional",
        "--subset",
        str(tmp_path / "membership.json"),
        "--output",
        str(output),
        "--guide-mode",
        "d0",
        "--chains-per-rank",
        "2",
        "--steps",
        "1",
        "--save-initial",
        "--save-update-state",
        "--resource-budget",
        str(budget_path),
    ]
    raw = {
        "id": "native-update",
        "kind": "train",
        "arguments": arguments,
        "output": str(output),
        "processes": 4,
        "port": 29509,
        "accelerate_config": str(config),
        "accelerate_config_sha256": sha256(config),
        "dependencies": [],
        "completion": {"checkpoint": str(output / "checkpoints/lora_weights_step_00001.safetensors"), "step": 1},
    }
    job_path = tmp_path / "original_job.json"
    job_path.write_text(json.dumps(raw))
    prepared = queue.prepare_job(raw, tmp_path)
    launch = queue.training_launch_record(prepared)
    output.mkdir()
    (output / "checkpoints").mkdir()
    saved = {"world_size": 4, "queue_job_sha256": prepared["sha256"], "queue_launch": launch}
    policy = {
        "param_dtype": "torch.bfloat16",
        "reduce_dtype": "torch.bfloat16",
        "buffer_dtype": None,
        "cast_forward_inputs": False,
        "cast_root_forward_inputs": False,
        "keep_low_precision_grads": False,
    }
    ranks = [
        {
            "schema_version": 2,
            "rank": rank,
            "world_size": 4,
            "mixed_precision": "bf16",
            "distributed_type": "FSDP",
            "conditioning_precision": "float32",
            "adapter_storage_dtypes": ["torch.float32"],
            "fsdp_policies": [deepcopy(policy)],
            "numerics": {**numerics.POLICY, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        }
        for rank in range(4)
    ]
    saved["runtime"] = {"schema_version": 2, "world_size": 4, "mixed_precision": "bf16", "ranks": ranks}
    budget = resources.read_budget(budget_path)
    saved["resource_budget"] = budget
    for rank in range(4):
        observations = [
            {
                "schema_version": 1,
                "rank": rank,
                "phase": phase,
                "device": f"cuda:{rank}",
                "elapsed_s": 10,
                "peak_allocated_bytes": 1000,
                "peak_reserved_bytes": 2000,
                "budget_sha256": budget["sha256"],
                "state": "passed",
                "error": None,
            }
            for phase in ("load", "export:0", "update:1", "export:1")
        ]
        (output / f"resources_rank{rank}.jsonl").write_text("\n".join(json.dumps(row) for row in observations) + "\n")
        resources.save_snapshot(output, rank, 1)
    marker = deepcopy(saved)
    marker["resource_evidence"] = resources.read_records(output, 4, step=1)[1]
    checkpoint = output / "checkpoints/lora_weights_step_00001.safetensors"
    checkpoint.write_bytes(b"controlled step-one checkpoint bytes; launch validation only")
    marker.update(schema_version=2, state="complete", step=1, path=str(checkpoint), sha256=sha256(checkpoint))
    marker_path = checkpoint.with_suffix(".complete.json")
    (output / "config.json").write_text(json.dumps(saved))
    marker_path.write_text(json.dumps(marker))
    return raw, prepared, launch, job_path, saved, marker_path


def test_single_job_authority_matches_list_and_prepared_reconstruction(tmp_path: Path) -> None:
    raw, prepared, launch, _, _, _ = original_job(tmp_path)
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"schema_version": 1, "jobs": [raw]}))
    assert queue.prepare_jobs(path)[0] == prepared
    assert queue.prepare_job(prepared, tmp_path) == prepared
    assert queue.verify_training_launch(launch, raw) == prepared
    assert base64.b64decode(launch["accelerate_config_bytes_base64"]) == (tmp_path / "fsdp.yaml").read_bytes()
    assert prepared["resource_budget_sha256"] == sha256(tmp_path / "native_protocol.json")


def test_changed_budget_refuses_prepared_command_and_dispatch_launch_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    raw, _, _, _, _, _ = original_job(tmp_path)
    output = tmp_path / "pending_training"
    raw["arguments"][raw["arguments"].index("--output") + 1] = str(output)
    raw["output"] = str(output)
    raw["completion"]["checkpoint"] = str(output / "checkpoints/lora_weights_step_00001.safetensors")
    prepared = queue.prepare_job(raw, tmp_path)
    budget_path = tmp_path / "native_protocol.json"
    # Preserve valid values while changing the exact predeclared bytes.
    budget_path.write_text(budget_path.read_text() + "\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("started a child"))
    with pytest.raises(ValueError, match="resource budget changed"):
        queue.prepare_job(prepared, tmp_path)
    with pytest.raises(ValueError, match="resource budget changed"):
        queue.job_command(prepared, queue.TRAIN_GPUS)
    with pytest.raises(ValueError, match="resource budget changed"):
        queue.training_launch_record(prepared)
    released = []
    claims = SimpleNamespace(owned=set(queue.TRAIN_GPUS), release=lambda: released.append(True))
    with pytest.raises(ValueError, match="resource budget changed"):
        queue.run_child(prepared, claims, tmp_path / "logs/child.log")
    assert released == [True]
    assert not output.exists()
    assert not (tmp_path / "logs").exists()


def test_claimed_budget_hash_must_match_actual_bytes(tmp_path: Path) -> None:
    raw, _, _, _, _, _ = original_job(tmp_path)
    raw["resource_budget_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="resource budget changed from its claimed hash"):
        queue.prepare_job(raw, tmp_path)




@pytest.mark.parametrize("step", [0, 1])
def test_shared_marker_reader_checks_actual_bytes_for_zero_and_one(tmp_path: Path, step: int) -> None:
    checkpoint = tmp_path / f"lora_weights_step_{step:05d}.safetensors"
    checkpoint.write_bytes(b"controlled checkpoint bytes; readiness only")
    marker = {"schema_version": 2, "state": "complete", "step": step,
              "path": str(checkpoint), "sha256": sha256(checkpoint)}
    checkpoint.with_suffix(".complete.json").write_text(json.dumps(marker))
    assert queue.read_training_marker(checkpoint, step) == marker
    checkpoint.write_bytes(b"changed checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint completion marker"):
        queue.read_training_marker(checkpoint, step)
















@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("defect", ["world", "precision", "distributed"])
def test_engine_actual_accelerator_refuses_checked_queued_launch_before_models_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, defect: str
) -> None:
    from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_ENV

    raw, _, _, _, _, _ = original_job(tmp_path)
    output = tmp_path / "fresh_training"
    raw["arguments"][raw["arguments"].index("--mode") + 1] = mode
    raw["arguments"][raw["arguments"].index("--output") + 1] = str(output)
    raw["output"] = str(output)
    raw["completion"]["checkpoint"] = str(output / "checkpoints/lora_weights_step_00001.safetensors")
    job = queue.prepare_job(raw, tmp_path)
    launch_path = tmp_path / "fresh_training.launch.json"
    launch_path.write_text(json.dumps(queue.training_launch_record(job)))
    monkeypatch.setenv(LAUNCH_ENV, str(launch_path))
    for key, value in numerics.ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    settings = config.parse_settings(job["arguments"])
    store = SimpleNamespace(membership={"sha256": "e" * 64})
    plan = {"sha256": "d" * 64, "samples": [{"split": settings.split}]}
    monkeypatch.setattr(engine, "prepare_run", lambda *_args, **_kwargs: (store, plan, None, False))
    actual = SimpleNamespace(
        num_processes=2 if defect == "world" else 4,
        mixed_precision="no" if defect == "precision" else "bf16",
        distributed_type=DistributedType.MULTI_GPU if defect == "distributed" else DistributedType.FSDP,
        device=torch.device("cpu"), process_index=0,
    )
    monkeypatch.setattr(engine, "Accelerator", lambda: actual)
    monkeypatch.setattr(engine.prompt_cache, "get_or_build", lambda *_args, **_kwargs: pytest.fail("loaded text"))
    monkeypatch.setattr(engine, "build_transformer", lambda *_args: pytest.fail("loaded model"))
    with pytest.raises(ValueError, match="actual Accelerator"):
        engine._run_settings(settings, SimpleNamespace(job=job["sha256"], token="b" * 32))
    assert not output.exists()
