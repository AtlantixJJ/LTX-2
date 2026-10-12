"""Shared saved-input builders and fixtures; never a collected test module."""

import json
from copy import deepcopy
from pathlib import Path

import torch

from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.training import numerics, resources


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
    prepared = queue_jobs.prepare_job(raw, tmp_path)
    launch = queue_jobs.training_launch_record(prepared)
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
