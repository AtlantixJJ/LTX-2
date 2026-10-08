"""Original launch controls use the real queue normalization and replay gate."""

import base64
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from scripts.onestep_avatar import queue
from scripts.onestep_avatar import training_update_check as check
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.training import numerics, resources


def original_job(tmp_path: Path, *, tolerance: dict | None = check.TOLERANCE) -> tuple:
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "pending"),
        ("schema_version", 1),
        ("schema_version", True),
        ("step", 0),
        ("step", True),
        ("path", "/forged/adapter.safetensors"),
        ("sha256", "f" * 64),
    ],
)
def test_replay_readiness_rejects_changed_marker_before_scientific_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value
) -> None:
    _, _, _, job_path, _, marker_path = original_job(tmp_path)
    marker = json.loads(marker_path.read_text())
    marker[field] = value
    marker_path.write_text(json.dumps(marker))
    monkeypatch.setattr(check.engine, "prepare_run", lambda *_args, **_kwargs: pytest.fail("opened scientific inputs"))
    monkeypatch.setattr(check.engine, "build_transformer", lambda *_args: pytest.fail("loaded model"))
    output = tmp_path / "serial"
    with pytest.raises(ValueError, match="checkpoint completion marker"):
        check.execute(job_path, output, 4)
    assert not output.exists()


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


def test_unchanged_launch_passes_actual_runtime_and_resource_gate(tmp_path: Path) -> None:
    _, prepared, _, path, _, _ = original_job(tmp_path)
    checked, _, precision = check.check_launch(path, 4)
    assert checked == prepared
    assert precision == "bf16"


@pytest.mark.parametrize(
    "tolerance",
    [
        None,
        {},
        {**check.TOLERANCE, "relative_l2": 0.03},
        {**check.TOLERANCE, "near_zero_rms": 1e-7},
        {**check.TOLERANCE, "absolute_rms": 1e-7},
    ],
)
def test_fully_bound_wrong_frozen_tolerance_refuses_before_scientific_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tolerance: dict | None
) -> None:
    _, prepared, launch, job_path, _, _ = original_job(tmp_path, tolerance=tolerance)
    # All job, marker and resource hashes agree with these predeclared bytes.
    assert queue.verify_training_launch(launch, prepared) == prepared
    monkeypatch.setattr(check.engine, "prepare_run", lambda *_args, **_kwargs: pytest.fail("opened scientific inputs"))
    monkeypatch.setattr(check.engine, "build_transformer", lambda *_args: pytest.fail("loaded model"))
    output = tmp_path / "serial"
    with pytest.raises(ValueError, match="frozen resource budget tolerance"):
        check.execute(job_path, output, 4)
    assert not output.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "runtime_world",
        "runtime_precision",
        "rank_disagreement",
        "cast_inputs",
        "adapter_dtype",
        "conditioning_precision",
        "missing_runtime",
        "missing_budget",
        "changed_budget",
        "missing_measurement",
        "allocated_limit",
        "wall_limit",
        "changed_snapshot",
        "journal_disagreement",
    ],
)
def test_applied_runtime_and_resource_defects_refuse_before_scientific_inputs(  # noqa: PLR0912 -- explicit negative controls
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    _, _, _, path, saved, marker_path = original_job(tmp_path)
    if defect == "runtime_world":
        saved["runtime"]["world_size"] = 2
    elif defect == "runtime_precision":
        saved["runtime"]["mixed_precision"] = "no"
    elif defect == "rank_disagreement":
        saved["runtime"]["ranks"][2]["fsdp_policies"][0]["reduce_dtype"] = "torch.float32"
    elif defect in ("cast_inputs", "adapter_dtype", "conditioning_precision"):
        for rank in saved["runtime"]["ranks"]:
            if defect == "cast_inputs":
                rank["fsdp_policies"][0]["cast_root_forward_inputs"] = True
            elif defect == "adapter_dtype":
                rank["adapter_storage_dtypes"] = ["torch.bfloat16"]
            else:
                rank["conditioning_precision"] = "bfloat16"
    elif defect == "missing_runtime":
        saved.pop("runtime")
    elif defect == "missing_budget":
        saved.pop("resource_budget")
    elif defect == "changed_budget":
        (tmp_path / "native_protocol.json").write_text(
            json.dumps({"wall_seconds_per_phase": 3600, "memory_limit_allocated_bytes": 48000000000})
        )
    else:
        resource_path = tmp_path / "distributed/resources_rank2.jsonl"
        snapshot_path = tmp_path / "distributed/resource_snapshots/step_00001/resources_rank2.jsonl"
        observations = [json.loads(line) for line in resource_path.read_text().splitlines()]
        if defect == "missing_measurement":
            observations.pop()
        elif defect == "allocated_limit":
            observations[-1]["peak_allocated_bytes"] = 48000000001
            observations[-1]["peak_reserved_bytes"] = 48000000002
        elif defect == "wall_limit":
            observations[-1]["elapsed_s"] = 1801
        elif defect == "changed_snapshot":
            snapshot = [json.loads(line) for line in snapshot_path.read_text().splitlines()]
            snapshot[-1]["peak_allocated_bytes"] += 1
            snapshot_path.write_text("\n".join(json.dumps(row) for row in snapshot) + "\n")
        else:
            observations[-1]["peak_allocated_bytes"] += 1
        resource_path.write_text("\n".join(json.dumps(row) for row in observations) + "\n")
        if defect in ("missing_measurement", "allocated_limit", "wall_limit"):
            snapshot_path.write_bytes(resource_path.read_bytes())
    (tmp_path / "distributed/config.json").write_text(json.dumps(saved))
    marker = json.loads(marker_path.read_text())
    if "runtime" in saved:
        marker["runtime"] = saved["runtime"]
    if defect != "changed_snapshot":
        marker["resource_evidence"] = resources.read_records(tmp_path / "distributed", 4, step=1)[1]
    marker_path.write_text(json.dumps(marker))
    monkeypatch.setattr(check.engine, "prepare_run", lambda *_args, **_kwargs: pytest.fail("opened scientific inputs"))
    with pytest.raises(
        ValueError, match=r"launch|precision|runtime|resource|FSDP|rank|queue|budget|identity|snapshot|world"
    ):
        check.execute(path, tmp_path / "serial", 4)
    assert not (tmp_path / "serial").exists()


@pytest.mark.parametrize(
    "defect",
    [
        "yaml",
        "processes",
        "hash",
        "port",
        "command",
        "snapshot",
        "saved_world",
        "saved_hash",
        "marker_hash",
        "missing_launch",
    ],
)
def test_replay_rejects_changed_original_launch_before_inputs_models_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    raw, _, _launch, job_path, saved, marker_path = original_job(tmp_path)
    if defect == "yaml":
        (tmp_path / "fsdp.yaml").write_text("mixed_precision: 'no'\n")
    elif defect == "processes":
        raw["processes"] = 2
    elif defect == "hash":
        raw["sha256"] = "f" * 64
    elif defect == "port":
        raw["port"] += 1
    elif defect == "command":
        saved["queue_launch"]["command"][0] = "/forged/python"
    elif defect == "snapshot":
        saved["queue_launch"]["accelerate_config_bytes_base64"] = base64.b64encode(b"mixed_precision: no").decode()
    elif defect == "saved_world":
        saved["world_size"] = 2
    elif defect == "saved_hash":
        saved["queue_job_sha256"] = "e" * 64
    elif defect == "marker_hash":
        marker = json.loads(marker_path.read_text())
        marker["queue_job_sha256"] = "e" * 64
        marker_path.write_text(json.dumps(marker))
    else:
        del saved["queue_launch"]
    job_path.write_text(json.dumps(raw))
    (tmp_path / "distributed/config.json").write_text(json.dumps(saved))
    monkeypatch.setattr(check.engine, "prepare_run", lambda *_args, **_kwargs: pytest.fail("opened scientific inputs"))
    monkeypatch.setattr(check.engine, "build_transformer", lambda *_args, **_kwargs: pytest.fail("loaded model"))
    replay_output = tmp_path / "serial"
    with pytest.raises(
        ValueError, match=r"launch|precision|runtime|resource|FSDP|rank|queue|budget|identity|snapshot|world"
    ):
        check.execute(job_path, replay_output, 4)
    assert not replay_output.exists()


def test_late_launch_change_blocks_replay_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, job_path, _, _ = original_job(tmp_path)
    monkeypatch.setattr(check.software, "check_current", lambda _record: None)
    identities = {str(job_path): sha256(job_path), str(tmp_path / "fsdp.yaml"): sha256(tmp_path / "fsdp.yaml")}
    (tmp_path / "fsdp.yaml").write_text("mixed_precision: 'no'\n")
    with pytest.raises(ValueError, match="input changed"):
        check.check_current(identities, {}, job_path=job_path)
    assert not (tmp_path / "serial/result.json").exists()


@pytest.mark.parametrize("defect", ["world", "precision", "distributed"])
def test_actual_serial_accelerator_refuses_before_models_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    _, _, _, job_path, _, _ = original_job(tmp_path)
    settings = check.config.parse_settings(json.loads(job_path.read_text())["arguments"])
    # Preparation is model-free and separately checked; this isolates the actual
    # Accelerator setup gate from the synthetic distributed evidence.
    monkeypatch.setattr(check, "prepare", lambda *_args: (settings, None, None, None, [], [], {}, {}, {}, None))
    monkeypatch.setattr(check.software, "capture", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(check.software, "check_current", lambda *_args: None)
    actual = SimpleNamespace(
        num_processes=2 if defect == "world" else 1,
        mixed_precision="no" if defect == "precision" else "bf16",
        distributed_type=DistributedType.FSDP if defect == "distributed" else DistributedType.NO,
    )
    monkeypatch.setattr(check, "Accelerator", lambda **_kwargs: actual)
    monkeypatch.setattr(check.engine, "build_transformer", lambda *_args: pytest.fail("loaded model"))
    output = tmp_path / "serial"
    with pytest.raises(ValueError, match="actual Accelerator|ordinary process"):
        check.execute(job_path, output, 4)
    assert not output.exists()


def test_input_change_during_serial_accelerator_setup_refuses_before_models_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, job_path, _, _ = original_job(tmp_path)
    settings = check.config.parse_settings(json.loads(job_path.read_text())["arguments"])
    budget_path = tmp_path / "native_protocol.json"
    identities = {str(budget_path): sha256(budget_path)}
    monkeypatch.setattr(check, "prepare", lambda *_args: (settings, None, None, None, [], [], {}, {}, identities, None))
    monkeypatch.setattr(check.software, "capture", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(check.software, "check_current", lambda *_args: None)

    def accelerator(**_kwargs):
        budget_path.write_text(budget_path.read_text() + "\n")
        return SimpleNamespace(num_processes=1, mixed_precision="bf16", distributed_type=DistributedType.NO)

    monkeypatch.setattr(check, "Accelerator", accelerator)
    monkeypatch.setattr(check.engine, "build_transformer", lambda *_args: pytest.fail("loaded model"))
    output = tmp_path / "serial"
    with pytest.raises(ValueError, match="input changed"):
        check.execute(job_path, output, 4)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("defect", ["world", "precision", "distributed"])
def test_engine_actual_accelerator_refuses_checked_queued_launch_before_models_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, defect: str
) -> None:
    from scripts.onestep_avatar.queue_protocol import LAUNCH_ENV

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
    settings = check.config.parse_settings(job["arguments"])
    store = SimpleNamespace(membership={"sha256": "e" * 64})
    plan = {"sha256": "d" * 64, "samples": [{"split": settings.split}]}
    monkeypatch.setattr(check.engine, "prepare_run", lambda *_args, **_kwargs: (store, plan, None, False))
    actual = SimpleNamespace(
        num_processes=2 if defect == "world" else 4,
        mixed_precision="no" if defect == "precision" else "bf16",
        distributed_type=DistributedType.MULTI_GPU if defect == "distributed" else DistributedType.FSDP,
        device=torch.device("cpu"), process_index=0,
    )
    monkeypatch.setattr(check.engine, "Accelerator", lambda: actual)
    monkeypatch.setattr(check.engine.prompt_cache, "get_or_build", lambda *_args, **_kwargs: pytest.fail("loaded text"))
    monkeypatch.setattr(check.engine, "build_transformer", lambda *_args: pytest.fail("loaded model"))
    with pytest.raises(ValueError, match="actual Accelerator"):
        check.engine._run_settings(settings, SimpleNamespace(job=job["sha256"], token="b" * 32))
    assert not output.exists()


@pytest.mark.parametrize("world", [1, 2, 8])
def test_requested_world_cannot_replace_original_processes(tmp_path: Path, world: int) -> None:
    _, _, _, job_path, _, _ = original_job(tmp_path)
    with pytest.raises(ValueError, match="world"):
        check.check_launch(job_path, world)


def test_replay_phase_preserves_execution_failure(tmp_path: Path) -> None:
    output = tmp_path / "serial"
    output.mkdir()
    records = []
    with (
        pytest.raises(RuntimeError, match="controlled update failure"),
        check.measured_phase(output, torch.device("cpu"), "update", None, records),
    ):
        raise RuntimeError("controlled update failure")
    saved = json.loads((output / "resources_rank0.jsonl").read_text())
    assert saved == records[0]
    assert saved["state"] == "failed"
    assert "controlled update failure" in saved["error"]
    assert not (output / "result.json").exists()


@pytest.mark.parametrize("limit", ["allocated", "wall"])
def test_replay_preserves_measured_limit_breach_before_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    _, _, _, _, original, _ = original_job(tmp_path)
    budget = original["resource_budget"]
    monkeypatch.setattr(resources.torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(resources.torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    allocated = budget["memory_limit_allocated_bytes"] + 1 if limit == "allocated" else 1000
    monkeypatch.setattr(resources.torch.cuda, "max_memory_allocated", lambda *_: allocated)
    monkeypatch.setattr(resources.torch.cuda, "max_memory_reserved", lambda *_: allocated + 1)
    ticks = iter([0, 1801 if limit == "wall" else 10])
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(ticks))
    output = tmp_path / "serial"
    output.mkdir()
    records = []
    with (
        pytest.raises(ValueError, match="exceeded"),
        check.measured_phase(output, torch.device("cuda:0"), "update", budget, records),
    ):
        pass
    saved = json.loads((output / "resources_rank0.jsonl").read_text())
    assert saved == records[0]
    assert saved["state"] == "failed"
    assert saved["peak_allocated_bytes"] == allocated
    assert not (output / "result.json").exists()
