"""Shared saved-input builders and fixtures; never a collected test module."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import adapter_effect_check as effect
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters
from scripts.onestep_avatar.tests.support.contracts import _contract
from scripts.onestep_avatar.tests.support.corpus import (
    evaluation_membership as evaluation_membership,  # noqa: PLC0414 -- explicit fixture re-export
)
from scripts.onestep_avatar.training import checkpoints, config, numerics, resources


@pytest.fixture(params=[False, True], ids=["one_update", "pilot"])
def saved_effect(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Controlled CPU tensors/journals test integrity; they are not native acceptance."""
    directory = tmp_path / "effect"
    directory.mkdir()
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps({"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48_000_000_000}))
    budget = resources.read_budget(budget_path)
    pilot = request.param
    states = effect.PILOT_STATES if pilot else effect.STATES
    steps = [0, 20, 60] if pilot else [0, 1]
    selected = {}
    for state, step in zip(states[1:], steps, strict=True):
        contract = _contract()
        contract["adapter"]["step"] = step
        values = {
            name: torch.ones(shape, dtype=torch.bfloat16)
            for name, shape in contract["adapter"]["tensor_shapes"].items()
        }
        for name, value in values.items():
            if ".lora_B." in name:
                value.fill_(0 if step == 0 else 0.125)
        checkpoint = tmp_path / f"{state}.safetensors"
        save_file(values, checkpoint, metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        selected[state] = {"path": str(checkpoint), "sha256": sha256(checkpoint), "contract": contract}
    noise_path = tmp_path / "noise.pt"
    torch.save(torch.zeros(1, 28, 128, dtype=torch.bfloat16), noise_path)
    source = "actor/views/view01_cam57"
    membership = evaluation_membership(tmp_path, [source])
    membership_path = tmp_path / "evaluation.json"
    membership_path.write_text(json.dumps(membership))
    membership_identity = {
        "path": str(membership_path),
        "sha256": sha256(membership_path),
        "membership_sha256": membership["sha256"],
    }
    pins = {
        str(path): sha256(path)
        for path in (budget_path, noise_path, membership_path, *(Path(item["path"]) for item in selected.values()))
    }
    policy = {**numerics.POLICY, "cudnn_allow_tf32": True}
    producer = software.capture("evaluation", "bidirectional", extra_sources=effect.EXTRA_SOURCES)
    protocol = {
        "kind": "onestep_avatar.adapter_effect_check",
        "schema_version": 1,
        "scope": "shared_API_adapter_correction",
        "software": producer,
        "input_files": pins,
        "resource_budget": budget,
        "checkpoints": selected,
        "evaluation_membership": membership_identity,
        "tolerance": effect.TOLERANCE,
        "schedule": [0.725, 0.0],
        "seed": 42,
        "runtime": {"world_size": 1, "device": "cuda:0", "numerics": policy},
    }
    if pilot:
        # This fixture tests the saved array/journal reader. Actual producer lineage has separate controls.
        def check_protocol(_protocol: dict, _prepared: dict, paths: dict, contracts: dict) -> None:
            assert set(paths) == {"zero", "step20", "step60"}
            assert sorted(row["adapter"]["step"] for row in contracts.values()) == steps

        monkeypatch.setattr(effect, "_verify_pilot_protocol", check_protocol)
    (directory / "protocol.json").write_text(json.dumps(protocol))
    conditions = {
        "application_method": adapters.UNMERGED,
        "global_sigma_dtype": "float32",
        "mode": "bidirectional",
        "mode_settings": asdict(config.BidirectionalSettings(7)),
        "model": contract["model"],
        "task": contract["task"],
        "shape": {"channels": 128, "height": 2, "width": 2, "frames": 7},
        "schedule": [0.725, 0.0],
    }
    base = torch.zeros(1, 128, 7, 2, 2, dtype=torch.bfloat16)
    trained = base.clone()
    trained[:, :, 1:] = 0.001
    outputs = {
        path: {
            "base": base.clone(),
            "zero": base.clone(),
            **{state: trained.clone() * (2 if state == "step60" else 1) for state in states[2:]},
        }
        for path in effect.PATHS
    }
    raw, journals = {}, []
    for path in effect.PATHS:
        raw[path] = {}
        for state in states:
            record = {
                "path": path,
                "adapter_state": state,
                "source": source,
                "source_coverage": [0, 7],
                "fps": 30,
                "seed": 42,
                "software": producer,
                "input_files": pins,
                "numerics": policy,
                "conditions": conditions,
                "checkpoint": None if state == "base" else selected[state]["path"],
                "application_method": "base" if state == "base" else adapters.UNMERGED,
                "call_counts": {"model_calls": 1},
                **dict.fromkeys(("guide_sha256", "c0_sha256", "noise_sha256", "text_sha256"), "1" * 64),
            }
            raw[path][state] = evaluate.save_case(outputs[path][state], record, directory / "case_00" / path / state)
            for operation in ("load", "sample"):
                journals.append(
                    {
                        "schema_version": 1,
                        "phase": f"{operation}:{source}:{path}:{state}",
                        "rank": 0,
                        "device": "cuda:0",
                        "elapsed_s": 0.1,
                        "peak_allocated_bytes": 1,
                        "peak_reserved_bytes": 1,
                        "budget_sha256": budget["sha256"],
                        "state": "passed",
                        "error": None,
                    }
                )
    journal = directory / "case_00/resources_rank0.jsonl"
    journal.write_text("".join(json.dumps(item) + "\n" for item in journals))
    result = {
        "schema_version": 1,
        "protocol": protocol,
        "state": "passed",
        "e2_complete": False,
        "coverage": {"sources": 1, "views": 1, "camera_labels": ["cam57"], "checkpoint_steps": steps, "decoded": False},
        "cases": [
            {
                "source": source,
                "source_coverage": [0, 7],
                "comparison": effect.compare_paths(outputs),
                "raw_records": raw,
                "renderings": [],
            }
        ],
        "resource_measurements": journals,
        "output_files": {
            str(path.relative_to(directory)): sha256(path) for path in directory.rglob("*") if path.is_file()
        },
    }
    (directory / "result.json").write_text(json.dumps(result))
    return directory, result, noise_path


build_saved_effect = saved_effect.__wrapped__
