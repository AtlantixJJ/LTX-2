"""Version-two records and actual exported matrix shapes gate adapter use."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from safetensors.torch import save_file

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import subset
from scripts.onestep_avatar.training import checkpoints, config

A = "diffusion_model.block.to_q.lora_A.weight"
B = "diffusion_model.block.to_q.lora_B.weight"


def _contract(mode: str = "bidirectional", teacher: bool = False, *, random_start: bool = False) -> dict:
    options = [
        "--mode",
        mode,
        "--subset",
        "/unused/list",
        "--output",
        "/unused/output",
        "--variant",
        "dev",
        "--objective",
        "white",
        "--lora-rank",
        "2",
    ]
    if mode == "bidirectional":
        options += ["--span-latent-frames", "7"]
    elif teacher:
        options += ["--teacher-forcing"]
    if random_start:
        options += ["--start-policy", "random"]
        if mode == "causal":
            options += ["--blocks-per-sample", "1", "--span-latent-frames", "3"]
    settings = config.parse_settings(options)
    settings.base_identity = {"base_transformer_file": "base.safetensors", "base_transformer_sha256": "a" * 64}
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "corpus_root": "/unused",
        "splits": {"train": ["actor"]},
        "sources": [
            {
                "relative_dir": "actor/view",
                "actor": "actor",
                "split": "train",
                "n_latent_frames": 7,
                "shape": [128, 7, 2, 2],
                "fps": 30,
            }
        ],
        "excluded": {},
    }
    membership["sha256"] = subset.membership_hash(membership)
    plan = config.build_frame_plan(settings, membership, SpatioTemporalScaleFactors(8, 32, 32))
    record = checkpoints.make_contract(settings, membership, plan, step=0)
    record["adapter"]["tensor_shapes"] = {A: [2, 4], B: [4, 2]}
    return record


def test_new_random_adapter_records_template_and_start_draw() -> None:
    record = _contract(random_start=True)
    selection = record["data"]["segment_selection"]
    assert selection["coverage_role"] == "window_templates"
    assert selection["start_bounds_inclusive"] == {"actor/view": [0, 0]}
    assert selection["start_draw"] == config.window_start_draw(record["training"]["seeds"]["noise"])
    assert selection["known_gap"] == "G9"


@pytest.mark.parametrize("defect", ["missing", "seed", "bounds", "first_image", "positions", "known_gap"])
def test_random_adapter_refuses_missing_or_changed_selection(defect: str) -> None:
    record = _contract(random_start=True)
    selection = record["data"]["segment_selection"]
    if defect == "missing":
        record["data"].pop("segment_selection")
    elif defect == "seed":
        selection["start_draw"]["seed"] += 1
    elif defect == "bounds":
        selection["start_bounds_inclusive"]["actor/view"] = [0, 1]
    else:
        selection[defect] = "different"
    with pytest.raises(ValueError, match="random segment selection"):
        checkpoints.validate_contract(record)


def test_random_causal_adapter_cannot_claim_multiple_blocks() -> None:
    record = _contract("causal", random_start=True)
    record["mode_settings"]["blocks_per_sample"] = 2
    record["causal"]["blocks_per_sample"] = 2
    with pytest.raises(ValueError, match="one block and window length"):
        checkpoints.validate_contract(record)


def test_random_full_master_templates_have_only_zero_start() -> None:
    selection = checkpoints.random_segment_selection(
        [{"source": "a", "ranges": [[0, 5]]}, {"source": "b", "ranges": [[0, 9]]}],
        {"a": 5, "b": 9}, None, 42,
    )
    assert selection["start_bounds_inclusive"] == {"a": [0, 0], "b": [0, 0]}
    assert selection["window_latent_frames"] is None


def _request(record: dict) -> dict:
    requested = {
        "application_method": "peft_unmerged_fp32",
        "global_sigma_dtype": "float32",
        "mode": record["mode"],
        "model": copy.deepcopy(record["model"]),
        "task": copy.deepcopy(record["task"]),
        "shape": {"channels": 128, "height": 2, "width": 2, "frames": 7},
        "schedule": [0.725, 0],
        "mode_settings": copy.deepcopy(record["mode_settings"]),
    }
    if record["mode"] == "causal":
        requested.update(history_mode="cache", kv_source="refresh")
    return requested


def test_metadata_roundtrip_and_actual_tensor_inventory(tmp_path: Path) -> None:
    record = _contract()
    path = tmp_path / "adapter.safetensors"
    save_file(
        {A: torch.ones(2, 4), B: torch.zeros(4, 2)}, path, metadata={checkpoints.CONTRACT_KEY: json.dumps(record)}
    )
    read = checkpoints.read_contract(path)
    assert read == record
    checkpoints.validate_adapter_tensors(path, read)
    request = _request(read)
    request["evaluation_people"] = ["another person"]
    request["membership_sha256"] = "new evaluation data"
    assert checkpoints.check_contract(read, request, product=True) == []


@pytest.mark.parametrize("field", ["mode", "model", "task", "shape", "schedule", "mode_settings"])
def test_changed_conditions_require_explicit_research_override(field: str) -> None:
    record = _contract()
    request = _request(record)
    if field == "mode":
        request[field] = "causal"
        request.update(history_mode="cache", kv_source="refresh")
    elif field == "model":
        request[field]["base_sha256"] = "b" * 64
    elif field == "task":
        request[field]["guide_mode"] = "d0"
    elif field == "shape":
        request[field]["height"] = 3
    elif field == "schedule":
        request[field] = [0.725, 0.421875, 0]
    else:
        request[field]["start_policy"] = "random"
    with pytest.raises(ValueError, match="incompatible"):
        checkpoints.check_contract(record, request)
    assert checkpoints.check_contract(record, request, override=True)
    with pytest.raises(ValueError, match="incompatible"):
        checkpoints.check_contract(record, request, override=True, product=True)


def test_generated_history_is_a_recorded_change_for_capture_history_adapter() -> None:
    record = _contract("causal", teacher=True)
    request = _request(record)
    request["mode_settings"]["teacher_forcing"] = False
    differences = checkpoints.check_contract(record, request, override=True)
    assert any("mode_settings" in difference for difference in differences)
    with pytest.raises(ValueError):
        checkpoints.check_contract(record, request, product=True)


@pytest.mark.parametrize("field,value", [("history_mode", "recompute"), ("history_mode", "joint"),
                                         ("kv_source", "denoise")])
def test_changed_causal_computation_requires_override(field, value):
    record = _contract("causal")
    request = _request(record)
    assert checkpoints.check_contract(record, request, product=True) == []
    request[field] = value
    with pytest.raises(ValueError, match="incompatible.*" + field):
        checkpoints.check_contract(record, request)
    differences = checkpoints.check_contract(record, request, override=True)
    assert len(differences) == 1
    assert field in differences[0] and repr(value) in differences[0]
    assert "cached_refresh_global_sigma0" in differences[0]
    with pytest.raises(ValueError, match="incompatible.*" + field):
        checkpoints.check_contract(record, request, override=True, product=True)


@pytest.mark.parametrize("field", ["history_mode", "kv_source"])
@pytest.mark.parametrize("value", [None, "unsupported", 1, [], {}])
def test_malformed_causal_computation_cannot_be_overridden(field, value):
    record = _contract("causal")
    request = _request(record)
    request[field] = value
    with pytest.raises(ValueError, match="missing or unsupported"):
        checkpoints.check_contract(record, request, override=True)
    del request[field]
    with pytest.raises(ValueError, match="missing or unsupported"):
        checkpoints.check_contract(record, request, override=True)


@pytest.mark.parametrize("history,teacher", [("recompute", False), ("joint", False), ("cache", True)])
def test_invalid_denoise_history_pair_cannot_be_overridden(history, teacher):
    record = _contract("causal", teacher=teacher)
    request = _request(record)
    request.update(history_mode=history, kv_source="denoise")
    with pytest.raises(ValueError, match="denoise K/V requires cached generated history"):
        checkpoints.check_contract(record, request, override=True)


def test_bidirectional_request_cannot_add_history_with_override():
    record = _contract()
    request = _request(record)
    request.update(history_mode="cache", kv_source="refresh")
    with pytest.raises(ValueError, match="bidirectional requests cannot contain"):
        checkpoints.check_contract(record, request, override=True)


def test_fused_application_requires_recorded_override_and_product_refuses_it():
    record = _contract()
    request = _request(record)
    request['application_method'] = 'fused_bf16'
    with pytest.raises(ValueError, match='incompatible.*application_method'):
        checkpoints.check_contract(record, request)
    differences = checkpoints.check_contract(record, request, override=True)
    assert len(differences) == 1 and 'peft_unmerged_fp32' in differences[0] and 'fused_bf16' in differences[0]
    with pytest.raises(ValueError, match='incompatible.*application_method'):
        checkpoints.check_contract(record, request, override=True, product=True)
    del request['application_method']
    with pytest.raises(ValueError, match='missing or unsupported'):
        checkpoints.check_contract(record, request, override=True)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_mode",
        "base_hash",
        "shape",
        "sigma_nan",
        "sigma_duplicate",
        "alpha",
        "rank",
        "tensor_rank",
        "missing_tensor_record",
        "history_in_bidirectional",
        "missing_data_hash",
    ],
)
def test_incomplete_or_unsupported_records_fail(defect: str) -> None:
    record = _contract()
    if defect == "missing_mode":
        record.pop("mode")
    elif defect == "base_hash":
        record["model"].pop("base_sha256")
    elif defect == "shape":
        record["shape"]["channels"] = 0
    elif defect == "sigma_nan":
        record["training"]["sigma_levels"] = [float("nan")]
    elif defect == "sigma_duplicate":
        record["training"]["sigma_levels"] = [0.725, 0.725]
    elif defect == "alpha":
        record["adapter"]["alpha"] = 1
    elif defect == "rank":
        record["adapter"]["rank"] = False
    elif defect == "tensor_rank":
        record["adapter"]["tensor_shapes"][B] = [4, 3]
    elif defect == "missing_tensor_record":
        record["adapter"].pop("tensor_shapes")
    elif defect == "missing_data_hash":
        record["data"].pop("membership_sha256")
    else:
        record["causal"] = {}
    with pytest.raises(ValueError):
        checkpoints.validate_contract(record)


def test_old_unclassified_adapter_is_not_silently_accepted(tmp_path: Path) -> None:
    path = tmp_path / "old.safetensors"
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, path, metadata={"onestep_avatar_attention": "block_causal"})
    with pytest.raises(ValueError, match="explicitly convert"):
        checkpoints.read_contract(path)


def test_actual_shape_mutation_fails_before_transformer_loading(tmp_path: Path) -> None:
    record = _contract()
    path = tmp_path / "wrong_shape.safetensors"
    save_file(
        {A: torch.ones(2, 4), B: torch.zeros(4, 3)}, path, metadata={checkpoints.CONTRACT_KEY: json.dumps(record)}
    )
    with pytest.raises(ValueError, match="shape differs"):
        checkpoints.validate_adapter_tensors(path, record)


def test_atomic_save_records_actual_shapes_and_preserves_destination_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        "base_model.model.block.to_q.lora_A.weight": torch.ones(2, 4),
        "base_model.model.block.to_q.lora_B.weight": torch.zeros(4, 2),
    }
    accelerator = SimpleNamespace(
        wait_for_everyone=lambda: None,
        get_state_dict=lambda model: state,
        is_main_process=True,
        unwrap_model=lambda model, **kw: model,
        distributed_type=DistributedType.NO,
    )
    monkeypatch.setattr(checkpoints, "get_peft_model_state_dict", lambda *args, **kwargs: state)
    record = _contract()
    record["adapter"]["tensor_shapes"] = {}
    metadata = {checkpoints.CONTRACT_KEY: json.dumps(record)}
    path = checkpoints.save_lora(torch.nn.Linear(1, 1), accelerator, tmp_path, 0, metadata, verify_noop=True)
    read = checkpoints.read_contract(path)
    assert read["adapter"]["tensor_shapes"] == {A: [2, 4], B: [4, 2]}
    checkpoints.validate_adapter_tensors(path, read)
    before = path.read_bytes()

    def fail_after_write(tensors, path, metadata):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("save failed")

    monkeypatch.setattr(checkpoints, "save_file", fail_after_write)
    with pytest.raises(RuntimeError, match="save failed"):
        checkpoints.save_lora(torch.nn.Linear(1, 1), accelerator, tmp_path, 0, metadata)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp.*"))


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_sigma_precision_contract_preserves_history_but_requires_known_execution(mode):
    record = _contract(mode)
    assert record["training"]["global_sigma_dtype"] == "float32"
    requested = _request(record)
    assert checkpoints.check_contract(record, requested) == []
    historical = copy.deepcopy(record)
    del historical["training"]["global_sigma_dtype"]
    checkpoints.validate_contract(historical)
    for override in (False, True):
        with pytest.raises(ValueError, match="global_sigma_dtype is unknown"):
            checkpoints.check_contract(historical, requested, override=override)
    calibrated = copy.deepcopy(record)
    calibrated["training"]["global_sigma_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="incompatible.*global_sigma_dtype"):
        checkpoints.check_contract(calibrated, requested)
    differences = checkpoints.check_contract(calibrated, requested, override=True)
    assert len(differences) == 1 and "global_sigma_dtype" in differences[0]
    with pytest.raises(ValueError, match="incompatible.*global_sigma_dtype"):
        checkpoints.check_contract(calibrated, requested, override=True, product=True)
    for value in (None, "float16", "", 32):
        requested["global_sigma_dtype"] = value
        with pytest.raises(ValueError, match="requested global_sigma_dtype"):
            checkpoints.check_contract(record, requested, override=True)
