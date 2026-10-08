"""Actual tiny LTX/PEFT consumers, native cast control and bounded evidence behavior."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.distributed.fsdp._runtime_utils import _cast_forward_inputs

from scripts.onestep_avatar.model import adapters, bidirectional, causal, common
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.training.consumer_trace import Trace, validate

BINDING = {
    "rank": 0,
    "world_size": 1,
    "queue_job_sha256": "a" * 64,
    "queue_attempt_token": "bounded-test-attempt",
    "launch_sha256": "b" * 64,
}


def _fixture(*, bf16: bool = True, checkpointing: bool = False) -> tuple:
    base = _model()
    if bf16:
        base = base.bfloat16()
    model = adapters.attach(base, rank=2, alpha=2, target="attn", init_seed=11).train()
    base.set_gradient_checkpointing(checkpointing)
    for name, parameter in model.named_parameters():
        if ".lora_B." in name:
            with torch.no_grad():
                parameter.fill_(0.02)
    grid = _grid(_geometry())
    generator = torch.Generator().manual_seed(33)
    dtype = torch.bfloat16 if bf16 else torch.float32
    capture = torch.randn(1, 28, 8, generator=generator).to(dtype)
    guide = torch.randn(1, 28, 8, generator=generator).to(dtype)
    context = torch.randn(1, 3, 16, generator=generator).to(dtype)
    return model, grid, capture, guide, context


def _run(model, grid, capture, guide, context, mode):  # noqa: ANN001, ANN202
    options = {"sigma": 0.725, "seed": 7, "accumulation": 1}
    if mode == "bidirectional":
        return bidirectional.train_sample(model, context, grid, capture, guide, lambda loss: loss.backward(), **options)
    return causal.train_sample(
        model, context, grid, capture, guide, _geometry(), [0, 1, 2], lambda loss: loss.backward(), **options
    )


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_trace_preserves_real_training_outputs_and_gradients(mode: str, checkpointing: bool) -> None:
    model, grid, capture, guide, context = _fixture(checkpointing=checkpointing)
    reference = copy.deepcopy(model)
    expected = _run(reference, grid, capture, guide, context, mode)
    with Trace(model, BINDING) as trace, trace.sample(mode=mode, step=1, slot=0, index=0):
        actual = _run(model, grid, capture, guide, context, mode)
    record = trace.record()
    validate(record, BINDING)
    assert {key: value for key, value in expected.items() if key != "cache"} == {
        key: value for key, value in actual.items() if key != "cache"
    }
    if mode == "causal":
        for left, right in zip(expected["cache"].caches, actual["cache"].caches, strict=True):
            assert left.length == right.length
            assert torch.equal(left.k[:, : left.length], right.k[:, : right.length])
            assert torch.equal(left.v[:, : left.length], right.v[:, : right.length])
    expected_gradients = {name: p.grad for name, p in reference.named_parameters() if p.requires_grad}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter.grad, expected_gradients[name])
    inputs = [event for event in record["events"] if event["kind"] == "ltx_input"]
    assert [event["role"] for event in inputs] == (
        ["denoise"] if mode == "bidirectional" else ["prime", "denoise", "refresh", "denoise", "refresh", "denoise"]
    )
    assert all(event["sigma"]["dtype"] == "torch.float32" for event in inputs)
    assert all(event["timesteps"]["dtype"] == "torch.float32" for event in inputs)
    denoise = next(event for event in inputs if event["role"] == "denoise")
    assert denoise["timesteps"]["sample_values"][:4] == [0, 0, 0, 0]
    assert denoise["sigma"]["sample_values"] == [pytest.approx(0.725)]
    blocks = [event for event in record["events"] if event["kind"] == "block_input"]
    assert any(event["block_pass"] == "checkpoint_recompute" for event in blocks) == checkpointing
    assert all(event["role"] == "denoise" for event in blocks if event["block_pass"] == "checkpoint_recompute")
    assert all(event["timesteps"]["hash_scope"] == "complete_tensor" for event in blocks)
    assert all(event["positional_embeddings"][0]["hash_scope"] == "complete_tensor" for event in blocks)
    assert record["adapter_storage"]
    assert all(item["dtype"] == "torch.float32" for item in record["adapter_storage"])
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())


def test_autocast_observes_adapter_storage_input_and_output_separately() -> None:
    model, grid, capture, guide, context = _fixture(bf16=True)
    with (
        Trace(model, BINDING) as trace,
        trace.sample(mode="bidirectional", step=1, slot=0, index=0),
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        _run(model, grid, capture, guide, context, "bidirectional")
    record = trace.record()
    validate(record)
    adapter_inputs = [event for event in record["events"] if event["kind"] == "adapter_input"]
    outputs = [event for event in record["events"] if event["kind"] == "adapter_output"]
    assert adapter_inputs
    assert outputs
    assert all(event["weight"]["dtype"] == "torch.float32" for event in adapter_inputs)
    assert all(len(event["weight"]["sha256"]) == 64 for event in adapter_inputs)
    assert any(event["input"]["dtype"] == "torch.float32" for event in adapter_inputs)
    assert all(event["output"]["dtype"] == "torch.bfloat16" for event in outputs)
    assert all(item["dtype"] == "torch.float32" for item in record["adapter_storage"])


def test_actual_consumer_exposes_installed_fsdp_recursive_input_cast() -> None:
    model, grid, capture, _guide, context = _fixture(bf16=True)
    modality = common.block_modality(grid, capture, context, 0.725, token_slices=[(0, 28)], clean_prefix_tokens=4)
    _, unchanged = _cast_forward_inputs(None, video=modality, audio=None, perturbations=None)
    _, cast = _cast_forward_inputs(torch.bfloat16, video=modality, audio=None, perturbations=None)
    records = []
    for arguments in (unchanged, cast):
        expected, _ = model(**arguments)
        with Trace(model, BINDING) as trace, trace.sample(mode="bidirectional", step=1, slot=0, index=0):
            actual, _ = model(**arguments)
        assert torch.equal(expected, actual)
        record = trace.record()
        validate(record)
        records.append(next(e for e in record["events"] if e["kind"] == "ltx_input"))
    for field in ("sigma", "timesteps", "positions"):
        assert records[0][field]["dtype"] == "torch.float32"
        assert records[1][field]["dtype"] == "torch.bfloat16"
        assert records[0][field]["sha256"] != records[1][field]["sha256"]
    expected = hashlib.sha256(modality.sigma.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert records[0]["sigma"]["sha256"] == expected


@pytest.mark.parametrize("limit", ["events", "tensor"])
def test_bounded_trace_marks_missing_evidence_incomplete(limit: str) -> None:
    model, grid, capture, guide, context = _fixture()
    options = {"max_events": 2} if limit == "events" else {"max_tensor_elements": 1}
    with Trace(model, BINDING, **options) as trace, trace.sample(mode="bidirectional", step=1, slot=0, index=0):
        _run(model, grid, capture, guide, context, "bidirectional")
    record = trace.record()
    assert not record["complete"]
    assert record["dropped_events"] if limit == "events" else record["observation_errors"]
    with pytest.raises(ValueError, match="incomplete"):
        validate(record)
    validate(record, require_complete=False)
    record["complete"] = True
    with pytest.raises(ValueError, match="missing observations"):
        validate(record)


def test_failed_scopes_preserve_original_exception_and_remove_hooks(tmp_path: Path) -> None:
    model, *_unused = _fixture()
    trace = Trace(model, BINDING)
    with (
        pytest.raises(RuntimeError, match="original failure"),
        trace,
        trace.sample(mode="bidirectional", step=1, slot=0, index=0),
    ):
        raise RuntimeError("original failure")
    record = trace.record()
    assert not record["complete"]
    assert record["samples"][0]["error"] == "RuntimeError"
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())
    path = tmp_path / "failed.json"
    receipt = trace.write(path, require_complete=False)
    assert receipt["complete"] is False
    assert json.loads(path.read_text()) == record


@pytest.mark.parametrize("parent_exists", [False, True])
@pytest.mark.parametrize("path_exists", [False, True])
def test_failure_path_preserves_error_and_never_creates_output_or_restamps(
    tmp_path: Path, parent_exists: bool, path_exists: bool
) -> None:
    model, grid, capture, guide, context = _fixture()
    parent = tmp_path / "output"
    path = parent / "failure.json"
    if parent_exists:
        parent.mkdir()
        if path_exists:
            path.write_text("original diagnostic bytes")
    trace = Trace(model, BINDING, failure_path=path)

    def failing_operation() -> None:
        with trace.sample(mode="bidirectional", step=1, slot=0, index=0):
            _run(model, grid, capture, guide, context, "bidirectional")
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"), trace:
        failing_operation()
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())
    if parent_exists and path_exists:
        assert path.read_text() == "original diagnostic bytes"
    elif parent_exists:
        record = json.loads(path.read_text())
        validate(record, require_complete=False)
        assert not record["complete"]
    else:
        assert not parent.exists()


def test_large_transformed_conditions_keep_metadata_without_claiming_value_hashes() -> None:
    model, grid, capture, guide, context = _fixture()
    # Actual positions need 168 values. Transformed AdaLN per-token values exceed 200.
    with (
        Trace(model, BINDING, max_tensor_elements=200) as trace,
        trace.sample(mode="bidirectional", step=1, slot=0, index=0),
    ):
        _run(model, grid, capture, guide, context, "bidirectional")
    record = trace.record()
    validate(record)
    blocks = [event for event in record["events"] if event["kind"] == "block_input"]
    assert blocks
    assert all(event["timesteps"]["hash_scope"] == "not_copied_exceeds_limit" for event in blocks)
    assert all(event["timesteps"]["sha256"] is None for event in blocks)


def test_snapshot_is_detached_from_later_samples_exclusive_and_bound(tmp_path: Path) -> None:
    model, grid, capture, guide, context = _fixture()
    with Trace(model, BINDING) as trace:
        initial = trace.record()
        validate(initial)
        with trace.sample(mode="bidirectional", step=1, slot=0, index=0):
            _run(model, grid, capture, guide, context, "bidirectional")
        path = tmp_path / "trace.json"
        receipt = trace.write(path)
        record = json.loads(path.read_text())
        validate(record, BINDING)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        with pytest.raises(FileExistsError):
            trace.write(path)
        assert initial["samples"] == []
        assert initial["events"] == []
        with pytest.raises(ValueError, match="binding differs"):
            validate(record, {**BINDING, "rank": 1, "world_size": 2})
    with pytest.raises(ValueError, match="reopened"):
        trace.start()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "event_call", "visit"])
def test_snapshot_validator_rejects_missing_or_rebound_consumer_evidence(mutation: str) -> None:
    model, grid, capture, guide, context = _fixture()
    with Trace(model, BINDING) as trace, trace.sample(mode="bidirectional", step=1, slot=0, index=0):
        _run(model, grid, capture, guide, context, "bidirectional")
    record = trace.record()
    event = next(e for e in record["events"] if e["kind"] == "ltx_input")
    if mutation == "missing":
        record["events"].remove(event)
        for index, item in enumerate(record["events"]):
            item["event"] = index
    elif mutation == "duplicate":
        record["events"].append({**event, "event": len(record["events"])})
    elif mutation == "hash":
        event["sigma"]["sha256"] = "z" * 64
    elif mutation == "event_call":
        event["model_call"] = 123
    else:
        record["samples"][0]["slot"] = True
    with pytest.raises(ValueError, match="consumer"):
        validate(record)


@pytest.mark.parametrize(
    "changes", [{"rank": True}, {"world_size": 0}, {"launch_sha256": "wrong"}, {"queue_attempt_token": ""}]
)
def test_invalid_launch_binding_is_rejected(changes: dict) -> None:
    with pytest.raises(ValueError, match="trace"):
        Trace(_model(), {**BINDING, **changes})
