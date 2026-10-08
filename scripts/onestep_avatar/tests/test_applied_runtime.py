"""Applied policy inventory and early Accelerator gate are independent of YAML claims."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from torch.distributed.fsdp import MixedPrecision

from scripts.onestep_avatar.training import numerics, runtime


def inventory(*, numerical: bool = False) -> dict:
    policy = {"param_dtype": "torch.bfloat16", "reduce_dtype": "torch.bfloat16", "buffer_dtype": None,
              "cast_forward_inputs": False, "cast_root_forward_inputs": False, "keep_low_precision_grads": False}
    schema = 2 if numerical else 1
    ranks = [{"schema_version": schema, "rank": rank, "world_size": 4, "mixed_precision": "bf16",
              "distributed_type": "FSDP", "conditioning_precision": "float32",
              "adapter_storage_dtypes": ["torch.float32"], "fsdp_policies": [deepcopy(policy)]}
             for rank in range(4)]
    if numerical:
        for rank in ranks:
            rank["numerics"] = {**numerics.POLICY, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    return {"schema_version": schema, "world_size": 4, "mixed_precision": "bf16", "ranks": ranks}


def test_applied_policy_inventory_agrees() -> None:
    runtime.validate(inventory(), 4, "bf16", native=True)


@pytest.mark.parametrize(("field", "value"), [("rank", True), ("world_size", 2), ("mixed_precision", "no"),
                                        ("conditioning_precision", "float16"), ("distributed_type", "invented"),
                                        ("adapter_storage_dtypes", [])])
def test_one_rank_changed_policy_refuses(field: str, value: object) -> None:
    record = inventory()
    record["ranks"][2][field] = value
    with pytest.raises(ValueError, match=r"runtime|policy|precision"):
        runtime.validate(record, 4, "bf16", native=True)


@pytest.mark.parametrize("change", ["root_cast", "missing_policies", "bf16_masters", "serial", "bad_dtype"])
def test_all_rank_agreement_cannot_authorize_wrong_applied_runtime(change: str) -> None:
    record = inventory()
    for rank in record["ranks"]:
        if change == "root_cast":
            rank["fsdp_policies"][0]["cast_root_forward_inputs"] = True
        elif change == "missing_policies":
            rank["fsdp_policies"] = []
        elif change == "bf16_masters":
            rank["adapter_storage_dtypes"] = ["torch.bfloat16"]
        elif change == "serial":
            rank["distributed_type"] = "NO"
        elif change == "bad_dtype":
            rank["fsdp_policies"][0]["reduce_dtype"] = "invented"
    with pytest.raises(ValueError, match=r"runtime|policy|precision"):
        runtime.validate(record, 4, "bf16", native=True)


@pytest.mark.parametrize(("world", "precision"), [(2, "bf16"), (4, "no")])
def test_actual_accelerator_mismatch_fails_before_model(world: int, precision: str) -> None:
    accelerator = SimpleNamespace(num_processes=world, mixed_precision=precision)
    with pytest.raises(ValueError, match="actual Accelerator"):
        runtime.check_accelerator(accelerator, 4, "bf16")


def test_actual_distributed_type_mismatch_fails_before_model() -> None:
    accelerator = SimpleNamespace(num_processes=4, mixed_precision="bf16", distributed_type=DistributedType.MULTI_GPU)
    with pytest.raises(ValueError, match="actual Accelerator"):
        runtime.check_accelerator(accelerator, 4, "bf16", distributed_type=DistributedType.FSDP)


def test_capture_observes_applied_policy_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    # Controlled module identity exposes a real PyTorch policy without requiring CUDA/FSDP initialization.
    class Wrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                 cast_root_forward_inputs=False)
            self.lora_A = torch.nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(runtime, "FSDP", Wrapped)
    model = torch.nn.Module()
    model.wrapped = Wrapped()
    accelerator = SimpleNamespace(num_processes=1, process_index=0, mixed_precision="bf16",
                                  distributed_type=DistributedType.FSDP)
    observed = runtime.capture(model, accelerator, "float32")
    assert observed["adapter_storage_dtypes"] == ["torch.float32"]
    assert observed["fsdp_policies"][0]["reduce_dtype"] == "torch.float32"
    gathered = runtime.gather(observed, accelerator)
    runtime.validate(gathered, 1, "bf16", native=True)
