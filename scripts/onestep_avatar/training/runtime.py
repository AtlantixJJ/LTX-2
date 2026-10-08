"""Observe applied distributed precision; see doc/training/runtime.md."""

from __future__ import annotations

import json

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817 -- native conventional name

from scripts.onestep_avatar.training import numerics


def _dtype(value: torch.dtype | None) -> str | None:
    return None if value is None else str(value)


def numerical_policy(record: dict) -> dict:
    """Read an explicit supported original policy, never a current-default substitute."""
    if (not isinstance(record, dict) or record.get("schema_version") != 2
            or not isinstance(record.get("ranks"), list) or not record["ranks"]):
        raise ValueError("current numerical policy requires explicit schema-two runtime evidence")
    observed = record["ranks"][0].get("numerics")
    numerics.validate(observed, required=True)
    for rank in record["ranks"]:
        numerics.validate(rank.get("numerics"), required=True, expected=observed)
    return dict(observed)


def check_accelerator(
    accelerator: Accelerator, world: int, precision: str, *, distributed_type: DistributedType | None = None
) -> None:
    """Reject a changed actual process group before opening transformer/text sessions."""
    if (accelerator.num_processes != world or accelerator.mixed_precision != precision
            or (distributed_type is not None and accelerator.distributed_type != distributed_type)):
        raise ValueError("actual Accelerator world size, mixed precision or distributed type differs from launch")


def capture(model: torch.nn.Module, accelerator, conditioning_precision: str) -> dict:  # noqa: ANN001
    """Read actual wrapped-module policies, not an assumed YAML configuration."""
    policies = []
    for module in model.modules():
        if isinstance(module, FSDP):
            policy = module.mixed_precision
            record = {key: _dtype(getattr(policy, key)) for key in ("param_dtype", "reduce_dtype", "buffer_dtype")}
            record.update({key: getattr(policy, key) for key in
                           ("cast_forward_inputs", "cast_root_forward_inputs", "keep_low_precision_grads")})
            if record not in policies:
                policies.append(record)
    return {"schema_version": 2, "rank": accelerator.process_index,
            "world_size": accelerator.num_processes,
            "mixed_precision": getattr(accelerator, "mixed_precision", "no"),
            "distributed_type": accelerator.distributed_type.name,
            "conditioning_precision": conditioning_precision,
            "adapter_storage_dtypes": sorted({str(parameter.dtype) for name, parameter in model.named_parameters()
                                               if ".lora_" in name}),
            "fsdp_policies": sorted(policies, key=lambda policy: json.dumps(policy, sort_keys=True)),
            "numerics": numerics.capture()}


def gather(record: dict, accelerator) -> dict:  # noqa: ANN001
    """Use tensor collectives for small fixed runtime records, never GPU object gather."""
    if accelerator.num_processes == 1:
        ranks = [record]
    else:
        payload = json.dumps(record, sort_keys=True).encode()
        size = 65536
        if len(payload) >= size:
            raise ValueError("runtime policy record exceeds fixed collective buffer")
        tensor = torch.zeros(size, dtype=torch.uint8, device=accelerator.device)
        tensor[:len(payload)] = torch.tensor(list(payload), dtype=torch.uint8, device=accelerator.device)
        values = accelerator.gather(tensor).reshape(accelerator.num_processes, size).cpu().tolist()
        ranks = [json.loads(bytes(value).split(b"\x00", 1)[0]) for value in values]
    result = {"schema_version": record["schema_version"], "world_size": accelerator.num_processes,
              "mixed_precision": record["mixed_precision"], "ranks": ranks}
    validate(result, accelerator.num_processes, record["mixed_precision"])
    return result


def validate(record: dict, world: int, precision: str, *, native: bool = False,
             numerical_policy: bool | dict | None = None) -> None:
    """Reject absent/applied-policy disagreement before replay loads weights."""
    if (type(world) is not int or world < 1 or precision not in ("no", "fp16", "bf16")
            or not isinstance(record, dict)
            or set(record) != {"schema_version", "world_size", "mixed_precision", "ranks"}
            or type(record.get("world_size")) is not int or type(record.get("schema_version")) is not int
            or record.get("schema_version") not in (1, 2)
            or record.get("world_size") != world or record.get("mixed_precision") != precision
            or not isinstance(record.get("ranks"), list) or len(record["ranks"]) != world):
        raise ValueError("applied runtime world size or mixed precision differs from launch")
    ranks = record["ranks"]
    reference = None
    for rank, observed in enumerate(ranks):
        fields = {"schema_version", "rank", "world_size", "mixed_precision", "distributed_type",
                  "conditioning_precision", "adapter_storage_dtypes", "fsdp_policies"}
        if record["schema_version"] == 2:
            fields.add("numerics")
        if (not isinstance(observed, dict) or set(observed) != fields or type(observed.get("rank")) is not int
                or observed.get("rank") != rank or type(observed.get("world_size")) is not int
                or type(observed.get("schema_version")) is not int
                or observed.get("schema_version") != record["schema_version"] or observed.get("world_size") != world
                or observed.get("mixed_precision") != precision
                or observed.get("distributed_type") not in {member.name for member in DistributedType}
                or observed.get("conditioning_precision") not in ("float32", "bfloat16")
                or not isinstance(observed.get("fsdp_policies"), list)
                or not isinstance(observed.get("adapter_storage_dtypes"), list)):
            raise ValueError("applied runtime rank inventory is malformed")
        if record["schema_version"] == 2:
            numerics.validate(observed["numerics"], required=bool(numerical_policy),
                             expected=numerical_policy if isinstance(numerical_policy, dict) else None)
        elif numerical_policy:
            raise ValueError("current numerical policy requires explicit schema-two runtime evidence")
        current = {key: value for key, value in observed.items() if key != "rank"}
        if reference is not None and current != reference:
            raise ValueError("ranks disagree on applied precision policy")
        reference = current
        dtypes = observed["adapter_storage_dtypes"]
        if (not dtypes or any(dtype not in ("torch.float32", "torch.bfloat16", "torch.float16") for dtype in dtypes)
                or dtypes != sorted(set(dtypes))):
            raise ValueError("applied adapter storage precision is malformed")
        if native and (observed.get("distributed_type") != DistributedType.FSDP.name
                       or observed["conditioning_precision"] != "float32"
                       or dtypes != ["torch.float32"] or not observed["fsdp_policies"]):
            raise ValueError("native runtime requires actual FSDP, float32 conditioning and fp32 adapter masters")
        for policy in observed["fsdp_policies"]:
            fields = {"param_dtype", "reduce_dtype", "buffer_dtype", "cast_forward_inputs",
                      "cast_root_forward_inputs", "keep_low_precision_grads"}
            if (not isinstance(policy, dict) or set(policy) != fields
                    or any(policy[key] not in (None, "torch.float32", "torch.float16", "torch.bfloat16")
                           for key in ("param_dtype", "reduce_dtype", "buffer_dtype"))
                    or any(type(policy[key]) is not bool for key in
                           ("cast_forward_inputs", "cast_root_forward_inputs", "keep_low_precision_grads"))):
                raise ValueError("applied FSDP policy is malformed")
            if (observed["conditioning_precision"] == "float32" and policy["cast_root_forward_inputs"]
                    and policy["param_dtype"] not in (None, "torch.float32")):
                raise ValueError("FSDP root casting violates float32 conditioning precision")
