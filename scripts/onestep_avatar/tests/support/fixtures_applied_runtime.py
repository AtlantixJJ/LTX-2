"""Shared saved-input builders and fixtures; never a collected test module."""

from __future__ import annotations

from copy import deepcopy

import torch

from scripts.onestep_avatar.training import numerics


def inventory() -> dict:
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
    return {"schema_version": 2, "world_size": 4, "mixed_precision": "bf16", "ranks": ranks}
