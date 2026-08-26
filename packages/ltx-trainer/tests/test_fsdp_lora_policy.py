"""Regression coverage for FSDP LoRA mixed-precision wrapping."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from torch import nn
from torch.distributed import destroy_process_group, init_process_group, is_initialized
from torch.distributed.fsdp import FullyShardedDataParallel


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)
        self.to_k = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.to_k(self.to_q(hidden_states))


class _ToyTransformer(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["_ToyBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.block = _ToyBlock()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


def test_fsdp_lora_policy_separates_bf16_base_and_fp32_adapters(tmp_path: Path) -> None:
    """PEFT's FSDP policy must prevent mixed-dtype FSDP flat parameters."""
    if is_initialized():
        pytest.skip("requires an isolated one-rank process group")

    model = _ToyTransformer().to(torch.bfloat16)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k"]))

    assert {parameter.dtype for parameter in model.parameters() if not parameter.requires_grad} == {torch.bfloat16}
    assert {parameter.dtype for parameter in model.parameters() if parameter.requires_grad} == {torch.float32}

    store_path = tmp_path / "fsdp-store"
    init_process_group("gloo", init_method=f"file://{store_path}", rank=0, world_size=1)
    try:
        wrapped = FullyShardedDataParallel(
            model,
            auto_wrap_policy=fsdp_auto_wrap_policy(model),
            device_id=torch.device("cpu"),
            use_orig_params=True,
        )
        flat_params = [
            module._flat_param for module in FullyShardedDataParallel.fsdp_modules(wrapped) if module._has_params
        ]
    finally:
        destroy_process_group()

    assert len(flat_params) == 5
    assert all(flat_param is not None for flat_param in flat_params)
    assert {flat_param.dtype for flat_param in flat_params if not flat_param.requires_grad} == {torch.bfloat16}
    assert {flat_param.dtype for flat_param in flat_params if flat_param.requires_grad} == {torch.float32}
