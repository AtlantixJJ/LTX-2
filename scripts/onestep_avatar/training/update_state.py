"""Export diagnostic named Adam moments after a shared training update.

Inputs: the prepared model, optimizer, Accelerator and completed update number.
Collect full optimizer state through FSDP on all ranks, offloaded to main-rank
CPU. For an ordinary optimizer map parameter objects to model names. Normalize
PEFT names to the adapter export namespace; require finite fp32 moments and the
declared step. Write only adapter moments atomically, never frozen model weights.
The engine owns software identity and the JSON binding. This is numerical
evidence, not a resumable optimizer checkpoint or a second training runtime.
For first-update Adam, exp_avg / (1-beta1) is the clipped averaged gradient.
"""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817 -- native conventional name

from scripts.onestep_avatar.dataset import atomic_write


def export_name(name: str) -> str:
    """Use the same namespace as exported adapter matrices, without wrapper names."""
    name = name.replace("_fsdp_wrapped_module.", "")
    if not name.startswith("base_model.model.") or ".lora_" not in name:
        raise ValueError("optimizer state is not a named PEFT adapter parameter")
    return "diffusion_model." + name.removeprefix("base_model.model.").replace(".default.", ".")


def validate_moments(states: dict, step: int) -> dict:
    """Reject incomplete/nonfinite moments before their diagnostic publication."""
    if step < 1 or not states:
        raise ValueError("update state requires a completed positive optimizer step")
    checked = {}
    for name, state in states.items():
        key = export_name(name)
        if key in checked or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("duplicate or unsupported Adam parameter state")
        if float(state["step"]) != step:
            raise ValueError("Adam state step differs from completed update")
        first, second = state["exp_avg"], state["exp_avg_sq"]
        if (
            first.shape != second.shape
            or first.dtype != torch.float32
            or second.dtype != torch.float32
            or not torch.isfinite(first).all()
            or not torch.isfinite(second).all()
            or torch.any(second < 0)
        ):
            raise ValueError("Adam moments must be matching finite fp32 arrays")
        checked[key] = {
            "step": step,
            "exp_avg": first.detach().cpu().contiguous(),
            "exp_avg_sq": second.detach().cpu().contiguous(),
        }
    return checked


def collect_adam_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, accelerator: Accelerator, step: int
) -> dict | None:
    """All ranks collect moments; only the main rank validates/returns full arrays."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    if accelerator.distributed_type == DistributedType.FSDP:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            states = FSDP.optim_state_dict(model, optimizer).get("state", {})
    else:
        names = {parameter: name for name, parameter in model.named_parameters()}
        states = {names[parameter]: value for parameter, value in optimizer.state.items()}
    if not accelerator.is_main_process:
        return None
    return validate_moments(states, step)


def save_adam_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, accelerator: Accelerator, path: Path, step: int
) -> dict | None:
    """Atomically save main-rank moments; return shape facts for the engine's record."""
    states = collect_adam_state(model, optimizer, accelerator, step)
    if states is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda temporary: torch.save(states, temporary))
    return {name: list(value["exp_avg"].shape) for name, value in states.items()}
