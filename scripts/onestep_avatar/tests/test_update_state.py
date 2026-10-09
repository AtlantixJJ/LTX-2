"""Analytic Adam evidence and fixed rank/slot replay controls."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.training import update_state


def _states() -> dict:
    gradient = torch.tensor([[0.2, -0.3], [0.0, 0.1]])
    return {
        "base_model.model.block.lora_B.default.weight": {
            "step": torch.tensor(1.0),
            "exp_avg": gradient * 0.1,
            "exp_avg_sq": gradient.square() * 0.001,
        }
    }


def test_actual_adam_moments_reconstruct_gradient(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = torch.nn.Module()
            self.base_model.model = torch.nn.Module()
            self.base_model.model.block = torch.nn.Module()
            self.base_model.model.block.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})

    model = Model()
    parameter = next(model.parameters())
    gradient = torch.tensor([[0.2, -0.3], [0.0, 0.1]])
    parameter.grad = gradient.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    optimizer.step()
    accelerator = SimpleNamespace(distributed_type=DistributedType.NO, is_main_process=True)
    path = tmp_path / "adam.pt"
    shapes = update_state.save_adam_state(model, optimizer, accelerator, path, 1)
    states = torch.load(path, weights_only=True)
    name = "diffusion_model.block.lora_B.weight"
    assert shapes == {name: [2, 2]}
    torch.testing.assert_close(states[name]["exp_avg"] / 0.1, gradient)
    torch.testing.assert_close(states[name]["exp_avg_sq"], gradient.square() * 0.001)


def test_fsdp_nonmain_empty_rank_zero_only_result(monkeypatch):
    calls = []
    monkeypatch.setattr(
        update_state,
        "FSDP",
        SimpleNamespace(
            state_dict_type=lambda *args: nullcontext(), optim_state_dict=lambda *args: calls.append("collect") or {}
        ),
    )
    accelerator = SimpleNamespace(distributed_type=DistributedType.FSDP, is_main_process=False)
    assert update_state.collect_adam_state(torch.nn.Linear(2, 2), object(), accelerator, 1) is None
    assert calls == ["collect"]


@pytest.mark.parametrize("defect", ["step", "dtype", "nan", "negative_second", "base", "extra_field"])
def test_invalid_moments_refused(defect):
    states = _states()
    name = next(iter(states))
    if defect == "step":
        states[name]["step"] = torch.tensor(2.0)
    elif defect == "dtype":
        states[name]["exp_avg"] = states[name]["exp_avg"].bfloat16()
    elif defect == "nan":
        states[name]["exp_avg"][0, 0] = float("nan")
    elif defect == "negative_second":
        states[name]["exp_avg_sq"][0, 0] = -1
    elif defect == "base":
        states["base_model.model.block.weight"] = states.pop(name)
    else:
        states[name]["extra"] = 1
    with pytest.raises(ValueError, match=r"Adam|adapter|moments"):
        update_state.validate_moments(states, 1)








@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", "missing.py", "README.md"])
def test_extra_owner_paths_refused(name):
    with pytest.raises(ValueError, match="contained"):
        software.capture("training", "causal", extra_sources=(name,))
