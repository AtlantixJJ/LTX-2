"""Analytic Adam evidence and fixed rank/slot replay controls."""

import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from scripts.onestep_avatar import software
from scripts.onestep_avatar import training_update_check as check
from scripts.onestep_avatar.training import config, update_state


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


def test_near_zero_and_nonfinite_comparisons():
    assert check.gap(torch.zeros(4), torch.zeros(4))["passed"]
    result = check.gap(torch.full((4,), 2e-8), torch.zeros(4))
    assert result["near_zero"]
    assert result["relative_l2"] is None
    assert not result["passed"]
    with pytest.raises(ValueError, match="finite"):
        check.gap(torch.tensor([float("nan")]), torch.zeros(1))


def test_four_rank_accumulation_keeps_original_seed_keys():
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            "/unused",
            "--output",
            "/unused",
            "--chains-per-rank",
            "2",
            "--guide-mode",
            "d0",
        ]
    )
    samples = [{"source": "a", "ranges": [[0, 7]]}, {"source": "b", "ranges": [[0, 7]]}]
    visits = check.first_update_visits(settings, samples, 4)
    assert len(visits) == 8
    assert [visit["index"] for visit in visits] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert len({visit["noise_seed"] for visit in visits}) == 8
    logs = []
    for rank in range(4):
        group = [visit for visit in visits if visit["rank"] == rank]
        logs.append(
            {
                "rank": rank,
                "step": 1,
                "sigma0": group[0]["sigma"],
                "samples": [
                    {"source": v["sample"]["source"], "ranges": [[0, 7]], "noise_seed": v["noise_seed"]} for v in group
                ],
                "call_counts": {"prime": 0, "denoise": 2, "backward": 2, "refresh": 0},
            }
        )
    check.check_visits(visits, logs, settings)
    changed = deepcopy(logs)
    changed[2]["samples"][1]["noise_seed"] += 1
    with pytest.raises(ValueError, match="noise"):
        check.check_visits(visits, changed, settings)


def test_experiment_owner_bound_without_ordinary_dependency(monkeypatch):
    normal = software.capture("training", "causal")
    assert check.ENTRY not in normal["sources"]
    record = software.capture("training", "causal", extra_sources=(check.ENTRY,))
    software.check_current(record)
    original = software.sha256
    monkeypatch.setattr(
        software,
        "sha256",
        lambda path: "f" * 64 if str(path.relative_to(software.ROOT)) == check.ENTRY else original(path),
    )
    software.validate(record)
    with pytest.raises(ValueError, match="changed"):
        software.check_current(record)
    software.check_current(normal)


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", "missing.py", "README.md"])
def test_extra_owner_paths_refused(name):
    with pytest.raises(ValueError, match="contained"):
        software.capture("training", "causal", extra_sources=(name,))


@pytest.mark.parametrize("missing", ["--save-update-state", "--save-initial", "--steps"])
def test_invalid_replay_design_fails_before_input_or_model_access(tmp_path, monkeypatch, missing):
    arguments = [
        "--mode",
        "bidirectional",
        "--subset",
        "/absent",
        "--output",
        "/absent",
        "--guide-mode",
        "d0",
        "--chains-per-rank",
        "2",
        "--steps",
        "1",
        "--save-initial",
        "--save-update-state",
    ]
    index = arguments.index(missing)
    del arguments[index : index + (2 if missing == "--steps" else 1)]
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"arguments": arguments}))
    monkeypatch.setattr(check.engine, "prepare_run", lambda *args, **kwargs: pytest.fail("opened inputs"))
    with pytest.raises(ValueError, match="fresh one-update"):
        check.execute(job, tmp_path / "output", 4)
    assert not (tmp_path / "output").exists()
