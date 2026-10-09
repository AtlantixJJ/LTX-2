"""E4 serial visits, numerical comparisons and experiment source binding."""

import json
from copy import deepcopy

import pytest
import torch

from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import training_update_check as check
from scripts.onestep_avatar.training import config


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
    record = software.capture("training", "causal", extra_sources=check.EXTRA_SOURCES)
    software.check_current(record)
    original = software.sha256
    monkeypatch.setattr(
        software,
        "sha256",
        lambda path: "f" * 64 if str(path.relative_to(software.LTX_ROOT)) == check.ENTRY else original(path),
    )
    software.validate(record)
    with pytest.raises(ValueError, match="changed"):
        software.check_current(record)
    software.check_current(normal)


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
