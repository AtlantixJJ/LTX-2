"""E4 serial replay reuses completed ordinary tiny-CPU updates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from scripts.onestep_avatar.experiments import training_update_check
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.test_training_runtime import CPUAccelerator, run_bounded_update
from scripts.onestep_avatar.training import engine, resources


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("preview_failure", [False, True])
@pytest.mark.parametrize("resource_check", ["off", "pass"])
@pytest.mark.parametrize("consumer_trace", [False, True])
def test_serial_replay_uses_shared_completed_update(
    mode: str, preview_failure: bool, resource_check: str, consumer_trace: bool,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = run_bounded_update(mode, preview_failure, False, True, resource_check,
                              consumer_trace, tmp_path, monkeypatch)
    assert saved is not None
    settings, options, initial, updated, initial_tensors, logs, state_path, budget_path = (
        saved[key] for key in ("settings", "options", "initial", "updated", "initial_tensors",
                              "logs", "state_path", "budget_path"))
    states = torch.load(state_path, weights_only=True)
    job_path = tmp_path / "job.json"
    launch_config = tmp_path / "accelerate.yaml"
    launch_config.write_text("compute_environment: LOCAL_MACHINE\ndistributed_type: 'NO'\nmixed_precision: 'no'\n")
    job_path.write_text(json.dumps({"arguments": options, "accelerate_config": str(launch_config)}))
    monkeypatch.setattr(training_update_check, "Accelerator", lambda **kwargs: CPUAccelerator())
    output = tmp_path / "serial"
    if not preview_failure:
        # This one-process fixture has no native dispatch/runtime/resource inventory.
        with pytest.raises(ValueError):
            training_update_check.execute(job_path, output, 1)
        assert not output.exists()
    if resource_check == "pass" and not preview_failure:
        # Exercise actual replay model/update/export calculation under controlled native-evidence gates.
        # Gate acceptance uses the separate real-normalizer tests; this CPU fixture establishes no native result.
        replay_store, replay_plan, replay_spec, _ = engine.prepare_run(settings, require_fresh_output=False)
        samples = [sample for sample in replay_plan["samples"] if sample["split"] == settings.split]
        visits = training_update_check.first_update_visits(settings, samples, 1)
        context = torch.load(settings.output / "update_states/text.pt", weights_only=True)
        source_paths = [job_path, settings.subset, settings.output / "config.json",
                        settings.output / "frame_plan.json", state_path, initial, updated]
        identities = {str(path.resolve()): sha256(path) for path in source_paths}
        monkeypatch.setattr(training_update_check, "prepare", lambda *_args: (
            settings, replay_store, replay_plan, replay_spec, visits, logs, states, initial_tensors,
            identities, context))
        budget = resources.read_budget(budget_path)
        original_cpu_runtime = json.loads((settings.output / "config.json").read_text())["runtime"]
        monkeypatch.setattr(training_update_check, "check_launch", lambda *_args: (
            {"sha256": "c" * 64}, {"queue_launch": {"scope": "controlled CPU evidence gate"},
                                 "runtime": original_cpu_runtime,
                                 "resource_budget": budget}, "no"))
        result = training_update_check.execute(job_path, output, 1)
        assert result["comparison"]["passed"]
        assert result["comparison"]["actual_step_one_reload_exact"]
        assert [record["phase"] for record in resources.read_records(output, 1)[0]] == ["load", "update", "export"]
    doubled = {name: {**value, "exp_avg": value["exp_avg"] * 2} for name, value in states.items()}
    measured = training_update_check.compare_update(
        doubled, states, load_file(updated), load_file(updated), beta1=0.9, norms=(1.0, 1.0), losses=(1.0, 1.0))
    assert not measured["passed"]
