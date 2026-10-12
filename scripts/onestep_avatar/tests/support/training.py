"""Shared CPU builders; never collected as test modules."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import precompute, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, causal
from scripts.onestep_avatar.tests.support.corpus import build_runtime_membership
from scripts.onestep_avatar.tests.support.model import _model
from scripts.onestep_avatar.tests.support.tokens import token
from scripts.onestep_avatar.training import checkpoints, config, engine


class CPUAccelerator:
    num_processes = 1
    process_index = 0
    is_main_process = True
    distributed_type = DistributedType.NO
    device = torch.device("cpu")
    mixed_precision = "no"

    def prepare(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple:
        return model, optimizer

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters: object, maximum: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(parameters, maximum)

    def wait_for_everyone(self) -> None:
        pass

    def get_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return model.state_dict()

    def unwrap_model(self, model: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        del kwargs
        return model


def run_bounded_update(  # noqa: PLR0912, PLR0915 -- one complete fixture lifecycle with all assertions and cleanup
    mode: str,
    preview_failure: bool,
    queued: bool,
    update_evidence: bool,
    resource_check: str,
    consumer_trace: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict | None:
    if queued:
        from scripts.onestep_avatar.execution.queue_protocol import (  # noqa: PLC0415 -- defer loader or fixture imports until preflight and test hooks are installed
            JOB_ENV,
            TOKEN_ENV,
        )

        monkeypatch.setenv(JOB_ENV, "a" * 64)
        monkeypatch.setenv(TOKEN_ENV, token())
    _membership, list_path = build_runtime_membership(tmp_path, seed=17)
    options = [
        "--mode",
        mode,
        "--subset",
        str(list_path),
        "--output",
        str(tmp_path / "run"),
        "--variant",
        "dev",
        "--objective",
        "white",
        "--guide-mode",
        "d0",
        "--lora-rank",
        "2",
        "--steps",
        "1",
        "--save-initial",
        "--no-wandb",
        "--log-every",
        "2",
    ]
    if mode == "causal":
        options += ["--blocks-per-sample", "2", "--context-latent-frames", "1"]
    if update_evidence:
        options += ["--chains-per-rank", "2", "--save-update-state"]
    if resource_check != "off":
        from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- fixture-local import
            resources,
        )

        budget_path = tmp_path / "resource_budget.json"
        budget_path.write_text(json.dumps({"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 1000}))
        options += ["--resource-budget", str(budget_path)]
        native_phase = resources.Phase
        peak = {"allocated": 100}

        def controlled_phase(device: torch.device, phase: str, rank: int, budget: object) -> object:
            # CUDA counters are controlled; model execution stays on CPU. This is publication evidence only.
            del device
            peak["allocated"] = 1001 if phase == "update:1" and resource_check == "update_failure" else 100
            return native_phase(torch.device("cuda:0"), phase, rank, budget)

        monkeypatch.setattr(resources, "Phase", controlled_phase)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: peak["allocated"])
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2000)
    if consumer_trace:
        options += ["--consumer-trace"]
    settings = config.parse_settings(options)
    if preview_failure:
        settings.preview_record = {"test": "enqueue failure"}

        def failed_enqueue(*args: object) -> Never:
            del args
            raise RuntimeError("preview-only failure")

        monkeypatch.setattr(engine, "enqueue_preview", failed_enqueue)
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=8),
        sigmas=[0.725],
        paths=SimpleNamespace(
            transformer=lambda: tmp_path / "unused_base.safetensors",
            video_vae=lambda: tmp_path / "unused_vae.safetensors",
        ),
    )
    specification.paths.transformer().write_bytes(b"controlled native-base identity")
    monkeypatch.setattr(engine.backbone, "resolve", lambda *_args: specification)
    monkeypatch.setattr(precompute, "file_fingerprint", lambda _path: "runtime VAE identity")
    monkeypatch.setattr(
        engine.backbone,
        "identity",
        lambda *_args, **_kwargs: {
            "base_transformer_sha256": sha256(specification.paths.transformer()),
            "base_transformer_file": "base.safetensors",
            "base_variant": "dev",
            "model_key": "2.5",
        },
    )
    monkeypatch.setattr(engine, "Accelerator", CPUAccelerator)
    monkeypatch.setattr(engine.prompt_cache, "get_or_build", lambda *_args: torch.zeros(1, 3, 16, dtype=torch.bfloat16))
    calls = []

    def build(*args: object) -> torch.nn.Module:
        del args
        model = _model(prompt_adaln=True).bfloat16()
        model.requires_grad_(False)
        torch.manual_seed(settings.init_seed)
        wrapped = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=adapters.LORA_TARGETS["attn"],
                lora_dropout=0.0,
                init_lora_weights=True,
            ),
        )
        wrapped.get_base_model().set_gradient_checkpointing(True)
        wrapped.register_forward_hook(
            lambda _module, _args, kwargs, _result: calls.append(kwargs["video"]), with_kwargs=True
        )
        return wrapped

    monkeypatch.setattr(engine, "build_transformer", build)
    if queued:
        # A token/hash fixture without actual dispatch evidence cannot certify a launch.
        with pytest.raises(ValueError, match="original dispatch launch evidence"):
            engine.run_settings(settings)
        assert not settings.output.exists()
        assert not calls
        return
    if mode == "bidirectional":
        monkeypatch.setattr(
            causal.BlockCache,
            "allocate",
            lambda *_args, **_kwargs: pytest.fail("bidirectional allocated a causal cache"),
        )
    if resource_check == "update_failure":
        with pytest.raises(ValueError, match="allocated-memory limit"):
            engine.run_settings(settings)
        assert (settings.output / "checkpoints/lora_weights_step_00000.complete.json").is_file()
        assert not (settings.output / "checkpoints/lora_weights_step_00001.complete.json").exists()
        measurements, _hashes = resources.read_records(settings.output, 1)
        assert measurements[-1]["state"] == "failed"
        assert measurements[-1]["peak_allocated_bytes"] == 1001
        return
    assert engine.run_settings(settings) == 0
    initial = settings.output / "checkpoints/lora_weights_step_00000.safetensors"
    updated = settings.output / "checkpoints/lora_weights_step_00001.safetensors"
    initial_tensors = load_file(initial)
    checkpoints.assert_exported_lora_is_noop(initial_tensors)
    assert any(torch.count_nonzero(value) > 0 for key, value in load_file(updated).items() if ".lora_B." in key)
    for path in (initial, updated):
        record = checkpoints.read_contract(path)
        assert record["mode"] == mode
        checkpoints.validate_adapter_tensors(path, record)
        complete = json.loads(path.with_suffix(".complete.json").read_text())
        assert complete["sha256"] == sha256(path)
        assert complete["state"] == "complete"
        assert complete["queue_job_sha256"] == ("a" * 64 if queued else None)
        assert complete["producer_source_sha256"] == sha256(Path(engine.__file__))
        assert complete["training_record"] == {
            "config_sha256": sha256(settings.output / "config.json"),
            "frame_plan_sha256": sha256(settings.output / "frame_plan.json"),
        }
        if consumer_trace:
            from scripts.onestep_avatar.training.consumer_trace import (  # noqa: PLC0415 -- fixture-local import
                validate,
            )

            evidence = complete["consumer_trace_evidence"]
            assert len(evidence) == 1
            trace_path, digest = next(iter(evidence.items()))
            assert sha256(Path(trace_path)) == digest
            trace_record = json.loads(Path(trace_path).read_text())
            validate(trace_record)
            assert len(trace_record["samples"]) == record["adapter"]["step"] * settings.chains_per_rank
    logs = [json.loads(line) for line in (settings.output / "metrics_rank0.jsonl").read_text().splitlines()]
    assert len(logs) == 1
    assert "anchor" not in logs[0]
    assert logs[0]["step"] == 1
    accumulation = 2 if update_evidence else 1
    assert len(calls) == accumulation * (1 if mode == "bidirectional" else 4)
    expected_counts = (
        {"prime": 0, "denoise": 1, "backward": 1, "refresh": 0}
        if mode == "bidirectional"
        else {"prime": 1, "denoise": 2, "backward": 2, "refresh": 1}
    )
    assert logs[0]["call_counts"] == {key: value * accumulation for key, value in expected_counts.items()}
    saved_plan = json.loads((settings.output / "frame_plan.json").read_text())
    assert saved_plan["sha256"] == subset.record_hash(saved_plan)
    for key in ("mode", "membership_sha256", "frame_plan_sha256"):
        assert key in json.loads((settings.output / "config.json").read_text())
    if resource_check == "pass":
        records, _journal_hashes = resources.read_records(settings.output, 1)
        initial_records, initial_hashes = resources.read_records(settings.output, 1, step=0)
        final_records, final_hashes = resources.read_records(settings.output, 1, step=1)
        assert final_records == records
        assert [record["phase"] for record in initial_records] == ["load", "export:0"]
        assert initial_records == final_records[:2]
        assert json.loads(initial.with_suffix(".complete.json").read_text())["resource_evidence"] == initial_hashes
        assert json.loads(updated.with_suffix(".complete.json").read_text())["resource_evidence"] == final_hashes
        resources.validate_records(
            records, 1, ["load", "export:0", "update:1", "export:1"], resources.read_budget(budget_path)
        )
    if update_evidence:
        from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- fixture-local import
            update_state,
        )

        state_path = settings.output / "update_states/step_00001.pt"
        states = torch.load(state_path, weights_only=True)
        assert set(states) == set(initial_tensors)
        assert all(value["step"] == 1 for value in states.values())
        assert all(value["exp_avg"].dtype == torch.float32 for value in states.values())
        assert (
            update_state.export_name("base_model.model.block.lora_A.default.weight")
            == "diffusion_model.block.lora_A.weight"
        )
    return {
        "settings": settings,
        "options": options,
        "initial": initial,
        "updated": updated,
        "initial_tensors": initial_tensors,
        "logs": logs,
        "state_path": state_path if update_evidence else None,
        "budget_path": budget_path if resource_check != "off" else None,
    }
