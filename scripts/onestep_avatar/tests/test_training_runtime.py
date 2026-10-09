"""Bounded CPU updates exercise the typed engine with real tiny LTX/PEFT modules."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, causal
from scripts.onestep_avatar.tests.test_causal_core import _model
from scripts.onestep_avatar.training import checkpoints, config, engine


class CPUAccelerator:
    num_processes = 1
    process_index = 0
    is_main_process = True
    distributed_type = DistributedType.NO
    device = torch.device("cpu")
    mixed_precision = "no"

    def prepare(self, model, optimizer):
        return model, optimizer

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, maximum):
        return torch.nn.utils.clip_grad_norm_(parameters, maximum)

    def wait_for_everyone(self):
        pass

    def get_state_dict(self, model):
        return model.state_dict()

    def unwrap_model(self, model, **kwargs):
        return model


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_native_builder_preserves_typed_fsdp_modality_precision(
    mode, legacy, mixed_precision, tmp_path, monkeypatch
):
    from dataclasses import asdict

    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp._runtime_utils import _cast_forward_inputs

    from ltx_core.model.transformer.modality import Modality

    settings = config.parse_settings([
        "--mode", mode, "--subset", str(tmp_path / "membership.json"),
        "--output", str(tmp_path / "output"), "--lora-rank", "2",
    ])
    settings.init_device = "cpu"
    if legacy:
        settings = SimpleNamespace(**vars(settings))
    policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16) if mixed_precision else None
    original = None if policy is None else asdict(policy)
    plugin = SimpleNamespace(mixed_precision_policy=policy)
    accelerator = SimpleNamespace(distributed_type=DistributedType.FSDP, state=SimpleNamespace(fsdp_plugin=plugin))
    specification = SimpleNamespace(paths=SimpleNamespace(transformer=lambda: "checked-base.safetensors"))
    monkeypatch.setattr(engine, "load_transformer", lambda **kwargs: _model().bfloat16())
    def wrapping(module, *, recurse=False, nonwrapped_numel=0):
        return isinstance(module, torch.nn.Linear) or module.__class__.__name__ == 'BasicAVTransformerBlock'
    monkeypatch.setattr(engine, "fsdp_auto_wrap_policy", lambda model: wrapping)
    model = engine.build_transformer(specification, settings, accelerator)
    if legacy:
        assert plugin.auto_wrap_policy is wrapping
    else:
        from torch.distributed.fsdp.wrap import CustomPolicy
        assert isinstance(plugin.auto_wrap_policy, CustomPolicy)
        chosen = plugin.auto_wrap_policy._run_policy(model, set(), {'mixed_precision': policy})
        assert set(chosen) == {module for module in model.modules() if wrapping(module)}
        for module, options in chosen.items():
            trainable = list(module.parameters()) and all(p.requires_grad for p in module.parameters())
            assert options['mixed_precision'] is (None if trainable else policy)
    assert all(p.dtype == torch.float32 for name, p in model.named_parameters() if ".lora_" in name)
    if policy is None:
        assert plugin.mixed_precision_policy is None
        return
    expected = {**original, "cast_root_forward_inputs": legacy}
    assert asdict(plugin.mixed_precision_policy) == expected
    assert asdict(policy) == original
    # Exercise the installed recursive cast with the actual native dataclass.
    sigma = torch.tensor([0.725], dtype=torch.float32)
    video = Modality(
        latent=torch.ones(1, 2, 8, dtype=torch.bfloat16), sigma=sigma,
        timesteps=sigma.expand(1, 2), positions=torch.full((1, 3, 2, 2), 0.123456),
        context=torch.ones(1, 2, 16, dtype=torch.bfloat16),
    )
    dtype = policy.param_dtype if plugin.mixed_precision_policy.cast_root_forward_inputs else None
    _, forwarded = _cast_forward_inputs(dtype, video=video)
    observed = forwarded["video"]
    assert observed.sigma.dtype == (torch.bfloat16 if legacy else torch.float32)
    if not legacy:
        assert torch.equal(observed.sigma, sigma)
        assert torch.equal(observed.timesteps, video.timesteps)
        assert torch.equal(observed.positions, video.positions)


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("preview_failure", [False, True])
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("update_evidence", [False, True])
@pytest.mark.parametrize("resource_check", ["off", "pass", "update_failure"])
@pytest.mark.parametrize("consumer_trace", [False, True])
def test_one_bounded_update_saves_real_zero_and_updated_lora(
    mode: str, preview_failure: bool, queued: bool, update_evidence: bool,
    resource_check: str, consumer_trace: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if queued:
        from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV
        monkeypatch.setenv(JOB_ENV, "a" * 64)
        monkeypatch.setenv(TOKEN_ENV, "b" * 32)
    view = tmp_path / "corpus" / "actor" / "view"
    view.mkdir(parents=True)
    path = view / dataset.capture_bundle_name("white")
    generator = torch.Generator().manual_seed(17)
    capture = torch.randn(8, 7, 2, 2, generator=generator)
    bundle = {"schema_version": 2, "master": capture, "fps": 30.0, "objective": "white",
              "source": "actor/view", "vae_fingerprint": "runtime VAE identity"}
    torch.save(bundle, path)
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "corpus_root": str(tmp_path / "corpus"),
        "objective": "white",
        "splits": {"train": ["actor"]},
        "sources": [
            {
                "relative_dir": "actor/view",
                "actor": "actor",
                "split": "train",
                "n_latent_frames": 7,
                "shape": [8, 7, 2, 2],
                "fps": 30.0,
                "capture_latent_sha256": sha256(path),
                "capture_encode_record": {
                    k: bundle[k] for k in ("schema_version", "fps", "objective", "source", "vae_fingerprint")
                },
            }
        ],
    }
    membership["sha256"] = subset.membership_hash(membership)
    list_path = tmp_path / "membership.json"
    list_path.write_text(json.dumps(membership))
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
        from scripts.onestep_avatar.training import resources
        budget_path = tmp_path / "resource_budget.json"
        budget_path.write_text(json.dumps({"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 1000}))
        options += ["--resource-budget", str(budget_path)]
        native_phase = resources.Phase
        peak = {"allocated": 100}
        def controlled_phase(device, phase, rank, budget):
            # CUDA counters are controlled; model execution stays on CPU. This is publication evidence only.
            peak["allocated"] = 1001 if phase == "update:1" and resource_check == "update_failure" else 100
            return native_phase(torch.device("cuda:0"), phase, rank, budget)
        monkeypatch.setattr(resources, "Phase", controlled_phase)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: peak["allocated"])
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2000)
    if consumer_trace:
        options += ["--consumer-trace"]
    settings = config.parse_settings(options)
    if preview_failure:
        settings.preview_record = {"test": "enqueue failure"}

        def failed_enqueue(*args):
            raise RuntimeError("preview-only failure")

        monkeypatch.setattr(engine, "enqueue_preview", failed_enqueue)
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=8),
        sigmas=[0.725],
        paths=SimpleNamespace(transformer=lambda: tmp_path / "unused_base.safetensors",
                              video_vae=lambda: tmp_path / "unused_vae.safetensors"),
    )
    specification.paths.transformer().write_bytes(b"controlled native-base identity")
    monkeypatch.setattr(engine.backbone, "resolve", lambda *args: specification)
    from scripts.onestep_avatar.corpus import precompute
    monkeypatch.setattr(precompute, "file_fingerprint", lambda path: "runtime VAE identity")
    monkeypatch.setattr(
        engine.backbone,
        "identity",
        lambda *args, **kwargs: {
            "base_transformer_sha256": sha256(specification.paths.transformer()),
            "base_transformer_file": "base.safetensors",
            "base_variant": "dev",
            "model_key": "2.5",
        },
    )
    monkeypatch.setattr(engine, "Accelerator", CPUAccelerator)
    monkeypatch.setattr(engine.prompt_cache, "get_or_build", lambda *args: torch.zeros(1, 3, 16, dtype=torch.bfloat16))
    calls = []

    def build(*args):
        model = _model(prompt_adaln=True).bfloat16()
        model.requires_grad_(False)
        torch.manual_seed(settings.init_seed)
        wrapped = get_peft_model(
            model,
            LoraConfig(
                r=2, lora_alpha=2, target_modules=adapters.LORA_TARGETS["attn"],
                lora_dropout=0.0, init_lora_weights=True
            ),
        )
        wrapped.get_base_model().set_gradient_checkpointing(True)
        wrapped.register_forward_hook(
            lambda module, args, kwargs, result: calls.append(kwargs["video"]), with_kwargs=True
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
            causal.BlockCache, "allocate", lambda *args, **kwargs: pytest.fail("bidirectional allocated a causal cache")
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
            from scripts.onestep_avatar.training.consumer_trace import validate
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
        resources.validate_records(records, 1, ["load", "export:0", "update:1", "export:1"],
                                   resources.read_budget(budget_path))
    if update_evidence:
        from scripts.onestep_avatar import training_update_check
        from scripts.onestep_avatar.training import update_state
        state_path = settings.output / "update_states/step_00001.pt"
        states = torch.load(state_path, weights_only=True)
        assert set(states) == set(initial_tensors)
        assert all(value["step"] == 1 for value in states.values())
        assert all(value["exp_avg"].dtype == torch.float32 for value in states.values())
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
        assert (update_state.export_name("base_model.model.block.lora_A.default.weight")
                == "diffusion_model.block.lora_A.weight")
