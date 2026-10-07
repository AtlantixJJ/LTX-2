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
from scripts.onestep_avatar import dataset, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import causal
from scripts.onestep_avatar.tests.test_causal_core import _model
from scripts.onestep_avatar.training import checkpoints, config, engine


class CPUAccelerator:
    num_processes = 1
    process_index = 0
    is_main_process = True
    distributed_type = DistributedType.NO
    device = torch.device("cpu")

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
@pytest.mark.parametrize("preview_failure", [False, True])
@pytest.mark.parametrize("queued", [False, True])
def test_one_bounded_update_saves_real_zero_and_updated_lora(
    mode: str, preview_failure: bool, queued: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if queued:
        from scripts.onestep_avatar.queue_protocol import JOB_ENV, TOKEN_ENV
        monkeypatch.setenv(JOB_ENV, "a" * 64)
        monkeypatch.setenv(TOKEN_ENV, "b" * 32)
    view = tmp_path / "corpus" / "actor" / "view"
    view.mkdir(parents=True)
    path = view / dataset.capture_bundle_name("white")
    generator = torch.Generator().manual_seed(17)
    capture = torch.randn(8, 7, 2, 2, generator=generator)
    bundle = {"schema_version": 2, "master": capture, "fps": 30.0, "objective": "white", "source": "actor/view", "vae_fingerprint": "runtime VAE identity"}
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
                "capture_encode_record": {k: bundle[k] for k in ("schema_version", "fps", "objective", "source", "vae_fingerprint")},
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
        paths=SimpleNamespace(transformer=lambda: tmp_path / "unused_base.safetensors", video_vae=lambda: tmp_path / "unused_vae.safetensors"),
    )
    monkeypatch.setattr(engine.backbone, "resolve", lambda *args: specification)
    from scripts.onestep_avatar import precompute
    monkeypatch.setattr(precompute, "file_fingerprint", lambda path: "runtime VAE identity")
    monkeypatch.setattr(
        engine.backbone,
        "identity",
        lambda *args, **kwargs: {
            "base_transformer_sha256": "a" * 64,
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
                r=2, lora_alpha=2, target_modules=config.LORA_TARGETS["attn"], lora_dropout=0.0, init_lora_weights=True
            ),
        )
        wrapped.register_forward_hook(
            lambda module, args, kwargs, result: calls.append(kwargs["video"]), with_kwargs=True
        )
        return wrapped

    monkeypatch.setattr(engine, "build_transformer", build)
    if mode == "bidirectional":
        monkeypatch.setattr(
            causal.BlockCache, "allocate", lambda *args, **kwargs: pytest.fail("bidirectional allocated a causal cache")
        )
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
    logs = [json.loads(line) for line in (settings.output / "metrics_rank0.jsonl").read_text().splitlines()]
    assert len(logs) == 1
    assert "anchor" not in logs[0]
    assert logs[0]["step"] == 1
    assert len(calls) == (1 if mode == "bidirectional" else 4)
    assert logs[0]["call_counts"] == (
        {"prime": 0, "denoise": 1, "backward": 1, "refresh": 0}
        if mode == "bidirectional"
        else {"prime": 1, "denoise": 2, "backward": 2, "refresh": 1}
    )
    saved_plan = json.loads((settings.output / "frame_plan.json").read_text())
    assert saved_plan["sha256"] == subset.record_hash(saved_plan)
    for key in ("mode", "membership_sha256", "frame_plan_sha256"):
        assert key in json.loads((settings.output / "config.json").read_text())
