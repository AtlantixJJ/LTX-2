"""Bounded CPU updates exercise the typed engine with real tiny LTX/PEFT modules."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp._runtime_utils import _cast_forward_inputs
from torch.distributed.fsdp.wrap import CustomPolicy

from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import precompute
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters
from scripts.onestep_avatar.tests.support.corpus import build_runtime_membership
from scripts.onestep_avatar.tests.support.model import _model
from scripts.onestep_avatar.tests.support.training import CPUAccelerator, run_bounded_update
from scripts.onestep_avatar.training import config, engine


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_native_builder_preserves_typed_fsdp_modality_precision(
    mode: str, mixed_precision: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    settings = config.parse_settings(
        [
            "--mode",
            mode,
            "--subset",
            str(tmp_path / "membership.json"),
            "--output",
            str(tmp_path / "output"),
            "--lora-rank",
            "2",
        ]
    )
    settings.init_device = "cpu"
    policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16) if mixed_precision else None
    original = None if policy is None else asdict(policy)
    plugin = SimpleNamespace(mixed_precision_policy=policy)
    accelerator = SimpleNamespace(distributed_type=DistributedType.FSDP, state=SimpleNamespace(fsdp_plugin=plugin))
    specification = SimpleNamespace(paths=SimpleNamespace(transformer=lambda: "checked-base.safetensors"))
    monkeypatch.setattr(engine, "load_transformer", lambda **_kwargs: _model().bfloat16())

    def wrapping(module: torch.nn.Module, *, recurse: bool = False, nonwrapped_numel: int = 0) -> bool:
        del nonwrapped_numel, recurse
        return isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "BasicAVTransformerBlock"

    monkeypatch.setattr(engine, "fsdp_auto_wrap_policy", lambda _model: wrapping)
    model = engine.build_transformer(specification, settings, accelerator)
    assert isinstance(plugin.auto_wrap_policy, CustomPolicy)
    chosen = plugin.auto_wrap_policy._run_policy(model, set(), {"mixed_precision": policy})
    assert set(chosen) == {module for module in model.modules() if wrapping(module)}
    for module, options in chosen.items():
        trainable = list(module.parameters()) and all(p.requires_grad for p in module.parameters())
        assert options["mixed_precision"] is (None if trainable else policy)
    assert all(p.dtype == torch.float32 for name, p in model.named_parameters() if ".lora_" in name)
    if policy is None:
        assert plugin.mixed_precision_policy is None
        return
    expected = {**original, "cast_root_forward_inputs": False}
    assert asdict(plugin.mixed_precision_policy) == expected
    assert asdict(policy) == original
    # Exercise the installed recursive cast with the actual native dataclass.
    sigma = torch.tensor([0.725], dtype=torch.float32)
    video = Modality(
        latent=torch.ones(1, 2, 8, dtype=torch.bfloat16),
        sigma=sigma,
        timesteps=sigma.expand(1, 2),
        positions=torch.full((1, 3, 2, 2), 0.123456),
        context=torch.ones(1, 2, 16, dtype=torch.bfloat16),
    )
    dtype = policy.param_dtype if plugin.mixed_precision_policy.cast_root_forward_inputs else None
    _, forwarded = _cast_forward_inputs(dtype, video=video)
    observed = forwarded["video"]
    assert observed.sigma.dtype == torch.float32
    assert torch.equal(observed.sigma, sigma)
    assert torch.equal(observed.timesteps, video.timesteps)
    assert torch.equal(observed.positions, video.positions)


UPDATE_CASES = [
    ("bidirectional", False, False, False, "off", False),
    ("bidirectional", True, False, True, "pass", True),
    ("causal", False, False, False, "update_failure", True),
    ("causal", True, False, True, "off", False),
    ("bidirectional", False, False, True, "update_failure", False),
    ("causal", False, False, False, "pass", False),
    ("bidirectional", True, False, False, "off", True),
    ("bidirectional", True, False, False, "update_failure", False),
    ("bidirectional", False, True, False, "off", False),
    ("causal", False, True, False, "off", False),
]


def test_update_cases_cover_all_level_pairs() -> None:
    import itertools  # noqa: PLC0415 -- fixture-only imports follow the CPU/import blocker setup

    selected = [tuple(row[i] for i in (0, 1, 3, 4, 5)) for row in UPDATE_CASES if not row[2]]
    levels = [
        ["bidirectional", "causal"],
        [False, True],
        [False, True],
        ["off", "pass", "update_failure"],
        [False, True],
    ]
    for i, j in itertools.combinations(range(5), 2):
        assert {(row[i], row[j]) for row in selected} == set(itertools.product(levels[i], levels[j]))


@pytest.mark.parametrize(
    ("mode", "preview_failure", "queued", "update_evidence", "resource_check", "consumer_trace"), UPDATE_CASES
)
def test_one_bounded_update_saves_real_zero_and_updated_lora(
    mode: str,
    preview_failure: bool,
    queued: bool,
    update_evidence: bool,
    resource_check: str,
    consumer_trace: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_bounded_update(
        mode, preview_failure, queued, update_evidence, resource_check, consumer_trace, tmp_path, monkeypatch
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_only_main_rank_archives_a_used_output_after_setup(
    rank: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``run_settings``, an explicit overwrite archives the prior run once, on the
    main rank only, after model setup and between barriers; another rank never touches the
    prior contents."""
    _membership, _ = build_runtime_membership(tmp_path, seed=3)
    output = tmp_path / "run"
    output.mkdir()
    (output / "config.json").write_text("prior config")
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(tmp_path / "membership.json"),
            "--output",
            str(output),
            "--variant",
            "dev",
            "--objective",
            "white",
            "--guide-mode",
            "d0",
            "--lora-rank",
            "2",
            "--steps",
            "0",
            "--no-wandb",
            "--overwrite",
        ]
    )
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=8),
        sigmas=[0.725],
        paths=SimpleNamespace(transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: tmp_path / "vae"),
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
    monkeypatch.setattr(engine.prompt_cache, "get_or_build", lambda *_args: torch.zeros(1, 3, 16, dtype=torch.bfloat16))
    events = []

    class RankAccelerator(CPUAccelerator):
        num_processes = 2
        process_index = rank
        is_main_process = rank == 0

        def prepare(self, model: torch.nn.Module, optimizer: object) -> tuple[torch.nn.Module, object]:
            events.append("prepare")
            return model, optimizer

        def wait_for_everyone(self) -> None:
            events.append("barrier")

        def gather(self, tensor: torch.Tensor) -> torch.Tensor:
            """Both ranks report this rank's record; the runtime policy keeps each rank's index."""
            if tensor.dtype != torch.uint8:
                return torch.cat([tensor, tensor])
            record = json.loads(bytes(tensor.tolist()).split(b"\x00", 1)[0])
            rows = []
            for index in range(self.num_processes):
                payload = json.dumps({**record, "rank": index}, sort_keys=True).encode()
                row = torch.zeros_like(tensor)
                row[: len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
                rows.append(row)
            return torch.cat(rows)

    monkeypatch.setattr(engine, "Accelerator", RankAccelerator)

    def build(*_args: object) -> torch.nn.Module:
        model = _model().bfloat16()
        model.requires_grad_(False)
        return get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=adapters.LORA_TARGETS["attn"],
                lora_dropout=0.0,
                init_lora_weights=True,
            ),
        )

    monkeypatch.setattr(engine, "build_transformer", build)
    archive = engine.archive_existing_run
    monkeypatch.setattr(engine, "archive_existing_run", lambda target: events.append("archive") or archive(target))
    assert engine.run_settings(settings) == 0
    assert "prepare" in events
    if rank == 0:
        assert events.count("archive") == 1
        index = events.index("archive")
        assert events.index("prepare") < index
        assert events[index - 1] == "barrier"
        assert events[index + 1] == "barrier"
        archived = next(output.glob("archived_*"))
        assert (archived / "config.json").read_text() == "prior config"
    else:
        assert "archive" not in events
        assert (output / "config.json").read_text() == "prior config"
