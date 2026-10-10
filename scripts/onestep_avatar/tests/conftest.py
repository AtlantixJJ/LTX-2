"""Shared test fixtures.

Controlled lifecycle substitutions (real launch-gate checks never use them) and the shared
saved-input fixtures `old_subset` and `checked_settings`, registered here once instead of
being imported into every test module that needs them.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import PACKAGE_ROOT
from scripts.onestep_avatar.execution import queue_launch
from scripts.onestep_avatar.hashing import sha256

if os.environ.get("ONESTEP_AVATAR_BLOCK_EXPERIMENTS") == "1":
    hook_directory = PACKAGE_ROOT / "tests/blocked_imports"
    hook_spec = importlib.util.spec_from_file_location(
        "onestep_avatar_experiment_import_blocker", hook_directory / "sitecustomize.py"
    )
    if hook_spec is None or hook_spec.loader is None:
        raise ImportError("experiment import blocker is absent")
    hook_spec.loader.exec_module(importlib.util.module_from_spec(hook_spec))
    previous_path = os.environ.get("PYTHONPATH")
    if previous_path is None or str(hook_directory) not in previous_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = str(hook_directory) + ("" if previous_path is None else os.pathsep + previous_path)
    collect_ignore = ["experiments"]


@pytest.fixture
def controlled_training_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the synthetic marker test on hash/mode checks; science has its own tests."""
    from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- defer loader or fixture imports until preflight and test hooks are installed
        engine,
    )

    monkeypatch.setattr(engine, "verify_training_conditions", lambda _job, _checkpoint, _contract: None)


@pytest.fixture
def controlled_evaluation_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope synthetic lifecycle records to lifecycle and generic tensor checks.

    These tests do not contain checked masters, weights or actual sampler output.
    Scientific result binding is tested separately against the production verifier.
    No production path bypasses the verifier.
    """
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- install the import blocker before model owners

    monkeypatch.setattr(evaluate, "verify_evaluation_conditions", lambda _arguments, _paths: None)
    monkeypatch.setattr(evaluate, "evaluation_evidence_paths", lambda _arguments, paths: paths)


@pytest.fixture
def controlled_queue_launch(monkeypatch: pytest.MonkeyPatch, controlled_queue_observation: None) -> None:  # noqa: ARG001 -- fixture installs controlled observer
    """Keep old fake-Popen tests scoped to receipts/retries, not OS registration.

    Each caller already replaces Popen with an immediate synthetic completion.
    These substitutions do not establish bootstrap/grant or process correctness.
    Real-process integration tests retain every production protocol function.
    """
    monkeypatch.setattr(queue_launch, "guard_command", lambda path: queue_launch.load_request(path)["command"])

    def registration(path: Path, child, _command, _refresh, **_limits) -> dict:  # noqa: ANN001 -- controlled child interfaces
        request = queue_launch.load_request(path)
        return {"identity": {"pid": child.pid, "start_ticks": 1, "command": request["command"], "terminal": False}}

    monkeypatch.setattr(queue_launch, "wait_registration", registration)
    monkeypatch.setattr(queue_launch, "publish_grant", lambda _path: {})


@pytest.fixture
def controlled_queue_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope fake observations to receipt/guard fixtures, never native acceptance."""
    from scripts.onestep_avatar.execution import queue, runtime, supervision  # noqa: PLC0415 -- fixture only

    def observe(child: object, **arguments: object) -> dict:
        code = child.poll()
        if code is None:
            code = child.wait(timeout=5)
        result = {"state": "passed", "error": None, "returncode": code, "workers_absent": True}
        queue_launch.publish_exclusive(arguments["evidence_path"], result)
        return result

    monkeypatch.setattr(supervision, "supervise", observe)
    monkeypatch.setattr(runtime, "verify_result", lambda *_: {})
    monkeypatch.setattr(queue, "verify_attempt_evidence", lambda *_: [])


@pytest.fixture
def old_subset(tmp_path: Path) -> dict:
    from scripts.onestep_avatar.corpus import dataset  # noqa: PLC0415 -- fixture-only saved-input producer

    sources = []
    chains = []
    for actor, group in (("1", "train"), ("2", "held_out")):
        name = f"Part_1/{actor}/views/view00"
        view = tmp_path / name
        view.mkdir(parents=True)
        (view / "rgb.mp4").write_bytes(b"saved capture bytes" + actor.encode())
        torch.save(
            {
                "schema_version": 2,
                "master": torch.arange(56).reshape(2, 7, 2, 2).float(),
                "fps": 30.0,
                "objective": "white",
                "source": name,
                "vae_fingerprint": "saved VAE identity",
                "box_xyxy": [0, 0, 64, 64],
                "edge": 64,
                "encode_contract_version": 1,
            },
            view / dataset.capture_bundle_name("white"),
        )
        sources.append(
            {
                "relative_dir": name,
                "actor": actor,
                "fps": 30.0,
                "n_latent_frames": 7,
                "n_blocks": 3,
                "rgb_sha256": sha256(view / "rgb.mp4"),
                "guide_sha256": None,
            }
        )
        chains += [
            {"source": name, "actor": actor, "split": group, "blocks": [0, 1], "seed_is_clip_start": True},
            {"source": name, "actor": actor, "split": group, "blocks": [1, 2], "seed_is_clip_start": False},
        ]
    return {
        "kind": "one_step_argavatar_block_chains",
        "schema_version": 1,
        "objective": "white",
        "corpus_root": str(tmp_path),
        "capture_manifest": {"sha256": "original crop record"},
        "splits": {"train": ["1"], "held_out": ["2"]},
        "sources": sources,
        "chains": chains,
        "geometry": {
            "latent_time_scale": 8,
            "block_latent_frames": 2,
            "context_latent_frames": 1,
            "sink_latent_frames": 1,
        },
        "chain_length": 2,
    }


@pytest.fixture
def checked_settings(old_subset: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    from ltx_core.types import SpatioTemporalScaleFactors  # noqa: PLC0415 -- needed only by preflight fixtures
    from scripts.onestep_avatar.corpus import precompute, subset  # noqa: PLC0415 -- fixture-only saved-input producers
    from scripts.onestep_avatar.training import config, engine  # noqa: PLC0415 -- import after the architecture blocker

    membership, _ = subset.convert_legacy(old_subset, original_file_sha256="a" * 64)
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(membership))
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(path),
            "--output",
            str(tmp_path / "output"),
            "--variant",
            "dev",
            "--objective",
            "white",
            "--guide-mode",
            "d0",
            "--span-latent-frames",
            "7",
        ]
    )
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=2),
        sigmas=[0.725],
        paths=SimpleNamespace(
            transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: tmp_path / "vae.safetensors"
        ),
    )
    monkeypatch.setattr(engine.backbone, "resolve", lambda *_args: specification)
    monkeypatch.setattr(precompute, "file_fingerprint", lambda _path: "saved VAE identity")
    monkeypatch.setattr(
        engine.backbone,
        "identity",
        lambda *_args, **_kwargs: {
            "base_transformer_sha256": "a" * 64,
            "base_transformer_file": "base.safetensors",
            "base_variant": "dev",
            "model_key": "2.5",
        },
    )
    monkeypatch.setattr(engine, "build_transformer", lambda *_args: pytest.fail("preflight loaded a transformer"))
    return settings, membership
