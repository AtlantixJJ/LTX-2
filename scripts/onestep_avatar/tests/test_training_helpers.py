"""CPU tests for typed-engine helpers.

They cover output archiving, the sigma and noise streams, step-zero LoRA export, W&B
settings, random segment starts and the startup refusal of a defective second video
before any model, accelerator or output mutation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import train
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import bidirectional
from scripts.onestep_avatar.tests.support.corpus import (
    two_videos as two_videos,  # noqa: PLC0414 -- explicit fixture re-export
)
from scripts.onestep_avatar.training import checkpoints, config, engine

# --- the one typed entry point ---------------------------------------------------------


@pytest.mark.parametrize("argv", [[], ["--subset", "s.json", "--output", "out"], ["--dry-run", "--overwrite"]])
def test_train_without_mode_exits_nonzero_before_any_write(tmp_path: Path, argv: list[str]) -> None:
    """The mode-less runtime is gone: no environment change, import of the engine or write."""
    environment = dict(os.environ)
    before = set(tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as refused:
        train.main([str(tmp_path / value) if value in ("s.json", "out") else value for value in argv])
    assert refused.value.code == train.MODE_REQUIRED  # a message exits with status 1
    assert dict(os.environ) == environment
    assert set(tmp_path.rglob("*")) == before


# --- output archiving ------------------------------------------------------------------


def test_archive_existing_run_moves_prior_contents_aside(tmp_path: Path) -> None:
    """The whole prior run moves into one timestamped subdirectory instead of being deleted."""
    output = tmp_path / "out"
    output.mkdir()
    (output / "metrics_rank0.jsonl").write_text('{"step": 0}\n')
    (output / "checkpoints").mkdir()
    (output / "checkpoints" / "lora_weights_step_00100.safetensors").write_bytes(b"")
    archived = engine.archive_existing_run(output)
    assert archived is not None
    assert archived.parent == output
    assert archived.name.startswith("archived_")
    assert (archived / "metrics_rank0.jsonl").is_file()
    assert (archived / "checkpoints" / "lora_weights_step_00100.safetensors").is_file()
    assert [p.name for p in output.iterdir()] == [archived.name]


def test_archive_existing_run_is_a_noop_on_an_empty_directory(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    assert engine.archive_existing_run(output) is None
    assert list(output.iterdir()) == []


# --- sigma and noise streams -----------------------------------------------------------


def test_multilevel_sigma_is_an_iid_uniform_draw() -> None:
    """Each (update, rank) draws a level i.i.d. uniformly, reproducibly from the noise seed."""
    levels = (0.909375, 0.725, 0.421875)
    assert config.training_sigmas(argparse.Namespace(sigma0=0.725, sigma_levels=list(levels))) == levels
    draws = [config.sigma_for_rank(levels, rank, step, 42) for step in range(3000) for rank in range(4)]
    assert set(draws) == set(levels)
    for level in levels:  # about a third of 12,000 draws each (binomial sd ~52)
        assert abs(draws.count(level) - 4000) < 300
    assert draws == [config.sigma_for_rank(levels, rank, step, 42) for step in range(3000) for rank in range(4)]
    assert draws != [config.sigma_for_rank(levels, rank, step, 43) for step in range(3000) for rank in range(4)]
    # Independent draws, not a rotation: some update repeats a level across ranks or skips one.
    assert any(len({config.sigma_for_rank(levels, rank, step, 42) for rank in range(3)}) < 3 for step in range(50))
    assert config.sigma_for_rank((0.725,), 3, 17, 42) == 0.725


def test_sigma_zero_is_refused() -> None:
    """sigma=0.0 adds no noise, so its loss and gradient are identically zero -- never trained."""
    with pytest.raises(SystemExit, match=r"sigma=0\.0"):
        config.training_sigmas(argparse.Namespace(sigma0=0.725, sigma_levels=[0.909375, 0.725, 0.421875, 0.0]))


def _noise_args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{"noise_policy": "fresh", "noise_seed": 42, **overrides})


def test_fresh_noise_changes_per_visit_and_is_shared_across_candidates() -> None:
    """Fresh noise is new per visit and ignores sigma/arm; fixed_per_chain is the old rule."""
    seeds = {
        config.training_noise_seed(_noise_args(), step=step, rank=rank, slot=slot, chain_index=0)
        for step in range(50)
        for rank in range(4)
        for slot in range(2)
    }
    assert len(seeds) == 50 * 4 * 2
    candidate = _noise_args(sigma0=0.725, guide_mode="d0")
    assert config.training_noise_seed(_noise_args(), step=7, rank=1, slot=0, chain_index=3) == (
        config.training_noise_seed(candidate, step=7, rank=1, slot=0, chain_index=5)
    )
    fixed = _noise_args(noise_policy="fixed_per_chain")
    assert (
        config.training_noise_seed(fixed, step=0, rank=0, slot=0, chain_index=3)
        == config.training_noise_seed(fixed, step=99, rank=1, slot=1, chain_index=3)
        == 42 * 100003 + 3 * 101
    )


# --- step-zero LoRA export -------------------------------------------------------------


def test_step_zero_lora_export_requires_exactly_zero_b() -> None:
    """The saved adapter, not just PEFT's in-memory module, is a provable no-op at step 0."""
    exported = {
        "diffusion_model.block.to_q.lora_A.weight": torch.randn(2, 4, dtype=torch.bfloat16),
        "diffusion_model.block.to_q.lora_B.weight": torch.zeros(4, 2, dtype=torch.bfloat16),
    }
    checkpoints.assert_exported_lora_is_noop(exported)
    exported["diffusion_model.block.to_q.lora_B.weight"][0, 0] = 1
    with pytest.raises(RuntimeError, match="non-zero LoRA delta"):
        checkpoints.assert_exported_lora_is_noop(exported)


def test_step_zero_lora_export_requires_b_weights() -> None:
    with pytest.raises(RuntimeError, match="no lora_B weights"):
        checkpoints.assert_exported_lora_is_noop({"diffusion_model.block.to_q.lora_A.weight": torch.zeros(2, 4)})


# --- W&B settings ----------------------------------------------------------------------

_SETTINGS = ["--mode", "bidirectional", "--subset", "/fake/subset.json", "--output", "/fake/out"]


def test_wandb_default_enabled() -> None:
    """W&B logging is enabled by default to 'onestep-avatar' in online mode."""
    settings = config.parse_settings(_SETTINGS)
    assert settings.wandb_project == "onestep-avatar"
    assert settings.wandb_mode == "online"
    assert not settings.no_wandb
    assert engine.wandb_is_enabled(settings) is True


@pytest.mark.parametrize(
    "extra", [["--no-wandb"], ["--wandb-mode", "disabled"], ["--wandb-project", ""], ["--wandb-project", "none"]]
)
def test_wandb_disable_options(extra: list[str]) -> None:
    settings = config.parse_settings([*_SETTINGS, *extra])
    assert engine.wandb_is_enabled(settings) is False
    assert engine.init_wandb(settings, config={}) is None


def test_init_wandb_calls_wandb_init(monkeypatch: pytest.MonkeyPatch) -> None:
    called_with = {}

    class FakeWandb:
        @staticmethod
        def init(**kwargs) -> str:
            called_with.update(kwargs)
            return "fake_run"

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    assert engine.init_wandb(config.parse_settings(_SETTINGS), config={"test_key": 123}) == "fake_run"
    assert called_with["project"] == "onestep-avatar"
    assert called_with["mode"] == "online"
    assert called_with["name"] == "out"
    assert called_with["config"] == {"test_key": 123}


# --- random segment starts -------------------------------------------------------------

# Recorded random starts for (seed, frames, span) over steps 0-5, ranks 0-3 and slots 0-1.
# plan_samples must reproduce this exact stream, so saved random-start runs stay reproducible.
ORIGINAL_STARTS = {
    (42, 28, 17): [7, 7, 11, 6, 7, 9, 2, 3, 10, 7, 11, 6],
    (7, 29, 18): [8, 0, 9, 9, 6, 0, 11, 8, 10, 7, 5, 6],
    (123, 18, 9): [0, 0, 1, 0, 3, 3, 5, 8, 9, 1, 8, 5],
}


def _start(seed: int, frames: int, span: int, step: int, rank: int, slot: int) -> int:
    return bidirectional.plan_samples(
        frames, span_latent_frames=span, start_policy="random", seed=seed, step=step, rank=rank, slot=slot
    )[0]


def test_random_start_keeps_the_original_seed_key_and_stream() -> None:
    key = config.window_start_draw(42)["key"]
    for (seed, frames, span), expected in ORIGINAL_STARTS.items():
        starts = [_start(seed, frames, span, s, r, sl) for s in range(6) for r in range(4) for sl in range(2)]
        assert starts[:12] == expected
        documented = [
            random.Random(key.format(seed=seed, step=s, rank=r, slot=sl)).randrange(frames - span + 1)
            for s in range(6)
            for r in range(4)
            for sl in range(2)
        ]
        assert starts == documented


def test_random_start_is_uniform_reproducible_and_bounded() -> None:
    """Starts cover 0..F-span uniformly; a full-length span always starts at 0."""
    starts = [_start(42, 28, 17, s, r, 0) for s in range(2000) for r in range(4)]
    assert set(starts) == set(range(12))
    for value in range(12):
        assert abs(starts.count(value) - 8000 / 12) < 150
    assert starts == [_start(42, 28, 17, s, r, 0) for s in range(2000) for r in range(4)]
    assert {_start(7, 17, 17, s, 0, 0) for s in range(50)} == {0}
    with pytest.raises(ValueError, match="segment length"):
        _start(42, 16, 17, 0, 0, 0)


# --- startup refusal of a defective second video ---------------------------------------


def _repin(path: Path, membership: dict, source: dict, role: str, bundle: Path) -> None:
    """Record the defective bytes, so the shared reader's deeper checks are what refuses them."""
    source[f"{role}_latent_sha256"] = sha256(bundle)
    membership["sha256"] = subset.membership_hash(membership)
    path.write_text(json.dumps(membership))


DEFECTS = {
    "missing_capture": ("capture", "No such file|does not exist"),
    "changed_bytes": ("capture", "encoded content changed"),
    "missing_fps": ("capture", "fps must be a finite positive"),
    "zero_fps": ("capture", "fps must be a finite positive"),
    "nan_fps": ("capture", "fps must be a finite positive"),
    "bad_tensor": ("capture", "nonempty floating"),
    "capture_geometry": ("capture", "shape or frame rate differs from membership"),
    "missing_guide": ("guide", "No such file|does not exist"),
    "guide_shape": ("guide", "shape or frame rate differs from membership"),
    "guide_fps": ("guide", "shape or frame rate differs from membership"),
}


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_startup_refuses_a_defective_second_video_without_touching_the_run(
    two_videos: tuple[Path, dict], tmp_path: Path, defect: str, dry_run: bool
) -> None:
    path, membership = two_videos
    role, error = DEFECTS[defect]
    source = membership["sources"][1]
    bundle = (
        Path(membership["corpus_root"])
        / source["relative_dir"]
        / (dataset.guide_bundle_name("white") if role == "guide" else dataset.capture_bundle_name("white"))
    )
    record = torch.load(bundle, weights_only=True)
    if defect in ("missing_guide", "missing_capture"):
        bundle.unlink()
    elif defect == "changed_bytes":
        bundle.write_bytes(b"changed encoded input")
    else:
        if defect in ("capture_geometry", "guide_shape"):
            record["master"] = record["master"][:, :, :1, :]
        elif defect == "guide_fps":
            record["fps"] = 25.0
        elif defect == "bad_tensor":
            record["master"] = record["master"][0]
        else:
            record.pop("fps")
            if defect != "missing_fps":
                record["fps"] = 0.0 if defect == "zero_fps" else math.nan
        torch.save(record, bundle)
        _repin(path, membership, source, role, bundle)
    output = tmp_path / "run"
    output.mkdir()
    old_log = output / "metrics_rank0.jsonl"
    old_log.write_bytes(b"old run\n")
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(path),
            "--output",
            str(output),
            "--overwrite",
            "--variant",
            "dev",
            "--objective",
            "white",
            "--guide-mode",
            "d1" if role == "guide" else "d0",
            "--span-latent-frames",
            "7",
            *(["--dry-run"] if dry_run else []),
        ]
    )
    with pytest.raises((ValueError, SystemExit, FileNotFoundError), match=error):
        engine.run_settings(settings)
    assert old_log.read_bytes() == b"old run\n"
    assert list(output.iterdir()) == [old_log]


def test_the_complete_two_video_membership_passes_preflight(two_videos: tuple[Path, dict], tmp_path: Path) -> None:
    """The fixture is valid in both arms, so each defect above is the only failure."""
    path, membership = two_videos
    for guide_mode in ("d0", "d1"):
        settings = config.parse_settings(
            [
                "--mode",
                "bidirectional",
                "--subset",
                str(path),
                "--output",
                str(tmp_path / guide_mode),
                "--variant",
                "dev",
                "--objective",
                "white",
                "--guide-mode",
                guide_mode,
                "--span-latent-frames",
                "7",
            ]
        )
        store, plan, _, archive = engine.prepare_run(settings)
        assert len(store) == 2
        assert plan["membership_sha256"] == membership["sha256"]
        assert archive is False


@pytest.mark.usefixtures("two_videos")  # base resolution, Accelerator and model building fail if reached
@pytest.mark.parametrize("kind", ["one_step_argavatar_window_chains", "one_step_argavatar_block_chains"])
def test_an_old_subset_is_refused_before_model_or_output_work(tmp_path: Path, kind: str) -> None:
    """An old window- or block-chain subset is never reinterpreted: only version-two
    membership reaches preflight, so old indices cannot select different frames."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"kind": kind, "corpus_root": "/nonexistent", "chains": [], "sources": []}))
    output = tmp_path / "used"
    output.mkdir()
    (output / "metrics_rank0.jsonl").write_bytes(b"old run\n")
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(old),
            "--output",
            str(output),
            "--overwrite",
            "--variant",
            "dev",
            "--objective",
            "white",
        ]
    )
    with pytest.raises(ValueError, match="version-two onestep_avatar_membership"):
        engine.run_settings(settings)
    assert (output / "metrics_rank0.jsonl").read_bytes() == b"old run\n"
    assert list(output.iterdir()) == [output / "metrics_rank0.jsonl"]
