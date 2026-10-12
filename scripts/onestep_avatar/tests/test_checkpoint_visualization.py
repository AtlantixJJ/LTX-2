"""Checkpoint videos use actual official entries and complete four-adapter groups."""

import json
from pathlib import Path

import pytest
import torch

from ltx_core.components.schedulers import LTX2Scheduler
from scripts.onestep_avatar.comparisons import checkpoint_groups, checkpoint_schedules, prepare_checkpoint_visualization
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.hashing import sha256


def test_official_tail_keeps_the_requested_minimum_and_even_indices() -> None:
    result = checkpoint_schedules(0.38, 0.9)
    official = LTX2Scheduler().execute(steps=30).tolist()
    assert result["selected_indices"] == [13, 18, 22, 27]
    assert result["four_step"] == [official[index] for index in [13, 18, 22, 27]] + [0.0]
    assert result["one_step"] == [result["four_step"][0], 0.0]
    assert all(0.38 <= value <= 0.9 for value in result["four_step"][:-1])


def test_groups_cover_every_checkpoint_with_a_shared_anchor() -> None:
    steps = [0, 1, 200, 400, 600, 800, 1000]
    groups = checkpoint_groups(steps)
    assert groups == [[0, 1, 200, 400], [400, 600, 800, 1000]]
    assert all(len(group) == 4 for group in groups)
    assert set(groups[0]) | set(groups[1]) == set(steps)
    assert set(groups[0]) & set(groups[1]) == {400}


@pytest.mark.parametrize("bounds", [(0, 0.9), (0.9, 0.4), (0.38, 1.1), (0.88, 0.9)])
def test_invalid_or_too_short_tail_is_refused(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match=r"noise bounds|bounded official tail"):
        checkpoint_schedules(*bounds)


@pytest.mark.parametrize("steps", [[0, 1], [0, 1, 200, 400, 600, 800, 800], [-1, 1, 200, 400, 600, 800, 1000]])
def test_incomplete_or_duplicate_groups_are_refused(steps: list[int]) -> None:
    with pytest.raises(ValueError, match="seven unique increasing"):
        checkpoint_groups(steps)


@pytest.mark.parametrize("wrong_group", [False, True])
def test_preparation_dry_run_preserves_inputs_and_checks_original_groups(
    checked_settings: tuple, tmp_path: Path, wrong_group: bool
) -> None:
    settings, membership = checked_settings
    run = tmp_path / "finished"
    run.mkdir()
    config = {"mode": "bidirectional", "variant": "dev", "guide_mode": "d0",
              "membership_sha256": membership["sha256"]}
    (run / "config.json").write_text(json.dumps(config))
    (run / "checkpoints").mkdir()
    for step in (0, 1, 200, 400, 600, 800, 1000):
        for suffix in ("safetensors", "complete.json"):
            (run / "checkpoints" / f"lora_weights_step_{step:05d}.{suffix}").touch()
    output = tmp_path / "visualization"
    original_bytes = settings.subset.read_bytes()
    arguments = ["--prepare-checkpoint-visualization", "--training-run", str(run),
                 "--membership", str(settings.subset), "--train-source", "Part_1/1/views/view00",
                 "--held-out-source", f"Part_1/{1 if wrong_group else 2}/views/view00",
                 "--output", str(output), "--dry-run"]
    if wrong_group:
        with pytest.raises(ValueError, match="original held_out group"):
            prepare_checkpoint_visualization(arguments)
    else:
        assert prepare_checkpoint_visualization(arguments) == 0
    assert not output.exists()
    assert settings.subset.read_bytes() == original_bytes


def test_preparation_uses_each_complete_master_and_detected_dev_guidance(
    checked_settings: tuple, tmp_path: Path
) -> None:
    settings, membership = checked_settings
    held = membership["sources"][1]
    membership.pop("excluded")
    membership["excluded_counts"] = {}
    held["n_latent_frames"] = 9
    held["shape"][1] = 9
    held["capture_encode_record"]["pixel_frames"] = 65
    capture = Path(membership["corpus_root"]) / held["relative_dir"] / dataset.capture_bundle_name("white")
    torch.save({**held["capture_encode_record"], "master": torch.arange(72).reshape(2, 9, 2, 2).float()}, capture)
    held["capture_latent_sha256"] = sha256(capture)
    membership["sha256"] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    run = tmp_path / "finished"
    (run / "checkpoints").mkdir(parents=True)
    config = {"mode": "bidirectional", "variant": "dev", "guide_mode": "d0", "model": "2.5",
              "objective": "white", "seed": 42, "mode_settings": {"span_latent_frames": 2},
              "membership_sha256": membership["sha256"]}
    (run / "config.json").write_text(json.dumps(config))
    for step in (0, 1, 200, 400, 600, 800, 1000):
        for suffix in ("safetensors", "complete.json"):
            (run / "checkpoints" / f"lora_weights_step_{step:05d}.{suffix}").touch()
    output = tmp_path / "visualization"
    assert prepare_checkpoint_visualization([
        "--prepare-checkpoint-visualization", "--training-run", str(run), "--membership", str(settings.subset),
        "--train-source", "Part_1/1/views/view00", "--held-out-source", "Part_1/2/views/view00",
        "--output", str(output),
    ]) == 0
    plan = json.loads((output / "visualization.json").read_text())
    assert plan["coverage"] == "complete_saved_encoding"
    assert plan["training_encoded_frames"] == 2
    assert plan["guidance"] == {"cfg": 3.0, "stg": 1.0, "stg_blocks": [28], "rescale": 0.7,
                                "skip_step": 0, "source": plan["guidance"]["source"]}
    assert Path(plan["guidance"]["source"]).name == "ltx-2.5-22b-dev-transformer-bf16.safetensors"
    for video in plan["videos"]:
        length = 7 if video["actor"] == "1" else 9
        assert video["encoded_frames"] == length
        assert video["pixel_frames"] == (length - 1) * 8 + 1
        spec = json.loads(Path(video["spec"]).read_text())["comparisons"][0]
        assert spec["span"] == length
    jobs = json.loads((output / "jobs.json").read_text())["jobs"]
    assert len(jobs) == 40
    for job in (j for j in jobs if j["kind"] == "evaluate"):
        arguments = job["arguments"]
        length = 7 if job["id"].startswith("train_") else 9
        assert arguments[arguments.index("--span-latent-frames") + 1] == str(length)
        frame_plan = json.loads(Path(arguments[arguments.index("--frame-plan") + 1]).read_text())
        assert frame_plan["samples"][0]["ranges"] == [[0, length]]
        noise = torch.load(arguments[arguments.index("--noise-file") + 1], weights_only=True)
        assert noise.shape == (1, length * 4, 2)
        for flag, value in [("--cfg", "3.0"), ("--stg", "1.0"), ("--rescale", "0.7"), ("--stg-blocks", "28")]:
            assert arguments[arguments.index(flag) + 1] == value
