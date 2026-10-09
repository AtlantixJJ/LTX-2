"""Explicit settings and frame plans fail before any model or output mutation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import subset
from scripts.onestep_avatar.training import config

SCALE = SpatioTemporalScaleFactors(8, 32, 32)
BASE = ["--subset", "/unused/videos.json", "--output", "/unused/output"]


def _membership() -> dict:
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "corpus_root": "/unused",
        "splits": {"train": ["actor"]},
        "sources": [
            {
                "relative_dir": "actor/view",
                "actor": "actor",
                "split": "train",
                "n_latent_frames": 17,
                "shape": [128, 17, 2, 2],
                "fps": 30,
            }
        ],
        "excluded": {},
    }
    membership["sha256"] = subset.membership_hash(membership)
    return membership


def test_mode_is_required() -> None:
    with pytest.raises(SystemExit):
        config.parse_settings(BASE)


@pytest.mark.parametrize(
    "options",
    [
        ["--block-latent-frames", "16"],
        ["--context-latent-frames", "0"],
        ["--blocks-per-sample", "1"],
        ["--teacher-forcing"],
    ],
)
def test_bidirectional_rejects_explicit_causal_settings(options: list[str]) -> None:
    with pytest.raises(SystemExit):
        config.parse_settings([*BASE, "--mode", "bidirectional", *options])


def test_bidirectional_resolves_without_any_causal_fields() -> None:
    settings = config.parse_settings(
        [*BASE, "--mode", "bidirectional", "--objective", "white", "--span-latent-frames", "17", "--seed", "7"]
    )
    assert isinstance(settings, config.RunSettings)
    assert isinstance(settings.mode_settings, config.BidirectionalSettings)
    assert set(asdict(settings.mode_settings)) == {"span_latent_frames", "start_policy", "attention"}
    assert settings.init_seed == settings.data_seed == settings.noise_seed == 7
    plan = config.build_frame_plan(settings, _membership(), SCALE)
    assert plan["samples"][0]["ranges"] == [[0, 17]]
    assert "blocks" not in plan["samples"][0]
    assert "geometry" not in plan
    assert plan["sha256"] == subset.record_hash(plan)


def test_causal_groups_blocks_and_discards_incomplete_sample_tail() -> None:
    settings = config.parse_settings(
        [*BASE, "--mode", "causal", "--objective", "white", "--block-latent-frames", "2", "--blocks-per-sample", "3"]
    )
    plan = config.build_frame_plan(settings, _membership(), SCALE)
    assert [s["blocks"] for s in plan["samples"]] == [[0, 1, 2], [3, 4, 5]]
    assert plan["samples"][0]["ranges"] == [[0, 3], [3, 5], [5, 7]]
    assert plan["samples"][1]["ranges"] == [[7, 9], [9, 11], [11, 13]]
    assert plan["mode_settings"]["context_latent_frames"] == 8


@pytest.mark.parametrize(
    "extra",
    [
        ["--lora-alpha", "4"],
        ["--lr", "nan"],
        ["--sigma-levels", "nan"],
        ["--sigma-levels", "0.725", "0.725"],
        ["--sigma0", "0"],
        ["--chains-per-rank", "0"],
        ["--context-latent-frames", "17"],
        ["--start-policy", "random", "--blocks-per-sample", "2"],
    ],
)
def test_invalid_settings_fail_before_file_access(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        config.parse_settings([*BASE, "--mode", "causal", *extra])


def test_random_segment_records_the_original_start_seed_key() -> None:
    settings = config.parse_settings(
        [
            *BASE,
            "--mode",
            "bidirectional",
            "--objective",
            "white",
            "--span-latent-frames",
            "17",
            "--start-policy",
            "random",
        ]
    )
    plan = config.build_frame_plan(settings, _membership(), SCALE)
    assert plan["start_draw"]["key"] == "onestep_avatar.window:{seed}:{step}:{rank}:{slot}"


@pytest.mark.parametrize("defect", ["missing", "seed", "key", "rule"])
def test_reproduction_plan_refuses_changed_start_draw(tmp_path: Path, defect: str) -> None:
    settings = config.parse_settings([*BASE, "--mode", "bidirectional", "--objective", "white",
                                     "--span-latent-frames", "17", "--start-policy", "random"])
    membership = _membership()
    plan = config.build_frame_plan(settings, membership, SCALE)
    if defect == "missing":
        plan.pop("start_draw")
    else:
        plan["start_draw"][defect] = 99 if defect == "seed" else "different"
    plan["sha256"] = subset.record_hash(plan)
    settings.frame_plan = tmp_path / "plan.json"
    settings.frame_plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="random start draw"):
        config.select_frame_plan(settings, membership, SCALE)


def test_explicit_frame_plan_is_checked_without_rewriting_it(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    settings = config.parse_settings([*BASE, "--mode", "causal", "--objective", "white"])
    membership = _membership()
    plan = config.build_frame_plan(settings, membership, SCALE)
    text = json.dumps(plan)
    path.write_text(text)
    settings.frame_plan = path
    assert config.select_frame_plan(settings, membership, SCALE) == plan
    assert path.read_text() == text
    plan["samples"][0]["ranges"][0][1] = 4
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="hash"):
        config.select_frame_plan(settings, membership, SCALE)


def test_random_causal_plan_cannot_select_later_original_blocks(tmp_path: Path) -> None:
    settings = config.parse_settings([*BASE, "--mode", "causal", "--objective", "white",
                                     "--span-latent-frames", "3", "--start-policy", "random",
                                     "--blocks-per-sample", "1"])
    membership = _membership()
    plan = config.build_frame_plan(settings, membership, SCALE)
    plan["samples"][0].update(blocks=[1], ranges=[[3, 5]], seed_is_clip_start=False)
    plan["sha256"] = subset.record_hash(plan)
    settings.frame_plan = tmp_path / "plan.json"
    settings.frame_plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="independent block-zero template"):
        config.select_frame_plan(settings, membership, SCALE)
