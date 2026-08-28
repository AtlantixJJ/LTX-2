from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.prune import corpus, refine_task


def test_fps_is_read_per_clip_and_not_uniform():
    assert {corpus.fps(source) for source in corpus.sources()} == {24.0, 30.0}


def test_pick_clip_skips_clip_shorter_than_window(model):
    geometry = refine_task.deployed_geometry(model.scale_factors)
    picked = corpus.pick_clip(geometry, windows=6)
    assert corpus.frame_count(picked) >= geometry.window_frames + 5 * geometry.stride_frames


def test_pick_clip_by_name_wins(model):
    geometry = refine_task.deployed_geometry(model.scale_factors)
    picked = corpus.pick_clip(geometry, 1, name="2K2K_00052_0__man_dance_2_crop")
    assert picked.parent.name == "2K2K_00052_0__man_dance_2_crop"


def test_pick_clip_prefers_held_out_split(model):
    geometry = refine_task.deployed_geometry(model.scale_factors)
    picked = corpus.pick_clip(geometry, 1, key="2.5", prefer="held_out")
    assert corpus.split("2.5")[picked.parent.name] == "held_out"


def test_pick_one_per_subject_returns_distinct_subjects():
    picked = corpus.pick_one_per_subject(3, min_frames=25)
    assert len({corpus.subject_of(source.parent.name) for source in picked}) == 3


def test_impossible_request_names_corpus_and_need(model):
    geometry = refine_task.deployed_geometry(model.scale_factors)
    with pytest.raises(SystemExit, match="frames needed for 999 window"):
        corpus.pick_clip(geometry, windows=999)


def test_no_module_builds_latent_tools_at_a_hardcoded_frame_rate():
    """fps is RoPE (VideoLatentTools: positions[:,0] /= fps), never a literal.

    Two defaults are deliberately allowed and are asserted by name, so adding a
    THIRD is a test failure rather than a silent regression:
      metrics.t3_video(fps=24.0)  -- an ffmpeg display rate, not a model input
      bench_refiner --fps 24.0    -- a synthetic benchmark that never opens a clip
    """
    offenders = set()
    for f in Path("scripts/prune").glob("*.py"):
        for line in f.read_text().splitlines():
            if "fps" in line and re.search(r"=\s*24(\.0)?\b", line):
                offenders.add(f.name)
                break
    assert offenders == {"metrics.py", "bench_refiner.py"}


def test_the_allowed_defaults_are_not_on_a_model_input_path():
    src = Path("scripts/prune/metrics.py").read_text()
    assert "VideoLatentTools" not in src  # metrics never builds RoPE positions
