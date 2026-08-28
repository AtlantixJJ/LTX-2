from __future__ import annotations

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
