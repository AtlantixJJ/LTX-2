import torch


def test_saved_only_report_missing_evidence_and_questions(tmp_path):
    import json
    import runpy

    import pytest

    from scripts.prune.data.learned_head_study import report_resources

    plan = tmp_path / "plan.md"
    plan.write_text(
        "\n".join(
            f"### R{i}. Does this comparison work?\n**Baseline:** fixed.\n**Changed variable:** one.\n"
            for i in range(1, 18)
        )
    )
    report_resources(tmp_path, plan)
    narrative_path = tmp_path / "configs/narrative.json"
    narrative = json.loads(narrative_path.read_text())
    assert len(narrative["questions"]) == 17
    namespace = runpy.run_path(str(tmp_path / "code/report.py"))
    narrative["required_files"] = [{"path": "missing_metric.json", "sha256": "0" * 64}]
    with pytest.raises(FileNotFoundError, match="missing report evidence"):
        namespace["checked_inputs"](narrative)


from scripts.prune.data.learned_head_study import draw_list, select_splits


def sources():
    return [
        {
            "actor": str(a),
            "relative_dir": f"actor{a}/view{v}",
            "n_latent_frames": 18 if v == 0 else 28,
            "split": "train" if a < 6 else "held_out",
        }
        for a in range(10)
        for v in range(4)
    ]


def test_disjoint_rounds_and_draws():
    split = select_splits(sources(), ["6"], validation_actors=2, learning_views=8, test_actors=3)
    assert not set(split["learning_actors"]) & set(split["validation_actors"])
    assert "6" not in split["test_actors"] and len(split["learning"]) == 8
    assert len({s["actor"] for s in split["learning"][:4]}) == 4
    assert draw_list(split["learning"], 42) == draw_list(split["learning"], 42)
    draws = draw_list(split["learning"], 42)
    assert len(draws) == 3000 and draws[-1]["noise_seed"] == 1422999
    assert [d["sigma_index"] for d in draws[:8]] == [0, 7, 1, 6, 2, 5, 3, 4]


def test_balanced_bucket_sigma_coverage():
    pool = [{"actor": str(i), "relative_dir": f"{i}"} for i in range(6)]
    draws = draw_list(pool, 42, buckets={str(i): i for i in range(6)})
    assert {(int(d["view"]), d["sigma_index"]) for d in draws[:48]} == {(b, i) for b in range(6) for i in range(8)}
    assert torch.tensor([d["sigma_index"] for d in draws[:8]]).unique().numel() == 8


def test_seven_sigma_draws_keep_views_and_noise_but_exclude_lowest_level():
    from scripts.prune.data.head_score_dataset import SIGMAS
    from scripts.prune.data.learned_head_study import study_sigmas

    pool = sources()
    cycle = [0, 6, 1, 5, 2, 4, 3]
    old = draw_list(pool, 42)
    new = draw_list(pool, 42, sigma_indices=cycle)
    assert [(d["view"], d["noise_seed"]) for d in old] == [(d["view"], d["noise_seed"]) for d in new]
    assert [d["sigma_index"] for d in new[:14]] == cycle * 2
    grid = study_sigmas({"config": {"sigmas": SIGMAS[:-1]}})
    assert set(grid[d["sigma_index"]] for d in new) == set(SIGMAS[:-1])
    assert SIGMAS[-1] not in grid
    assert len(new) == 3000


def test_sigma_cycle_rejects_duplicate_or_missing_indices():
    import pytest

    for cycle in [[], [0, 0], [0, 2], [0, 1.0]]:
        with pytest.raises(ValueError, match="sigma cycle"):
            draw_list(sources(), 42, sigma_indices=cycle)


def test_ten_level_preset_covers_high_dev_tail_without_changing_noise_or_validation():
    import json
    from collections import Counter
    from pathlib import Path

    from ltx_core.components.schedulers import LTX2Scheduler
    from scripts.prune.data.head_score_dataset import SIGMAS
    from scripts.prune.data.learned_head_study import monitor_sigma_indices, validate_sigma_grid

    path = Path(__file__).resolve().parents[1] / "configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json"
    config = json.loads(path.read_text())
    stock = LTX2Scheduler().execute(steps=30).tolist()
    validate_sigma_grid(config, {"original_grid": stock})
    expected_high = {s for s in stock if 0.9 <= s <= 0.95}
    assert len(expected_high) == 5 and expected_high <= set(config["sigmas"])
    assert set(config["sigmas"]) == set(SIGMAS[:-1]) | expected_high
    assert config["gate_lr"] == 0.01 and config["lambda_max"] == 0.01
    assert [config["sigmas"][i] for i in monitor_sigma_indices({"config": config})] == [SIGMAS[i] for i in (0, 4, 6)]
    old = draw_list(sources(), 42, sigma_indices=[0, 6, 1, 5, 2, 4, 3], draws_per_update=4)
    new = draw_list(sources(), 42, sigma_indices=config["sigma_indices_cycle"], draws_per_update=4)
    assert [(d["view"], d["noise_seed"]) for d in new] == [(d["view"], d["noise_seed"]) for d in old]
    assert Counter(d["sigma_index"] for d in new[:20]) == {i: 2 for i in range(10)}


def test_sigma_validation_rejects_nonstock_levels_and_invalid_monitor_indices():
    import copy

    import pytest

    from scripts.prune.data.head_score_dataset import SIGMAS
    from scripts.prune.data.learned_head_study import monitor_sigma_indices, validate_sigma_grid

    cfg = {"sigmas": SIGMAS[:-1], "sigma_indices_cycle": [0, 6, 1, 5, 2, 4, 3]}
    schedule = {"original_grid": SIGMAS}
    validate_sigma_grid(cfg, schedule)
    assert monitor_sigma_indices({"config": cfg}) == [0, 4, 6]
    for grid in ([0.945, *SIGMAS[1:-1]], [*SIGMAS[:-2], SIGMAS[0]], [*SIGMAS[:-2], 0.0]):
        with pytest.raises(ValueError, match="training sigmas"):
            validate_sigma_grid({**cfg, "sigmas": grid}, schedule)
    for indices in ([0, 0, 6], [0, 4, 7], [0, 4], [0, 4, 6.0]):
        bad = copy.deepcopy(cfg)
        bad["monitor_sigma_indices"] = indices
        with pytest.raises(ValueError, match="monitor sigma indices"):
            validate_sigma_grid(bad, schedule)
