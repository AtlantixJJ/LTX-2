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
