"""Check rank charts and unique-case averages without contacting W&B."""

import copy

import pytest

import wandb
from scripts.prune.learn.wandb_logger import published_updates, rank_series, training_metrics, validation_metrics


def receipt(update=1):
    return {
        "update": update,
        "phase": "soft",
        "draws": [{"official": i, "penalty": 0.4} for i in (1, 3, 5, 7)],
        "sparsity_lambda": 0.0109,
        "seconds": 3,
        "legal_mask_changes": 0,
        "adapter_gradient_norm": 0.2,
        "logit_gradient_norm": 0.3,
        "velocity_logit_gradient_norm": 0.1,
        "sparsity_logit_gradient_norm": 0.2,
        "gates": {"mean": 0.9, "le_0_1": 0, "lt_0_5": 0, "ge_0_9": 0.5, "quantiles": [0.8, 0.85, 0.9, 0.95, 0.98]},
        "rank_metrics": [{"rank": r, "sigma": 0.9 - r * 0.1, "seconds": r + 2} for r in range(4)],
        "devices": [
            {"free": (r + 3) * 2**30, "peak_allocated": 30 * 2**30, "peak_reserved": 40 * 2**30} for r in range(4)
        ],
    }


def test_total_loss_is_single_mean_and_gate_lr_has_no_warmup():
    cfg = {"global_unique_clips": 4, "adapter_lr": 0.0001, "gate_lr": 0.1, "warmup_updates": 50}
    values = training_metrics(receipt(), cfg)
    assert values["train/loss"] == pytest.approx(4.4)
    assert values["train/velocity_mse"] == 4
    assert values["train/sparsity_loss"] == pytest.approx(0.4)
    assert values["train/lr_gates"] == 0.1 and values["train/update_s"] == 5
    assert not any("learned_joint" in key or "rank_" in key for key in values)
    regional = receipt()
    for draw in regional["draws"]:
        draw["total_loss"] = draw["official"] + draw["penalty"] + 2
    assert training_metrics(regional, cfg)["train/loss"] == pytest.approx(6.4)
    with pytest.raises(ValueError, match="draw coverage"):
        training_metrics({**receipt(), "draws": receipt()["draws"][:2]}, cfg)


def test_one_sigma_chart_has_exactly_four_rank_lines_and_all_updates():
    series = rank_series([receipt(1), receipt(2)], 4)
    assert "train/loss" not in series and not any("validation" in key for key in series)
    plot = wandb.plot.line_series(**series["train/sigma"], xname="optimizer update")
    assert plot.table.columns == ["step", "lineKey", "lineVal"]
    assert {row[1] for row in plot.table.data} == {"rank 0", "rank 1", "rank 2", "rank 3"}
    assert len(plot.table.data) == 8
    assert series["memory/free_GiB"]["ys"] == [[3, 3], [4, 4], [5, 5], [6, 6]]
    bad = receipt()
    bad["rank_metrics"] = bad["rank_metrics"][:3]
    with pytest.raises(ValueError, match="per-rank"):
        rank_series([bad], 4)


def test_validation_averages_cases_across_ranks_without_four_lines():
    rows = []
    for sigma in (0.9, 0.7, 0.4):
        for rank in range(4):
            rows.append(
                {
                    "view": str(rank),
                    "rank": rank,
                    "sigma": sigma,
                    "states": {state: {"metrics": {"capture_mse": rank + 1}} for state in ("trained", "hardened")},
                }
            )
    # A duplicate-rank replay contributes no second copy of the case.
    rows += [copy.deepcopy(rows[0])]
    values = validation_metrics({"update": 100, "rows": rows})
    assert values["validation/trained_sigma_2_gt_latent_mse"] == 2.5
    assert values["validation/trained_gt_latent_mse"] == 2.5
    assert values["validation/cases"] == 12
    assert not any("rank" in key or "learned_joint" in key for key in values)
    rows[-1]["states"]["trained"]["metrics"]["capture_mse"] = 999
    with pytest.raises(ValueError, match="conflicting"):
        validation_metrics({"update": 100, "rows": rows})


def test_published_receipts_ignore_atomic_temporary_files(tmp_path):
    for name in ("0001.json", "0002.json", "0010.json", ".publish-broken.json", "metadata.json"):
        (tmp_path / name).write_text("unfinished")
    assert [p.name for p in published_updates(tmp_path)] == ["0001.json", "0002.json", "0010.json"]
