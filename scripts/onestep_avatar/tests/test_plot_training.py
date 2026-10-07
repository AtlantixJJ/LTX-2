from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.onestep_avatar import plot_training as pt


def _write_run(run_dir: Path, records: list[dict]) -> pt.RunData:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics_rank0.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return pt.load_run(run_dir)


def test_per_position_values_groups_by_chain_position_across_steps(tmp_path: Path) -> None:
    """Historical causal breakdown, read back the way block_position.png consumes it."""
    run = _write_run(
        tmp_path / "run",
        [
            {
                "step": 1,
                "rank": 0,
                "lr": 1e-4,
                "grad_norm": 0.1,
                "elapsed_s": 1.0,
                "source": "s",
                "loss": 0.5,
                "mse": 0.5,
                "anchor": 0.0,
                "per_block": [
                    {"chain_position": 0, "block_index": 3, "mse": 0.4, "anchor": 0.0},
                    {"chain_position": 1, "block_index": 4, "mse": 0.6, "anchor": 0.0},
                ],
            },
            {
                "step": 2,
                "rank": 0,
                "lr": 1e-4,
                "grad_norm": 0.1,
                "elapsed_s": 2.0,
                "source": "s",
                "loss": 0.3,
                "mse": 0.3,
                "anchor": 0.0,
                "per_block": [
                    {"chain_position": 0, "block_index": 3, "mse": 0.2, "anchor": 0.0},
                    {"chain_position": 1, "block_index": 4, "mse": 0.35, "anchor": 0.0},
                ],
            },
        ],
    )

    per_position = pt._per_position_values(run, "mse")

    assert set(per_position) == {0, 1}
    assert per_position[0] == {1: [0.4], 2: [0.2]}
    assert per_position[1] == {1: [0.6], 2: [0.35]}


def test_plot_block_position_returns_none_for_a_run_predating_per_block_logging(tmp_path: Path) -> None:
    """A run logged before SS7.4(a) has no `per_block` key -- must skip cleanly, not crash,
    so passing an old and a new run to `--run` together still works."""
    run = _write_run(
        tmp_path / "old_run",
        [
            {
                "step": 1,
                "rank": 0,
                "lr": 1e-4,
                "grad_norm": 0.1,
                "elapsed_s": 1.0,
                "source": "s",
                "loss": 0.5,
                "mse": 0.5,
                "anchor": 0.0,
            }
        ],
    )

    assert pt.plot_block_position([run], smooth=1, output=tmp_path / "figs") is None


def test_plot_block_position_writes_a_figure_when_data_exists(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "run",
        [
            {
                "step": s,
                "rank": 0,
                "lr": 1e-4,
                "grad_norm": 0.1,
                "elapsed_s": float(s),
                "source": "s",
                "loss": 0.5,
                "mse": 0.5,
                "anchor": 0.0,
                "per_block": [{"chain_position": 0, "block_index": s, "mse": 0.5 / s, "anchor": 0.0}],
            }
            for s in (1, 2, 3)
        ],
    )
    output = tmp_path / "figs"
    output.mkdir()

    path = pt.plot_block_position([run], smooth=1, output=output)

    assert path == output / "block_position.png"
    assert path.is_file()


def _distributed_run(tmp_path: Path) -> pt.RunData:
    run_dir = tmp_path / "distributed"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"world_size": 2, "mode": "causal"}))
    for rank, losses, elapsed in ((0, [2, 100, 4], [10, 15, 18]), (1, [6, 8], [12, 21])):
        steps = [1, 2, 3] if rank == 0 else [1, 3]
        records = [
            {
                "schema_version": 2,
                "mode": "causal",
                "rank": rank,
                "step": step,
                "loss": loss,
                "mse": loss,
                "elapsed_s": wall,
                "lr": 0.01,
                "grad_norm": rank + 1,
                "sigma0": 0.725,
                "samples": [{"source": "actor/view", "per_block": [{"block_index": 7, "mse": loss}]}],
            }
            for step, loss, wall in zip(steps, losses, elapsed, strict=True)
        ]
        (run_dir / f"metrics_rank{rank}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return pt.load_run(run_dir)


def test_complete_batch_means_exclude_missing_rank(tmp_path: Path) -> None:
    run = _distributed_run(tmp_path)
    steps, means = pt.step_mean(run, "loss")
    assert steps.tolist() == [1, 3]
    assert means.tolist() == [4, 6]
    coverage = pt.update_coverage(run)
    assert coverage[2]["missing"] == [1]
    assert not coverage[2]["complete"]
    _, values, disagreement = pt.scalar_series(run, "grad_norm")
    assert values.tolist() == [1, 1]
    assert disagreement == 1
    assert pt._per_position_values(run, "mse") == {0: {1: [2, 6], 3: [4, 8]}}
    summary = pt._run_summary(run)
    assert summary["steps_completed"] == 3
    assert summary["complete_updates"] == [1, 3]
    assert summary["loss_min"] == 4
    assert summary["loss_last_10pct_mean"] == 6
    assert summary["wall_clock_minutes"] == 0.35
    assert summary["distinct_sources_seen"] == 1
    assert summary["sigma_counts_per_rank_update"] == {"0.725": 4}


def test_unequal_accumulation_is_not_a_batch_mean(tmp_path: Path) -> None:
    run = _distributed_run(tmp_path)
    run.by_rank[1][0]["samples"] *= 2
    assert not pt.update_coverage(run)[1]["complete"]
    assert pt.step_mean(run, "loss")[0].tolist() == [3]


def test_duplicate_update_is_rejected(tmp_path: Path) -> None:
    row = {"rank": 0, "step": 1}
    with pytest.raises(ValueError, match="duplicate update"):
        _write_run(tmp_path / "run", [row, row])


def test_bidirectional_plots_do_not_invent_block_positions(tmp_path: Path) -> None:
    run = _distributed_run(tmp_path)
    run.config["mode"] = "bidirectional"
    assert pt._per_position_values(run, "mse") == {}


def test_smoothing_padding_does_not_change_raw_summary(tmp_path: Path) -> None:
    assert pt._smooth(np.array([2.0, 6.0, 10.0]), 2).tolist() == [4, 4, 8]
    run = _distributed_run(tmp_path)
    summary = pt._run_summary(run)
    assert summary["raw_series"]["loss"] == {"steps": [1, 3], "values": [4, 6]}
