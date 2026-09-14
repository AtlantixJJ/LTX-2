from __future__ import annotations

import json
from pathlib import Path

from scripts.onestep_avatar import plot_training as pt


def _write_run(run_dir: Path, records: list[dict]) -> pt.RunData:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics_rank0.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return pt.load_run(run_dir)


def test_per_position_values_groups_by_chain_position_across_steps(tmp_path: Path) -> None:
    """SS7.4(a)'s per-window breakdown, read back the way window_position.png consumes it."""
    run = _write_run(
        tmp_path / "run",
        [
            {
                "step": 1, "rank": 0, "lr": 1e-4, "grad_norm": 0.1, "elapsed_s": 1.0, "source": "s",
                "loss": 0.5, "mse": 0.5, "anchor": 0.0,
                "per_window": [
                    {"chain_position": 0, "window_index": 3, "mse": 0.4, "anchor": 0.0},
                    {"chain_position": 1, "window_index": 4, "mse": 0.6, "anchor": 0.0},
                ],
            },
            {
                "step": 2, "rank": 0, "lr": 1e-4, "grad_norm": 0.1, "elapsed_s": 2.0, "source": "s",
                "loss": 0.3, "mse": 0.3, "anchor": 0.0,
                "per_window": [
                    {"chain_position": 0, "window_index": 3, "mse": 0.2, "anchor": 0.0},
                    {"chain_position": 1, "window_index": 4, "mse": 0.35, "anchor": 0.0},
                ],
            },
        ],
    )

    per_position = pt._per_position_values(run, "mse")

    assert set(per_position) == {0, 1}
    assert per_position[0] == {1: [0.4], 2: [0.2]}
    assert per_position[1] == {1: [0.6], 2: [0.35]}


def test_plot_window_position_returns_none_for_a_run_predating_per_window_logging(tmp_path: Path) -> None:
    """A run logged before SS7.4(a) has no `per_window` key -- must skip cleanly, not crash,
    so passing an old and a new run to `--run` together still works."""
    run = _write_run(
        tmp_path / "old_run",
        [{"step": 1, "rank": 0, "lr": 1e-4, "grad_norm": 0.1, "elapsed_s": 1.0, "source": "s",
          "loss": 0.5, "mse": 0.5, "anchor": 0.0}],
    )

    assert pt.plot_window_position([run], smooth=1, output=tmp_path / "figs") is None


def test_plot_window_position_writes_a_figure_when_data_exists(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "run",
        [
            {
                "step": s, "rank": 0, "lr": 1e-4, "grad_norm": 0.1, "elapsed_s": float(s), "source": "s",
                "loss": 0.5, "mse": 0.5, "anchor": 0.0,
                "per_window": [{"chain_position": 0, "window_index": s, "mse": 0.5 / s, "anchor": 0.0}],
            }
            for s in (1, 2, 3)
        ],
    )
    output = tmp_path / "figs"
    output.mkdir()

    path = pt.plot_window_position([run], smooth=1, output=output)

    assert path == output / "window_position.png"
    assert path.is_file()
