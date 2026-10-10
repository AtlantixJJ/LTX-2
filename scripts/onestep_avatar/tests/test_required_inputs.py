"""Ordinary public interfaces reject missing inputs before model or publication work."""

from pathlib import Path

import pytest

from scripts.onestep_avatar import plot_training, previews


def test_preview_requires_its_explicit_output() -> None:
    with pytest.raises(SystemExit):
        previews.parse_args(["--preview-job", "/not-read"])


@pytest.mark.parametrize("missing", ["world_size", "samples", "schema_version"])
def test_training_plot_refuses_missing_current_evidence(missing: str, tmp_path: Path) -> None:
    config = {"world_size": 2, "mode": "causal"}
    row = {"schema_version": 2, "rank": 0, "step": 1, "samples": [{"source": "actor/view"}]}
    if missing == "world_size":
        del config[missing]
    else:
        del row[missing]
    run = plot_training.RunData("run", tmp_path, config, {0: [row]})
    with pytest.raises(ValueError, match=r"world size|schema-two"):
        plot_training.update_coverage(run)
