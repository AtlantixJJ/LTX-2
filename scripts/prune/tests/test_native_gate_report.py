"""Report publication guards and ineligible-stability presentation controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prune.score import native_gate_analysis as analysis
from scripts.prune.score.native_gate_accumulator import atomic_json


def test_report_preview_keeps_canonical_report_and_status(tmp_path: Path) -> None:
    atomic_json(tmp_path / "configs/frozen_study.json", {"steps": 1300, "sigmas": [0.94, 0.91, 0.77, 0.48]})
    atomic_json(tmp_path / "configs/narrative.json", {"sections": [{"design": "Frozen design."} for _ in range(5)]})
    atomic_json(tmp_path / "comparison/masks.json", {})
    (tmp_path / "provenance").mkdir()
    (tmp_path / "provenance/plan.md").write_text("Frozen plan.")
    (tmp_path / "REPORT.md").write_text("Current published report.")
    target = tmp_path / "preview.md"
    analysis.build_report(tmp_path, destination=target)
    assert (tmp_path / "REPORT.md").read_text() == "Current published report."
    text = target.read_text()
    assert "Skipped scheduled records: **0**" in text
    assert "**ROC AUC:**" in text
    assert "Verification and source appendix" in text
    assert target.with_suffix(".validation.json").exists()
    assert not (tmp_path / "monitoring/report_validation.json").exists()


def test_real_report_requires_complete_figure_inventory(tmp_path: Path) -> None:
    study = {"format": "native_gate_training_stream_v1", "identity": "fixture"}
    directory = tmp_path / "milestones/step_0100"
    atomic_json(directory / "complete.json", {"study": "fixture", "step": 100, "draws": 160, "per_sigma": [40] * 4})
    atomic_json(tmp_path / "comparison/masks.json", {"references": []})
    atomic_json(tmp_path / "configs/examples.json", {"brief_activation": []})
    with pytest.raises(ValueError, match="required scientific figure missing"):
        analysis.validate_report_inputs(tmp_path, study, 100)
    for name in [
        "E1a_mean_gates",
        "E1b_distributions",
        "E3a_noise_response",
        "E3b_rank_agreement",
        "coverage",
        "E4d_event_drops",
    ]:
        for suffix in (".png", ".pdf", ".data.json"):
            path = directory / f"figures/{name}{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_text("test fixture")
    analysis.validate_report_inputs(tmp_path, study, 100)
    (directory / "figures/E1b_distributions.pdf").unlink()
    with pytest.raises(ValueError, match=r"E1b_distributions\.pdf"):
        analysis.validate_report_inputs(tmp_path, study, 100)
    atomic_json(directory / "complete.json", {"study": "fixture", "step": 100, "draws": 159, "per_sigma": [40] * 4})
    with pytest.raises(ValueError, match="complete balanced source prefix"):
        analysis.validate_report_inputs(tmp_path, study, 100)


def test_stability_figure_omits_ineligible_comparisons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for step in (100, 200, 300, 400, 500):
        atomic_json(
            tmp_path / f"milestones/step_{step:04d}/analysis.json",
            {"step": step, "convergence": {"stable_fraction": [0.25, 0.5, 0.75, 1]}},
        )
    plotted = []

    def capture(fig: object, _directory: Path, _name: str, data: dict) -> None:
        plotted.append(data)
        analysis.plt.close(fig)

    monkeypatch.setattr(analysis, "save_figure", capture)
    analysis.eligible_stability_figure(tmp_path, 500, [0.94, 0.91, 0.77, 0.48])
    assert [h["step"] for h in plotted[0]["history"]] == [400, 500]
    assert plotted[0]["first_eligible_step"] == 400
