"""Saved-only report controls: event eligibility, original IDs and fixed examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.prune.score import native_gate_review as review
from scripts.prune.score.native_gate_accumulator import atomic_json, save_arrays


def test_event_correlations_keep_noise_groups_separate() -> None:
    scores = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.4, 0.3], [0.2, 0.1]]])
    events = [{"update": 50, "heads": [{"block": 0, "head": 0, "drop": 0.1}, {"block": 1, "head": 1, "drop": 0.4}]}]
    result = review.event_correlations(scores, events)[0]
    assert result["eligible_heads"] == 2
    np.testing.assert_allclose(result["by_sigma"], [1, -1])
    assert result["equal_sigma"] is None


def test_head_records_use_original_ids_and_distinct_masks() -> None:
    scores = np.arange(8).reshape(2, 2, 2) / 4
    reference = {"kept": [[True, False], [False, True]]}
    metrics = {
        "by_sigma": [
            {"selected": [1, 2], "within_block_selected": [0, 3]},
            {"selected": [0, 3], "within_block_selected": [1, 2]},
        ],
        "aggregate": {"selected": [0, 1], "within_block_selected": [0, 2]},
    }
    rows = review.head_records(scores, reference, metrics)
    assert [(r["block"], r["head"]) for r in rows] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert rows[1]["learned_removed"]
    assert rows[1]["native_global_removed"] == [True, False, True]
    assert rows[1]["native_within_block_removed"] == [False, True, False]
    assert rows[1]["mean_by_sigma"] == [0.25, 1.25]


@pytest.fixture
def saved_evidence(tmp_path: Path) -> Path:
    sigmas = [0.94, 0.91, 0.77, 0.48]
    atomic_json(tmp_path / "configs/frozen_study.json", {"sigmas": sigmas, "identity": "fixture"})
    atomic_json(tmp_path / "configs/examples.json", {"heads": {str(j): [0, 1, 2] for j in range(4)}})
    atomic_json(tmp_path / "configs/narrative.json", {"sections": [{"design": "Frozen design."} for _ in range(5)]})
    kept = np.ones((48, 32), dtype=bool)
    kept[0, :2] = False
    atomic_json(tmp_path / "comparison/masks.json", {"references": [{"name": "primary", "kept": kept.tolist()}]})
    heads = [
        {"block": 0, "head": i, "drop": i / 10, "reason": role}
        for i, role in enumerate(("kept", "strong", "persistent"))
    ]
    atomic_json(tmp_path / "comparison/events.json", [{"update": u, "heads": heads} for u in range(50, 601, 50)])
    scores = np.broadcast_to(np.linspace(0.2, 1.8, 1536).reshape(1, 48, 32), (4, 48, 32)).copy()
    aggregate = {
        "selected": [0, 1],
        "within_block_selected": [0, 1],
        "removed_mean": 0.2,
        "kept_mean": 1,
        "intersection": 2,
        "random_expected_intersection": 4 / 1536,
    }
    metric = {"aggregate": aggregate, "by_sigma": [aggregate] * 4}
    for step in (100, 200):
        directory = tmp_path / f"milestones/step_{step:04d}"
        atomic_json(
            tmp_path / f"monitoring/audits/step_{step:04d}.json", {"passed": True, "step": step, "study": "fixture"}
        )
        result = {
            "step": step,
            "coverage": [{"actors": 1, "views": 1}] * 4,
            "agreement": {"primary": metric},
            "convergence": {
                "mean_change": {"median": [0] * 4},
                "recent_window_distance": {"median": [0.01] * 4},
                "stable_fraction": [0] * 4,
            },
        }
        atomic_json(directory / "analysis.json", result)
        save_arrays(directory / "analysis.npz", {"mean": scores * 40 / 40}, {})
        save_arrays(directory / "cumulative.npz", {"gate_sum": scores * 40, "tokens": np.full(4, 40)}, {})
        save_arrays(
            directory / "window.npz",
            {},
            {"receipts": [{"actor": "actor", "view": "view", "sigma_index": j} for j in range(4)]},
        )
        probability = np.zeros(2048)
        probability[256] = 1
        curves = [
            {"sigma": sigmas[j], "block": 0, "head": h, "probability": probability.tolist()}
            for j in (3, 2, 1, 0)
            for h in (0, 1, 2)
        ]
        atomic_json(directory / "figures/E1b_distributions.data.json", {"examples": curves})
    return tmp_path


def test_review_and_narrative_rebuild_are_idempotent(saved_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def save(fig: object, directory: Path, name: str, data: dict) -> None:
        seen.append(name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.png").write_bytes(b"test fixture")
        (directory / f"{name}.pdf").write_bytes(b"test fixture")
        atomic_json(directory / f"{name}.data.json", data)
        review.plt.close(fig)

    monkeypatch.setattr(review, "save_figure", save)
    review.review_milestone(saved_evidence, 100)
    first = (saved_evidence / "configs/narrative.json").read_text()
    review.review_milestone(saved_evidence, 100)
    assert (saved_evidence / "configs/narrative.json").read_text() == first
    result = review.review_milestone(saved_evidence, 200)
    assert len(result["event_correlations_by_sigma"]) == 12
    assert result["displayed_events"] == [50, 300, 600]
    assert "E4b_primary_global_by_sigma" in seen
    assert "E2c_sampling_coverage" in seen
    data = review.read(saved_evidence / "milestones/step_0200/review/figures/E1b_prefix_overlays.data.json")
    assert data["prefixes"] == [100, 200]
    text = (saved_evidence / "configs/narrative.json").read_text()
    assert "Practical-stability labels require step 400" in text


def test_frozen_example_change_is_rejected(saved_evidence: Path) -> None:
    path = saved_evidence / "milestones/step_0100/figures/E1b_distributions.data.json"
    data = review.read(path)
    data["examples"][0]["head"] = 3
    atomic_json(path, data)
    with pytest.raises(ValueError, match="frozen first-milestone head IDs"):
        review.density_overlays(saved_evidence, 100, saved_evidence / "review")
    review.plt.close("all")


def test_review_reader_imports_no_model_stack() -> None:
    code = (
        "import sys; from scripts.prune.score import native_gate_review; "
        "assert not any(k == 'torch' or k.startswith(('ltx_core', 'ltx_pipelines', 'scripts.onestep_avatar')) "
        "for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_inconsistent_saved_means_are_rejected(saved_evidence: Path) -> None:
    save_arrays(saved_evidence / "milestones/step_0100/analysis.npz", {"mean": np.zeros((4, 48, 32))}, {})
    with pytest.raises(ValueError, match="audited sufficient statistics"):
        review.review_milestone(saved_evidence, 100)
