"""Original-ID joins, exact worker prefixes and comparison export provenance."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.prune.score import native_gate_records as records
from scripts.prune.score.native_gate_accumulator import atomic_json, digest


def heads() -> list[dict]:
    return [
        {
            "block": i // 2,
            "head": i % 2,
            "learned_removed": i in (1, 2),
            "native_global_removed": [i in (0, 1)] * 5,
            "native_within_block_removed": [i in (0, 2)] * 5,
        }
        for i in range(4)
    ]


def test_disagreement_lists_preserve_ids_under_row_permutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records, "HEAD_SHAPE", (2, 2))
    groups = [0.94, 0.91, 0.77, 0.48, "equal_sigma"]
    kept = [[True, False], [False, True]]
    first = records.disagreements(heads(), kept, groups)
    reordered = records.disagreements(heads()[::-1], kept, groups)
    assert first == reordered
    assert first["global"][0]["both"] == [[0, 1]]
    assert first["global"][0]["learned_only"] == [[1, 0]]
    assert first["global"][0]["native_only"] == [[0, 0]]
    assert first["within_block"][0]["both"] == [[1, 0]]


def test_rejects_duplicate_heads_and_wrong_block_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records, "HEAD_SHAPE", (2, 2))
    rows = heads()
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="unique complete original head IDs"):
        records.disagreements(rows, [[True, False], [False, True]], list(range(5)))
    rows = heads()
    rows[1]["native_within_block_removed"] = [True] * 5
    rows[2]["native_within_block_removed"] = [False] * 5
    with pytest.raises(ValueError, match="saved block counts"):
        records.disagreements(rows, [[True, False], [False, True]], list(range(5)))


@pytest.fixture
def reviewed_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(records, "HEAD_SHAPE", (2, 2))
    mask = {"references": [{"name": "primary", "kept": [[True, False], [False, True]]}]}
    atomic_json(tmp_path / "comparison/masks.json", mask)
    cycle = [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]
    selected = [i for i in range(400) if cycle[i % 10] in (1, 4, 7, 9)]
    study = {
        "steps": 100,
        "gpus": [4, 5, 6],
        "sigmas": [0.94, 0.91, 0.77, 0.48],
        "selected_draw_indices": selected,
        "files": {"comparison/masks.json": records.file_hash(tmp_path / "comparison/masks.json")},
    }
    study["identity"] = digest(study)
    atomic_json(tmp_path / "configs/frozen_study.json", study)
    directory = tmp_path / "milestones/step_0100"
    head_path = directory / "review/head_records.json"
    atomic_json(head_path, {"score_group_order": [*study["sigmas"], "equal_sigma"], "references": {"primary": heads()}})
    review_path = directory / "review/review.json"
    atomic_json(review_path, {"event_correlations_by_sigma": [{"update": 50, "by_sigma": [0.1] * 4}]})
    atomic_json(
        directory / "review/complete.json",
        {
            "study": study["identity"],
            "step": 100,
            "files": {p.relative_to(tmp_path).as_posix(): records.file_hash(p) for p in [head_path, review_path]},
        },
    )
    receipts = [{"draw_index": i, "sigma_index": [1, 4, 7, 9].index(cycle[i % 10])} for i in selected]
    for r in receipts:
        atomic_json(tmp_path / f"draws/{r['draw_index']:05d}.json", r)
    atomic_json(
        directory / "complete.json",
        {"study": study["identity"], "step": 100, "per_sigma": [40] * 4, "receipt_chain_sha256": digest(receipts)},
    )
    workers = [{"draw_index": i, "gpu": study["gpus"][i % 3], "shard_sha256": "fixture_shard"} for i in selected]
    atomic_json(
        tmp_path / "monitoring/audits/step_0100.json",
        {"passed": True, "study": study["identity"], "step": 100, "window_worker_provenance": workers},
    )
    return tmp_path


def test_exported_workers_match_receipt_chain_and_real_source_indices(reviewed_prefix: Path) -> None:
    result = records.export_records(reviewed_prefix, 100)
    assert result["indexed_steps"] == [100]
    workers = records.read(reviewed_prefix / "comparison/step_0100/draw_workers.json")
    assert workers["observed_draws"] == 160
    assert workers["per_sigma"] == [40] * 4
    assert workers["minimum_source_draw_id"] == 1
    assert workers["maximum_source_draw_id"] == 398
    for row in workers["workers"]:
        assert records.file_hash(reviewed_prefix / row["source_receipt"]) == row["source_receipt_sha256"]


def test_missing_worker_draw_does_not_publish_comparison_index(reviewed_prefix: Path) -> None:
    path = reviewed_prefix / "monitoring/audits/step_0100.json"
    audit = records.read(path)
    audit["window_worker_provenance"].pop()
    atomic_json(path, audit)
    with pytest.raises(ValueError, match="exact complete retained source prefix"):
        records.export_records(reviewed_prefix, 100)
    assert not (reviewed_prefix / "comparison/index.json").exists()
