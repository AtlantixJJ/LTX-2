"""Actor exclusions use a pinned declared set, independent of other experiment folders."""

import json

import pytest

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import declared_evidence


def test_only_declared_files_supply_exclusions_and_all_hashes_are_checked(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text("Part_1/0012_01/views/view00 actor: 17")
    (tmp_path / "other.md").write_text("actor: 999")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"rows": [{"actor": "21"}], "cases": [{"actor": "22"}]}))
    listing = tmp_path / "evidence.json"
    listing.write_text(
        json.dumps(
            {
                "format": "prune_seen_evidence_v1",
                "files": [{"path": p.name, "sha256": file_sha256(p)} for p in (report, metrics)],
            }
        )
    )
    actors, pins = declared_evidence(listing)
    assert actors == ["12", "17", "21", "22"] and len(pins) == 3
    report.write_text("actor: 123")
    with pytest.raises(ValueError, match="frozen input changed"):
        declared_evidence(listing)
