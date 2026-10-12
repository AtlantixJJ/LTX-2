"""Profiles cover the exact loaded owners in isolated clean forks."""

import json
import os
from pathlib import Path

import pytest

from scripts.onestep_avatar.tests.support.children import blocked_environment, fork_profiles
from scripts.onestep_avatar.tests.support.profile_entry import SOURCE

pytestmark = [pytest.mark.boundary, pytest.mark.subprocess]


def test_profiles_cover_loaded_entry_owners(tmp_path: object, subtests: object) -> None:
    cases = [
        ("training", "causal"),
        ("training", "bidirectional"),
        ("evaluation", "causal"),
        ("evaluation", "bidirectional"),
        ("inference", "causal"),
        ("inference", "bidirectional"),
        ("decoding", None),
        ("preparation", None),
        ("preparation", "causal"),
        ("preparation", "bidirectional"),
    ]
    rows = fork_profiles(
        [
            {
                "id": profile + "_" + (mode or "none"),
                "source": SOURCE,
                "argv": [
                    "profile",
                    profile,
                    str(tmp_path / (profile + "_" + (mode or "none") + ".json")),
                    str(tmp_path / "output"),
                    mode or "none",
                ],
            }
            for profile, mode in cases
        ],
        blocked_environment(),
    )
    for row in rows:
        with subtests.test(profile=row["id"]):
            assert row["returncode"] == 0, row["stdout"]
            result = json.loads(row["stdout"].splitlines()[-1])
            assert result["loaded"]
            assert set(result["loaded"]).issubset(result["declared"]), sorted(
                set(result["loaded"]) - set(result["declared"])
            )
            assert not (tmp_path / "output").exists()
            if evidence_directory := os.environ.get("ONESTEP_AVATAR_PROFILE_OBSERVATIONS"):
                evidence = Path(evidence_directory)
                evidence.mkdir(parents=True, exist_ok=True)
                (evidence / (row["id"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
