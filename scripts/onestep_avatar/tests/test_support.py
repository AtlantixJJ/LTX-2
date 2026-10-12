"""Support preserves mutation isolation and child row identity."""

from pathlib import Path

import pytest

from scripts.onestep_avatar.tests.support.children import normal_environment, run_dry_run_rows
from scripts.onestep_avatar.tests.support.tree import restore, snapshot


def test_restore_removes_new_files_and_restores_links_and_bytes(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "original").write_bytes(b"original")
    (tmp_path / "link").symlink_to("original")
    saved = snapshot(tmp_path)
    (tmp_path / "original").write_bytes(b"changed")
    (tmp_path / "link").unlink()
    (tmp_path / "link").write_bytes(b"replacement")
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra/file").write_bytes(b"created")
    restore(tmp_path, saved)
    assert snapshot(tmp_path) == saved


@pytest.mark.subprocess
def test_child_rows_keep_success_failure_and_output_separate() -> None:
    rows = run_dry_run_rows(
        [
            {"id": "ok", "argv": ["probe"], "source": "print('usage: example')"},
            {"id": "bad", "argv": ["probe"], "source": "raise SystemExit(2)"},
            {"id": "last", "argv": ["probe"], "source": "print('last')"},
        ],
        normal_environment(),
    )
    assert [row["returncode"] for row in rows] == [0, 2, 0]
    assert [row["id"] for row in rows] == ["ok", "bad", "last"]
    assert rows[0]["stdout"] == "usage: example\n"
    assert rows[2]["stdout"] == "last\n"
