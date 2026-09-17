"""CPU-only, no data/GPU needed. Run with: python -m pytest scripts/onestep_avatar/tests"""

from __future__ import annotations

import pytest

from scripts.onestep_avatar import dataset


def test_atomic_write_replaces_the_destination_and_returns_write_tos_result(tmp_path) -> None:  # noqa: ANN001
    def write_to(temp):  # noqa: ANN001, ANN202
        temp.write_text("hi")
        return 42

    destination = tmp_path / "out.json"
    result = dataset.atomic_write(destination, write_to)
    assert destination.read_text() == "hi"
    assert result == 42


def test_atomic_write_uses_a_hidden_temp_name_with_the_pid_and_real_suffix(tmp_path) -> None:  # noqa: ANN001
    import os

    seen: list = []

    def write_to(temp):  # noqa: ANN001, ANN202
        seen.append(temp)
        temp.write_text("x")

    dataset.atomic_write(tmp_path / "render.mp4", write_to)
    (temp,) = seen
    assert temp.name == f".render.tmp.{os.getpid()}.mp4"


def test_atomic_write_leaves_no_temp_file_and_no_destination_on_failure(tmp_path) -> None:  # noqa: ANN001
    destination = tmp_path / "out.json"

    def write_to(temp):  # noqa: ANN001, ANN202
        temp.write_text("partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        dataset.atomic_write(destination, write_to)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_creates_missing_parent_directories(tmp_path) -> None:  # noqa: ANN001
    destination = tmp_path / "nested" / "dir" / "out.json"
    dataset.atomic_write(destination, lambda temp: temp.write_text("hi"))
    assert destination.read_text() == "hi"
