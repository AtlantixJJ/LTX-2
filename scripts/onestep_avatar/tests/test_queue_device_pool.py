"""Explicit pools keep single-device work from taking the training devices."""

import json
from pathlib import Path

import pytest

from scripts.onestep_avatar.execution import devices, queue


def test_fixed_training_pool_waits_instead_of_spilling_to_other_free_devices() -> None:
    memory = dict.fromkeys(range(8), 4)
    memory[2] = 1024
    assert devices.select_devices(memory, training=True) == (0, 1, 3, 4)
    assert devices.select_devices(memory, training=True, pool=(0, 1, 2, 3)) is None
    memory[2] = 4
    assert devices.select_devices(memory, training=True, pool=(0, 1, 2, 3)) == (0, 1, 2, 3)
    assert devices.select_devices(memory, training=False, pool=(4, 5, 6, 7), busy=(4,)) == (5,)


@pytest.mark.parametrize("pool", [(), (0, 0), (8,), (-1,), (True,), [0]])
def test_bad_device_pool_fails(pool: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="GPU pool"):
        devices.select_devices(dict.fromkeys(range(8), 4), training=False, pool=pool)


@pytest.mark.parametrize("pool", ["", "0,0", "8", "-1", "abc"])
def test_bad_cli_pool_fails_before_job_reads(tmp_path: Path, pool: str) -> None:
    with pytest.raises(SystemExit):
        queue.main(["--jobs", str(tmp_path / "missing"), "--state", str(tmp_path / "state"),
                    "--dry-run", "--gpu-pool", pool])
    assert not list(tmp_path.iterdir())


def test_receipt_check_runs_completion_verifier_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"complete": True}))
    job = {"id": "decode", "kind": "decode", "sha256": "a" * 64,
           "completion": {"manifest": str(manifest)}}
    calls = []
    monkeypatch.setattr(queue, "verify_completion", lambda value: calls.append(value["id"]) or True)
    receipt = queue.completion_receipt(job)
    assert calls == ["decode"]
    calls.clear()
    assert queue.verify_receipt(job, receipt)
    assert calls == ["decode"]
