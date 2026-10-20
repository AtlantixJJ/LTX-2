"""Every readiness, deadline or predicate failure publishes an attributable receipt."""

import json
import time

import pytest

from scripts.prune.core import waits


@pytest.mark.parametrize("status", ["ready", "timeout", "failed"])
def test_wait_receipt_records_termination(tmp_path, status):
    def ready():
        if status == "failed":
            raise FileNotFoundError("receipt missing")
        return status == "ready"

    if status == "ready":
        waits.until(ready, time.monotonic() + 1, "fixture", receipts=tmp_path)
    else:
        with pytest.raises(TimeoutError if status == "timeout" else FileNotFoundError):
            waits.until(ready, time.monotonic() - 1, "fixture", receipts=tmp_path)
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert receipt["status"] == status and receipt["checks"] == 1
    assert receipt["condition"] == "fixture" and receipt["waited_seconds"] >= 0
