"""Bounded polling with one immutable receipt for every completed or failed wait.

The caller supplies a monotonic deadline and a readiness predicate. The receipt
records the requested condition, timeout, checks, elapsed time, process and exit
status. Unique names preserve prior waits when an evaluation CLI is rerun.
This helper starts no workload and never stops another process.
"""

import os
import time
from pathlib import Path

from scripts.prune.core import records


def until(ready, deadline, what, *, receipts, poll_seconds=20):
    started, wall = time.monotonic(), time.time_ns()
    timeout = max(0.0, deadline - started)
    checks, status, error = 0, "ready", None
    try:
        while True:
            checks += 1
            if ready():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"waited too long for {what}")
            time.sleep(min(poll_seconds, remaining))
    except BaseException as exc:
        status, error = "timeout" if isinstance(exc, TimeoutError) else "failed", repr(exc)
        raise
    finally:
        records.publish_json(
            Path(receipts) / f"wait_{wall}_{os.getpid()}.json",
            dict(
                condition=what,
                pid=os.getpid(),
                started_unix_ns=wall,
                timeout_seconds=timeout,
                waited_seconds=time.monotonic() - started,
                checks=checks,
                status=status,
                error=error,
            ),
        )
