"""Keep one original overall deadline across retries, restarts and recovery.

Inputs: prepared budget, current journal row and observation time. Validate that
the first start plus overall_seconds equals the saved deadline and that every
attempt retains both values. Set the clock only at the first launch; GPU waiting
before that launch does not count. Return the same clock and remaining seconds.
An expired clock raises OverallDeadlineError; malformed or reset clocks fail
validation. Jobs without an overall budget use their existing per-phase bounds.
QueueChildError identifies reaped workload failures eligible for retry review.
"""

from __future__ import annotations

import math
import time


class QueueChildError(ValueError):
    """A reaped nonzero child; validation and ownership errors use other failures."""


class OverallDeadlineError(QueueChildError):
    """The original job clock expired; retries cannot replace its deadline."""


def validate_execution_clock(job: dict, row: dict) -> None:
    """Require one original clock across every attempt, including restart reads."""
    overall = job.get("effective_budget", {}).get("overall_seconds")
    start, deadline = row.get("first_attempt_started_at"), row.get("overall_deadline_at")
    if overall is None:
        if start is not None or deadline is not None:
            raise ValueError("queue overall clock requires an explicit overall budget")
        return
    if start is None and deadline is None and not row.get("attempts"):
        return  # A job waiting for devices has not begun an attempt.
    if (
        type(start) not in (int, float)
        or not math.isfinite(start)
        or type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or deadline != start + overall
    ):
        raise ValueError("queue original overall deadline is missing or changed")
    attempts = row.get("attempts", [])
    if attempts and attempts[0].get("started_at") != start:
        raise ValueError("queue overall clock differs from its first attempt")
    for attempt in attempts:
        if attempt.get("first_attempt_started_at") != start or attempt.get("overall_deadline_at") != deadline:
            raise ValueError("queue attempt reset its original overall deadline")


def remaining_deadline(row: dict, *, now: float | None = None) -> float | None:
    """Bound registration, grants and supervision by the same saved clock."""
    deadline = row.get("overall_deadline_at")
    if deadline is None:
        return None
    remaining = deadline - (time.time() if now is None else now)
    if remaining <= 0:
        raise OverallDeadlineError("overall_deadline: original job deadline expired")
    return remaining


def begin_execution_clock(job: dict, row: dict, *, now: float) -> dict:
    """Set the clock once, in the first-attempt journal transaction."""
    validate_execution_clock(job, row)
    overall = job.get("effective_budget", {}).get("overall_seconds")
    if overall is None:
        return {}
    if row.get("first_attempt_started_at") is None:
        row.update(first_attempt_started_at=now, overall_deadline_at=now + overall)
    remaining_deadline(row, now=now)
    return {name: row[name] for name in ("first_attempt_started_at", "overall_deadline_at")}
