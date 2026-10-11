"""Display saved queue logs and translate owner stop signals into bounded cleanup.

Inputs: an existing child log and the CLI owner's SIGINT/SIGTERM signals.
Logic: a reader copies appended log bytes to stdout, then drains on exit.
The first stop signal raises KeyboardInterrupt; later signals wait for cleanup.
Outputs: terminal text and the received signal number. Prior handlers restore.
This module never launches, signals or owns a child; the queue supervisor does.
"""

from __future__ import annotations

import codecs
import signal
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType


@contextmanager
def stop_signals() -> Iterator[list[int]]:
    """Keep repeated stop signals from interrupting exact-worker shutdown."""
    received = []

    def stop(number: int, _frame: FrameType | None) -> None:
        if not received:
            received.append(number)
            raise KeyboardInterrupt(f"received {signal.Signals(number).name}")

    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        for number in previous:
            signal.signal(number, stop)
        yield received
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextmanager
def follow_log(path: Path, *, enabled: bool) -> Iterator[None]:
    """Read the durable log independently of supervision; retain split UTF-8 bytes."""
    if not enabled:
        yield
        return
    stopped = threading.Event()
    output = sys.stdout

    def follow() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with path.open("rb") as reader:
            while True:
                chunk = reader.read(65536)
                if chunk:
                    output.write(decoder.decode(chunk))
                    output.flush()
                    continue
                if stopped.is_set():
                    output.write(decoder.decode(b"", final=True))
                    output.flush()
                    return
                stopped.wait(0.1)

    reader = threading.Thread(target=follow, name="avatar-queue-log", daemon=True)
    reader.start()
    try:
        yield
    finally:
        stopped.set()
        reader.join(timeout=5)
