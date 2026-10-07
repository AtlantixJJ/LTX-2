"""Report queued startup contention without owning retry or model execution.

Inputs: optional paired queue token/job hash and distributed rank environment.
Logic: bind plain JSON events to this attempt; mark the update boundary before
training starts; report CUDA OOM or typed port contention before that boundary and propagate
errors.
Outputs: stdout events consumed by the queue after the complete child session
terminates. Nonqueued runs emit nothing. Tracebacks are not retry authority.
"""
from __future__ import annotations

import errno
import json
import os
import re
from types import TracebackType

import torch

from scripts.onestep_avatar.queue_protocol import JOB_ENV, PREFIX, TOKEN_ENV


class StartupEvents:
    """Separate startup from all training computation, including the first update."""

    def __init__(self) -> None:
        self.token = os.environ.get(TOKEN_ENV)
        self.job = os.environ.get(JOB_ENV)
        self.rank = int(os.environ.get('RANK', '0'))
        self.updating = False
        if (self.token is None) != (self.job is None):
            raise ValueError('queue startup identity requires both token and job hash')
        if self.token is not None and (
            re.fullmatch(r'[0-9a-f]{32}', self.token) is None
            or re.fullmatch(r'[0-9a-f]{64}', self.job) is None
            or not 0 <= self.rank < 4
        ):
            raise ValueError('invalid queue startup identity or global rank')

    def _emit(self, event: str, reason: str | None = None) -> None:
        if self.token is not None:
            print(PREFIX + json.dumps({  # noqa: T201 -- machine-readable child evidence
                'schema_version': 1, 'event': event, 'token': self.token,
                'job_sha256': self.job, 'rank': self.rank,
                'reason': reason,
            }, sort_keys=True), flush=True)

    def begin_updates(self) -> None:
        """Set the boundary before a forward/backward or optimizer update starts."""
        self.updating = True
        self._emit('updates_begin')

    def __enter__(self) -> StartupEvents:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if isinstance(error, torch.cuda.OutOfMemoryError) and not self.updating:
            self._emit('startup_contended', 'cuda_oom')
        elif isinstance(error, OSError) and error.errno == errno.EADDRINUSE and not self.updating:
            self._emit('startup_contended', 'port_in_use')
        elif error is not None:
            self._emit('startup_failed')
