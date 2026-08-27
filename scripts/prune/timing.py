"""``StageTimer`` + FLOP counting shared by every scripts/prune/* benchmark.

``StageTimer`` is a copy (not an import -- vae_refine_sliding_window.py is a run
script, not a library) of the pattern already proven there: reset peak memory
stats, synchronize, ``perf_counter``. See plans/2026-08-26-refiner-head-ffn-pruning.md §5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


@dataclass
class StageTimer:
    label: str
    device: torch.device
    t0: float = 0.0
    elapsed_s: float = 0.0
    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0

    def __enter__(self) -> "StageTimer":
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        torch.cuda.synchronize(self.device)
        self.elapsed_s = time.perf_counter() - self.t0
        self.peak_alloc_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
        self.peak_reserved_gb = torch.cuda.max_memory_reserved(self.device) / 1e9


def count_flops(fn: Callable[[], object]) -> int:
    """Run ``fn()`` once under ``FlopCounterMode`` and return total FLOPs.
    Call on an already-warmed model/inputs; the counted call still executes for
    real (not a dry trace), so pair with a separate timed loop for ms/step --
    this is FLOPs only, not wall-clock.
    """
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()
