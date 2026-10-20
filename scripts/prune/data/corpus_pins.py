"""Pinned DNARendering membership and the official LTX-2.5 dev schedule.

`CSV_SHA` and `JSON_SHA` pin the membership files; `SEMANTIC_SHA` pins their
parsed content. `official_schedule` rebuilds the stock 30-step dev grid and
fails when the checkpoint defaults or the eight selected tail levels change.
Learned-gate preparation reads these pins before it freezes a study.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_pipelines.utils.constants import detect_params
from scripts.prune.core.provenance import file_sha256

CSV_SHA = "9ff5416a8bd1e2c6b9f70e620139f9c99a1d5146ec0bcfb6bb0f1b4a8e4bdbd7"
JSON_SHA = "ab373ebdad2cd804bb1d30f23046543da27edeeecdc7d4c727abf78b9535617c"
SEMANTIC_SHA = "6585d3ef8b6c177d2c3738cf7d6db38938520880afbea5eed1b3a7ca7ad2ff8e"
SIGMAS = [
    0.9489720463752747,
    0.9208537936210632,
    0.8844786882400513,
    0.8355841040611267,
    0.7663628458976746,
    0.6608133316040039,
    0.48016369342803955,
    0.10000002384185791,
]


def official_schedule(checkpoint):
    steps = detect_params(str(checkpoint)).num_inference_steps
    if steps != 30:
        raise ValueError("official checkpoint defaults changed")
    stock = LTX2Scheduler().execute(steps=steps)
    eligible = [i for i, value in enumerate(stock) if 0 < float(value) <= 0.95]
    indices = list(range(8, 30, 3))
    selected = [float(stock[i]) for i in indices]
    if eligible != list(range(8, 30)) or selected != SIGMAS:
        raise ValueError("official schedule changed; reconcile before execution")
    return {
        "reference_grid_steps": 30,
        "eligible_tail_steps": 22,
        "selected_level_count": 8,
        "executed_calls_per_view": 8,
        "original_grid": stock.tolist(),
        "eligible_tail": stock[8:].tolist(),
        "selected_tail_indices": list(range(0, 22, 3)),
        "stock_step_indices": indices,
        "sigmas": selected,
        "scheduler_arguments": {"steps": 30, "latent": None, "others": "defaults"},
        "scheduler_source_sha256": file_sha256(Path(inspect.getfile(LTX2Scheduler))),
        "evaluation_schedules": [[s, 0.0] for s in selected],
    }
