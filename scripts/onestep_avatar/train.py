"""Train avatar LoRA weights through the shared training engine.

Inputs: command-line options, encoded video records, and a fresh output path.
Logic: the engine checks settings and inputs, prepares distributed training,
updates weights, records numeric logs, and saves completed adapters.
Typed commands configure the shared numerical environment before native imports.
Outputs: run settings, numeric logs, and adapter checkpoints.
Rules: no model/cache/update logic belongs in this CLI. Run from LTX-2 in ltx.
Invalid requests and dry runs must not change an existing output directory.
"""

from __future__ import annotations

import os
import sys

from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_ENV
from scripts.onestep_avatar.training import numerics


def main(argv: list[str] | None = None) -> int:
    """Explicit-mode commands use typed settings; live old queues retain their entry path."""
    argv = sys.argv[1:] if argv is None else argv
    if any(option == "--mode" or option.startswith("--mode=") for option in argv):
        numerics.configure_environment(required=LAUNCH_ENV in os.environ)
        from scripts.onestep_avatar.training.config import parse_settings  # noqa: PLC0415 -- environment first
        from scripts.onestep_avatar.training.engine import run_settings  # noqa: PLC0415 -- environment first
        return run_settings(parse_settings(argv))
    from scripts.onestep_avatar.training.engine import main as queued_main  # noqa: PLC0415 -- legacy path unchanged
    return queued_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
