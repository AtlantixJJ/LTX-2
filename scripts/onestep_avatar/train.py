"""Train avatar LoRA weights through the shared training engine.

Inputs: command-line options, encoded video records, and a fresh output path.
Logic: the engine checks settings and inputs, prepares distributed training,
updates weights, records numeric logs, and saves completed adapters.
Outputs: run settings, numeric logs, and adapter checkpoints.
Rules: no model/cache/update logic belongs in this CLI. Run from LTX-2 in ltx.
Invalid requests and dry runs must not change an existing output directory.
"""

from __future__ import annotations

import sys

from scripts.onestep_avatar.training.config import parse_settings
from scripts.onestep_avatar.training.engine import main as queued_main
from scripts.onestep_avatar.training.engine import run_settings


def main(argv: list[str] | None = None) -> int:
    """Explicit-mode commands use typed settings; live old queues retain their entry path."""
    argv = sys.argv[1:] if argv is None else argv
    if any(option == "--mode" or option.startswith("--mode=") for option in argv):
        return run_settings(parse_settings(argv))
    return queued_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
