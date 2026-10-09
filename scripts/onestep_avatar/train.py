"""Train avatar LoRA weights through the shared typed training engine.

Inputs: command-line options with an explicit ``--mode bidirectional|causal``, a
version-two fixed video list, and a fresh output path.
Logic: a command without ``--mode`` is refused before any environment change, import of
native libraries or write; the mode-less runtime retired in refactor group G11. A typed
command configures the shared numerical environment before native imports, then the
engine checks settings and inputs, prepares distributed training, updates weights,
records numeric logs and saves completed adapters.
Outputs: run settings, numeric logs and adapter checkpoints.
Rules: no model/cache/update logic belongs in this CLI. Run from LTX-2 in ltx.
Invalid requests and dry runs must not change an existing output directory.
"""

from __future__ import annotations

import os
import sys

from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_ENV
from scripts.onestep_avatar.training import numerics

MODE_REQUIRED = "train requires --mode bidirectional or --mode causal; the mode-less training runtime is retired"


def main(argv: list[str] | None = None) -> int:
    """Route only explicit-mode commands to typed settings and ``engine.run_settings``."""
    argv = sys.argv[1:] if argv is None else argv
    if not any(option == "--mode" or option.startswith("--mode=") for option in argv):
        raise SystemExit(MODE_REQUIRED)
    numerics.configure_environment(required=LAUNCH_ENV in os.environ)
    from scripts.onestep_avatar.training.config import parse_settings  # noqa: PLC0415 -- environment first
    from scripts.onestep_avatar.training.engine import run_settings  # noqa: PLC0415 -- environment first

    return run_settings(parse_settings(argv))


if __name__ == "__main__":
    raise SystemExit(main())
