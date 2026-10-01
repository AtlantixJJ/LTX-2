# `data/prompt_cache.py`

## Objective and data flow

Cache prompt conditioning by model key and prompt hash. `get_or_build` loads a
cached video encoding or runs the text encoder once and stores it on CPU.
The CLI uses `session.DEFAULT_PROMPT`; other consumers pass explicit text.

## Invariants and verification

Changing text changes the cache path. `--verify` recomputes the encoding and
checks bit equality. Native reconstruction separately checks actual context bytes
against saved baseline provenance. Use a free GPU in the ltx environment when
building or verifying a real prompt cache.
