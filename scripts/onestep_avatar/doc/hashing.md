# `hashing.py` — the one file-hash helper

## Objective

`sha256(path)`, read in 1 MiB chunks. Nothing else. Split out in S1 of the 2026-09-17 cleanup
plan because `precompute.py` (the corpus pass) and `windows.py` (the subset freezer) each
carried a byte-identical copy — the same algorithm, the same chunk size — after the 2026-09-15
consolidation removed the two-tree reason for the duplication.

## Data flow

```
Path ─▶ sha256() ─▶ hex digest
```

Two callers: `precompute.py` hashes the video VAE checkpoint and each capture/guide bundle for
provenance; `windows.py` hashes the selected `rgb.mp4` and guide render to content-pin a frozen
subset (`windows.py`'s own docstring, SS5.0).

## Organization logic

Pure stdlib (`hashlib`, `pathlib`), no dataset or GPU dependency — the same reason `geometry.py`
is its own module rather than folded into `dataset.py` or `precompute.py`. A hash is not
geometry and not corpus layout; it earns three lines of its own rather than living in either.

## Invariants

- **One implementation.** Do not reintroduce a second `sha256` "to avoid an import" —
  `CLAUDE.md` invariant 3 names exactly this shape of duplication.

## Tests

`tests/test_hashing.py`.
