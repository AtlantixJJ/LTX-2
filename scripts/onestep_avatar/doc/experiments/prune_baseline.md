# `experiments/prune_baseline.py` — queued raw pruning baseline

## Objective

Expose the existing pruning-owned whole-clip numerical producer through the
fixed `prune_baseline` experiment selector. The wrapper does not sample, decode,
launch children or allocate devices. The ordinary queue owns its one-rank,
generic bounded runtime phase; `execution.runtime.run` records that phase.
`scripts.prune.evaluate.whole_clip_baseline` retains the numerical producer and
its original manifest inside `baseline/`.

## Organization logic

The outer parser and `spec_protocol` own schema and scientific arguments;
`input_paths` and acceptance records own fixed byte identity; the pruning
producer owns computation and saved scientific checks. Public spec-path
completion and evidence APIs match the existing queue contract and raise on
invalid completion. No ordinary profile or pruning dependency points back to
the orchestration module.

## Data flow

1. `spec_protocol` reads a `schema_version: 1` scientific spec and supplies queue-owned
   output/device arguments. Restrict this route to one absolute view, one seed,
   raw-only white distilled unguided one-step D0, optionally with D1. The shared
   spec protocol still rejects repeated options, including repeated `--view`.
2. Before native work, resolve public backbone/registry component paths and pin
   full SHA-256 of capture, guide, transformer, text encoder, VAE, existing prompt
   context cache, optional prompt file and the scientific spec. Capture the current evaluation/causal software
   manifest with reviewed producer dependencies as explicit extra sources.
   Ordinary software profiles remain unchanged. A dry run parses the scientific
   spec only, writes nothing and never hashes or loads model components.
3. Call the unchanged producer's `main` directly with its scientific arguments
   and `--output <root>/baseline`. No process or controller is introduced.
4. Recheck every input hash and current source/runtime identity. The producer's
   saved-only verifier checks settings, exact coverage, local artifact hashes,
   BF16 finite shapes, shared saved epsilon and clean capture frame `c0`.
   Registered scale factors, capabilities and component fingerprints must match
   the producer stamps, and actual BF16 cache bytes must match its context hash.
5. Publish `acceptance.json`, then the queue-facing root `manifest.json` through
   `spec_protocol`. Resume verification repeats the input/software checks,
   producer checks and exact local output inventory. Evidence includes every
   contained baseline file and acceptance record.

## Invariants

One view, one seed and one global saved epsilon define the comparison. The
original producer manifest remains untouched inside `baseline/`.

## Gotchas

This route reproduces raw baseline values, not decoded media. The direct prune
CLI keeps its broader options. Completion verifies saved noise identity and
geometry; CPU verification cannot independently regenerate device-specific
seeded CUDA noise. A native old/new comparison must compare the saved epsilon
tensors and D0 outputs directly. Equal seeds alone do not prove equal noise.
Full component hashes are intentionally required for native acceptance and
resume, and can take time for large weights. A partial output never becomes a
completion receipt; it is preserved for diagnosis rather than overwritten.
The selected prompt cache must already exist. Cache construction and fresh
text-encoder parity remain owned by the pruning prompt-cache command.

## Tests

`tests/experiments/test_prune_baseline.py` exercises scientific restrictions,
queue routing and source/input pin refusals with small files. Prune's
`tests/test_whole_clip_baseline.py` exercises saved-only coverage, tensor,
hash, `c0` and containment refusals using its tiny CPU producer fixture.
