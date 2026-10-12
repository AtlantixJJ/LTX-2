# `experiments/prune_baseline.py` — queued raw pruning baseline

## Objective

Run the existing whole-clip numerical producer of the pruning package through
the fixed `prune_baseline` experiment selector. The wrapper does not sample,
decode, start children or allocate devices.

- The ordinary queue owns its one-rank, generic bounded runtime phase.
  `execution.runtime.run` records that phase.
- `scripts.prune.evaluate.whole_clip_baseline` keeps the numerical producer and
  its original manifest inside `baseline/`.

## Usage

Humans can use the foreground queue or run this module directly.
See [launch, output and stop commands](../usage.md). Agents use the queue.
The queue recipes below describe managed execution.

Start the baseline as a queue `experiment` job with selector `prune_baseline`.
See [run guide §9](../../configs/README.md#pruning-baseline-prune_baseline) and
the [spec format](spec_protocol.md).

This is the agent path. A person can also run the same module directly, or start the queue in the foreground; see [usage](../usage.md).

```json
{"schema_version": 1, "protocol": "prune_baseline",
 "arguments": ["--view", "/abs/corpus/Part_1/0008_01/views/view00_cam51", "--model", "2.5",
               "--variant", "distilled", "--objective", "white", "--sigmas", "0.725", "0.909375", "1.0",
               "--context-latent-frames", "8", "--prompt", "", "--seed", "42", "--raw-only", "--arms", "d0"]}
```

The spec allows exactly one view and one seed, raw output, `white` distilled weights or explicit direct dev D0, no guidance and context eight.
Direct dev requires `--dev-schedule direct`, no transformer override, D0 only, and all sigmas at most 0.95. `--arms d0 d1` adds the D1 arm. The
other arguments are the producer's (see
[`scripts/prune/doc/evaluate/whole_clip_baseline.md`](../../../prune/doc/evaluate/whole_clip_baseline.md)).

Check the spec on CPU first. A dry run parses the spec only. It writes nothing
and never hashes or loads model components:

```bash
cd LTX-2
"$PY" -m scripts.onestep_avatar.experiments.prune_baseline \
  --spec <SPEC.json> --output <FRESH_OUT> --dry-run
```

Outputs: the producer's files under `<FRESH_OUT>/baseline/`,
`acceptance.json` and `manifest.json`. The selected prompt cache must already
exist.

## Data flow

1. `spec_protocol` reads a `schema_version: 1` scientific spec and supplies the
   queue-owned output and device arguments. This route allows one absolute
   view, one seed and raw-only `white` unguided one-step D0.
   Distilled can also produce D1. Direct dev uses the unmodified dev checkpoint. The shared spec protocol still refuses repeated options,
   including a repeated `--view`.
2. Before native work, resolve the public backbone and registry component
   paths. Pin the full SHA-256 of the capture, the optional D1 guide, the transformer, the
   text encoder, the VAE, the existing prompt context cache, the optional prompt
   file and the scientific spec. Record the current evaluation/causal software
   manifest, with the reviewed producer dependencies as explicit extra sources.
   The ordinary software profiles do not change.
3. Call the unchanged producer's `main` directly, with its scientific arguments
   and `--output <root>/baseline`. No process or controller is added.
4. Check every input hash and the current source and runtime identity again.
   The producer's saved-only verifier checks the settings, the exact coverage,
   the local file hashes, finite BF16 shapes, the shared saved epsilon and the
   clean capture frame `c0`. The registered scale factors, capabilities and
   component fingerprints must match the producer stamps. The actual BF16 cache
   bytes must match its context hash.
5. Publish `acceptance.json`, then the queue-facing root `manifest.json`
   through `spec_protocol`.

Resume verification repeats the input and software checks, the producer checks
and the exact local output list. The evidence includes every contained
baseline file and the acceptance record.

## Organization logic

- The outer parser and `spec_protocol` own the schema and the scientific
  arguments.
- `input_paths` and the acceptance records own the fixed byte identity.
- The pruning producer owns the computation and the saved scientific checks.

The public spec-path completion and evidence APIs match the existing queue
contract. They raise on invalid completion. No ordinary profile or pruning
dependency points back to this orchestration module. The queue can call
`evidence_paths(..., verified=True)` only after it ran this verifier in the same
receipt check. Direct callers keep the default `verified=False`.

## Invariants

- One view, one seed and one global saved epsilon define the comparison.
- The original producer manifest stays unchanged inside `baseline/`.

## Gotchas

- This route reproduces raw baseline values, not decoded media. The direct
  prune CLI keeps its broader options.
- Completion verifies the saved noise identity and geometry. CPU verification
  cannot regenerate device-specific seeded CUDA noise. A native old/new
  comparison must compare the saved epsilon tensors and the D0 outputs
  directly. Equal seeds alone do not prove equal noise.
- Full component hashes are required for native acceptance and resume. They can
  take time for large weights.
- A partial output never becomes a completion receipt. It is kept for
  diagnosis, not overwritten.
- The selected prompt cache must already exist. The pruning prompt-cache
  command owns cache construction and fresh text-encoder parity.

## Tests

- `tests/experiments/test_prune_baseline.py` checks the scientific
  restrictions, the queue routing and the refusals of source and input pins,
  with small files.
- Prune's `tests/test_whole_clip_baseline.py` checks the saved-only refusals of
  coverage, tensor, hash, `c0` and containment, with its tiny CPU producer
  fixture.

## Direct dev input checks

Direct dev validates exact `[sigma,0]` calls through the producer.
It does not apply the distilled grid check to dev sigmas.
Capture-only D0 excludes guides from input pins and never opens a guide file.
The existing distilled grid and optional D1 recipe retain their checks.
CPU tests cover off-grid dev values, input pins and rejection of D1, checkpoint overrides and sigma above 0.95.
