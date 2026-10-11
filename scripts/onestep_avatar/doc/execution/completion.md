# `execution/completion.py`

Source owner: [`execution/completion.py`](../../execution/completion.py).

## Objective

Verify saved outputs and bind scientific and execution evidence into receipts.

## Data flow

Prepared jobs and saved files enter model-free verifiers. Verified file inventories
and supervision records produce immutable hash receipts.

## Organization logic

ready_jobs checks receipts needed by pending dependencies and returns the eligible saved jobs.
It keeps non-training jobs first and checks saved rendering inputs before declaring a render ready.
The scheduler and direct persistent attempts share this dependency-evidence decision.

Keep each job kind's scientific verifier and all existing tensor, media and identity checks.
Check supervised execution separately from scientific artifacts. Neither can replace the other.
Build each receipt inventory after one completion verification. Rechecking a receipt verifies
its files, current completion and exact inventory. No verifier loads a native model.
The selected experiment comes only from jobs.experiment_entry.

### Artifacts and receipts

[journal](journal.md) owns saved state and its locked publication.

`verify_completion` checks each kind through the owner of its saved files.
Missing files return false, and the job stays pending. Malformed or changed
evidence raises an error. Evidence paths must stay inside the recorded output.

- `train`: the checkpoint and its `.complete.json` marker (integer schema two,
  `state="complete"`, exact path, current byte hash and the expected final
  step); the typed adapter contract and the actual matrices; a contract mode
  equal to the command's mode; and the engine's scientific verifier for the
  typed settings, data, base and parent identities, frame plan and queue and
  source digests. With `--save-update-state`, the receipt also binds the
  training text, the Adam moments and the records of each update.
- `evaluate`: every listed `result.json`, with a complete state and the
  command's mode, its encoding hash, recorded shape and finite content. Then
  `evaluate.verify_evaluation_conditions` checks the exact source and variant
  list, adapters, saved text and noise and the clean `c0`. One matching result
  cannot complete a different schedule, seed, source or adapter.
- `decode`: the declared manifest; the exact job and comparison IDs; matching
  source records; the VAE hash of the selected model; native decoder settings;
  the source-code hash, seed and encoded shape; every video and sample hash;
  the frame count, rate and size of each saved video (`(F-1)*time+1`,
  `H*height`, `W*width`); and the format and size of the sample PNGs. It loads
  saved tensors and hashes VAE files, but it opens no decoder.
- `render`: the exact spec, decoder and software identity; the comparison order
  and fields; every input hash and shape; coverage and fps; panel titles and
  layout; and the video and poster hashes.
- `preview`, `product`, `bench`, `experiment`: the `verify_completion` of the
  selected owner. A benchmark list must equal the owner's list.

`completion_receipt` verifies all evidence first. Then it records the job hash
and the hash of every declared file. `verify_receipt` requires the same
identity and unchanged evidence, and it runs the artifact checks again. A
completed row without a receipt is invalid. Missing receipt files block the
dependent jobs. Changed evidence raises an error.

One receipt check runs its completion verifier once. Building the evidence list
does not run that verifier again. Dispatch verifies the dependency receipts
once. Then it binds a copy of those receipts to the selected launch. Before
journal publication, `run_child` checks that every dependency is still complete
with the same job hash and receipt. A changed dependency stops the launch.
Direct persistent callers without a dispatch proof still verify their
dependencies through `ready_jobs`.


## Invariants

Keep job hashes, saved schemas, scientific settings, budgets and verification gates unchanged.
Keep ordinary execution independent of experiment imports. Preserve original attempt evidence.

## Gotchas

Public queue aliases provide import compatibility, not another implementation.
Patch the defining owner in tests. Source-byte changes need fresh affected scientific evidence.

## Tests

Worked check: Change a saved encoding after completion: verify_receipt must reject its hash.
An unchanged output with missing original supervision cannot complete a current job.

Existing coverage: test_ordinary_queue_completion.py, test_evaluation_completion.py and test_queue_render.py.
