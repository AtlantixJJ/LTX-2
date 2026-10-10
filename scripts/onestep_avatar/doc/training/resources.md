# `training/resources.py` — measure declared process resources

## Objective

Measure CUDA allocated and reserved peaks in the process that performs the work.
Bind the unchanged budget file to each phase. Sampled total-device memory is a
separate external observation; it cannot substitute for these measurements.
Outside the queue, ordinary runs may omit a budget. Queued children always use
the same effective budget as the queue's supervision/notification contract.
Current native update acceptance requires complete local CUDA measurements.

## Data flow

Read the budget bytes once, parse the two positive limits and retain their path
and SHA-256. Before each phase, recheck those bytes, synchronize the local CUDA
device and reset Torch peak counters. After the phase, synchronize again and
read allocated/reserved peaks and monotonic elapsed time. Retain one record per
rank and phase, including a failed record when execution or a limit check fails.
The engine/checker owns publication of these records and its failure evidence.
An operation failure after a phase starts retains a failed record. Before fresh
output exists, the trainer emits that record in its original queue log; it does
not create a scientific output directory to report startup failure. Budget
validation or CPU refusal before a phase starts is a preflight error with no
allocator measurement. Neither route can publish a complete checkpoint.
At each checkpoint, copy each rank's complete journal into an immutable
`resource_snapshots/step_<step>/` record. Marker hashes identify these snapshots;
later updates do not invalidate earlier checkpoint evidence. Final verification
also requires the snapshot records to equal the final live journals.

```mermaid
flowchart LR
  B[("declared resource budget")] --> R["read_budget"] --> P("checked limits and budget hash")
  P --> S["Phase.start"] --> C{{"local CUDA peak counters"}}
  W["engine: load, update or export"] --> C
  C --> F["Phase.finish"] --> M("elapsed time and allocated/reserved peaks")
  P --> F
  M --> E["engine: save phase record"] --> O[("resources_rankN.jsonl")]
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef tensor fill:#dff2e0,stroke:#43834a,color:#16341b;
  classDef state fill:#fdecc8,stroke:#b07d18,color:#3d2a05;
  class R,S,W,F,E proc;
  class B,O disk;
  class P,M tensor;
  class C state;
```

This diagram shows one native CUDA phase. The engine orders `Phase.start`, the
work and `Phase.finish`. A failed finish still yields a failed measurement for
the engine to preserve; it cannot certify completion. CPU runs record null peaks.

## Organization logic

`read_budget` consumes the existing protocol's `wall_seconds_per_phase` and
`memory_limit_allocated_bytes`; it does not choose new limits or tolerances.
It retains optional tolerance data from the same parsed byte buffer. The
experiment owner checks that data against its already declared rule. A second
filename read must not pair tolerance from new bytes with an old budget hash.
Optional `overall_seconds` is retained only when present and must be a finite
positive number, excluding booleans. It is an overall queue deadline; a local
phase never resets or substitutes for that deadline.

`default_budget` binds the declared 1,800-second phase bound and
48,000,000,000-byte allocated bound to canonical JSON bytes, with `path=null`
and a reproducible content SHA. `effective_budget(path)` selects an unchanged
explicit file record or that default without writing a file. `read_budget(None)`
still returns null outside this explicit queue policy.
`budget_environment` serializes the checked effective record and its SHA into
`ONESTEP_AVATAR_EFFECTIVE_BUDGET` and
`ONESTEP_AVATAR_EFFECTIVE_BUDGET_SHA256`. `budget_from_environment` requires
both fields, valid queue token/job bindings and unchanged file/default identity.
Partial, malformed or mismatched bindings fail before phase work.
`Phase(..., budget=None)` adopts only this bound queue budget. An explicit
phase budget must equal any inherited queue budget. Beginning and ending a
phase rechecks both identity and environment; events and saved measurements
carry the effective SHA. No default applies to an unqueued CPU measurement.
`Phase.start` and `Phase.finish` bracket load, each update and each export.
An allocated peak or elapsed time greater than its bound fails the phase.
Reserved peaks remain measurements, not a limit with an invented threshold.
CPU runs explicitly record null CUDA peaks and cannot satisfy native acceptance.
The final validator requires every declared rank and phase exactly once, finite
nonnegative measurements, unchanged budget and passed phase states. A missing
record, changed budget, duplicate rank/phase or breached limit refuses completion.
The external exact-owned-process wall-time supervisor remains required: measuring
elapsed time at a completed boundary alone cannot interrupt a hung operation.
When a checked supervision contract is present, publish token/job/rank/budget-bound
phase begin before the initial synchronize and phase end after measurement.
Notification failure fails the measurement while preserving an original operation
error. Without this contract the hooks do nothing. The shared
`execution/supervision.py` observer uses these events to time an in-progress phase; its
separate startup and transition guards cover intervals outside a local phase.
Historical supervisor text remains preserved and is not a current executor.
The local `load` phase starts after Accelerator setup; a separate startup guard
must cover process-group initialization. No existing sampled total-device bound
is relabeled as an allocated-byte bound.

`training_phases` supplies the chronological notification inventory from the
actual save rule: load, optional step-zero export, then each update followed by
its requested initial/periodic/final export. The queue freezes this order and
checkpoint verification uses the same inventory through that checkpoint.

Worked check: with a 48,000,000,000-byte allocated bound, a peak of
48,000,000,001 fails, even if sampled total-device memory was lower. A reserved
peak above that number does not by itself fail the allocated-memory criterion.
A missing rank-three export record cannot certify a four-rank update.
With a queued canonical default and a fake 100-second CUDA phase, the saved
record and both notifications carry the default SHA. A measured peak of
48,000,000,001 bytes fails under this default even if the phase caller supplied
null. Changing only the environment SHA or using a different explicit phase
budget refuses work; changing budget bytes during a phase preserves a failed
record with its original SHA.

## Invariants

- Use synchronized Torch counters on the actual local device, never nvidia-smi.
- Reset counters once per phase; include allocations still live at the reset.
- Preserve the budget's exact bytes and tolerances; never rewrite old evidence.
- Missing/non-CUDA/failed records cannot publish native acceptance.
- This owner launches no process and imports no experiment code.

## Gotchas

Torch counters exclude allocations outside its allocator. They measure the
declared allocated quantity, not total device occupancy. Synchronization adds
diagnostic overhead. Phase checks supplement external wall-time supervision.

## Tests

Use controlled CUDA counters to check reset/synchronization order, peaks, elapsed
time, exception preservation, exact limits and budget changes. Validate missing,
duplicate and malformed rank/phase inventories and explicit CPU refusal.
Effective-budget controls check canonical default identity, optional overall
validation, paired queue bindings, explicit/default disagreement, inherited
phase events and original no-budget behavior outside the queue.
