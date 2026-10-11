# `execution/recovery.py`

Source owner: [`execution/recovery.py`](../../execution/recovery.py).

## Objective

Classify interrupted attempts without launching or retrying work.

## Data flow

The journal supplies its locked state. Process observations and saved evidence
produce recovery decisions. The journal publishes those decisions atomically.

## Organization logic

Check every running child before changing any row. A surviving or uncertain child refuses recovery.
An unapproved ended launch fails without adopting output.
An expired original clock fails. Otherwise verify saved scientific completion and supervision.
Valid completion with valid original supervision can produce a receipt.
Missing original supervision needs the exact bounded ledger closure and always fails scientifically.
This owner never opens a journal transaction, selects GPUs or starts a process.

### Recovery

`--recover` cannot be used with `--execute` or `--dry-run`. It needs an
existing journal. It checks the job identities. Then it recovers every running
entry in one locked transaction. It never queries GPU memory, retries failed
work, moves files or starts processes. A journal without running entries gets a
read-only summary.

For each running entry, recovery inspects the recorded child, even when this
process owns the journal.

- A surviving, unidentified or reused handle, or a live foreign owner, refuses
  the whole transaction.
- A matching terminal or missing handle allows artifact verification:
  - a valid receipt marks the entry complete;
  - absent completion marks it failed with an interruption reason;
  - corrupt evidence refuses recovery and keeps the earlier bytes.

Failed outputs stay in place. A new attempt uses a new job ID and output.

**Unapproved guarded launch.** A registered-child row with no PID or identity
is classified `unapproved` only when all of these are true:

- the launch owner is gone;
- the request and attempt bindings match;
- no grant exists;
- any saved bootstrap is a known, terminated, matching guard;
- no owned worker survives.

Such a row is marked failed. Its output is not adopted. A row without the
registered-child protocol, and an unknown launch window, refuse takeover.

**Missing supervision.** An approved attempt can lose its owner before the
owner publishes a supervisor result. Such an attempt can be recorded as failed
before its deadline, after the matching bounded closure:
`queue_launch --recover-owned`, or `--recover-interrupted` when the workload
stopped early. That closure certifies only the exact registered handles. It
does not certify unknown descendants or continuous supervision. Recovery then
does these steps:

1. Check again the request, the approval, the unchanged journal, the original
   owned-row hash and the ended handles.
2. Set the reason `missing_supervision`.
3. Keep the earlier attempt under `interruption_recovery.prior_attempt`.
4. Keep all outputs and the original deadline.

The selected owner still checks the saved scientific files. Corrupt files
refuse recovery. Valid outputs never replace the supervisor. This branch writes
no success receipt.

Worked check: the deadline is 3,700 and recovery runs at 105. The exact owned
handles have ended, and the bounded closure matches. Missing supervisor fields
give `failed` with `missing_supervision`, never a completion receipt. The first
start stays 100 and the deadline stays 3,700.


## Invariants

Keep job hashes, saved schemas, scientific settings, budgets and verification gates unchanged.
Keep ordinary execution independent of experiment imports. Preserve original attempt evidence.

## Gotchas

Public queue aliases provide import compatibility, not another implementation.
Patch the defining owner in tests. Source-byte changes need fresh affected scientific evidence.

## Tests

Worked check: At time 105, an approved ended attempt without supervision and with deadline 3700 fails
with missing_supervision. Its deadline and original attempt remain preserved.

Existing coverage: test_queue_missing_supervision_recovery.py, test_guarded_recovery.py and test_queue_zombie.py.
