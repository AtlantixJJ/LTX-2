# `execution/journal.py`

Source owner: [`execution/journal.py`](../../execution/journal.py).

## Objective

Read and publish the queue journal under one exclusive transaction.

## Data flow

Saved JSON enters read_queue_state. A locked transaction validates ownership,
applies updates and publishes fsynced JSON only after the body succeeds.

## Organization logic

Reading validates append-only job order, identities, states and original clocks.
A new job becomes pending with an empty attempt list.
record_attempt sets the current running fields and appends an independent attempt record.
update_attempt applies shared fields to the current row and latest attempt together.
A failed transaction leaves previous bytes intact.
_publish_state writes the temporary JSON, fsyncs its bytes, replaces the journal and fsyncs its directory.
The transaction keeps the lock held through that publication.
Opening a transaction checks live owners and interrupted children. It does not adopt outputs.
Only recover=True invokes recovery decisions, inside the same journal lock.
Recovery retains original historical evidence where the existing format requires it.


## Invariants

Keep job hashes, saved schemas, scientific settings, budgets and verification gates unchanged.
Keep ordinary execution independent of experiment imports. Preserve original attempt evidence.

## Gotchas

Public queue aliases provide import compatibility, not another implementation.
Patch the defining owner in tests. Source-byte changes need fresh affected scientific evidence.

## Tests

Worked check: Register a child identity: the current row and last attempt must receive identical handles.
Raise inside the transaction: the saved journal must remain byte-identical.

Existing coverage: test_queue.py, test_queue_launch.py and test_queue_loop.py.
