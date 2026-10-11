# `execution/processes.py`

Source owner: [`execution/processes.py`](../../execution/processes.py).

## Objective

Read stable Linux process identities and classify owned workers.

## Data flow

PID, session and attempt-token inputs produce stable process observations.
Ledger-backed rows use targeted registered descendants. Legacy fixtures use session/token scans.

## Organization logic

Read stat before and after cmdline. Reject a changed start tick.
Missing handles and terminal zombies differ from unreadable live handles.
Check owned workers before declaring a leader terminal. Guard command transitions remain
verified by queue_launch. These helpers never import the scheduler.

### Process identity

The identity uses the start ticks from `/proc/<pid>/stat` and the exact
NUL-separated command arguments.

- Parse the stat fields after the last closing parenthesis, because the process
  name can contain spaces or parentheses.
- Read stat before and after the command. A changed start tick rejects the
  observation.
- A zombie or dead process is terminal, even if signal zero succeeds.
- Missing process files mean that the handle disappeared. Permission errors and
  other I/O errors stay errors. They never prove death.

`inspect_child` compares the saved PID and start ticks. A live handle must have
the saved command. A matching terminal handle can have the saved command or an
empty one, because Linux clears `cmdline` before the parent reaps a zombie. The
function checks for surviving owned workers before it returns terminal.

`owned_workers_live` uses the targeted ledger observation for rows that record
a process ledger. Other rows use the two-pass scan by session and token. Both
fail closed on unreadable or reused handles. A vanished session leader never
proves that the workers ended.

Worked check: save PID 42, start ticks 100 and command `python worker.py` while
the process is live. After exit, observe PID 42, ticks 100, terminal true and
command `[]`. With no owned workers, the result is terminal. With a live
token-bound worker, it is live. Ticks 101 refuse recovery in both cases.


## Invariants

Keep job hashes, saved schemas, scientific settings, budgets and verification gates unchanged.
Keep ordinary execution independent of experiment imports. Preserve original attempt evidence.

## Gotchas

Public queue aliases provide import compatibility, not another implementation.
Patch the defining owner in tests. Source-byte changes need fresh affected scientific evidence.

## Tests

Worked check: PID 42 with changed ticks is not the original child.
An ended leader with a surviving owned worker remains live.

Existing coverage: test_queue_proc_races.py, test_queue_zombie.py and test_supervision.py.
