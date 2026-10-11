# `execution/jobs.py`

Source owner: [`execution/jobs.py`](../../execution/jobs.py).

## Objective

Normalize saved jobs and build fixed package commands.

## Data flow

Raw JSON jobs flow through owner parsers, path resolution and content hashes.
Prepared jobs flow to the scheduler, attempt runner and completion verifier.

## Organization logic

The literal experiment table moves here. Ordinary owners never read it.
Owner imports stay lazy. Preparation preserves canonical hashes and omitted defaults.
The queue keeps public aliases for existing callers. Internal callers use this owner.

### Job list and identity

The list has `schema_version: 1` and an ordered `jobs` list. Each job has:

- a unique `id` and a `kind`;
- an `arguments` array of strings and an `output`;
- `dependencies` (earlier job IDs) and `completion` evidence.

Train jobs also give `accelerate_config`, `processes` (exactly four) and
`port`. Train and evaluate arguments contain exactly one
`--mode bidirectional|causal`.

`prepare_job` normalizes one job. `prepare_jobs` calls it after the list
checks. Replay calls it with the directory of the saved job. It does these
steps:

1. Resolve path arguments against the directory of the job list.
2. Parse the arguments with the public parser of the selected owner. This
   reads no inputs and opens no model.
3. Refuse `--gpu-id`, `--dry-run`, `--help`, `--overwrite`, `--saved-metrics`
   and `--verify`. Refuse `--preview-job` outside a `preview` job.
4. Require the recorded output to equal the parsed `--output`.

The canonical job SHA-256 hashes the normalized job with the resolved parser
settings, including omitted defaults. So adding a seed changes the identity,
but preparing the same list twice does not. A supplied prepared digest must
match the rebuilt one. Two output texts that resolve to the same directory
fail. A new revision of a list can add jobs. A change to the hash of an
existing job fails.

Kind-specific pins are part of the identity:

- `train`: the full SHA-256 of the Accelerate file
  (`accelerate_config_sha256`), and the final completion step. The step must
  equal the parsed `--steps`.
- `decode`: the resolved `--jobs` path and its content hash.
- `render`: the resolved `--render-saved-comparisons` path and hash. The
  completion manifest must be `<output>/render_manifest.json`.
- `experiment`: the selector, the spec path (equal to the one `--spec`
  argument) and its required `spec_sha256`.
- `preview`, `product`, `bench`, `experiment`: the `completion` descriptor of
  the owner parser must equal the job's `completion`. The descriptor is a
  nonempty mapping with a `manifest` string, a nonempty `records` list, or
  both. Every path is inside the output. `preview` also pins its fixed
  `preview_identity`.
- Every kind: the effective resource budget. `train` names it with
  `--resource-budget`. Other kinds use `resource_budget` and
  `resource_budget_sha256`. A supplied digest must match the file. A digest
  without a file fails. Without a file, the job binds the scientific default:
  1,800 seconds per phase and 48,000,000,000 allocated bytes. An experiment
  owner that declares its own scientific budget must receive exactly that
  budget.
- `numerical_policy` is absent or `"deterministic"`.

Worked check:

- A prepared four-process job records the bf16 YAML. A change of the YAML to
  `mixed_precision: no` changes the digest. The original claimed digest then
  fails normalization.
- A whitespace-only change in a budget file also changes the digest.
- A two-process job fails the four-process rule.
- An unchanged raw or prepared job rebuilds the same identity.


### Experiment selection

The `experiment` kind reads one literal `EXPERIMENTS` table. The table maps
each selector to its module: `stock_parity`, `adapter_effect_check`,
`continuation_check`, `continuation_identity_check`, `conditioning_check`,
`training_update_check`, `training_slice_check`, `causality`,
`fusion_parity`, `sigma_sweep`, `stats`, `prune_baseline` and
`prune_reproduction`.

- Only this branch reads the table. It imports the selected module when it
  needs it.
- An unknown selector fails before input reads, claims, state writes or
  children.
- Ordinary kinds never import an experiment. There is no discovery or
  registration API.
- The selected module owns its evidence layout and its
  `verify_completion(spec, root)`. For `sigma_sweep`, the descriptor is exactly
  `<output>/manifest.json`.
- `prune_baseline` runs the prune-owned whole-clip producer for one view and
  one seed in the generic one-rank phase. `prune_reproduction` runs its
  controls in the same kind of phase.


### Commands and environment

`job_command` builds argument arrays only for the fixed owners.

- Training uses the current Python with `-m accelerate.commands.launch`, the
  recorded configuration, `--num_processes 4` and the recorded port. It also
  sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Every child gets `CUDA_VISIBLE_DEVICES` set to its claimed physical IDs.
  Single-GPU children get `--gpu-id 0`, which is their local device after
  masking. The state keeps the physical ID.
- A deterministic job gets the numerical environment from
  `training/numerics.py`.
- The queue records only the environment changes, the attempt token
  (`ONESTEP_AVATAR_QUEUE_TOKEN`), the job hash
  (`ONESTEP_AVATAR_QUEUE_JOB_SHA256`) and the paths of the supervision and
  runtime contracts.
- Children run with `cwd=LTX_ROOT`, an argument array and no shell.

The device policy comes from `execution/devices.py`. The pool is GPUs 0–7, and
training uses four processes. A GPU is free when `nvidia-smi` reports less
than 1,024 MiB used and no active row of this ledger claims it. Training takes
the first four free devices in pool order. They need not be adjacent. Other
kinds take one device. A malformed or failed `nvidia-smi` read selects
nothing.


### Training launch binding

Before dispatch, `training_launch_record` records the launch
(`schema_version: 2`): the exact prepared job, the command, the physical
devices, the Accelerate bytes in base64 and the numerical environment. The
record is written beside the child log and passed through
`ONESTEP_AVATAR_QUEUE_LAUNCH`. Training copies it into its checked config and
completion evidence.

`verify_training_launch` accepts only schema two. It rebuilds the job and the
command. It compares the original bytes with the current file. It refuses
changed arguments, processes, port, command, claimed hash or YAML before the
models load. These checks bind the requested launch facts. The engine
separately records the actual world size, the FSDP dtype policies and the rank
agreement after setup. A record without these facts cannot pass replay.


## Invariants

Keep job hashes, saved schemas, scientific settings, budgets and verification gates unchanged.
Keep ordinary execution independent of experiment imports. Preserve original attempt evidence.

## Gotchas

Public queue aliases provide import compatibility, not another implementation.
Patch the defining owner in tests. Source-byte changes need fresh affected scientific evidence.

## Tests

Worked check: Prepare a relative output twice: both prepared jobs must have the same hash.
Change the Accelerate bytes: the old claimed hash must fail.

Existing coverage: test_queue.py, test_experiment_queue.py and the ordinary completion tests.
