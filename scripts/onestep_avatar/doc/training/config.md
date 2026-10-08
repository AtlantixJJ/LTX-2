# `training/config.py` — check run settings

The unsupported `--anchor-weight` option is removed from both explicit-mode and
transitional parsing. Old saved configurations remain historical input records;
they do not authorize restoring the deleted offline-teacher path.

Status: **Typed settings, mode plans, and explicit-mode CLI integration implemented. Fixed preview settings and execution are implemented; complete native workflow acceptance and removal of transitional parsing remain pending. Read [current acceptance](../known_gaps.md#current-acceptance-and-next-step).**
`parse_args`, `training_sigmas`, `training_noise_seed`, and `sigma_for_rank`
are extracted without changing their calculations. The engine imports them.
The transitional mode-less route still accepts old subset settings.

`parse_settings` produces `RunSettings` with a `BidirectionalSettings` or
`CausalSettings` value. The old parser remains temporarily while existing live
queues finish or transfer ownership. Its removal remains a Stage D gate;
explicit-mode commands already use the typed public CLI.
`--frame-plan` can select a saved reproduction plan. The reader checks its hash,
membership hash, mode, geometry, and sample lengths before model loading.
For a random-start plan, `window_start_draw` owns the saved draw descriptor:
uniform integer starts, the noise seed and
`onestep_avatar.window:{seed}:{step}:{rank}:{slot}`. Reproduction plans must
match this descriptor exactly, even if their content hash is valid. Clip-start
plans reject a random descriptor. Adapter records consume the same descriptor.
Causal random-start plans require only block zero and its `[0,B+1]` template.
Do not accept a later block index that the runtime would apply to a newly sliced
independent window.
`--allow-cross-mode-init` permits only explicit parent-weight initialization with
the same base, rank, alpha, and targets. Record the parent mode; do not call this resume.

## Objective

`save_update_state` defaults to false. The explicit `--save-update-state` flag
requests diagnostic named Adam moments from the shared engine after updates.
It changes saved evidence only; it changes no noise, sample, optimizer or adapter
contract. Record it with the resolved run settings. The artifact is not optimizer
resume support. Older/default jobs do not request the extra collection.

`--resource-budget PATH` optionally pins the existing frozen protocol bytes.
It adds no experiment loader or new limits. The resource owner reads the exact
allocated-memory and per-phase wall-time limits before execution and measures
load, update and export phases. Current native replay requires this evidence;
ordinary CPU controls and older runs without the option keep their original scope.

`--consumer-trace` is an optional diagnostic flag, default false. It records
actual model-consumer conditioning and adapter compute through observational
hooks after wrapping. It changes evidence and measured overhead, not the
training calculation. Completion binds each checkpoint's immutable per-rank
trace. Incomplete or changed requested traces refuse completion. It does not
establish native numerical agreement by itself.

Resolve one explicit mode before loading model weights or changing output files.
Keep command-line configuration. Do not introduce an experiment-YAML loader or implicit resume.

## Data flow

Read CLI arguments, the fixed video list, and the base weight identity.
Produce checked common settings, mode settings, and a frame selection plan.
The engine receives those checked records.

## Organization logic

Require `--mode bidirectional` or `--mode causal` in `train.py`, `evaluate.py`, and `infer.py`.
Training selects `plan_samples` and `train_sample` from that mode.
Evaluation and product generation select its `sample` function and check adapter settings.
See the [README workflows](../../README.md#explicit-training-and-generation-workflows).

Common settings include paths, train/evaluation group, model version, base variant,
D0/D1, background, noise levels, seeds, LoRA, optimizer, accumulation, and logging.
Keep model version 2.5 and the current distilled default.
Examples state their base variant explicitly.

Bidirectional settings select the number of encoded frames and the start rule.
Reject explicitly supplied block, cache, priming, or history-policy options.
Causal settings select block length, blocks per sample, cache depth, and history policy.
Keep the current one-block limit for random-start independent segments.

Check settings in this order:

1. Parse types. Distinguish omitted options from explicitly supplied options.
2. Resolve defaults. Keep the current seed fallback to `seed`.
3. Check finite noise levels, unique level lists, LoRA values, and accumulation bounds.
4. Check background, train/evaluation group, video hashes, frame counts, frame rate, and guide availability.
5. Build the selected mode's frame plan. Check sample/block counts across GPU processes and cache capacity.
6. Check a parent adapter's base/LoRA settings. Record the previous training stage.
7. Check output-directory handling. Archive an old output only after all checks succeed.

A dry run prints resolved settings.
It does not archive files, create text caches, or load model weights.
Save resolved settings in `config.json`.

Queued training also saves `queue_launch`, the original dispatch binding from
the queue owner. It includes exact normalized arguments/processes/port/command
and original Accelerate bytes/hash. Applied runtime evidence is separate from
CLI settings: the engine records actual world and each rank's precision policy
after Accelerator/FSDP setup. Readers compare those facts with the launch;
they cannot infer missing historical settings from today's file.

Keep current random start and noise-seed rules.
Each update/GPU process draws one sigma uniformly from its levels.
Accumulated samples on that process share the draw.

Visualization settings specify plot output and optional preview jobs.
A preview job records its checkpoint step, fixed inputs, mode, exact levels, seed, and output path.
Keep preview execution separate from distributed training; see [engine](engine.md) and [media](../media.md).
Check preview inputs before scheduling a job.

### Core resolution logic

The typed mode/data interface below is implemented. Mode-less live queue commands temporarily retain the old parser.
Keep a record of explicitly supplied options before applying defaults.
For bidirectional mode, reject explicit causal options even when their values equal a causal default.
An omitted causal option never enters the resolved bidirectional record.
For causal mode, require positive block length and K, nonnegative history depth,
and enough complete blocks for every planned sample.
For bidirectional mode, require a nonempty legal segment within each selected master.
Reject an empty frame plan before any distributed work.
The causal frame plan records sample stride (default K), ordered starts, and sources yielding no full sample.
Reproduction plans retain original starts/indices and seed mappings.

The checked result contains four parts:

- Common data/model identity: resolved paths, background, D0/D1, groups, actual base identity,
  checked video-list hash, and frame-plan hash.
- One mode record: segment length/start rule, or block length/K/history depth/policy.
- Training runtime: LoRA/optimizer values, accumulation, steps, seeds, noise rule, and save/log settings.
- Preview settings: fixed inputs, levels, output roles, and checkpoint-trigger rule; no preview model session.

Print these same records for a dry run and save them for an accepted run.
The engine receives them directly. It does not re-parse the CLI or infer mode from K.
Only after successful checks may the engine create/archive run output and build model/text caches.

### Reproducible noise and level choices

Keep the extracted seed calculations exactly:

- Fixed-per-chain base noise seed: `noise_seed*100003 + chain_index*101`.
- Fresh base noise seed: `((noise_seed*1000003 + step)*1009 + rank)*131 + slot*17`.
- Existing block noise uses the base seed plus its block index.
- For one sigma level, use that value. For multiple levels, seed a local Python generator with
  `onestep_avatar.sigma:{seed}:{step}:{rank}` and select `rng.randrange(number_of_levels)`.

Fresh noise changes on the next update/visit. Fixed-per-chain noise reuses the same chain draw.
The sigma draw is independent of the noise stream and data order.
One draw applies to all accumulation slots on that process/update.
Equal exposure to levels is expected over many draws; it is not guaranteed per update or epoch.
Reproduction plans preserve original chain indices where that seed rule needs them.

## Invariants

- One block does not automatically select bidirectional mode.
- A resolved bidirectional record contains no causal settings.
- Invalid requests and dry runs do not change an existing output directory.
- A parent adapter starts a new stage with fresh optimizer, step, and random state.
- Frame plans use the fixed video list and declared rules, not directory names.
- Changing modes starts a new run; it does not convert an existing adapter.

## Gotchas

An omitted cache option must not make a bidirectional request invalid.
Keep GPU-process settings in Accelerate configuration.
Keep model mode and preview settings in run configuration.

## Tests

[V6–V8](../verification.md) check mode fields, data changes, and incompatible adapters.
After implementation, test these checks before any model load.
Check dry-run and archive behavior.
Check that preview jobs record fixed inputs and do not execute inside the training loop.
Worked mode check: bidirectional mode with an explicit history-depth option fails,
even if the value is zero. Omitting that option succeeds when segment/data checks pass.
Worked side-effect check: an existing output directory and invalid D1 guide must remain unchanged.
A valid dry run prints the same mode/frame plan as execution but creates no model/cache/output files.
Worked seed check: `noise_seed=2`, `chain_index=3` gives fixed base seed `200309`.
Changing only sigma does not change that base seed.
Explicit-mode train commands run through the typed engine. They require a version-two fixed video list.
Preview CLI settings and checked input records are implemented. Native preview output remains an acceptance gate.

`preview_inputs` is an optional JSON path. `preview_record` is the checked,
immutable input record retained by preflight. A dry run reads and validates it
but enqueues no job. Preview settings add no training or decoder session.
