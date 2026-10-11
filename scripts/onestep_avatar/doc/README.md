# `onestep_avatar` — current design index

## Reading order

Read the documents in this order:

1. [Architecture](architecture.md): who owns each part of the code, and which
   imports are allowed.
2. [Core algorithm](core_algorithm.md): tensor shapes, conditioning and the
   calculation for each mode.
3. [Training choices](training_choices.md): the supported settings.
4. [Verification](verification.md): the worked checks and the acceptance gates.
5. [Known gaps](known_gaps.md): the current limits.

The [run guide](../configs/README.md) gives the full commands for each workflow.
Each entry-script document also has a short **Usage** section.
The [human and agent usage guide](usage.md) explains both launch paths, output and stopping.

## Documentation rules

**Which file gets a document.** A production file with more than 100 physical
lines has a design document at `doc/<relative-source-path>.md`. Count blank
lines, comments and headers. A file with 100 lines or fewer explains its logic
in its header. The header gives the objective, the ordered steps, the inputs
and outputs, the failures and the invariants. Empty package markers and tests
need no document. Do not split or pad a file to change which rule applies.

**Sections.** A design document has these sections: Objective, Usage (entry
scripts only), Data flow, Organization logic, Invariants, Gotchas and Tests.
Explain the actual decisions, calculations, state changes and failure meanings.
A list of symbols does not replace the core logic. Give a worked input and its
expected result. For a visual output, state the data, positions, labels, time
map and readability rules.

**Usage section.** Each entry script (a module with a `python -m` entry) has a
Usage section after the Objective. It states:

- who starts the module: agents through the queue; humans through the queue or directly;
- the conda environment;
- one complete example command, with a CPU `--dry-run` check when one exists;
- a table of the main arguments;
- the output files;
- a link to the matching [run guide](../configs/README.md) section.

**Diagrams.** Use small left-to-right Mermaid flows with the
[shared legend](core_algorithm.md#7-end-to-end-data-flow). Render each diagram
and look at it. A diagram cannot prove numerical or distributed behavior.

**Language.** Write in simplified technical English, about 80% to the
ASD-STE100 rules:

- Write one topic in each sentence. Keep a sentence to 20 words or fewer for an
  instruction and 25 words or fewer for a description.
- Keep a paragraph to six sentences or fewer.
- Use the active voice and the present tense. Use the imperative for an
  instruction.
- Use the articles "a", "an" and "the". Do not stack more than three nouns
  together.
- Use one term for one concept. Define a term before you use it.
- Put steps and conditions in a numbered or bulleted list.
- Keep code names, file names, numbers and equations exactly as they are.
- Use "must" for a rule and "can" for an option.

## Current source owners

The index lists only the source files that exist now. Each package has one job:

- `corpus/` makes the data and the checked video lists.
- `model/` holds the numerical calculations.
- `training/` holds the optimizer loop and the typed settings.
- `execution/` holds the one queue and the process supervision.
- The root modules do evaluation, product generation, media, input preparation,
  plots and cost measurement.
- `experiments/` runs scientific comparisons through the public functions of
  the packages above.

Ordinary imports and software profiles do not include `experiments/`. Report
code under `expr/` reads saved inputs only. It cannot run a model, decode or
start queue work.

| Source | Design | Physical lines |
|---|---|---:|
| `__init__.py` | source header | 13 |
| `bench.py` | [bench.md](bench.md) | 577 |
| `comparisons.py` | [comparisons.md](comparisons.md) | 501 |
| `corpus/build_guidance.py` | [corpus/build_guidance.md](corpus/build_guidance.md) | 778 |
| `corpus/dataset.py` | [corpus/dataset.md](corpus/dataset.md) | 397 |
| `corpus/geometry.py` | [corpus/geometry.md](corpus/geometry.md) | 111 |
| `corpus/mask_video.py` | [corpus/mask_video.md](corpus/mask_video.md) | 181 |
| `corpus/motion.py` | [corpus/motion.md](corpus/motion.md) | 187 |
| `corpus/precompute.py` | [corpus/precompute.md](corpus/precompute.md) | 1390 |
| `corpus/qa.py` | source header | 29 |
| `corpus/subset.py` | [corpus/subset.md](corpus/subset.md) | 400 |
| `decode_saved.py` | [decode_saved.md](decode_saved.md) | 192 |
| `evaluate.py` | [evaluate.md](evaluate.md) | 740 |
| `execution/clock.py` | source header | 72 |
| `execution/processes.py` | [execution/processes.md](execution/processes.md) | 206 |
| `execution/completion.py` | [execution/completion.md](execution/completion.md) | 500 |
| `execution/journal.py` | [execution/journal.md](execution/journal.md) | 105 |
| `execution/recovery.py` | [execution/recovery.md](execution/recovery.md) | 204 |
| `execution/attempt.py` | [execution/attempt.md](execution/attempt.md) | 571 |
| `execution/jobs.py` | [execution/jobs.md](execution/jobs.md) | 556 |
| `execution/devices.py` | source header | 84 |
| `execution/process_registry.py` | [execution/process_registry.md](execution/process_registry.md) | 358 |
| `execution/queue.py` | [execution/queue.md](execution/queue.md) | 396 |
| `execution/queue_launch.py` | [execution/queue_launch.md](execution/queue_launch.md) | 1061 |
| `execution/queue_protocol.py` | source header | 11 |
| `execution/runtime.py` | [execution/runtime.md](execution/runtime.md) | 173 |
| `execution/software.py` | [execution/software.md](execution/software.md) | 202 |
| `execution/supervision.py` | [execution/supervision.md](execution/supervision.md) | 521 |
| `execution/terminal.py` | source header | 72 |
| `experiments/adapter_effect_check.py` | [experiments/adapter_effect_check.md](experiments/adapter_effect_check.md) | 957 |
| `experiments/capacity_review.py` | [experiments/capacity_review.md](experiments/capacity_review.md) | 256 |
| `experiments/causality.py` | [experiments/causality.md](experiments/causality.md) | 764 |
| `experiments/conditioning_check.py` | [experiments/conditioning_check.md](experiments/conditioning_check.md) | 595 |
| `experiments/continuation_check.py` | [experiments/continuation_check.md](experiments/continuation_check.md) | 917 |
| `experiments/continuation_identity_check.py` | [experiments/continuation_identity_check.md](experiments/continuation_identity_check.md) | 474 |
| `experiments/fusion_parity.py` | [experiments/fusion_parity.md](experiments/fusion_parity.md) | 330 |
| `experiments/past_influence_check.py` | [experiments/past_influence_check.md](experiments/past_influence_check.md) | 594 |
| `experiments/prune_baseline.py` | [experiments/prune_baseline.md](experiments/prune_baseline.md) | 201 |
| `experiments/prune_reproduction.py` | [experiments/prune_reproduction.md](experiments/prune_reproduction.md) | 417 |
| `experiments/prepare_continuation.py` | [experiments/prepare_continuation.md](experiments/prepare_continuation.md) | 521 |
| `experiments/reference_inputs.py` | [experiments/reference_inputs.md](experiments/reference_inputs.md) | 149 |
| `experiments/saved_probe_metrics.py` | [experiments/saved_probe_metrics.md](experiments/saved_probe_metrics.md) | 141 |
| `experiments/short_causal_check.py` | [experiments/short_causal_check.md](experiments/short_causal_check.md) | 573 |
| `experiments/sigma_sweep.py` | [experiments/sigma_sweep.md](experiments/sigma_sweep.md) | 329 |
| `experiments/sigma_sweep_results.py` | source header | 70 |
| `experiments/spec_protocol.py` | [experiments/spec_protocol.md](experiments/spec_protocol.md) | 139 |
| `experiments/stats.py` | [experiments/stats.md](experiments/stats.md) | 470 |
| `experiments/stock_parity.py` | [experiments/stock_parity.md](experiments/stock_parity.md) | 654 |
| `experiments/training_slice_check.py` | [experiments/training_slice_check.md](experiments/training_slice_check.md) | 462 |
| `experiments/training_update_check.py` | [experiments/training_update_check.md](experiments/training_update_check.md) | 703 |
| `hashing.py` | source header | 34 |
| `infer.py` | [infer.md](infer.md) | 696 |
| `media.py` | [media.md](media.md) | 714 |
| `metrics.py` | [metrics.md](metrics.md) | 151 |
| `model/adapters.py` | [model/adapters.md](model/adapters.md) | 145 |
| `model/backbone.py` | source header | 68 |
| `model/bidirectional.py` | [model/bidirectional.md](model/bidirectional.md) | 141 |
| `model/causal.py` | [model/causal.md](model/causal.md) | 899 |
| `model/common.py` | [model/common.md](model/common.md) | 326 |
| `model/sampling.py` | [model/sampling.md](model/sampling.md) | 150 |
| `plot_training.py` | [plot_training.md](plot_training.md) | 481 |
| `prepare_inputs.py` | [prepare_inputs.md](prepare_inputs.md) | 381 |
| `previews.py` | [previews.md](previews.md) | 606 |
| `train.py` | source header | 38 |
| `training/checkpoints.py` | [training/checkpoints.md](training/checkpoints.md) | 584 |
| `training/config.py` | [training/config.md](training/config.md) | 406 |
| `training/consumer_trace.py` | [training/consumer_trace.md](training/consumer_trace.md) | 492 |
| `training/engine.py` | [training/engine.md](training/engine.md) | 962 |
| `training/numerics.py` | source header | 79 |
| `training/resources.py` | [training/resources.md](training/resources.md) | 234 |
| `training/runtime.py` | [training/runtime.md](training/runtime.md) | 132 |
| `training/startup.py` | source header | 63 |
| `training/update_state.py` | source header | 93 |

## Entry scripts

Run every entry script from the `LTX-2` directory as
`python -m scripts.onestep_avatar.<module>`. The **Started by** column tells
you who starts it. "Queue" names the managed job kind; humans can also use
the [direct CLI commands](usage.md#human-direct-cli).
Each link goes to the Usage section of the module document.

| Module | What it does | Started by | Env |
|---|---|---|---|
| `corpus.precompute` | VAE-encode capture and guide masters | you, one free GPU | `ltx` |
| `corpus.build_guidance` | Render ARGAvatar guide videos | you, one free GPU | `argavatar` |
| `corpus.subset` | Write a fixed video list (membership) | you, CPU | `ltx` |
| `train` | Train an adapter | human CLI or queue `train` | `ltx` |
| `evaluate` | Generate and score with the corpus target | human CLI or queue `evaluate` | `ltx` |
| `infer` | Product generation from a supplied image | human CLI or queue `product` | `ltx` |
| `previews` | Fixed preview at a saved checkpoint | human CLI or queue `preview` | `ltx` |
| `bench` | Measure generation cost | human CLI or queue `bench` | `ltx` |
| `decode_saved` | Decode saved encodings to video | human CLI or queue `decode` | `ltx` |
| `comparisons` | Render saved comparison videos | human CLI or queue `render` | `ltx` |
| `media` | Prepare training reference pixels | you, one free GPU | `ltx` |
| `prepare_inputs` | Prepare preview inputs and supplied images | you, one free GPU | `ltx` |
| `plot_training` | Plot training curves | you, CPU | `ltx` |
| `execution.queue` | Run a job list | you, CPU owner process | `ltx` |
| `execution.queue_launch` | Close a ledger row after an owner ended | you, CPU | `ltx` |
| `experiments.<selector>` | One scientific comparison | human CLI or queue `experiment` | `ltx` |
| `experiments.prepare_continuation` | Plan and seal E3 inputs | you, CPU | `ltx` |
| `experiments.saved_probe_metrics` | Measure saved probe encodings | you, CPU | `ltx` |

Usage sections:
[precompute](corpus/precompute.md#usage),
[build_guidance](corpus/build_guidance.md#usage),
[subset](corpus/subset.md#usage),
[train](training/config.md#usage),
[evaluate](evaluate.md#usage),
[infer](infer.md#usage),
[previews](previews.md#usage),
[bench](bench.md#usage),
[decode_saved](decode_saved.md#usage),
[comparisons](comparisons.md#usage),
[media](media.md#usage),
[prepare_inputs](prepare_inputs.md#usage),
[plot_training](plot_training.md#usage),
[queue](execution/queue.md#usage),
[queue_launch](execution/queue_launch.md#usage).
Each experiment document has its own Usage section.

## Core logic review map

Read these documents to review the core logic:

- [Common model operations](model/common.md): tokens, the clean `c0`, noise and
  the full-frame loss.
- [Causal computation](model/causal.md): block 0, block 1, a mid-clip start,
  and how the cache keeps and refreshes entries.
- [Bidirectional computation](model/bidirectional.md): the whole segment, the
  clean first image, the target and the loss reduction.
- [Adapter application](model/adapters.md): the frozen bf16 base, the fp32
  PEFT tensors and the unmerged operation order.
- [Training engine](training/engine.md): which samples each rank sees, the
  update order, the distributed reduction and the exports.
- [Queue](execution/queue.md): GPU occupancy, process identity, startup and
  phase bounds, and completion.
- [Media](media.md): aligned RGB references, labeled panels, and the full and
  narrow reading formats.

## Artifact and evidence boundaries

The code checks the same identities at three points: before the run starts, at
the model call and at the saved completion check. The identities are the
content hash, schedule, geometry, noise, image, context and source.

- A changed file fails the check. Readers do not rebuild missing scientific
  files.
- Agents use the queue for native work. Humans can also use direct task CLIs.
  Only the queue produces external supervision and queue receipts.
- A child process records runtime receipts for each declared phase. These
  receipts do not replace the continuous supervision.
- The saved completion check verifies identity and structure. It opens no
  model.
- A report rebuild reads saved files only. It fails before it writes when a
  cited file is missing.

The [verification contract](verification.md) describes the controls and the
acceptance gates. No entry in this index claims native acceptance.
