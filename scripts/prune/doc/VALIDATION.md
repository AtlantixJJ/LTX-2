# Evidence and verification

A candidate needs attributable inputs, the declared computation, held-out quality and measured execution cost.
These checks answer different questions, so save each result separately.
The [glossary](glossary.md) defines terms; [overview](overview.md) and [algorithm](core_algorithm.md) explain the computation.

## CPU gate

Run from `LTX-2` using the `ltx` interpreter:

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES='' python -m pytest scripts/prune/tests -q -m 'not gpu'
ruff check --select F,I scripts/prune
```

Collection skips only tests already marked `gpu` when CUDA is unavailable or hidden.
CPU fixtures retain their imports and small-model behavior checks.
For docs, the contract checks source coverage, symbol/file references, glossary labels, placeholders, links, headings and CLI flags.
It also compares documented defaults and worked calculations with code and presets.
Render changed Mermaid diagrams and inspect readable wide/narrow versions.

## Evidence gates

### G1: reproduce the saved baseline

Reconstruct exact capture, saved epsilon, VAE, fps, geometry, text bytes and `[sigma,0]` schedules.
Use one bidirectional block, BF16 tensors, FP32 timesteps and clean c0.
Require finite outputs and finite differences, with maximum absolute latent difference at most **0.02**.
This is the default native replay and export-parity tolerance used throughout this package; a stricter frozen study can require more.
Compare saved noise tensors directly: a seed identifies the RNG request, not its resulting device-specific bytes.

The saved completion reader verifies hashes, coverage, finite BF16 tensors and c0 without opening a model.
The queued baseline wrapper additionally pins complete components and software.
Use its [reproduction recipe](usage.md#queued-baseline-reproduction) when the wrapper owns the task.

### G2: bind artifacts and held-out scope

Require the declared versioned candidate family, complete calibration-manifest content and actual source/noise/context pins.
Keep calibration actors outside held-out data across clips, Parts, views and aliases; reject a reused capture too.
Held-out sigmas must occur in calibration.
Keep pinned calibration rows in the selected baseline for distribution validation, then report held-out rows separately.
The saved analyzer visits every row, including calibration rows.
The [input contract](data/whole_clip.md) owns manifest and recovered-family dispatch.

### G3: reproduce the functional export reference

Use identical saved inputs for the in-memory intervention and exported model, loading one transformer at a time.
Use functional masks for width export and an ordered retained-block list for depth export.
Require matching shapes, finite outputs/differences and the G1 tolerance on every declared case.
Include a no-prune control and record relative L2 as supporting evidence.

Learned candidates need separate structural-transfer, adapter-transfer, fusion and merged-reload checks.
Their envelopes must cover retained IDs, consumed draws, components and merged tensors.
Use the split-bundle loader when adapters remain separate; the base checkpoint alone has different arithmetic.
A changed rounding point, normalization or fitted weight needs its own functional reference.
Depth metadata verification covers architecture and accounting; establish payload and native numerical agreement with their own checks.

### G4: assess held-out decoded quality

Predeclare acceptance thresholds, actors, seeds, exact schedules and coverage before blind evaluation.
Report generated-frame direction relative L2/cosine, capture latent MSE and candidate/baseline disagreement.
Exclude clean c0; it is supplied conditioning.
Latent agreement measures numerical change, not identity, action correctness or useful motion.

Inspect synchronized capture VAE, native D0 VAE and candidate D0 VAE videos for identity, detail, pose, motion and temporal consistency.
Include matched capture RGB when judging original capture fidelity, and label RGB versus VAE decoding.
Verify recorded crop, scale, matte, frame map and source hashes before interpreting alignment.
State sample size, selection rule and the clips actually inspected.
Required region metrics need reviewed GT tracks and available pinned LPIPS weights; missing required coverage fails acceptance.

### G5: measure execution cost

Use warmed, synchronized repeated forwards on the same GPU with fixed input, software, precision and numerical policy.
The baseline benchmark requires at least one warmup and three timed calls.
Use both ABA and BAB brackets when gains approach drift; repeat the outside arm to expose changing device conditions.
Claim a speedup only when it exceeds drift and timing spread and satisfies the frozen study's declared gate.

Report checkpoint bytes, resident-video parameter counts, peak allocated memory and forward latency with separate denominators.
A smaller checkpoint establishes storage savings; actual execution measurements establish latency and device-memory savings.
Include every required split-adapter or padded full-width GEMM in the timed computation.

## Learned native acceptance

Head feasibility covers both complete source lengths and every sigma in the frozen config.
The run guide's [preset section](usage.md#preset-values) owns the exact grid source and monitor indices.
Check native velocity conversion, wrapper parity, all-one gates, zero adapters and the distinct initialized sigmoid-gate student.
The bounded full-source feasibility phase uses ten backwards per length and its configured device-headroom requirement.

For uniform temporal crops, `checks.ddp_native` independently verifies full F18 and nonzero cropped-F28 inputs,
soft/binary gradients, frozen base weights, device headroom and serialized resume.
The receipt must match study identity, method and configured world size before sustained training.
Full-source monitoring capacity needs its own check because a crop fits less memory.

FFN and joint families run separate disposable-test and formal-rank acceptance through the shared owner.
Both exercise all-one parity, checkpointed backward, exact resume and fixed-mask recovery.
Match receipts to the frozen device list and rank count; a smaller test group cannot substitute for the formal group.
Joint optimizer replay, native channel ablations and export-isolation review remain separately pinned prerequisites.
The descriptive review names in the [state contract](learn/readmission.md) are distinct from G1–G5.

## Test ownership

- `tests/test_input_loading.py`, `tests/test_whole_clip_baseline.py` and `tests/test_whole_clip_d0.py` cover saved input and replay refusals.
- `tests/test_estimators.py`, `tests/test_projected_energy.py` and `tests/test_token_sampling.py` cover score meaning and sample coordinates.
- `tests/test_export_pruned.py`, `tests/test_export_depth.py`, `tests/test_safetensors_stream.py` and `tests/test_joint_exports.py` cover surgery and payloads.
- `tests/test_head_losses.py`, `tests/test_learned_gates.py` and `tests/test_head_adapters.py` cover loss, binary decisions and original-ID transfer.
- `tests/test_ffn_learning.py`, `tests/test_ddp_training.py` and `tests/test_readmission.py` compare numerical families and real CPU reducer gradients with serial controls.
- `tests/test_ddp_owner_loop.py` checks full-state resume, receipt digests, draw order, preview RNG and missing-input failure votes.
  Its fixture pins CPU reduction order; native NCCL process-restart equivalence remains a separate requirement.
- `tests/test_bench_learned_heads.py` checks nonfinite benchmark refusals and timing drift decisions.
- `tests/test_compact_recovery.py` checks aligned compact setup, fresh adapters, frozen weights and serialized Adam replay.
- `tests/test_doc_contract.py` checks the documentation contract and its intentional failure cases.
- `tests/test_device_policy.py` checks relocated disjoint groups, frozen rank counts, native capacity and physical-ID remapping refusals.

The [change-impact map](ARCHITECTURE.md#change-impact-map) selects checks for a particular edit.
CPU correctness checks complement native BF16 replay, decoded quality and timing evidence.

## Supported scope and persistent limits

Native pruning readers require white-capture, unguided, BF16, single-step D0 inputs.
The baseline producer has broader dev/guided/D1 options; verify reader support before using those outputs for pruning.
The registry includes multiple releases; scorer, functional ablation and baseline benchmark currently open LTX-2.5 sessions.

Prompt-cache names bind model key and prompt text, without a component hash.
Revalidate after changing the text encoder. Its CLI verifies `DEFAULT_PROMPT`; empty-text context uses the public API in [usage](usage.md#preflight-and-prompt-cache).
Native baseline rows use one seed per input root; create separate roots for additional seeds.
Raw-only evidence requires separately produced media for decoded comparisons.
Sampled checkpoint fingerprints identify a checkpoint; full-file hashes establish full-file integrity.

Depth export consumes a selected block set; this package supplies no automatic depth selector.
The scoring proxy and its spatial samplers have the limits defined in the [algorithm](core_algorithm.md#training-free-scores-and-masks).
Study-specific unrun checks, measured failures and quality/speed outcomes live in the [workspace handoffs](../../../../plans/README.md) and their saved reports.

## Comprehension review

After a method change, ask a fresh reader to use README, overview and algorithm alone to answer:
which branches change; how c0 stays clean; what the score measures and ignores; how export modes execute;
when units leave, return or become permanent; why score and gate moments reset; how selection ends and recovery runs;
and what evidence supports acceptance or speed.

Then check three maintenance tasks: write a launch command from usage, identify the input-builder change impact,
and explain a removal/return rule. Compare answers with source and record the review with the change.
This is a review activity; the CPU doc contract supplies repeatable checks between reviews.
