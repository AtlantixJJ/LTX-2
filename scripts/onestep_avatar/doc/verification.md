# Cross-module checks and acceptance limits

Status: **Design arithmetic checked; shared implementation has scoped CPU/native evidence.**
This file describes checks across modules.
It is not a per-source-file doc, so the 100-line rule does not apply.
Read the [symbols](core_algorithm.md#1-symbols) first.

The equations and frame traces in V1–V6 were checked mechanically.
V7–V8 define condition and conversion checks. Current CPU tests cover these
paths. Their existence does not certify the full native handoff. The
[implementation ledger](../../../../plans/assets/2026-10-07-onestep-avatar-implementation-ledger.md)
records current executed evidence and remaining native scope.

## V1 — Is the first image a clean model input?

Use one number per encoded frame:
capture `[10,20,30]`, guide `[1,2,3]`, random noise `[-1,0,1]`, and sigma `.5`.

D1 noise mixing gives `[0,1,2]`.
Replace the first value with capture `c0=10`.
The model receives `[10,1,2]`, token noise levels `[0,.5,.5]`, and whole-model sigma `.5`.

D0 noise mixing gives `[4.5,10,15.5]`.
The model receives `[10,10,15.5]` after first-image replacement.

**Design result:** both inputs contain unchanged first-image value `10` at token noise level zero.
Replacing only the output would fail this input check.
**Planned test:** inspect actual model inputs, output, and refresh input.
Test both backgrounds and both causal history policies.

## V2 — Do D0 and D1 agree when source weight is zero?

At sigma one, `(1-sigma)*source+sigma*epsilon` equals `epsilon`.
With the same weights, adapter, text, first image, noise, frame layout, and history,
D0/D1 submit identical model inputs.
In V1, both inputs are `[10,0,1]`.

**Design result:** deterministic predictions agree.
This does not compare different adapters or different noise/history.
**Planned test:** inspect bit-identical inputs and deterministic CPU output.
Run the established real-model control with matching settings.

## V3 — Is the loss scale unchanged?

Prediction `[10,18,32]` against capture `[10,20,30]` gives squared errors `[0,4,4]`.
Full-frame MSE is `8/3`. The unchanged first frame stays in the denominator.

For two causal blocks with MSE `2` and `6`, sample loss is `(2+6)/2=4`.
With two accumulated samples, backpropagate each block's MSE divided by `K*A=4`.

**Design result:** gradients represent the mean of sample block means.
Different block sizes do not change this to a token-weighted loss.
Bidirectional processing uses one segment.
**Planned test:** compare old/new losses and gradients.
Check that the engine does not divide again.

## V4 — Are cache positions and removal correct?

Use five encoded frames, block length two, stored-history limit one, and a pinned first frame.
The block ranges are `[0,3)` and `[3,5)`.

| Step | Cached frame data before denoising | Current frames | Cached frame data after refresh |
|---|---|---|---|
| Block 0 | empty | frame 0 unchanged; frames 1,2 noisy | 0,2 |
| Block 1 | 0,2 | frames 3,4 noisy | 0,4 if refreshed |

**Design result:** block 1 sees first-image data and the last stored past frame.
Cache entries keep original positions 0,2; they are not relabeled 0,1.
Training skips its last refresh.
A seven-frame video adds `[5,7)`, which sees cached data for frames 0,4.
At history limit zero, keep only frame zero.
Write capacity is `min(F,1+D+1+B)=5` frames.

**Planned test:** inspect layer-cache data, counts, and positions before and after removal.
Include zero-history and first-frame retention cases.

## V5 — What changes when training starts at a later block?

Starting at block 1 primes from capture frames 0 and 2.
Use one call without gradients, whole-model/token sigma zero, and original positions/block IDs.
Then add noise to input frames 3,4.
This applies even to self-forced training.

**Design result:** priming supplies capture past frames that product generation cannot provide.
Training from video start makes one discarded no-cache priming call and leaves history empty.
**Planned test:** inspect priming input data, positions, and levels.
Compare old/new code under the same approximate priming rule.

## V6 — Do GPU processes make the same calls?

For causal `K=3`, each process makes one prime, three denoises, and two refreshes.
That is six explicit model calls and three immediate backward calls.
The counts hold for sequences starting at frame zero or a later block.
Two accumulated samples require twelve model calls and six backward calls.
Two bidirectional samples require two model calls and two backward calls.

**Design result:** reject mismatched K, sample-group length, or mode plans before distributed model calls.
Equal explicit counts are necessary but not sufficient.
Repeated checkpointed calculations, reductions, and saves also need compatible order.
**Required implementation check:** record call order and compare the first
distributed update against the fixed serial reference. The initial native
bidirectional comparison failed. The root-input precision correction requires
fresh four-process acceptance in both modes. See G12 and the ledger; do not
treat compatible call counts or exact adapter reload as E4 acceptance.

## V7 — Does an incompatible request fail before model loading?

An example adapter records bidirectional mode, dev weight hash A, white D1, and direct sigma `.5`.
A causal request differs in mode.
Distilled hash B differs in base weights.
Sigma `.7` differs unless the adapter's level list includes it.
Schedule `[.5,.25,0]` differs from direct one-step training.
An old adapter without mode metadata needs explicit classification.

**Design result:** the checker can name differences before loading 22B weights.
Evaluation records explicit overrides; product generation rejects them.
Evaluation can use people excluded from training.
**Required implementation check:** use checker cases and a loader sentinel.
Rejected requests must not load weights or archive an output directory.
Current CPU coverage includes `tests/test_checkpoint_contract.py`,
`tests/test_evaluate.py` and `tests/test_infer.py`. Full native adapter-function
and distributed-update acceptance remain E2/E4.

## V8 — Does the new video list preserve old data?

Copy exact video IDs, background, person groups, and file hashes from an old subset.
Write a new versioned video list.
Keep the original hash and exact old block sequences in a separate reproduction frame plan.

**Design result:** this schema change need not change people, encoded data, or selected samples.
The mode can select frames without rebuilding media or crops.
**Required implementation check:** compare full video lists, hashes, and reconstructed frame ranges.
Reject missing D1 guides before model loading.
Keep original subset files unchanged.
`tests/test_subset.py` covers exact original people, hashes and frame ranges,
changed bytes/coverage, D0 without guides, and refusal to overwrite destinations.
The direct version-two corpus survey is still proposed; conversion is implemented.

## Repository ownership checks

Training, generation, evaluation, reusable metrics, and decoding execute in LTX-2.
Code under `expr/` only generates reports from saved results.
Study settings, narratives, runs, metrics, and media can remain under `expr/` as data.
Package code must not import executable study code.

Worked design check: save checkpoint 100, its metrics, preview video, poster, and input records.
The report generator reads those saved files and builds the report.
Disable training, transformer, and VAE loaders during the rebuild.
Remove a required preview from a test fixture and rebuild again.
The expected result is a named missing-file error and zero model/decoder jobs.
The report must not recreate the preview or start an `expr/` execution wrapper.

After implementation, audit avatar `expr/` imports and subprocess commands.
Check required training/evaluation queues and launchers have package owners.
Check transferred code uses matching docs or small-file headers.
Check retired `expr/` executor docs and forwarding wrappers are removed.
Rebuild existing reports from saved results with model executors disabled.
Full ownership closure remains a Stage D gate. The historical training report
has a checked saved-only publication guard; other report families and import
restoration paths need their own controls. One verified family does not certify
the workspace.

## Limits of these checks

Arithmetic and frame traces do not establish video quality, speed, memory use, or implemented-code equality.
G7 cache-refresh differences, G8 bf16 LoRA fusion, and accepted G9 first-image differences remain.
After review, run deterministic CPU tests, native saved-input checks, a stock-pipeline comparison,
and small distributed updates.

Diagram review and Markdown-change checks are recorded in [README](README.md).
