# Verification — worked checks and acceptance gates

This file defines checks across modules. It is not a document for one source
file, so the 100-line rule does not apply. Read the
[symbols](core_algorithm.md#1-symbols) first.

- V1–V8 are design checks with worked numbers. The CPU test suite covers each
  of them.
- The acceptance gates E1–E5 define the native evidence.

A passing CPU check never proves a native numerical result, video quality,
speed or memory use.

## V1 — Is the first image a clean model input?

Use one number per encoded frame:
capture `[10,20,30]`, guide `[1,2,3]`, random noise `[-1,0,1]`, and sigma `.5`.

D1 noise mixing gives `[0,1,2]`.
Replace the first value with capture `c0=10`.
The model receives `[10,1,2]`, token noise levels `[0,.5,.5]`, and whole-model sigma `.5`.

D0 noise mixing gives `[4.5,10,15.5]`.
The model receives `[10,10,15.5]` after first-image replacement.

**Result:** both inputs contain the unchanged first-image value `10` at token noise level zero.
Replacing only the output would fail this input check.
**Tests** inspect the actual model inputs, output and refresh input for both
backgrounds and both causal history policies.

## V2 — Do D0 and D1 agree when the source weight is zero?

At sigma one, `(1-sigma)*source+sigma*epsilon` equals `epsilon`.
With the same weights, adapter, text, first image, noise, frame layout and history,
D0/D1 submit identical model inputs. In V1, both inputs are `[10,0,1]`.

**Result:** deterministic predictions agree.
This does not compare different adapters or different noise or history.
**Tests** check bit-identical inputs and deterministic CPU output; E5 repeats
the control on the real model.

## V3 — Is the loss scale unchanged?

Prediction `[10,18,32]` against capture `[10,20,30]` gives squared errors `[0,4,4]`.
Full-frame MSE is `8/3`. The unchanged first frame stays in the denominator.

For two causal blocks with MSE `2` and `6`, the sample loss is `(2+6)/2=4`.
With two accumulated samples, backpropagate each block's MSE divided by `K*A=4`.

**Result:** gradients represent the mean of sample block means.
Different block sizes do not change this to a token-weighted loss.
Bidirectional processing uses one segment.
**Tests** compare losses and gradients and check that the engine does not divide again.

## V4 — Are cache positions and removal correct?

Use five encoded frames, block length two, stored-history limit one, and a pinned first frame.
The block ranges are `[0,3)` and `[3,5)`.

| Step | Cached frame data before denoising | Current frames | Cached frame data after refresh |
|---|---|---|---|
| Block 0 | empty | frame 0 unchanged; frames 1,2 noisy | 0,2 |
| Block 1 | 0,2 | frames 3,4 noisy | 0,4 if refreshed |

**Result:** block 1 sees first-image data and the last stored past frame.
Cache entries keep original positions 0,2; they are not relabeled 0,1.
Training skips its last refresh.
A seven-frame video adds `[5,7)`, which sees cached data for frames 0,4.
At history limit zero, keep only frame zero.
Write capacity is `min(F,1+D+1+B)=5` frames.

**Tests** inspect layer-cache data, counts and positions before and after
removal, including zero history and first-frame retention.

## V5 — What changes when training starts at a later block?

Starting at block 1 primes from capture frames 0 and 2.
Use one call without gradients, whole-model/token sigma zero, and original positions/block IDs.
Then add noise to input frames 3,4. This applies even to self-forced training.

**Result:** priming supplies capture past frames that product generation cannot provide.
Training from the video start makes one discarded no-cache priming call and leaves history empty.
**Tests** inspect priming input data, positions and levels.

## V6 — Do GPU processes make the same calls?

For causal `K=3`, each process makes one prime, three denoises and two refreshes:
six explicit model calls and three immediate backward calls.
The counts hold for sequences starting at frame zero or a later block.
Two accumulated samples require twelve model calls and six backward calls.
Two bidirectional samples require two model calls and two backward calls.

**Result:** reject mismatched K, sample-group length or mode plans before distributed model calls.
Equal counts are necessary but not sufficient; checkpointed recomputation,
reductions and saves also need compatible order. E4 compares the first
distributed update against a fixed serial reference.

## V7 — Does an incompatible request fail before model loading?

An example adapter records bidirectional mode, dev weight hash A, white D1 and direct sigma `.5`.
A causal request differs in mode. Distilled hash B differs in base weights.
Sigma `.7` differs unless the adapter's level list includes it.
Schedule `[.5,.25,0]` differs from direct one-step training.
An adapter without the typed contract is refused.

**Result:** the checker names differences before loading 22B weights.
Evaluation records explicit overrides; product generation rejects them.
Evaluation can use people excluded from training.
**Tests** (`tests/test_checkpoint_contract.py`, `tests/test_evaluate.py`,
`tests/test_infer.py`) use checker cases and a loader sentinel; rejected
requests load no weights and write no output directory.

## V8 — Does the membership identify unchanged inputs?

Survey checked crop records and stored masters. Pin exact video IDs,
background, person groups and selected producer bytes. Membership creation
writes only a new list; the settings owner selects encoded frames separately.

**Required checks:** deterministic person grouping and exclusions, actual
stored master coverage, capture/guide provenance, input-change detection and
refusal to overwrite a destination. Relocation preserves the membership hash.
Missing D1 guides fail before model loading. `tests/test_subset.py` covers
these cases; frame-plan tests verify mode-specific ranges without changing
membership or producer data.

## Repository ownership check

Training, generation, evaluation, reusable metrics and decoding execute in LTX-2.
Code under `expr/` only builds reports from saved results.

Worked check: save checkpoint 100, its metrics, preview video, poster and input
records. The report generator reads those saved files and builds the report
with training, transformer and VAE loaders disabled. Remove one required
preview from the fixture and rebuild. The expected result is a named
missing-file error and zero model or decoder jobs; the report never recreates
the preview.

## Acceptance gates

Native acceptance uses the [run guide](../configs/README.md#9-experiments)
recipes, fresh output directories and the current source. A result keeps the
source and inputs that made it. A changed producer needs a fresh check of the
affected results. Each gate states its scope. Numerical agreement does not
prove appearance quality.

### E1 — Supported sampler (`experiments/stock_parity.py`)

Compare the native stock video sampler with the package sampler. Keep these
items the same: weights, first image, text, saved noise, geometry, fps,
schedule and guidance calls.

- Build stock schedules with `LTX2Scheduler().execute(steps=N)` and no latent
  argument.
- Compare the raw encodings, the intermediate calls, repeats and the decoded
  17-frame (129 RGB) output with a declared dtype tolerance.
- The direct `[sigma, 0]` endpoint differs from the stock bf16 endpoint by
  terminal rounding
  ([known gaps](known_gaps.md#direct-endpoint-and-stock-terminal-rounding)).

Pure-noise parity does not prove D1 or video quality. Outer stock
preprocessing and joint audio-video are outside this video-only check.

### E2 — Learned adapter function (`experiments/adapter_effect_check.py`)

In both modes, compare three paths: the training reference, ordinary
evaluation and the product. Use the same `c0`, guide, text and noise. Compare
three model states: the base, a zero adapter and the trained unmerged adapter.

- Report the correction RMS, the absolute and relative gaps, the raw output,
  the decoded appearance and the measured cost.
- The relative effect L2 must be **under 0.05**. When the reference RMS is
  below `1e-8`, the absolute difference RMS must be **at most `1e-8`**.
- A zero adapter equals its own base.
- Complete **two camera views and trained steps 20 and 60**.

Near-zero agreement allows the bounded pilot. It does not prove a learned
effect. Fused bf16 application is a separate diagnostic.

### E3 — Causal continuation (`experiments/continuation_check.py`, `experiments/causality.py`)

Require an exact same-input cached repeat, and independent `c0`, prefix and
future-noise controls. Compare the cache refresh with block-causal
recomputation. Keep fixed: weights, adapter method, source, text, saved noise,
schedule, B and D, and coverage.

1. Use **17 encoded frames**. Separate the blocks before and after eviction at
   D8. Keep the after-eviction block `[11,13)`.
2. Start with one view at direct `[1,0]` and `[0.909375,0]`. Save the block
   output and the layerwise K/V.
3. Keep capture-history diagnostics separate from generated-history
   comparisons.
4. Measure seams, identity, motion, calls, allocated and reserved peaks and
   warmed latency. Review the matched capture RGB and VAE reconstruction.
5. Then complete **three held-out people, two seeds and one matched view
   each**.

The seven-frame short controls do not replace the eviction or quality
acceptance.

### E4 — Distributed training update (`experiments/training_update_check.py`)

In each mode, compare the deterministic first update (four ranks, accumulation
two) with a fixed serial reference. Use the same visits, text, noise and
configuration. Compare:

- 768 named gradients, the Adam moments and the exports;
- the loss and the norm, a zero no-op and the actual step-one reload.

The relative L2 limit is **0.02**. The near-zero and absolute RMS limit is
**`1e-8`**.

Causal samples keep block-zero and later starts, equal K3, block sets
`[0,1,2]` and `[3,4,5]`, refresh and eviction, and checkpoint recomputation.
Also check:

- canonical launch reconstruction, the actual world size and dtype agreement;
- resource evidence for each rank;
- negative controls for a changed launch, budget and source.

Gradients with a norm below one prove that clipping is a no-op. They do not
prove active clipping.

Complete **one fixed base/step-one preview and one public supplied-image
product for each mode**. Each must show `c0`, finite outputs, call counts,
generated history, absent workers and synchronized full and narrow media.
`experiments/training_slice_check.py` finds a mismatch on one GPU. It cannot
complete this gate.

### E5 — Bounded learning, controls and cost

Fixed settings for both modes:

- the same **two checked clips**, with a separate run for each mode;
- D1, `white`, dev LTX-2.5, direct **`[0.725,0]`**;
- seven encoded frames (49 RGB) and clip-start `c0`;
- LoRA rank and alpha 8 on attention, fresh noise and seed 42;
- learning rate `1e-4`, a 20-update warmup and norm-one clipping;
- four ranks with one sample per rank for each update;
- causal: B2/K3/D8 with generated history and sigma-zero refresh.

Save updates **0, 20 and 60**. Stop at **60 updates or 7,200 seconds overall
for each mode**, whichever comes first. Report incomplete coverage without a
change of settings.

Then do these comparisons:

1. Compare the fixed base, zero, 20 and 60 states.
2. Prove **D0 ≡ D1 at sigma one with the same model, adapter, `c0`, noise,
   text and history**.
3. Run wrong-guide and wrong-text controls. Judge identity and action from
   matched videos.
4. Measure the separate D0 capacity question.
5. Complete **three held-out people × two seeds**, and the warmed latency and
   memory.

Latent MSE or reduced motion alone cannot prove quality. Do not tune on the
final test set. Do not claim longer coverage from this pilot.

## Limits of these checks

Arithmetic and frame traces do not establish video quality, speed, memory use or
equality of implemented code. The cache-refresh, fused-adapter, random-start
first-image and terminal-rounding differences in [known gaps](known_gaps.md)
remain until accepted evidence resolves them. Diagrams do not prove numerical
or distributed behavior.
