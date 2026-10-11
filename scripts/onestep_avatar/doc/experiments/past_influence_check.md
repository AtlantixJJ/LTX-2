# `experiments/past_influence_check.py` — independent changed-past observations

## Objective

The `past_influence` operation of the `conditioning_check` selector checks one
question: do independently changed generated past frames reach the native
history and affect the next block? A future-noise control cannot answer this
question. This owner calls the public causal operations and defines no
sampler. Its saved measurements prove input use, not identity or action
quality.

## Usage

Humans can use the foreground queue or run this module directly.
See [launch, output and stop commands](../usage.md). Agents use the queue.
The queue recipes below describe managed execution.

This module has no selector of its own. Run it as a queue `experiment` job
with selector `conditioning_check` and operation `past_influence`. See
[run guide §9, E3](../../configs/README.md#e3--causal-continuation-continuation_check-continuation_identity_check-causality).

```json
{"schema_version": 1, "protocol": "conditioning_check", "operation": "past_influence",
 "arguments": ["--snapshot", "/abs/generated_capture/continuation.json",
               "--reference", "/abs/generated_reference/continuation.json",
               "--sigma", "0.909375", "--history", "cache"]}
```

| Scientific argument | Meaning |
|---|---|
| `--snapshot` | The fresh generated-history `continuation_check` capture. |
| `--reference` | The matching `continuation_check` reference. |
| `--sigma {1.0,0.909375}` | The schedule. It must match both producers. |
| `--history {cache,recompute}` | The history calculation. |
| `--review-of <DIR>` | Decoder-only review of an already verified past output. |

- Both continuation producers must be complete, matching and declared queue
  dependencies.
- Run both schedules, and both `cache` and `recompute`. Each attempt covers the
  cases before and after eviction.
- For a decoder-only review job, add `--review-of /abs/completed_past_output`
  and declare that output as another dependency. It makes fresh exact-prefix
  RGB and VAE references and full and narrow diagnostic videos. It runs no
  other rollout.

## Data flow

The operation requires a fresh, complete generated-history
`continuation_check` capture and its exact reference.

1. Run both public saved scientific verifiers. Their sigma, protocol, source
   snapshot, fixed inputs and current software must match.
2. Both producers contain the pre-eviction block `[3,5)` and the
   after-eviction block `[11,13)`, in the same 17-frame B2/D8 case.
3. The source manifests, specs and records stay byte-pinned dependencies. The
   original corpus and producer data do not change.

Each source snapshot contains the full generated clean history, the current
noisy block, the capture, the guide, the full noise, `c0`, the text, the native
positions, the timesteps and the K/V of every retained layer.

```mermaid
flowchart LR
  S[("Fresh prefix + fixed noise")] --> O["Public history replay"]
  O --> E("Exact prediction + every layer K/V")
  E -->|"negate past frames; c0 held"| C["Changed public history replay"]
  C --> M(["Saved input, K/V and output differences"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef tensor fill:#dff2e0,stroke:#43834a,color:#16341b;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class S disk;
  class O,C proc;
  class E tensor;
  class M out;
```

## Organization logic

### Original replay: cache mode

For each target, the operation first rebuilds the original calculation.

1. Start with an empty cache.
2. Call `causal.refresh_block` on every completed block, in time order. Every
   refresh uses the saved generated clean block, writes at sigma zero and calls
   the public eviction policy.
3. The original denoise forwards were read-only and wrote nothing to the cache.
   So this replay keeps the cache transaction order exactly. It never calls the
   approximate `prime_cache` operation.

**Memory layout.** The snapshot recorder copies token values into one
full-span history buffer. Its narrow views do not keep the memory layout of the
original refresh block. So the replay restores the layout:

- The first block is contiguous, as from the rollout's clean-`c0` `torch.cat`.
- Later blocks use a preserve-format clone. This packs the channel-major slices
  of the saved buffer into the original dense channel-major blocks (or keeps
  dense row-major blocks in the tiny fixture).

This restoration keeps every value exactly. It applies equally to the original
and changed arms. Native GEMM rounding can depend on strides, so equal values
alone do not justify replaying a padded view. Equality of all-layer K/V and of
the prediction stays mandatory. No tolerance is added.

Before denoising, the operation streams and compares the actual cache of every
rebuilt layer with the capture's saved K/V, bit for bit. `causal.denoise_block`
must also reproduce the saved cached prediction exactly. The original cache
tensors reuse the current verified source file only after the live all-layer
comparison. The new record includes the independently observed tensor hashes.

### Original replay: recompute mode

Recompute mode calls `causal.denoise_with_clean_history` with the saved
original history and the current noisy block. Its original prediction and the
native history K/V of every layer must equal the fresh reference producer
exactly. The `ReferenceObserver` of the frozen continuation owner delegates
each native attention function transparently. It observes the history K after
normalization and RoPE, and the native V. It does not change q, k, v, masks or
the attention result.

### Changed arm

The changed arm runs only after the original prediction and the all-layer
rebuild pass.

- It negates the completed history frames `[1, target_start)`.
- It keeps frame zero, all current and future bytes, the guide, the current
  noise, the text, the positions, the schedule and the geometry.
- The same public cache or recompute operations consume this deliberately
  mismatched past.
- It saves the actual all-layer K/V and the current-block prediction.

The differences are recorded separately for each calculation and each target.
Do not compare cache with recompute as the changed factor of this
intervention.

### Functions

- `source_records` verifies both producers and their exact lineage before any
  native work.
- `changed_history` owns the one data intervention.
- `run_arm` calls the public causal operations. It records native inputs
  through a public PyTorch transformer pre-hook. Cached K/V comes through public
  cache reads. Recomputed K/V uses the existing transparent continuation
  observer. No observer changes a numerical input.
- Each target and arm saves the prediction, the input trace and all-layer
  observations. The trace records the actual latent, the global sigma, the
  token timesteps, the positions, the keyframe mask, the causal mask, the text
  hash, the cache read and write boundary and all layer lengths.
- `verify_trace` rebuilds every expected refresh and denoise input from the
  saved source. It checks the full transaction order, including the sigma-zero
  refresh payload and the retained context after eviction.
- `verify` reads saved bytes only. It repeats the source scientific
  verification. It checks the native coordinate constructions without
  tolerances. It verifies both original predictions and every original layer
  against their saved producers, and recomputes every difference. Every
  observed history layer and the current prediction must change in the
  intervention.
- `execute` opens one base dev LTX-2.5 native session, gives it the saved text
  directly, and checks free disk space before it writes full layer
  observations. The selector publishes its manifest only after saved
  verification.

A missing boundary or layer, an unchanged intervention, source or profile
drift or a corrupted input fails.

Worked example:

- Before `[3,5)`, the replay refreshes `[0,3)`, then denoises the fixed current
  block.
- Before `[11,13)`, the replay refreshes five completed blocks. It keeps `c0`
  and eight later frames before the fixed current block.
- For each arm, cache makes two and six forwards. Recompute makes one for each
  target. So one operation makes 16 actual cache forwards or four recompute
  forwards.

Original byte equality is a prerequisite. A large difference in the changed
arm cannot make up for a failed original rebuild.

### Decoder-only review

The optional `--review-of <completed-past-output>` selects a separate
decoder-only attempt through the same frozen operation. Its dependency list
adds the exact past producer.

1. The saved native verification must pass first. The review pins its result
   and queue manifest.
2. Its decoding profile includes the frozen `reference_inputs` helper as an
   operation-specific source.
3. It checks the original membership through the full native source verifier
   and the public `ClipStore.verify`. Then it derives a copied RGB source
   descriptor through `reference_inputs.checked_source`. The original
   membership and crop bytes stay unchanged.
4. The current VAE fingerprint must equal the original capture encoder.

**Preparation reuse.** `execute_review` gets one fresh `review_preparation`
before the decoder opens. It keeps that verified input object in the same
invocation. Then it calls the private `_verify_review_prepared` to check the
saved decoder outputs: all presentation latents, RGB identities, reference
attribution, labels, layouts, frame maps, videos and posters. It does not
repeat the completed native prerequisite calculation inside that child. The
prepared object is not saved or exposed as a verification option.

The public `verify_review(args)` always gets a new `review_preparation` before
it calls the same private output checker. So standalone and queue completion
checks repeat the full native prerequisite verification independently of the
child. They cannot reuse its prepared object. Source software checks, the exact
original replay, all-layer checks and the declared budgets do not change.

**Presentation latents.** For each boundary, the review joins the saved
generated prefix and that arm's exact current prediction. It unpatchifies them
through the public grid and saves the exact presentation latent. Fresh public
VAE decoding makes lossless RGB tensors. Public reference preparation also runs
fresh at each exact prefix length, five or thirteen encoded frames. It does not
slice a longer VAE decode. Recompute uses the saved cached generated prefix
with a recomputed current prediction, and its caption states this origin.

**Video layout.** Each synchronized video shows Capture RGB, Capture VAE, Guide
RGB, the original current-block diagnostic and the changed current-block
diagnostic.

- The full view has five panels of equal height in two rows of three.
- The narrow view uses the shared compact layout, with the original and changed
  panels next to each other. It can stack panels to keep the exact labels
  readable.
- The titles give the actual encoded current interval. The caption records its
  RGB start, the schedule, the history, the fixed current noise and the
  artificial negated past.
- The source map is exactly zero through 32 before eviction, or zero through
  96 after eviction, at 30 fps.
- The original value line is `Diagnostic; VAE decoded`. The changed value line
  is `Negated past; VAE decoded`. Both must fit the actual public full and
  narrow layout geometry.
- The poster is the final frame of the current block. Public scaling keeps the
  aspect ratio.

Saved verification rebuilds the presentation latents, the reference crop and
pixels, all decode keys, labels, frame maps, full and narrow layouts, video
dimensions and poster pixels. It opens no model or decoder.

## Invariants

- Source members, source images and baseline results do not change.
- The current noise, guide, text, `c0`, positions, sigma and B2/D8 stay fixed.
- The original prediction and every original layer require exact equality.
- The cache replay keeps every original refresh and eviction transaction. There
  is no priming.
- Both targets and every native layer are required.
- A decoder verifies the native prerequisites before its session opens.
  Prepared inputs are reused only within one child. Public saved verification
  prepares again.
- Source and owner pins are operation-specific. The other conditioning,
  reference and capacity profiles do not include this helper.
- Generic queue supervision limits the whole operation. Observations are
  diagnostic cost, not warmed deployment latency.

## Gotchas

- Negated past latents are a mismatched input control. They are not a real
  second action or a perceptual-quality reference.
- Before and after eviction use different retained histories. But each arm
  pair keeps its current block fixed.
- Cache history was written at sigma zero. Recomputed history uses zero token
  times and the current global sigma. These stay separate calculations.
- Full native K/V can use many gigabytes. Layers are read and compared one at a
  time. Only one GPU cache is resident.
- Native CUDA and CPU timestamp division can differ by one float32 ULP. Save
  the actual native full grid, require exact public CPU or CUDA arithmetic and
  bind every observed call to that saved grid.
- The deadline of the whole attempt includes the queue's saved completion
  checks after the decoder exits. Less repeated child preparation does not
  prove that the full attempt meets its limit. Fresh native and decoder
  evidence must prove it (see
  [known gaps](../known_gaps.md#changed-past-decoder-total-deadline)).

## Tests

- Tiny real-transformer tests cover both sigmas, both history calculations,
  both eviction targets, original all-layer and prediction equality and
  changed-past effects.
- Channel-major saved histories also test the original `c0` concatenation and
  the packed strides of later refreshes. They bind the original replay to those
  layouts without changing token values.
- Negative tests change `c0`, past content, current noise, positions,
  timesteps, text, cache lengths, transaction order and an original layer.
- Source tests refuse a different sigma, history or snapshot, and missing
  targets.
- Saved-only checks block model and session loading. They refuse changed
  predictions, actual inputs, K/V bytes, measurements and position grids.
- Dispatcher tests check the literal operation and the exact two producer
  dependencies.
- Decoder tests require separate five- and thirteen-frame reference requests
  and matching native decode calls. They verify the exact supplied past and
  current latents. They refuse a changed latent, RGB bytes, source map, crop or
  poster. The shared media owner gives real layout tests. Local decoder tests
  use a bounded pixel fixture and never start CUDA.
- Decoder lifecycle tests refuse a failed native preparation before a decoder
  can open. They require one preparation for each child. They refuse corrupt
  saved outputs, even with that prepared input. Standalone verification must
  refuse a changed native prediction after decoding. An earlier child check
  cannot supply that proof.
- Real public geometry tests cover the complete reference and diagnostic labels
  at both boundaries, five panels in the full two-by-three grid, readable
  narrow titles, unchanged frame maps and adjacent original and changed panels.

CPU checks do not prove native or perceptual acceptance.
