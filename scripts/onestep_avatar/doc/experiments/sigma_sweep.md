# `experiments/sigma_sweep.py` — decode and score a sigma sweep

## Objective

Decode and score the generated outputs of a result-bound sigma sweep: four
sigma levels by two arms (D0, D1) plus the capture and guide masters. Open one
decoder-only session after checking every input. Never regenerate a rollout or
prompt. Report sheets are assembled separately from the saved samples.

The queue runs this module as the `sigma_sweep` experiment. `parse_args` is the
shared argument owner for the CLI and the queue; parsing alone performs no
decoding or GPU discovery. Its entry calls the shared
[runtime](../execution/runtime.md) helper. `--verify` runs the saved-output
verifier with the same spec and output arguments, rejects `--gpu-id` and opens
no session.

## Data flow

The input spec (`schema_version: 2`) names the capture and guide masters with
their hashes, model `2.5`, decode seed 42, fps 30 and the VAE path and hash. Each
of the eight cells embeds its evaluation job as `evaluation_job`.
`sigma_sweep_results.resolve_cells` requires the exact causal, distilled,
cached-refresh, unguided 17-frame setup and the same source, prompt, seed and
saved noise in all eight jobs; job arguments must name the spec's pinned
masters. It verifies each job's queue scientific completion before it resolves
`generated.pt` and its hash; a missing result fails and is never regenerated.
Result, text, noise, membership and original noise-file bytes are bound before
verification, rechecked after it and included in the decoder's input binding.
The output manifest keeps the embedded jobs and adds resolved tensor
identities without rewriting the source spec.

Decode ten matched 17-frame tensors, 129 RGB frames each. Save ten videos,
selected PNG frames, `metrics.json` and a final manifest. A report reads those
artifacts.

```mermaid
flowchart LR
  results[(Verified generation results)] --> resolve[sigma_sweep_results.resolve_cells]
  resolve --> preflight
  queue[Package queue: pinned spec and shared claim] --> preflight
  spec[(Checked sweep spec)] --> preflight[prepare]
  preflight --> decode[media.decode]
  decode --> pixels(Floating RGB)
  pixels --> score[sigma_sweep boundary metrics]
  pixels --> samples[(Saved videos and samples)]
  score --> metrics[(Saved metrics)]
  samples --> report([Report sheets])
  classDef code fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef tensor fill:#d9f7e7,stroke:#20804b,color:#102c20;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class queue,resolve,preflight,decode,score code;
  class results,spec,samples,metrics disk;
  class pixels tensor;
  class report out;
```

## Organization logic

`parse_args(argv)` returns parsed values plus the queue completion descriptor
`{"manifest": "<output>/manifest.json"}`. The decoding software profile stays
ordinary; this experiment adds its own module, the experiment package marker
and `sigma_sweep_results.py` through `extra_sources`. The shared decoding
software manifest is captured before input preparation, checked before the VAE
opens, before media writes and before final publication, and saved in the
manifest.

The cells are sigma 0.421875 (`one_step`, one call), 0.725 (official, two
calls), 0.909375 (official, three calls) and 1 (official, eight calls), each
with D0 and D1, in this order. Masters may be longer than 17 frames; generated
tensors contain exactly 17. Require matching finite bf16 B,C,F,H,W shapes after
master selection and matching recorded fps. Before a decoder opens, require a
fresh output, checked input hashes and the registered VAE path and hash. Each
input decodes through `media.decode` with a new seed-42 generator; no text
context or transformer is created.

Convert decoded FCHW to floating FHWC, dividing byte-range pixels by 255, and
require exactly 129 aligned RGB frames. Form one mask from the union of
min-channel below 0.9 in capture, guide and all eight outputs. Call
`sigma_sweep_boundary_metrics` and `metrics.masked_rgb_transition_steps` on the
uncompressed floats. Record the tag, boundary and eviction metadata, `c0`
equality against sigma-one D0, and sigma-one D0/D1 equality and maximum delta.

`sigma_sweep_boundary_metrics` lives here because its fixed 129-frame
inventory, seven boundary locations, four-neighbor references and
post-eviction attribution belong to this study; the general aligned-mask
transition operation stays in `metrics.py`. Zero denominators produce explicit
undefined statuses; arrays use four decimals.

Save each video at 30 fps and PNG samples at frames 0, 16, 17, 48, 49, 63, 64,
65, 66, 96, 97 and 128. PNGs use `media.frame`'s rounded quantization; scores
are calculated before any quantization. Record sample indices, media hashes,
pixel digests, decode settings, input and spec hashes and source hashes. Recheck
inputs, VAE and sources before atomically publishing the manifest. A partial
directory is not complete and is never reused.

Before publication, probe each MP4 for 129 frames at 30 fps and the declared
size, and check all twelve PNG sizes per role. `verify_completion` reconstructs
the expected spec, input, VAE and source identities without a model. It
requires the exact ten-role inventory, contained paths, video and PNG hashes,
sample order, actual frame counts, timebase and size, pixel digest syntax and
the metrics-file hash; it checks cell identities and call labels and
recalculates the `c0` and sigma-one controls from the saved tensors. Scores stay
producer measurements and are not recomputed from MP4 or PNG. The report reader
calls this verifier before loading samples.

## Invariants

All inputs decode in one session with the same settings and seed. Missing,
changed, short, nonfinite or mismatched inputs fail before a session or output
exists. Old outputs are kept; a rerun uses a fresh destination. A report rebuild
cannot start decoding or repair missing samples. This is saved decoding only;
generation, adapter quality and native parity need separate evidence.

## Gotchas

Capture and guide paths come from saved producer manifests, not actor names or
guessed corpus roots; a spec must match their recorded hashes. A VAE mismatch
refuses the comparison. A spec without embedded evaluation jobs is refused.

## Tests

- Eight real small CPU causal evaluations across all levels and arms verify
  sigma-one equality and resolve ten tensors without sessions or writes; changed
  seed, arm, paired masters, noise, result metadata, saved text and late
  evidence fail. Decoder readiness requires all eight unchanged receipts.
- A controlled decoder with real small media writes checks seed reset, raw
  scores, cell identities, `c0` controls and output hashes; missing or changed
  cells, geometry, fps, VAE, nonfinite tensors and used outputs fail before
  model work, and mutating an input during decoding blocks publication.
- The complete-movie mutation control changes saved movie bytes while every
  poster PNG stays identical and requires rejection; restoring the movie passes.
- A schema-one tensor spec fails before a decoder opens.
