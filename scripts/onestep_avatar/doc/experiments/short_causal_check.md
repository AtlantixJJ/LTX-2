# `experiments/short_causal_check.py` — exact short causal controls

## Objective

Check three things through the shared causal evaluation and product functions:

- repeated generation gives identical output;
- the supplied first image is kept;
- earlier output does not depend on future noise.

The check uses one accepted E4 step-one adapter, seven encoded frames and
generated cache history. It proves numerical agreement. It proves no video
quality and no behavior after cache eviction.

## Usage

Humans can use the foreground queue or run this module directly.
See [launch, output and stop commands](../usage.md). Agents use the queue.
The queue recipes below describe managed execution.

This module has no selector of its own. Run it through the
`continuation_check` selector with `--phase short_controls`, as a queue
`experiment` job. See
[run guide §9, E3](../../configs/README.md#e3--causal-continuation-continuation_check-continuation_identity_check-causality).

```json
{"schema_version": 1, "protocol": "continuation_check",
 "arguments": ["--phase", "short_controls", "--job", "/abs/causal_train_job.json",
               "--inputs", "/abs/fixed_inputs.json", "--update-check", "/abs/e4_causal_reference/result.json",
               "--checkpoints", "/abs/step0.safetensors", "/abs/step1.safetensors",
               "--changed-noise-3", "/abs/noise_changed_3.pt", "--changed-noise-5", "/abs/noise_changed_5.pt"]}
```

| Scientific argument | Meaning |
|---|---|
| `--job` | The original causal training job. |
| `--inputs` | The fixed E2 inputs file (see [adapter_effect_check](adapter_effect_check.md#inputs-file)). |
| `--update-check` | The passed serial update result (E4 reference). The job depends on it. |
| `--checkpoints` | The step-zero and step-one adapters. |
| `--changed-noise-3`, `--changed-noise-5` | Saved noise, changed from encoded frame 3 or 5. |

The accepted queue plan also depends on the matching four-rank comparison.
Outputs: nine raw outputs with their records, resource records and
`manifest.json`.

## Data flow

```mermaid
flowchart LR
  I[("Fixed inputs and accepted adapter")] --> P["prepare"]
  P --> S["sample_case / generate"]
  S --> O[("Nine raw outputs and actual calls")]
  O --> V["verify_saved"]
  V --> R(["Exact controls and bounded phases"])
  classDef code fill:#dbeafe,stroke:#2563eb;
  classDef saved fill:#e5e7eb,stroke:#6b7280;
  classDef result fill:#f3e8ff,stroke:#9333ea;
  class P,S,V code;
  class I,O saved;
  class R result;
```

The ordinary queue owns the device, the process ledger, the runtime contract
and supervision. This module starts no process. The continuation owner loads
this module only for the short phase. The larger continuation phases keep
their original import and source closure.

The public E2 preparation checks these items before the weights load: the
calibration, the checkpoint lineage, the actual saved training context, the
independently supplied image, the continuous guide and the exact saved noise.
This module adds its fixed short-control restrictions. It binds both experiment
owners to the current software manifest.

The shared grid, the adapter loader, `evaluate.sample_case`,
`causality.probe_future_noise` and `infer.generate` make all outputs. The
shared saved-case and manifest owners publish the files. Completion reads CPU
tensors and checks their relationships. It opens no model or decoder.

## Organization logic

### Fixed settings

D1, `white`, LTX-2.5 dev, seed 42, direct `[0.725, 0]`, B=2, K=3, D=8 and
generated history. The training span stays null. The module selects seven
encoded frames at 30 fps and 1024-pixel square RGB geometry. It reads the
actual training context and never rebuilds it from a prompt. It uses the
independent supplied image for `c0`, also in the capture tokens given to
evaluation.

### Changed noise

The saved noise must be BF16. At boundary three or five, its earlier values
stay exact, and all later values equal the negated original noise. The module
checks this relationship before any transformer loads. The two changed arrays
are inputs. This module does not generate noise.

### Rollouts

The module loads the checked unmerged FP32 step-one adapter once. Then it runs:

1. two independent evaluation rollouts with the original noise;
2. one product rollout with the original noise;
3. for each boundary, the public two-output future-noise probe and one product
   rollout with the same changed noise.

This gives nine outputs and 54 measured native forwards. Every rollout starts a
fresh shared cache. Each has three denoises, zero primes and three refreshes.

### Pass rules

- All arrays are finite BF16 with the exact source geometry and the supplied
  `c0`.
- The repeated arrays, and the matching evaluation and product arrays, are bit
  identical.
- The original branch of each intervention equals the first repeat.
- Before its boundary, the changed output is exact, with a maximum difference
  of zero.
- Later differences are recorded. No acceptance tolerance is fitted.

### Resources

The module measures eight phases: `load`, `repeat_a`, `repeat_b`,
`product_original`, `future_noise_3`, `product_changed_3`, `future_noise_5`
and `product_changed_5`. It keeps the accepted E4 budget: 1,800 seconds per
phase and 48 billion allocated bytes. It uses the shared Phase owner and keeps
failed records. It checks all pinned bytes, the source identity and the
numerical policy again before the final result and the strict spec manifest.
A partial failure cannot complete.

## Invariants

- One numerical owner for conditioning, noising, generation, adapters and
  caches.
- The same saved context, supplied image, guide and matrices enter every path.
- The controls are exact on full values. The E4 calibration tolerances do not
  relax them.
- Both the step-zero and step-one exports prove the accepted lineage. The
  rollouts use step one.
- Every raw record binds its consumed noise, guide, text, image, calls and
  producer.
- CPU completion checks all nine arrays, the resource phases and the local file
  paths.

## Gotchas

- Seven encoded frames cover 49 RGB frames. This check has no cache eviction.
- The supplied image can differ from capture frame zero. Replacing only the
  output's first frame does not satisfy the input rule.
- Each of the two future-noise phases has two rollouts. Count the actual
  forwards through the shared measurement hook, and also check each rollout's
  call record.
- Do not keep a model reference after its adapter lifecycle ends.
- The larger 17-frame continuation phases keep their separate scientific
  preparation, sampled device guard and completion checks. This module does not
  change their geometry, history, sigma or file requirements.

## Tests

A real small causal transformer runs with the shared evaluation, product and
future-noise APIs. The tests require all nine outputs, preservation of the
supplied image, exact repeat, product and future controls and 54 actual
forwards. They check parser isolation and the existing continuation profiles.

Saved-only controls block model loading. They refuse rehashed output changes,
incorrect calls, changed source or input identity, redirected files and missing
resource phases. Native acceptance needs the same fixed full-weight inputs and
a completed E4 calibration.
