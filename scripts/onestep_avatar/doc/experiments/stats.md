# `experiments/stats.py` — measurement, no training

## Objective

Four numbers, each of which decides something the design would otherwise be guessing at:

| | What | Decides |
|---|---|---|
| **(a)** | ε-sensitivity: spread of one-step outputs from one `z_g` over N seeds | whether ε is a sampling variable here at all |
| **(b)** | the excursion `a = ‖Φ(x_σ₀) − z_g‖/√d` | how far the base model moves its own input, in the same per-dimension RMS units as `r` |
| **(c)** | latent moments of `z_g` vs `z_y` | whether an affine correction is needed (answer: no) |
| **(d)** | the gap `r = ‖z_y − z_g‖/√d`, full **and** subject-interior | how much a plain blend has to do |

## Data flow

```mermaid
flowchart LR
  PAIRS[("--pairs &lt;corpus root&gt;<br/>master z_g / z_y per view")]
  MODEL[("model + guide latents")]
  RATIO["r_full, r_subject, moments<br/>--no-gpu"]
  FWD["one-step forwards at σ₀<br/>eps spread, excursion a · GPU"]
  OUTJ[("analysis_summary*.json")]

  PAIRS --> RATIO --> OUTJ
  MODEL --> FWD --> OUTJ

  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  class RATIO,FWD proc;
  class PAIRS,MODEL,OUTJ disk;
```

## Organization logic

Everything here is measurement and writes only JSON — no checkpoints, no training state. The
GPU half and the `--no-gpu` half are one module because they report into one summary and share
the latent-loading path.

`r` is measured over **whole clips** since the causal rewrite, rather than over overlapping
windows that double-counted every second latent frame.

### Core measurement calculations

For arrays `a,b` with `d` values, `rms_gap` is `sqrt(sum((a-b)^2)/d)` in fp32.
For encoded coverage weights `w[F,H,W]`, expand weights across channels/batch and use
`sqrt(sum(w*(a-b)^2)/sum(w))`.
Current code returns NaN for zero total weight. Treat that as an invalid measurement, not zero error.
These weights are measurement inputs; they do not change the training loss.

`measure_pairs` reads the capture and guide for the same relative video/view and background.
Current code truncates both to the smaller encoded frame count; record that common coverage.
Calculate a full-frame gap for each view.
When both masks exist, use render coverage, capture coverage, their maximum (union),
or their minimum (intersection), according to the declared mask choice.
Save per-view values and the count of views with subject measurements.
Do not silently describe a missing subject score as a full-frame score.

Per-channel moments flatten each channel's frame/spatial values and calculate mean and sample standard deviation.
Current global summaries average per-view means/standard deviations with equal view weight.
Current per-channel moments use the first loaded pair, not every corpus frame.
`_summary` reports count, mean, p10/p50/p90, and maximum over the supplied per-view values.

For the guide-only map measurement, keep one guide encoding, prompt, first guide frame, and sigma fixed.
Use seeds `seed+k` for `k=0,...,N-1` and produce one output `o_k` for each seed.
Excursion is the mean `rms_gap(o_k,z_g)`.
Noise spread is the mean `rms_gap(o_i,o_j)` over all `i<j`; with one output it is zero by construction.
The recorded heuristic compares median spread/excursion with `.1`.
Current code assigns ratio zero for zero excursion. That special case or one output does not prove noise is irrelevant.

This unpaired diagnostic keeps the guide's first encoded frame unchanged.
It is not a product input with a real supplied first image, and has no capture target.
Label that distinction when reporting its values.
Use relative video paths for per-video output keys; corpus guide basenames are repeated.

## Reading the numbers

The numbers below are observations from one set of encoded inputs.
They are context, not corpus acceptance rules.
Check the original measurement records and regenerate matched coverage before reusing them.

- `r_subject` ≈ 0.89–0.93 against a 0.6 "comfort" threshold: a plain blend asks the LoRA to
  move the output by ~90 % of its natural scale. Substantial, **not** a rejection — the base
  model's own excursion `a` = 0.247 is only a quarter of that, so whether a LoRA closes the
  gap is open.
- Moments: per-channel std ratio p50 1.002; the worst mean offset reaches the transformer as
  an 8 % perturbation on 1 channel of 128. No correction worth building.
- Mind the reference: under the linear interpolant `Var(x_σ) = (1−σ)² + σ² = 0.60` at σ₀ **by
  design**. Matching to unit variance would be the wrong correction.

## Gotchas

- Reports `r_full` *and* `r_subject`, and they move very differently: the pixel composite
  collapsed `r_full` 1.39 → 0.53 while `r_subject` moved only 0.89–0.93 → 0.80–0.91. Quoting
  the wrong one overstates what compositing bought.
- Current per-channel moments use the first loaded pair. Global summaries aggregate view statistics.
  State those scopes before making a corpus-wide claim about unusual channels.

## Whole-clip map measurement

`measure_map` encodes the complete VAE-aligned guide prefix, keeps its first latent
frame clean, and executes one full-bidirectional `[sigma0, 0]` forward per saved
seed. Excursion and epsilon spread cover that complete clip. There is no 25-frame
window baseline; numbers must be regenerated under this coverage before comparing
with newly encoded capture/guide pairs.

## Invariants

- Gap units are RMS per encoded value, not an unnormalized vector norm.
- Full-frame and subject scores remain separate.
- Saved records identify background, coverage, mask choice, sigma, and seed count.
- A guide-only diagnostic is not described as product/capture fidelity.

## Tests

Worked gap check: differences `[0,2]` give RMS `sqrt(2)`.
Weights `[0,1]` give weighted RMS `2`.
For scalar non-first-frame outputs `0,2` around guide value `1`, excursion is `1` and spread is `2`.
These are arithmetic examples, not observed corpus results.
After implementation changes, check zero-weight/missing-mask records, coverage counts,
repeated filenames, and the exact scope of moment summaries before interpreting results.

## Queue interface and completion

The queue owns launch, device selection, process records and bounded supervision.
The checker accepts a pinned `schema_version: 1` specification with its literal
protocol and scientific `arguments`, plus queue-owned `--output` and local
`--gpu-id`. Execution flags in the scientific arguments fail before writes.
Public `parse_args`, `read_spec`, `resource_phases`, `verify_completion` and
`evidence_paths` provide normalization, phase/topology data and saved-only
completion. The manifest binds exact spec bytes, normalized scientific settings
and the complete local artifact inventory. Each run binds
the spec helper as an explicit experiment extra.

Completion verifies the original scientific controls from actual saved artifacts.
It loads no model or decoder, starts no child, and never repairs missing output.
A missing, changed or escaped artifact prevents completion. CPU fixtures prove
these refusals.

The strict outer output is a directory. The original scientific `--out` setting
is injected as `analysis_summary.json` inside that directory. Scientific specs
cannot supply their own `--out`, GPU, process or supervision setting. The queue
records one generic `runtime` phase. `--no-gpu` retains the pair-only arithmetic.

New map runs also save `map_raw.pt`: each exact guide latent and its ordered
outputs, keyed by source-relative video name. Source path/hash records bind
these tensors to the selected video inventory. Completion requires the saved
output count, shape, finite values and unchanged clean first frame. It recomputes
excursion, pairwise spread, quantiles and the original `.1` heuristic from those
actual tensors. Pair completion reuses the original CPU paired-master reader
and metric function. `inputs.json` binds the exact selected masters, masks and
videos plus producer software. Extra files, escaped links, missing sources,
changed inputs or rehashed false measurements fail completion.

Float32 GPU and CPU reductions can differ in their last bits. Completion uses
fixed `rel_tol=1e-5`, `abs_tol=1e-6` for numerical metric readback only; this does
not change the original `.1` and `.6` heuristic thresholds. Zero weight produces
an invalid nonfinite measurement and cannot pass completion. The source decoder
and transformer are never invoked by this saved verifier.

The queue may request `evidence_paths(..., verified=True)` only after it has
completed this owner's scientific verifier in the same receipt check. That path
reads the checked inventory without repeating the verifier. Direct callers keep
the default `verified=False`, which verifies before exposing evidence.
