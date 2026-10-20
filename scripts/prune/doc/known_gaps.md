# Current pruning limits

## Learned FFN evidence

Individual binary FFN masks, joint head/FFN training, exact state and compact fusion/export are implemented.
The FFN CPU controls establish numerical and distributed helper behavior.
Fresh native test, four-rank training, decoded quality and speed are separate saved-evidence gates.
The initial aligned width and straight-through estimator are experimental choices, not accepted compression results.
Existing historical head-only receipts do not qualify the new mask family.

## Task coverage

Native consumers accept white-capture, unguided, BF16, one-step whole-clip D0 inputs.
The baseline producer also supports dev schedules, guidance and optional D1 outputs.
Those broader modes do not establish support in calibration, parity or timing.

The registry supports 2.3 and 2.5.
The scorer, functional ablation and benchmark currently open 2.5 sessions.
Use 2.5 for new work. Check each entrypoint before assuming registry-wide support.

## Numerical checks

Compact executed widths can change BF16 GEMM rounding.
`compact_faithful` keeps original execution geometry and separates storage from speed.
Native export parity remains necessary for every candidate.

The scorer explicitly rejects nonfinite baseline differences.
Functional ablation and benchmarking currently use threshold-only comparisons.
Inspect finite outputs and differences before accepting their evidence.

Numerical policy can change attention kernels and rounding.
Keep it fixed in paired runs. The queue reproduction protocol supplies diagnostic controls.
CPU small-model checks do not replace native BF16 checks.

## Calibration scope

The RMS proxy ignores cancellation, downstream interactions and full-token contributions.
Neither spatial sampler covers the full grid.
The balanced sampler's production BF16 saved-baseline check remains a documented GPU gate.

The depth exporter accepts a declared block set.
It does not implement a selection or recovery method.
New depth artifacts are `unqualified`.

## Saved inputs and media

Prompt cache filenames contain model key and prompt hash, but no component hash.
Revalidate after changing the text encoder.
The prompt-cache CLI uses `DEFAULT_PROMPT`; explicit empty text needs its public API.

Native input records index rows by view and sigma.
Multi-seed raw producer output is outside that ordinary single-seed input contract.
Run independent seed directories for calibration and paired comparisons.

Raw-only baselines have no decoded comparison video.
Saved paired analysis visits every row; it does not automatically exclude calibration actors.
The selected baseline must include pinned calibration rows.
Identify held-out rows and inspect synchronized capture VAE, baseline and candidate videos.

The VAE comparisons show decoded latents, not original capture RGB.
Keep that label explicit when judging capture fidelity.

## Evidence boundaries

Sampled checkpoint fingerprints are identity cues, not full-file integrity proof.
The queued baseline wrapper adds complete component and software hashes.
Saved verification does not regenerate device-specific CUDA noise.
Compare saved epsilon tensors directly.

A smaller file does not establish reduced latency or VRAM.
One actor or geometry does not establish general quality or throughput.
The [verification guide](VALIDATION.md) defines the required independent gates.

## Learned head implementation and scientific evidence

The +3 gate learner, legal selector, official velocity loss and adapter/moment transfer are implemented.
CPU preparation, bounded native phases, paired teacher records, compact export, strict recovered validation and media readers are implemented.
The execution probe phase tests head layouts and FFN widths separately.
Main, soft-only, fixed-ranking, balanced, regional, sequential and common fresh recovery paths remain scientific comparisons.
Their implementation does not establish a quality winner or speed candidate.

GT face and hand tracks require a reviewed source-pixel schema and pinned producer.
The learner never guesses the existing keypoint layout or substitutes body boxes.
Preparation pins installed AlexNet LPIPS and cached weights when available.
Missing LPIPS or required region coverage prevents decoded quality acceptance.
Native feasibility, sustained paired training, final quality and cost results must use saved stage receipts.

## Uniform-F18 DDP training

The production DDP route, deterministic temporal crop and shared CPU W&B logger are implemented.
Four-rank CPU checks establish mean gradients and exact resume; they do not prove native memory or quality.
The prior two-rank F18 native diagnostic passed three updates. Full F28 training failed its memory test.
A new four-rank uniform-crop study requires its own `checks.ddp_native` receipt before formal training.
Native crop acceptance, sustained training, full-source monitoring memory and final compact quality remain separate gates.
Implementation work does not authorize restarting a stopped study.

## Refactor acceptance

CPU tests and lint establish local integration only.
Shared-owner native gradient parity, real V3 export identity, saved-media reproduction and I/O timing remain separate gates.
The refactor does not authorize restarting historical studies or replacing their pinned source snapshots.
