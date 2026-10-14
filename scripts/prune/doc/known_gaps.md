# Current pruning limits

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

The ridge helper fits a bounded local projection on CPU.
It has no native collector or fitted-checkpoint exporter.
Local calibration MSE is not held-out model improvement.
Cache identity checks cannot prove how arbitrary features were collected.

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

## MobileWan implementation status

Squared projected-energy, cohort statistics and global allocation now have focused CPU controls.
The complete capture-only audit CLI checks pinned membership and source content.
The resident dataset runner and native dev queue integration are implemented.
Complete dataset execution, removal evaluation and final reporting remain pending.
These helpers do not establish native scoring or removal quality.

The producer now supports explicit direct dev schedules and capture-only D0.
The restricted agent baseline wrapper now accepts unmodified direct dev D0 at sigma <=0.95.
Per-view score records now reject incomplete observations and changed resume identities.

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

The single-process block placement and four-process FSDP methods have separate native acceptance gates.
Passing a small-model or block-placement check does not establish full-length FSDP acceptance.
Keep run-specific memory results, failures and retired artifacts in experiment records and the active plan.

## GT-reference FSDP pilot

The direct four-rank FSDP method is implemented with replicated small FP32 trainable state.
Small distributed/serial gradient and exact resume checks pass. Native full-clip FSDP acceptance passes for the frozen primary pilot.
The direct supervisor runs formal training. Full training completion and decoded quality remain pending.
The user-directed GT reference mode requires no teacher cache or speed probe before training.
Historical score reuse declares its actual learning-view coverage. Reviewed small-region tracks and decoded quality acceptance remain separate gaps.
Unweighted training defers foreground replay and bucket preparation. Regional and balanced arms require a separate full preparation.
Compact export lazily creates missing structural input manifests after selection; it does not rerun calibration or teacher forwards.
