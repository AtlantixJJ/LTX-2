# Known gaps — current limits

Each entry states the required behavior, the current behavior, its impact and
the evidence that would close it. A source change or a passing CPU test does not
close a gap. Only accepted native evidence closes it. The
[acceptance gates](verification.md#acceptance-gates) define that evidence.

## Saved causal block ranges for guide-study measurements

**Required.** The approved guide-study revision uses the public canonical planner
with saved frame counts and mode settings. Check coverage and measured calls.
Label reconstructed ranges clearly; they do not prove saved executed boundaries.
Readers must not build a second block planner.

**Current.** `evaluate.sample_case` computes `geometry.plan` for causal sampling.
It saves the frame count, mode settings, schedule and call counts, but no block ranges.
`evaluate.save_case` adds output identity and completion state without these ranges.

**Evidence.** The 2026-10-12 rerun has a CPU small-model saved-record check at
`expr/onestep_avatar/ar_sigma_rollouts_rerun_20261012/provenance/blocking_gap.json`
in the workspace. The saved causal record has three denoise and three refresh
calls, but no block-range field. This checks the record contract, not native quality.

**Impact.** The producer still does not save executed ranges. On 2026-10-12, the
user approved using the public canonical planner in this study's metrics reader.
The reader checks saved frame counts, coverage and calls. It labels ranges as
reconstructed. Shared producers and the running training remain unchanged.

**Closure evidence.** Authorize a shared-producer change and resolve affected software
pins first. Save the ranges used by sampling, then test saved seam and first-block
measurements against those ranges. Fresh native smoke receipts must pass afterward.

## Cached history and causal recomputation

**Required.** Do not treat the cached refresh as only an optimization of
explicit causal continuation until the two are compared on the real
checkpoint. The comparison must match the text conditioning, the clean
per-token history timesteps, the positions and the retention. If zero-sigma
refresh is the intended streaming model, show its quality with generated
history.

**Current.**

- `refresh_block` computes the history K/V with global sigma zero.
- `transformer_args.py` uses the global sigma for prompt AdaLN, independently
  of the token timesteps. The LTX-2.5 checkpoint enables this branch.
- On the real checkpoint, the clean block-0 K/V are equal at layer 0. They
  differ from layer 1 onward when only the global sigma changes from 0 to
  0.909375 or 1.
- A two-block real rollout differs at block 1 by a relative latent L2 of
  0.12084 (D0) and 0.11428 (D1) at 0.909375, and 0.23675 in both arms at
  sigma 1.
- Block 0 and a repeated cached run agree exactly.

`rollout(history_mode="recompute")` is the explicit block-causal reference. Its
retained history stays at token timestep zero, while the global sigma follows
each denoise step. After eviction, recomputation also changes the context that
the old frames can see. So comparisons are split into before and after
eviction.

**Impact.** The difference is measured on one view. Its effect on appearance
jumps is not measured. Teacher-forced cache history also differs from the
displayed output by definition. Remove `--teacher-forcing` to test that
separate mismatch.

**Closing evidence.** Keep the sigma-zero cached refresh as the selected
calculation. E3 compares these items on held-out views:

- native block output and layerwise K/V before and after eviction, at
  0.909375 and at sigma 1;
- boundary quality with both history calculations and both forcing policies;
- generated-history quality and the measured cost.

## Fused adapter application

**Required.** Evaluation and product run the function training optimised, the
unmerged fp32 adapter against frozen bf16 base weights.

**Current.** Training, ordinary evaluation and the product share
`model.adapters`. Inference wraps that velocity function once in stock x0. CPU
controls with small real LTX/PEFT models match the loaded training reference
bit for bit in both modes, including zero adapters.
`--adapter-application fused_bf16` is an evaluation diagnostic, and it needs
`--research-override`. The product has no fusion choice and no fallback to it.

**Mechanism.** Fusing `W + BA` into bf16 base weights loses low-order bits when a
small correction is added to a large base value. It shrinks and perturbs the
learned correction, most for small, early adapters.

**Closing evidence.** E2 measures the unmerged effect against the loaded
training reference (relative L2 under 0.05, with the near-zero rule) on two
views and trained steps 20 and 60, with decoded inspection and cost. Fusion is
measured separately.

## Random-start first-image distribution

**Required.** Training's `c0` has the distribution of what deployment supplies:
one real image encoded by the causal VAE as latent frame 0.

**Current.** `--start-policy random` selects a segment `[s, s+L)` from the
master and uses master frame `s` as `c0` without re-encoding. For `s = 0` this
is the single-image keyframe encode; for `s > 0` it is a latent that encodes
eight pixel frames, with different statistics. Evaluation and product start at
`s = 0`. Random-start causal segments are single independent blocks.

**Impact.** A share `1 − 1/(F − L + 1)` of random-start samples conditions on a
`c0` that deployment never produces (one half for an 18-frame master with
`L = 17`).

**Closing it.** Re-encode each segment from pixel frame `8s` so its frame 0 is a
true single-image encode, or train only on `s = 0` (`--start-policy clip_start`).

## Direct endpoint and stock terminal rounding

**Required.** Positive next levels use the stock step's operation order and
dtype conversions. A direct `[sigma, 0]` step returns the prediction exactly.

**Current.** `model.sampling.euler_to` calls the native
`EulerDiffusionStep.step` at positive next levels and returns the prediction
directly at zero. The stock bf16 step rounds velocity before reconstructing its
endpoint, so terminal outputs differ slightly. This difference is kept and
reported, never hidden by a parity claim or an unmeasured tolerance.

**Closing evidence.** CPU tests match the native step exactly at positive levels
and keep the exact direct endpoint. E1 measures full-model raw and decoded
differences with matched inputs.

## Several model states in one evaluation process

**Required.** An ordinary evaluation or bench request that names several model
states (the base and one or more adapters) runs them one after another within
its memory bound. Each model is released before the next one loads.

**Current.** On a 48 GB GPU, one process completes two 22B model states. It
fails while it loads the third. Two native four-state requests (base, update 0,
update 20, update 60) failed this way: a causal warmed bench and a
bidirectional held-out evaluation. Each request saved the base and update-0
outputs. Then it raised CUDA out of memory while it loaded the update-20
weights. At the failure, PyTorch had 46.30 GiB allocated in that process. The
source of the retained allocation is not known.

**Impact.** A request with three or more model states (`--include-base` with
two or more `--checkpoint`, or three or more `--checkpoint`) is not usable on
these GPUs. Run one model state per job. Consumers that compare states read
one-state outputs: `experiments/capacity_review.py` groups four one-state jobs
per case, and a saved comparison render names each state's output file.

**Closing evidence.** Find and release the retained allocation. Then a native
four-state evaluation must equal the matching one-state outputs bit-for-bit
under the deterministic policy, with its peak memory recorded.

## Thread-dependent saved-metric verification

**Required.** Verifying a saved result gives the same answer in any process
that reads the same bytes.

**Current.** Completion checks recompute encoded metrics on CPU and compare
them with the saved floats exactly. CPU reductions depend on the thread count.
A saved conditioning control verified at `OMP_NUM_THREADS=8`, the queue
owners' setting, and failed at 4 with "conditioning arm metrics differ from
saved outputs".

**Impact.** A valid result can fail verification in a process with another
thread count. Run every verifier (receipt checks, `--verify`, saved reviews)
with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.

**Closing evidence.** Metric verification that passes at several thread counts
on the same saved outputs, for example a fixed-order float64 reduction.

## Changed-past decoder total deadline

**Required.** A changed-past decoder attempt must complete two things within
its declared deadline for the whole attempt: the child execution and the
queue's independent saved completion checks. A child exit code of zero does not
prove complete bounded execution.

**Current.**

- The decoder checks its full native prerequisites before it opens a session.
  It reuses those prepared inputs for its own saved-output checks in the same
  child.
- The public `verify_review` always prepares again. So the queue still checks
  the native all-layer evidence and every saved decoder output independently.
- That completion work counts toward the original deadline.
- Native attempts finished their supervised child successfully, but failed the
  total deadline during the parent's completion checks.
- The current reuse of prepared inputs has CPU checks. Its complete native
  timing is not yet known.

**Impact.** Saved videos or a passed child supervisor cannot accept an attempt
whose queue records `overall_deadline`. Keep those attempts failed and keep
their failure observations. A source change requires fresh native and decoder
jobs for the affected results.

**Closing evidence.** Fresh native interventions and decoder attempts that
match the source complete their canonical queue checks within the unchanged
bounds. Record the child timing and the total attempt timing. Do not increase
the budget. Do not skip the independent completion verification.

## Eight-block causality diagnostic

The `experiments.causality` protocol `eight_block` uses the typed adapter
contract and the shared unmerged loader. It runs only an adapter whose recorded
conditions match eight causal blocks over 17 encoded frames. A shorter trained
span needs its own protocol, not an override.

## Supervision and reproduction records

**E3 changed-past native verification.** The isolated past observer restores
the original refresh layouts from the saved generated-history values. The
`c0` block uses a contiguous layout. Later blocks use packed preserve-format
copies. Focused CPU checks cover both history calculations, the schedules and
the eviction targets. A fresh native replay of the original must still
reproduce every layer's K/V and the current prediction bit for bit. Only then
can changed-past results be accepted. The saved failed reconstruction stays
failed. No tolerance is relaxed.

Scientific output completion and continuous queue supervision are separate.
An ended queue owner or a complete tensor file cannot create missing
supervisor evidence. Recovery records the attempt as failed with reason
`missing_supervision`. A raw reproduction mismatch stays a failure, even when
the noise and first-image inputs match. Find source, input, precision and
backend-policy differences with one-factor controls. Never fit a tolerance to a
failed result.

**E4 compare budget.** Both `training_update_check` operations bind the
absolute resource budget of the original training job. A different declared
budget fails normalization. This producer still needs fresh native reference
and compare evidence.

**Saved pruning reference.** The saved whole-clip D0 reference
`expr/refiner_prune/2.5/recheck_20260930/baseline` differs from the current
producer in two settings:

| Setting | Saved reference | Current producer |
|---|---|---|
| Global sigma dtype | BF16 | float32 |
| Numerical policy | PyTorch defaults: attention runs the cuDNN kernel | deterministic (`torch.use_deterministic_algorithms(True)`): attention runs the Flash kernel |

Per-token timesteps are float32 in both. `experiments/prune_reproduction.py`
reproduces the reference bit for bit with the default settings and a BF16
global sigma. The dense mask, cuDNN determinism and BF16 reduction change
nothing. So the current outputs differ from that reference at every sigma.
Keep pruning comparisons within one producer setting until a baseline is
rebuilt or the reference settings are adopted.

## Related

[Core algorithm](core_algorithm.md) defines required calculations.
[Training choices](training_choices.md) defines supported settings.
[Verification](verification.md) defines checks and acceptance gates.
[Architecture](architecture.md) defines owners and software boundaries.
