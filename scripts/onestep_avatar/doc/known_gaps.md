# Known gaps — current limits

Each entry states the required behavior, the current behavior, its impact and
the evidence that would close it. A source change or a passing CPU test does not
close a gap; accepted native evidence does. The
[acceptance gates](verification.md#acceptance-gates) define that evidence.

## Cached history and causal recomputation

**Required.** Before cached refresh is treated as an optimization of explicit
causal continuation, compare the two under matched text conditioning, clean
per-token history timesteps, positions and retention on the real checkpoint. If
zero-sigma refresh is the intended streaming model, its quality must be shown
under generated history.

**Current.** `refresh_block` computes history K/V with global sigma zero.
`transformer_args.py` uses global sigma for prompt AdaLN, independently of token
timesteps, and the LTX-2.5 checkpoint enables this branch. On the real
checkpoint, clean block-0 K/V are equal at layer 0 but differ from layer 1 onward
when only global sigma changes from 0 to 0.909375 or 1. A two-block real rollout
differs at block 1 by relative latent L2 0.12084 (D0) / 0.11428 (D1) at
0.909375, and 0.23675 in both arms at sigma 1. Block 0 and a repeated cached run
agree exactly. `rollout(history_mode="recompute")` is the explicit block-causal
reference: its retained history stays at token timestep zero while global sigma
follows each denoise step. After eviction, recomputation also changes the old
frames' available context, so comparisons split before and after eviction.

**Impact.** The difference is measured on one view; its contribution to
appearance jumps is unmeasured. Teacher-forced cache history also differs from
the displayed output by definition; that separate mismatch is tested by
dropping `--teacher-forcing`.

**Closing evidence.** Keep sigma-zero cached refresh as the selected
calculation. E3 compares native block output and layerwise K/V before and after
eviction at 0.909375 and sigma 1, boundary quality under both history
calculations and both forcing policies on held-out views, generated-history
quality and measured cost.

## Fused adapter application

**Required.** Evaluation and product run the function training optimised, the
unmerged fp32 adapter against frozen bf16 base weights.

**Current.** Training, ordinary evaluation and product share `model.adapters`.
Inference wraps that velocity function once in stock x0. CPU controls with small
real LTX/PEFT models match the loaded training reference bit-for-bit in both
modes, including zero adapters. `--adapter-application fused_bf16` is an
evaluation diagnostic that needs `--research-override`; product has no fusion
choice and no fallback to it.

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

**Current.** On a 48 GB GPU, one process completes two 22B model states and
fails while it loads the third. Two native four-state requests (base, update 0,
update 20, update 60) failed this way: a causal warmed bench and a
bidirectional held-out evaluation. Each saved the base and update-0 outputs,
then raised CUDA out of memory while loading the update-20 weights. At the
failure the process itself held 46.30 GiB allocated by PyTorch. The source of
the retained allocation is not established.

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

## Eight-block causality diagnostic

`experiments.causality` protocol `eight_block` uses the typed adapter contract
and the shared unmerged loader. It runs only an adapter whose recorded
conditions match eight causal blocks over 17 encoded frames; a shorter trained
span needs its own protocol, not an override.

## Supervision and reproduction records

**E3 changed-past native verification.** The isolated past observer restores
the original refresh layouts from saved generated-history values: contiguous
for the c0 block, packed preserve-format copies for later blocks. Focused CPU
checks cover both history computations, schedules and eviction targets. Fresh
native original replay must still reproduce every layer's K/V and the current
prediction bit-for-bit before changed-past results can be accepted. The saved
failed original reconstruction remains failed; no tolerance is relaxed.

Scientific output completion and continuous queue supervision are separate. An
ended queue owner or a complete tensor file cannot create missing supervisor
evidence; recovery records the attempt as failed with reason
`missing_supervision`. A raw reproduction mismatch stays a failure even when
noise and first-image inputs match. Diagnose source, input, precision and
backend-policy differences with one-factor controls; never fit a tolerance to a
failed result.

**E4 compare budget.** Both `training_update_check` operations now bind the
original training job's absolute resource budget. A different declared budget
fails normalization. Fresh native reference and compare evidence remains
required for this producer.

**Saved pruning reference.** The saved whole-clip D0 reference
`expr/refiner_prune/2.5/recheck_20260930/baseline` was made with a BF16 global
sigma and PyTorch default numerical settings, under which scaled-dot-product
attention runs the cuDNN kernel. The current producer passes a float32 global
sigma (per-token timesteps are float32 in both), and the deterministic policy
(`torch.use_deterministic_algorithms(True)`) makes attention run the Flash
kernel. `experiments/prune_reproduction.py` reproduces the reference
bit-for-bit with default settings and a BF16 global sigma; the dense mask, cuDNN
determinism and BF16 reduction change nothing. Current outputs therefore differ
from that reference at every sigma. Pruning comparisons stay within one
producer setting until a baseline is rebuilt or the reference's settings are
adopted.

## Related

[Core algorithm](core_algorithm.md) defines required calculations.
[Training choices](training_choices.md) defines supported settings.
[Verification](verification.md) defines checks and acceptance gates.
[Architecture](architecture.md) defines owners and software boundaries.
