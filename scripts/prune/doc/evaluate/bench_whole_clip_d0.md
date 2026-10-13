# `evaluate/bench_whole_clip_d0.py` — same-device forward timing

## Objective

Measure baseline and exported D0 transformer latency on the same saved whole-clip input.

## Usage

A human or agent can run the direct CLI from `LTX-2` in `ltx`.
Check `nvidia-smi`. The selected GPU must have at least 44 GiB free.
There is no `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.evaluate.bench_whole_clip_d0 \
  --baseline ../expr/refiner_prune/2.5/prune_doc_example/baseline \
  --candidate ../expr/refiner_prune/2.5/prune_doc_example/candidate \
  --view /abs/corpus/Part_1/0264_01/views/view00_cam51 \
  --sigma 0.909375 --gpu-id 0 --warmup 2 --repeats 5 --order ABA \
  --output ../expr/refiner_prune/2.5/prune_doc_example/benchmark_aba.json
```

| Argument | Meaning |
|---|---|
| `--baseline`, `--candidate` | Matched saved rollout directories |
| `--view`, `--sigma` | One recorded view and exact level |
| `--gpu-id` | Required visible device |
| `--warmup` | Untimed calls per arm; minimum 1, default 2 |
| `--repeats` | Timed calls per arm; minimum 3, default 5 |
| `--order` | `ABA` or `BAB`; A is baseline, B is candidate |
| `--output` | Benchmark JSON |

See the [timing run guide](../usage.md#held-out-quality-and-timing).

## Data flow

Validate the paired manifests, export provenance and actual saved epsilon equality.
Reconstruct the baseline modality with `data.whole_clip.build_input`.
Use one whole-clip block, clean capture frame zero and `attention_mask=None`.
Run each checkpoint sequentially on that same modality.

## Organization logic

Before timing an arm, compare its first output with the saved D0 tensor.
A maximum absolute difference above 0.02 raises.
Warm up, synchronize CUDA before and after each timed call, and record wall-clock milliseconds.
The timed call includes prediction and clean-prefix restoration.
Loading, noising, dense unpatchification, VAE decoding and video encoding are outside the timing.

Each arm records raw repetitions, median, mean, range and peak allocated CUDA memory.
The JSON combines arm medians into the baseline/candidate means.
Speedup is `baseline_ms/candidate_ms`.
Bracket drift is the percentage change between the last and first arm medians.

Worked example: `ABA` measures baseline, candidate, then baseline again.
Medians 100, 80 and 110 ms give baseline mean 105 ms, speedup 1.3125 and bracket drift 10%.

## Invariants

- Both checkpoints use the same saved capture, epsilon, context, geometry and exact sigma.
- Validate checkpoint fingerprints before loading.
- Only one transformer is resident at a time.
- Report peak allocated memory separately from checkpoint storage.

## Gotchas

The current implementation opens a 2.5 session.
This does not establish benchmarking support for other registry keys.
`compact_faithful` restores original GEMM widths; storage savings do not imply faster execution.
The baseline guard uses a threshold comparison. Inspect finite recorded differences before acceptance.
Repeat with `BAB` when a gain is near bracket drift.
One geometry and one view do not establish general throughput.

## Tests

Shared manifest and candidate checks are covered by the CPU suite.
A native benchmark must verify saved-output differences and inspect all timing repetitions.
See [acceptance gates](../VALIDATION.md).

## Dev and recovered timing

Session setup now passes the baseline's explicit transformer path, including the dev override.
Recovered candidates enter strict shared validation before loading.
The benchmark sets the native deterministic policy before CUDA initialization.
Recorded outputs, measured outputs and differences must all be finite.
The relative denominator uses the documented diagnostic floor, without hiding an invalid output.
Default timing uses two warmups and seven repetitions.
Outside-arm drift above two percent triggers twenty repetitions and preserves the first timing rows.
The result records recovered learning provenance separately from timing and ordinary saved-output checks.
