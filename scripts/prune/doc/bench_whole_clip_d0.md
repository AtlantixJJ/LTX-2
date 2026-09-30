# `evaluate/bench_whole_clip_d0.py` — same-GPU D0 forward latency

## Objective

Measure the unpruned and compact-pruned LTX-2.5 **one-step whole-clip D0 transformer forward** on the same GPU, independently of the historical `k2` window gate. This prices the dense shape-reduced checkpoint directly; it does not infer latency from sparse gather/scatter or FLOP counts.

## Data flow

The CLI reads the two `visualize_d1 --whole-clip` manifests from the pruning comparison, validates their fixed setup and actual saved noise equality. `data.whole_clip.build_input` reconstructs the exact D0 modality from the baseline alone: `lerp(capture, epsilon, sigma)`, clean frame 0, global positions, one block `[0,T)`, all-ones bidirectional mask, and the same prompt context. Before timing an arm, it compares that arm's first forward with its saved latent output and refuses a maximum absolute difference above 0.02. It then runs a warmup and repeated synchronized wall-clock forwards.

The default order is baseline → candidate → baseline; `--order BAB` reverses it. The repeated outside arm exposes run-order drift. The JSON saves every timing, median, mean, range, peak allocated CUDA memory, input/source fingerprints, geometry, hardware, and saved-output parity check. Checkpoint loading, noising, VAE decoding and MP4 encoding are outside the timed scope.

## Command

Run from `LTX-2` in the `ltx` environment after checking `nvidia-smi`:

```bash
python -m scripts.prune.evaluate.bench_whole_clip_d0 \
  --baseline ../expr/onestep_avatar/d1_diagnostic/ar_sigma_rollouts/runs/s1d_prompts_20260929/P1_3actors \
  --candidate ../expr/refiner_prune/2.5/whole_clip_d0/p05_compact \
  --historical-transfer \
  --view /data1/datasets/AnimatableHuman/DNARenderingVideo/Part_1/0008_01/views/view00_cam51 \
  --sigma 0.909375 --gpu-id 3 --warmup 2 --repeats 5 \
  --output ../expr/refiner_prune/2.5/whole_clip_d0/benchmark_0008_01_s0909375.json
```

## Invariants and gotchas

- Candidate provenance is checked before GPU model loading. Historical k2 exports require `--historical-transfer`. `compact_faithful` stores compact parameters but pads GEMMs at execution; report its measured cost rather than assuming reduced dense work.

- A free A6000 needs at least 44 GiB available for the 1024-pixel, 18-latent-frame modality plus the 22B transformer. Do not share its GPU with a training run.
- The same saved capture, epsilon, sigma, empty prompt, attention mask and clean keyframe are used for both checkpoint loads. The single block has no history or K/V cache.
- Benchmarking a prebuilt `Modality` isolates transformer execution. It does not measure data loading or total video pipeline throughput.
- The repeated outside arm exposes thermal/load drift. Interpret a small speed ratio only if it exceeds that drift and timing spread. A one-case benchmark does not establish throughput across other geometries.
- The first forward is checked against the saved rollout because an apparently faster call on a different attention path would not answer this experiment.

## Tests

Run Ruff and `py_compile` on this module, then inspect the saved-output difference and raw repetitions in a real benchmark result. The pair analyzer's CPU tests cover the shared manifest validation.
