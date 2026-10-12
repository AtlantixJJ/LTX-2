# Native MobileWan direct dev checks

## Objective

Verify a complete fresh dev baseline through the stock denoiser, score hooks and all-one masks.
The check uses one recorded full capture and its eight exact task sigmas.

## Usage

Launch this standalone prune check from `LTX-2` in the `ltx` environment after checking GPU occupancy.
The baseline must have completed through the restricted queue recipe first.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.checks.mobilewan_native \
  --baseline ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/validation/native_18/baseline \
  --gpu-id 4 \
  --output ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/validation/native_check_18.json
```

`--baseline` selects pinned saved inputs and outputs. `--output` receives the validation record.
`--gpu-id` selects the free CUDA device. Repeat for the other complete clip length.
See the [run guide](../usage.md#mobilewan-native-check).

## Data flow

The shared `data.whole_clip` reader reconstructs the capture, saved epsilon, context and native modality.
The stock `SimpleDenoiser` receives a `LatentState` with exactly the same tokens, positions and keyframe markers.
Its float32 denoise mask is zero for the clean first frame and one elsewhere.
The exact float32 `[sigma,0]` schedule produces one stock call.

Score hooks observe gated self-attention vectors without returning replacements.
They use stride one and exclude the clean first frame.
A third call attaches all-one masks to both video self-attention and text cross-attention.
The check restores exact clean frame zero before comparing the saved prediction.

## Organization logic

For each sigma, compare the stock output with the score-hook output and the all-one-mask output exactly.
Reject any mismatch, incomplete block coverage or nonfinite saved-baseline comparison.
Require the fresh prediction to equal the saved baseline bytes as tensor values.
Record energies, token counts, index hashes and equality controls per sigma.
Publish the output only when all eight observations pass.

The CLI applies the queue's deterministic Torch policy before opening CUDA.
It requires the same cuBLAS workspace, disables CUDA matmul TF32 and disables cuDNN benchmarking.
It preserves cuDNN's separate TF32 default, as the queue policy does.

## Invariants

The baseline must declare dev and explicit direct schedules. Distilled or multistep baselines fail.
The check does not convert the X0 wrapper's result as velocity.
Model and VAE fingerprints, source, noise and context checks remain in the shared input reader.
Every discovered eligible block contributes every head on the same generated tokens.
This checker supplies native validation evidence, not complete dataset statistics or pruning quality.

## Gotchas

The stock comparison covers `SimpleDenoiser`, not the top-level stock video CLI.
The official scheduler grid is verified separately by the study configuration.
An all-one mismatch must be diagnosed before interpreting removal damage.
A missing output record means the check did not complete; it does not mean the check passed.

## Tests

`tests/test_mobilewan_native.py` intercepts the stock call and verifies all input tensors and dtypes exactly.
The full-clip checker itself supplies the independent native acceptance evidence.

The report pins the projected-energy source hash. A changed source during verification fails before publication.
Dataset workers require that hash to match their current score implementation.
