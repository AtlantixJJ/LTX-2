# `evaluate/whole_clip_d0.py` — matched D0 quality comparison

## Objective

Compare saved baseline and candidate predictions on identical whole-clip D0 inputs.
Optionally measure functional head, FFN and combined ablations before export.

## Usage

A human or agent can run the saved comparison directly on CPU in `ltx`.
Functional ablation uses one free GPU. Run from `LTX-2`.
Neither mode has a `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.evaluate.whole_clip_d0 \
  --baseline ../expr/refiner_prune/2.5/prune_doc_example/baseline \
  --candidate ../expr/refiner_prune/2.5/prune_doc_example/candidate \
  --output ../expr/refiner_prune/2.5/prune_doc_example/comparison
```

For functional ablation, replace `--candidate` with the following arguments.
Use a JSON file, rather than a directory, for this mode's output.

```bash
conda run -n ltx python -m scripts.prune.evaluate.whole_clip_d0 \
  --baseline ../expr/refiner_prune/2.5/prune_doc_example/baseline \
  --functional-mask ../expr/refiner_prune/2.5/prune_doc_example/mask.json \
  --view /abs/corpus/Part_1/0264_01/views/view00_cam51 \
  --sigmas 0.725 0.909375 1.0 --gpu-id 0 \
  --output ../expr/refiner_prune/2.5/prune_doc_example/ablation.json
```

| Argument | Meaning |
|---|---|
| `--baseline` | Saved native D0 run |
| `--candidate` | Saved exported-checkpoint run for paired analysis |
| `--functional-mask` | Native width mask for GPU ablation |
| `--view`, `--sigmas`, `--gpu-id` | Required in functional mode |
| `--output` | Comparison directory, or functional JSON path |

The saved mode writes `comparison.json`, optional comparison MP4s and title PNGs.
The functional mode writes metrics only.
See [quality and timing](../usage.md#held-out-quality-and-timing)
and [functional ablation](../usage.md#calibration-and-functional-ablation).

## Data flow

Use `data.whole_clip` to validate manifests, candidate provenance, saved noise and D0 output hashes.
Read the capture with `corpus.dataset.load_training_master`.
The dataset reader checks schema, shape and fps without importing training.

Saved mode compares every recorded view/sigma pair.
Functional mode reconstructs one held-out view at requested calibrated sigmas.
It first compares an unmasked forward with the saved baseline, then attaches each mask family.

## Organization logic

Build the noised input in float32, then cast to capture dtype.
For each prediction, the direction is `(x_sigma-predicted_x0)/sigma`.
Exclude clean latent frame zero from all reported direction and MSE metrics.

Direction relative L2 is `norm(candidate_direction-baseline_direction)/norm(baseline_direction)`.
Cosine compares their angle. Capture MSE measures error against the capture latent.
Candidate-versus-baseline latent MSE measures prediction disagreement.
These metrics do not judge identity or correct motion.

Worked example: `x_sigma=3`, sigma 0.5, baseline prediction 1 and candidate prediction 2 give directions 4 and 2.
Relative L2 is 0.5 and cosine is 1. Equal direction angle does not imply equal prediction.

For decoded comparisons, crop capture and baseline D0 from the baseline producer video.
Crop candidate D0 from the candidate producer video.
Place those equal-size panels left to right and overlay role labels.
The labels identify VAE decoding. Keep source fps and stop at shared coverage.
The producer video itself has capture, D0 and D1 panels.

## Invariants

- Saved inputs match in capture, context, fps, geometry, seed and schedule.
- Check each epsilon file hash, then compare actual tensor values.
- Candidate exports carry complete native source and family provenance.
- Schedules are exactly `[sigma,0]`.

## Gotchas

Equal seeds do not prove equal noise.
Raw-only runs produce metrics without comparison videos.
The saved comparison visits all rows, including calibration rows when present.
Identify held-out rows explicitly when making a final quality claim.
The selected baseline must include the pinned calibration rows.
Functional mode enforces actor separation and calibrated sigmas.
It opens a 2.5 session with the default 4 GB preflight threshold. This threshold does not establish model capacity.
A functional mask does not establish export parity or faster execution.
The direct-baseline guard uses a threshold comparison; inspect finite differences before acceptance.

## Tests

`tests/test_whole_clip_d0.py` checks direction mathematics and invalid paired inputs.
Mask and actor tests cover functional input restrictions.
Inspect real synchronized videos before claiming perceptual quality.
