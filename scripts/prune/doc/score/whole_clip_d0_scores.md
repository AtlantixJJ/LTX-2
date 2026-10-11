# `score/whole_clip_d0_scores.py` — D0 whole-clip activation screen

## Objective

Rank attention heads and FFN channels from exact whole-clip bidirectional
forwards. Pin calibration manifest content and full input provenance. The score
is a local screening proxy, not a measured ablation or quality verdict.

## Usage

A human or agent can run the direct CLI from `LTX-2` in `ltx`.
It requires one free GPU. There is no `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.score.whole_clip_d0_scores \
  --baseline ../expr/refiner_prune/2.5/prune_doc_example/baseline \
  --view /abs/corpus/Part_1/0008_01/views/view00_cam51 \
  --sigmas 0.725 0.909375 1.0 --head-fraction 0.10 --ffn-fraction 0.10 \
  --sampler stride --spatial-stride 16 --gpu-id 0 \
  --output ../expr/refiner_prune/2.5/prune_doc_example/mask.json
```

| Argument | Meaning |
|---|---|
| `--baseline` | Saved unpruned native D0 run |
| `--view` | Calibration view; repeat for more views |
| `--sigmas` | Exact saved calibration levels |
| `--head-fraction`, `--ffn-fraction` | Per-branch removal fractions; default 0.10 |
| `--sampler` | `stride` or `balanced_2d_midpoint_v1` |
| `--spatial-stride` | Sample budget control; default 16 |
| `--gpu-id` | Required device with at least 44 GiB free |
| `--output` | Score and mask JSON |

See the [calibration run guide](../usage.md#calibration-and-functional-ablation).

## Data flow

Read the baseline manifest and select every requested calibration view/sigma pair.
Reconstruct each D0 input with `data.whole_clip.build_input`.
Load the unpruned transformer once.
Observe activations before attention and FFN output projections.

Each forward must reproduce its saved D0 latent.
Reject nonfinite maximum differences or values above 0.02.
Only after this check can the run contribute accepted scores.

Sample every generated latent frame, excluding c0.
The default `stride` sampler selects flattened spatial offsets.
The optional balanced sampler uses actual latent height and width at the same budget.
Record sampler, geometry, rows, columns and integer-encoded index hashes.

Write `whole_clip_d0_mask_v1` with scores, binary masks, removal counts and per-case forward checks.
Pin the full calibration-manifest content and native input distribution.
Calibration needs no candidate checkpoint.

## Organization logic

Hooks collect post-activation square means per branch and calibration case.
Average those means across cases before taking the square root.
For heads, average activation squares across sampled tokens and head dimensions.
Multiply the resulting RMS by the Frobenius norm of that head's output-weight slice.
For FFN channels, multiply activation RMS by the corresponding output-column norm.
Allocate removals independently in each branch.
Stable score ties follow source order. Keep at least one head or channel.

Worked example: four FFN scores `[1,1,3,4]` at fraction 0.5 remove the first two channels.
The forward still executes the complete clip.

## Invariants

- Scoring observes native D0 forwards without gradients or checkpoint changes.
- Check capture and saved epsilon hashes through the shared input builder.
- Compare every calibration output with the saved baseline.

## Gotchas

The implementation opens a registered 2.5 session.
Registry support for 2.3 does not establish scorer support for 2.3.

- Default stride sampling selects columns 0 and 16 in every row on a 32×32
  grid. The balanced control uses an 8×8 midpoint grid at the same 64/frame
  budget, expanding column coverage but still sampling only eight rows/columns.
  Neither is a full-token contribution calculation. The forward and held-out
  metrics cover the complete clip. RMS times output-weight norm ignores
  cancellation and downstream effects. Evaluate the compact checkpoint directly.

## Tests

Ruff, sampler CPU tests, and real saved-output parity checks are required. Inspect
mask widths and provenance with `hooks.read_mask_artifact` before exporting.
`test_token_sampling.py` checks selection and actual patchifier geometry;
the real parity and held-out rollout establish model-facing results.
The runtime saved-baseline guard rejects nonfinite differences as well as a
maximum absolute difference above 0.02. The new balanced scorer has not yet run
the production native BF16 baseline check; that remains a GPU gate.
