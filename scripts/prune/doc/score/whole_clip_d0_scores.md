# `score/whole_clip_d0_scores.py` — D0 whole-clip activation screen

## Objective

Rank attention heads and FFN channels from exact whole-clip bidirectional
forwards. Pin calibration manifest content and full input provenance. The score
is a local screening proxy, not a measured ablation or quality verdict.

## Data flow

Read the saved baseline manifest and reconstruct each D0 modality with `data.whole_clip.build_input`. Calibration does not need a candidate checkpoint or manifest. On a free GPU load the unpruned transformer once. For each calibration view and sigma, run the exact full bidirectional one-step forward with hooks at the attention output and FFN output projections. Before any scores are accepted, compare the direct output with the saved baseline latent (maximum absolute difference ≤0.02).

Hooks sample across *every generated latent frame*, excluding the clean first
frame. The default `--sampler stride` preserves every `--spatial-stride`-th
flattened spatial token. Opt-in `--sampler balanced_2d_midpoint_v1` selects exactly
the same token budget using actual latent H/W from the native grid. Each run
records geometry, sampler version, selected rows/columns and integer-encoded
spatial/full-token hashes. For each head, score post-attention activation RMS
across sampled tokens and head dimensions multiplied by the Frobenius norm of
that head's output-projection slice. For each FFN channel, score post-activation
RMS times its output-projection column norm. Equal fractional removal **per
attention branch and per FFN layer** keeps every branch executable. Squared
statistics are averaged across calibration clip/sigma pairs before taking RMS.
The JSON contains all scores, binary masks, model fingerprint, input list,
sampled-token identities and forward parity; its width-mask schema is unchanged.

New results use `candidate_format=whole_clip_d0_mask_v1` and stamp task, full-bidirectional attention, clean-frame conditioning, calibration views, exact sigmas, seed, context, baseline manifest and checkpoint fingerprints. Native export and parity reject missing task provenance. Use fresh calibration artifacts.

## Invariants and limits

- Scoring only observes the D0 capture one-step forward. It does not backpropagate, reconstruct weights or alter the baseline checkpoint.
- Saved epsilon values and the original capture hash are checked through the shared `build_input` builder. The model's first output at every calibration point is compared with the saved baseline.
- Default stride sampling selects columns 0 and 16 in every row on a 32×32
  grid. The balanced control uses an 8×8 midpoint grid at the same 64/frame
  budget, expanding column coverage but still sampling only eight rows/columns.
  Neither is a full-token contribution calculation. The forward and held-out
  metrics cover the complete clip. RMS times output-weight norm ignores
  cancellation and downstream effects. Evaluate the compact checkpoint directly.

## Verification

Ruff, sampler CPU tests, and real saved-output parity checks are required. Inspect
mask widths and provenance with `hooks.read_mask_artifact` before exporting.
`test_token_sampling.py` checks selection and actual patchifier geometry;
the real parity and held-out rollout establish model-facing results.
The runtime saved-baseline guard rejects nonfinite differences as well as a
maximum absolute difference above 0.02. The new balanced scorer has not yet run
the production native BF16 baseline check; that remains a GPU gate.
