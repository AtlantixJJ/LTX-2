# `score/whole_clip_d0_scores.py` — D0 whole-clip activation screen

## Objective

Rank attention heads and FFN channels from exact whole-clip bidirectional
forwards. Pin calibration manifest content and full input provenance. The score
is a local screening proxy, not a measured ablation or quality verdict.

## Data flow

Read the saved baseline manifest and reconstruct each D0 modality with `data.whole_clip.build_input`. Calibration does not need a candidate checkpoint or manifest. On a free GPU load the unpruned transformer once. For each calibration view and sigma, run the exact full bidirectional one-step forward with hooks at the attention output and FFN output projections. Before any scores are accepted, compare the direct output with the saved baseline latent (maximum absolute difference ≤0.02).

Hooks sample every `--spatial-stride`-th token across *every generated latent frame*, excluding the clean first frame. For each head, score post-attention activation RMS across sampled tokens and head dimensions multiplied by the Frobenius norm of that head's output-projection slice. For each FFN channel, score post-activation RMS times its output-projection column norm. Equal fractional removal **per attention branch and per FFN layer** keeps every branch executable and avoids comparing unlike branch scales globally. Squared statistics are averaged across calibration clip/sigma pairs before taking RMS. The resulting JSON contains all scores, binary masks, model fingerprint, input list, sampled token count and forward parity; it is accepted directly by `score.export_pruned`.

New results use `candidate_format=whole_clip_d0_mask_v1` and stamp task, full-bidirectional attention, clean-frame conditioning, calibration views, exact sigmas, seed, context, baseline manifest and checkpoint fingerprints. Native export and parity reject missing task provenance. Use fresh calibration artifacts.

## Invariants and limits

- Scoring only observes the D0 capture one-step forward. It does not backpropagate, reconstruct weights or alter the baseline checkpoint.
- Saved epsilon values and the original capture hash are checked through the shared `build_input` builder. The model's first output at every calibration point is compared with the saved baseline.
- Sampling covers every generated frame using a stride over flattened spatial indices; it is not a balanced two-dimensional or full-token contribution calculation. On a 32×32 grid, stride 16 selects columns 0 and 16 in every row. Treat this lattice as a calibration limitation; the forward and held-out metrics still cover the complete clip. RMS times output-weight norm ignores cancellation and downstream effects. The compact checkpoint must be evaluated directly after export.

## Verification

Ruff, import/compile check, and the real saved-output parity checks are required. Inspect mask widths and provenance with `hooks.read_mask_artifact` before exporting. No new small-tensor test is needed for this reversible scoring proxy; the real parity and held-out rollout are the meaningful checks.
