# `score/whole_clip_d0_scores.py` — D0 whole-clip activation screen

## Objective

Rank structural attention-head and FFN-channel pruning units under the actual one-step D0 whole-video input, rather than `k2` sliding-window calibration records. This is a **screening proxy**, not a measured per-unit ablation or a quality verdict.

## Data flow

Read the saved baseline manifest and reconstruct each D0 modality with `data.whole_clip.build_input`. Calibration does not need a candidate checkpoint or manifest. On a free GPU load the unpruned transformer once. For each calibration view and sigma, run the exact full bidirectional one-step forward with hooks at the attention output and FFN output projections. Before any scores are accepted, compare the direct output with the saved baseline latent (maximum absolute difference ≤0.02).

Hooks sample every `--spatial-stride`-th token across *every generated latent frame*, excluding the clean first frame. For each head, score post-attention activation RMS across sampled tokens and head dimensions multiplied by the Frobenius norm of that head's output-projection slice. For each FFN channel, score post-activation RMS times its output-projection column norm. Equal fractional removal **per attention branch and per FFN layer** keeps every branch executable and avoids comparing unlike branch scales globally. Scores are averaged across calibration clip/sigma pairs. The resulting JSON contains all scores, binary masks, model fingerprint, input list, sampled token count and forward parity; it is accepted directly by `score.export_pruned`.

New results use `candidate_format=whole_clip_d0_mask_v1` and stamp task, full-bidirectional attention, clean-frame conditioning, calibration views, exact sigmas, seed, context, baseline manifest and checkpoint fingerprints. Native export and parity reject missing task provenance. The earlier score artifact predates this schema; re-score to create a native candidate.

## First screen

Use actors `0008_01` and `0012_09` for scoring at sigma 0.725, 0.909375 and 1.0; reserve `0025_11` as a held-out visual/direction check. Start with 10% heads in each self- and cross-attention branch and 10% FFN channels in each layer. The actual head count is three of 32 per branch (9.375%) because a head is indivisible. This is a larger structural probe than the earlier p05 export, which changed only cross-attention layers. Do not call this mask D0-optimal until held-out direct rollout and decoded outputs have been inspected.

## Invariants and limits

- Scoring only observes the D0 capture one-step forward. It does not backpropagate, reconstruct weights or alter the baseline checkpoint.
- Saved epsilon values and the original capture hash are checked through the shared `_input` builder. The model's first output at every calibration point is compared with the saved baseline.
- Sampling spans time and space uniformly; it is not a full-token exact contribution calculation. RMS times output-weight norm ignores cancellation and downstream effects. The compact checkpoint must be evaluated directly after export.
- The `k2` contribution, Michel and FFN results cannot be reused as scores for this mask. The two tasks use different attention patterns and inputs.

## Verification

Ruff, import/compile check, and the real saved-output parity checks are required. Inspect mask widths and provenance with `hooks.read_mask_artifact` before exporting. No new small-tensor test is needed for this reversible scoring proxy; the real parity and held-out rollout are the meaningful checks.
