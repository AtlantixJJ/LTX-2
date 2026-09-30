# `evaluate/whole_clip_d0.py` — matched one-step pruning comparison

## Objective

Compare a compact or sparse pruned transformer with the unpruned LTX-2.5 transformer on the actual requested diagnostic: one bidirectional block covering each original DNARendering capture, with one denoising step from several start noise levels. This is an offline D0 capacity probe, not the `k2` sliding-window deployment gate.

Paired manifest validation, saved-noise equality and D0 latent path resolution are shared through `data.whole_clip`; the evaluator owns direction metrics and media assembly.

## Data flow

Run `scripts.onestep_avatar.visualize_d1` twice with identical `--view`, `--whole-clip`, `--sigmas`, `--prompt`, `--seed`, and model/guidance flags, changing only `--transformer`. Keep both fresh output directories. The producer saves the capture bundle hash, one global epsilon tensor per view, D0 predicted latents per sigma, and a decoded GT|D0|D1 MP4. Then run:

```bash
conda run -n ltx python -m scripts.prune.evaluate.whole_clip_d0 \
  --baseline BASELINE_OUTPUT --candidate PRUNED_OUTPUT --output COMPARISON_OUTPUT
```

The analyzer verifies matching model key and VAE fingerprint, objective, sigma list, seed, geometry, text context, guidance, source hashes, block plan, one-step schedule and **actual saved epsilon tensor**. It calculates noise-facing one-step direction `(x_sigma - predicted_x0) / sigma` from the exact `torch.lerp` noising mix, using only latent frames 1 onward. Latent frame 0 is the clean condition. It reports candidate versus baseline direction cosine and relative L2, plus MSE of each predicted latent against the capture latent. It assembles labeled GT capture VAE | baseline D0 VAE | pruned D0 VAE videos from the producer's already decoded panels; all panels have equal size and synchronized frames. `comparison.json` is the machine-readable output.

For a diagnostic before export, `--functional-mask <native-mask.json> --view <view> --sigmas <levels> --gpu-id N --output <ablation.json>` runs heads-only, FFN-only and combined masks on the baseline checkpoint. It verifies an unmasked direct forward against the saved D0 latent first. These scores describe functional-mask effects; they do not establish compact-export parity or timing. No candidate manifest is required in this mode.

## Invariants and gotchas

- A saved epsilon's file SHA can change with serialization. Compare tensor values, not the `.pt` file bytes.
- Both manifests must say `whole_clip: true` and `attention: full_bidirectional`. The schedule for every row must be exactly `[sigma, 0]`.
- A matching seed is insufficient proof of matching noise; verify the tensor values.
- The direction metric excludes the clean keyframe and describes deviation from the baseline model. Capture MSE is a separate accuracy reference. Neither metric alone determines perceptual quality; inspect the synchronized decoded videos.
- The three-panel producer MP4 contains D1 as its third panel; the analyzer crops the matched GT and D0 panels only. The new video title bars identify VAE decoding. Keep both source videos for provenance.

## Tests

`scripts/prune/tests/test_whole_clip_d0.py` checks a hand-computed direction comparison and rejects unmatched manifests. A real run should additionally verify source/noise hashes in `comparison.json` and inspect the assembled MP4s.
