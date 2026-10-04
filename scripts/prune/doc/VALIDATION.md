# Whole-clip validation

1. Reconstruct each saved baseline input: capture, VAE, fps, geometry, prompt
   bytes, seed, original epsilon, clean frame 0 and exact `[sigma, 0]` schedule.
   Calibration checks the direct output against the saved D0 latent.
2. Require the separate `whole_clip_d0_mask_v1` width or
   `whole_clip_d0_depth_v1` intact-block format, native task/attention,
   checkpoint/VAE identity and content-pinned calibration distribution.
   Reject calibration actors across views, clips and Parts, including filesystem
   aliases, and reject uncalibrated held-out sigmas.
3. Compare the functional width mask or in-memory ordered retained blocks against
   the exported checkpoint on identical inputs using
   `checks.export_parity`. The default maximum absolute tolerance is 0.02.
   Include a no-prune export control. Compare baseline against candidate separately.
4. Report generated-frame direction relative L2 and cosine, latent capture MSE
   and synchronized capture/baseline/candidate VAE videos on held-out actors.
5. Measure repeated warmed, synchronized forwards on one GPU with ABA/BAB
   bracketing. A speed claim must exceed baseline-arm drift. Report allocated
   memory separately; parameter count alone cannot establish a speed gain.

Run CPU tests with `-m 'not gpu'` and check changed code with Ruff. Fresh GPU
baseline/export checks are required before interpreting new experimental results.

`test_export_depth.py` covers physical depth surgery with deterministic real
small video/AV models, strict checkpoint reload and cached/full block-causal
forward comparisons. Its FP32 CPU results do not replace the native BF16 gate.
