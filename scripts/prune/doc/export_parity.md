# `checks/export_parity.py` — functional mask versus export

## Objective

Test whether an exported native D0 checkpoint computes the same one-step whole-video output as the source checkpoint with its mask attached. This is separate from baseline-versus-candidate quality and from runtime measurement.

## Data flow

The default CLI takes a saved baseline D0 directory, native mask artifact, exported checkpoint, held-out view and exact sigma list. `data.whole_clip.build_input` reconstructs each input from the baseline alone. The source transformer runs with functional head/FFN hooks; the exported transformer loads sequentially and runs without hooks. A `source` checkpoint fingerprint, mask SHA256 and `task=whole_clip_d0` in export metadata must match. Each output is compared by maximum absolute difference and relative L2; default maximum absolute tolerance is 0.02. The command writes `export_parity.json` through `core.artifacts` and exits nonzero on a failed numerical comparison.

```bash
python -m scripts.prune.checks.export_parity \
  --baseline ../expr/onestep_avatar/d1_diagnostic/ar_sigma_rollouts/runs/s1d_prompts_20260929/P1_3actors \
  --masks <native-d0-mask.json> --exported-checkpoint <export.safetensors> \
  --view <held-out-view-path> --sigmas 0.725 0.909375 --gpu-id N
```

`--historical-k2` explicitly selects the older frozen-record and two-window check, with its original `--model`, `--states`, `--video` and source-hash arguments. It remains available for deployed-refiner regression work, but is not native D0 evidence.

## Invariants and checks

Both models see the same saved noise, capture hash, fps, geometry, clean first frame, context and sigma. One 22B transformer is resident at a time. A no-prune export should match the baseline control; compact BF16 shapes may exceed the tolerance. A passing parity check does not imply good decoded quality or speed. Focused CPU tests cover manifest and mask rejection; a real D0 parity run validates the model execution path.

Native mask distribution is bound to both a content-pinned calibration manifest and the selected baseline. Holdout excludes the actor across views. `compact_faithful` is a separate compact-storage mode that restores the original GEMM and attention geometry at execution to avoid the reduced-shape BF16 discrepancy; it uses the same 0.02 gate and does not claim reduced execution FLOPs.
