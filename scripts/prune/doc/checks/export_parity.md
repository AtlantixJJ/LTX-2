# `checks/export_parity.py` — in-memory intervention versus export

## Objective

Test whether an exported native D0 checkpoint computes the same one-step
whole-video output as its in-memory source intervention. Width exports use
functional masks; depth exports use the ordered retained `ModuleList`. This is
separate from baseline-versus-candidate quality and from runtime measurement.

## Data flow

The CLI takes a saved baseline D0 directory, exactly one of `--masks` or
`--depth-artifact`, an exported checkpoint, held-out view and exact sigma list.
`data.whole_clip.build_input` reconstructs each input from the baseline alone.
The source transformer runs with head/FFN hooks or a temporary physically
shortened block list; the original block list is restored afterward. The exported
transformer loads sequentially and runs without interventions. The source
fingerprint, family artifact SHA256 and native task must match. Depth additionally
checks the retained key inventory, transformed config and parameter accounting.
Each output is compared by maximum absolute difference and relative L2; default
maximum absolute tolerance is 0.02, and nonfinite tolerances are rejected.
The command writes `export_parity.json` through `core.artifacts` and exits nonzero
on a failed numerical comparison.

```bash
python -m scripts.prune.checks.export_parity \
  --baseline <fresh-baseline-dir> \
  --masks <native-d0-mask.json> --exported-checkpoint <export.safetensors> \
  --view <held-out-view-path> --sigmas 0.725 0.909375 --gpu-id N
```

For depth exports, replace `--masks <native-d0-mask.json>` with
`--depth-artifact <native-d0-depth.json>`. Its result identifies depth and the
in-memory retained-block reference. A passing result establishes faithful
serialization of that smaller architecture, not its quality relative to the
original teacher. A cached identity-bypass model is not used as the reference:
skipped blocks would leave holes in the cache list.


## Invariants and checks

Both models see the same saved noise, capture hash, fps, geometry, clean first frame, context and sigma. One 22B transformer is resident at a time. A no-prune export should match the baseline control; compact BF16 shapes may exceed the tolerance. A passing parity check does not imply good decoded quality or speed. Focused CPU tests cover manifest and mask rejection; a real D0 parity run validates the model execution path.

Native mask distribution is bound to both a content-pinned calibration manifest and the selected baseline. Holdout excludes the actor across views. `compact_faithful` is a separate compact-storage mode that restores the original GEMM and attention geometry at execution to avoid the reduced-shape BF16 discrepancy; it uses the same 0.02 gate and does not claim reduced execution FLOPs.
