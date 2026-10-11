# `checks/export_parity.py` — in-memory intervention versus export

## Objective

Test whether an exported native D0 checkpoint computes the same one-step
whole-video output as its in-memory source intervention. Width exports use
functional masks; depth exports use the ordered retained `ModuleList`. This is
separate from baseline-versus-candidate quality and from runtime measurement.

## Usage

A human or agent can run the direct CLI from `LTX-2` in `ltx`.
Check `nvidia-smi` and select one free GPU. There is no `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.checks.export_parity \
  --baseline ../expr/refiner_prune/2.5/prune_doc_example/baseline \
  --masks ../expr/refiner_prune/2.5/prune_doc_example/mask.json \
  --exported-checkpoint ../expr/refiner_prune/2.5/prune_doc_example/candidate.safetensors \
  --view /abs/corpus/Part_1/0264_01/views/view00_cam51 \
  --sigmas 0.725 0.909375 1.0 --gpu-id 0 --max-abs 0.02
```

| Argument | Meaning |
|---|---|
| `--baseline` | Saved native source run including the held-out view |
| `--masks`, `--depth-artifact` | Exactly one native family artifact |
| `--exported-checkpoint` | Export to test |
| `--view`, `--sigmas` | Held-out view and calibrated levels |
| `--gpu-id` | Device for sequential model loads |
| `--max-abs` | Nonnegative finite tolerance; default 0.02 |

Output is `export_parity.json` in an indexed `d0-export-parity` run directory.
The process exits 1 on numerical failure.
See the [run guide](../usage.md#width-export-and-parity).

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

For depth exports, replace `--masks <native-d0-mask.json>` with
`--depth-artifact <native-d0-depth.json>`. Its result identifies depth and the
in-memory retained-block reference. A passing result establishes faithful
serialization of that smaller architecture, not its quality relative to the
original teacher. A cached identity-bypass model is not used as the reference:
skipped blocks would leave holes in the cache list.


## Organization logic

Validate artifact and export provenance before opening sessions.
Evaluate the source intervention at each sigma, keeping outputs on CPU.
Close the source model, then evaluate the exported model without hooks.
Compute maximum absolute difference and relative L2 in float32.

Worked example: deleting depth block 1 from four blocks compares the exported three-block model with source blocks `[0,2,3]`.
The temporary source block list is restored on exit.

## Invariants

Both models use the same saved noise, capture, fps, geometry, c0, context and sigma.
Only one transformer is resident at a time.

## Gotchas

A no-prune export should match its source control.
Compact BF16 shapes can exceed the tolerance.
Passing parity does not establish decoded quality or speed.

Native mask distribution is bound to both a content-pinned calibration manifest and the selected baseline. Holdout excludes the actor across views. `compact_faithful` is a separate compact-storage mode that restores the original GEMM and attention geometry at execution to avoid the reduced-shape BF16 discrepancy; it uses the same 0.02 gate and does not claim reduced execution FLOPs.

## Tests

`tests/test_export_pruned.py` and `tests/test_export_depth.py` check small-model export behavior.
Depth tests also check the mutually exclusive artifact parser.
Mask and manifest tests cover input rejection.
A real held-out BF16 parity run validates the native execution path.
