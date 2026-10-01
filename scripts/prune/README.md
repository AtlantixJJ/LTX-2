# Whole-clip bidirectional pruning

Calibrate and judge attention-head and FFN-channel pruning using one full-video,
bidirectional D0 forward at each exact `[sigma, 0]` schedule. Frame 0 is a clean
capture condition; all later frames are generated. Use fresh baseline, calibration,
export and evaluation directories under `../expr/refiner_prune/2.5/`.

Run from the LTX-2 root in the `ltx` conda environment. Check `nvidia-smi` before
loading the transformer.

## Workflow

1. Save a baseline with `python -m scripts.onestep_avatar.visualize_d1 --whole-clip`.
   Use the original capture, unguided BF16 inference, one step at each sigma, and
   saved epsilon and output latents. Fix prompt, geometry, fps and seed.
   Whole-clip attention uses no dense mask. Keep sigma and token timesteps in
   float32 as in the stock pipeline; BF16 applies to the weights and latents.
2. Calibrate on selected actors and exact sigma levels:
   `python -m scripts.prune.score.whole_clip_d0_scores --baseline <baseline-dir> --view <calibration-view> --sigmas 0.725 0.909375 1.0 --head-fraction 0.10 --ffn-fraction 0.10 --gpu-id N --output <mask.json>`.
   Every calibration forward must reproduce its saved baseline.
3. Export:
   `python -m scripts.prune.score.export_pruned --model 2.5 --masks <mask.json> --mode compact_faithful --output <candidate.safetensors>`.
   `masked_full` is the default full-width control. `compact_faithful` compresses
   stored parameters while executing original GEMM shapes; `compact` reduces
   supported execution shapes and needs independent numerical validation.
   `sparse` selects retained attention heads with full-width projections.
4. Verify the export against the functional mask on a held-out actor:
   `python -m scripts.prune.checks.export_parity --baseline <baseline-dir> --masks <mask.json> --exported-checkpoint <candidate.safetensors> --view <held-out-view> --sigmas 0.725 0.909375 --gpu-id N`.
   The default maximum absolute tolerance is 0.02; failure exits nonzero.
5. Generate the same saved whole-clip setup with
   `visualize_d1 --whole-clip --transformer <candidate.safetensors>`, then compare:
   `python -m scripts.prune.evaluate.whole_clip_d0 --baseline <baseline-dir> --candidate <candidate-dir> --output <comparison-dir>`.
   This verifies input identity and renders synchronized capture VAE, baseline
   D0 VAE and candidate D0 VAE panels.
6. Measure cost:
   `python -m scripts.prune.evaluate.bench_whole_clip_d0 --baseline <baseline-dir> --candidate <candidate-dir> --view <held-out-view> --sigma 0.909375 --gpu-id N --order ABA --output <benchmark.json>`.
   Repeat in reversed order when the effect is near measured bracket drift.

Use `evaluate.whole_clip_d0 --functional-mask <mask.json> --view <held-out-view>
--sigmas <levels> --gpu-id N` with baseline and output arguments for head-only,
FFN-only and combined ablations before exporting. Held-out actors must differ
from calibration actors across all views. All mask consumers require pinned
native whole-clip provenance.

## Validation and design

Read [methods](doc/METHODS.md), [architecture](doc/ARCHITECTURE.md),
[validation](doc/VALIDATION.md), [module docs](doc/README.md) and
[working rules](CLAUDE.md).

```bash
conda run -n ltx python -m pytest scripts/prune/tests -q -m 'not gpu'
conda run -n ltx ruff check --select F,I scripts/prune
```

A smaller checkpoint or a sparse functional mask is not a speed result. Acceptance
requires matched inputs, export parity, held-out direction and decoded quality,
and measured same-device latency improvement beyond drift.
