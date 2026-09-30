# `scripts/prune` — native whole-video D0 pruning

The active pruning experiment calibrates and judges an LTX-2.5 transformer on the **original complete capture video**: one full-bidirectional D0 forward from each specified noise level, with the first latent frame kept clean. Compare the unpruned and pruned **noise directions and VAE-decoded videos**. The older k2 sliding-window refiner study is [historical](doc/HISTORY.md) for this decision; its deployed-method regression check remains in place.

Run commands from the LTX-2 root in the `ltx` conda environment. Check `nvidia-smi` before a model run. Generated artifacts belong under the workspace's ignored `expr/refiner_prune/2.5/` tree.

## Active sequence

1. **Saved baseline:** use a `scripts.onestep_avatar.visualize_d1 --whole-clip` run on the original D0 capture with full bidirectional attention, exact `[sigma,0]` schedules, saved epsilon and D0 latents. The current three-actor [baseline manifest](../../../expr/onestep_avatar/d1_diagnostic/ar_sigma_rollouts/runs/s1d_prompts_20260929/P1_3actors/manifest.json) is an example.
2. **Calibrate and rank:** `python -m scripts.prune.score.whole_clip_d0_scores --baseline <baseline-dir> --view <calibration-view> [--view <second-view>] --sigmas 0.725 0.909375 1.0 --head-fraction 0.10 --ffn-fraction 0.10 --gpu-id N --output <native-mask.json>`. This reads the baseline only and verifies every calibration forward against the saved latent.
3. **Export:** `python -m scripts.prune.score.export_pruned --model 2.5 --masks <native-mask.json> --mode compact --output <candidate.safetensors>`. Native masks require task, source checkpoint, attention, conditioning, calibration-view and sigma provenance. `--historical-k2-mask` explicitly permits an older mask for a separate control experiment.
4. **Export parity:** `python -m scripts.prune.checks.export_parity --baseline <baseline-dir> --masks <native-mask.json> --exported-checkpoint <candidate.safetensors> --view <held-out-view> --sigmas 0.725 0.909375 --gpu-id N`. This checks functional-mask versus exported-model output at a default maximum absolute tolerance of 0.02. It exits nonzero on failure. `--historical-k2` selects the old record/two-window gate explicitly.
5. **Held-out output:** run the same `visualize_d1 --whole-clip` capture/noise/context/sigma setup with `--transformer <candidate.safetensors>`. Then run `python -m scripts.prune.evaluate.whole_clip_d0 --baseline <baseline-dir> --candidate <candidate-dir> --output <comparison-dir>`. The evaluator checks actual saved noise tensor equality and creates synchronized GT capture VAE | baseline D0 VAE | candidate D0 VAE videos.
6. **Cost:** run `python -m scripts.prune.evaluate.bench_whole_clip_d0 --baseline <baseline-dir> --candidate <candidate-dir> --view <held-out-view> --sigma 0.909375 --gpu-id N --order ABA --output <benchmark.json>`. Use a reversed order if the result is near the bracket drift. The benchmark times a warmed, synchronized dense-transformer forward; it records peak allocated memory separately.

A native candidate is acceptable only after input identity, export parity, held-out direction and decoded quality, and an actual same-device speed gain beyond measured drift are established. Parameter count or a functional sparse mask alone is not a speed result. The [current report](../../../expr/refiner_prune/2.5/FINDINGS.md) rejects the combined 10% head/FFN screen on these criteria.

## Documentation and tests

Start with [methods](doc/METHODS.md), [architecture](doc/ARCHITECTURE.md), [validation](doc/VALIDATION.md), and [history](doc/HISTORY.md). The [module index](doc/README.md) retains one design page for every production script. The local [CLAUDE.md](CLAUDE.md) states code invariants.

```bash
python -m pytest scripts/prune/tests -q -m 'not gpu'
uv run ruff check scripts/prune/data/whole_clip.py scripts/prune/score/whole_clip_d0_scores.py scripts/prune/evaluate/whole_clip_d0.py scripts/prune/evaluate/bench_whole_clip_d0.py scripts/prune/checks/export_parity.py
```

`checks/method_parity --model 2.5 --gpu-id N --windows 3` guards tensor-affecting changes to the shared deployed k2 rollout. Old experiment commands and source files remain available during the call-site audit; they are not the active D0 acceptance path.
