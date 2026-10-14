# learn/run_full

## Objective

Run and monitor the user-authorized full paired GT training through direct prune CLIs.

## Usage

Launch directly in ltx from LTX-2. No avatar queue is used.

```bash
CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 setsid python -u -m scripts.prune.learn.run_full --study /abs/frozen_study.json > /abs/run/logs/full.log 2>&1 < /dev/null &
```

Arguments select the study, arm list and replicate seed. Defaults are learned_joint and energy, seed 42.
Outputs include phase logs, immutable launch records, a mutable thirty-second monitoring/status.json and PID records.
See the prune run guide. Stop the recorded owner PID with SIGTERM; it forwards termination to owned child groups.

## Data flow

A frozen GT/FSDP study enters direct CPU score reuse, native FSDP acceptance, full selection and compact recovery.
Each learning phase reads the same paired draw list. Export and evaluation remain separate package CLIs.

## Organization logic

Verify four free GPUs before native work. Skip only already-completed, matching artifacts.
Run native parity, gradient and resume acceptance before the first sustained learner.
Start four ranks through torchrun. Internal twelve-hour learning bounds save the last completed update.
An external thirteen-hour timeout has ten-minute termination grace.
Resume only TimeoutError exits with increased saved progress. Other errors stop this assigned run.
Selection uses 1,000 updates; recovery uses 500. Export the compact base between phases and merge afterward.
At compact export, save missing seed-42 structural inputs on GPU zero without a transformer forward. CPU surgery follows in the same child.
This metadata step occurs after selection and needs no teacher or calibration pass.
After the pair completes, GT numeric validation and decoded quality review remain the next separate gates.

## Invariants

Only physical GPUs zero through three. No teacher preparation or speed testing.
No global job queue, reservation database or unrelated process control.
The driver starts only this study's direct children and monitors its own saved evidence.
Do not edit production sources during a live frozen study.

## Gotchas

A diagnostic pass does not establish pruning quality. A complete numeric validation does not establish perceptual quality.
The historical energy pool can be partial; its imported record declares actual coverage.

## Tests

Check frozen study validation and child failure handling. Native correctness is owned by fsdp_small and fsdp_native.
