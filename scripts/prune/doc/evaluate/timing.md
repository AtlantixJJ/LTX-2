# evaluate/timing

## Objective

Measure synchronized model forwards and summarize matched ABA/BAB timing brackets.

## Data flow

`measure_arm` loads one checkpoint in a session and checks its prediction against a saved rollout.
It warms the model, times repeated forwards in milliseconds and returns parameter and peak-memory observations.
`timed` measures an already resident velocity model in seconds.
`bracket` receives three arm results, their ABA or BAB order, and the timing field name.

## Organization logic

`measure_arm` restores clean frame zero and unpatchifies before checking the recorded output.
It rejects nonfinite values and maximum absolute differences above 0.02.
Each timed forward has CUDA synchronization before and after execution.
`timed` applies masks when supplied, runs two warmups, then measures seven forwards.
Callers reset peak memory before resident-model timing.
`bracket` averages the A observations and B observations separately.
Speedup is mean A divided by mean B. Saving is mean A minus mean B.
Outer-arm drift is the absolute last-minus-first difference divided by the first observation.

Worked example: ABA medians of 10, 8 and 10.2 give A=10.1, B=8, speedup=1.2625 and drift=0.02.
A caller can reject a gain when drift or timing spread prevents a reliable claim.
The V3 deployment owner requires both speedups at least 1.05 and both drift fractions below 0.02.

## Invariants

Keep checkpoint inputs, device, geometry, numerical policy and source noise fixed.
Use exactly three rows in ABA or BAB order. Keep milliseconds and seconds explicit.
A speed claim needs separate parity and quality evidence.

## Gotchas

The helper summarizes drift; each experiment owns its declared acceptance threshold.
`timed` does not reset peak memory or validate a saved prediction.
Different adapter execution paths must include their actual arithmetic in the timed call.

## Tests

`tests/test_bench_learned_heads.py` checks drift statistics and benchmark gate behavior.
Native same-device measurements remain required before a speed claim.
