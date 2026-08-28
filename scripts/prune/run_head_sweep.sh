#!/usr/bin/env bash
# Head-pruning sparsity sweep across 4 GPUs, at the DEPLOYED refine geometry.
#
# Run from the LTX-2 repo root in the `ltx` conda env:
#     bash scripts/prune/run_head_sweep.sh 2.5 "0 1 2 3"
#
# Per sparsity target it does two things on one GPU:
#   1. scripts.prune.head_scores  --target-sparsity P   -> a head_scores.json whose
#      `iterative.masks` is the exact per-attention-module mask
#   2. scripts.prune.phase1_gates --head-masks <that>   -> the full T0/T1/T2/T3 gate
#
# Prerequisites, in order (this script checks for the first two and refuses otherwise):
#   scripts.prune.method_parity      -> PASS  (the harness reproduces the refine script)
#   scripts.prune.source_target --build-calibration  (format-2 records at ctx=1, real fps)
#   scripts.prune.phase1_gates ... --output phase1_gates_baseline.json
#
# Two operational bugs this script exists to avoid, both hit by the first sweep:
#   * `conda run` appends a blank line to stdout, so `$(... | tail -1)` captured "" and
#     `--head-masks ""` resolved to Path(".") -> IsADirectoryError. The path is extracted
#     with grep on the expected filename instead.
#   * provenance.run_id() HAD 1-second resolution, so two jobs finishing in the same second landed in
#     the same output directory and one clobbered the other's head_scores.json (a launch stagger does
#     not help -- run_id is called at the END). run_id now carries a PID; each job still
#     re-reads its own schedule's target_sparsity to prove it got the file it asked for.
set -uo pipefail

MODEL="${1:-2.5}"
read -r -a GPUS <<< "${2:-0 1 2 3}"
SPARSITIES=(${SPARSITY_LIST:-0.05 0.10 0.15 0.20 0.25 0.30 0.40})
MAX_RECORDS="${MAX_RECORDS:-24}"
ROUNDS="${ROUNDS:-2}"
PYTHON="${LTX_PYTHON:-$(conda run -n ltx which python 2>/dev/null | tail -1)}"
[ -x "$PYTHON" ] || PYTHON="/home/jianjinx/data/miniconda3/envs/ltx/bin/python"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/../expr/refiner_prune/$MODEL"
cd "$ROOT" || exit 1

[ -f "$OUT/method_parity.json" ] || { echo "missing $OUT/method_parity.json -- run scripts.prune.method_parity first"; exit 1; }
grep -q '"pass": true' "$OUT/method_parity.json" || { echo "method_parity did not PASS -- fix that before measuring pruning deltas"; exit 1; }
[ -f "$OUT/calibration/index.json" ] || { echo "missing $OUT/calibration/index.json -- run scripts.prune.source_target --build-calibration first"; exit 1; }

run_one() {
    local sparsity="$1" gpu="$2"
    local tag; tag="p$(awk -v v="$sparsity" 'BEGIN{printf "%02d", v*100+0.5}')"
    local log="$OUT/sweep_${tag}_gpu${gpu}.log"
    {
        echo "=== $tag on GPU $gpu: head_scores --target-sparsity $sparsity"
        "$PYTHON" -m scripts.prune.head_scores --model "$MODEL" --gpu-id "$gpu" \
            --methods michel --iterative-method michel \
            --target-sparsity "$sparsity" --rounds "$ROUNDS" --max-records "$MAX_RECORDS"
    } >"$log" 2>&1
    local schedule
    schedule="$(grep -oE '/[^ ]*head_scores\.json' "$log" | tail -1)"
    if [ -z "$schedule" ] || [ ! -f "$schedule" ]; then
        echo "[$tag] FAILED: no head_scores.json produced (see $log)"
        return 1
    fi
    # Belt-and-braces against a clobbered schedule: the file on disk must be the one this
    # job asked for. provenance.run_id() now carries a PID so directories cannot collide,
    # but the check is cheap and a wrong mask is invisible in the resulting numbers.
    # Compared NUMERICALLY -- json writes 0.10 back out as 0.1, and a string compare here
    # failed a perfectly good p10 run.
    if ! "$PYTHON" -c "import json,sys; sys.exit(0 if abs(json.load(open(sys.argv[1]))['iterative']['target_sparsity'] - float(sys.argv[2])) < 1e-9 else 1)" "$schedule" "$sparsity"; then
        echo "[$tag] FAILED: $schedule is not the target_sparsity=$sparsity schedule this job built"
        return 1
    fi
    {
        echo "=== $tag on GPU $gpu: phase1_gates --head-masks $schedule"
        "$PYTHON" -m scripts.prune.phase1_gates --model "$MODEL" --gpu-id "$gpu" \
            --head-masks "$schedule" \
            --output "$OUT/phase1_gates_pruned_${tag}.json" \
            --figures-dir "$OUT/figures_pruned_${tag}" \
            --t0-max-records 12
    } >>"$log" 2>&1 || { echo "[$tag] FAILED in phase1_gates (see $log)"; return 1; }
    echo "[$tag] done -> $OUT/phase1_gates_pruned_${tag}.json"
}

i=0
while [ "$i" -lt "${#SPARSITIES[@]}" ]; do
    pids=()
    for gpu in "${GPUS[@]}"; do
        [ "$i" -lt "${#SPARSITIES[@]}" ] || break
        run_one "${SPARSITIES[$i]}" "$gpu" &
        pids+=($!)
        i=$((i + 1))
        sleep 2
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
done
echo "sweep complete; results under $OUT"
