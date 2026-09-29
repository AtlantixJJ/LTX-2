#!/usr/bin/env bash
# One iterative head-sparsity candidate and Phase-1 evaluation per GPU.
# Run from LTX-2 in the ltx env: bash scripts/prune/run_head_sweep.sh 2.5 "0 1 2 3"
set -uo pipefail

MODEL="${1:-2.5}"
read -r -a GPUS <<< "${2:-0 1 2 3}"
SPARSITIES=(${SPARSITY_LIST:-0.05 0.10 0.15 0.20 0.25 0.30 0.40})
MAX_RECORDS="${MAX_RECORDS:-24}"
ROUNDS="${ROUNDS:-2}"
PYTHON="${LTX_PYTHON:-$(conda run -n ltx which python 2>/dev/null | grep -m1 '/python' || true)}"
[ -x "$PYTHON" ] || PYTHON="/data1/users/jianjinx/miniconda3/envs/ltx/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

if [ "${#GPUS[@]}" -eq 0 ] || [ "${#SPARSITIES[@]}" -eq 0 ]; then
    echo "at least one GPU and sparsity are required" >&2
    exit 1
fi
if ! PREREQS="$("$PYTHON" -m scripts.prune.core.preflight --model "$MODEL" --check-sweep-prereqs)"; then
    echo "sweep prerequisites failed; rerun method_parity, calibration, and unpruned phase1_gates" >&2
    exit 1
fi
SWEEP_DIR="$("$PYTHON" -c 'import sys; from scripts.prune.core import artifacts; print(artifacts.run_dir(sys.argv[1], "head-sweep", script="run_head_sweep", argv=sys.argv[2:]))' "$MODEL" "$@")"
if [ ! -d "$SWEEP_DIR" ]; then
    echo "could not create sweep directory" >&2
    exit 1
fi
printf '%s\n' "$PREREQS" > "$SWEEP_DIR/prerequisites.json"

status_file() {
    "$PYTHON" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"sparsity": float(sys.argv[2]), "gpu": int(sys.argv[3]), "status": sys.argv[4], "scores": sys.argv[5] or None, "evaluation": sys.argv[6] or None}, indent=2))' \
        "$SWEEP_DIR/status_$1.json" "$2" "$3" "$4" "$5" "$6"
}

run_one() {
    local sparsity="$1" gpu="$2" tag="$3"
    local log="$SWEEP_DIR/${tag}_gpu${gpu}.log"
    local schedule="" evaluation="$SWEEP_DIR/phase1_${tag}.json"
    if ! "$PYTHON" -m scripts.prune.score.head_scores --model "$MODEL" --gpu-id "$gpu" \
        --methods michel --iterative-method michel --target-sparsity "$sparsity" \
        --rounds "$ROUNDS" --max-records "$MAX_RECORDS" >"$log" 2>&1; then
        status_file "$tag" "$sparsity" "$gpu" "score_failed" "" ""
        echo "[$tag] scoring failed: $log" >&2
        return 1
    fi
    schedule="$(grep -oE '/[^ ]*head_scores\.json' "$log" | tail -1)"
    if [ -z "$schedule" ] || [ ! -f "$schedule" ]; then
        status_file "$tag" "$sparsity" "$gpu" "score_missing" "" ""
        echo "[$tag] no head_scores.json: $log" >&2
        return 1
    fi
    if ! "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert abs(r["iterative"]["target_sparsity"]-float(sys.argv[2]))<1e-9' "$schedule" "$sparsity"; then
        status_file "$tag" "$sparsity" "$gpu" "score_mismatch" "$schedule" ""
        echo "[$tag] score report has the wrong sparsity" >&2
        return 1
    fi
    local t2_args=() rollout_args=()
    if [ -n "${T2_VIDEO:-}" ]; then
        t2_args=(--t2-video "$T2_VIDEO" --expected-source-sha256 "${T2_SHA256:?T2_SHA256 is required with T2_VIDEO}")
    fi
    if [ -n "${SWEEP_ROLLOUT_WINDOWS:-}" ]; then
        rollout_args=(--rollout-windows "$SWEEP_ROLLOUT_WINDOWS")
    fi
    if ! "$PYTHON" -m scripts.prune.evaluate.phase1_gates --model "$MODEL" --gpu-id "$gpu" \
        --head-masks "$schedule" --output "$evaluation" \
        --figures-dir "$SWEEP_DIR/figures_$tag" --t0-max-records 12 \
        "${t2_args[@]}" "${rollout_args[@]}" >>"$log" 2>&1; then
        status_file "$tag" "$sparsity" "$gpu" "evaluation_failed" "$schedule" "$evaluation"
        echo "[$tag] evaluation failed: $log" >&2
        return 1
    fi
    status_file "$tag" "$sparsity" "$gpu" "complete" "$schedule" "$evaluation"
    echo "[$tag] done: $evaluation"
}

i=0
failed=0
while [ "$i" -lt "${#SPARSITIES[@]}" ]; do
    pids=()
    for gpu in "${GPUS[@]}"; do
        [ "$i" -lt "${#SPARSITIES[@]}" ] || break
        tag="p$(awk -v v="${SPARSITIES[$i]}" 'BEGIN{printf "%02d", v*100+0.5}')"
        run_one "${SPARSITIES[$i]}" "$gpu" "$tag" &
        pids+=($!)
        i=$((i + 1))
        sleep "${SWEEP_STAGGER_SECONDS:-2}"
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
done
"$PYTHON" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); rows=[json.loads(p.read_text()) for p in sorted(root.glob("status_*.json"))]; (root/"sweep_manifest.json").write_text(json.dumps({"model":sys.argv[2],"candidates":rows,"pass":bool(rows) and all(row["status"]=="complete" for row in rows)},indent=2))' "$SWEEP_DIR" "$MODEL"
if [ "$failed" -ne 0 ]; then
    echo "sweep failed; see $SWEEP_DIR/sweep_manifest.json" >&2
    exit 1
fi
echo "sweep complete: $SWEEP_DIR/sweep_manifest.json"
