#!/usr/bin/env bash
# B2b (plan 2026-09-10 §5 B2b) — render ARGAvatar guides into the manifest's crop box.
#
# **This is the binding constraint on the whole project.** Every number measured so far --
# the IoU band, `r`, the preliminary training run -- rests on 2 pairs from 2 clips, view01
# only. T2 wants ~8 actors. Captures are the cheap half and are already 24 % of the way
# through all 8 views; renders are ~20 min each and are what a pair actually needs.
#
# Per §10 decision 7: keep RENDERING at 2 driving views even though the capture pass is doing
# all 8. Views 1 and 5 are opposite sides of the rig (front-right / back-left), so two renders
# per clip cover the two hardest poses rather than two similar ones.
#
#   LTX-2/scripts/onestep_avatar/run_b2b.sh 3            # GPU id, defaults to a 8-view review batch
#   LTX-2/scripts/onestep_avatar/run_b2b.sh 3 0          # 0 = no limit, render everything available
set -euo pipefail

GPU_ID="${1:?usage: run_b2b.sh <gpu-id> [limit] [driving-views...]}"
LIMIT="${2:-8}"
shift $(( $# > 2 ? 2 : $# ))
# Two real elements, not one "1 5" string: `VIEWS=("${@:-1 5}")` would produce a single
# element that only works because the expansion downstream is unquoted.
if [ "$#" -gt 0 ]; then VIEWS=("$@"); else VIEWS=(1 5); fi

# Since the 2026-09-15 consolidation this script lives in LTX-2/scripts/onestep_avatar/, so
# `../..` is the LTX-2 repo -- the directory `python -m scripts.onestep_avatar.*` must run
# from. Experiment output (expr/, data/) still lives one level up, in the workspace.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PY="/home/jianjinx/data2/miniconda3/envs/argavatar/bin/python"
LOG="$WORKSPACE/expr/onestep_avatar/logs/b2b_$(date +%Y%m%d-%H%M%S).log"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID")
if [ "$free_mib" -lt 20000 ]; then
  echo "GPU $GPU_ID has only ${free_mib} MiB free; a render peaks around 15 GB." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
echo "logging to $LOG"

# --visualize keeps argavatar_render.mp4 next to rgb.mp4 plus qa/overlay_view<D>.mp4.
# THE REVIEW GATE IS NOT OPTIONAL on the first batch: look at the overlays before rendering
# anything else. A crop, camera-convention or matte error is visible there and nowhere
# cheaper -- and every downstream number would inherit it silently.
env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u -m scripts.onestep_avatar.build_guidance \
  --driving-views "${VIEWS[@]}" \
  --limit "$LIMIT" \
  --visualize \
  --device cuda:0 \
  2>&1 | tee "$LOG"
