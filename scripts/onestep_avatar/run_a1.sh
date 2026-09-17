#!/usr/bin/env bash
# A1 (plan 2026-09-10 §5 A1) — characterise the distilled map at sigma0 = 0.725.
#
# This is the job that decides D1 vs D2. §0.3 measured the gap the LoRA must close as
# r = 0.89-0.93 on the subject interior, against §4.2's "r >~ 0.6 means D1 is overpowering the
# prior". That verdict is only safe next to the base model's OWN excursion
# a = ||Phi(x_sigma0) - z_g|| / sqrt(d): if a is also ~0.9, the task is the size of what the
# model already does and D1 is not overreaching; if a is ~0.3, it is.
#
# `--pairs` is the CORPUS ROOT, not an experiment tree: since SS4.4 (2026-09-14) every master
# latent lives beside its source video and `expr/onestep_avatar/precomputed/` is neither
# written nor read. Pointing --pairs there made `stats.measure_pairs` find no guide bundle and
# exit before any measurement ran.
#
# Runs on the REAL corpus guides by default, not the ARGAvatar smoke renders: `a` is only
# comparable with `r` at the same geometry, and the corpus guides are the deployed 1024**2 /
# 4096-token window. Needs ONE free GPU for ~2 h.
#
#   LTX-2/scripts/onestep_avatar/run_a1.sh 2          # GPU id
set -euo pipefail

GPU_ID="${1:?usage: run_a1.sh <gpu-id> [max-videos] [eps-samples]}"
MAX_VIDEOS="${2:-4}"
EPS_SAMPLES="${3:-8}"

# Since the 2026-09-15 consolidation this script lives in LTX-2/scripts/onestep_avatar/, so
# `../..` is the LTX-2 repo -- the directory `python -m scripts.onestep_avatar.*` must run
# from. Experiment output (expr/, data/) still lives one level up, in the workspace.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PY="/home/jianjinx/data2/miniconda3/envs/ltx/bin/python"
OUT="$WORKSPACE/expr/onestep_avatar/analysis_summary.json"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID")
if [ "$free_mib" -lt 34000 ]; then
  echo "GPU $GPU_ID has only ${free_mib} MiB free; the video-only transformer needs ~28 GB plus the VAE." >&2
  exit 1
fi

cd "$ROOT"
exec "$PY" -u -m scripts.onestep_avatar.stats \
  --gpu-id "$GPU_ID" \
  --renders "$WORKSPACE/data/AnimatableHuman/DNARenderingVideo" \
  --render-glob "argavatar_render.mp4" \
  --pairs "$WORKSPACE/data/AnimatableHuman/DNARenderingVideo" \
  --max-videos "$MAX_VIDEOS" \
  --eps-samples "$EPS_SAMPLES" \
  --out "$OUT"
