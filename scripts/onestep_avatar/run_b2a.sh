#!/usr/bin/env bash
# B2a (plan 2026-09-10 §1.3) -- capture -> target latents, restartable.
#
# The first two attempts (09-11 evening, 09-12 morning) were both SIGKILLed (exit 137), and
# the inline "restarting in 15s" retry loop that was supposed to catch that died WITH the
# job it was supervising: it shared this shell's process group and session, so whatever
# signal reached the child reached the loop too. This script exists to not repeat that --
# launch it with `setsid` (see the usage note below) so the supervisor loop runs in its own
# session, immune to a signal delivered to the launching shell's process group or a SIGHUP
# from a closed terminal.
#
#   setsid -f scripts/onestep_avatar/run_b2a.sh 1 3 </dev/null >/dev/null 2>&1 &
#   disown
#
# `1` = GPU id (use one of 0-3), `3` = --crop-workers (load-bearing, see the README: the
# default os.cpu_count() fan-out is what caused the very first, pre-supervisor OOM on
# 09-10). Check `nvidia-smi`/`free -h` before raising either.
#
# Progress: tail -f the log below, or watch `capture_latent_manifest.json`'s sibling bundle
# count grow (`find <corpus> -name 'ltx_vae_latent*.pt' | wc -l`, target 6720 for both
# objectives). A restart looks
# like a hang for a while even with the plan cache (`.capture_plan_cache.json`, corpus root)
# now landed: only SOURCES NOT ALREADY CACHED cost a decode, but a bundle write only happens
# once a source's VAE encode finishes, which is still one full source at a time on this GPU.
set -uo pipefail

GPU_ID="${1:?usage: run_b2a.sh <gpu-id 0-3> [crop-workers]}"
CROP_WORKERS="${2:-3}"

# Since the 2026-09-15 consolidation this script lives in LTX-2/scripts/onestep_avatar/, so
# `../..` is the LTX-2 repo -- the directory `python -m scripts.onestep_avatar.*` must run
# from. Experiment output (expr/, data/) still lives one level up, in the workspace.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PY="/home/jianjinx/data2/miniconda3/envs/ltx/bin/python"
LOG="$WORKSPACE/expr/onestep_avatar/logs/precompute_capture_only_gpu${GPU_ID}.log"
PIDFILE="$WORKSPACE/expr/onestep_avatar/logs/precompute_capture_only_gpu${GPU_ID}.supervisor.pid"

mkdir -p "$(dirname "$LOG")"
cd "$ROOT"

echo $$ > "$PIDFILE"
{
  echo "=== $(date -Iseconds) supervisor started, pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ') pgid=$(ps -o pgid= -p $$ | tr -d ' ') ==="
} >> "$LOG"

while true; do
  free_before=$(free -m | awk '/^Mem:/{print $7}')
  {
    echo "=== $(date -Iseconds) launching precompute --capture-only (gpu=$GPU_ID crop-workers=$CROP_WORKERS free_mem=${free_before}MiB) ==="
  } >> "$LOG"

  "$PY" -u -m scripts.onestep_avatar.precompute \
    --model 2.5 --gpu-id "$GPU_ID" --capture-only \
    --objective bg white \
    --views 0 1 2 3 4 5 6 7 --edge 1024 --pad-factor 1.2 --crop-workers "$CROP_WORKERS" \
    >> "$LOG" 2>&1
  code=$?

  if [ "$code" -eq 0 ]; then
    echo "=== $(date -Iseconds) completed successfully, exit 0 ===" >> "$LOG"
    break
  fi
  echo "=== $(date -Iseconds) exited with code $code, restarting in 15s ===" >> "$LOG"
  sleep 15
done

rm -f "$PIDFILE"
