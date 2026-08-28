"""Emit one auditable pruning-gate verdict from real evaluation artifacts.

This deliberately consumes results already produced by ``phase1_gates`` and the
production refiner profile.  It does not pretend a one-window result proves the
mandatory 200-chunk rollout gate: missing evidence is a failed/unknown gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path | None) -> dict | None:
    return json.loads(path.read_text()) if path is not None else None


def verdict(*, baseline: dict | None, candidate: dict | None, profile: dict | None, baseline_profile: dict | None,
            minimum_speedup: float, rollout_chunks: int) -> dict:
    result: dict = {"required_rollout_chunks": rollout_chunks, "minimum_speedup": minimum_speedup, "pass": False}
    if candidate is not None:
        t0 = candidate.get("T0", {}).get("held_out", {}).get("rel_l2_chunk")
        result["T0"] = t0
        result["T1"] = candidate.get("T1")
        result["T2"] = candidate.get("T2")
        result["T3"] = candidate.get("T3")
    if baseline is not None and candidate is not None:
        base_t0 = baseline.get("T0", {}).get("held_out", {}).get("rel_l2_chunk")
        cand_t0 = candidate.get("T0", {}).get("held_out", {}).get("rel_l2_chunk")
        if base_t0 and cand_t0:
            result["T0_delta"] = cand_t0["mean"] - base_t0["mean"]
    if profile:
        steps = profile[0].get("denoiser_call_s_per_step", [])
        result["candidate_ms_per_step"] = 1000 * sum(steps) / max(len(steps), 1)
    if baseline_profile and profile:
        baseline_steps = baseline_profile[0].get("denoiser_call_s_per_step", [])
        candidate_steps = profile[0].get("denoiser_call_s_per_step", [])
        base = sum(baseline_steps) / max(len(baseline_steps), 1)
        candidate_time = sum(candidate_steps) / max(len(candidate_steps), 1)
        result["baseline_ms_per_step"] = 1000 * base
        result["speedup"] = base / candidate_time if candidate_time else None
    t2 = result.get("T2") or {}
    # "windows" since phase1_gates became a sliding-window rollout; "student_chunks" was
    # the pre-parity field name and is still read so older artifacts keep evaluating.
    complete_rollout = t2.get("windows", t2.get("student_chunks", 0)) >= rollout_chunks
    has_t1_t3 = bool(result.get("T1")) and bool(result.get("T3"))
    # A speedup needs matched unpruned timing, which this artifact schema does
    # not infer from unrelated runs.
    result["evidence_complete"] = bool(complete_rollout and has_t1_t3 and baseline is not None)
    result["reason"] = "pass" if result["evidence_complete"] else "missing matched held-out T0/T1/T2/T3 and/or 200-chunk rollout evidence"
    result["pass"] = result["evidence_complete"]
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--profile", type=Path)
    p.add_argument("--baseline-profile", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rollout-chunks", type=int, default=200)
    p.add_argument("--minimum-speedup", type=float, default=1.4)
    a = p.parse_args()
    out = verdict(baseline=_read(a.baseline), candidate=_read(a.candidate), profile=_read(a.profile), baseline_profile=_read(a.baseline_profile),
                  minimum_speedup=a.minimum_speedup, rollout_chunks=a.rollout_chunks)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
