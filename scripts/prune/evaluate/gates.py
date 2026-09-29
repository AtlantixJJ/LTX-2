"""Evaluate matched pruning artifacts against explicit quality, coverage, and speed gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read(path: Path | None):
    return json.loads(path.read_text()) if path is not None else None


def _get(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _source_fingerprint(data):
    return (_get(data, "source_transformer_fingerprint")
            or _get(data, "provenance", "source_transformer_fingerprint")
            or _get(data, "provenance", "transformer_fingerprint"))


def _profile_ms(profile, evaluation):
    """Bare legacy profile lists have no run identity and cannot prove a speedup."""
    if not isinstance(profile, dict) or not isinstance(evaluation, dict):
        return None
    for key in ("geometry", "seed", "clip"):
        expected = _get(evaluation, "T2", "clip") if key == "clip" else evaluation.get(key)
        if profile.get(key) != expected:
            return None
    if _source_fingerprint(profile) != _source_fingerprint(evaluation):
        return None
    rows = profile.get("rows")
    if not isinstance(rows, list) or len(rows) != _get(evaluation, "T2", "windows") or not rows:
        return None
    samples = []
    for index, row in enumerate(rows):
        if _get(row, "window_index") != index:
            return None
        value = _number(_get(row, "refine_s"))
        if value is None or value <= 0:
            return None
        samples.append(value)
    # The first window pays kernel warm-up; the remaining windows are the
    # steady-state deployment quantity the pruning speed gate measures.
    steady = samples[1:] if len(samples) > 1 else samples
    return 1000 * sum(steady) / len(steady)


def verdict(*, baseline: dict | None, candidate: dict | None, profile: dict | None, baseline_profile: dict | None,
            minimum_speedup: float = 1.4, rollout_chunks: int = 200, mode: str = "long",
            max_t0_delta: float = 0.05, max_t1_psnr_drop_db: float = 0.5,
            max_t2_slope_drop_db_per_100: float = 5.0) -> dict:
    if mode not in {"short", "long"} or rollout_chunks < 2 or _number(minimum_speedup) is None or minimum_speedup <= 0:
        raise ValueError("mode must be short/long, rollout_chunks >= 2, and minimum_speedup positive")
    if mode == "long" and rollout_chunks < 200:
        raise ValueError("long-form verdict requires at least 200 windows")
    if any(_number(v) is None or v < 0 for v in (max_t0_delta, max_t1_psnr_drop_db, max_t2_slope_drop_db_per_100)):
        raise ValueError("quality tolerances must be finite and nonnegative")
    baseline_ok, candidate_ok = isinstance(baseline, dict), isinstance(candidate, dict)
    base, cand = baseline if baseline_ok else {}, candidate if candidate_ok else {}
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def require(name, passed, reason):
        checks[name] = bool(passed)
        if not passed:
            reasons.append(reason)

    require("artifacts", baseline_ok and candidate_ok, "baseline/candidate evaluation missing")
    same = all(base.get(k) is not None and base.get(k) == cand.get(k)
               for k in ("geometry", "seed", "student_sigmas", "target"))
    same &= all(_get(base, "provenance", k) is not None and
                _get(base, "provenance", k) == _get(cand, "provenance", k)
                for k in ("model_key", "video_vae_path"))
    same &= _source_fingerprint(base) is not None and _source_fingerprint(base) == _source_fingerprint(cand)
    same &= all(_get(base, "T2", k) is not None and _get(base, "T2", k) == _get(cand, "T2", k)
                for k in ("clip", "source_sha256", "fps", "stride_frames", "source_frame_windows"))
    require("matched_inputs", same, "checkpoint, geometry, seed, schedule, target or source clip differs")

    b0, c0 = _number(_get(base, "T0", "held_out", "rel_l2_chunk", "mean")), _number(_get(cand, "T0", "held_out", "rel_l2_chunk", "mean"))
    t0_delta = c0 - b0 if b0 is not None and c0 is not None else None
    require("T0", t0_delta is not None and t0_delta <= max_t0_delta, "held-out chunk T0 missing or exceeds delta ceiling")
    b1, c1 = _number(_get(base, "T1", "psnr_vs_teacher")), _number(_get(cand, "T1", "psnr_vs_teacher"))
    t1_drop = b1 - c1 if b1 is not None and c1 is not None else None
    require("T1", t1_drop is not None and t1_drop <= max_t1_psnr_drop_db, "T1 PSNR missing or drop exceeds ceiling")
    bs, cs = _number(_get(base, "T2", "psnr_slope_db_per_100_chunks")), _number(_get(cand, "T2", "psnr_slope_db_per_100_chunks"))
    slope_drop = bs - cs if bs is not None and cs is not None else None
    windows = (_get(base, "T2", "windows"), _get(cand, "T2", "windows"))
    require("T2_coverage", all(isinstance(v, int) and v >= rollout_chunks for v in windows),
            f"{mode} rollout has fewer than {rollout_chunks} matched windows")
    require("T2_drift", slope_drop is not None and slope_drop <= max_t2_slope_drop_db_per_100,
            "T2 PSNR slope missing or degraded beyond ceiling")
    media = [_get(cand, "T3", name) for name in ("grid", "video")]
    require("T3", all(isinstance(p, str) and Path(p).is_file() for p in media),
            "candidate T3 grid or synchronized video missing")
    hardware_match = isinstance(baseline_profile, dict) and isinstance(profile, dict) and all(
        baseline_profile.get(k) is not None and baseline_profile.get(k) == profile.get(k)
        for k in ("gpu_name", "gpu_index")
    )
    base_ms, cand_ms = _profile_ms(baseline_profile, base), _profile_ms(profile, cand)
    speedup = base_ms / cand_ms if base_ms is not None and cand_ms is not None else None
    require("speed", hardware_match and speedup is not None and speedup >= minimum_speedup,
            "matched timing profiles missing or speedup below threshold")
    return {"mode": mode, "required_rollout_chunks": rollout_chunks,
            "thresholds": {"minimum_speedup": minimum_speedup, "max_t0_delta": max_t0_delta,
                           "max_t1_psnr_drop_db": max_t1_psnr_drop_db,
                           "max_t2_slope_drop_db_per_100": max_t2_slope_drop_db_per_100},
            "measurements": {"T0_delta": t0_delta, "T1_psnr_drop_db": t1_drop,
                             "T2_slope_drop_db_per_100": slope_drop, "baseline_ms_per_window": base_ms,
                             "candidate_ms_per_window": cand_ms, "speedup": speedup},
            "checks": checks, "reasons": reasons, "pass": all(checks.values())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--baseline-profile", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("short", "long"), default="long")
    p.add_argument("--rollout-chunks", type=int, default=200)
    p.add_argument("--minimum-speedup", type=float, default=1.4)
    p.add_argument("--max-t0-delta", type=float, default=0.05)
    p.add_argument("--max-t1-psnr-drop-db", type=float, default=0.5)
    p.add_argument("--max-t2-slope-drop-db-per-100", type=float, default=5.0)
    a = p.parse_args()
    out = verdict(baseline=_read(a.baseline), candidate=_read(a.candidate), profile=_read(a.profile),
                  baseline_profile=_read(a.baseline_profile), minimum_speedup=a.minimum_speedup,
                  rollout_chunks=a.rollout_chunks, mode=a.mode, max_t0_delta=a.max_t0_delta,
                  max_t1_psnr_drop_db=a.max_t1_psnr_drop_db,
                  max_t2_slope_drop_db_per_100=a.max_t2_slope_drop_db_per_100)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
