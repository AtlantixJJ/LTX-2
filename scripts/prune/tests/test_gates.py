from __future__ import annotations

from scripts.prune import gates


def _candidate(windows):
    return {
        "T0": {"held_out": {"rel_l2_chunk": {"mean": 0.3}}},
        "T1": {"psnr_vs_source": 30.0},
        "T2": {"windows": windows},
        "T3": {"video": "x.mp4"},
    }


def test_short_rollout_fails():
    verdict = gates.verdict(baseline=_candidate(200), candidate=_candidate(4), profile=None, baseline_profile=None, minimum_speedup=1.4, rollout_chunks=200)
    assert verdict["pass"] is False and "200-chunk" in verdict["reason"]


def test_missing_baseline_fails_even_with_full_rollout():
    verdict = gates.verdict(baseline=None, candidate=_candidate(200), profile=None, baseline_profile=None, minimum_speedup=1.4, rollout_chunks=200)
    assert verdict["pass"] is False


def test_legacy_student_chunks_field_still_evaluates():
    old = _candidate(0)
    old["T2"] = {"student_chunks": 200}
    verdict = gates.verdict(baseline=_candidate(200), candidate=old, profile=None, baseline_profile=None, minimum_speedup=1.4, rollout_chunks=200)
    assert verdict["pass"] is True


def test_t0_delta_is_candidate_minus_baseline():
    baseline, candidate = _candidate(200), _candidate(200)
    candidate["T0"]["held_out"]["rel_l2_chunk"]["mean"] = 0.35
    verdict = gates.verdict(baseline=baseline, candidate=candidate, profile=None, baseline_profile=None, minimum_speedup=1.4, rollout_chunks=200)
    assert abs(verdict["T0_delta"] - 0.05) < 1e-9
