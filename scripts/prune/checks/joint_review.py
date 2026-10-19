"""Review complete saved preflight coverage without loading a model.

Run in ltx: python -m scripts.prune.checks.joint_review --study STUDY --pilot PILOT
--p1 P1_DIRECTORY --p2 P2_DIRECTORY --output NEW_RECEIPT. Native P1 must contain
all 1152 ablations. Each case compares median early-block and late-block costs.
All comparisons must favor early blocks for an automatic P1 pass; mixed findings
stay unresolved. Pilot binary MSE must stay within 3% of native. P0 correlation
and P2 remain separate required gates. No input is regenerated here.
"""

import argparse
import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.learn.checkpoints import publish_json


def review(study, pilot, p1, p2):
    rows = [torch.load(p, map_location="cpu", weights_only=False) for p in Path(p1).glob("*.pt")]
    actors = study["splits"]["validation_actors"][:4]
    sigmas = [study["config"]["sigmas"][i] for i in study["config"]["monitor_sigma_indices"]]
    expected = {(a, s, b, n) for a in actors for s in sigmas for b in range(48) for n in (1638, 8192)}
    seen = {(r["actor"], r["sigma"], r["block"], r["count"]) for r in rows}
    if seen != expected or len(rows) != 1152:
        raise ValueError(
            f"P1 coverage incomplete or duplicated: expected 1152, got {len(rows)}; missing {len(expected - seen)}"
        )
    comparisons = []
    for a in actors:
        for s in sigmas:
            for n in (1638, 8192):
                subset = [r for r in rows if (r["actor"], r["sigma"], r["count"]) == (a, s, n)]
                for metric in ("capture_mse", "velocity_mse"):
                    early = torch.tensor([r["delta_per_channel"][metric] for r in subset if r["block"] < 16])
                    late = torch.tensor([r["delta_per_channel"][metric] for r in subset if 24 <= r["block"] <= 40])
                    comparisons.append(
                        dict(
                            actor=a,
                            sigma=s,
                            count=n,
                            metric=metric,
                            early_median=float(early.median()),
                            late_median=float(late.median()),
                            early_cheaper=bool(early.median() < late.median()),
                        )
                    )
    monitor = {}
    receipts = []
    for label, filename in (
        ("native", "update_0000.json"),
        ("fractional", "update_0050_fractional_before_reset.json"),
        ("binary", "update_0050.json"),
    ):
        path = Path(pilot) / "monitor" / filename
        payload = json.loads(path.read_text())
        if len(payload["cases"]) != 12:
            raise ValueError("pilot monitor coverage differs")
        monitor[label] = payload["capture_mse"]
        receipts.append(dict(path=str(path.resolve()), sha256=file_sha256(path)))
    increase = monitor["binary"] / monitor["native"] - 1
    p0_path = Path(pilot) / "checks/p0/replay.json"
    p0 = json.loads(p0_path.read_text())
    p2_path = Path(p2) / "acceptance.json"
    p2_receipt = json.loads(p2_path.read_text())
    receipts += [dict(path=str(p.resolve()), sha256=file_sha256(p)) for p in (p0_path, p2_path)]
    checks = dict(
        p0_correlated_replay=p0["correlation"]["status"] == "complete",
        p1_consistent_early_cost=all(c["early_cheaper"] for c in comparisons),
        pilot_binary_increase_at_most_three_percent=increase <= 0.03,
        p2_compact_parity=p2_receipt["passed"],
    )
    return dict(
        study=study["identity"],
        passed=all(checks.values()),
        checks=checks,
        p1_comparisons=comparisons,
        pilot_monitor=monitor,
        pilot_increase=increase,
        source_receipts=receipts,
        limitations="Mixed early/late findings require review. Random ablations do not establish learned-mask quality.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "pilot", "p1", "p2", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    publish_json(args.output, review(study, args.pilot, args.p1, args.p2))


if __name__ == "__main__":
    main()
