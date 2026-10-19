"""Index audited native-gate head comparisons and physical worker provenance.

This model-free reader materializes the comparison/ deliverables from immutable
review records and draw receipts. Original IDs, not input row order, own the join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.prune.score.native_gate_accumulator import atomic_json, digest

HEAD_SHAPE = (48, 32)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def disagreements(rows: list[dict], kept: list[list[bool]], groups: list) -> dict:
    expected = {(b, h) for b in range(HEAD_SHAPE[0]) for h in range(HEAD_SHAPE[1])}
    keyed = {(row["block"], row["head"]): row for row in rows}
    if len(rows) != len(keyed) or set(keyed) != expected:
        raise ValueError("comparison requires unique complete original head IDs")
    ordered = [keyed[key] for key in sorted(keyed)]
    for row in ordered:
        if row["learned_removed"] != (not bool(kept[row["block"]][row["head"]])):
            raise ValueError("original-head record differs from frozen binary mask")
    removed = sum(row["learned_removed"] for row in ordered)
    output = {}
    for field, role in (("native_global_removed", "global"), ("native_within_block_removed", "within_block")):
        by_group = []
        if any(len(row[field]) != len(groups) for row in ordered):
            raise ValueError("native-list score groups differ")
        for group, label in enumerate(groups):
            if sum(row[field][group] for row in ordered) != removed:
                raise ValueError("native list does not match the saved removal count")
            if role == "within_block":
                for block in range(HEAD_SHAPE[0]):
                    actual = sum(row[field][group] for row in ordered if row["block"] == block)
                    required = sum(row["learned_removed"] for row in ordered if row["block"] == block)
                    if actual != required:
                        raise ValueError("within-block native list differs from the saved block counts")
            categories = {name: [] for name in ("both", "learned_only", "native_only", "neither")}
            for row in ordered:
                learned, native = row["learned_removed"], row[field][group]
                category = (
                    "both"
                    if learned and native
                    else "learned_only"
                    if learned
                    else "native_only"
                    if native
                    else "neither"
                )
                categories[category].append([row["block"], row["head"]])
            by_group.append({"score_group": label, "removed_count": removed, **categories})
        output[role] = by_group
    return output


def export_records(root: Path, through_step: int) -> dict:  # noqa: PLR0912, PLR0915
    # Keep each prefix's identity checks and its publication in one transaction.
    study = read(root / "configs/frozen_study.json")
    identity = study["identity"]
    if digest({k: v for k, v in study.items() if k != "identity"}) != identity:
        raise ValueError("frozen comparison study identity differs")
    if through_step < 100 or through_step > study["steps"] or through_step % 100:
        raise ValueError("comparison export requires a complete milestone endpoint")
    mask_path = root / "comparison/masks.json"
    if file_hash(mask_path) != study["files"]["comparison/masks.json"]:
        raise ValueError("comparison mask bytes changed")
    references = {r["name"]: r["kept"] for r in read(mask_path)["references"]}
    workers = {}
    indexed = []
    for step in range(100, through_step + 1, 100):
        directory = root / f"milestones/step_{step:04d}"
        audit = read(root / f"monitoring/audits/step_{step:04d}.json")
        marker = read(directory / "review/complete.json")
        if (
            not audit["passed"]
            or audit["study"] != identity
            or marker["study"] != identity
            or audit["step"] != step
            or marker["step"] != step
        ):
            raise ValueError("comparison requires matching passing audits and review identities")
        source = directory / "review/head_records.json"
        relative = source.relative_to(root).as_posix()
        if file_hash(source) != marker["files"][relative]:
            raise ValueError("reviewed original-head records changed")
        records = read(source)
        review_path = directory / "review/review.json"
        if file_hash(review_path) != marker["files"][review_path.relative_to(root).as_posix()]:
            raise ValueError("reviewed historical correlations changed")
        reviewed = read(review_path)
        if records["score_group_order"] != [*study["sigmas"], "equal_sigma"]:
            raise ValueError("comparison score-group order differs from the frozen sigmas")
        if records["references"].keys() != references.keys():
            raise ValueError("comparison reference inventory differs")
        lists = {
            name: disagreements(rows, references[name], records["score_group_order"])
            for name, rows in records["references"].items()
        }
        for row in audit["window_worker_provenance"]:
            index = row["draw_index"]
            if index in workers:
                raise ValueError("duplicate draw in worker provenance")
            slot = index % len(study["gpus"])
            if row["gpu"] != study["gpus"][slot] or not (step - 100) * 4 <= index < step * 4:
                raise ValueError("worker provenance assignment or source window differs")
            receipt_path = root / f"draws/{index:05d}.json"
            workers[index] = {
                **row,
                "source_receipt": receipt_path.relative_to(root).as_posix(),
                "source_receipt_sha256": file_hash(receipt_path),
            }
        expected = [i for i in study["selected_draw_indices"] if i < step * 4]
        if sorted(workers) != expected:
            raise ValueError("worker provenance lacks the exact complete retained source prefix")
        receipts = [read(root / workers[i]["source_receipt"]) for i in expected]
        complete = read(directory / "complete.json")
        counts = [sum(r["sigma_index"] == j for r in receipts) for j in range(4)]
        if (
            complete["study"] != identity
            or complete["step"] != step
            or len(expected) != step * 8 // 5
            or counts != [step * 2 // 5] * 4
            or counts != complete["per_sigma"]
            or any(r["draw_index"] != i for i, r in zip(expected, receipts, strict=True))
        ):
            raise ValueError("comparison receipt population or sigma counts differ")
        if digest(receipts) != complete["receipt_chain_sha256"]:
            raise ValueError("worker provenance does not match the cumulative receipt chain")
        target = root / f"comparison/step_{step:04d}"
        atomic_json(target / "head_records.json", records)
        atomic_json(
            target / "event_correlations.json",
            {
                "study": identity,
                "step": step,
                "sigmas": study["sigmas"],
                "events": reviewed["event_correlations_by_sigma"],
                "source_sha256": file_hash(review_path),
            },
        )
        atomic_json(target / "disagreements.json", {"study": identity, "step": step, "references": lists})
        atomic_json(
            target / "draw_workers.json",
            {
                "study": identity,
                "step": step,
                "observed_draws": len(expected),
                "minimum_source_draw_id": min(expected),
                "maximum_source_draw_id": max(expected),
                "per_sigma": complete["per_sigma"],
                "receipt_chain_sha256": complete["receipt_chain_sha256"],
                "workers": [workers[i] for i in expected],
            },
        )
        indexed.append(step)
    result = {
        "study": identity,
        "through_step": through_step,
        "indexed_steps": indexed,
        "complete_original_ids": True,
        "complete_worker_provenance": True,
    }
    atomic_json(root / "comparison/index.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--through-step", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(export_records(args.root.resolve(), args.through_step), indent=2))


if __name__ == "__main__":
    main()
