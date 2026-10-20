"""Follow saved learned FFN or V3 joint receipts on CPU and publish one W&B run.

Use --study to select the frozen config; its `format` selects the mask metrics.
Read immutable optimizer and monitor receipts, log each global step once, and
retain a mutable local cursor. Rank sigma and memory use labeled series. Loss
and validation use global means. Stop after the training owner finishes or fails.
This process never loads a model or creates a CUDA context.
"""

import argparse
import json
import os
import time
from pathlib import Path


def mask_metrics(row, study_format):
    """Training scalars plus the format's mask counts for one global update receipt."""
    data = {f"train/{k}": v for k, v in row["metrics"].items()}
    data["train/update"] = row["update"]
    if study_format == "ffn_mask_study_v1":
        data["ffn/removed_channels"] = row["ffn"]["removed_channels"]
        data["ffn/retained_width"] = row["ffn"]["retained_per_block"][0]
        return data
    if study_format != "joint_study_v3":
        raise ValueError(f"unsupported receipt study format: {study_format}")
    data["channels/removed"] = row["ffn"]["counts"]["removed"]
    data["channels/kept"] = row["ffn"]["counts"]["kept"]
    for family, snapshot in (("heads", row["heads"]), ("channels", row["ffn"])):
        for key, value in snapshot["counts"].items():
            if isinstance(value, (int, float)):
                data[f"{family}/{key}"] = value
        for block, count in enumerate(snapshot["retained_per_block"]):
            data[f"{family}/kept_block_{block}"] = count
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    study = json.loads(args.study.read_text())
    cfg, root = study["config"], Path(study["root"])
    settings = cfg.get("wandb", {})
    if not settings.get("enabled"):
        return
    import wandb

    run = wandb.init(
        project=settings["project"],
        entity=settings.get("entity"),
        id="ffn_" + study["identity"][:20],
        resume="allow",
        mode=settings.get("mode", "online"),
        name=root.name,
        config=cfg,
        dir=str(root),
    )
    cursor_path = root / "monitoring/wandb_cursor.json"
    cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else dict(update=0, monitor=-1)
    print(run.url, flush=True)
    while True:
        files = sorted((root / "training/updates").glob("*.json"))
        for path in files:
            row = json.loads(path.read_text())
            if row["update"] <= cursor["update"]:
                continue
            data = mask_metrics(row, study["format"])
            for rank, item in enumerate(row["draws"]):
                data[f"ranks/sigma_rank_{rank}"] = item["sigma"]
            for item in row["memory"]:
                data[f"memory/allocated_rank_{item['rank']}"] = item["allocated"]
            run.log(data)
            cursor["update"] = row["update"]
        for path in sorted((root / "monitor").glob("update_*.json")):
            row = json.loads(path.read_text())
            if row["update"] <= cursor["monitor"]:
                continue
            run.log({"validation/update": row["update"], "validation/capture_mse": row["capture_mse"]})
            cursor["monitor"] = row["update"]
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cursor_path.with_suffix(".tmp")
        temp.write_text(json.dumps(cursor) + "\n")
        temp.replace(cursor_path)
        owner = root / "monitoring/owner.json"
        if owner.exists() and json.loads(owner.read_text())["status"] in ("complete", "stopped"):
            break
        if (root / "monitoring/failure.json").exists():
            break
        time.sleep(15)
    run.finish()


if __name__ == "__main__":
    main()
