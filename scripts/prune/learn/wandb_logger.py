"""One CPU W&B writer per arm, reading published global receipts only."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

from scripts.prune.learn import protocol


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def published_updates(folder):
    return sorted((p for p in folder.glob("*.json") if p.stem.isdecimal()), key=lambda p: int(p.stem))


def training_metrics(row, config):
    draws = row["draws"]
    if len(draws) != config.get("global_unique_clips", 2):
        raise ValueError("incomplete global draw coverage for W&B")
    official = sum(d["official"] for d in draws) / len(draws)
    penalty = sum(d["penalty"] for d in draws) / len(draws)
    total = sum(d.get("total_loss", d["official"] + d["penalty"]) for d in draws) / len(draws)
    phase, step = row["phase"], row["update"]
    # Older frozen configs predate explicit budgets in this reader; they used 700/300/500.
    budget = {"soft_updates": 700, "binary_updates": 300, "compact_updates": 500, **config}
    soft_end, selection_end, _ = protocol.phase_budget(budget)
    phase_start = selection_end if phase == "compact" else soft_end if phase == "binary" else 0
    rank_rows = row.get("rank_metrics", [])
    values = {
        "update": step,
        "train/loss": total,
        "train/velocity_mse": official,
        "train/sparsity_loss": penalty,
        "train/lambda": row["sparsity_lambda"],
        "train/phase_code": {"soft": 0, "binary": 1, "compact": 2, "fixed": 3}[phase],
        "train/update_s": max((r["seconds"] for r in rank_rows), default=row["seconds"]),
        "train/lr_adapters": config["adapter_lr"] * min((step - phase_start) / config["warmup_updates"], 1),
        "train/lr_gates": config["gate_lr"] if phase in ("soft", "binary") else 0,
        "train/legal_mask_changes": row["legal_mask_changes"],
    }
    for name in (
        "adapter_gradient_norm",
        "logit_gradient_norm",
        "velocity_logit_gradient_norm",
        "sparsity_logit_gradient_norm",
    ):
        values["train/" + name] = row[name]
    for name in ("mean", "le_0_1", "lt_0_5", "ge_0_9"):
        if row["gates"][name] is not None:
            values["gates/" + name] = row["gates"][name]
    if "progressive" in row:
        for name, count in row["progressive"]["counts"].items():
            values["gates/" + name] = count
        for name in (
            "eligible_count",
            "requested_count",
            "selected_count",
            "shortfall",
            "one_eligible_count",
            "one_requested_count",
            "one_selected_count",
            "one_shortfall",
        ):
            if name in row["progressive"]:
                values["gates/" + name] = row["progressive"][name]
    for name, value in zip(
        ("min", "p10", "median", "p90", "max"), row["gates"]["quantiles"] or [None] * 5, strict=True
    ):
        if value is not None:
            values["gates/" + name] = value
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("nonfinite W&B training metrics")
    return values


def validation_metrics(monitor):
    # Count each actor/view/sigma once, even if an older monitor repeated it on ranks.
    unique = {}
    for row in monitor["rows"]:
        key = row["view"], row["sigma"]
        if key in unique and row["states"] != unique[key]["states"]:
            raise ValueError("duplicate validation case has conflicting results")
        unique[key] = row
    rows = list(unique.values())
    if not rows:
        raise ValueError("empty validation monitor")
    values = {"validation/update": monitor["update"], "validation/cases": len(rows)}
    for state in ("trained",) if "progressive" in monitor else ("trained", "hardened"):
        values[f"validation/{state}_gt_latent_mse"] = sum(
            r["states"][state]["metrics"]["capture_mse"] for r in rows
        ) / len(rows)
        for index, sigma in enumerate(sorted({r["sigma"] for r in rows}, reverse=True)):
            cases = [r for r in rows if r["sigma"] == sigma]
            values[f"validation/{state}_sigma_{index}_gt_latent_mse"] = sum(
                r["states"][state]["metrics"]["capture_mse"] for r in cases
            ) / len(cases)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("nonfinite W&B validation metrics")
    return values


def rank_series(history, world):
    specs = {
        "train/sigma": "Sigma per rank",
        "ranks/update_s": "Update seconds per rank",
        "memory/free_GiB": "Free memory per rank",
        "memory/peak_allocated_GiB": "Peak allocated memory per rank",
        "memory/peak_reserved_GiB": "Peak reserved memory per rank",
    }
    series = {
        key: {
            "title": title,
            "xs": [],
            "ys": [[] for _ in range(world)],
            "keys": [f"rank {rank}" for rank in range(world)],
        }
        for key, title in specs.items()
    }
    for row in history:
        ranks = sorted(row["rank_metrics"], key=lambda r: r["rank"])
        if [r["rank"] for r in ranks] != list(range(world)) or len(row["devices"]) != world:
            raise ValueError("incomplete per-rank chart evidence")
        for spec in series.values():
            spec["xs"].append(row["update"])
        for rank, (metrics, memory) in enumerate(zip(ranks, row["devices"], strict=True)):
            series["train/sigma"]["ys"][rank].append(metrics["sigma"])
            series["ranks/update_s"]["ys"][rank].append(metrics["seconds"])
            for name in ("free", "peak_allocated", "peak_reserved"):
                series[f"memory/{name}_GiB"]["ys"][rank].append(memory[name] / 2**30)
    if any(not math.isfinite(value) for spec in series.values() for ys in spec["ys"] for value in ys):
        raise ValueError("nonfinite per-rank chart evidence")
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--arm", default="learned_joint")
    parser.add_argument("--replicate-seed", type=int, default=42)
    args = parser.parse_args()
    study = read(args.study)
    body = {k: v for k, v in study.items() if k != "identity"}
    identity = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if identity != study["identity"] or args.replicate_seed not in study["config"]["replicate_seeds"]:
        raise ValueError("W&B study identity or seed differs")
    root = Path(study["root"])
    cfg = study["config"]
    options = cfg.get("wandb", {})
    stem = (
        "wandb"
        if args.arm == "learned_joint" and args.replicate_seed == 42
        else f"wandb_{args.arm}_{args.replicate_seed}"
    )
    directory = root / "monitoring"
    directory.mkdir(exist_ok=True)
    lock = (directory / f"{stem}_logger.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cursor_path = directory / f"{stem}_cursor.json"
    cursor = read(cursor_path) if cursor_path.exists() else {"train_update": 0, "validation": []}
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    import wandb

    run_id = "heads-" + identity[:16] + ("" if args.arm == "learned_joint" else "-" + args.arm)
    if args.replicate_seed != 42:
        run_id += "-" + str(args.replicate_seed)
    run = wandb.init(
        project=options.get("project", "onestep-avatar"),
        entity=options.get("entity", "jianjinx"),
        name=root.name if args.arm == "learned_joint" else root.name + "-" + args.arm,
        group="learned-head-pruning",
        id=run_id,
        resume="allow",
        mode=options.get("mode", "online"),
        dir=str(directory),
        config={**cfg, "arm": args.arm, "study_identity": identity, "replicate_seed": args.replicate_seed},
        settings=wandb.Settings(x_disable_stats=True),
    )
    run.define_metric("update")
    run.define_metric("validation/update")
    for prefix in ("train", "gates", "memory", "ranks"):
        run.define_metric(prefix + "/*", step_metric="update")
    run.define_metric("validation/*", step_metric="validation/update")
    metadata = {
        "status": "logging",
        "pid": os.getpid(),
        "run_id": run_id,
        "url": run.url,
        "arm": args.arm,
        "source": "published global receipts; no model or decoder",
    }
    save(directory / f"{stem}.json", metadata)
    (directory / f"{stem}_logger.pid").write_text(str(os.getpid()) + "\n")
    updates = root / f"training/{args.arm}/replicate_{args.replicate_seed}/updates"
    world = len(cfg["gpu_ids"])
    interval = options.get("chart_interval", 10)
    if type(interval) is not int or interval < 1:
        raise ValueError("W&B chart interval must be positive")
    history = [read(p) for p in published_updates(updates) if int(p.stem) <= cursor["train_update"]]
    last_chart = 0

    def log_charts():
        nonlocal last_chart
        if history and "rank_metrics" in history[-1]:
            charts = {
                key: wandb.plot.line_series(**spec, xname="optimizer update")
                for key, spec in rank_series(history, world).items()
            }
            run.log({"update": history[-1]["update"], **charts})
            last_chart = history[-1]["update"]

    try:
        while not stop[0]:
            for path in published_updates(updates):
                row = read(path)
                if row["update"] <= cursor["train_update"]:
                    continue
                if row["study"] != identity:
                    raise ValueError("W&B receipt belongs to another study")
                run.log(training_metrics(row, cfg))
                history.append(row)
                cursor["train_update"] = row["update"]
                save(cursor_path, cursor)
                if row["update"] in (1, 2) or row["update"] % interval == 0:
                    log_charts()
            for path in sorted((root / f"checks/{args.arm}/replicate_{args.replicate_seed}").glob("monitor_*.json")):
                if str(path) in cursor["validation"]:
                    continue
                monitor = read(path)
                if monitor["study"] != identity:
                    raise ValueError("W&B monitor belongs to another study")
                run.log(validation_metrics(monitor))
                cursor["validation"].append(str(path))
                save(cursor_path, cursor)
            metadata.update(
                last_logged_update=cursor["train_update"],
                last_poll_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            status_path = directory / "status.json"
            status = read(status_path)["status"] if status_path.exists() else "starting"
            if status in (
                "failed",
                "stopped",
                "paired_training_complete_GT_validation_pending",
                "progressive_training_complete",
            ):
                metadata["status"] = status
                break
            save(directory / f"{stem}.json", metadata)
            time.sleep(5)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if stop[0]:
            metadata["status"] = "stopped"
        try:
            if last_chart != cursor["train_update"]:
                log_charts()
        finally:
            run.finish()
            save(directory / f"{stem}.json", metadata)
            lock.close()


if __name__ == "__main__":
    main()
