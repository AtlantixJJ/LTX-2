"""Render ``train.py``'s per-rank JSONL logs as reviewable training-curve figures.

``train.py`` writes one record per ``(rank, step)`` to ``metrics_rank<r>.jsonl`` -- loss/mse,
lr, grad_norm, elapsed_s, and sample sources -- and does no aggregation and no
plotting of its own (see its ``main()``). This turns one or more run directories into the
curves a training review actually needs, plus a ``training_summary.json`` with the numbers a
plan write-up would quote, following the same figures + JSON + ``INDEX.md`` reporting convention.

**Why the mean-across-ranks loss is "the" loss curve.** Ranks are sharded over DIFFERENT
chains (``train.py``'s deterministic per-rank stride, not a ``DataLoader``), and FSDP
backward accumulates gradients from every rank before the one optimizer step -- so a step's
mean loss across ranks is the effective batch loss that step's update actually saw, the same
role averaging plays over a mini-batch's per-example losses. ``grad_norm`` and ``lr`` are
already rank-invariant by construction (``clip_grad_norm_`` all-reduces the norm; ``lr`` is
computed from ``step`` alone, identically on every rank) -- read once and cross-checked
against the other ranks rather than averaged, so a real desync shows up as a nonzero
disagreement in ``training_summary.json`` instead of being silently smoothed away.

Example:
    conda run -n ltx python -m scripts.onestep_avatar.plot_training \\
        --run ../expr/onestep_avatar/runs/prelim-r8 \\
        --run ../expr/onestep_avatar/runs/prelim-r8-attn-ffn --label attn+ffn
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from scripts.onestep_avatar.corpus.dataset import atomic_write

COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")


@dataclass
class RunData:
    label: str
    run_dir: Path
    config: dict
    by_rank: dict[int, list[dict]]  # rank -> records, sorted by step


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_run(run_dir: Path, label: str | None = None) -> RunData:
    rank_paths = sorted(run_dir.glob("metrics_rank*.jsonl"))
    if not rank_paths:
        raise SystemExit(f"{run_dir}: no metrics_rank*.jsonl -- is this a train.py --output directory?")
    by_rank: dict[int, list[dict]] = {}
    for path in rank_paths:
        records = sorted(_read_jsonl(path), key=lambda r: r["step"])
        if records:
            rank = int(path.stem.removeprefix("metrics_rank"))
            if any(r["rank"] != rank for r in records) or rank in by_rank:
                raise ValueError(f"{path}: mixed or duplicate rank logs")
            if len({r["step"] for r in records}) != len(records):
                raise ValueError(f"{path}: duplicate update records")
            by_rank[rank] = records
    if not by_rank:
        raise SystemExit(f"{run_dir}: metrics_rank*.jsonl are all empty")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return RunData(label=label or run_dir.name, run_dir=run_dir, config=config, by_rank=by_rank)


def update_coverage(run: RunData) -> dict[int, dict]:
    """Coverage is evidence separate from the mean; missing ranks are never zeros."""
    expected = run.config.get("world_size", len(run.by_rank))
    if type(expected) is not int or expected < 1 or any(rank not in range(expected) for rank in run.by_rank):
        raise ValueError("logged ranks disagree with the expected world size")
    groups = {}
    for rank, records in run.by_rank.items():
        for record in records:
            groups.setdefault(record["step"], {})[rank] = record
    coverage = {}
    for step, records in sorted(groups.items()):
        sample_counts = [len(row.get("samples", [])) for row in records.values() if row.get("schema_version") == 2]
        equal = not sample_counts or (
            len(sample_counts) == len(records) and min(sample_counts) > 0 and len(set(sample_counts)) == 1
        )
        coverage[step] = {
            "ranks": sorted(records),
            "expected": expected,
            "missing": sorted(set(range(expected)) - records.keys()),
            "complete": len(records) == expected and equal,
            "equal_accumulation": equal,
            "world_size_inferred": "world_size" not in run.config,
        }
    return coverage


def _complete_steps(run: RunData) -> set[int]:
    return {step for step, coverage in update_coverage(run).items() if coverage["complete"]}


def _per_step_values(run: RunData, field: str) -> dict[int, list[float]]:
    per_step: dict[int, list[float]] = {}
    complete = _complete_steps(run)
    for records in run.by_rank.values():
        for r in records:
            if r["step"] in complete and r.get(field) is not None:
                if not math.isfinite(r[field]):
                    raise ValueError(f"nonfinite {field} at update {r['step']}")
                per_step.setdefault(r["step"], []).append(r[field])
    expected = run.config.get("world_size", len(run.by_rank))
    per_step = {step: values for step, values in per_step.items() if len(values) == expected}
    return per_step


def step_mean(run: RunData, field: str) -> tuple[np.ndarray, np.ndarray]:
    """``(steps, mean-across-ranks value)``, in step order -- the batch-loss curve (see module doc)."""
    per_step = _per_step_values(run, field)
    steps = sorted(per_step)
    means = [sum(per_step[s]) / len(per_step[s]) for s in steps]
    return np.array(steps), np.array(means)


def scalar_series(run: RunData, field: str) -> tuple[np.ndarray, np.ndarray, float]:
    """``(steps, value, max cross-rank disagreement)`` for a field that should be rank-invariant."""
    per_step = _per_step_values(run, field)
    steps = sorted(per_step)
    values = np.array([per_step[s][0] for s in steps])
    disagreement = max((max(v) - min(v) for v in per_step.values()), default=0.0)
    return np.array(steps), values, float(disagreement)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; initial display padding repeats the first full-window mean."""
    if not len(values):
        return values
    window = max(1, min(window, len(values)))
    if window == 1:
        return values
    kernel = np.ones(window) / window
    # 'valid' shortens the series by window-1; pad the front by repeating the first smoothed
    # value so the smoothed line still spans every step (this is a display curve, not data).
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, smoothed[0]), smoothed])


def _per_position_values(run: RunData, field: str) -> dict[int, dict[int, list[float]]]:
    """``{chain_position: {step: [values across chains/ranks at that step]}}``.

    Version-two records use samples[].per_block. Older causal records use a
    top-level per_block list with explicit chain_position. Unmapped historical
    windows are not relabeled as causal blocks.
    """
    per_position: dict[int, dict[int, list[float]]] = {}
    if run.config.get("mode") == "bidirectional":
        return per_position
    complete = _complete_steps(run)
    for records in run.by_rank.values():
        for r in records:
            if r["step"] not in complete:
                continue
            if r.get("schema_version") == 2:
                for sample in r["samples"]:
                    for position, block in enumerate(sample.get("per_block", [])):
                        per_position.setdefault(position, {}).setdefault(r["step"], []).append(block[field])
                continue
            for w in r.get("per_block") or []:
                per_position.setdefault(w["chain_position"], {}).setdefault(r["step"], []).append(w[field])
    return per_position


def plot_block_position(runs: list[RunData], smooth: int, output: Path) -> Path | None:
    """SS7.4(c) ``block_position.png``: mean mse by position in the AR chain, over training.

    Named ``plot_window_position`` / ``window_position.png`` before S3 of the 2026-09-17
    cleanup plan -- a name left over from the sliding-window loop SS4.4 replaced with the
    causal AR one on 2026-09-14. The unit plotted here has been a **block**, not a window,
    since that rewrite; see ``doc/plot_training.md`` for the date this rename split at.

    Position means describe training error within sampled sequences. They do not establish
    correct motion, quality, or the cause of a higher loss at a later position.

    Returns ``None`` (plots nothing) if no run in ``runs`` has per-position data -- an older
    run predating SS7.4(a)'s logging change, not an error.
    """
    figure, axis = plt.subplots(figsize=(9, 4.5))
    linestyles = ("-", "--", ":", "-.")
    any_data = False
    for run, color in zip(runs, COLORS * (len(runs) // len(COLORS) + 1), strict=False):
        per_position = _per_position_values(run, "mse")
        for position in sorted(per_position):
            per_step = per_position[position]
            steps = sorted(per_step)
            means = np.array([sum(per_step[s]) / len(per_step[s]) for s in steps])
            if means.size == 0:
                continue
            any_data = True
            label = f"{run.label} pos {position}" if len(runs) > 1 else f"chain position {position}"
            axis.plot(steps, means, color=color, alpha=0.25, linewidth=0.8)
            axis.plot(
                steps,
                _smooth(means, smooth),
                color=color,
                linestyle=linestyles[position % len(linestyles)],
                linewidth=1.6,
                label=label,
            )
    if not any_data:
        plt.close(figure)
        return None
    axis.set(xlabel="completed updates", ylabel="full-frame MSE", title="Loss by sampled block position")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "block_position.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def _default_smooth_window(runs: list[RunData]) -> int:
    total_steps = max((max(r["step"] for records in run.by_rank.values() for r in records) for run in runs), default=1)
    return max(1, total_steps // 40)


def plot_loss_curves(runs: list[RunData], smooth: int, output: Path) -> Path:
    metrics = ("loss", "mse")
    figure, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.0 * len(metrics)), squeeze=False, sharex=True)
    single_run = len(runs) == 1
    for row, metric in enumerate(metrics):
        axis = axes[row, 0]
        for run, color in zip(runs, COLORS * (len(runs) // len(COLORS) + 1), strict=False):
            if single_run:
                # Enough headroom to also show each rank's own (noisier) chain-local loss --
                # useful for spotting one bad chain/rank rather than only the averaged curve.
                for rank in sorted(run.by_rank):
                    records = run.by_rank[rank]
                    rank_steps = np.array([r["step"] for r in records])
                    rank_values = np.array([r[metric] for r in records])
                    axis.plot(rank_steps, rank_values, color=color, alpha=0.18, linewidth=0.8, zorder=1)
                for step, coverage in update_coverage(run).items():
                    if not coverage["complete"]:
                        axis.axvline(step, color="gray", linestyle=":", linewidth=0.8)
                        axis.annotate(
                            "incomplete",
                            (step, 0.96),
                            xycoords=("data", "axes fraction"),
                            rotation=90,
                            va="top",
                            ha="right",
                            fontsize=8,
                        )
            steps, means = step_mean(run, metric)
            if steps.size == 0:
                continue
            axis.plot(steps, means, color=color, alpha=0.25, linewidth=1.0, zorder=2)
            axis.plot(steps, _smooth(means, smooth), color=color, linewidth=2.0, label=run.label, zorder=3)
        axis.set_ylabel(metric)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("completed updates")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    figure.suptitle("Full-frame training error")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = output / "loss_curves.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def plot_lr_grad_norm(runs: list[RunData], output: Path) -> Path:
    figure, (lr_axis, grad_axis) = plt.subplots(1, 2, figsize=(11, 3.6))
    for run, color in zip(runs, COLORS * (len(runs) // len(COLORS) + 1), strict=False):
        lr_steps, lr_values, _ = scalar_series(run, "lr")
        grad_steps, grad_values, _ = scalar_series(run, "grad_norm")
        if lr_steps.size:
            lr_axis.plot(lr_steps, lr_values, color=color, linewidth=1.6, label=run.label)
        if grad_steps.size:
            grad_axis.plot(grad_steps, grad_values, color=color, linewidth=1.2, alpha=0.8, label=run.label)
    lr_axis.set(xlabel="step", ylabel="learning rate", title="LR schedule")
    grad_axis.set(xlabel="completed updates", ylabel="gradient norm before clipping", title="Gradient norm")
    lr_axis.grid(alpha=0.25)
    grad_axis.grid(alpha=0.25)
    lr_axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "lr_grad_norm.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def plot_throughput(runs: list[RunData], output: Path) -> Path:
    figure, (step_axis, wall_axis) = plt.subplots(1, 2, figsize=(11, 3.6))
    for run, color in zip(runs, COLORS * (len(runs) // len(COLORS) + 1), strict=False):
        # Wall clock is the SLOWEST rank at each step (FSDP forward/backward are collective, so
        # every rank waits for it) -- max across ranks, not mean, is the number a run actually
        # took to reach that step.
        per_step = _per_step_values(run, "elapsed_s")
        steps = sorted(per_step)
        if not steps:
            continue
        wall = np.array([max(per_step[s]) for s in steps])
        step_time = np.diff(wall, prepend=0.0)
        step_axis.plot(steps, _smooth(step_time, max(1, len(steps) // 40)), color=color, linewidth=1.6, label=run.label)
        wall_axis.plot(steps, wall / 60.0, color=color, linewidth=1.6, label=run.label)
    step_axis.set(xlabel="step", ylabel="s / step (smoothed)", title="Step time")
    wall_axis.set(xlabel="step", ylabel="elapsed (min)", title="Wall clock")
    step_axis.grid(alpha=0.25)
    wall_axis.grid(alpha=0.25)
    step_axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "throughput.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def _run_summary(run: RunData) -> dict:
    steps_seen = sorted(_complete_steps(run))
    tail = steps_seen[-max(1, len(steps_seen) // 10) :]  # last ~10% of steps
    loss_steps, loss_means = step_mean(run, "loss")
    tail_mask = np.isin(loss_steps, tail)
    _, lr_values, lr_disagreement = scalar_series(run, "lr")
    _, grad_values, grad_disagreement = scalar_series(run, "grad_norm")
    per_step_wall = _per_step_values(run, "elapsed_s")
    wall = max((max(v) for v in per_step_wall.values()), default=0.0)
    sources = {
        sample["source"]
        for records in run.by_rank.values()
        for r in records
        if r["step"] in steps_seen
        for sample in r.get("samples", [{"source": r.get("source")}])
        if sample.get("source") is not None
    }
    sigma_counts = Counter(
        str(r["sigma0"])
        for records in run.by_rank.values()
        for r in records
        if r["step"] in steps_seen and "sigma0" in r
    )
    return {
        "run_dir": str(run.run_dir),
        "config": {
            key: run.config.get(key)
            for key in (
                "lora_rank",
                "lora_alpha",
                "lora_target",
                "mode",
                "objective",
                "sigma_levels",
                "guide_mode",
                "sigma0",
                "lr",
                "steps",
                "world_size",
            )
        },
        "steps_completed": steps_seen[-1] if steps_seen else 0,
        "ranks_logged": sorted(run.by_rank),
        "complete_updates": steps_seen,
        "coverage": update_coverage(run),
        "sigma_counts_per_rank_update": dict(sigma_counts),
        "loss_definition": "full_frame_x0_mse; equal sample means; causal samples average block means",
        "raw_series": {
            field: {"steps": step_mean(run, field)[0].tolist(), "values": step_mean(run, field)[1].tolist()}
            for field in ("loss", "mse")
        },
        "raw_block_position": _per_position_values(run, "mse"),
        "raw_timing": {
            "steps": sorted(per_step_wall),
            "elapsed_s": [max(per_step_wall[step]) for step in sorted(per_step_wall)],
            "interval_s": np.diff([max(per_step_wall[step]) for step in sorted(per_step_wall)], prepend=0).tolist(),
        },
        "final_lr": float(lr_values[-1]) if lr_values.size else None,
        "final_grad_norm": float(grad_values[-1]) if grad_values.size else None,
        "max_grad_norm_rank_disagreement": grad_disagreement,
        "max_lr_rank_disagreement": lr_disagreement,
        "loss_last_10pct_mean": float(loss_means[tail_mask].mean()) if tail_mask.any() else None,
        "loss_min": float(loss_means.min()) if loss_means.size else None,
        "loss_min_step": int(loss_steps[loss_means.argmin()]) if loss_means.size else None,
        "wall_clock_minutes": round(wall / 60.0, 2),
        "mean_observed_interval_time_s": round(wall / len(steps_seen), 2) if steps_seen else None,
        "distinct_sources_seen": len(sources),
    }


def render(runs: list[RunData], output: Path, smooth: int | None) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    window = smooth if smooth is not None else _default_smooth_window(runs)
    figures = [
        plot_loss_curves(runs, window, output),
        plot_lr_grad_norm(runs, output),
        plot_throughput(runs, output),
    ]
    lines = [
        "# onestep_avatar training figures",
        "",
        "- `loss_curves.png`: loss/mse vs completed updates -- faint per-step mean, bold smoothed; "
        "single-run plots also show each rank's own chain-local loss.",
        "- `lr_grad_norm.png`: learning rate and gradient norm before clipping vs completed updates.",
        "- `throughput.png`: smoothed seconds/step and cumulative wall clock vs step.",
    ]
    position_figure = plot_block_position(runs, window, output)
    if position_figure is not None:
        figures.append(position_figure)
        lines.append(
            "- `block_position.png`: mean mse by position in the AR chain, over training "
            "-- positions are relative to each sampled sequence, not original video block numbers."
        )
    else:
        lines.append("- `block_position.png`: not applicable; no complete causal position records.")
    summary = {
        "schema_version": 2,
        "runs": {run.label: _run_summary(run) for run in runs},
        "smooth_window": window,
        "smoothing": "equal-weight trailing window; initial display padding repeats first full-window mean",
    }
    atomic_write(
        output / "training_summary.json",
        lambda temporary: temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n"),
    )
    lines += [
        "- `training_summary.json`: the numbers above, plus the last-10%-of-run loss, the best "
        "step, and each run's config.",
        "",
        "## Runs",
        "",
    ] + [f"- **{run.label}**: `{run.run_dir}`" for run in runs]
    (output / "INDEX.md").write_text("\n".join(lines) + "\n")
    return figures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run",
        type=Path,
        action="append",
        dest="runs",
        required=True,
        help="train.py --output directory; repeatable to compare runs",
    )
    p.add_argument(
        "--label",
        action="append",
        default=[],
        help="one per --run, in the same order; default is the directory name",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="default: figures/ next to the first --run")
    p.add_argument(
        "--smooth",
        type=int,
        default=None,
        help="moving-average window in steps (default: total_steps // 40, min 1)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.label and len(args.label) != len(args.runs):
        raise SystemExit(f"--label given {len(args.label)} times but --run given {len(args.runs)} times")
    labels = args.label or [None] * len(args.runs)
    runs = [load_run(run_dir, label) for run_dir, label in zip(args.runs, labels, strict=True)]
    output = args.output_dir or args.runs[0] / "figures"
    for path in render(runs, output, args.smooth):
        print(path)  # noqa: T201 -- CLI's requested output list.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
