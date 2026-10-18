# ruff: noqa: E501
# Markdown paragraph literals keep complete sentences together in the source.
"""Saved-statistics-only milestone analysis and Markdown report/sketch generation.

This module imports no model, torch, decoder or execution owner. A report rebuild
cannot repair missing scientific evidence. It plots real milestone data and keeps
explicit placeholders for measurements that have not been observed yet.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.prune.score.native_gate_accumulator import (
    atomic_json,
    distribution_distance,
    load_arrays,
    means,
    probabilities,
    save_arrays,
)

QUESTIONS = [
    "How are native gates distributed across heads?",
    "Do the distributions settle as samples accumulate?",
    "Do different noise levels use different native gates?",
    "Do native gate statistics agree with our pruning decisions?",
]
DISPLAY = [3, 2, 1, 0]
COLORS = ["#3b6fb6", "#d17b28", "#44935b", "#9c59a7"]


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first, second = np.asarray(first).ravel(), np.asarray(second).ravel()
    if not np.ptp(first) or not np.ptp(second):
        return None
    return float(spearmanr(first, second).statistic)


def agreement(scores: np.ndarray, kept: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=float)
    kept = np.asarray(kept, dtype=bool)
    if scores.shape != kept.shape or scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("native score and original-head mask geometry differ")
    removed = ~kept
    labels = removed.ravel()
    values = scores.ravel()
    total, count = values.size, int(labels.sum())
    if count in (0, total):
        raise ValueError("agreement requires both kept and removed heads")
    ranked = rankdata(-values, method="average")
    auc = (ranked[labels].sum() - count * (count + 1) / 2) / (count * (total - count))
    order = np.argsort(values, kind="stable")
    ordered = labels[order]
    # Threshold AP groups score ties, unlike index-dependent individual positives.
    ends = np.r_[np.flatnonzero(np.diff(values[order]) != 0), total - 1]
    true = np.cumsum(ordered)[ends]
    recall = true / count
    average_precision = np.sum(np.diff(np.r_[0, recall]) * true / (ends + 1))
    selected = np.zeros(total, dtype=bool)
    selected[order[:count]] = True
    intersection = int((selected & labels).sum())
    block_selected = np.zeros_like(removed)
    by_block = removed.sum(axis=1)
    for block, k in enumerate(by_block):
        block_selected[block, np.argsort(scores[block], kind="stable")[:k]] = True
    within = int((block_selected & removed).sum())
    rng = np.random.default_rng(42)
    null = np.zeros(10000, dtype=int)
    for k in by_block:
        null += rng.hypergeometric(int(k), scores.shape[1] - int(k), int(k), size=len(null))
    return {
        "removed": count,
        "kept": total - count,
        "roc_auc": float(auc),
        "average_precision": float(average_precision),
        "prevalence": count / total,
        "intersection": intersection,
        "recall": intersection / count,
        "precision": intersection / count,
        "jaccard": intersection / (2 * count - intersection),
        "random_expected_intersection": count * count / total,
        "within_block_intersection": within,
        "within_block_recall": within / count,
        "within_block_null_mean": float(null.mean()),
        "within_block_null_std": float(null.std()),
        "within_block_null_interval": np.quantile(null, [0.025, 0.975]).tolist(),
        "within_block_reference_tail": float((1 + (null >= within).sum()) / (len(null) + 1)),
        "removed_mean": float(values[labels].mean()),
        "kept_mean": float(values[~labels].mean()),
        "removed_median": float(np.median(values[labels])),
        "kept_median": float(np.median(values[~labels])),
        "selected": np.flatnonzero(selected).tolist(),
        "within_block_selected": np.flatnonzero(block_selected).tolist(),
    }


def summaries(values: np.ndarray) -> dict:
    return {
        name: np.quantile(values.reshape(len(values), -1), q, axis=1).tolist()
        for name, q in (("median", 0.5), ("p95", 0.95), ("maximum", 1))
    }


def quantiles(histogram: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(histogram, axis=-1)
    count = cumulative[..., -1]
    return np.stack(
        [np.maximum(0, (cumulative < count[..., None] * q).sum(-1)) / 1024 for q in (0.01, 0.05, 0.5, 0.95, 0.99)],
        axis=-1,
    )


def analyze(root: Path, step: int) -> dict:
    directory = root / f"milestones/step_{step:04d}"
    pin = json.loads((directory / "complete.json").read_text())
    study = json.loads((root / "configs/frozen_study.json").read_text())
    groups = len(study["sigmas"])
    arrays, metadata = load_arrays(directory / "cumulative.npz")
    window, _ = load_arrays(directory / "window.npz")
    if (
        pin["draws"] != step * 160 // 100
        or len(metadata["receipts"]) != step * 160 // 100
        or not np.all(arrays["samples"] == step * 0.4)
    ):
        raise ValueError("analysis requires an exact, balanced completed milestone")
    scores = means(arrays)
    fraction = arrays["threshold_counts"] / arrays["tokens"][:, None, None, None]
    output = {
        "step": step,
        "draws": step * 160 // 100,
        "scheduled_draws": step * 4,
        "samples": arrays["samples"].tolist(),
        "mean_gate_by_sigma": scores.mean(axis=(1, 2)).tolist(),
        "coverage": [
            {
                "actors": len({r["actor"] for r in metadata["receipts"] if r["sigma_index"] == j}),
                "views": len({r["view"] for r in metadata["receipts"] if r["sigma_index"] == j}),
            }
            for j in range(groups)
        ],
    }
    stable = np.zeros_like(scores, dtype=np.int64)
    if step > 100:
        previous, _ = load_arrays(root / f"milestones/step_{step - 100:04d}/cumulative.npz")
        before = means(previous)
        delta = np.abs(scores - before)
        distance = distribution_distance(probabilities(arrays), probabilities(previous))
        recent = distribution_distance(probabilities(window), probabilities(previous))
        fraction_delta = np.abs(fraction - previous["threshold_counts"] / previous["tokens"][:, None, None, None])
        qualified = (delta <= 0.01) & (distance <= 0.02) & (fraction_delta.max(axis=2) <= 0.01)
        prior, _ = load_arrays(root / f"milestones/step_{step - 100:04d}/analysis.npz")
        stable = np.where(qualified, prior["stable_streak"] + 1, 0)
        output["convergence"] = {
            "mean_change": summaries(delta),
            "distribution_change": summaries(distance),
            "recent_window_distance": summaries(recent),
            "stable_fraction": (stable >= 3).mean((1, 2)).tolist(),
            "unstable_head_sigma_ids": np.argwhere(stable < 3).tolist(),
        }
        del previous, prior
    references = json.loads((root / "comparison/masks.json").read_text())["references"]
    output["agreement"] = {}
    for reference in references:
        kept = np.asarray(reference["kept"], dtype=bool)
        output["agreement"][reference["name"]] = {
            "by_sigma": [agreement(score, kept) for score in scores],
            "aggregate": agreement(scores.mean(0), kept),
        }
    output["sigma_rank_correlation"] = [[correlation(a, b) for b in scores] for a in scores]
    events = json.loads((root / "comparison/events.json").read_text())
    aggregate = scores.mean(0)
    output["events"] = [
        {
            "update": e["update"],
            "heads": len(e["heads"]),
            "spearman": correlation(
                np.array([aggregate[r["block"], r["head"]] for r in e["heads"]]),
                np.array([r["drop"] for r in e["heads"]]),
            ),
        }
        for e in events
    ]
    variances = np.maximum(0, arrays["gate_square_sum"] / arrays["tokens"][:, None, None] - scores**2)
    frame_mean = arrays["frame_gate_sum"] / (arrays["clean_tokens"][:, None, None, None])
    frame_high = arrays["frame_high_count"] / (arrays["clean_tokens"][:, None, None, None])
    summary = {
        "mean": scores,
        "std": np.sqrt(variances),
        "threshold_fractions": fraction,
        "quantiles": quantiles(arrays["histogram"]),
        "frame_mean": frame_mean,
        "frame_high_fraction": frame_high,
        "stable_streak": stable,
    }
    save_arrays(directory / "analysis.npz", summary, {"step": step, "study": pin["study"]})
    atomic_json(directory / "analysis.json", output)
    if step == 100:
        examples = {}
        for j in DISPLAY:
            order = np.argsort(scores[j].ravel(), kind="stable")
            examples[str(j)] = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
        rare = (scores.mean(0) < 0.5) & (summary["quantiles"][..., -1].max(0) > 1.5)
        eligible = np.flatnonzero(rare)
        eligible = sorted(eligible, key=lambda i: (aggregate.ravel()[i], i))[:3]
        atomic_json(
            root / "configs/examples.json",
            {
                "heads": examples,
                "brief_activation": eligible,
                "rule": "first-milestone mean <0.5 and any-sigma p99 >1.5",
            },
        )
    del arrays, window
    figures(root, step, summary, output, references, events)
    return output


def save_figure(fig: plt.Figure, directory: Path, name: str, data: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    atomic_json(directory / f"{name}.data.json", data)
    plt.close(fig)


def figures(root: Path, step: int, summary: dict, output: dict, references: list, events: list) -> None:  # noqa: PLR0912, PLR0915
    # Keep the fixed scientific panel ordering together; each saved figure has its own data receipt.
    directory = root / f"milestones/step_{step:04d}/figures"
    study = json.loads((root / "configs/frozen_study.json").read_text())
    sigmas = study["sigmas"]
    scores = summary["mean"]
    examples = json.loads((root / "configs/examples.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(15, 7), sharey=True, layout="constrained")
    for axis, j in zip(axes, DISPLAY, strict=True):
        plotted = axis.imshow(scores[j], vmin=0, vmax=2, aspect="auto", cmap="viridis")
        axis.set(title=f"sigma {sigmas[j]:.3f}", xlabel="Original head ID")
    axes[0].set_ylabel("Block ID")
    fig.colorbar(plotted, ax=list(axes), label="Cumulative mean native gate")
    fig.suptitle(f"Native gate means — {step} steps, {int(step * 0.4)} draws per noise level")
    save_figure(
        fig,
        directory,
        "E1a_mean_gates",
        {
            "x": "original head ID",
            "y": "block ID",
            "color": scores[DISPLAY].tolist(),
            "sigmas": [sigmas[j] for j in DISPLAY],
        },
    )
    arrays, _ = load_arrays(root / f"milestones/step_{step:04d}/cumulative.npz")
    probs = probabilities(arrays)
    centers = (np.arange(2048) + 0.5) / 1024
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    density_data = []
    for axis, j in zip(axes.ravel(), DISPLAY, strict=True):
        for index in examples["heads"][str(j)]:
            block, head = divmod(index, 32)
            # Coarsen adjacent bins only for readable display; source histogram is preserved.
            density = probs[j, block, head].reshape(128, 16).sum(-1) / (2 / 128)
            axis.plot(centers.reshape(128, 16).mean(-1), density, label=f"B{block} H{head}")
            density_data.append(
                {"sigma": sigmas[j], "block": block, "head": head, "probability": probs[j, block, head].tolist()}
            )
        axis.set(xlabel="Native gate value", ylabel="Probability density", title=f"sigma {sigmas[j]:.3f}", xlim=(0, 2))
        axis.legend(fontsize=8)
    save_figure(fig, directory, "E1b_distributions", {"examples": density_data, "histogram_bin_width": 1 / 1024})
    if examples["brief_activation"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for index in examples["brief_activation"]:
            block, head = divmod(index, 32)
            axes[0].plot(np.arange(18), summary["frame_mean"][:, block, :, head].mean(0), label=f"B{block} H{head}")
            axes[1].plot(
                np.arange(18), summary["frame_high_fraction"][:, block, :, head].mean(0), label=f"B{block} H{head}"
            )
        for axis, label in zip(axes, ["Mean native gate", "Fraction above 1.5"], strict=True):
            axis.set(xlabel="Local latent-frame index (0 is clean)", ylabel=label)
            axis.legend()
        save_figure(
            fig,
            directory,
            "E1c_brief_activation",
            {
                "heads": examples["brief_activation"],
                "frame_mean": summary["frame_mean"].tolist(),
                "high_fraction": summary["frame_high_fraction"].tolist(),
            },
        )
    histories = [
        json.loads((root / f"milestones/step_{s:04d}/analysis.json").read_text()) for s in range(100, step + 1, 100)
    ]
    if step >= 200:
        for field, name, ylabel in [
            ("mean_change", "E2a_mean_stability", "Absolute mean-gate change"),
            ("distribution_change", "E2b_distribution_stability", "Cumulative distribution distance"),
            ("recent_window_distance", "E2c_new_sample_check", "New-window vs prior cumulative distance"),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
            for axis, j in zip(axes.ravel(), DISPLAY, strict=True):
                for statistic, style in (("median", "-"), ("p95", "--"), ("maximum", ":")):
                    axis.plot(
                        [h["step"] for h in histories[1:]],
                        [h["convergence"][field][statistic][j] for h in histories[1:]],
                        style,
                        marker=".",
                        label=statistic,
                    )
                axis.set(xlabel="Completed logical steps", ylabel=ylabel, title=f"sigma {sigmas[j]:.3f}")
                axis.legend()
            save_figure(
                fig,
                directory,
                name,
                {
                    "history": [{"step": h["step"], "values": h["convergence"][field]} for h in histories[1:]],
                    "sigmas": sigmas,
                },
            )
        fig, axis = plt.subplots(figsize=(10, 4), layout="constrained")
        for j in range(len(sigmas)):
            axis.plot(
                [h["step"] for h in histories[1:]],
                [h["convergence"]["stable_fraction"][j] for h in histories[1:]],
                label=f"{sigmas[j]:.3f}",
            )
        axis.set(xlabel="Completed logical steps", ylabel="Fraction of practically stable heads", ylim=(-0.02, 1.02))
        axis.legend(title="sigma", ncol=5, fontsize=8)
        save_figure(
            fig,
            directory,
            "E2d_stable_coverage",
            {
                "history": [{"step": h["step"], "stable": h["convergence"]["stable_fraction"]} for h in histories[1:]],
                "sigmas": sigmas,
            },
        )
        fig, axes = plt.subplots(1, 4, figsize=(15, 7), sharey=True, layout="constrained")
        for axis, j in zip(axes, DISPLAY, strict=True):
            axis.imshow(summary["stable_streak"][j] < 3, vmin=0, vmax=1, aspect="auto", cmap="Greys")
            axis.set(title=f"sigma {sigmas[j]:.3f}", xlabel="Original head ID")
        axes[0].set_ylabel("Block ID")
        fig.suptitle("Black: not yet practically stable; no stability labels before step 400")
        save_figure(
            fig,
            directory,
            "E2d_unstable_heads",
            {"unstable": (summary["stable_streak"] < 3).tolist(), "sigmas": sigmas},
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fixed = sorted({i for values in examples["heads"].values() for i in values})
    order = np.argsort(sigmas)
    for index in fixed:
        block, head = divmod(index, 32)
        axes[0].plot(np.asarray(sigmas)[order], scores[:, block, head][order], label=f"B{block} H{head}")
        axes[1].plot(np.asarray(sigmas)[order], summary["threshold_fractions"][:, block, 2, head][order])
    axes[0].set(xlabel="Sigma", ylabel="Cumulative mean native gate")
    axes[1].set(xlabel="Sigma", ylabel="Fraction of native gates above 1.5")
    axes[0].legend(ncol=3, fontsize=7)
    save_figure(
        fig,
        directory,
        "E3a_noise_response",
        {
            "sigmas": sigmas,
            "heads": fixed,
            "means": scores.reshape(len(sigmas), -1)[:, fixed].tolist(),
            "upper_fraction": summary["threshold_fractions"][:, :, 2].reshape(len(sigmas), -1)[:, fixed].tolist(),
        },
    )
    fig, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    matrix = np.array(output["sigma_rank_correlation"], dtype=float)
    plotted = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set(
        xticks=range(len(sigmas)),
        yticks=range(len(sigmas)),
        xticklabels=[f"{s:.3f}" for s in sigmas],
        yticklabels=[f"{s:.3f}" for s in sigmas],
        xlabel="Sigma",
        ylabel="Sigma",
    )
    axis.tick_params(axis="x", rotation=90)
    fig.colorbar(plotted, ax=axis, label="Spearman correlation of head rankings")
    save_figure(
        fig, directory, "E3b_rank_agreement", {"sigmas": sigmas, "correlation": output["sigma_rank_correlation"]}
    )
    coverage_fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for j in DISPLAY:
        for axis, key in zip(axes, ("actors", "views"), strict=True):
            axis.plot(
                [h["step"] for h in histories], [h["coverage"][j][key] for h in histories], label=f"{sigmas[j]:.3f}"
            )
            axis.set(xlabel="Completed logical steps", ylabel=f"Distinct training {key}")
            axis.legend(title="sigma")
    save_figure(
        coverage_fig,
        directory,
        "coverage",
        {"history": [{"step": h["step"], "coverage": h["coverage"]} for h in histories]},
    )
    for reference in references:
        name = reference["name"]
        kept = np.asarray(reference["kept"], dtype=bool)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        cdf_data = []
        for axis, j in zip(axes.ravel(), DISPLAY, strict=True):
            for label, select in (("Kept", kept), ("Removed", ~kept)):
                x = np.sort(scores[j][select])
                axis.step(x, np.arange(1, len(x) + 1) / len(x), label=label)
                cdf_data.append({"sigma": sigmas[j], "group": label, "scores": x.tolist()})
            axis.set(
                xlabel="Per-head mean native gate",
                ylabel="Cumulative fraction of heads",
                title=f"sigma {sigmas[j]:.3f}",
                xlim=(0, 2),
                ylim=(0, 1),
            )
            axis.legend()
        save_figure(fig, directory, f"E4a_{name}", {"reference": name, "curves": cdf_data})
        fig, axes = plt.subplots(1, 2, figsize=(10, 7), layout="constrained")
        metrics = output["agreement"][name]["aggregate"]
        roles = []
        for axis, key, title in zip(
            axes,
            ("selected", "within_block_selected"),
            ("Global matched count", "Within-block matched counts"),
            strict=True,
        ):
            predicted = np.zeros(1536, dtype=bool)
            predicted[metrics[key]] = True
            category = (~kept).astype(int) + 2 * predicted.reshape(48, 32)
            axis.imshow(
                category,
                vmin=0,
                vmax=3,
                aspect="auto",
                cmap=matplotlib.colors.ListedColormap(["#eeeeee", "#d48045", "#568ec5", "#5eaa71"]),
            )
            axis.set(title=title, xlabel="Original head ID", ylabel="Block ID")
            roles.append(category.tolist())
        fig.suptitle("Grey: neither; orange: learned only; blue: native list only; green: both")
        save_figure(
            fig,
            directory,
            f"E4b_{name}",
            {"reference": name, "categories": roles, "equal_sigma_score": scores.mean(0).tolist()},
        )
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for axis, metric, label in zip(
            axes.ravel(),
            ("roc_auc", "average_precision", "recall", "jaccard"),
            ("ROC AUC", "Average precision", "Matched-count recall", "Matched-count Jaccard"),
            strict=True,
        ):
            for j, color in zip(DISPLAY, COLORS, strict=True):
                axis.plot(
                    [h["step"] for h in histories],
                    [h["agreement"][name]["by_sigma"][j][metric] for h in histories],
                    marker=".",
                    label=f"{sigmas[j]:.3f}",
                    color=color,
                )
            axis.plot(
                [h["step"] for h in histories],
                [h["agreement"][name]["aggregate"][metric] for h in histories],
                "k-",
                marker=".",
                label="equal-sigma score",
            )
            if metric in ("roc_auc", "average_precision", "recall"):
                chance = 0.5 if metric == "roc_auc" else metrics["prevalence"]
                axis.axhline(chance, color="grey", linestyle=":", label="chance/prevalence reference")
            axis.set(xlabel="Completed logical steps", ylabel=label, ylim=(0, 1))
            axis.legend(fontsize=7)
        save_figure(
            fig,
            directory,
            f"E4c_{name}",
            {"reference": name, "history": [{"step": h["step"], "agreement": h["agreement"][name]} for h in histories]},
        )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    scatter = []
    for axis, historical in zip(axes, [events[0], events[5], events[-1]], strict=True):
        x = [scores.mean(0)[r["block"], r["head"]] for r in historical["heads"]]
        y = [r["drop"] for r in historical["heads"]]
        colors = [
            {"kept": "#aaaaaa", "strong": "#d17b28", "persistent": "#3b6fb6"}[r["reason"]] for r in historical["heads"]
        ]
        axis.scatter(x, y, s=6, c=colors, alpha=0.65)
        axis.set(
            xlabel="Native mean gate (equal sigma weight)",
            ylabel="Historical pre-reset pruning drop",
            title=f"Pruning event {historical['update']}",
        )
        scatter.append(
            {
                "update": historical["update"],
                "native_mean": x,
                "drop": y,
                "reason": [r["reason"] for r in historical["heads"]],
            }
        )
    save_figure(fig, directory, "E4d_event_drops", {"events": scatter, "all_event_correlations": output["events"]})
    del arrays, probs


def build_report(root: Path) -> None:  # noqa: PLR0912, PLR0915
    # One document assembly keeps availability branches beside their scientific claims.
    """Create a sketch after launch; rebuild current sections from completed evidence."""
    study = json.loads((root / "configs/frozen_study.json").read_text())
    narrative = json.loads((root / "configs/narrative.json").read_text())
    milestones = sorted((root / "milestones").glob("step_*/analysis.json"))
    latest = json.loads(milestones[-1].read_text()) if milestones else None
    step = latest["step"] if latest else 0
    acceptance = [json.loads(p.read_text()) for p in sorted((root / "provenance").glob("acceptance_gpu_*.json"))]
    status_path = root / "monitoring/status.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {"status": "launched; native acceptance pending"}
    )
    lines = [
        "# Native head gates: training-stream statistics",
        "",
        "## Problem",
        "",
        "Measure native gate distributions, their stability with more training draws, and their agreement with saved pruning decisions.",
        "",
        "**Terms**",
        "",
        "- **Native gate:** the frozen dev model's token-dependent `2 * sigmoid(logit)` multiplier, in native units from zero to two.",
        "- **Step:** four scheduled training records; only selected-noise records receive F18 native forwards. Weights stay fixed.",
        "- **Milestone:** another 100 source steps: 400 scheduled records, 160 observed draws, and 40 per selected sigma.",
        "- **Sigma:** the draw's saved noise level; the four selected levels remain separate.",
        "- **Distribution distance:** binned Wasserstein-1 distance, the area between cumulative histograms in native gate units.",
        "- **Practical stability:** three consecutive mean/distribution/threshold-fraction changes within the saved tolerances; first eligible at step 400.",
        "- **Pruned reference:** a saved binary head mask, joined by original block/head IDs; agreement does not establish safe removal.",
        "",
        "**Shared setup:** fixed BF16 LTX-2.5 dev, empty text, 256 training views / 93 actors, original draw order filtered to four sigmas, F18 crops, one direct `[sigma,0]` call per draw, GPUs 4-6.",
        "",
        "**Progress**",
        "",
        f"Status: {status['status']}. Validated prefix: **{step} / {study['steps']} steps**, **{step * 160 // 100} observed draws** out of **{4 * step} scheduled records**.",
        f"Cumulative draws per noise level: **{int(step * 0.4)}**. Latest validated milestone: **{step if step else 'not available yet'}**.",
        "Noise levels: " + ", ".join(f"`{s}`" for s in study["sigmas"]) + ".",
        "",
        "**Main findings**",
        "",
    ]
    if latest:
        metrics = latest["agreement"]["clipped_reset_u600"]["aggregate"]
        lines.extend(
            [
                f"1. The current native mean gate across heads and noise levels is {np.mean(latest['mean_gate_by_sigma']):.4f}. [Distributions](#how-are-native-gates-distributed-across-heads)",
                f"2. Adjacent-prefix convergence is {'available below; practical stability requires step 400 or later' if step >= 200 else 'not available yet; step 200 is required'}. [Convergence](#do-the-distributions-settle-as-samples-accumulate)",
                "3. Noise-specific rankings and coverage are shown separately. [Noise](#do-different-noise-levels-use-different-native-gates)",
                f"4. At {step} steps, the equal-sigma native ranking overlaps {metrics['intersection']} of 292 learned removals; ROC AUC is {metrics['roc_auc']:.3f}. This is descriptive agreement. [Pruning](#do-native-gate-statistics-agree-with-our-pruning-decisions)",
            ]
        )
    else:
        lines.extend(f"{i + 1}. **Not available yet.** {question}" for i, question in enumerate(QUESTIONS))
    lines.extend(
        [
            "",
            "## Does observation preserve the native forward?",
            "",
            f"**Short answer:** {'Passed on ' + str(len(acceptance)) + ' workers, with bit-identical outputs and unchanged weight hashes.' if acceptance else 'Not available yet; worker native acceptance is pending.'}",
            "",
        ]
    )
    for row in acceptance:
        gpu = row["gpu"]
        lines.append(
            f"GPU {gpu}: maximum output difference `{row['max_abs']}`, {row['modules']} blocks / {row['heads']} heads; [receipt](provenance/acceptance_gpu_{gpu}.json)."
        )
    for number, question in enumerate(QUESTIONS, 1):
        lines.extend(["", f"## {question}", ""])
        if latest and number == 1:
            lines.append(
                f"**Short answer:** Native distributions are available through {step} steps, with {int(step * 0.4)} draws per noise level. Shapes and per-head differences appear below."
            )
        elif latest and number == 2 and step >= 200:
            conv = latest["convergence"]
            lines.append(
                f"**Short answer:** Median cumulative distribution changes range from {min(conv['distribution_change']['median']):.5f} to {max(conv['distribution_change']['median']):.5f} across noise levels. Practically stable head fractions range from {min(conv['stable_fraction']):.1%} to {max(conv['stable_fraction']):.1%}; this does not prove a limit distribution."
            )
        elif latest and number == 3:
            matrix = np.array(latest["sigma_rank_correlation"], dtype=float)
            lower = matrix[np.tril_indices(len(study["sigmas"]), -1)]
            lines.append(
                f"**Short answer:** Cross-noise head-rank correlations range from {np.nanmin(lower):.3f} to {np.nanmax(lower):.3f}. Assigned samples differ across noise levels; this is not a same-input intervention."
            )
        elif latest and number == 4:
            metrics = latest["agreement"]["clipped_reset_u600"]["aggregate"]
            lines.append(
                f"**Short answer:** The equal-sigma ranking has ROC AUC {metrics['roc_auc']:.3f}, average precision {metrics['average_precision']:.3f}, and matched-count recall {metrics['recall']:.1%}. Its overlap is {metrics['intersection']} heads versus {metrics['random_expected_intersection']:.1f} under global random selection; within-block overlap is {metrics['within_block_intersection']} versus a conditional-reference mean of {metrics['within_block_null_mean']:.1f}."
            )
        else:
            required = 200 if number == 2 else 100
            lines.append(
                f"**Short answer:** Not available yet. This section requires the validated step-{required} milestone."
            )
        lines.extend(["", narrative["sections"][number]["design"], ""])
        diagram = root / f"figures/path_E{number}.png"
        if diagram.exists():
            lines.extend([f"![Measurement path for E{number}.](figures/path_E{number}.png)", ""])
        available = []
        if latest:
            available = sorted((root / f"milestones/step_{step:04d}/figures").glob(f"E{number}*.png"))
        if available:
            for path in available:
                relative = str(path.relative_to(root))
                title = path.stem.replace("_", " ")
                lines.extend(
                    [
                        f"![{title}; real evidence through step {step}.]({relative})",
                        "",
                        f"*{title}, through {step} steps. Figure axes and its purpose are specified above.* "
                        f"[PDF]({relative[:-4]}.pdf) · [Plotted data]({relative[:-4]}.data.json)",
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "**Figure placeholders:** the figures, axes and expected patterns listed above are planned; no synthetic result curves are shown.",
                    "",
                ]
            )
        if number == 1 and latest:
            examples = json.loads((root / "configs/examples.json").read_text())
            if not examples["brief_activation"]:
                lines.append(
                    "**Brief-activation coverage:** no first-milestone head meets the saved mean <0.5 and any-sigma p99 >1.5 example rule. E1c is omitted; the full statistics remain saved."
                )
        if number == 2 and latest and step < 400:
            lines.append(
                "**Practical-stability labels are not available yet:** three adjacent-milestone comparisons require step 400."
            )
    lines.extend(
        [
            "",
            "## Evidence and limitations",
            "",
            "These are training-data statistics from one frozen draw order and checkpoint. Repeated views, tokens and noise levels are correlated. The study estimates neither blind-test quality nor a new mask's removal safety.",
            "",
            "Cumulative curves can flatten because each new window has decreasing weight. Read the disjoint-window distances and coverage beside them. Model weights never change.",
            "",
            "[Frozen plan](provenance/plan.md) · [Frozen study](configs/frozen_study.json) · [Pruning references](comparison/masks.json)",
            "",
        ]
    )
    text = "\n".join(lines)
    # Validate links and scientific figure companions after every rebuild.

    missing = []
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if target.startswith("#"):
            continue
        if not (root / target.split("#")[0]).exists():
            missing.append(target)
    if missing:
        raise ValueError(f"report links missing saved evidence: {missing}")
    for path in (root / f"milestones/step_{step:04d}/figures").glob("*.png"):
        for suffix in (".pdf", ".data.json"):
            if not path.with_suffix(suffix).is_file():
                raise ValueError(f"scientific figure companion missing: {path}")
    temporary = root / f".REPORT.{os.getpid()}.md.tmp"
    temporary.write_text(text)
    temporary.replace(root / "REPORT.md")
    atomic_json(
        root / "monitoring/report_validation.json",
        {"passed": True, "step": step, "local_links_checked": True, "synthetic_result_curves": False},
    )
