"""Complete per-sigma report evidence from saved native-gate milestones only.

Collection and its frozen analysis snapshot remain unchanged. Supplemental figures
live in each milestone's review/ directory. Narrative updates are atomic, so the
running report owner can include the new evidence without changing model code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

from scripts.prune.score.native_gate_accumulator import atomic_json, load_arrays
from scripts.prune.score.native_gate_analysis import DISPLAY, correlation, plt, save_figure


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def event_correlations(scores: np.ndarray, events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        ids = np.array([[r["block"], r["head"]] for r in event["heads"]], dtype=int)
        drops = np.array([r["drop"] for r in event["heads"]])
        x = scores[:, ids[:, 0], ids[:, 1]]
        rows.append(
            {
                "update": event["update"],
                "eligible_heads": len(ids),
                "by_sigma": [correlation(values, drops) for values in x],
                "equal_sigma": correlation(x.mean(0), drops),
            }
        )
    return rows


def head_records(scores: np.ndarray, reference: dict, metrics: dict) -> list[dict]:
    kept = np.asarray(reference["kept"], dtype=bool)
    groups = [*metrics["by_sigma"], metrics["aggregate"]]
    global_sets = [set(g["selected"]) for g in groups]
    block_sets = [set(g["within_block_selected"]) for g in groups]
    return [
        {
            "block": b,
            "head": h,
            "learned_removed": not bool(kept[b, h]),
            "mean_by_sigma": scores[:, b, h].tolist(),
            "equal_sigma_mean": float(scores[:, b, h].mean()),
            "native_global_removed": [i in selected for selected in global_sets],
            "native_within_block_removed": [i in selected for selected in block_sets],
        }
        for i, (b, h) in enumerate(np.ndindex(kept.shape))
    ]


def density_overlays(root: Path, step: int, target: Path) -> None:
    prefixes = sorted({100, max(100, step - 100), step})
    examples = read(root / "configs/examples.json")["heads"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    data = []
    colors = ["#3b6fb6", "#d17b28", "#44935b"]
    for prefix in prefixes:
        curves = read(root / f"milestones/step_{prefix:04d}/figures/E1b_distributions.data.json")["examples"]
        for axis, group in zip(axes.ravel(), DISPLAY, strict=True):
            ids = examples[str(group)]
            rows = [r for r in curves if r["sigma"] == read(root / "configs/frozen_study.json")["sigmas"][group]]
            if [r["block"] * 32 + r["head"] for r in rows] != ids:
                raise ValueError("distribution overlay changed frozen first-milestone head IDs")
            for color, row in zip(colors, rows, strict=True):
                density = np.array(row["probability"]).reshape(128, 16).sum(-1) * 64
                style = "-" if prefix == step else "--" if prefix == 100 else ":"
                axis.plot(
                    (np.arange(128) + 0.5) / 64,
                    density,
                    style,
                    color=color,
                    alpha=1 if prefix == step else 0.55,
                    label=f"B{row['block']} H{row['head']}, step {prefix}",
                )
                data.append({"step": prefix, **row})
            axis.set(
                title=f"sigma {rows[0]['sigma']:.3f}", xlabel="Native gate", ylabel="Probability density", xlim=(0, 2)
            )
            axis.legend(fontsize=6)
    save_figure(
        fig,
        target,
        "E1b_prefix_overlays",
        {"prefixes": prefixes, "curves": data, "source_bin_width": 1 / 1024, "display_bins": 128},
    )


def identity_figures(target: Path, name: str, kept: np.ndarray, metrics: dict, sigmas: list) -> None:
    cmap = matplotlib.colors.ListedColormap(["#eeeeee", "#d48045", "#568ec5", "#5eaa71"])
    for key, role in (("selected", "global"), ("within_block_selected", "within_block")):
        fig, axes = plt.subplots(1, 4, figsize=(15, 7), sharey=True, layout="constrained")
        values = []
        for axis, group in zip(axes, DISPLAY, strict=True):
            selected = np.zeros(kept.size, dtype=bool)
            selected[metrics["by_sigma"][group][key]] = True
            category = (~kept).astype(int) + 2 * selected.reshape(kept.shape)
            axis.imshow(category, vmin=0, vmax=3, cmap=cmap, aspect="auto")
            axis.set(title=f"sigma {sigmas[group]:.3f}", xlabel="Original head ID")
            values.append(category.tolist())
        axes[0].set_ylabel("Block ID")
        fig.suptitle(f"{name}: {role} count matching\nGrey neither; orange learned only; blue native only; green both")
        save_figure(
            fig,
            target,
            f"E4b_{name}_{role}_by_sigma",
            {"sigmas": [sigmas[j] for j in DISPLAY], "reference": name, "categories": values, "method": key},
        )


def drop_figures(target: Path, scores: np.ndarray, events: list, sigmas: list, correlations: list) -> None:
    selected = sorted({0, len(events) // 2 - 1, len(events) - 1})
    for index in selected:
        event = events[index]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        b, h = np.array([[r["block"], r["head"]] for r in event["heads"]]).T
        drops = [r["drop"] for r in event["heads"]]
        colors = [
            {"kept": "#aaaaaa", "strong": "#d17b28", "persistent": "#3b6fb6"}[r["reason"]] for r in event["heads"]
        ]
        for axis, group in zip(axes.ravel(), DISPLAY, strict=True):
            axis.scatter(scores[group, b, h], drops, s=6, c=colors, alpha=0.65)
            rho = correlations[index]["by_sigma"][group]
            label = "undefined" if rho is None else f"{rho:.3f}"
            axis.set(
                title=f"sigma {sigmas[group]:.3f}, Spearman {label}",
                xlabel="Native mean gate",
                ylabel="Historical pre-reset drop",
                xlim=(0, 2),
            )
        fig.suptitle(
            f"Historical event {event['update']}: eligible heads only; grey kept, orange strong, blue persistent"
        )
        save_figure(
            fig,
            target,
            f"E4d_event_{event['update']:04d}_by_sigma",
            {
                "sigmas": sigmas,
                "event": event,
                "native_mean_by_sigma": scores[:, b, h].tolist(),
                "correlation": correlations[index],
            },
        )


def figure_markdown(root: Path, path: Path, caption: str) -> str:
    relative = path.relative_to(root).as_posix()
    return (
        f"\n\n![{caption}]({relative}.png)\n\n*{caption}* "
        f"[Full-size PNG]({relative}.png) · [PDF]({relative}.pdf) · [Plotted data]({relative}.data.json)\n"
    )


def sampling_coverage(root: Path, step: int, current: dict, target: Path) -> None:
    history = [read(root / f"milestones/step_{s:04d}/review/review.json") for s in range(100, step, 100)]
    history.append(current)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, key in zip(axes, ("actors", "views"), strict=True):
        for group in DISPLAY:
            (line,) = axis.plot(
                [r["step"] for r in history],
                [r["coverage"][group][key] for r in history],
                label=f"sigma {current['sigmas'][group]:.3f}, cumulative",
            )
            axis.plot(
                [r["step"] for r in history],
                [r["window_coverage"][group][key] for r in history],
                "--",
                color=line.get_color(),
                label=f"sigma {current['sigmas'][group]:.3f}, window",
            )
        axis.set(xlabel="Completed source steps", ylabel=f"Distinct training {key}")
        axis.legend(fontsize=6)
    save_figure(
        fig,
        target,
        "E2c_sampling_coverage",
        {
            "history": [{k: r[k] for k in ("step", "coverage", "window_coverage")} for r in history],
            "sigmas": current["sigmas"],
        },
    )


def write_narrative(root: Path, step: int, result: dict) -> None:
    path = root / "configs/narrative.json"
    baseline = root / "configs/narrative_design.json"
    if not baseline.exists():
        atomic_json(baseline, read(path))
    narrative = read(baseline)
    sections = narrative["sections"]
    base = root / f"milestones/step_{step:04d}/review/figures"
    sections[1]["design"] += (
        f"\n\n**Results through step {step}.** Each sigma has {step * 2 // 5} equally weighted F18 draws. "
        "Per-head mean ranges, in source sigma order, are "
        + ", ".join(f"[{low:.3f}, {high:.3f}]" for low, high in result["head_mean_ranges"])
        + ". "
        "Histogram quantiles use a 1/1024-wide bin approximation. "
        "The first-milestone examples remain fixed; overlays compare the first, preceding and current prefixes."
        + figure_markdown(root, base / "E1b_prefix_overlays", f"Fixed-head distribution overlays through step {step}.")
    )
    if step >= 200:
        conv = result["convergence"]
        sections[2]["design"] += (
            f"\n\n**Results through step {step}.** Cumulative mean-change medians are "
            + ", ".join(f"{v:.5f}" for v in conv["mean_change"]["median"])
            + "; recent-window distribution-distance medians are "
            + ", ".join(f"{v:.5f}" for v in conv["recent_window_distance"]["median"])
            + ", in source sigma order. "
            + (
                "Practical-stability labels require step 400; they are not available yet."
                if step < 400
                else "Stable fractions are " + ", ".join(f"{v:.1%}" for v in conv["stable_fraction"]) + "."
            )
            + " Repeated views and tokens are correlated. These changes are descriptive; "
            "no independent-subject confidence intervals are reported."
        )
    sections[3]["design"] += (
        f"\n\n**Observed sampling coverage through step {step}.** "
        + "; ".join(
            f"sigma {sigma}: {coverage['actors']} actors and {coverage['views']} views"
            for sigma, coverage in zip(result["sigmas"], result["coverage"], strict=True)
        )
        + ". The assigned views and noise differ, so this is not a same-crop sigma intervention."
    )
    sections[3]["design"] += figure_markdown(
        root,
        root / f"milestones/step_{step:04d}/figures/coverage",
        f"Training actor/view coverage by sigma through step {step}; this bounds the sampling interpretation.",
    )
    if step >= 200:
        sections[2]["design"] += figure_markdown(
            root,
            base / "E2c_sampling_coverage",
            f"Cumulative and disjoint-window actor/view coverage through step {step}; window sampling can fluctuate.",
        )
    sections[4]["design"] += f"\n\n**Results through step {step}.** "
    for name, metric in result["agreement"].items():
        aggregate = metric["aggregate"]
        recipe_note = "The r3 mask also reflects joint FFN training. " if name.startswith("r3_") else ""
        sections[4]["design"] += (
            f"For {name}, removed heads have mean {aggregate['removed_mean']:.3f}, versus "
            f"{aggregate['kept_mean']:.3f} for retained heads. The matched-count intersection is "
            f"{aggregate['intersection']}, versus {aggregate['random_expected_intersection']:.1f} "
            "under global random selection. "
            + recipe_note
            + "This is agreement with that saved recipe; it does not establish removal safety."
        )
        for role in ("global", "within_block"):
            sections[4]["design"] += figure_markdown(
                root,
                base / f"E4b_{name}_{role}_by_sigma",
                f"{name}: {role.replace('_', ' ')} matched head identities by sigma, through step {step}.",
            )
    sections[4]["design"] += (
        "\n\n**Historical event coverage.** All twelve eligible-population correlations are saved separately by sigma. "
        "The plots show early, middle and final selection events. "
        "Reset scores are not treated as a continuous final ranking."
    )
    for event in result["displayed_events"]:
        sections[4]["design"] += figure_markdown(
            root,
            base / f"E4d_event_{event:04d}_by_sigma",
            f"Event {event}: native means versus pre-reset drops by sigma, using eligible heads only.",
        )
    for fig in base.glob("*.png"):
        if not fig.with_suffix(".pdf").exists() or not fig.with_suffix(".data.json").exists():
            raise ValueError(f"supplemental figure companion missing: {fig}")
    atomic_json(path, narrative)


def review_milestone(root: Path, step: int) -> dict:
    audit = read(root / f"monitoring/audits/step_{step:04d}.json")
    study = read(root / "configs/frozen_study.json")
    if not audit["passed"] or audit["study"] != study["identity"] or audit["step"] != step:
        raise ValueError("supplemental analysis requires the matching saved-prefix audit")
    source = root / f"milestones/step_{step:04d}"
    result = read(source / "analysis.json")
    summary, _ = load_arrays(source / "analysis.npz")
    scores = summary["mean"]
    raw, _ = load_arrays(source / "cumulative.npz")
    if not np.array_equal(scores, raw["gate_sum"] / raw["tokens"][:, None, None]):
        raise ValueError("saved analysis means differ from audited sufficient statistics")
    del raw
    events = read(root / "comparison/events.json")
    target = source / "review/figures"
    density_overlays(root, step, target)
    records = {}
    for reference in read(root / "comparison/masks.json")["references"]:
        name = reference["name"]
        metrics = result["agreement"][name]
        identity_figures(target, name, np.asarray(reference["kept"], dtype=bool), metrics, study["sigmas"])
        records[name] = head_records(scores, reference, metrics)
    correlations = event_correlations(scores, events)
    drop_figures(target, scores, events, study["sigmas"], correlations)
    _, window = load_arrays(source / "window.npz")
    output = {
        **result,
        "sigmas": study["sigmas"],
        "head_mean_ranges": [[float(s.min()), float(s.max())] for s in scores],
        "event_correlations_by_sigma": correlations,
        "displayed_events": [events[i]["update"] for i in sorted({0, len(events) // 2 - 1, len(events) - 1})],
        "window_coverage": [
            {
                "actors": len({r["actor"] for r in window["receipts"] if r["sigma_index"] == j}),
                "views": len({r["view"] for r in window["receipts"] if r["sigma_index"] == j}),
            }
            for j in range(4)
        ],
    }
    atomic_json(
        source / "review/head_records.json",
        {"score_group_order": [*study["sigmas"], "equal_sigma"], "references": records},
    )
    atomic_json(source / "review/review.json", output)
    if step >= 200:
        sampling_coverage(root, step, output, target)
    write_narrative(root, step, output)
    files = [source / "review/review.json", source / "review/head_records.json", *sorted(target.glob("*"))]
    atomic_json(
        source / "review/complete.json",
        {
            "study": study["identity"],
            "step": step,
            "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        },
    )
    return output
