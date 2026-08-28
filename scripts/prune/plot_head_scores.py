"""Render Phase 2 head-importance reports as reviewable figures.

The heatmaps use a rank within each attention block, rather than raw estimator
units, so a row directly answers the pruning question: which heads are the
lowest-ranked candidates in that block?  The agreement plot compares complete
rankings from estimators run on the same checkpoint and calibration records.

Example:
    conda run -n ltx python -m scripts.prune.plot_head_scores \
        --scores expr/refiner_prune/2.5/*-head-scores/head_scores.json \
        --output-dir expr/refiner_prune/2.5/head-score-figures
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ("contribution", "michel", "gauss_newton")


def _load(path: Path) -> dict:
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"score report does not exist: {path}") from exc
    if not report.get("methods"):
        raise ValueError(f"{path}: no score methods found")
    return report


def _rank(values: np.ndarray) -> np.ndarray:
    """Return average-tie ranks scaled to [0, 1], without a SciPy dependency."""
    flat = values.reshape(-1)
    if flat.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        end = start + 1
        while end < flat.size and flat[order[end]] == flat[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return (ranks / (flat.size - 1)).reshape(values.shape)


def _score_grid(scores: dict[str, list[float]], attention: str) -> np.ndarray:
    """Build a dense ``(layers, heads)`` score grid from ``'<layer>.<attn>'`` keys."""
    selected: dict[int, np.ndarray] = {}
    for name, values in scores.items():
        layer_text, kind = name.split(".", 1)
        if kind == attention:
            selected[int(layer_text)] = np.asarray(values, dtype=np.float64)
    if not selected:
        raise ValueError(f"report has no {attention} scores")
    heads = {row.size for row in selected.values()}
    if len(heads) != 1:
        raise ValueError(f"{attention} has inconsistent head counts: {sorted(heads)}")
    layers = max(selected) + 1
    if set(selected) != set(range(layers)):
        raise ValueError(f"{attention} layers are not contiguous: {sorted(selected)}")
    return np.stack([selected[layer] for layer in range(layers)])


def _validate_compatible(reports: list[tuple[Path, dict]]) -> None:
    reference = reports[0][1]
    expected = (
        reference["provenance"].get("model_key"),
        reference["provenance"].get("transformer_fingerprint"),
        reference.get("split"),
        reference.get("records"),
    )
    for path, report in reports[1:]:
        found = (
            report["provenance"].get("model_key"),
            report["provenance"].get("transformer_fingerprint"),
            report.get("split"),
            report.get("records"),
        )
        if found != expected:
            raise ValueError(
                f"{path} is not comparable to {reports[0][0]}: model/checkpoint/split/records differ"
            )


def _collect(reports: list[tuple[Path, dict]]) -> dict[str, dict[str, np.ndarray]]:
    methods: dict[str, dict[str, np.ndarray]] = {}
    for path, report in reports:
        for method, scores in report["methods"].items():
            if method in methods:
                raise ValueError(f"duplicate {method!r} scores: provide exactly one report per method")
            methods[method] = {attention: _score_grid(scores, attention) for attention in ("attn1", "attn2")}
    if not methods:
        raise ValueError("no head-score methods supplied")
    return dict(sorted(methods.items(), key=lambda item: (METHOD_ORDER.index(item[0]) if item[0] in METHOD_ORDER else 99, item[0])))


def _write_heatmaps(methods: dict[str, dict[str, np.ndarray]], output: Path, title: str) -> Path:
    figure, axes = plt.subplots(len(methods), 2, figsize=(14, max(3.5 * len(methods), 4)), squeeze=False)
    image = None
    for row, (method, grids) in enumerate(methods.items()):
        for col, attention in enumerate(("attn1", "attn2")):
            grid = grids[attention]
            # Normalize separately in each block: pruning makes choices among the
            # 32 competing heads in a block, not between estimator unit systems.
            ranked = np.stack([_rank(block) for block in grid])
            axis = axes[row, col]
            image = axis.imshow(ranked, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
            axis.set_title(f"{method}: {attention}")
            axis.set_xlabel("head")
            axis.set_ylabel("transformer block")
    assert image is not None
    # Reserve an explicit margin for the shared scale.  ``tight_layout`` cannot
    # reliably place a colorbar spanning this six-panel layout and can obscure
    # the right-hand attention maps.
    figure.subplots_adjust(left=0.06, right=0.89, bottom=0.06, top=0.90, hspace=0.35, wspace=0.14)
    color_axis = figure.add_axes((0.915, 0.17, 0.018, 0.68))
    figure.colorbar(image, cax=color_axis, label="within-block head rank (low → prune first)")
    figure.suptitle(title)
    path = output / "head_score_heatmaps.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = _rank(left).reshape(-1), _rank(right).reshape(-1)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _write_agreement(methods: dict[str, dict[str, np.ndarray]], output: Path) -> tuple[Path | None, list[dict]]:
    pairs = list(itertools.combinations(methods, 2))
    agreements = []
    if not pairs:
        return None, agreements
    figure, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.8), squeeze=False)
    for axis, (left_name, right_name) in zip(axes[0], pairs, strict=True):
        left = np.concatenate([methods[left_name][kind].reshape(-1) for kind in ("attn1", "attn2")])
        right = np.concatenate([methods[right_name][kind].reshape(-1) for kind in ("attn1", "attn2")])
        split = methods[left_name]["attn1"].size
        rho = _spearman(left, right)
        agreements.append({"left": left_name, "right": right_name, "spearman_rho": rho, "heads": int(left.size)})
        left_rank, right_rank = _rank(left), _rank(right)
        axis.scatter(left_rank[:split], right_rank[:split], s=5, alpha=0.35, label="attn1", rasterized=True)
        axis.scatter(left_rank[split:], right_rank[split:], s=5, alpha=0.35, label="attn2", rasterized=True)
        axis.plot((0, 1), (0, 1), color="black", linewidth=0.7, linestyle="--")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"{left_name} global rank", ylabel=f"{right_name} global rank")
        axis.set_title(f"Spearman ρ = {rho:.3f}" if rho is not None else "Spearman ρ undefined")
        axis.legend(markerscale=2)
    figure.suptitle("Estimator agreement across all attention heads")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = output / "head_score_agreement.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path, agreements


def render(score_paths: list[Path], output: Path, title: str | None = None) -> list[Path]:
    reports = [(path, _load(path)) for path in score_paths]
    _validate_compatible(reports)
    methods = _collect(reports)
    output.mkdir(parents=True, exist_ok=True)
    provenance = reports[0][1]["provenance"]
    figure_title = title or f"LTX-{provenance['model_key']} head importance ({len(reports[0][1]['records'])} calibration record(s))"
    heatmaps = _write_heatmaps(methods, output, figure_title)
    agreement, agreement_rows = _write_agreement(methods, output)
    summary = {
        "model_key": provenance["model_key"],
        "transformer_fingerprint": provenance.get("transformer_fingerprint"),
        "records": reports[0][1]["records"],
        "score_reports": [str(path) for path, _ in reports],
        "methods": list(methods),
        "layers": int(next(iter(methods.values()))["attn1"].shape[0]),
        "heads_per_attention": int(next(iter(methods.values()))["attn1"].shape[1]),
        "agreements": agreement_rows,
    }
    (output / "head_score_visualization.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Phase 2 head-score figures", "", "- `head_score_heatmaps.png`: rank of each head within its block; dark heads are pruning candidates."]
    if agreement is not None:
        lines.append("- `head_score_agreement.png`: global rank agreement between estimators, split by attention kind.")
    lines += ["", "## Inputs", ""] + [f"- `{path}`" for path, _ in reports]
    (output / "INDEX.md").write_text("\n".join(lines) + "\n")
    return [path for path in (heatmaps, agreement) if path is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True, help="One or more head_scores.json files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    output = args.output_dir or args.scores[-1].parent / "figures"
    for path in render(args.scores, output, args.title):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
