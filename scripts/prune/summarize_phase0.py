"""Collect every Phase 0 artifact into one ``analysis_summary.json`` + markdown tables.

Follows the ``scripts/ltx23_diag/README.md`` convention the plan points at (§12):
numbers are produced mechanically here, and the prose findings report is written
*from* this output rather than from re-reading logs. Every scalar quoted in
``expr/refiner_prune/<key>/FINDINGS.md`` should be traceable to a key emitted here.

    conda run -n ltx python -m scripts.prune.summarize_phase0 --model 2.5 --model 2.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.prune import artifacts
# A6000 dense bf16 tensor-core peak, for the MFU column. Not measured here -- it is
# the vendor number, quoted so "TFLOPS achieved" has a denominator.
A6000_BF16_PEAK_TFLOPS = 154.8


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def collect(key: str) -> dict:
    root = artifacts.root(key)
    gates = {
        "caps": _load(artifacts.gate(key, "caps")),
        "prompt_cache": _load(artifacts.gate(key, "prompt_cache_check")),
        "kv_cache": _load(artifacts.gate(key, "kv_cache_check")),
        "video_only": _load(artifacts.gate(key, "video_only_check")),
        "parity": _load(artifacts.gate(key, "parity_check")),
        "manifest": _load(artifacts.manifest(key)),
    }
    bench = _load(artifacts.bench(key, "baseline"))
    compile_bench = _load(artifacts.bench(key, "compile"))

    summary: dict = {"model": key, "gates": {}, "bench": {}, "teacher": {}}

    summary["gates"]["prompt_cache_bit_exact"] = (gates["prompt_cache"] or {}).get("bit_exact")
    summary["gates"]["kv_cache_bit_exact"] = (gates["kv_cache"] or {}).get("bit_exact")
    summary["gates"]["video_only_pass"] = (gates["video_only"] or {}).get("pass")
    summary["gates"]["parity_pass"] = (gates["parity"] or {}).get("pass")
    summary["gates"]["caps_dumped"] = gates["caps"] is not None

    if gates["video_only"]:
        mem = gates["video_only"]["memory"]
        summary["memory"] = {
            "audio_video_resident_gb": round(mem["audio_video"]["resident_alloc_gb"], 2),
            "video_only_resident_gb": round(mem["video_only"]["resident_alloc_gb"], 2),
            "resident_saved_gb": round(mem["resident_saved_gb"], 2),
            "build_peak_gb_audio_video": round(mem["audio_video"]["build_peak_alloc_gb"], 2),
            "build_peak_gb_video_only": round(mem["video_only"]["build_peak_alloc_gb"], 2),
        }
        summary["gates"]["video_only_max_abs_diff"] = max(c["max_abs_diff"] for c in gates["video_only"]["clips"])
        summary["gates"]["video_only_subjects"] = [c["subject"] for c in gates["video_only"]["clips"]]

    if gates["kv_cache"]:
        summary["kv_cache"] = {
            "cached_mb": gates["kv_cache"]["cached_mb"],
            "cached_tensors": gates["kv_cache"]["cached_tensors"],
            "reuse_hits": gates["kv_cache"]["reuse_pass"]["hits"],
            "reuse_misses": gates["kv_cache"]["reuse_pass"]["misses"],
        }

    if bench:
        rows = {(r["n_new_latent_frames"], r["ctx_latent_frames"], r["kv_cache"]): r for r in bench["rows"]}
        table = []
        for (n, c, kv), r in sorted(rows.items()):
            if kv:
                continue
            cached = rows.get((n, c, True))
            an = r["flops_per_fwd_analytic"]
            table.append(
                {
                    "n_new": n,
                    "ctx": c,
                    "tokens_video": r["tokens_video"],
                    "ms_per_fwd": round(r["ms_per_fwd"], 1),
                    "ms_per_fwd_kv_cached": round(cached["ms_per_fwd"], 1) if cached else None,
                    "kv_cache_saving_pct": round(100 * (r["ms_per_fwd"] - cached["ms_per_fwd"]) / r["ms_per_fwd"], 1)
                    if cached
                    else None,
                    "ms_per_output_latent_frame": round(r["ms_per_fwd"] / n, 1),
                    "tflops_achieved": round(r["tflops_achieved"], 1),
                    "mfu_pct": round(100 * r["tflops_achieved"] / A6000_BF16_PEAK_TFLOPS, 1),
                    "tflop_measured": round(r["flops_per_fwd_measured"] / 1e12, 1),
                    "tflop_analytic": round(an["total"] / 1e12, 1),
                    "analytic_error_pct": round(
                        100 * (r["flops_per_fwd_measured"] - an["total"]) / an["total"], 2
                    ),
                    "video_gemm_pct": round(100 * an["video_gemm"] / an["total"], 1),
                    "context_gemm_pct": round(100 * an["context_gemm"] / an["total"], 1),
                    "attn_pct": round(100 * (an["self_attn"] + an["cross_attn"]) / an["total"], 1),
                    "peak_alloc_gb": round(r["peak_alloc_gb"], 1),
                }
            )
        summary["bench"]["rows"] = table
        summary["bench"]["build"] = bench["builds"]
        summary["bench"]["provenance"] = bench["provenance"]
        summary["bench"]["analytic_error_pct_max"] = max(abs(t["analytic_error_pct"]) for t in table)

    if compile_bench:
        by = {(r["variant"], r["n_new_latent_frames"], r["ctx_latent_frames"]): r for r in compile_bench["rows"]}
        variants = sorted({v for v, _, _ in by})
        comp = []
        for (variant, n, c), r in sorted(by.items()):
            base = by.get(("eager", n, c))
            comp.append(
                {
                    "variant": variant,
                    "n_new": n,
                    "ctx": c,
                    "ms_per_fwd": round(r["ms_per_fwd"], 1),
                    "speedup_vs_eager": round(base["ms_per_fwd"] / r["ms_per_fwd"], 3) if base else None,
                }
            )
        summary["compile"] = {
            "variants": variants,
            "rows": comp,
            "builds": compile_bench["builds"],
        }

    if gates["manifest"]:
        m = gates["manifest"]
        summary["teacher"]["spec"] = m["target"]
        summary["teacher"]["student_sigmas"] = m["student"]["sigmas"]
        summary["teacher"]["corpus_clips"] = len(m["corpus"])
        summary["teacher"]["corpus_subjects"] = len({c["subject"] for c in m["corpus"]})
        summary["teacher"]["calibration_clips"] = len(m["split"]["calibration"])
        summary["teacher"]["held_out_clips"] = len(m["split"]["held_out"])
        summary["teacher"]["held_out_subjects"] = m["split"]["holdout_subjects"]
        k8 = [
            c["k_psnr_vs_source"]["k8"]
            for c in m["corpus"]
            if c.get("k_psnr_vs_source", {}).get("k8") is not None
        ]
        if k8:
            summary["teacher"]["on_disk_k8_psnr_mean"] = round(sum(k8) / len(k8), 2)
            summary["teacher"]["on_disk_k8_psnr_max"] = round(max(k8), 2)
    summary["phase1"] = _phase1(root)
    return summary


def _phase1(root: Path) -> dict:
    """The §6 gate: calibration cache, sampler A/B, and the unpruned T0/T1/T2 run.

    Kept mechanical for the same reason as everything above -- "Phase 1 passed" has
    to be readable off one JSON, not reconstructed by opening four files and
    remembering which one had the rollout in it.
    """
    out: dict = {}
    index = _load(artifacts.calibration_index(root.name))
    if index:
        records = index["records"]
        splits = {s: sum(1 for r in records if r["split"] == s) for s in ("calibration", "held_out")}
        out["calibration_cache"] = {
            "records": len(records),
            "clips": len({r["clip"] for r in records}),
            "by_split": splits,
            "families": sorted({r["family"] for r in records}),
            "chunk_sizes": sorted({r["chunk_latent_frames"] for r in records}),
            # Records carry both counts; the gap is the index-0 keyframe that
            # `denoise_mask` includes and the AR chunk does not.
            "keyframe_share_of_fresh_tokens": sorted({
                round(1 - r["chunk_tokens"] / r["fresh_tokens"], 3)
                for r in records if r.get("fresh_tokens") and r.get("chunk_tokens") is not None
            }),
            # A calibration cache with no calibration-split records cannot drive
            # Phase 2 at all, however many records it holds.
            "usable_for_phase2": splits.get("calibration", 0) > 0,
        }
    ab = _load(artifacts.gate(root.name, "sampler_ab"))
    if ab:
        out["sampler_ab"] = {
            "states": len(ab["states"]),
            "euler_t0_mean": ab["euler_t0_mean"],
            "ancestral_t0_mean": ab["ancestral_t0_mean"],
            "chosen_sampler": ab["chosen_sampler"],
        }
    g = _load(artifacts.phase1(root.name))
    if g:
        out["T0"] = {
            k: g["T0"].get(k) for k in ("calibration", "held_out", "loss_nonzero_at_every_step",
                                        "min_per_step_x0_mse_chunk")
        }
        out["T1"] = g.get("T1")
        t2 = g.get("T2") or {}
        out["T2"] = {k: v for k, v in t2.items() if k != "psnr_vs_teacher"}
        out["T3"] = g.get("T3")
    out["gate_complete"] = all(k in out for k in ("calibration_cache", "sampler_ab", "T0", "T2"))
    return out


def markdown(summary: dict) -> str:
    lines = [f"### {summary['model']}", ""]
    g = summary["gates"]
    lines += [
        "| Gate | Result |",
        "|---|---|",
        f"| `ModelCaps` dumped | {'yes' if g.get('caps_dumped') else 'NO'} |",
        f"| prompt-context cache bit-exact | {g.get('prompt_cache_bit_exact')} |",
        f"| cross-attn K/V cache bit-exact | {g.get('kv_cache_bit_exact')} |",
        f"| video-only == audio-video | {g.get('video_only_pass')} "
        f"(max abs diff {g.get('video_only_max_abs_diff')}) |",
        f"| registry refactor bit-for-bit | {g.get('parity_pass')} |",
        "",
    ]
    if "bench" in summary and summary["bench"].get("rows"):
        lines += [
            "| n_new | ctx | tokens | ms/fwd | +KV cache | KV Δ% | ms/out-frame | TFLOPS | MFU% | "
            "measured TF | analytic TF | err% | video GEMM% | ctx GEMM% | peak GB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in summary["bench"]["rows"]:
            lines.append(
                f"| {r['n_new']} | {r['ctx']} | {r['tokens_video']} | {r['ms_per_fwd']} | "
                f"{r['ms_per_fwd_kv_cached']} | {r['kv_cache_saving_pct']} | "
                f"{r['ms_per_output_latent_frame']} | {r['tflops_achieved']} | {r['mfu_pct']} | "
                f"{r['tflop_measured']} | {r['tflop_analytic']} | {r['analytic_error_pct']} | "
                f"{r['video_gemm_pct']} | {r['context_gemm_pct']} | {r['peak_alloc_gb']} |"
            )
        lines.append("")
    if "compile" in summary:
        lines += ["| variant | n_new | ctx | ms/fwd | speedup vs eager |", "|---|---:|---:|---:|---:|"]
        for r in summary["compile"]["rows"]:
            lines.append(
                f"| {r['variant']} | {r['n_new']} | {r['ctx']} | {r['ms_per_fwd']} | {r['speedup_vs_eager']} |"
            )
        lines.append("")
    p1 = summary.get("phase1") or {}
    if p1:
        cache, ab, t2 = p1.get("calibration_cache"), p1.get("sampler_ab"), p1.get("T2")
        lines += ["| Phase 1 gate | Result |", "|---|---|"]
        lines.append(
            f"| calibration cache | {cache['records']} records / {cache['clips']} clips, "
            f"calib {cache['by_split'].get('calibration', 0)} / held-out {cache['by_split'].get('held_out', 0)}, "
            f"usable {cache['usable_for_phase2']} |" if cache else "| calibration cache | MISSING |"
        )
        lines.append(
            f"| sampler A/B | {ab['chosen_sampler']} (euler {ab['euler_t0_mean']:.4f} vs "
            f"ancestral {ab['ancestral_t0_mean']:.4f}, {ab['states']} states) |" if ab else "| sampler A/B | MISSING |"
        )
        t0 = p1.get("T0")
        lines.append(
            f"| T0 unpruned (held-out, chunk tokens) | {t0['held_out']['rel_l2_chunk']['mean']:.4f} mean |"
            if t0 and (t0.get("held_out") or {}).get("rel_l2_chunk") else "| T0 unpruned | MISSING |"
        )
        lines.append(
            f"| T2 drift floor | {t2['chunks']} chunks, "
            f"{t2['psnr_slope_db_per_100_chunks']:.2f} dB/100 chunks |" if t2 else "| T2 drift floor | MISSING |"
        )
        lines += [f"| §6 gate complete | {p1.get('gate_complete')} |", ""]
    return "\n".join(lines)


def write_figures(summary: dict) -> list[Path]:
    """Render the Phase 0 measurements as durable review artifacts.

    JSON is authoritative for downstream computation; these plots make the
    small-chunk latency cliff and measured-vs-analytic FLOP agreement reviewable
    without manually reconstructing a chart from a table.
    """
    rows = summary.get("bench", {}).get("rows", [])
    if not rows:
        return []
    root = artifacts.figures(summary["model"])
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ctx in sorted({r["ctx"] for r in rows}):
        subset = sorted((r for r in rows if r["ctx"] == ctx), key=lambda r: r["n_new"])
        fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
        ax.plot([r["n_new"] for r in subset], [r["ms_per_fwd"] for r in subset], "o-", label="uncached")
        cached = [r for r in subset if r["ms_per_fwd_kv_cached"] is not None]
        if cached:
            ax.plot([r["n_new"] for r in cached], [r["ms_per_fwd_kv_cached"] for r in cached], "o--", label="K/V cached")
        ax.set(xlabel="fresh latent frames", ylabel="ms / transformer forward", title=f"LTX-{summary['model']} Phase 0 latency (ctx={ctx})")
        ax.grid(alpha=0.25); ax.legend()
        path = root / f"phase0_latency_ctx{ctx}.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    measured = [r["tflop_measured"] for r in rows]; analytic = [r["tflop_analytic"] for r in rows]
    hi = max(measured + analytic) * 1.05
    ax.scatter(analytic, measured, c=[r["ctx"] for r in rows], cmap="viridis", s=60)
    ax.plot([0, hi], [0, hi], "k--", linewidth=1, label="agreement")
    ax.set(xlabel="analytic TFLOP / forward", ylabel="measured TFLOP / forward", title=f"LTX-{summary['model']} FLOP cross-check", xlim=(0, hi), ylim=(0, hi))
    ax.grid(alpha=0.25); ax.legend()
    path = root / "phase0_flops_measured_vs_analytic.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    (root / "INDEX.md").write_text("# Phase 0 figures\n\n" + "\n".join(f"- `{p.name}`" for p in paths) + "\n")
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", dest="models", default=None)
    args = ap.parse_args()
    models = args.models or ["2.5", "2.3"]

    out = {}
    for key in models:
        summary = collect(key)
        out[key] = summary
        path = artifacts.gate(key, "analysis_summary")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        figures = write_figures(summary)
        print(markdown(summary))
        if figures:
            print("<!-- wrote figures: " + ", ".join(str(p) for p in figures) + " -->")
        print(f"<!-- wrote {path} -->\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
