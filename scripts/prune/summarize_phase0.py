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

from scripts.prune.model_registry import WORKSPACE_ROOT

OUT_ROOT = WORKSPACE_ROOT / "expr" / "refiner_prune"
# A6000 dense bf16 tensor-core peak, for the MFU column. Not measured here -- it is
# the vendor number, quoted so "TFLOPS achieved" has a denominator.
A6000_BF16_PEAK_TFLOPS = 154.8


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def collect(key: str) -> dict:
    root = OUT_ROOT / key
    gates = {
        "caps": _load(root / "caps.json"),
        "prompt_cache": _load(root / "prompt_cache_check.json"),
        "kv_cache": _load(root / "kv_cache_check.json"),
        "video_only": _load(root / "video_only_check.json"),
        "parity": _load(root / "parity_check.json"),
        "teacher_manifest": _load(root / "teacher" / "teacher_manifest.json"),
        "teacher_validation": _load(root / "teacher" / "teacher_validation.json"),
    }
    bench = _load(root / "bench_baseline.json")
    compile_bench = _load(root / "bench_compile.json")

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

    if gates["teacher_manifest"]:
        m = gates["teacher_manifest"]
        summary["teacher"]["spec"] = {k: v for k, v in m["teacher"].items() if k != "sigmas"}
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
    if gates["teacher_validation"]:
        summary["teacher"]["validation"] = gates["teacher_validation"]["clips"]

    return summary


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
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", dest="models", default=None)
    args = ap.parse_args()
    models = args.models or ["2.5", "2.3"]

    out = {}
    for key in models:
        summary = collect(key)
        out[key] = summary
        path = OUT_ROOT / key / "analysis_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        print(markdown(summary))
        print(f"<!-- wrote {path} -->\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
