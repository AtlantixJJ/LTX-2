"""Decode matched native, joint, head-only and budget-control monitor predictions.

Legacy --study-a/--study-b/--study-c flags identify those saved arms. Verify
common model/context, exact source/noise hashes and the complete shared case
set before decoding. Render synchronized Capture RGB/VAE and four prediction
panels, with labels checked at narrow width before GPU work. These are matched
development monitors; no transformer or optimizer is constructed.

CLI: python -m scripts.prune.evaluate.joint_monitor_media --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from scripts.prune.core import numerics, provenance, records, waits
from scripts.prune.data import study_io
from scripts.prune.evaluate import ffn
from scripts.prune.evaluate.joint_media import gpu_idle


def read(path):
    return json.loads(Path(path).read_text())


def presentation(update):
    """Check the exact labels before waiting for a GPU or decoding a capture."""
    titles = [("native", "Native"), ("A", "A joint clipped"),
              ("B", "B heads only"), ("C", "C joint budget")]
    question = f"A, B and C at step {update}?"
    roles = [("capture_rgb", "Capture RGB"), ("capture_vae", "Capture VAE"), *titles]
    panels = [ffn.media.Panel(role, title, None, value="RGB" if role == "capture_rgb" else "VAE decoding")
              for role, title in roles]
    ffn.media.layout_geometry(panels, question=question, layout="comparison",
                              panel_size=(384, 384), viewing_width=480)
    return titles, question


def paired_rows(monitors):
    def key(row):
        return row["actor"], row["view"], row["sigma"]

    indexed = {arm: {key(row): row for row in monitor["cases"]} for arm, monitor in monitors.items()}
    expected = set(indexed["native"])
    if len(expected) != 12 or any(len(monitor["cases"]) != 12 for monitor in monitors.values()):
        raise ValueError("Monitor media requires exactly 12 distinct cases per arm")
    if any(set(values) != expected for values in indexed.values()):
        raise ValueError("Monitor media populations differ")
    for case in expected:
        reference = indexed["native"][case]
        for arm in indexed:
            row = indexed[arm][case]
            if any(row[name] != reference[name] for name in ("input_sha256", "noise_sha256")):
                raise ValueError("Monitor comparison uses mismatched input or noise")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-a", "study-b", "study-c", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, choices=(4, 5), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid monitor media shard")
    titles, question = presentation(args.update)
    numerics.deterministic_policy()
    torch.set_num_threads(4)
    studies = {
        arm: study_io.read_frozen(path, formats=("joint_study_v3",), verify_code=False)
        for arm, path in [("A", args.study_a), ("B", args.study_b), ("C", args.study_c)]
    }
    study = studies["C"]
    for other in studies.values():
        if any(other[name] != study[name] for name in ("components", "text_context", "splits")):
            raise ValueError("Monitor arms have different component/context/source identities")
    paths = {arm: Path(value["root"]) / f"monitor/update_{args.update:04d}.json" for arm, value in studies.items()}
    paths["native"] = Path(study["root"]) / "monitor/update_0000.json"
    monitors = {arm: read(path) for arm, path in paths.items()}
    indexed = paired_rows(monitors)
    waits.until(
        lambda: gpu_idle(args.gpu_id),
        time.monotonic() + 48 * 3600,
        f"idle GPU {args.gpu_id}",
        receipts=args.output / "waits",
        poll_seconds=20,
    )
    current = ffn.current_for(study, args.gpu_id)
    import lpips

    metric = lpips.LPIPS(net="alex", verbose=False).to(current.device).eval()
    sources = {source["relative_dir"]: source for source in study["splits"]["validation"]}
    views = sorted({case[1] for case in indexed["native"]})
    rows = []
    with current.decoder() as decoder:
        for i, view in enumerate(views):
            if i % args.shard_count != args.shard_index:
                continue
            source = sources[view]
            refs = ffn.references(current, decoder, metric, source)
            for case in sorted(key for key in indexed["native"] if key[1] == view):
                key = ffn.key(source, 2042, case[2])
                dest = args.output / key
                predictions = {arm: values[case] for arm, values in indexed.items()}
                for row in predictions.values():
                    if provenance.file_sha256(row["input_manifest"]) != row["input_sha256"]:
                        raise ValueError("Monitor input manifest bytes changed")
                errors = ffn.compare_case(
                    current,
                    decoder,
                    metric,
                    refs,
                    predictions,
                    titles,
                    question=question,
                    dest=dest,
                    source=source,
                    sigma=case[2],
                    seed=2042,
                )
                result = dict(
                    actor=case[0],
                    view=view,
                    sigma=case[2],
                    seed=2042,
                    update=args.update,
                    study=study["identity"],
                    metrics=errors,
                    population="development monitor",
                    input_sha256=predictions["native"]["input_sha256"],
                    noise_sha256=predictions["native"]["noise_sha256"],
                    comparison=str(dest / "comparison.mp4"),
                    poster=str(dest / "poster.png"),
                    rendering=str(dest / "rendering.json"),
                )
                records.publish_json(dest / "results.json", result)
                rows.append(result)
                print(f"Monitor media shard {args.shard_index}: {len(rows)} cases", flush=True)
                gc.collect()
            del refs
    records.publish_json(
        args.output / "results.json",
        dict(
            study=study["identity"],
            update=args.update,
            cases=rows,
            arms=["native", "A", "B", "C"],
            shard=args.shard_index,
            source_receipts={
                arm: dict(path=str(path), sha256=provenance.file_sha256(path)) for arm, path in paths.items()
            },
        ),
    )


if __name__ == "__main__":
    main()
