"""Decode A, selection, native and r3 on 36 prespecified held-out cases.

Run in ltx with --study and --gpu-id 4, 5 or 6. Waits, within --wait-hours, for
both fixed endpoint inference result sets and a free assigned GPU. No training
changes or retries. Decoding and scoring use the shared `evaluate.ffn` media steps.
See doc/evaluate/joint_media.md for metrics, source frames and limits.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch

from scripts.prune.core import numerics, provenance, records, waits
from scripts.prune.evaluate import ffn, joint_final


def read(path):
    return json.loads(Path(path).read_text())


def gpu_idle(gpu):
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    )
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    )
    return uuid.strip() not in active


def run(study, gpu, shard_index=0, shard_count=1, wait_hours=24.0):
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid disjoint media shard")
    root = Path(study["root"])
    deadline = time.monotonic() + wait_hours * 3600
    waits.until(
        lambda: all((root / f"evaluation/test/{arm}/results.json").exists() for arm in ("selection", "final")),
        deadline,
        "fixed endpoint inference results",
        receipts=root / "media/waits",
    )
    waits.until(lambda: gpu_idle(gpu), deadline, f"an idle GPU {gpu}", receipts=root / "media/waits")
    current = ffn.current_for(study, gpu)
    parent = Path(study["parent"]["input_root"])
    old = read(parent / "configs/frozen_study.json")
    if any(study[k] != old[k] for k in ("splits", "components", "text_context", "software", "schedule")):
        raise ValueError("V3 study inputs differ from the r3 parent")
    import lpips

    metric = lpips.LPIPS(net="alex", verbose=False).to(current.device).eval()
    titles = [
        ("native", "Native"),
        ("selection", "A before recovery"),
        ("final", "A after recovery"),
        ("r3", "Earlier r3 final"),
    ]
    summaries = []
    expected = sum(i % shard_count == shard_index for i in range(len(study["splits"]["test"]))) * 3
    with current.decoder() as decoder:
        for source_index, source in enumerate(study["splits"]["test"]):
            if source_index % shard_count != shard_index:
                continue
            refs = ffn.references(current, decoder, metric, source)
            for index in study["config"]["monitor_sigma_indices"]:
                sigma, seed = study["config"]["sigmas"][index], 2042
                k = ffn.key(source, seed, sigma)
                dest = root / f"media/test/{k}"
                if (dest / "results.json").exists():
                    summaries.append(read(dest / "results.json"))
                    continue
                paths = {
                    "native": parent / f"evaluation/test/native/{k}/case.json",
                    "selection": root / f"evaluation/test/selection/{k}/case.json",
                    "final": root / f"evaluation/test/final/{k}/case.json",
                    "r3": parent / f"evaluation/test/final/{k}/case.json",
                }
                rows = {arm: read(path) for arm, path in paths.items()}
                if (
                    len({r["input_sha256"] for r in rows.values()}) != 1
                    or len({r["noise_sha256"] for r in rows.values()}) != 1
                ):
                    raise ValueError("compared arms use different input or noise bytes")
                errors = ffn.compare_case(
                    current,
                    decoder,
                    metric,
                    refs,
                    rows,
                    titles,
                    question="Does A preserve the person better than r3?",
                    dest=dest,
                    source=source,
                    sigma=sigma,
                    seed=seed,
                )
                row = dict(
                    study=study["identity"],
                    actor=source["actor"],
                    view=source["relative_dir"],
                    sigma=sigma,
                    seed=seed,
                    metrics=errors,
                    vae_to_rgb_ceiling=refs["ceiling"],
                    matte_identity=refs["matte_identity"],
                    rendering=str(dest / "rendering.json"),
                    input_sha256=rows["native"]["input_sha256"],
                    noise_sha256=rows["native"]["noise_sha256"],
                    source_receipts={
                        arm: dict(path=str(path), sha256=provenance.file_sha256(path)) for arm, path in paths.items()
                    },
                )
                records.publish_json(dest / "results.json", row)
                summaries.append(row)
                print(
                    f"A media shard={shard_index} {len(summaries)}/{expected} actor={source['actor']} sigma={sigma}",
                    flush=True,
                )
                gc.collect()
            del refs
            gc.collect()
    result = "results.json" if shard_count == 1 else f"results_shard_{shard_index}.json"
    records.publish_json(
        root / f"media/test/{result}",
        dict(
            study=study["identity"],
            cases=summaries,
            shard_index=shard_index,
            shard_count=shard_count,
            population="Disjoint source views, seed 2042 and all three schedules",
            arms=["native", "selection", "final", "r3"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--wait-hours", type=float, default=24.0)
    args = parser.parse_args()
    numerics.deterministic_policy()
    torch.set_num_threads(4)
    run(joint_final.load(args.study), args.gpu_id, args.shard_index, args.shard_count, args.wait_hours)


if __name__ == "__main__":
    main()
