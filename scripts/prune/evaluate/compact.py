"""Measure fixed compact recovery against its native and joint references.

Read saved checkpoints without optimizer updates. Export/infer native, qualified
joint split, initial compact, recovered split and recovered fused variants on
matched cases, then measure quality, bracketed cost and memory independently.
Media uses the same recorded capture crop, source frames and VAE coordinates.
The compact_A identifier is the recorded CLI/schema name for this workflow.

CLI: python -m scripts.prune.evaluate.compact --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import gc
import platform
import statistics
import time
from pathlib import Path

import torch

from scripts.prune.core import ltx_adapter, numerics, provenance, records, session
from scripts.prune.data import compact_study, study_io
from scripts.prune.evaluate import ffn, joint_final, timing
from scripts.prune.learn import adapters, compact_training, ddp_owner
from scripts.prune.score import export_joint_split, export_learned_ffn

# Model panel titles of the decoded comparison; each must fit one 384-pixel panel at the
# 480-pixel viewing width. The caption says the merged model is the update-500 result.
MEDIA_QUESTION = "Does compact recovery help?"
MEDIA_TITLES = (("native", "Native"), ("A", "A final"), ("initial", "Compact start"), ("merged", "Compact merged"))


def payload(study):
    root = Path(study["root"])
    owner = records.read_json(root / "monitoring/owner.json")
    if owner["status"] != "complete" or owner["update"] != study["config"]["recovery_updates"]:
        raise ValueError("Compact final evaluation requires completed training")
    return compact_training.read(root / f"training/checkpoints/update_{owner['update']:04d}.pt",
                                 study=study["identity"], world_size=4)


def attach(model, study, state):
    attached = adapters.attach(model, study["config"]["seed"], study["config"]["head_rank"])
    attached.update(adapters.attach_ffn(model, study["config"]["seed"], study["config"]["ffn_rank"]))
    adapters.load_state(attached, state["adapters"])
    return attached


def current_for(study, gpu):
    return session.open_frozen(study, gpu, script="prune.evaluate.compact")


def export(study):
    state = payload(study)
    root = Path(study["root"])
    path = root / "exports/recovered_merged.safetensors"
    if not path.exists():
        export_learned_ffn.merge(study["compact"]["path"], state["adapters"], path,
                                 pruning=dict(family="compact_A_retrain_v1", recovery="merged_fp32_delta_to_bf16"))
    record = dict(study=study["identity"], path=str(path), sha256=provenance.file_sha256(path),
                  execution="actual_compact_merged", qualification="pending_quality_and_fusion_replay")
    records.publish_json(root / "exports/recovered_merged.json", record)
    return path


def infer(study, gpu, arm):
    state = payload(study)
    root = Path(study["root"]) / f"evaluation/test/{arm}"
    current = current_for(study, gpu)
    path = export(study) if arm == "merged" else Path(study["compact"]["path"])
    identity = dict(study=study["identity"], arm=arm, split="test", checkpoint_update=state["update"],
                    schedules=[[study["config"]["sigmas"][i], 0.0] for i in study["config"]["monitor_sigma_indices"]],
                    seeds=study["config"]["evaluation_seeds"], weights=str(path))
    records.publish_json(root / "executor.json", dict(code=study_io.code_identity(), task=f"compact {arm} inference"))
    records.publish_json(root / "identity.json", identity)
    parent = Path(study["parent"]["input_root"])
    native = Path(records.read_json(parent / "configs/frozen_study.json")["parent"]["input_root"])

    def expected_input(dest):
        return records.read_json(native / f"evaluation/test/native/{dest.name}/case.json")["input_sha256"]

    with current.transformer(path) as model, torch.no_grad():
        model.eval()
        if arm == "split":
            attach(model, study, state)
        del state
        ffn.run_cases(study, current, model, root, identity, ffn.cases(study, "test"),
                      label=f"compact {arm}", input_case=ddp_owner.input_case, expected_input=expected_input)


def measure(model, video, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = sum(p.numel() for p in model.parameters())
    before = torch.cuda.memory_allocated(device)
    for _ in range(2):
        ltx_adapter.velocity_forward(model, video)
    samples = []
    for _ in range(7):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = ltx_adapter.velocity_forward(model, video)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
        del result
    return dict(seconds=samples, median_seconds=statistics.median(samples),
                peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device), resident_allocated=before,
                resident_parameters=resident, tokens=int(video.latent.shape[1]),
                tokens_per_second=video.latent.shape[1] / statistics.median(samples))


def benchmark(study, gpu):
    root = Path(study["root"])
    state, final = payload(study), export(study)
    current = current_for(study, gpu)
    parent = joint_final.load(Path(study["parent"]["path"]))
    split_bundle = Path(parent["root"]) / "exports/final_split.json"
    # Held-out test sources are all F28; the fixed F18 source is the first F18 validation view.
    sources = [next(s for s in study["splits"]["validation"] if s["n_latent_frames"] == 18),
               study["splits"]["test"][0]]
    all_rows, summaries = [], []
    for source in sources:
        case = ddp_owner.input_case(study, current, source, 2042, study["config"]["sigmas"][0])
        for baseline in ("native", "A_split"):
            for order in ((baseline, "merged", baseline), ("merged", baseline, "merged")):
                rows = []
                for arm in order:
                    manager = (export_joint_split.open_bundle(current, split_bundle) if arm == "A_split"
                               else current.transformer(final if arm == "merged" else None))
                    with manager as model, torch.no_grad():
                        model.eval()
                        value = dict(arm=arm, frames=source["n_latent_frames"], view=source["relative_dir"],
                                     **measure(model, case[1], current.device))
                        rows.append(value)
                        all_rows.append(value)
                        del model
                    gc.collect()
                    torch.cuda.empty_cache()
                result = timing.bracket(rows, "ABA" if order[0] == baseline else "BAB", "median_seconds")
                summaries.append(dict(baseline=baseline, frames=source["n_latent_frames"], order=list(order),
                                      speedup=result["speedup"], drift=result["drift_fraction"], rows=rows))
        del case
    # Separate adapter overhead from actual compact GEMM savings on the F18 source.
    case = ddp_owner.input_case(study, current, sources[0], 2042, study["config"]["sigmas"][0])
    for arm in ("initial", "split"):
        with current.transformer(Path(study["compact"]["path"])) as model, torch.no_grad():
            if arm == "split":
                attach(model, study, state)
            model.eval()
            all_rows.append(dict(arm=arm, frames=18, view=sources[0]["relative_dir"],
                                 **measure(model, case[1], current.device)))
            del model
        gc.collect()
        torch.cuda.empty_cache()
    records.publish_json(root / "timing/results.json", dict(
        study=study["identity"], gpu=gpu, device=torch.cuda.get_device_name(current.device),
        software=dict(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version()),
        code=study_io.code_identity(), sigma=study["config"]["sigmas"][0], seed=2042,
        sources=[dict(view=s["relative_dir"], actor=s["actor"], frames=s["n_latent_frames"]) for s in sources],
        warmups=2, repeats=7, rows=all_rows, brackets=summaries,
        speed_qualified=all(r["speedup"] >= 1.05 and r["drift"] < 0.02 for r in summaries),
        bytes=dict(native=Path(study["components"]["transformer"]["path"]).stat().st_size,
                   initial=Path(study["compact"]["path"]).stat().st_size, merged=final.stat().st_size),
        scope="transformer only; identical inputs; F18 and F28; no VAE/text encoding/loading in timed calls",
    ))


def media_keys(study):
    """Ordered (source, sigma, key) of the 36 decoded seed-2042 cases."""
    return [(source, study["config"]["sigmas"][index], ffn.key(source, 2042, study["config"]["sigmas"][index]))
            for source in study["splits"]["test"] for index in study["config"]["monitor_sigma_indices"]]


def media(study, gpu, shard=0, shards=1):
    """Decode this shard's test sources; a case with a published `results.json` is kept, not redone.

    Shards split by test-source index so each loads its capture references once. Whichever shard
    finds all cases published writes the combined `media/final/results.json`.
    """
    root = Path(study["root"])
    parent = Path(study["parent"]["input_root"])
    ancestor = Path(records.read_json(parent / "configs/frozen_study.json")["parent"]["input_root"])
    mine = [s for n, s in enumerate(study["splits"]["test"]) if n % shards == shard]
    todo = [(s, sigma, key) for s, sigma, key in media_keys(study)
            if s in mine and not (root / f"media/final/{key}/results.json").exists()]
    if todo:
        current = current_for(study, gpu)
        import lpips

        metric = lpips.LPIPS(net="alex", verbose=False).to(current.device).eval()
        with current.decoder() as decoder:
            for source in mine:
                cases_left = [(sigma, key) for s, sigma, key in todo if s is source]
                if not cases_left:
                    continue
                refs = ffn.references(current, decoder, metric, source)
                for sigma, key in cases_left:
                    dest = root / f"media/final/{key}"
                    paths = dict(native=ancestor / f"evaluation/test/native/{key}/case.json",
                                 A=parent / f"evaluation/test/final/{key}/case.json",
                                 initial=root / f"evaluation/test/initial/{key}/case.json",
                                 merged=root / f"evaluation/test/merged/{key}/case.json")
                    cases = {arm: records.read_json(path) for arm, path in paths.items()}
                    if len({r["input_sha256"] for r in cases.values()}) != 1:
                        raise ValueError("Compact video inputs differ")
                    metrics = ffn.compare_case(
                        current, decoder, metric, refs, cases,
                        list(MEDIA_TITLES), question=MEDIA_QUESTION, dest=dest, source=source, sigma=sigma, seed=2042,
                    )
                    row = dict(actor=source["actor"], view=source["relative_dir"], sigma=sigma, metrics=metrics,
                               comparison=str(dest / "comparison.mp4"), poster=str(dest / "poster.png"),
                               rendering=str(dest / "rendering.json"), input_sha256=cases["native"]["input_sha256"])
                    records.publish_json(dest / "results.json", row)
                    print("Compact decoded case", key, flush=True)
                del refs
    published = [root / f"media/final/{key}/results.json" for _, _, key in media_keys(study)]
    if all(path.exists() for path in published):
        records.publish_json(root / "media/final/results.json",
                             dict(study=study["identity"], cases=[records.read_json(path) for path in published]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2, 3, 4, 5), required=True)
    parser.add_argument("--stage", choices=("export", "infer", "benchmark", "media", "parity"), required=True)
    parser.add_argument("--arm", choices=("initial", "split", "merged"), default="merged")
    parser.add_argument("--shard", type=int, default=0, help="media only: this shard index")
    parser.add_argument("--shards", type=int, default=1, help="media only: number of source shards")
    args = parser.parse_args()
    numerics.deterministic_policy()
    torch.set_num_threads(4)
    # A completed study may be evaluated by a later source tree; each stage records its executor.
    study = compact_study.load(args.study, verify_code=False)
    if args.stage == "export":
        export(study)
    elif args.stage == "infer":
        infer(study, args.gpu_id, args.arm)
    elif args.stage == "benchmark":
        benchmark(study, args.gpu_id)
    elif args.stage == "parity":
        root = Path(study["root"])
        split = records.read_json(root / "evaluation/test/split/results.json")["rows"]
        merged = records.read_json(root / "evaluation/test/merged/results.json")["rows"]
        index = {(r["view"], r["seed"], r["sigma"]): r for r in split}
        rows = []
        for row in merged:
            reference = index[row["view"], row["seed"], row["sigma"]]
            if reference["input_sha256"] != row["input_sha256"]:
                raise ValueError("Fusion replay input differs")
            for value in (reference, row):
                if provenance.file_sha256(value["prediction"]) != value["prediction_sha256"]:
                    raise ValueError("Fusion replay prediction changed")
            a = torch.load(reference["prediction"], map_location="cpu", weights_only=True).float()
            b = torch.load(row["prediction"], map_location="cpu", weights_only=True).float()
            difference = (a - b).abs()
            maximum = float(difference.max())
            rows.append(dict(actor=row["actor"], view=row["view"], seed=row["seed"], sigma=row["sigma"],
                             max_abs=maximum, mean_abs=float(difference.mean()),
                             passed=bool(torch.isfinite(difference).all() and maximum <= 0.02)))
        if len(rows) != 108 or len(index) != 108:
            raise ValueError("Fusion replay requires all 108 paired test cases")
        records.publish_json(root / "exports/recovered_merged.parity.json",
                             dict(study=study["identity"], tolerance=0.02, cases=rows,
                                  passed=all(r["passed"] for r in rows)))
    else:
        media(study, args.gpu_id, args.shard, args.shards)


if __name__ == "__main__":
    main()
