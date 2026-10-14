"""Reuse saved raw head energies on the frozen learning subset; never load a model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.prune.core import provenance
from scripts.prune.data import whole_clip
from scripts.prune.data.head_score_dataset import SIGMAS
from scripts.prune.data.learned_head_study import load_study
from scripts.prune.learn import checkpoints
from scripts.prune.score import mobilewan_records


def reuse(study, previous):
    previous = Path(previous).resolve()
    old_path = previous / "configs/study.json"
    old = json.loads(old_path.read_text())
    if old["model_variant"] != "dev" or old["prompt"] != "" or not old["full_clip"]:
        raise ValueError("historical scoring uses another model or input task")
    for name, component in study["components"].items():
        if old["components"][name]["file_sha256"] != component["file_sha256"]:
            raise ValueError("historical scoring component differs")
    context_path = Path(old["text_context"]["cache_path"])
    context = torch.load(context_path, map_location="cpu", weights_only=True)
    if checkpoints.tensor_hash(context) != study["text_context"]["tensor_sha256"]:
        raise ValueError("historical scoring actual context differs")
    dataset_path = previous / old["dataset"]["manifest"]
    if provenance.file_sha256(dataset_path) != old["dataset"]["file_sha256"]:
        raise ValueError("historical source membership changed")
    sources = {s["relative_dir"]: s for s in json.loads(dataset_path.read_text())["sources"]}
    pins = {
        str(old_path): provenance.file_sha256(old_path),
        str(context_path): provenance.file_sha256(context_path),
        str(dataset_path): provenance.file_sha256(dataset_path),
    }
    for frames in (18, 28):
        gate = previous / f"validation/native_check_{frames}.json"
        record = json.loads(gate.read_text())
        if len(record["runs"]) != 8 or [r["sigma"] for r in record["runs"]] != SIGMAS:
            raise ValueError("historical scoring native coverage differs")
        if any(
            not all(
                r.get(k) is True
                for k in (
                    "stock_parity",
                    "hook_transparent",
                    "all_one_equal",
                    "saved_baseline_equal",
                    "clean_frame_equal",
                )
            )
            for r in record["runs"]
        ):
            raise ValueError("historical scoring native controls failed")
        pins[str(gate)] = provenance.file_sha256(gate)
    widths = {f"{b}.attn1": 32 for b in range(48)}
    rows, covered, missing = [], [], []
    for source in study["splits"]["learning"]:
        view = source["relative_dir"]
        key = hashlib.sha256(view.encode()).hexdigest()
        path = previous / f"scores/per_view/{key}.json"
        if not path.exists():
            missing.append(view)
            continue
        if source["actor"] in study["splits"]["validation_actors"] + study["splits"]["test_actors"]:
            raise ValueError("held-out actor entered calibration")
        old_source = sources[view]
        if any(
            old_source[k] != source[k] for k in ("actor", "shape", "capture_latent_sha256", "capture_encode_record")
        ):
            raise ValueError("historical capture source differs")
        record = json.loads(path.read_text())
        identity = record["identity"]
        if (
            identity["view"] != view
            or identity["actor"] != source["actor"]
            or identity["capture_sha256"] != source["capture_latent_sha256"]
            or identity["context_sha256"] != study["text_context"]["bytes_sha256"]
            or identity["transformer_sha256"] != study["components"]["transformer"]["file_sha256"]
            or identity["sampler"]["latent_frames"] != source["n_latent_frames"]
            or identity["sampler"]["sample_tokens"] != (source["n_latent_frames"] - 1) * 1024
        ):
            raise ValueError("historical score native identity differs")
        mobilewan_records.validate_record(record, identity, widths)
        inputs = previous / f"inputs/views/{key}"
        manifest = whole_clip.load_manifest(inputs)
        if (
            manifest["sigmas"] != SIGMAS
            or manifest["seed"] != 42
            or manifest["videos"][0]["artifacts"]["epsilon_sha256"] != identity["epsilon_sha256"]
            or manifest["videos"][0]["artifacts"]["capture_sha256"] != identity["capture_sha256"]
        ):
            raise ValueError("historical scoring noise or capture provenance differs")
        whole_clip.load_epsilon(inputs, manifest["videos"][0])
        pins[str(path)] = provenance.file_sha256(path)
        pins[str(inputs / "manifest.json")] = provenance.file_sha256(inputs / "manifest.json")
        pins[str(inputs / manifest["videos"][0]["artifacts"]["epsilon"])] = identity["epsilon_sha256"]
        rows.append(torch.tensor([record["energies"][f"{b}.attn1"] for b in range(48)], dtype=torch.float64).mean(1))
        covered.append(view)
    if not rows:
        raise ValueError("no compatible learning-view scores available")
    raw = torch.stack(rows).mean(0)
    if not torch.isfinite(raw).all() or (raw.mean(1) <= 0).any():
        raise ValueError("invalid pooled historical energy")
    result = {
        "study": study["identity"],
        "scores": (raw / raw.mean(1, keepdim=True)).tolist(),
        "raw_pooled": raw.tolist(),
        "views": covered,
        "missing_views": missing,
        "requested_views": [s["relative_dir"] for s in study["splits"]["learning"]],
        "status": "complete_existing_subset",
        "scope": "existing_learning_subset",
        "records": pins,
        "source_root": str(previous),
        "normalization": "raw sigma mean, equal view mean, then block mean",
    }
    checkpoints.publish_json(Path(study["root"]) / "calibration/ranking.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = reuse(load_study(args.study), args.previous)
    print(f"reused {len(result['views'])} learning views; {len(result['missing_views'])} missing", flush=True)


if __name__ == "__main__":
    main()
