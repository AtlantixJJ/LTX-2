"""Export compact head widths, then merge recovered adapters without reselecting heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.prune.core import provenance
from scripts.prune.data import whole_clip
from scripts.prune.data.learned_head_study import load_study
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.score import export_pruned

FORMAT = "whole_clip_d0_learned_heads_v1"


def prepare_geometry_inputs(study, gpu_id):
    """Supply legacy seed-42 provenance without calibration or transformer forwards."""
    import hashlib

    from scripts.prune.learn import heads

    root = Path(study["root"])
    missing = []
    for source in study["splits"]["learning"]:
        key = hashlib.sha256(source["relative_dir"].encode()).hexdigest()
        inputs = root / "inputs" / key / "seed_42"
        if not (inputs / "manifest.json").exists():
            missing.append((source, inputs))
    if not missing:
        return
    heads.numerical_policy()
    current = heads.open_current(study, gpu_id)
    with torch.no_grad():
        for source, inputs in missing:
            whole_clip.prepare_score_inputs(inputs, source, current=current, sigmas=study["config"]["sigmas"], seed=42)
    print(f"saved {len(missing)} missing structural input manifests; no model forwards", flush=True)


def masks_from_tensor(mask, widths):
    names = [f"{b}.attn1" for b in range(len(mask))]
    if (
        mask.ndim != 2
        or not torch.isfinite(mask).all()
        or not torch.all((mask == 0) | (mask == 1))
        or any(widths.get(name) != mask.shape[1] for name in names)
        or (mask.sum(1) == 0).any()
    ):
        raise ValueError("invalid original-head mask")
    values = {name: mask[b].tolist() for b, name in enumerate(names)}
    values.update({name: [1] * width for name, width in widths.items() if name.endswith(".attn2")})
    return values


def geometry_wrappers(study, masks, destination):
    """Legacy manifests have one seed and one geometry; never combine clip lengths."""
    import copy
    import hashlib

    root = Path(study["root"])
    wrappers = {}
    for length in (18, 28):
        sources = [s for s in study["splits"]["learning"] if s["n_latent_frames"] == length]
        rows, first = [], None
        for source in sources:
            inputs = root / "inputs" / hashlib.sha256(source["relative_dir"].encode()).hexdigest() / "seed_42"
            native = whole_clip.load_manifest(inputs)
            if native["sigmas"] != study["config"]["sigmas"] or native["seed"] != 42:
                raise ValueError("legacy structural wrapper requires the pinned seed-42 learning inputs")
            if first is not None and native["geometry"] != first["geometry"]:
                raise ValueError("legacy wrapper contains different geometries")
            first = first or native
            for row in native["videos"]:
                row = copy.deepcopy(row)
                row["artifacts"]["epsilon"] = str(inputs / row["artifacts"]["epsilon"])
                rows.append(row)
        if first is None:
            raise ValueError("both complete clip lengths are required for structural wrappers")
        merged = {**first, "videos": rows}
        native_root = destination / f"learning_inputs_F{length}"
        checkpoints.publish_json(native_root / "manifest.json", merged)
        stamp = whole_clip.native_provenance(
            native_root, merged, [s["resolved_dir"] for s in sources], study["config"]["sigmas"]
        )
        path = destination / f"structural_mask_F{length}.json"
        checkpoints.publish_json(
            path, {"candidate_format": "whole_clip_d0_mask_v1", "provenance": stamp, "masks": masks}
        )
        wrappers[str(length)] = {"path": str(path), "sha256": provenance.file_sha256(path)}
    return wrappers


def merge(source, state, output, *, metadata=None, scaling=1.0, envelope=None, envelope_path=None):
    """Merge in FP32 and round once to the compact base dtype."""
    with safe_open(source, framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    expected = set()
    for name in state:
        if not name.endswith(".A"):
            continue
        prefix = name[:-2]
        block, kind, projection = prefix.split(".")
        if kind != "attn1" or projection not in ("v", "out"):
            raise ValueError("adapter targets unsupported projection")
        a, b = state[prefix + ".A"], state[prefix + ".B"]
        expected.update((prefix + ".A", prefix + ".B"))
        key = f"{export_pruned.PREFIX}.{block}.attn1." + ("to_v.weight" if projection == "v" else "to_out.0.weight")
        base = tensors[key]
        if (
            a.dtype != torch.float32
            or b.dtype != torch.float32
            or b.shape[1] != a.shape[0]
            or (b.shape[0], a.shape[1]) != base.shape
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise ValueError("compact adapter tensor shape or values differ")
        tensors[key] = (base.float() + scaling * b @ a).to(base.dtype).contiguous()
    if set(state) != expected:
        raise ValueError("incomplete adapter fusion inventory")
    inventory = {
        name: {"shape": list(value.shape), "sha256": checkpoints.tensor_hash(value)} for name, value in tensors.items()
    }
    if envelope is not None:
        envelope["tensor_inventory"] = inventory
        checkpoints.publish_json(envelope_path, envelope)
        metadata = {
            **(metadata or {}),
            "artifact": str(envelope_path),
            "artifact_sha256": provenance.file_sha256(envelope_path),
        }
    if metadata:
        config = json.loads(meta["config"])
        config["transformer"]["pruning"].update(metadata)
        meta["config"] = json.dumps(config)
    checkpoints.publish(output, lambda p: save_file(tensors, str(p), metadata=meta))
    return inventory


def run_export(study, arm, seed, stage):
    root = Path(study["root"])
    training = root / f"training/{arm}/replicate_{seed}"
    _, selection_end, compact_end = protocol.phase_budget(study["config"])
    selection_path = training / f"checkpoints/update_{selection_end:04d}.pt"
    selection = checkpoints.load(selection_path)
    if selection["identity"]["study"] != study["identity"]:
        raise ValueError("selection belongs to another frozen study")
    mask = selection["mask"].cpu()
    source = Path(study["components"]["transformer"]["path"])
    widths = export_pruned.checkpoint_mask_widths(source)
    values = masks_from_tensor(mask, widths)
    dest = root / f"exports/{arm}/replicate_{seed}"
    mask_path = root / f"masks/{arm}/replicate_{seed}/learned_heads.json"
    mask_record = {
        "format": FORMAT,
        "study": study["identity"],
        "mask": mask.tolist(),
        "head_ids": [torch.where(row == 1)[0].tolist() for row in mask],
        "removed_heads": int((mask == 0).sum()),
        "selection_checkpoint": str(selection_path),
        "selection_sha256": provenance.file_sha256(selection_path),
    }
    checkpoints.publish_json(mask_path, mask_record)
    wrappers = geometry_wrappers(study, values, mask_path.parent)
    base_path = dest / "compact_base.safetensors"
    pruning = {
        "family": "width_recovered",
        "source_transformer_fingerprint": provenance.checkpoint_fingerprint(source),
        "source_transformer_sha256": study["components"]["transformer"]["file_sha256"],
        "artifact": str(mask_path),
        "artifact_sha256": provenance.file_sha256(mask_path),
        "qualification": "compact_base_unqualified",
    }
    if not base_path.exists():
        checkpoints.publish(
            base_path,
            lambda p: export_pruned.export(
                source, values, p, model_key="2.5", mode="compact", provenance_block=pruning
            ),
        )
    if stage == "compact_base":
        return base_path
    final_step = compact_end + 1000 if arm == "learned_sequential" else compact_end
    final = checkpoints.load(training / f"checkpoints/update_{final_step}.pt")
    if final["identity"]["study"] != study["identity"] or not torch.equal(final["mask"].cpu(), mask):
        raise ValueError("recovery changed original head IDs or study")
    envelope_path = root / f"masks/{arm}/replicate_{seed}/recovered.json"
    output = dest / "final_merged.safetensors"
    envelope = {
        "format": FORMAT,
        "family": "width_recovered",
        "study_path": str(root / "configs/frozen_study.json"),
        "study_sha256": provenance.file_sha256(root / "configs/frozen_study.json"),
        "study_identity": study["identity"],
        "components": study["components"],
        "environment": {
            "path": str(root / "provenance/environment.json"),
            "sha256": provenance.file_sha256(root / "provenance/environment.json"),
        },
        "mask_path": str(mask_path),
        "mask_sha256": provenance.file_sha256(mask_path),
        "head_ids": mask_record["head_ids"],
        "budget": mask_record["removed_heads"],
        "compact_base": str(base_path),
        "compact_base_sha256": provenance.file_sha256(base_path),
        "recovery_checkpoint": str(training / f"checkpoints/update_{final_step}.pt"),
        "recovery_sha256": provenance.file_sha256(training / f"checkpoints/update_{final_step}.pt"),
        "adapter_rank": study["config"]["rank"],
        "adapter_scaling": 1.0,
        "adapter_hashes": {n: checkpoints.tensor_hash(v) for n, v in final["adapters"].items()},
        "training_identity": final["identity"],
        "config": study["config"],
        "lineage": final["lineage"],
        "learning_actors": study["splits"]["learning_actors"],
        "validation_actors": study["splits"]["validation_actors"],
        "reference_type": "in_memory_merged",
        "external_adapters": False,
        "qualification": "unqualified_until_native_checks_and_quality",
    }
    envelope["structural_wrappers"] = wrappers
    envelope["update_receipts"] = {
        **final["lineage"].get("prefix_receipts", {}),
        **{str(p): provenance.file_sha256(p) for p in sorted((training / "updates").glob("*.json"))},
    }
    # The envelope pins the inventory; the checkpoint pins the envelope. Avoid circular file hashes.
    with safe_open(base_path, framework="pt", device="cpu") as h:
        shapes = {k: h.get_slice(k).get_shape() for k in h.keys()}
    envelope["tensor_shapes"] = shapes
    merge(
        base_path,
        final["adapters"],
        output,
        metadata={**pruning, "qualification": envelope["qualification"]},
        envelope=envelope,
        envelope_path=envelope_path,
    )
    checkpoints.publish_json(
        dest / "final_merged.json",
        {
            "sha256": provenance.file_sha256(output),
            "artifact": str(envelope_path),
            "artifact_sha256": provenance.file_sha256(envelope_path),
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--replicate-seed", type=int, default=42)
    p.add_argument("--stage", choices=("compact_base", "final_merged"), required=True)
    p.add_argument("--input-gpu-id", type=int, help="Save missing seed-42 structural inputs on this GPU; no model load.")
    args = p.parse_args()
    study = load_study(args.study)
    if args.input_gpu_id is not None:
        prepare_geometry_inputs(study, args.input_gpu_id)
    print(run_export(study, args.arm, args.replicate_seed, args.stage))


if __name__ == "__main__":
    main()
