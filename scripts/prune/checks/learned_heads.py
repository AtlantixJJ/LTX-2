"""Validate recovered head envelopes and independently saved numerical checks.

Check full source/components, original IDs, geometry wrappers, recovery cursor,
consumed inputs and every merged tensor hash. structural_inventory checks
coupled tensor shapes against declared head choices. These saved validators do
not select masks, train adapters or supply missing native evidence.

CLI: python -m scripts.prune.checks.learned_heads --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.prune.core import provenance, records
from scripts.prune.data.learned_head_study import load_study
from scripts.prune.learn import adapters, checkpoints, protocol
from scripts.prune.score.export_learned_heads import FORMAT
from scripts.prune.score.export_pruned import PREFIX


def compare(reference, candidate, *, prefix_tokens=0, tolerance=0.02):
    if reference.shape != candidate.shape or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("numerical check requires matched finite tensors")
    a, b = reference[:, prefix_tokens:].float(), candidate[:, prefix_tokens:].float()
    delta = b - a
    max_abs = float(delta.abs().max())
    return {
        "max_abs": max_abs,
        "relative_l2": float(delta.norm() / a.norm().clamp_min(1e-8)),
        "bit_exact": torch.equal(reference, candidate),
        "tolerance": tolerance,
        "passed": max_abs <= tolerance,
    }


def validate_envelope(path, *, base=None, candidate=None, verify_components=True):
    path = Path(path)
    record = json.loads(path.read_text())
    if record.get("format") != FORMAT or record.get("family") != "width_recovered":
        raise ValueError("unsupported recovered candidate format")
    if record.get("external_adapters") is not False or record.get("reference_type") != "in_memory_merged":
        raise ValueError("merged candidate cannot use external adapters or an unmerged reload reference")
    if provenance.file_sha256(record["environment"]["path"]) != record["environment"]["sha256"]:
        raise ValueError("recovered runtime/package/backend identity changed")
    for field, hash_field in (
        ("study_path", "study_sha256"),
        ("mask_path", "mask_sha256"),
        ("compact_base", "compact_base_sha256"),
        ("recovery_checkpoint", "recovery_sha256"),
    ):
        if provenance.file_sha256(record[field]) != record[hash_field]:
            raise ValueError(f"recovered artifact changed: {field}")
    study = load_study(record["study_path"], verify_components=verify_components)
    if (
        study["identity"] != record["study_identity"]
        or study["components"] != record["components"]
        or study["config"] != record["config"]
    ):
        raise ValueError("recovered study/config/component identities differ")
    config = record["config"]
    if (
        config["alpha_init"] != 3
        or config["target"] != "epsilon_minus_capture"
        or config["reduction"] != "generated_tokens_fp32_mean"
        or config["temperature"] != 1
    ):
        raise ValueError("unannounced training target or gate policy")
    ids = record["head_ids"]
    if len(ids) != config["blocks"] or any(
        h != sorted(set(h))
        or len(h) not in config["allowed_counts"]
        or any(type(i) is not int or not 0 <= i < config["heads"] for i in h)
        for h in ids
    ):
        raise ValueError("illegal or changed original head identities")
    if config["heads"] * config["blocks"] - sum(map(len, ids)) != record["budget"]:
        raise ValueError("recovered budget differs")
    mask = json.loads(Path(record["mask_path"]).read_text())
    if mask["head_ids"] != ids or mask["study"] != study["identity"]:
        raise ValueError("frozen mask and recovered head IDs differ")
    if [[i for i, v in enumerate(row) if v == 1] for row in mask["mask"]] != ids:
        raise ValueError("raw mask and original head identities differ")
    state = checkpoints.load(record["recovery_checkpoint"])
    soft_end, selection_end, _ = protocol.phase_budget(config)
    final_step = protocol.recovery_end(config, record["training_identity"]["arm"])
    if state["update"] != final_step or state["identity"] != record["training_identity"]:
        raise ValueError("recovery lineage or update count differs")
    if [torch.where(row == 1)[0].tolist() for row in state["mask"].cpu()] != ids:
        raise ValueError("recovery changed the frozen binary mask")
    if {n: records.tensor_hash(v) for n, v in state["adapters"].items()} != record["adapter_hashes"]:
        raise ValueError("recovery adapter hashes differ")
    if not record.get("update_receipts"):
        raise ValueError("recovery requires complete draw and noise evidence")
    expected_updates = (
        set(range(selection_end + 1, final_step + 1))
        if record["training_identity"]["arm"].endswith("_fresh")
        else set(range(1, final_step + 1))
    )
    actual_updates = set()
    for p, sha in record["update_receipts"].items():
        if provenance.file_sha256(p) != sha:
            raise ValueError("training draw receipt changed")
        receipt = json.loads(Path(p).read_text())
        actual_updates.add(receipt["update"])
        expected_study = study["identity"]
        if receipt["update"] <= soft_end and p in record["lineage"].get("prefix_receipts", {}):
            expected_study = record["lineage"]["source_identity"]["study"]
        if receipt["study"] != expected_study or len(receipt["draws"]) != config.get("global_unique_clips", 2):
            raise ValueError("incomplete paired microbatch coverage")
        for draw in receipt["draws"]:
            manifest_path = Path(draw["input_manifest"])
            if provenance.file_sha256(manifest_path) != draw["input_manifest_sha256"]:
                raise ValueError("consumed draw input changed")
            native = json.loads(manifest_path.read_text())
            epsilon = manifest_path.parent / native["videos"][0]["artifacts"]["epsilon"]
            if provenance.file_sha256(epsilon) != draw["epsilon_sha256"]:
                raise ValueError("consumed training epsilon changed")
            if study["config"].get("uniform_f18", False):
                from scripts.prune.data.training_windows import validate_saved_window

                if "training_window" not in draw:
                    raise ValueError("uniform F18 export requires consumed window evidence")
                validate_saved_window(manifest_path.parent, draw["training_window"])
    if actual_updates != expected_updates:
        raise ValueError("training update coverage differs from declared arm budget")
    for wrapper in record["structural_wrappers"].values():
        if provenance.file_sha256(wrapper["path"]) != wrapper["sha256"]:
            raise ValueError("geometry-specific structural mask wrapper changed")
        from scripts.prune.data import whole_clip

        wrapped = json.loads(Path(wrapper["path"]).read_text())
        whole_clip.validate_native_provenance(wrapped["provenance"])
        if any(wrapped["masks"].get(f"{b}.attn1") != mask["mask"][b] for b in range(len(ids))):
            raise ValueError("structural wrapper changed the canonical binary mask")
    if (
        record["learning_actors"] != study["splits"]["learning_actors"]
        or record["validation_actors"] != study["splits"]["validation_actors"]
    ):
        raise ValueError("learning/validation actor identities changed")
    if set(record["learning_actors"]) & set(record["validation_actors"]) or set(record["learning_actors"]) & set(
        study["splits"]["test_actors"]
    ):
        raise ValueError("test/learning actor overlap")
    if base is not None:
        if (
            provenance.checkpoint_fingerprint(record["components"]["transformer"]["path"])
            != base["model"]["transformer_fingerprint"]
            or base["model"]["model_key"] != "2.5"
            or base["text_context"]["sha256"] != study["text_context"]["bytes_sha256"]
        ):
            raise ValueError("baseline source or context differs from recovered study")
    if candidate is not None:
        with safe_open(candidate, framework="pt", device="cpu") as handle:
            shapes = {k: handle.get_slice(k).get_shape() for k in handle.keys()}
            architecture = json.loads(handle.metadata()["config"])["transformer"]
            pruning = architecture["pruning"]
            expected_rope = [heads if len(heads) != config["heads"] else None for heads in ids]
            if (
                architecture.get("per_layer_video_attn1_heads") != [len(heads) for heads in ids]
                or architecture.get("per_layer_video_attn1_rope_head_indices") != expected_rope
                or architecture.get("video_pruning_preserve_qk_norm") is not True
                or architecture.get("video_pruning_shape_faithful") is not False
                or pruning.get("mode") != "compact"
                or pruning.get("family") != "width_recovered"
            ):
                raise ValueError("recovered executed head geometry or original Q/K policy differs")
            if {
                k: {
                    "shape": list(handle.get_slice(k).get_shape()),
                    "sha256": records.tensor_hash(handle.get_tensor(k)),
                }
                for k in handle.keys()
            } != record["tensor_inventory"]:
                raise ValueError("merged tensor inventory content changed")
        if shapes != record["tensor_shapes"] or pruning.get("artifact_sha256") != provenance.file_sha256(path):
            raise ValueError("merged checkpoint inventory or artifact identity differs")
    return record


def structural_inventory(source, compact, head_ids, head_dim=128):
    """Independently compare every coupled tensor and untouched source tensor."""
    with safe_open(source, framework="pt", device="cpu") as a, safe_open(compact, framework="pt", device="cpu") as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError("compact base changed tensor inventory")
        for key in a.keys():
            expected = a.get_tensor(key)
            for block, heads in enumerate(head_ids):
                prefix = f"{PREFIX}.{block}.attn1."
                if not key.startswith(prefix):
                    continue
                channels = adapters.channel_indices(heads, head_dim)
                suffix = key[len(prefix) :]
                if suffix in ("to_v.weight", "to_v.bias"):
                    expected = expected[channels]
                elif suffix == "to_out.0.weight":
                    expected = expected[:, channels]
                elif suffix in ("to_gate_logits.weight", "to_gate_logits.bias"):
                    expected = expected[heads]
                break
            if not torch.equal(expected, b.get_tensor(key)):
                raise ValueError(f"compact source tensor differs: {key}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--replicate-seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int)
    args = p.parse_args()
    study = load_study(args.study)
    root = Path(study["root"])
    record = validate_envelope(
        root / f"masks/{args.arm}/replicate_{args.replicate_seed}/recovered.json",
        candidate=root / f"exports/{args.arm}/replicate_{args.replicate_seed}/final_merged.safetensors",
    )
    structural_inventory(record["components"]["transformer"]["path"], record["compact_base"], record["head_ids"])
    records.publish_json(
        root / f"checks/{args.arm}/replicate_{args.replicate_seed}/inventory.json",
        {"study": study["identity"], "passed": True, "native_checks": "not_run"},
    )
    if args.gpu_id is not None:
        from scripts.prune.evaluate.learned_heads import native_checks

        native_checks(study, args.arm, args.replicate_seed, args.gpu_id)


if __name__ == "__main__":
    main()
