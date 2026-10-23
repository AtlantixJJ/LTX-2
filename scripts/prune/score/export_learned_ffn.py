"""Merge learned head/FFN adapters and export original-ID binary selections.

masks_from_state restores both structural families. stream_merge holds one
changed tensor while applying adapter deltas and coupled slicing. The recovered
envelope pins state, masks, consumed draws and every output tensor. Probe
exports remain separate from completed recovery candidates. native_parity
compares split, fused and compact arithmetic on the same saved inputs.

CLI: python -m scripts.prune.score.export_learned_ffn --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.prune.core import numerics, provenance, records
from scripts.prune.data import ffn_study
from scripts.prune.learn import adapters, ffn_training
from scripts.prune.score import export_pruned, safetensors_stream

FORMAT = "whole_clip_d0_learned_head_ffn_v1"
FAMILY = "width_head_ffn_recovered"


def masks_from_state(state, widths):
    masks = {}
    for name, width in widths.items():
        block, kind = name.split(".")
        value = (
            state["ffn_mask"][int(block)]
            if kind == "ff"
            else state["heads_mask"][int(block)]
            if kind == "attn1"
            else torch.ones(width)
        )
        if (
            value.shape != (width,)
            or not torch.isfinite(value).all()
            or not ((value == 0) | (value == 1)).all()
            or not value.any()
        ):
            raise ValueError(f"invalid frozen head/FFN mask: {name}")
        masks[name] = value.int().tolist()
    return masks


def compact_adapter_state(state, head_mask, ffn_mask, *, head_dim=128):
    head = {k: v for k, v in state.items() if ".attn1." in k}
    ffn = {k: v for k, v in state.items() if ".ff." in k}
    if len(head) + len(ffn) != len(state):
        raise ValueError("unsupported learned adapter family")
    return {**adapters.transfer(head, head_mask, head_dim), **adapters.transfer_ffn(ffn, ffn_mask)}


TARGETS = {
    ("attn1", "v"): "attn1.to_v.weight",
    ("attn1", "out"): "attn1.to_out.0.weight",
    ("ff", "up"): "ff.net.0.proj.weight",
    ("ff", "down"): "ff.net.2.weight",
}


def adapter_targets(tensors, state, *, kinds=("attn1", "ff")):
    """Validate complete FP32 adapter pairs against the declared merge-coordinate shapes."""
    expected, targets = set(), {}
    for name in state:
        if not name.endswith(".A"):
            continue
        prefix = name[:-2]
        block, kind, projection = prefix.split(".")
        if kind not in kinds or (kind, projection) not in TARGETS or prefix + ".B" not in state:
            raise ValueError("unsupported or incomplete adapter target")
        a, b = state[name], state[prefix + ".B"]
        key = f"{export_pruned.PREFIX}.{block}.{TARGETS[kind, projection]}"
        base = tensors[key]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.dtype != torch.float32
            or b.dtype != torch.float32
            or b.shape[1] != a.shape[0]
            or (b.shape[0], a.shape[1]) != base.shape
        ):
            raise ValueError("compact adapter shape or dtype differs")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite compact adapter")
        targets[key] = (a, b)
        expected.update((name, prefix + ".B"))
    if set(state) != expected:
        raise ValueError("incomplete adapter fusion inventory")
    return targets


def _transform(value, *, fusion, selection, merge_first, scaling):
    if selection is not None and not merge_first:
        value = selection(value)
    if fusion is not None:
        a, b = fusion
        value = (value.float() + scaling * b @ a).to(value.dtype).contiguous()
    return selection(value) if selection is not None and merge_first else value


def stream_merge(
    source,
    state,
    output,
    *,
    masks=None,
    merge_first=False,
    kinds=("attn1", "ff"),
    scaling=1.0,
    pruning=None,
    envelope=None,
    envelope_path=None,
    inventory_dtype=True,
):
    """Fuse and slice one tensor at a time, preserving each family's arithmetic order."""
    original, meta = safetensors_stream.inspect(source)
    tensors, selections = original, {}
    if masks is not None:
        tensors, meta, selections = export_pruned.stream_plan(
            source, masks, model_key="2.5", mode="compact", provenance_block=pruning
        )
    targets = adapter_targets(original if merge_first else tensors, state, kinds=kinds)
    transforms = {
        key: partial(
            _transform, fusion=targets.get(key), selection=selections.get(key), merge_first=merge_first, scaling=scaling
        )
        for key in targets.keys() | selections.keys()
    }
    original_config = meta.get("config", "{}")

    def metadata(artifact_hash=None):
        binding = dict(pruning or {})
        if envelope is not None:
            binding.update(artifact=str(envelope_path), artifact_sha256=artifact_hash)
        if binding:
            config = json.loads(original_config)
            config["transformer"].setdefault("pruning", {}).update(binding)
            meta["config"] = json.dumps(config)
        return meta

    def bind(inventory):
        envelope["tensor_inventory"] = inventory
        records.publish_json(envelope_path, envelope)
        return metadata(provenance.file_sha256(envelope_path))

    return safetensors_stream.write(
        source,
        output,
        tensors,
        metadata("0" * 64),
        transforms=transforms,
        inventory=True,
        inventory_dtype=inventory_dtype,
        on_inventory=bind if envelope is not None else None,
    )


def merge(source, state, output, *, pruning=None, envelope=None, envelope_path=None):
    return stream_merge(source, state, output, pruning=pruning, envelope=envelope, envelope_path=envelope_path)


def run_export(study, checkpoint, output, *, probe=False):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    state = ffn_training.read(checkpoint, study=study["identity"])
    end = study["config"]["selection_updates"] + study["config"]["recovery_updates"]
    if not state["frozen"] or state["heads_mask"] is None or (not probe and state["update"] != end):
        raise ValueError("FFN export requires final frozen masks and completed recovery")
    source = study["components"]["transformer"]["path"]
    if provenance.file_sha256(source) != study["components"]["transformer"]["file_sha256"]:
        raise ValueError("FFN source checkpoint changed")
    masks = masks_from_state(state, export_pruned.checkpoint_mask_widths(source))
    compact = compact_adapter_state(state["adapters"], state["heads_mask"], state["ffn_mask"])
    envelope = dict(
        format=FORMAT,
        family=FAMILY,
        study_path=str(Path(study["root"]) / "configs/frozen_study.json"),
        study_identity=study["identity"],
        checkpoint=str(checkpoint),
        checkpoint_sha256=provenance.file_sha256(checkpoint),
        components=study["components"],
        masks=masks,
        original_channel_order="ascending",
        learning_actors=study["splits"]["learning_actors"],
        update=state["update"],
        probe=probe,
        qualification="unqualified_until_native_parity_and_quality",
        external_adapters=False,
    )
    pruning = dict(
        family=FAMILY,
        qualification=envelope["qualification"],
        source_transformer_fingerprint=provenance.checkpoint_fingerprint(source),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stream_merge(
        source,
        compact,
        output,
        masks=masks,
        pruning=pruning,
        envelope=envelope,
        envelope_path=output.with_suffix(".envelope.json"),
    )
    records.publish_json(output.with_suffix(".json"), dict(path=str(output), sha256=provenance.file_sha256(output)))
    return output


def validate_envelope(path, *, base, candidate):
    """Validate payload and actor scope; numerical/decoded acceptance stays separate."""
    envelope = json.loads(Path(path).read_text())
    if envelope.get("format") != FORMAT or envelope.get("family") != FAMILY or envelope.get("probe"):
        raise ValueError("invalid or disposable head/FFN envelope")
    study = ffn_study.load(envelope["study_path"])
    if study["identity"] != envelope["study_identity"]:
        raise ValueError("head/FFN envelope study differs")
    if provenance.file_sha256(envelope["checkpoint"]) != envelope["checkpoint_sha256"]:
        raise ValueError("head/FFN recovery checkpoint changed")
    state = ffn_training.read(envelope["checkpoint"], study=study["identity"])
    if (
        not state["frozen"]
        or state["update"] != study["config"]["selection_updates"] + study["config"]["recovery_updates"]
    ):
        raise ValueError("head/FFN recovery is incomplete")
    widths = export_pruned.checkpoint_mask_widths(study["components"]["transformer"]["path"])
    if masks_from_state(state, widths) != envelope["masks"]:
        raise ValueError("head/FFN original IDs changed")
    from scripts.prune.learn.ddp_owner import verify_consumed

    verify_consumed(study, state["update"], state["input_chain"])
    if (
        provenance.checkpoint_fingerprint(study["components"]["transformer"]["path"])
        != base["model"]["transformer_fingerprint"]
    ):
        raise ValueError("head/FFN source differs from baseline")
    from scripts.prune.data.whole_clip import actor_identity

    if any(actor_identity(row["view"]) in set(envelope["learning_actors"]) for row in base["videos"]):
        raise ValueError("head/FFN evaluation reuses a learning actor")
    with safe_open(candidate, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(envelope["tensor_inventory"]):
            raise ValueError("head/FFN merged tensor inventory differs")
        for key, expected in envelope["tensor_inventory"].items():
            if records.tensor_hash(handle.get_tensor(key)) != expected["sha256"]:
                raise ValueError("head/FFN merged tensor changed")
    return envelope


def native_parity(study, checkpoint, candidate, gpu_id):
    """Compare split adapters, fused masked weights and reloaded compact weights."""
    from scripts.onestep_avatar.model import common
    from scripts.prune.core import ltx_adapter, session
    from scripts.prune.learn import heads

    if gpu_id not in study["config"]["test_gpu_ids"]:
        raise ValueError("FFN parity tests use GPUs 4–6")
    numerics.deterministic_policy()
    current = session.open_model(
        "2.5",
        gpu_id,
        script="prune.score.export_learned_ffn",
        prompt="",
        transformer_path=Path(study["components"]["transformer"]["path"]),
    )
    source = study["splits"]["validation"][0]
    sigma = study["config"]["sigmas"][0]
    case = heads.input_case(study, current, source, 2042, sigma)
    payload = ffn_training.read(checkpoint, study=study["identity"])
    with current.transformer() as model:
        step = ffn_training.JointStep(model, study["config"])
        ao, go = ffn_training.optimizers(step)
        ffn_training.restore(step, ao, go, payload, restore_rng=False)
        del payload
        with torch.no_grad():
            step.masks()
            _, clean = ltx_adapter.velocity_forward(model, case[1])
            functional = common.with_clean_prefix(clean, case[2]).cpu()
            for adapter in step.attached.values():
                adapter.base.weight.copy_(adapter.merged_weight())
                adapter.enabled = False
            _, clean = ltx_adapter.velocity_forward(model, case[1])
            fused = common.with_clean_prefix(clean, case[2]).cpu()
        step.close()
        del step, ao, go, model, clean
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    with current.transformer(candidate) as model, torch.no_grad():
        _, clean = ltx_adapter.velocity_forward(model, case[1])
        compact = common.with_clean_prefix(clean, case[2]).cpu()
    error = lambda a, b: float((a.float() - b.float()).abs().max())
    result = dict(
        study=study["identity"],
        source=source["relative_dir"],
        schedule=[sigma, 0.0],
        split_vs_fused_max_abs=error(functional, fused),
        fused_vs_compact_max_abs=error(fused, compact),
        split_vs_compact_max_abs=error(functional, compact),
        tolerance=0.02,
        input_sha256=provenance.file_sha256(case[-2] / "manifest.json"),
        candidate_sha256=provenance.file_sha256(candidate),
    )
    result["passed"] = all(
        result[key] <= 0.02
        for key in ("split_vs_fused_max_abs", "fused_vs_compact_max_abs", "split_vs_compact_max_abs")
    )
    records.publish_json(Path(candidate).with_suffix(".parity.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--probe", action="store_true", help="Export a disposable native-check checkpoint; never qualify it."
    )
    parser.add_argument("--parity-gpu-id", type=int, help="Also run native held-out parity on a testing GPU.")
    args = parser.parse_args()
    study = ffn_study.load(args.study)
    candidate = run_export(study, args.checkpoint, args.output, probe=args.probe)
    print(candidate)
    if args.parity_gpu_id is not None:
        print(json.dumps(native_parity(study, args.checkpoint, candidate, args.parity_gpu_id)))


if __name__ == "__main__":
    main()
