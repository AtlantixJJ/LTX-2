"""V3 combined-mask export with terminal alignment and strict state provenance.

Merge FP32 adapter deltas before structural slicing. This artifact has a separate
family from historical head/FFN exports. No compact or speed qualification follows
from serialization alone. See doc/score/export_joint.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from scripts.prune.core import provenance, records
from scripts.prune.data import joint_study
from scripts.prune.learn import joint_training
from scripts.prune.score import export_learned_ffn, export_pruned

FORMAT = "whole_clip_d0_joint_readmission_v3"
FAMILY = "width_joint_readmission_recovered"


def terminal(study, path, *, probe=False):
    state = joint_training.read(path, study=study["identity"])
    policy = state["policy"]
    if not state["frozen"] or not policy["aligned"]:
        raise ValueError("joint export requires terminal aligned binary gates")
    if not probe and state["update"] != policy["selection_end"] + study["config"]["recovery_updates"]:
        raise ValueError("joint export requires the prespecified final checkpoint")
    if not all(int(n) % 128 == 0 for n in state["ffn_mask"].sum(1)):
        raise ValueError("joint export contains unaligned FFN widths")
    return state


def run_export(study, checkpoint, output, *, probe=False):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    state = terminal(study, checkpoint, probe=probe)
    source = study["components"]["transformer"]["path"]
    expected = study["components"]["transformer"]
    if provenance.file_sha256(source) != expected.get("file_sha256", expected.get("sha256")):
        raise ValueError("joint source weights changed")
    masks = export_learned_ffn.masks_from_state(state, export_pruned.checkpoint_mask_widths(source))
    envelope = dict(
        format=FORMAT,
        family=FAMILY,
        study_path=str(Path(study["root"]) / "configs/frozen_study.json"),
        study_identity=study["identity"],
        checkpoint=str(checkpoint),
        checkpoint_sha256=provenance.file_sha256(checkpoint),
        masks=masks,
        learning_actors=study["splits"]["learning_actors"],
        probe=probe,
        update=state["update"],
        selection_end=state["policy"]["selection_end"],
        qualification="unqualified_until_native_parity_and_quality",
        external_adapters=False,
        original_channel_order="ascending",
    )
    pruning = dict(
        family=FAMILY,
        qualification=envelope["qualification"],
        source_transformer_fingerprint=provenance.checkpoint_fingerprint(source),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Merge FP32 deltas into full-width weights, slice the coupled projections, then write once.
    tensors, meta = export_pruned.load(source)
    export_learned_ffn.merge_adapters(tensors, state["adapters"])
    export_pruned.export_tensors(
        tensors,
        meta,
        masks,
        widths=export_pruned.checkpoint_mask_widths(source),
        model_key="2.5",
        mode="compact",
        provenance_block=pruning,
    )
    export_learned_ffn.finalize(
        tensors, meta, output, pruning=pruning, envelope=envelope, envelope_path=output.with_suffix(".envelope.json")
    )
    del tensors
    records.publish_json(output.with_suffix(".json"), dict(path=str(output), sha256=provenance.file_sha256(output)))
    return output


def validate_envelope(path, *, base, candidate):
    envelope = json.loads(Path(path).read_text())
    if envelope.get("format") != FORMAT or envelope.get("family") != FAMILY or envelope.get("probe"):
        raise ValueError("invalid or disposable joint envelope")
    study = joint_study.load(envelope["study_path"])
    if study["identity"] != envelope["study_identity"]:
        raise ValueError("joint envelope study differs")
    if provenance.file_sha256(envelope["checkpoint"]) != envelope["checkpoint_sha256"]:
        raise ValueError("joint checkpoint changed")
    state = terminal(study, envelope["checkpoint"])
    source = study["components"]["transformer"]["path"]
    if export_learned_ffn.masks_from_state(state, export_pruned.checkpoint_mask_widths(source)) != envelope["masks"]:
        raise ValueError("joint original IDs differ")
    from scripts.prune.data.whole_clip import actor_identity
    from scripts.prune.learn.ddp_owner import verify_consumed

    verify_consumed(study, state["update"], state["input_chain"])
    if provenance.checkpoint_fingerprint(source) != base["model"]["transformer_fingerprint"]:
        raise ValueError("joint source differs from baseline")
    if any(actor_identity(row["view"]) in set(envelope["learning_actors"]) for row in base["videos"]):
        raise ValueError("joint evaluation reuses a learning actor")
    with safe_open(candidate, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(envelope["tensor_inventory"]):
            raise ValueError("joint tensor inventory differs")
        for key, record in envelope["tensor_inventory"].items():
            tensor = handle.get_tensor(key)
            if (
                list(tensor.shape) != record["shape"]
                or str(tensor.dtype) != record["dtype"]
                or records.tensor_hash(tensor) != record["sha256"]
            ):
                raise ValueError("joint merged tensor differs")
    return envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    print(run_export(joint_study.load(args.study), args.checkpoint, args.output, probe=args.probe))


if __name__ == "__main__":
    main()
