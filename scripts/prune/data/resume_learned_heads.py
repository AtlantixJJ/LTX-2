"""Fork a stopped soft checkpoint with an explicit sparsity-loss amendment.

Keep the source study and receipts immutable. Copy no corpus or model weights.
Bind unchanged trainable/optimizer/RNG state to the new frozen study identity.
Only the sparsity loss and its coefficient can change; t remains the global update.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity
from scripts.prune.learn import checkpoints
from scripts.prune.learn.heads import train_identity

AMENDMENT_CODE = {
    "scripts/prune/data/resume_learned_heads.py",
    "scripts/prune/learn/losses.py",
    "scripts/prune/learn/heads.py",
    "scripts/prune/checks/fsdp_native.py",
}


def unchanged_state_hash(state):
    """Hash saved payloads without the deliberately changed identity and lineage."""

    def encode(value):
        if isinstance(value, torch.Tensor):
            return {"tensor": checkpoints.tensor_hash(value)}
        if isinstance(value, np.ndarray):
            return {
                "numpy": str(value.dtype),
                "shape": list(value.shape),
                "bytes": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [encode(v) for v in value]}
        return value

    return checkpoints.digest(encode({k: v for k, v in state.items() if k not in ("identity", "lineage")}))


def prepare(previous, checkpoint, plan, output):
    previous, checkpoint, plan, output = (Path(p).resolve() for p in (previous, checkpoint, plan, output))
    old = json.loads(previous.read_text())
    old_id = old["identity"]
    if old.get("format") != "learned_head_study_v1" or old_id != checkpoints.digest(
        {k: v for k, v in old.items() if k != "identity"}
    ):
        raise ValueError("source frozen study changed")
    if file_sha256(old["plan"]["path"]) != old["plan"]["sha256"]:
        raise ValueError("source frozen plan changed")
    current = code_identity()
    changed_code = {
        p
        for p in set(current["files"]) | set(old["code"]["files"])
        if current["files"].get(p) != old["code"]["files"].get(p)
    }
    if changed_code - AMENDMENT_CODE:
        raise ValueError(f"unrelated production code changed: {sorted(changed_code - AMENDMENT_CODE)}")
    source = checkpoints.load(checkpoint)
    identity = source["identity"]
    seed, arm = identity["seed"], identity["arm"]
    draw_record = old["draws"][f"{seed}:ordinary"]
    if identity != train_identity(old, arm, seed, identity["removed"], draw_record):
        raise ValueError("checkpoint belongs to another training protocol")
    if source["phase"] != "soft" or not 0 < source["update"] < old["config"]["soft_updates"]:
        raise ValueError("loss amendment requires an incomplete soft checkpoint")
    expected_checkpoint = Path(old["root"]) / f"training/{arm}/replicate_{seed}/checkpoints/{checkpoint.name}"
    if checkpoint != expected_checkpoint.resolve() or output.exists():
        raise ValueError("source checkpoint path or fresh output differs")
    for pin in old["draws"].values():
        if file_sha256(pin["path"]) != pin["sha256"]:
            raise ValueError("paired draw bytes changed")
    prefix = {}
    for update in range(1, source["update"] + 1):
        path = checkpoint.parent.parent / f"updates/{update:04d}.json"
        receipt = json.loads(path.read_text())
        if receipt["study"] != old_id or receipt["update"] != update or len(receipt["draws"]) != 2:
            raise ValueError("source prefix update coverage differs")
        for draw in receipt["draws"]:
            manifest = Path(draw["input_manifest"])
            if file_sha256(manifest) != draw["input_manifest_sha256"]:
                raise ValueError("consumed prefix input changed")
            native = json.loads(manifest.read_text())
            epsilon = manifest.parent / native["videos"][0]["artifacts"]["epsilon"]
            if file_sha256(epsilon) != draw["epsilon_sha256"]:
                raise ValueError("consumed prefix epsilon changed")
        prefix[str(path)] = file_sha256(path)
    config = {**old["config"], "lambda_max": 1.0, "sparsity_penalty": "relu_logits", "sparsity_logit_offset": 3.0}
    config_path = output / "configs/config.json"
    checkpoints.publish_json(config_path, config)
    frozen_plan = output / "provenance/plan.md"
    checkpoints.publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    amendment = {
        "source_study": old_id,
        "source_study_path": str(previous),
        "source_study_sha256": file_sha256(previous),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_update": source["update"],
        "changes": {
            "lambda_max": [old["config"]["lambda_max"], 1.0],
            "sparsity_penalty": [old["config"].get("sparsity_penalty", "mean_sigmoid"), "relu_logits"],
            "sparsity_logit_offset": 3.0,
        },
        "changed_code": sorted(changed_code),
        "update_count_reset": False,
        "optimizer_reset": False,
        "rng_reset": False,
    }
    study = {
        **old,
        "config": config,
        "config_sha256": file_sha256(config_path),
        "root": str(output),
        "plan": {"path": str(frozen_plan), "sha256": file_sha256(frozen_plan), "original_path": str(plan)},
        "code": current,
        "continuation": amendment,
    }
    study.pop("identity")
    study["identity"] = checkpoints.digest(study)
    checkpoints.publish_json(output / "configs/frozen_study.json", study)
    state = copy.deepcopy(source)
    state["identity"] = train_identity(study, arm, seed, identity["removed"], draw_record)
    state["lineage"] = {
        **source["lineage"],
        "source_identity": identity,
        "source_checkpoint": str(checkpoint),
        "source_sha256": file_sha256(checkpoint),
        "prefix_receipts": prefix,
        "loss_amendment": amendment,
    }
    target = output / f"training/{arm}/replicate_{seed}/checkpoints/{checkpoint.name}"

    def write(p):
        with p.open("wb") as stream:
            torch.save(state, stream)

    checkpoints.publish(target, write)
    checkpoints.publish_json(
        target.with_suffix(".json"),
        {
            "identity": state["identity"],
            "update": state["update"],
            "phase": state["phase"],
            "sha256": file_sha256(target),
            "tensor_hashes": {k: checkpoints.tensor_hash(v) for k, v in state["adapters"].items()},
            "alpha_sha256": checkpoints.tensor_hash(state["alpha"]),
            "lineage": state["lineage"],
        },
    )
    restored = checkpoints.load(target, state["identity"])
    if unchanged_state_hash(source) != unchanged_state_hash(restored):
        raise ValueError("fork changed trainable state, optimizer, RNG, mask or draw cursor")
    gate_path = Path(old["root"]) / "checks/feasibility.json"
    gate = json.loads(gate_path.read_text())
    if gate["study"] != old_id or not gate["passed"] or gate["method"] != "four_rank_fsdp":
        raise ValueError("source native FSDP execution was not accepted")
    checkpoints.publish_json(
        output / "checks/feasibility.json",
        {
            **gate,
            "study": study["identity"],
            "reuse": {
                "path": str(gate_path),
                "sha256": file_sha256(gate_path),
                "source_study": old_id,
                "scope": "unchanged FSDP/forward execution; amended loss is checked on resumed updates",
            },
        },
    )
    environment = Path(old["root"]) / "provenance/environment.json"
    checkpoints.publish(output / "provenance/environment.json", lambda p: p.write_bytes(environment.read_bytes()))
    checks = {
        "study": study["identity"],
        "checkpoint": str(target),
        "resumed_update": source["update"],
        "state_payload_bit_identical": True,
        "state_payload_sha256": unchanged_state_hash(restored),
        "prefix_receipts": len(prefix),
        "next_update": source["update"] + 1,
        "next_lambda": min((source["update"] + 1) / 300, 1),
        "amendment": amendment,
    }
    checkpoints.publish_json(output / "provenance/continuation.json", checks)
    return study, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    study, checks = prepare(args.previous, args.checkpoint, args.plan, args.output)
    print(json.dumps({"study": study["identity"], "continuation": checks}, indent=2))


if __name__ == "__main__":
    main()
