"""Fork a completed soft checkpoint into a new study with a longer phase budget.

Keep the source study and receipts immutable. Copy no corpus or model weights.
Bind unchanged trainable/optimizer/RNG state to the new frozen study identity.
Only the soft/binary/compact budgets change; draws extend with an identical prefix.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity, draw_list
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.heads import train_identity

AMENDMENT_CODE = {
    "scripts/prune/data/extend_learned_heads.py",
    "scripts/prune/learn/heads.py",
    "scripts/prune/learn/protocol.py",
    "scripts/prune/learn/run_full.py",
    "scripts/prune/learn/wandb_logger.py",
    "scripts/prune/score/export_learned_heads.py",
}
BUDGET_KEYS = ("soft_updates", "binary_updates", "compact_updates")


def extended_draws(old_study, seed, total_updates):
    """Regenerate one replicate's ordinary draws; the frozen prefix must be byte-for-byte equal."""
    config = old_study["config"]
    pin = old_study["draws"][f"{seed}:ordinary"]
    if file_sha256(pin["path"]) != pin["sha256"]:
        raise ValueError("paired draw bytes changed")
    old = json.loads(Path(pin["path"]).read_text())
    new = draw_list(
        old_study["splits"]["learning"],
        seed,
        updates=total_updates,
        sigma_indices=config["sigma_indices_cycle"],
        draws_per_update=protocol.draws_per_update(config),
    )
    if len(new) < len(old) or new[: len(old)] != old:
        raise ValueError("extended draws do not preserve the frozen prefix")
    return new


def prepare(previous, checkpoint, plan, output, budget):
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
    old_soft, _, _ = protocol.phase_budget(old["config"])
    new_config = {**old["config"], **dict(zip(BUDGET_KEYS, budget, strict=True))}
    soft_end, selection_end, compact_end = protocol.phase_budget(new_config)
    source = checkpoints.load(checkpoint)
    identity = source["identity"]
    seed, arm = identity["seed"], identity["arm"]
    if identity != train_identity(old, arm, seed, identity["removed"], old["draws"][f"{seed}:ordinary"]):
        raise ValueError("checkpoint belongs to another training protocol")
    if source["phase"] != "soft" or source["update"] != old_soft or soft_end <= source["update"]:
        raise ValueError("budget extension starts from the completed soft checkpoint and extends soft training")
    expected = Path(old["root"]) / f"training/{arm}/replicate_{seed}/checkpoints/{checkpoint.name}"
    if checkpoint != expected.resolve() or output.exists():
        raise ValueError("source checkpoint path or fresh output differs")
    count = protocol.draws_per_update(old["config"])
    prefix = {}
    for update in range(1, source["update"] + 1):
        path = checkpoint.parent.parent / f"updates/{update:04d}.json"
        receipt = json.loads(path.read_text())
        if receipt["study"] != old_id or receipt["update"] != update or len(receipt["draws"]) != count:
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
    config_path = output / "configs/config.json"
    checkpoints.publish_json(config_path, new_config)
    frozen_plan = output / "provenance/plan.md"
    checkpoints.publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    draws, old_draw_paths = {}, {v["path"] for v in old["draws"].values()}
    for key in old["draws"]:
        replicate, policy = key.split(":")
        if policy != "ordinary":
            raise ValueError("budget extension supports ordinary draws only")
        path = output / f"draws/replicate_{replicate}_ordinary.json"
        checkpoints.publish_json(path, extended_draws(old, int(replicate), compact_end))
        draws[key] = {"path": str(path), "sha256": file_sha256(path)}
    # Shared corpus/component pins stay; the source run's own draw and config pins are replaced.
    old_config = str(Path(old["root"]) / "configs/config.json")
    files = [pin for pin in old["files"] if pin["path"] not in old_draw_paths and pin["path"] != old_config]
    files += list(draws.values()) + [{"path": str(config_path), "sha256": file_sha256(config_path)}]
    amendment = {
        "source_study": old_id,
        "source_study_path": str(previous),
        "source_study_sha256": file_sha256(previous),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_update": source["update"],
        "changes": {k: [old["config"][k], new_config[k]] for k in BUDGET_KEYS if old["config"][k] != new_config[k]},
        "boundaries": {"soft_end": soft_end, "selection_end": selection_end, "compact_end": compact_end},
        "draw_prefix_preserved": True,
        "changed_code": sorted(changed_code),
        "update_count_reset": False,
        "optimizer_reset": False,
        "rng_reset": False,
    }
    study = {
        **old,
        "config": new_config,
        "config_sha256": file_sha256(config_path),
        "root": str(output),
        "draws": draws,
        "files": files,
        "plan": {"path": str(frozen_plan), "sha256": file_sha256(frozen_plan), "original_path": str(plan)},
        "code": current,
        "continuation": amendment,
    }
    study.pop("identity")
    study["identity"] = checkpoints.digest(study)
    checkpoints.publish_json(output / "configs/frozen_study.json", study)
    state = copy.deepcopy(source)
    state["identity"] = train_identity(study, arm, seed, identity["removed"], draws[f"{seed}:ordinary"])
    state["lineage"] = {
        **source["lineage"],
        "source_identity": identity,
        "source_checkpoint": str(checkpoint),
        "source_sha256": file_sha256(checkpoint),
        "prefix_receipts": prefix,
        "budget_amendment": amendment,
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
            "draws_per_update": state["draws_per_update"],
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
    if gate["study"] != old_id or not gate["passed"]:
        raise ValueError("source native execution was not accepted")
    checkpoints.publish_json(
        output / "checks/feasibility.json",
        {
            **gate,
            "study": study["identity"],
            "reuse": {
                "path": str(gate_path),
                "sha256": file_sha256(gate_path),
                "source_study": old_id,
                "scope": "unchanged execution, inputs and loss; only the phase budget changes",
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
    parser.add_argument("--soft-updates", type=int, required=True)
    parser.add_argument("--binary-updates", type=int, required=True)
    parser.add_argument("--compact-updates", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    budget = (args.soft_updates, args.binary_updates, args.compact_updates)
    study, checks = prepare(args.previous, args.checkpoint, args.plan, args.output, budget)
    print(json.dumps({"study": study["identity"], "continuation": checks}, indent=2))


if __name__ == "__main__":
    main()
