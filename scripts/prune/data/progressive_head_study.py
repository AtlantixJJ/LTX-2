"""Freeze a fresh progressive study from verified r13 bytes and checkpoint zero."""

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity, load_study
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.heads import train_identity
from scripts.prune.learn.progressive import ProgressivePolicy


def prepare(previous, config, plan, output):
    previous, config, plan, output = [Path(p).resolve() for p in (previous, config, plan, output)]
    if output.exists():
        raise ValueError("progressive preparation requires a fresh output")
    old = json.loads(previous.read_text())
    if old["identity"] != checkpoints.digest({k: v for k, v in old.items() if k != "identity"}):
        raise ValueError("source study identity changed")
    if old.get("format") != "learned_head_study_v1":
        raise ValueError("unknown source study")
    for pin in [old["plan"], *old["files"]]:
        if file_sha256(pin["path"]) != pin["sha256"]:
            raise ValueError(f"source bytes changed: {pin['path']}")
    for pin in old["components"].values():
        if file_sha256(pin["path"]) != pin["file_sha256"]:
            raise ValueError("source model component changed")
    cfg = {**old["config"], **json.loads(config.read_text())}
    protocol.validate(cfg)
    protocol.phase_budget(cfg)
    allowed = {
        "version",
        "soft_updates",
        "binary_updates",
        "compact_updates",
        "replicate_seeds",
        "progressive_policy",
        "removed_heads",
        "gpu_ids",
        "person_coefficient",
        "small_region_coefficient",
    }
    if any(cfg[k] != old["config"].get(k) for k in cfg if k not in allowed):
        raise ValueError("progressive overrides changed source numerical or input policy")
    if cfg["gpu_ids"] != [0, 1, 2, 3] or cfg["replicate_seeds"] != [42]:
        raise ValueError("progressive run requires GPUs 0-3 and seed 42")
    source_path = Path(old["root"]) / "training/learned_joint/replicate_42/checkpoints/update_0000.pt"
    source = checkpoints.load(source_path)
    if (
        source["update"] != 0
        or source["draw_cursor"] != 0
        or source["phase"] != "soft"
        or source["identity"] != train_identity(old, "learned_joint", 42, 152, old["draws"]["42:ordinary"])
        or not torch.equal(source["alpha"], torch.full((48, 32), 3.0))
    ):
        raise ValueError("source checkpoint zero differs")
    draw_pin = old["draws"]["42:ordinary"]
    draws = json.loads(Path(draw_pin["path"]).read_text())[:2000]
    if len(draws) != 2000:
        raise ValueError("source lacks 2000 draws")
    config_path = output / "configs/config.json"
    checkpoints.publish_json(config_path, cfg)
    draw_path = output / "draws/replicate_42_ordinary.json"
    checkpoints.publish_json(draw_path, draws)
    new_draw = {"path": str(draw_path), "sha256": file_sha256(draw_path)}
    frozen_plan = output / "provenance/plan.md"
    checkpoints.publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    replaced = {p["path"] for p in old["draws"].values()} | {str(Path(old["root"]) / "configs/config.json")}
    files = [p for p in old["files"] if p["path"] not in replaced]
    files += [new_draw, {"path": str(config_path), "sha256": file_sha256(config_path)}]
    study = {
        **old,
        "config": cfg,
        "config_sha256": file_sha256(config_path),
        "root": str(output),
        "draws": {"42:ordinary": new_draw},
        "files": files,
        "code": code_identity(),
        "plan": {"path": str(frozen_plan), "sha256": file_sha256(frozen_plan)},
        "progressive_source": {
            "study": str(previous),
            "sha256": file_sha256(previous),
            "checkpoint_zero": str(source_path),
            "checkpoint_sha256": file_sha256(source_path),
            "draw_prefix_sha256": checkpoints.digest(draws),
        },
    }
    study.pop("identity")
    study["identity"] = checkpoints.digest(study)
    checkpoints.publish_json(output / "configs/frozen_study.json", study)
    state = copy.deepcopy(source)
    state["identity"] = train_identity(study, "learned_progressive", 42, 0, new_draw)
    state["lineage"] = {"source_identity": source["identity"], **study["progressive_source"]}
    state["progressive"] = ProgressivePolicy(state["alpha"], cfg["progressive_policy"]).state_dict()
    target = output / "training/learned_progressive/replicate_42/checkpoints/update_0000.pt"

    def write(path):
        with path.open("wb") as stream:
            torch.save(state, stream)

    checkpoints.publish(target, write)
    checkpoints.publish_json(
        target.with_suffix(".json"),
        {
            "identity": state["identity"],
            "update": 0,
            "phase": "soft",
            "draws_per_update": 4,
            "sha256": file_sha256(target),
            "tensor_hashes": {k: checkpoints.tensor_hash(v) for k, v in state["adapters"].items()},
            "alpha_sha256": checkpoints.tensor_hash(state["alpha"]),
            "lineage": state["lineage"],
            "progressive_sha256": checkpoints.digest(state["progressive"]),
        },
    )
    restored = checkpoints.load(target, state["identity"])
    restored.pop("progressive")
    source.pop("progressive", None)
    if unchanged_state_hash(source) != unchanged_state_hash(restored):
        raise ValueError("checkpoint zero initial tensors, optimizers or RNG changed")
    if json.loads(draw_path.read_text()) != json.loads(Path(draw_pin["path"]).read_text())[:2000]:
        raise ValueError("draw prefix changed")
    (output / "logs").mkdir(exist_ok=True)
    checks = dict(
        study=study["identity"],
        initial_state_bit_identical=True,
        exact_draw_prefix=2000,
        source_bytes_verified=True,
        state_sha256=unchanged_state_hash(restored),
    )
    checkpoints.publish_json(output / "provenance/preparation.json", checks)
    load_study(output / "configs/frozen_study.json", verify_components=False)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "config", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(args.previous, args.config, args.plan, args.output), indent=2))


if __name__ == "__main__":
    main()
