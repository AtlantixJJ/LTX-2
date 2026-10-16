"""Freeze a zero/one continuation from the completed progressive step-500 state."""

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity, draw_list, load_study
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.heads import train_identity
from scripts.prune.learn.zero_one import ZeroOnePolicy

POLICY = dict(
    hardening_interval=50,
    removal_fraction=0.01,
    regret_threshold=0.1,
    fixed_one_fraction=0.005,
    fixed_one_regret_threshold=0.5,
)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(previous, plan, output, max_step=4000):
    previous, plan, output = (Path(v).resolve() for v in (previous, plan, output))
    if output.exists() or type(max_step) is not int or max_step <= 500 or max_step % 100:
        raise ValueError("requires fresh output and a final 100-update boundary beyond 500")
    old = read(previous)
    if old["identity"] != checkpoints.digest({k: v for k, v in old.items() if k != "identity"}):
        raise ValueError("source study digest changed")
    for pin in [old["plan"], *old["files"]]:
        if file_sha256(pin["path"]) != pin["sha256"]:
            raise ValueError(f"source pinned bytes changed: {pin['path']}")
    for pin in old["components"].values():
        if file_sha256(pin["path"]) != pin["file_sha256"]:
            raise ValueError("model component changed")
    parent_root = Path(old["root"])
    strict = sorted((parent_root / "analysis/audits").glob("*.json"), key=lambda p: p.stat().st_mtime)[-1]
    accepted = read(strict)
    if not accepted["complete"] or accepted["latest_audited_update"] != 500:
        raise ValueError("parent requires a complete strict training audit")
    path = parent_root / "training/learned_progressive/replicate_42/checkpoints/update_0500.pt"
    source = checkpoints.load(path)
    if (
        source["update"] != 500
        or source["draw_cursor"] != 2000
        or source["phase"] != "soft"
        or source["identity"] != train_identity(old, "learned_progressive", 42, 0, old["draws"]["42:ordinary"])
    ):
        raise ValueError("source checkpoint is not the accepted step-500 state")
    policy = ZeroOnePolicy(source["alpha"], POLICY, source["progressive"])
    if policy.counts() != dict(free=1376, removed=160, fixed_one=0):
        raise ValueError("source gate counts differ")
    if not torch.equal(policy.compose(source["alpha"]), source["alpha"].sigmoid() * policy.free):
        raise ValueError("migration changed executed gates")
    config = {
        **old["config"],
        "version": "progressive-zero-one-from-u500-v1",
        "soft_updates": max_step,
        "binary_updates": 0,
        "compact_updates": 0,
        "continuation_start_update": 500,
        "progressive_policy": POLICY,
    }
    protocol.validate(config)
    protocol.phase_budget(config)
    draws = draw_list(
        old["splits"]["learning"], 42, updates=max_step, sigma_indices=config["sigma_indices_cycle"], draws_per_update=4
    )
    parent_draws = read(old["draws"]["42:ordinary"]["path"])
    ancestor = read(old["progressive_source"]["study"])
    ancestor_pin = ancestor["draws"]["42:ordinary"]
    if file_sha256(ancestor_pin["path"]) != ancestor_pin["sha256"]:
        raise ValueError("ancestor draw bytes changed")
    ancestor_draws = read(ancestor_pin["path"])
    if draws[: len(parent_draws)] != parent_draws or draws[: len(ancestor_draws)] != ancestor_draws:
        raise ValueError("extended draws changed the existing parent/ancestor prefix")
    config_path, draw_path = output / "configs/config.json", output / "draws/replicate_42_ordinary.json"
    checkpoints.publish_json(config_path, config)
    checkpoints.publish_json(draw_path, draws)
    pin = dict(path=str(draw_path), sha256=file_sha256(draw_path))
    frozen_plan = output / "provenance/plan.md"
    checkpoints.publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    replaced = {old["draws"]["42:ordinary"]["path"], str(parent_root / "configs/config.json")}
    files = [v for v in old["files"] if v["path"] not in replaced]
    files += [pin, dict(path=str(config_path), sha256=file_sha256(config_path))]
    origin = dict(
        study=str(previous),
        study_sha256=file_sha256(previous),
        source_identity=old["identity"],
        checkpoint=str(path),
        checkpoint_sha256=file_sha256(path),
        update=500,
        strict_audit=str(strict),
        strict_audit_sha256=file_sha256(strict),
        source_environment_sha256=file_sha256(parent_root / "provenance/environment.json"),
        ancestor_draws_verified=len(ancestor_draws),
        parent_draws_verified=len(parent_draws),
    )
    study = {
        **old,
        "root": str(output),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "code": code_identity(),
        "files": files,
        "draws": {"42:ordinary": pin},
        "plan": dict(path=str(frozen_plan), sha256=file_sha256(frozen_plan)),
        "zero_one_source": origin,
    }
    study.pop("identity")
    study["identity"] = checkpoints.digest(study)
    checkpoints.publish_json(output / "configs/frozen_study.json", study)
    state = copy.deepcopy(source)
    state["progressive"] = policy.state_dict()
    state["identity"] = train_identity(study, "learned_progressive", 42, 0, pin)
    state["lineage"] = {**state["lineage"], "zero_one_source": origin}
    target = output / "training/learned_progressive/replicate_42/checkpoints/update_0500.pt"
    checkpoints.publish(target, lambda p: torch.save(state, p))
    checkpoints.publish_json(
        target.with_suffix(".json"),
        dict(
            identity=state["identity"],
            update=500,
            phase="soft",
            draws_per_update=4,
            sha256=file_sha256(target),
            lineage=state["lineage"],
            tensor_hashes={k: checkpoints.tensor_hash(v) for k, v in state["adapters"].items()},
            alpha_sha256=checkpoints.tensor_hash(state["alpha"]),
            progressive_sha256=checkpoints.digest(state["progressive"]),
        ),
    )
    restored = checkpoints.load(target, state["identity"])
    before, after = copy.deepcopy(source), copy.deepcopy(restored)
    before.pop("progressive")
    after.pop("progressive")
    if unchanged_state_hash(before) != unchanged_state_hash(after):
        raise ValueError("migration changed trainables, optimizer, RNG or cursor")
    for event in policy.history:
        oldpath = parent_root / f"training/learned_progressive/replicate_42/events/{event['update']:04d}.json"
        if read(oldpath) != event:
            raise ValueError("parent event receipt differs from checkpoint history")
        checkpoints.publish_json(
            output / f"training/learned_progressive/replicate_42/events/{event['update']:04d}.json", event
        )
    environment = parent_root / "provenance/environment.json"
    checkpoints.publish(output / "provenance/environment.json", lambda p: p.write_bytes(environment.read_bytes()))
    (output / "logs").mkdir()
    checks = dict(
        study=study["identity"],
        resumed_update=500,
        next_update=501,
        final_update=max_step,
        source=origin,
        preserved_state_sha256=unchanged_state_hash(after),
        trainables_optimizers_rng_cursor_identical=True,
        initial_executed_gates_identical=True,
        counts=policy.counts(),
        draws=len(draws),
        first_event=550,
        fresh_native_acceptance_required=True,
    )
    checkpoints.publish_json(output / "provenance/preparation.json", checks)
    load_study(output / "configs/frozen_study.json", verify_components=False)
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-step", type=int, default=4000)
    args = p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(args.previous, args.plan, args.output, args.max_step), indent=2))


if __name__ == "__main__":
    main()
