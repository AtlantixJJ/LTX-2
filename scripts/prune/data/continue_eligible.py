"""Freeze eligible-budget selection from a stopped checkpoint, preserving its partial window."""

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity, draw_list, load_study
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.eligible import EligiblePolicy
from scripts.prune.learn.heads import train_identity
from scripts.prune.learn.progressive import ProgressivePolicy

POLICY = dict(
    hardening_interval=50,
    removal_fraction=0.02,
    regret_threshold=0.1,
    fixed_one_fraction=0.02,
    fixed_one_regret_threshold=0.5,
    selection_population="eligible",
    minimum_count=16,
    finalize_updates=300,
)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(previous, plan, output, resume_update=1357, gate_only=False):
    policy_config = dict(POLICY)
    if gate_only:
        policy_config.update(fixed_one_selection="gate_only", fixed_one_regret_threshold=None)
    previous, plan, output = (Path(v).resolve() for v in (previous, plan, output))
    if output.exists() or type(resume_update) is not int or resume_update < 1:
        raise ValueError("requires fresh output and a positive source cursor")
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
    status = read(parent_root / "monitoring/status.json")
    if status["status"] != "stopped":
        raise ValueError("source owner must be stopped")
    strict = parent_root / "analysis/audit_latest.json"
    accepted = read(strict)
    if not accepted["passed"] or accepted["last_update"] != resume_update or accepted["study"] != old["identity"]:
        raise ValueError("parent requires a complete strict training audit")
    path = parent_root / f"training/learned_progressive/replicate_42/checkpoints/update_{resume_update:04d}.pt"
    source = checkpoints.load(path)
    if (
        source["update"] != resume_update
        or source["draw_cursor"] != resume_update * 4
        or source["phase"] != "soft"
        or source["identity"] != train_identity(old, "learned_progressive", 42, 0, old["draws"]["42:ordinary"])
    ):
        raise ValueError("source checkpoint is not the accepted step-500 state")
    parent_policy = ProgressivePolicy.from_state(source["progressive"])
    policy = EligiblePolicy(source["alpha"], policy_config, source["progressive"])
    # Every nonterminal event fixes at least one zero. This bound is not a target.
    first_event = (resume_update // POLICY["hardening_interval"] + 1) * POLICY["hardening_interval"]
    max_step = ((first_event + POLICY["hardening_interval"] * int(policy.free.sum()) + 300 + 99) // 100) * 100
    max_step = max(max_step, old["config"]["soft_updates"])
    if getattr(parent_policy, "final_update", None) is not None:
        raise ValueError("do not reset an already started finalization tail")
    if policy.counts() != accepted["counts"]:
        raise ValueError("source gate counts differ")
    if not torch.equal(policy.compose(source["alpha"]), parent_policy.compose(source["alpha"])):
        raise ValueError("migration changed executed gates")
    config = {
        **old["config"],
        "version": "progressive-gate-only-two-percent-final300-v1"
        if gate_only
        else "progressive-eligible-two-percent-final300-v1",
        "soft_updates": max_step,
        "binary_updates": 0,
        "compact_updates": 0,
        "continuation_start_update": resume_update,
        "progressive_policy": policy_config,
    }
    protocol.validate(config)
    protocol.phase_budget(config)
    draws = draw_list(
        old["splits"]["learning"], 42, updates=max_step, sigma_indices=config["sigma_indices_cycle"], draws_per_update=4
    )
    parent_draws = read(old["draws"]["42:ordinary"]["path"])
    if draws[: len(parent_draws)] != parent_draws:
        raise ValueError("extended draws changed the full source prefix")
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
        update=resume_update,
        strict_audit=str(strict),
        strict_audit_sha256=file_sha256(strict),
        source_environment_sha256=file_sha256(parent_root / "provenance/environment.json"),
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
        "comparison_parent_study": old.get("comparison_parent_study", old["zero_one_source"]["study"]),
        "comparison_baseline_study_identity": old.get(
            "comparison_baseline_study_identity", old["zero_one_source"]["source_identity"]
        ),
        "zero_one_source": origin,
        "eligible_source": origin,
    }
    study.pop("identity")
    study["identity"] = checkpoints.digest(study)
    checkpoints.publish_json(output / "configs/frozen_study.json", study)
    state = copy.deepcopy(source)
    state["progressive"] = policy.state_dict()
    state["identity"] = train_identity(study, "learned_progressive", 42, 0, pin)
    state["lineage"] = {**state["lineage"], "eligible_source": origin}
    target = output / f"training/learned_progressive/replicate_42/checkpoints/update_{resume_update:04d}.pt"
    checkpoints.publish(target, lambda p: torch.save(state, p))
    checkpoints.publish_json(
        target.with_suffix(".json"),
        dict(
            identity=state["identity"],
            update=resume_update,
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
        resumed_update=resume_update,
        next_update=resume_update + 1,
        mathematical_bound=max_step,
        final_update="first zero-ineligible event plus 300 updates",
        source=origin,
        preserved_state_sha256=unchanged_state_hash(after),
        trainables_optimizers_rng_cursor_identical=True,
        initial_executed_gates_identical=True,
        counts=policy.counts(),
        draws=len(draws),
        first_event=first_event,
        inherited_partial_snapshots=len(policy.window),
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
    p.add_argument("--resume-update", type=int, default=1357)
    p.add_argument("--gate-only", action="store_true", help="rank every remaining free head by gate, ignoring regret")
    args = p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(args.previous, args.plan, args.output, args.resume_update, args.gate_only), indent=2))


if __name__ == "__main__":
    main()
