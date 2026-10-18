"""Continue clipped-reset selection from a stopped capped run's step-500 event.

The source run ended selection at its cap. This owner replays every saved event
window through a new policy config with a larger cap and a stop rule, proves the
same removals and post-reset scores, and freezes a fresh study. Draws extend the
deterministic source order to the new bound; the prefix must match the source
and the old lineage's draws. See doc/data/continue_clipped.md.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity, draw_list, load_study
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.clipped_reset import ClippedResetPolicy, validate_config
from scripts.prune.learn.heads import train_identity


def read(path):
    return json.loads(Path(path).read_text())


def migrate(saved_state, config):
    """Replay saved windows under the new config; removals and resets must not change."""
    policy = ClippedResetPolicy(torch.ones(48, 32), config)
    for old in saved_state["history"]:
        trajectory = [torch.tensor(v, dtype=torch.float32) for v in old["trajectory"]]
        if not torch.equal(trajectory[0], policy.window[-1]):
            raise ValueError("saved window start differs from the replayed reset")
        for value in trajectory[1:]:
            event = policy.record(policy.update + 1, value.clone())
        if event is None or event["update"] != old["update"] or event["selected_ids"] != old["selected_ids"]:
            raise ValueError(f"replayed removal differs at event {old['update']}")
        if event["post_reset_scores"] != old["post_reset_scores"] or event["terminal_reason"] is not None:
            raise ValueError(f"replayed reset or stop differs at event {old['update']}")
    return policy


def prepare(previous, plan, output, resume_update, selection_updates, stop_rule, old_draws):
    previous, plan, output, old_draws = (Path(v).resolve() for v in (previous, plan, output, old_draws))
    if output.exists():
        raise ValueError("requires a fresh output root")
    old = read(previous)
    if old["identity"] != checkpoints.digest({k: v for k, v in old.items() if k != "identity"}):
        raise ValueError("source study digest changed")
    for pin in old["components"].values():
        if file_sha256(pin["path"]) != pin["file_sha256"]:
            raise ValueError("model component changed")
    parent_root = Path(old["root"])
    if read(parent_root / "monitoring/status.json")["status"] != "stopped":
        raise ValueError("source owner must be stopped")
    folder = parent_root / "training/learned_progressive/replicate_42"
    path = folder / f"checkpoints/update_{resume_update:04d}.pt"
    source = checkpoints.load(path)
    if (
        source["update"] != resume_update
        or source["draw_cursor"] != resume_update * 4
        or source["identity"] != train_identity(old, "learned_progressive", 42, 0, old["draws"]["42:ordinary"])
        or source["progressive"]["finalization_start"] != resume_update
        or source["gate_optimizer"]["state"]
    ):
        raise ValueError("source checkpoint is not the canonical post-reset cap event")
    policy_config = validate_config(
        dict(old["config"]["progressive_policy"], selection_updates=selection_updates, stop_rule=stop_rule)
    )
    policy = migrate(source["progressive"], policy_config)
    if not torch.equal(policy.window[-1], source["alpha"]) or policy.update != resume_update:
        raise ValueError("migrated policy differs from the saved post-reset scores")
    total = selection_updates + policy_config["finalize_updates"]
    config = {
        **old["config"],
        "version": "clipped-reset-continuation-v1",
        "soft_updates": selection_updates,
        "continuation_start_update": resume_update,
        "progressive_policy": policy_config,
    }
    protocol.validate(config)
    if protocol.phase_budget(config) != (selection_updates, selection_updates, total):
        raise ValueError("continuation budget differs")
    draws = draw_list(
        old["splits"]["learning"], 42, updates=total, sigma_indices=config["sigma_indices_cycle"], draws_per_update=4
    )
    parent_draws = read(old["draws"]["42:ordinary"]["path"])
    lineage_draws = read(old_draws)
    if draws[: len(parent_draws)] != parent_draws or draws != lineage_draws[: len(draws)]:
        raise ValueError("extended draws differ from the source or old-lineage order")
    config_path, draw_path = output / "configs/config.json", output / "draws/replicate_42_ordinary.json"
    checkpoints.publish_json(config_path, config)
    checkpoints.publish_json(draw_path, draws)
    pin = dict(path=str(draw_path), sha256=file_sha256(draw_path))
    frozen_plan = output / "provenance/plan.md"
    checkpoints.publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    replaced = {old["draws"]["42:ordinary"]["path"], str(parent_root / "configs/config.json")}
    files = [v for v in old["files"] if v["path"] not in replaced and Path(v["path"]).exists()]
    files += [pin, dict(path=str(config_path), sha256=file_sha256(config_path))]
    origin = dict(
        study=str(previous), study_sha256=file_sha256(previous), source_identity=old["identity"],
        checkpoint=str(path), checkpoint_sha256=file_sha256(path), update=resume_update,
        old_lineage_draws=str(old_draws), old_lineage_draws_sha256=file_sha256(old_draws),
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
        "clipped_continuation_source": origin,
    }
    study.pop("identity")
    study["identity"] = checkpoints.digest(study)
    checkpoints.publish_json(output / "configs/frozen_study.json", study)
    state = copy.deepcopy(source)
    state["progressive"] = policy.state_dict()
    state["identity"] = train_identity(study, "learned_progressive", 42, 0, pin)
    state["lineage"] = {**state["lineage"], "clipped_continuation_source": origin}
    target = output / f"training/learned_progressive/replicate_42/checkpoints/update_{resume_update:04d}.pt"
    checkpoints.publish(target, lambda p: torch.save(state, p))
    checkpoints.publish_json(
        target.with_suffix(".json"),
        dict(
            identity=state["identity"], update=resume_update, phase="soft", draws_per_update=4,
            sha256=file_sha256(target), lineage=state["lineage"],
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
        checkpoints.publish_json(
            output / f"training/learned_progressive/replicate_42/events/{event['update']:04d}.json", event
        )
    environment = parent_root / "provenance/environment.json"
    if environment.exists():
        checkpoints.publish(output / "provenance/environment.json", lambda p: p.write_bytes(environment.read_bytes()))
    (output / "logs").mkdir()
    checks = dict(
        study=study["identity"], resumed_update=resume_update, selection_cap=selection_updates,
        stop_rule=stop_rule, maximum_final_update=total, source=origin, counts=policy.counts(),
        replayed_events=len(policy.history), identical_removals_and_resets=True,
        preserved_state_sha256=unchanged_state_hash(after), draws=len(draws),
        draws_match_old_lineage=True, fresh_native_acceptance_required=True,
    )
    checkpoints.publish_json(output / "provenance/preparation.json", checks)
    load_study(output / "configs/frozen_study.json", verify_components=False)
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True, help="stopped capped study")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--old-draws", type=Path, required=True, help="old-lineage draw file whose order must match")
    p.add_argument("--resume-update", type=int, default=500)
    p.add_argument("--selection-updates", type=int, default=3000)
    p.add_argument("--stop-rule", default="no_strong_or_no_candidates")
    args = p.parse_args()
    torch.set_num_threads(4)
    checks = prepare(args.previous, args.plan, args.output, args.resume_update, args.selection_updates,
                     args.stop_rule, args.old_draws)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
