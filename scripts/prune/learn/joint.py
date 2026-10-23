"""Run joint clipped selection with allocation and alignment reviews.

Supply family hooks to learn.ddp_owner; learn.readmission owns decisions and
learn.joint_training owns numerical state. Pending review triggers stop new
updates, including legacy review records. New records use descriptive review
names so they are distinct from the package evidence gates.
See doc/learn/readmission.md for policy and review semantics.

CLI: python -m scripts.prune.learn.joint --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from scripts.prune.core import records
from scripts.prune.data import joint_study
from scripts.prune.learn import ddp_owner, joint_training

# The shared owner reads the numerical state owner through this attribute.
state = joint_training
label = "V3"
script = "prune.learn.joint"


def native_fixture(cfg):
    fixture = dict(
        cfg, policy=dict(interval=2, late=1, earliest=4, limit=6, recovery=1, alignment=128), recovery_updates=1
    )
    if cfg["arm"] == "budget_joint_v3":
        fixture.update(budget_ramp_start=0, budget_ramp_end=1)
    return fixture


def native_prepare(step, cfg):
    # Exercise returnable zero and permanent zero separately in both families.
    if cfg["arm"] != "budget_joint_v3":
        for gates in (step.heads,) if cfg["arm"] == "heads_clip_reset_v3" else (step.heads, step.ffn):
            gates.family.kept[0, :2] = False
            gates.family.permanent[0, 1] = True
            gates.family.reset_summary(gates.family.kept.float())
    step.policy.reset(step.heads.alpha, step.ffn.alpha, [])


def native_first_checks(step, first, cfg, runtime):
    if not first["adapter_gradient_norm"] > 0 or (
        cfg["arm"] != "heads_clip_reset_v3" and not first["ffn_gradient_norm"] > 0
    ):
        raise ValueError("native score or adapter gradient is zero")
    if not first["head_gradient_norm"] > 0:
        raise ValueError("native joint head gradient is zero")
    averaged = {}
    for name, gates in (("heads", step.heads), ("channels", step.ffn)):
        grad = gates.alpha.grad
        if (
            cfg["arm"] != "budget_joint_v3"
            and (name == "heads" or cfg["arm"] != "heads_clip_reset_v3")
            and (grad[0, 0] == 0 or grad[0, 1] != 0)
        ):
            raise ValueError("native returnable/permanent gradient check failed")
        digests = runtime.gather(records.tensor_hash(grad))
        if len(set(digests)) != 1:
            raise ValueError("native gate gradients differ between ranks")
        averaged[name] = digests[0]
    return dict(returnable_nonzero_permanent_zero=cfg["arm"] != "budget_joint_v3", averaged_gate_gradients=averaged)


def native_freeze(step, go):
    # Exact resume crosses a real event; the following disposable tail
    # explicitly freezes its binary result for recovery verification.
    if step.config["arm"] == "budget_joint_v3":
        step.policy.heads.kept = step.heads.decision.bool().cpu()
        step.policy.channels.kept = step.ffn.decision.bool().cpu()
    step.policy.selection_end = 2
    step.policy.reset(step.heads.alpha, step.ffn.alpha, go.items)
    step.policy.align(step.heads.alpha, step.ffn.alpha, go.items)
    return lambda: step.policy.channels.kept


def check_start(study, root, cfg, *, pilot, max_updates):
    if not pilot:
        prereq_record = study.get("prerequisites")
        if not prereq_record:
            raise ValueError("formal V3 training requires pinned P0/P1/P2 acceptance")
        joint_study.verify_files([prereq_record])
        prerequisite = json.loads(Path(prereq_record["path"]).read_text())
        if not prerequisite.get("passed") or not prerequisite.get("checks") or not all(prerequisite["checks"].values()):
            raise ValueError("formal V3 prerequisites are unresolved")
        joint_study.verify_files(prerequisite["source_receipts"])
    for review_path in (root / "reviews").glob("*.json"):
        review = json.loads(review_path.read_text())
        if review.get("triggers"):
            raise ValueError(f"review gate remains pending: {review_path}")
    if pilot and (max_updates is None or max_updates > 50):
        raise ValueError("disposable testing pilot is bounded to 50 updates")


def after_update(ctx, number):
    step, root = ctx.step, ctx.root
    payload = dict(
        study=ctx.study["identity"],
        update=number,
        dtype="float32",
        units="d_official_velocity_MSE/d_executed_gate",
        tensors=step.task_gradients,
        hashes={k: records.tensor_hash(v) for k, v in step.task_gradients.items()},
    )
    state.publish_record(root / f"training/gradients/{number:04d}.pt", payload)
    if step.trace is not None and not step.frozen:
        state.publish_record(root / f"training/traces/{number:04d}.pt", step.trace)
    if step.event:
        state.publish_record(root / f"training/events/{number:04d}.pt", step.event)


def receipt_fields(ctx, number):
    step = ctx.step
    return dict(
        phase="recovery" if step.frozen and number > step.policy.selection_end else "selection",
        ffn=step.ffn.snapshot(),
        heads=step.heads.snapshot(),
        pilot=ctx.pilot,
        projection=step.projection,
        optimizer_damping=step.optimizer_damping,
        head_phase=step.heads.phase,
    )


def reviews(ctx, number):
    """Review terminal alignment and early allocation independently of the evidence gates."""
    step, root, cfg, runtime = ctx.step, ctx.root, ctx.cfg, ctx.runtime
    review, stop = None, False
    if step.event and step.policy.recovering:
        # Preserve the pre-alignment binary result and score it.
        pre = ddp_owner.monitor(ctx.study, ctx.current, runtime, step, number, tag="pre_alignment")
        ids = step.policy.align(step.heads.alpha, step.ffn.alpha, ctx.go.items)
        post = ddp_owner.monitor(ctx.study, ctx.current, runtime, step, number, tag="aligned")
        baseline = json.loads((root / "monitor/update_0000.json").read_text())["capture_mse"]
        parity_record = ctx.study.get("prerequisites", {})
        parity = Path(parity_record.get("path", root / "checks/prerequisites.json"))
        prereq = json.loads(parity.read_text()) if parity.is_file() else {}
        triggers = []
        if max(pre["capture_mse"], post["capture_mse"]) / baseline > 1.03:
            triggers.append("monitor_above_three_percent")
        if not prereq.get("passed", False):
            triggers.append("prerequisites_unresolved")
        review = dict(
            gate="alignment_review",
            update=number,
            triggers=triggers,
            restored=ids,
            selection_end=number,
            budget=int(step.policy.heads.kept.sum()) * 704 + int(step.policy.channels.kept.sum()) - len(ids),
            pre=pre["capture_mse"],
            post=post["capture_mse"],
        )
        stop = stop or bool(triggers)
    if number == 100:
        result = ddp_owner.monitor(ctx.study, ctx.current, runtime, step, number)
        baseline = json.loads((root / "monitor/update_0000.json").read_text())["capture_mse"]
        triggers = []
        if (step.policy.channels.kept.sum(1) <= 0.6 * cfg["ffn_width"]).any():
            triggers.append("block_lost_forty_percent")
        if runtime.primary:
            last = [torch.load(root / f"training/events/{i:04d}.pt", weights_only=False) for i in (50, 100)]
            capped = all(e.get("families", {}).get("channels", {}).get("global_cap_bound", False) for e in last)
        else:
            capped = None
        if runtime.gather(capped)[0]:
            triggers.append("channel_cap_both_events")
        if result["capture_mse"] / baseline > 1.03:
            triggers.append("monitor_above_three_percent")
        review = dict(gate="allocation_review", update=number, triggers=triggers, monitor=result["capture_mse"])
        stop = stop or bool(triggers)
    if review and runtime.primary:
        records.publish_json(root / f"reviews/{review['gate']}_{number:04d}.json", review)
    return stop


def checkpoint_due(ctx, number, end):
    return number in (ctx.cfg["selection_updates"], end) or ctx.step.event is not None


def after_checkpoint(ctx, number):
    step = ctx.step
    if not (ctx.pilot and step.event and number == 50):
        return
    saved_h = step.heads.alpha.detach().clone()
    saved_c = step.ffn.alpha.detach().clone()
    saved_hp = step.policy.heads.permanent.clone()
    saved_cp = step.policy.channels.permanent.clone()
    with torch.no_grad():
        step.heads.alpha.copy_(step.event["families"]["heads"]["pre_scores"].to(step.heads.alpha.device))
        step.ffn.alpha.copy_(step.event["families"]["channels"]["pre_scores"].to(step.ffn.alpha.device))
    # Pre-event permanent masks exclude newly permanent removals.
    for family in (step.policy.heads, step.policy.channels):
        family.permanent &= family.final.eq(0)
    ddp_owner.monitor(ctx.study, ctx.current, ctx.runtime, step, number, tag="fractional_before_reset")
    with torch.no_grad():
        step.heads.alpha.copy_(saved_h)
        step.ffn.alpha.copy_(saved_c)
    step.policy.heads.permanent = saved_hp
    step.policy.channels.permanent = saved_cp


def monitor_due(ctx, number, end):
    return (number % ctx.cfg["monitor_interval"] == 0 or number == end) and number != 100


def final_update(ctx):
    return ctx.step.policy.final_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--phase", choices=("native", "formal", "train", "pilot"), default="formal")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    if args.testing and args.phase not in ("native", "pilot"):
        raise ValueError("the testing device group is reserved for disposable native tests")
    if args.phase == "pilot" and not args.testing:
        raise ValueError("pilot uses the frozen testing device list")
    ddp_owner.run(sys.modules[__name__], args, joint_study.load)


if __name__ == "__main__":
    main()
