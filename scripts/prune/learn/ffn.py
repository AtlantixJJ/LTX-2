"""Bounded native checks and DDP training for learned FFN and head masks.

This family module supplies FFN hooks to the shared DDP owner (`learn.ddp_owner`).
Selection learns aligned binary channel masks, and head gates in the joint arm;
fixed-mask adapter recovery follows. See doc/learn/ffn.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.prune.data import ffn_study
from scripts.prune.learn import ddp_owner, ffn_training

# The shared owner reads the numerical state owner through this attribute.
state = ffn_training
label = "FFN"
script = "prune.learn.ffn"


def native_fixture(cfg):
    return dict(cfg, head_soft_updates=1, selection_updates=2, recovery_updates=1, ffn_warmup=0, ffn_ramp_end=1)


def native_prepare(step, cfg):
    return None


def native_first_checks(step, first, cfg, runtime):
    if not first["adapter_gradient_norm"] > 0 or not first["ffn_gradient_norm"] > 0:
        raise ValueError("FFN native score or adapter gradient is zero")
    if cfg["arm"] == "joint_heads_ffn" and not first["head_gradient_norm"] > 0:
        raise ValueError("native joint head gradient is zero")
    return {}


def native_freeze(step, go):
    # The two-update fixture already froze its masks at its selection end.
    return lambda: step.ffn.decision


def check_start(study, root, cfg, *, pilot, max_updates):
    if pilot:
        raise ValueError("the FFN owner has no pilot phase")


def after_update(ctx, number):
    return None


def receipt_fields(ctx, number):
    step = ctx.step
    return dict(
        phase="recovery" if number > ctx.cfg["selection_updates"] else "selection",
        ffn=step.ffn.snapshot(),
        head_phase=step.heads.phase,
    )


def reviews(ctx, number):
    return False


def checkpoint_due(ctx, number, end):
    return number in (ctx.cfg["head_soft_updates"], ctx.cfg["selection_updates"], end)


def after_checkpoint(ctx, number):
    return None


def monitor_due(ctx, number, end):
    return number % ctx.cfg["monitor_interval"] == 0 or number in (ctx.cfg["selection_updates"], end)


def final_update(ctx):
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--phase", choices=("native", "formal", "train"), default="formal")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    if args.testing and args.phase != "native":
        raise ValueError("GPUs 4–6 are reserved for disposable native tests")
    ddp_owner.run(sys.modules[__name__], args, ffn_study.load)


if __name__ == "__main__":
    main()
