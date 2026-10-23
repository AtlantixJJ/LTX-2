"""Run native acceptance and fixed compact adapter recovery with the DDP owner.

CompactSession preserves native input-component identities but loads the frozen
compact transformer. Family hooks implement fixed-width checks, checkpoint and
monitor boundaries; learn.compact_training owns numerical state. The shared
update order is defined in doc/learn/ddp_owner.md.

CLI: python -m scripts.prune.learn.compact --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.prune.core import numerics, records
from scripts.prune.data import compact_study
from scripts.prune.learn import compact_training, ddp_owner

state = compact_training
label = "Compact A"
script = "prune.learn.compact"


class CompactSession:
    """Preserve the native input contract while loading the actual compact base."""
    def __init__(self, original, artifact):
        self.original, self.artifact = original, Path(artifact)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def transformer(self, transformer_path=None, **kwargs):
        return self.original.transformer(transformer_path or self.artifact, **kwargs)


def native_fixture(cfg):
    return cfg


def native_prepare(step, cfg):
    pass


def native_first_checks(step, first, cfg, runtime):
    if first["adapter_gradient_norm"] <= 0:
        raise ValueError("Compact adapter gradient must be nonzero")
    digests = runtime.gather(records.unchanged_state_hash(state.adapters.state_dict(step.attached)))
    if len(set(digests)) != 1:
        raise ValueError("Compact ranks applied different adapter updates")
    return dict(compact_forward=True, averaged_adapter_updates=digests[0])


def native_freeze(step, go):
    return lambda: torch.ones(1)


def check_start(study, root, cfg, *, pilot, max_updates):
    if pilot:
        raise ValueError("Compact testing exercises native acceptance only")


def after_update(ctx, number):
    pass


def receipt_fields(ctx, number):
    alignment = records.read_json(Path(ctx.study["alignment"]["path"]))
    return dict(phase="compact_recovery", ffn=dict(retained_per_block=alignment["compact_ffn"]),
                heads=dict(retained_per_block=alignment["compact_heads"]), pilot=False,
                execution="actual_compact_both_ffn_projections")


def reviews(ctx, number):
    return False


def checkpoint_due(ctx, number, end):
    return number == end


def after_checkpoint(ctx, number):
    pass


def monitor_due(ctx, number, end):
    return number % ctx.cfg["monitor_interval"] == 0 or number == end


def final_update(ctx):
    return ctx.cfg["recovery_updates"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--phase", choices=("native", "formal", "train"), default="formal")
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()
    if args.testing and args.phase != "native":
        raise ValueError("GPUs 4/5 support disposable native tests only")
    numerics.deterministic_policy()
    torch.set_num_threads(4)
    study = compact_study.load(args.study, verify_inputs=int(os.environ.get("RANK", "0")) == 0)
    runtime = ddp_owner.Runtime(study["config"], testing=args.testing)
    current = CompactSession(ddp_owner.open_current(study, runtime, script=script), study["compact"]["path"])
    try:
        family = sys.modules[__name__]
        if args.phase in ("native", "formal"):
            ddp_owner.native(study, current, runtime, family, testing=args.testing)
        if args.phase in ("train", "formal"):
            ddp_owner.train(study, current, runtime, family)
    except BaseException as error:
        if runtime.primary:
            records.write_json(Path(study["root"]) / "monitoring/failure.json",
                               dict(error=repr(error), phase=args.phase, pid=os.getpid()))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
