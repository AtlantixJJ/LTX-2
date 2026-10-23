"""Replay a saved head-only endpoint before new fixed-checkpoint evaluation.

Verify the original checkpoint, post-reset binary gates and saved monitor
predictions under the standard replay tolerance. Only a successful replay
permits held-out scoring. The recorded study supplies input provenance; this
reader constructs no optimizer and never resumes its training owner.

CLI: python -m scripts.prune.evaluate.historical_clip --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, provenance, records
from scripts.prune.data.study_io import code_identity
from scripts.prune.evaluate import ffn, joint_final
from scripts.prune.learn import adapters
from scripts.prune.learn.ddp_owner import input_case
from scripts.prune.score import hooks


def read_checkpoint(path, old):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if provenance.file_sha256(path) != receipt["sha256"]:
        raise ValueError("historical checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["format"] != "head_learning_state_v1" or state["identity"]["study"] != old["identity"]:
        raise ValueError("historical checkpoint identity differs")
    if state["update"] != 600 or state["draw_cursor"] != 2400 or receipt["identity"] != state["identity"]:
        raise ValueError("historical endpoint or draw cursor differs")
    if (
        records.tensor_hash(state["alpha"]) != receipt["alpha_sha256"]
        or {k: records.tensor_hash(v) for k, v in state["adapters"].items()} != receipt["tensor_hashes"]
    ):
        raise ValueError("historical saved tensors changed")
    policy = state["progressive"]
    if records.digest(policy) != receipt["progressive_sha256"] or policy["event_stage"] != "post_reset":
        raise ValueError("historical policy differs")
    if policy["finalization_start"] != 600 or not torch.equal(state["alpha"], torch.tensor(policy["free"]).float()):
        raise ValueError("historical endpoint is not the post-reset binary mask")
    return state


def monitor_tokens(prediction, source_shape):
    channels, frames, height, width = source_shape
    if tuple(prediction.shape) == (1, channels, frames, height, width):
        prediction = prediction.flatten(2).transpose(1, 2).contiguous()
    if tuple(prediction.shape) != (1, frames * height * width, channels) or not torch.isfinite(prediction).all():
        raise ValueError("historical prediction shape or finite values differ")
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "historical-study", "checkpoint", "monitor", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid historical evaluation shard")
    study = joint_final.load(args.study)
    old = json.loads(args.historical_study.read_text())
    for name in ("splits", "components", "text_context", "schedule"):
        if old[name] != study[name]:
            raise ValueError("historical comparison inputs differ: " + name)
    state = read_checkpoint(args.checkpoint, old)
    current = ffn.current_for(study, args.gpu_id)
    identity = dict(
        study=study["identity"],
        historical_study=old["identity"],
        arm="historical_clip_600",
        checkpoint_sha256=provenance.file_sha256(args.checkpoint),
        code=code_identity()["sha256"],
    )
    records.publish_json(args.output / "identity.json", identity)
    monitor = json.loads(args.monitor.read_text())
    parity = []
    with current.transformer() as model, torch.no_grad():
        model.eval()
        attached = adapters.attach(model, old["config"]["replicate_seeds"][0], old["config"]["rank"])
        adapters.load_state(attached, state["adapters"])
        handle = hooks.attach_head_masks(
            model, {f"{i}.attn1": value for i, value in enumerate(state["alpha"])}, requires_grad=False
        )
        for row in monitor["rows"]:
            saved = row["states"]["trained"]
            if provenance.file_sha256(saved["path"]) != saved["sha256"]:
                raise ValueError("historical monitor prediction changed")
            # Replay the historical manifest itself, including its original provenance.
            case = input_case(dict(study, root=old["root"]), current, row["source"], 2042, row["sigma"])
            if provenance.file_sha256(case[-2] / "manifest.json") != row["input_manifest_sha256"]:
                raise ValueError("historical monitor inputs differ")
            _, clean = ltx_adapter.velocity_forward(model, case[1])
            clean = common.with_clean_prefix(clean, case[2]).cpu()
            expected = torch.load(saved["path"], map_location="cpu", weights_only=True)
            expected = monitor_tokens(expected, row["source"]["shape"])
            difference = (clean.float() - expected.float()).abs()
            parity.append(
                dict(
                    actor=row["actor"],
                    sigma=row["sigma"],
                    max_abs=float(difference.max()),
                    passed=bool(torch.isfinite(difference).all() and difference.max() <= 0.02),
                )
            )
        passed = len(parity) == 12 and all(r["passed"] for r in parity)
        records.publish_json(args.output / "monitor_parity.json", dict(passed=passed, cases=parity, tolerance=0.02))
        if not passed:
            raise ValueError("historical monitor replay failed; no held-out scoring")
        native = Path(study["parent"]["input_root"]) / "evaluation/test/native"
        ffn.run_cases(
            study,
            current,
            model,
            args.output / "test",
            identity,
            [case for i, case in enumerate(ffn.cases(study, "test")) if i % args.shard_count == args.shard_index],
            label="Historical head-only update 600",
            input_case=input_case,
            expected_input=lambda dest: json.loads((native / dest.name / "case.json").read_text())["input_sha256"],
        )
        hooks.detach_all(handle)


if __name__ == "__main__":
    main()
