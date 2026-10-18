"""Freeze a new FFN study from verified existing corpus and draw evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity, draw_list
from scripts.prune.learn.checkpoints import digest, publish, publish_json
from scripts.prune.learn.ffn_gates import aligned_width


def validate_config(config):
    if config.get("format") != "learned_ffn_config_v1" or config.get("arm") not in ("ffn_only", "joint_heads_ffn"):
        raise ValueError("unknown FFN config or arm")
    for key in ("gate_lr", "adapter_lr", "wall_time_hours", "headroom_bytes"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"invalid FFN setting: {key}")
    if config["seed"] != 42:
        raise ValueError("the first FFN study uses the verified seed-42 draw stream")
    for key in (
        "head_rank",
        "ffn_rank",
        "selection_updates",
        "recovery_updates",
        "checkpoint_interval",
        "monitor_interval",
        "world_size",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"invalid FFN config: {key}")
    if (
        not 0 < config["head_soft_updates"] < config["selection_updates"]
        or config["ffn_ramp_end"] > config["selection_updates"]
    ):
        raise ValueError("FFN/head selection boundaries differ")
    aligned_width(
        0,
        config["ffn_width"],
        config["ffn_kept"],
        alignment=config["ffn_alignment"],
        warmup=config["ffn_warmup"],
        ramp_end=config["ffn_ramp_end"],
    )
    if config["world_size"] != 4 or config["gpu_ids"] != [0, 1, 2, 3] or config["test_gpu_ids"] != [4, 5, 6]:
        raise ValueError("FFN GPU assignment differs from the authorized launch")


def verify_files(items):
    seen = {}
    for item in items:
        path = item["path"]
        if path not in seen:
            seen[path] = file_sha256(path)
        if seen[path] != item.get("sha256", item.get("file_sha256")):
            raise ValueError(f"frozen FFN input changed: {path}")


def prepare(source, config, plan, output):
    source, config, plan, output = map(lambda p: Path(p).resolve(), (source, config, plan, output))
    parent = json.loads(source.read_text())
    parent_id = parent.pop("identity")
    if parent_id != digest(parent):
        raise ValueError("parent study identity changed")
    overrides = json.loads(config.read_text())
    validate_config(overrides)
    verify_files([*parent["files"], *parent["components"].values(), parent["plan"]])
    effective = copy.deepcopy(parent["config"])
    effective.pop("progressive_policy", None)
    effective.update(overrides)
    effective.update(
        uniform_f18=True,
        rank=overrides["head_rank"],
        global_unique_clips=4,
        person_coefficient=0.0,
        small_region_coefficient=0.0,
    )
    total = effective["selection_updates"] + effective["recovery_updates"]
    draws = draw_list(
        parent["splits"]["learning"],
        effective["seed"],
        updates=total,
        sigma_indices=effective["sigma_indices_cycle"],
        draws_per_update=4,
    )
    original_draws = parent["draws"][f"{effective['seed']}:ordinary"]
    verify_files([original_draws])
    old = json.loads(Path(original_draws["path"]).read_text())
    prefix = min(len(old), len(draws))
    if old[:prefix] != draws[:prefix]:
        raise ValueError("new FFN draws do not reproduce the saved head draw prefix")
    frozen_plan = output / "provenance/plan.md"
    publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    draw_path = output / "draws/replicate_42_ordinary.json"
    publish_json(draw_path, draws)
    publish_json(output / "configs/config.json", effective)
    study = {
        k: copy.deepcopy(parent[k])
        for k in ("components", "splits", "files", "software", "text_context", "schedule", "scale_factors")
    }
    # Parent draw/config files are historical inputs. The new draw file is the only consumed draw authority.
    study.update(
        format="ffn_mask_study_v1",
        config=effective,
        root=str(output),
        code=code_identity(),
        plan=dict(path=str(frozen_plan), sha256=file_sha256(frozen_plan)),
        parent=dict(
            path=str(source),
            sha256=file_sha256(source),
            identity=parent_id,
            input_root=parent["root"],
            matched_draw_prefix=prefix,
        ),
        draws=dict(path=str(draw_path), sha256=file_sha256(draw_path)),
    )
    study["files"].append(study["draws"])
    study["identity"] = digest(study)
    publish_json(output / "configs/frozen_study.json", study)
    return study


def load(path, *, verify_inputs=True):
    study = json.loads(Path(path).read_text())
    identity = study.pop("identity")
    if study.get("format") != "ffn_mask_study_v1" or identity != digest(study):
        raise ValueError("FFN study identity changed")
    study["identity"] = identity
    validate_config(study["config"])
    if code_identity() != study["code"]:
        raise ValueError("FFN source changed; freeze a new study")
    verify_files([study["plan"], study["draws"], study["parent"]])
    if verify_inputs:
        verify_files([*study["files"], *study["components"].values()])
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("scripts/prune/configs/learned_ffn_dev10_ddp_joint.json"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = prepare(args.source_study, args.config, args.plan, args.output)
    print(json.dumps(dict(study=study["identity"], root=study["root"])))


if __name__ == "__main__":
    main()
