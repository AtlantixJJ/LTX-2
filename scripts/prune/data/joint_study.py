"""Freeze joint clipped-selection studies from verified paired input authorities.

Copy exact schedules, source pins and the extended draw prefix into a new
identity. The selected arm fixes the event policy; the matched-budget arm also
pins a joint reference endpoint and budget receipt. Preparation writes no
training state and never modifies its parent. Device IDs are frozen config,
not machine constants. See doc/learn/readmission.md for policy state.

CLI: python -m scripts.prune.data.joint_study --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from scripts.prune.core.provenance import file_sha256
from scripts.prune.core.records import digest, publish, publish_json
from scripts.prune.data.study_io import code_identity, read_device_config, read_frozen, validate_devices, verify_files

SIGMAS = [0.9489720463752747, 0.7663628458976746, 0.48016369342803955]
ARMS = ("joint_clip_reset_v3", "heads_clip_reset_v3", "budget_joint_v3")


def validate_config(cfg):
    if cfg["format"] != "joint_config_v3" or cfg["arm"] not in ARMS:
        raise ValueError("unknown joint arm or config")
    validate_devices(cfg)
    if [cfg["sigmas"][i] for i in cfg["monitor_sigma_indices"]] != SIGMAS:
        raise ValueError("joint exact monitor schedules differ")
    expected = dict(interval=50, late=10, earliest=200, limit=1500, recovery=200, alignment=128)
    if cfg["policy"] != expected or cfg["rho"] != 1 / 704 or cfg["ffn_width"] != 16384:
        raise ValueError("joint frozen method differs")
    if cfg["arm"] == "budget_joint_v3":
        if not 200 <= cfg["selection_updates"] <= 1500 or not cfg.get("budget_receipt"):
            raise ValueError("budget_joint_v3 requires the joint reference endpoint and budget receipt")
        if not 48 * (704 + 128) <= cfg.get("budget", 0) <= 1536 * 704 + 786432:
            raise ValueError("budget_joint_v3 budget cannot satisfy charged block guards")
    elif cfg["selection_updates"] != 1500:
        raise ValueError("joint selection bound differs")


def prepare(source, continuation, plan, output, *, arm=ARMS[0], budget_receipt=None, prerequisites=None, device_config=None):
    source, continuation, plan, output = [Path(p).resolve() for p in (source, continuation, plan, output)]
    parent = json.loads(source.read_text())
    identity = parent.pop("identity")
    if identity != digest(parent):
        raise ValueError("joint parent identity changed")
    extended = json.loads(continuation.read_text())
    continuation_identity = extended.pop("identity")
    if continuation_identity != digest(extended):
        raise ValueError("draw continuation identity changed")
    cfg = copy.deepcopy(parent["config"])
    if device_config is not None:
        cfg.update(read_device_config(device_config))
    cfg.update(
        format="joint_config_v3",
        arm=arm,
        selection_updates=1500,
        recovery_updates=200,
        monitor_interval=50,
        rho=1 / 704,
        policy=dict(interval=50, late=10, earliest=200, limit=1500, recovery=200, alignment=128),
    )
    if budget_receipt:
        path = Path(budget_receipt).resolve()
        budget = json.loads(path.read_text())
        cfg.update(
            budget=budget["budget"],
            selection_updates=budget["selection_end"],
            budget_receipt=dict(path=str(path), sha256=file_sha256(path)),
        )
    validate_config(cfg)
    verify_files([*parent["files"], *parent["components"].values(), parent["plan"]])
    old_record = parent["draws"]
    new_record = extended["draws"]["42:ordinary"]
    verify_files([old_record, new_record])
    old = json.loads(Path(old_record["path"]).read_text())
    draws = json.loads(Path(new_record["path"]).read_text())[:6800]
    if len(draws) != 6800 or draws[: len(old)] != old:
        raise ValueError("joint draw extension does not reproduce all r3 draws")
    if cfg["sigmas"] != extended["config"]["sigmas"]:
        raise ValueError("continuation exact training schedules differ")
    frozen_plan = output / "provenance/plan.md"
    publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    draw_path = output / "draws/replicate_42_ordinary.json"
    publish_json(draw_path, draws)
    study = {
        k: copy.deepcopy(parent[k])
        for k in (
            "components",
            "splits",
            "files",
            "software",
            "text_context",
            "schedule",
            "scale_factors",
            "perceptual_metric",
        )
    }
    study.update(
        format="joint_study_v3",
        config=cfg,
        root=str(output),
        code=code_identity(),
        plan=dict(path=str(frozen_plan), sha256=file_sha256(frozen_plan)),
        parent=dict(
            path=str(source),
            sha256=file_sha256(source),
            identity=identity,
            input_root=parent["root"],
            matched_draw_prefix=len(old),
        ),
        continuation=dict(path=str(continuation), sha256=file_sha256(continuation)),
        draws=dict(path=str(draw_path), sha256=file_sha256(draw_path)),
    )
    if prerequisites:
        receipt_path = Path(prerequisites).resolve()
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("passed") or not receipt.get("checks") or not all(receipt["checks"].values()):
            raise ValueError("pre-launch checks have not passed")
        verify_files(receipt.get("source_receipts", []))
        study["prerequisites"] = dict(path=str(receipt_path), sha256=file_sha256(receipt_path))
    study["identity"] = digest(study)
    publish_json(output / "configs/config.json", cfg)
    publish_json(output / "configs/frozen_study.json", study)
    return study


def load(path, *, verify_inputs=True):
    study = read_frozen(path, formats=("joint_study_v3",))
    validate_config(study["config"])
    verify_files([study[k] for k in ("plan", "draws", "parent", "continuation")])
    if verify_inputs:
        verify_files([*study["files"], *study["components"].values()])
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-study", "continuation-study", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--arm", choices=ARMS, default=ARMS[0])
    parser.add_argument("--budget-receipt", type=Path)
    parser.add_argument("--prerequisites", type=Path)
    parser.add_argument("--device-config", type=Path, help="New device groups; the frozen parent remains unchanged")
    args = parser.parse_args()
    study = prepare(
        args.source_study,
        args.continuation_study,
        args.plan,
        args.output,
        arm=args.arm,
        budget_receipt=args.budget_receipt,
        prerequisites=args.prerequisites,
        device_config=args.device_config,
    )
    print(json.dumps(dict(identity=study["identity"], root=study["root"])))


if __name__ == "__main__":
    main()
