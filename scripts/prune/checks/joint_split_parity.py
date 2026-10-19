"""Verify compact head/FFN bundles while preserving split adapter arithmetic.

Historical studies supply immutable saved inputs only. This diagnostic records
current code and every source hash. It never resumes old training. Each bundle
gets independent masked and reloaded compact forwards on exact matched inputs.
See doc/checks/joint_split_parity.md.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, provenance, session
from scripts.prune.data.learned_head_study import code_identity
from scripts.prune.learn import adapters, checkpoints, ffn_training, heads, joint_training
from scripts.prune.learn.ffn import input_case
from scripts.prune.score import export_joint_split, export_pruned, hooks


def run(study_path, output, gpu=5, reuse=None, selected_kinds=None):
    heads.numerical_policy()
    torch.set_num_threads(4)
    if gpu not in (4, 5, 6):
        raise ValueError("Native tests use GPUs 4–6")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    study = json.loads(Path(study_path).read_text())
    identity = study.pop("identity")
    if identity != checkpoints.digest(study):
        raise ValueError("Historical study changed")
    study["identity"] = identity
    current = session.open_session(
        argparse.Namespace(model="2.5", gpu_id=gpu, seed=42),
        script="prune.checks.joint_split_parity",
        prompt="",
        transformer_path=Path(study["components"]["transformer"]["path"]),
    )
    source = study["components"]["transformer"]["path"]
    widths = export_pruned.checkpoint_mask_widths(source)
    full = {k: [1] * v for k, v in widths.items()}
    r3_path = Path(study["parent"]["input_root"]) / "training/checkpoints/update_0050.pt"
    pilot_path = Path(study["root"]) / "training/checkpoints/update_0050.pt"
    r3 = ffn_training.read(r3_path)["adapters"]
    pilot = joint_training.read(pilot_path)
    result = dict(
        format="joint_split_parity_v1",
        source_study=identity,
        code=code_identity(),
        tolerance=0.02,
        source_sha256=provenance.file_sha256(source),
        r3_checkpoint_sha256=provenance.file_sha256(r3_path),
        pilot_checkpoint_sha256=provenance.file_sha256(pilot_path),
        cases=[],
        merged_export_qualified=False,
        qualification_scope="compact attention and FFN up; original FFN down reduction; split adapters",
    )
    # Keep saved test actors out of this diagnostic. These are development inputs.
    monitor_sources = [
        next(s for s in study["splits"]["validation"] if s["actor"] == a)
        for a in study["splits"]["validation_actors"][:2]
    ]
    specs = [(monitor_sources[0], i) for i in study["config"]["monitor_sigma_indices"]] + [
        (monitor_sources[1], study["config"]["monitor_sigma_indices"][0])
    ]
    cached = []
    for src, si in specs:
        case = input_case(study, current, src, 2042, study["config"]["sigmas"][si])
        cached.append(case)
    if reuse:
        for kind in ("zero", "head"):
            for index in range(4):
                path = Path(reuse) / f"{kind}_case{index}.json"
                row = json.loads(path.read_text())
                if not row["passed"] or not row["bit_identical"]:
                    raise ValueError("Cannot reuse a failed replay observation")
                expected_hash = provenance.file_sha256(cached[index][-2] / "manifest.json")
                if row["input_sha256"] != expected_hash:
                    raise ValueError("Reused replay inputs differ")
                # The new branch affects only FFN down reductions. Zero/head
                # conditions retain every FFN channel, so that branch is a no-op.
                row["reused_from"] = dict(
                    path=str(path.resolve()),
                    sha256=provenance.file_sha256(path),
                    reason="FFN masks are all one; new FFN branch is a no-op",
                )
                result["cases"].append(row)
        kinds = ("ffn", "joint", "pilot_aligned")
    else:
        kinds = ("zero", "head", "ffn", "joint", "pilot_aligned")
    if selected_kinds is not None:
        kinds = tuple(selected_kinds)
    result["expected_cases"] = len(result["cases"]) + 4 * len(kinds)
    result["conditions"] = list(kinds)
    for kind in kinds:
        masks = {k: list(v) for k, v in full.items()}
        state = r3
        for name, mask in masks.items():
            b, family = name.split(".")
            if family == "attn1" and kind in ("head", "joint"):
                # Different block widths and noncontiguous IDs expose index bugs.
                count = 1 + int(b) % 5
                ids = torch.randperm(len(mask), generator=torch.Generator().manual_seed(4200 + int(b)))[:count]
                for index in ids:
                    mask[int(index)] = 0
            if family == "ff" and kind in ("ffn", "joint"):
                count = 128 * (1 + int(b) % 3)
                ids = torch.randperm(len(mask), generator=torch.Generator().manual_seed(8400 + int(b)))[:count]
                for index in ids:
                    mask[int(index)] = 0
            if kind == "pilot_aligned" and family in ("attn1", "ff"):
                value = pilot["heads_mask" if family == "attn1" else "ffn_mask"][int(b)].clone()
                if family == "ff":
                    need = (-int(value.sum())) % 128
                    value[(value == 0).nonzero().flatten()[:need]] = 1
                masks[name] = value.int().tolist()
                state = pilot["adapters"]
        envelope = output / f"{kind}.json"
        if not envelope.exists():
            export_joint_split.bundle(
                source, masks, state, envelope, provenance=dict(probe=True, kind=kind, source_study=identity)
            )
        paths = []
        with current.transformer() as model, torch.no_grad():
            attached = adapters.attach(model, 42, 128)
            attached.update(adapters.attach_ffn(model, 42, 16))
            adapters.load_state(attached, state)
            with (
                hooks.attach_head_masks(model, {k: torch.tensor(v) for k, v in masks.items()}, requires_grad=False),
                hooks.attach_ffn_masks(model, {k: torch.tensor(v) for k, v in masks.items()}, requires_grad=False),
            ):
                for index, case in enumerate(cached):
                    _, clean = ltx_adapter.velocity_forward(model, case[1])
                    clean = common.with_clean_prefix(clean, case[2]).cpu()
                    path = output / f"{kind}_case{index}_functional.pt"
                    checkpoints.publish_tensor(path, clean)
                    paths.append(path)
            del attached, model, clean
        gc.collect()
        torch.cuda.empty_cache()
        with export_joint_split.open_bundle(current, envelope) as model, torch.no_grad():
            for index, case in enumerate(cached):
                _, clean = ltx_adapter.velocity_forward(model, case[1])
                clean = common.with_clean_prefix(clean, case[2]).cpu()
                expected = torch.load(paths[index], map_location="cpu", weights_only=True)
                difference = (expected.float() - clean.float()).abs()
                maximum = float(difference.max())
                path = output / f"{kind}_case{index}_compact.pt"
                checkpoints.publish_tensor(path, clean)
                row = dict(
                    kind=kind,
                    case=index,
                    source=specs[index][0]["relative_dir"],
                    sigma=study["config"]["sigmas"][specs[index][1]],
                    input_sha256=provenance.file_sha256(case[-2] / "manifest.json"),
                    max_abs=maximum,
                    passed=bool(torch.isfinite(difference).all() and maximum <= 0.02),
                    bit_identical=torch.equal(expected, clean),
                    envelope=str(envelope),
                    envelope_sha256=provenance.file_sha256(envelope),
                )
                result["cases"].append(row)
                checkpoints.publish_json(output / f"{kind}_case{index}.json", row)
                print(json.dumps(row), flush=True)
            del model, clean
        gc.collect()
        torch.cuda.empty_cache()
    result["passed"] = len(result["cases"]) == result["expected_cases"] and all(r["passed"] for r in result["cases"])
    checkpoints.publish_json(output / "acceptance.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-id", type=int, default=5)
    p.add_argument("--reuse-zero-head", type=Path)
    p.add_argument("--kinds", nargs="+", choices=("zero", "head", "ffn", "joint", "pilot_aligned"))
    a = p.parse_args()
    run(a.study, a.output, a.gpu_id, a.reuse_zero_head, a.kinds)


if __name__ == "__main__":
    main()
