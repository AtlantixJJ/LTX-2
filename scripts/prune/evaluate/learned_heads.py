"""Evaluate merged head candidates and their independent native checks.

Use pinned saved teacher/GT evidence and exact full-source inputs for paired
latent metrics. Aggregate by actor for bootstrap uncertainty, preserving view
and sigma dependence. native_checks separates structural transfer, adapter
transfer, fusion and merged reload. Frozen evaluation never selects masks.

CLI: python -m scripts.prune.evaluate.learned_heads --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from scripts.onestep_avatar.model import common
from scripts.prune.checks.learned_heads import compare, validate_envelope
from scripts.prune.core import ltx_adapter, numerics, provenance, records
from scripts.prune.data.learned_head_study import load_study, study_sigmas
from scripts.prune.learn import adapters, checkpoints, losses, protocol
from scripts.prune.learn.heads import input_case, open_current, reference_identity
from scripts.prune.learn.teacher_references import TeacherReferences
from scripts.prune.score import hooks


def paired_metrics(velocity, student, teacher_velocity, teacher, capture, epsilon, *, sigma, prefix_tokens):
    values = (velocity, student, teacher_velocity, teacher, capture, epsilon)
    if any(v.shape != student.shape or not torch.isfinite(v).all() for v in values) or not 0 < sigma <= 0.95:
        raise ValueError("paired metrics require matched finite native tokens and exact sigma")
    if not torch.equal(student[:, :prefix_tokens], capture[:, :prefix_tokens]):
        raise ValueError("student clean frame differs from supplied capture")
    s, t, y = [v[:, prefix_tokens:].float() for v in (student, teacher, capture)]
    mse_s, mse_t = float((s - y).square().mean()), float((t - y).square().mean())
    result = {
        "teacher_latent_mse": float((s - t).square().mean()),
        "capture_mse": mse_s,
        "teacher_capture_mse": mse_t,
        "capture_mse_increase": (mse_s - mse_t) / max(mse_t, 1e-8),
        "official_velocity_mse": float(losses.velocity_mse(velocity, epsilon, capture, prefix_tokens)["official"]),
        "teacher_official_velocity_mse": float(
            losses.velocity_mse(teacher_velocity, epsilon, capture, prefix_tokens)["official"]
        ),
        "native_velocity_relative_l2": float(
            (velocity[:, prefix_tokens:].float() - teacher_velocity[:, prefix_tokens:].float()).norm()
            / teacher_velocity[:, prefix_tokens:].float().norm().clamp_min(1e-8)
        ),
        "clean_frame_equal": True,
    }
    return result


def capture_metrics(velocity, student, capture, epsilon, *, prefix_tokens):
    if (
        student.shape != capture.shape
        or not torch.isfinite(student).all()
        or not torch.equal(student[:, :prefix_tokens], capture[:, :prefix_tokens])
    ):
        raise ValueError("GT metrics require finite matched capture and exact clean frame")
    return {
        "capture_mse": float((student[:, prefix_tokens:].float() - capture[:, prefix_tokens:].float()).square().mean()),
        "official_velocity_mse": float(losses.velocity_mse(velocity, epsilon, capture, prefix_tokens)["official"]),
        "clean_frame_equal": True,
        "reference": "gt_capture",
    }


def actor_bootstrap(rows, metric, *, draws=2000, seed=42):
    actors = sorted({row["actor"] for row in rows})
    if len(actors) < 2:
        raise ValueError("uncertainty requires at least two independent actors")
    means = np.array([np.mean([r[metric] for r in rows if r["actor"] == actor]) for actor in actors])
    if not np.isfinite(means).all():
        raise ValueError("nonfinite actor metric")
    rng = np.random.default_rng(seed)
    estimates = means[rng.integers(0, len(actors), (draws, len(actors)))].mean(1)
    return {
        "actor_count": len(actors),
        "case_count": len(rows),
        "actor_mean": float(means.mean()),
        "one_sided_95_upper": float(np.quantile(estimates, 0.95)),
        "draws": draws,
        "seed": seed,
    }


def saved_case(path, identity):
    record = json.loads(path.read_text())
    if record["identity"] != identity:
        raise ValueError("completed evaluation identity changed")
    values = {}
    for role, item in record["tensors"].items():
        p = Path(item["path"])
        if provenance.file_sha256(p) != item["sha256"]:
            raise ValueError("completed evaluation tensor changed")
        value = torch.load(p, map_location="cpu", weights_only=True)
        if records.tensor_hash(value) != item["tensor_sha256"] or not torch.isfinite(value).all():
            raise ValueError("evaluation tensor content differs")
        values[role] = value
    return record, values


def evaluate(study, arm, seed, split, gpu_id):
    root = Path(study["root"])
    if split == "test":
        checkpoints.validate_finalists(study, arm=arm, seed=seed)
    envelope_path = root / f"masks/{arm}/replicate_{seed}/recovered.json"
    candidate = root / f"exports/{arm}/replicate_{seed}/final_merged.safetensors"
    validate_envelope(envelope_path, candidate=candidate)
    identity = {
        "study": study["identity"],
        "candidate_sha256": provenance.file_sha256(candidate),
        "envelope_sha256": provenance.file_sha256(envelope_path),
        "split": split,
        "arm": arm,
        "replicate_seed": seed,
    }
    numerics.deterministic_policy()
    current = open_current(study, gpu_id)
    gt_only = study["config"].get("monitor_reference") == "gt_capture"
    cache = None if gt_only else TeacherReferences(root / "teacher/references")
    rows = []
    expected_paths = [
        root
        / f"evaluation/{split}/{arm}/replicate_{seed}/records/{hashlib.sha256(s['relative_dir'].encode()).hexdigest()}/seed_{noise_seed}/sigma_{i}.json"
        for s in study["splits"][split]
        for noise_seed in study["config"]["evaluation_seeds"]
        for i in range(len(study_sigmas(study)))
    ]
    resident = nullcontext(None) if all(p.exists() for p in expected_paths) else current.transformer(candidate)
    with resident as model:
        for source in study["splits"][split]:
            if source["actor"] in study["splits"]["learning_actors"]:
                raise ValueError("evaluation actor was used for learning")
            key = hashlib.sha256(source["relative_dir"].encode()).hexdigest()
            for noise_seed in study["config"]["evaluation_seeds"]:
                for index, sigma in enumerate(study_sigmas(study)):
                    out = (
                        root
                        / f"evaluation/{split}/{arm}/replicate_{seed}/records/{key}/seed_{noise_seed}/sigma_{index}.json"
                    )
                    case_id = {
                        **identity,
                        "view": source["relative_dir"],
                        "actor": source["actor"],
                        "noise_seed": noise_seed,
                        "sigma": sigma,
                    }
                    grid, video, c0, capture, eps, inputs, manifest = input_case(
                        study, current, source, noise_seed, sigma
                    )
                    ref_id = (
                        None
                        if gt_only
                        else reference_identity(study, current, grid, video, eps, inputs, manifest, sigma)
                    )
                    ref = None if gt_only else cache.read(ref_id)
                    if out.exists():
                        record, values = saved_case(out, case_id)
                        velocity, clean = values["velocity"].to(current.device), values["clean"].to(current.device)
                    else:
                        velocity, clean = ltx_adapter.velocity_forward(model, video)
                        clean = common.with_clean_prefix(clean, c0)
                    if gt_only:
                        metrics = capture_metrics(
                            velocity, clean, capture, eps, prefix_tokens=grid.tokens_per_latent_frame
                        )
                        numeric_pass = None
                    else:
                        teacher_v, teacher_z = ref["velocity"].to(current.device), ref["clean"].to(current.device)
                        metrics = paired_metrics(
                            velocity,
                            clean,
                            teacher_v,
                            teacher_z,
                            capture,
                            eps,
                            sigma=sigma,
                            prefix_tokens=grid.tokens_per_latent_frame,
                        )
                        start = grid.tokens_per_latent_frame
                        dt = (video.latent[:, start:].float() - teacher_z[:, start:].float()) / sigma
                        ds = (video.latent[:, start:].float() - clean[:, start:].float()) / sigma
                        metrics["direction_relative_l2"] = float((ds - dt).norm() / dt.norm().clamp_min(1e-8))
                        sl, tl = [grid.unpatchify_block(v, grid.latent_frames).float() for v in (clean, teacher_z)]
                        metrics["temporal_disagreement"] = float(
                            ((sl[:, :, 2:] - sl[:, :, 1:-1]) - (tl[:, :, 2:] - tl[:, :, 1:-1])).square().mean()
                        )
                        numeric_pass = (
                            metrics["direction_relative_l2"] <= 0.05 and metrics["capture_mse_increase"] <= 0.05
                        )
                    if out.exists():
                        if record["metrics"] != metrics:
                            raise ValueError("saved paired metrics differ from revalidated tensors")
                    else:
                        tensors = {}
                        for role, value in (
                            ("velocity", velocity),
                            ("clean", clean),
                            ("latent", grid.unpatchify_block(clean, grid.latent_frames)),
                        ):
                            p = out.parent / f"sigma_{index}_{role}.pt"
                            records.publish_tensor(p, value)
                            tensors[role] = {
                                "path": str(p),
                                "sha256": provenance.file_sha256(p),
                                "tensor_sha256": records.tensor_hash(value),
                            }
                        record = {
                            "identity": case_id,
                            "teacher_key": None if gt_only else records.digest(ref_id),
                            "input_manifest_sha256": provenance.file_sha256(inputs / "manifest.json"),
                            "reference": "gt_capture" if gt_only else "original_teacher",
                            "tensors": tensors,
                            "metrics": metrics,
                            "schedule": [sigma, 0.0],
                            "numeric_pass": numeric_pass,
                            "quality_pass": None,
                            "missing": ["decoded region LPIPS", "GT region coverage", "video review"],
                        }
                        records.publish_json(out, record)
                    rows.append({"actor": source["actor"], **metrics, "record": str(out)})
    summary = {
        "identity": identity,
        "status": "complete_numeric_only",
        "case_count": len(rows),
        "rows": rows,
        "quality_pass": None,
        "uncertainty": {
            k: actor_bootstrap(rows, k)
            for k in (
                ("capture_mse", "official_velocity_mse")
                if gt_only
                else ("direction_relative_l2", "capture_mse_increase")
            )
        },
        "limitation": "native agreement and capture errors do not establish human quality",
    }
    records.publish_json(root / f"evaluation/{split}/{arm}/replicate_{seed}/metrics.json", summary)
    return summary


def native_checks(study, arm, seed, gpu_id):
    numerics.deterministic_policy()
    current = open_current(study, gpu_id)
    root = Path(study["root"])
    directory = root / f"training/{arm}/replicate_{seed}/checkpoints"
    _, selection_end, _ = protocol.phase_budget(study["config"])
    final_step = protocol.recovery_end(study["config"], arm)
    selection, final = [checkpoints.load(directory / f"update_{i:04d}.pt") for i in (selection_end, final_step)]
    selection_source_state = (
        adapters.fresh_source_state(seed, rank=study["config"]["rank"])
        if arm.endswith("_fresh")
        else selection["adapters"]
    )
    mask = selection["mask"].cpu()
    cases = [next(s for s in study["splits"]["validation"] if s["n_latent_frames"] == length) for length in (18, 28)]
    predictions = {}
    paths = [
        ("masked", Path(study["components"]["transformer"]["path"])),
        ("compact", root / f"exports/{arm}/replicate_{seed}/compact_base.safetensors"),
        ("reloaded", root / f"exports/{arm}/replicate_{seed}/final_merged.safetensors"),
    ]
    for role, path in paths:
        with current.transformer(path) as model:
            attached = None
            masked = (
                hooks.attach_head_masks(model, {f"{b}.attn1": row for b, row in enumerate(mask)}, requires_grad=False)
                if role == "masked"
                else hooks.MaskAttachments()
            )
            with masked:
                states = ("zero", "transferred") if role in ("masked", "compact") else ("merged",)
                for state_name in states:
                    if state_name == "transferred":
                        attached = adapters.attach(model, seed)
                        adapters.load_state(
                            attached,
                            selection_source_state
                            if role == "masked"
                            else adapters.transfer(selection_source_state, mask),
                        )
                    for source in cases:
                        for i, sigma in enumerate(study_sigmas(study)):
                            grid, video, c0, _, _, _, _ = input_case(study, current, source, 2042, sigma)
                            _, clean = ltx_adapter.velocity_forward(model, video)
                            predictions[(role, state_name, source["n_latent_frames"], i)] = common.with_clean_prefix(
                                clean, c0
                            ).cpu()
                if role == "compact":
                    adapters.load_state(attached, final["adapters"])
                    for fusion_role in ("unmerged", "in_memory_merged"):
                        if fusion_role == "in_memory_merged":
                            for module in attached.values():
                                module.base.weight.data.copy_(module.merged_weight())
                                module.enabled = False
                        for source in cases:
                            for i, sigma in enumerate(study_sigmas(study)):
                                _, video, c0, _, _, _, _ = input_case(study, current, source, 2042, sigma)
                                _, clean = ltx_adapter.velocity_forward(model, video)
                                predictions[("compact", fusion_role, source["n_latent_frames"], i)] = (
                                    common.with_clean_prefix(clean, c0).cpu()
                                )
    for name, reference, candidate in (
        ("structural", ("masked", "zero"), ("compact", "zero")),
        ("transfer", ("masked", "transferred"), ("compact", "transferred")),
        ("fusion", ("compact", "unmerged"), ("compact", "in_memory_merged")),
        ("reload", ("compact", "in_memory_merged"), ("reloaded", "merged")),
    ):
        results = []
        for length in (18, 28):
            for i, sigma in enumerate(study_sigmas(study)):
                results.append(
                    {
                        "frames": length,
                        "sigma": sigma,
                        **compare(
                            predictions[(*reference, length, i)],
                            predictions[(*candidate, length, i)],
                            prefix_tokens=1024,
                        ),
                    }
                )
        records.publish_json(
            root / f"checks/{arm}/replicate_{seed}/{name}.json",
            {"study": study["identity"], "results": results, "passed": all(r["passed"] for r in results)},
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--replicate-seed", type=int, default=42)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--gpu-id", type=int, required=True)
    args = p.parse_args()
    evaluate(load_study(args.study), args.arm, args.replicate_seed, args.split, args.gpu_id)


if __name__ == "__main__":
    main()
