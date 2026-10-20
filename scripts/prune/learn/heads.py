"""Bounded native learned-head phases; one resident model per process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import signal
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, numerics, provenance, records, session
from scripts.prune.data import foreground, training_windows, whole_clip
from scripts.prune.data.learned_head_study import load_study, monitor_sigma_indices, study_sigmas
from scripts.prune.evaluate import timing
from scripts.prune.learn import adapters, checkpoints, losses, protocol
from scripts.prune.learn.gates import HeadGates, nearest_removed, select_legal
from scripts.prune.learn.teacher_references import TeacherReferences
from scripts.prune.score import hooks, token_sampling
from scripts.prune.score.estimators import projected_energy

ARMS = (
    "learned_joint",
    "learned_soft_only",
    "energy",
    "random",
    "learned_no_sparsity",
    "learned_person",
    "learned_small_regions",
    "learned_balanced",
    "learned_sequential",
    "learned_fresh",
    "energy_fresh",
    "energy_counts_fresh",
    "dense_fresh",
)


def open_current(study, gpu_id):
    if gpu_id not in study["config"]["gpu_ids"]:
        raise ValueError("GPU outside frozen allowed device list")
    subprocess.run(["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index,memory.free", "--format=csv"], check=True)
    current = session.open_frozen(study, gpu_id, script="prune.learn.heads")
    from ltx_core.model.transformer.attention import attention_label, automatic_attention

    packages = {}
    for name in ("numpy", "safetensors", "transformers", "lpips"):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            directory = Path(spec.origin).parent
            package_files = {
                str(p.relative_to(directory)): provenance.file_sha256(p) for p in sorted(directory.rglob("*.py"))
            }
            packages[name] = {
                "version": importlib.metadata.version(name),
                "code_sha256": records.digest(package_files),
            }
    environment = {
        "software": study["software"],
        "packages": packages,
        "cudnn": torch.backends.cudnn.version(),
        "attention_dispatch": attention_label(automatic_attention()),
        "gpu_architecture": list(torch.cuda.get_device_capability(current.device)),
        "gpu_name": torch.cuda.get_device_name(current.device),
        "numerics": study["config"]["numerics"],
        "driver_version": subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    records.publish_json(Path(study["root"]) / "provenance/environment.json", environment)
    return current


def input_case(study, current, source, seed, sigma):
    key = hashlib.sha256(source["relative_dir"].encode()).hexdigest()
    root = Path(study["root"]) / f"inputs/{key}/seed_{seed}"
    import fcntl

    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f"{root.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return whole_clip.load_case(root, source, current=current, sigmas=study_sigmas(study), seed=seed, sigma=sigma)


def training_input_case(study, current, source, seed, sigma):
    """Apply only the frozen training crop; other consumers keep full sources."""
    case = input_case(study, current, source, seed, sigma)
    return training_windows.crop_case(case, current, seed, sigma) if training_windows.enabled(study["config"]) else case


def reference_identity(study, current, grid, video, epsilon, root, manifest, sigma):
    return {
        "components": study["components"],
        "input": {
            "manifest": provenance.file_sha256(root / "manifest.json"),
            "tokens": records.tensor_hash(video.latent),
        },
        "epsilon": records.tensor_hash(epsilon),
        "context": study["text_context"],
        "geometry": manifest["geometry"],
        "sigma": sigma,
        "code": study["code"]["sha256"],
        "numerics": study["config"]["numerics"],
        "software": {"environment_sha256": provenance.file_sha256(Path(study["root"]) / "provenance/environment.json")},
        "architecture": torch.cuda.get_device_name(current.device),
    }


def teacher_phase(study, current, split, deadline):
    sources = study["splits"][split]
    if split == "test":
        checkpoints.validate_finalists(study)
    cache = TeacherReferences(
        Path(study["root"]) / "teacher/references",
        study["config"]["teacher_disk_cap"],
        study["config"]["teacher_hot_cap"],
    )
    cache.budget([(1, s["n_latent_frames"] * 1024, 128) for s in sources for _ in range(24)])
    saved_records = []
    complete = (Path(study["root"]) / f"teacher/{split}_completion.json").is_file()
    resident = nullcontext(None) if complete else current.transformer(Path(study["components"]["transformer"]["path"]))
    with resident as model:
        for source in sources:
            for seed in study["config"]["evaluation_seeds"]:
                for sigma in study_sigmas(study):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("teacher phase bound reached; completed references remain immutable")
                    grid, video, c0, _, eps, root, manifest = input_case(study, current, source, seed, sigma)
                    identity = reference_identity(study, current, grid, video, eps, root, manifest, sigma)
                    if complete:
                        cache.read(identity)
                        saved_records.append(records.digest(identity))
                        continue

                    def forward(video=video, c0=c0):
                        velocity, clean = ltx_adapter.velocity_forward(model, video)
                        return velocity, common.with_clean_prefix(clean, c0)

                    cache.produce(identity, forward)
                    saved_records.append(records.digest(identity))
    records.publish_json(
        Path(study["root"]) / f"teacher/{split}_completion.json",
        {"study": study["identity"], "split": split, "keys": saved_records, "status": "complete"},
    )


def energy_phase(study, current, deadline):
    root = Path(study["root"])
    widths = {f"{b}.attn1": 32 for b in range(48)}
    saved_records = []
    with current.transformer(Path(study["components"]["transformer"]["path"])) as model:
        for source in study["splits"]["learning"]:
            key = hashlib.sha256(source["relative_dir"].encode()).hexdigest()
            path = root / f"calibration/energy/{key}.json"
            native_root = root / f"inputs/{key}/seed_42"
            manifest = whole_clip.prepare_score_inputs(
                native_root, source, current=current, sigmas=study_sigmas(study), seed=42
            )
            identity = {
                "study": study["identity"],
                "view": source["relative_dir"],
                "source": source["capture_latent_sha256"],
                "input_manifest": provenance.file_sha256(native_root / "manifest.json"),
                "epsilon": manifest["videos"][0]["artifacts"]["epsilon_sha256"],
            }
            if path.exists():
                record = json.loads(path.read_text())
                if record["identity"] != identity:
                    raise ValueError("energy input identity differs")
                saved_records.append(record)
                continue
            observations = []
            for j, sigma in enumerate(study_sigmas(study)):
                partial = path.with_suffix(f".sigma{j}.json")
                if partial.exists():
                    saved = json.loads(partial.read_text())
                    if saved["identity"] != identity or saved["sigma"] != sigma:
                        raise ValueError("energy observation identity changed")
                    observations.append(saved["energies"])
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("energy bound reached")
                grid, video, c0, _, _, _, _ = input_case(study, current, source, 42, sigma)
                indices = token_sampling.sample_indices(video.latent.shape[1], 32, 32, 1, current.device)
                energies = {}

                def observe(name, activation, attention):
                    if name.endswith(".attn1"):
                        energies[name] = (
                            projected_energy(activation, attention.to_out[0].weight, attention.heads, indices)
                            .cpu()
                            .tolist()
                        )

                with hooks.collect_activations(model, "head", observe):
                    output, _ = model(video=video, audio=None, perturbations=None)
                if set(energies) != set(widths) or not torch.isfinite(output).all():
                    raise ValueError("energy coverage or native output failed")
                records.publish_json(partial, {"identity": identity, "sigma": sigma, "energies": energies})
                observations.append(energies)
            record = {"identity": identity, "sigmas": study_sigmas(study), "energies": observations}
            records.publish_json(path, record)
            saved_records.append(record)
    raw = torch.tensor(
        [
            [r["energies"][i][f"{b}.attn1"] for b in range(48)]
            for r in saved_records
            for i in range(len(study_sigmas(study)))
        ],
        dtype=torch.float64,
    ).mean(0)
    if not torch.isfinite(raw).all() or (raw.mean(1) <= 0).any():
        raise ValueError("invalid pooled raw energy")
    records.publish_json(
        root / "calibration/ranking.json",
        {
            "study": study["identity"],
            "scores": (raw / raw.mean(1, keepdim=True)).tolist(),
            "views": [s["relative_dir"] for s in study["splits"]["learning"]],
            "status": "complete",
            "records": {
                str(
                    root / f"calibration/energy/{hashlib.sha256(s['relative_dir'].encode()).hexdigest()}.json"
                ): provenance.file_sha256(
                    root / f"calibration/energy/{hashlib.sha256(s['relative_dir'].encode()).hexdigest()}.json"
                )
                for s in study["splits"]["learning"]
            },
        },
    )


def optimizer(params, lr, config):
    return torch.optim.AdamW(params, lr=lr, betas=tuple(config["betas"]), eps=config["adam_epsilon"], weight_decay=0)


def train_identity(study, arm, seed, removed, draws):
    prefix_config = {
        k: v for k, v in study["config"].items() if k not in ("removed_heads", "test_actors", "gpu_ids", "gates")
    }
    prefix = records.digest(
        {
            "seed": seed,
            "draws": draws["sha256"],
            "config": prefix_config,
            "learning": study["splits"]["learning"],
            "context": study["text_context"]["tensor_sha256"],
            "components": study["components"],
            "code": study["code"],
            "software": study["software"],
            "arm": "learned_joint" if arm == "learned_soft_only" else arm,
        }
    )
    return {
        "study": study["identity"],
        "arm": arm,
        "seed": seed,
        "removed": removed,
        "draws": draws["sha256"],
        "prefix": prefix,
    }


def training_phase(study, current, args, deadline):
    root, config = Path(study["root"]), study["config"]
    soft_end, selection_end, compact_end = protocol.phase_budget(config)
    uniform_f18 = training_windows.enabled(config)
    draw_count = protocol.validate(config)
    if config.get("foreground_preparation") == "deferred" and args.arm in (
        "learned_balanced",
        "learned_person",
        "learned_small_regions",
    ):
        raise ValueError("regional and balanced arms require foreground preparation")
    fresh = args.arm.endswith("_fresh")
    if fresh and args.phase != "compact_recovery":
        raise ValueError("fresh controls run only compact recovery")
    if args.arm == "learned_small_regions" and study["region_tracks"]["status"] != "verified":
        raise ValueError("regional runs require verified canonical coverage and reviewed GT region records")
    allowed = [config["allowed_counts"] for _ in range(config["blocks"])]
    if (
        args.removed_heads not in config["removed_heads"]
        or nearest_removed(args.removed_heads, allowed, heads=config["heads"]) != args.removed_heads
    ):
        raise ValueError("removal budget differs from frozen feasible choices")
    devices = getattr(current, "execution_devices", (current.device,))
    runtime = getattr(current, "distributed", None)
    # Sustained training is gated by native execution evidence, never a schema-only receipt.
    feasibility_path = root / "checks/feasibility.json"
    feasibility = json.loads(feasibility_path.read_text())
    if feasibility.get("study") != study["identity"] or feasibility.get("passed") is not True:
        raise ValueError("sustained learning requires passed native feasibility")
    if runtime and feasibility.get("method") != runtime.feasibility_method:
        raise ValueError("distributed learning requires the matching native execution gate")
    gt_monitor = config.get("monitor_reference") == "gt_capture"
    if not gt_monitor:
        teacher_completion = json.loads((root / "teacher/validation_completion.json").read_text())
        if teacher_completion["study"] != study["identity"]:
            raise ValueError("monitoring requires original-teacher references for this study")
        probe_path = root / "checks/execution_probes.json"
        probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
        if (
            not (probe.get("study") == study["identity"] and probe.get("speed_layout_passed"))
            and not args.quality_research
        ):
            raise ValueError("no reproducible compact speed saving; explicitly request a quality research pilot")
    save_checkpoint = runtime.save if runtime else checkpoints.save
    policy = "balanced" if args.arm == "learned_balanced" else "ordinary"
    draw_record = study["draws"][f"{args.replicate_seed}:{policy}"]
    draws = json.loads(Path(draw_record["path"]).read_text())
    identity = train_identity(
        study,
        args.arm,
        args.replicate_seed,
        0 if args.arm == "dense_fresh" else args.removed_heads,
        draw_record,
    )
    gates = HeadGates(config["blocks"], config["heads"], allowed).to(current.device)
    kept = config["blocks"] * config["heads"] - args.removed_heads
    scores = None
    if args.arm == "energy":
        ranking = json.loads((root / "calibration/ranking.json").read_text())
        expected_views = [s["relative_dir"] for s in study["splits"]["learning"]]
        actual_views = ranking["views"]
        reused = config.get("energy_scope") == "existing_learning_subset"
        if (
            ranking["study"] != study["identity"]
            or (not reused and actual_views != expected_views)
            or (
                reused
                and (
                    ranking.get("requested_views") != expected_views
                    or not actual_views
                    or not set(actual_views).issubset(expected_views)
                    or set(actual_views) & set(ranking["missing_views"])
                    or set(actual_views) | set(ranking["missing_views"]) != set(expected_views)
                )
            )
        ):
            raise ValueError("energy ranking uses another learning pool")
        for p, sha in ranking["records"].items():
            if provenance.file_sha256(p) != sha:
                raise ValueError("energy evidence changed")
        scores = torch.tensor(ranking["scores"])
    elif args.arm == "random":
        scores = torch.rand(48, 32, generator=torch.Generator().manual_seed(args.replicate_seed + 30000))
    directory = root / f"training/{args.arm}/replicate_{args.replicate_seed}"
    paths = sorted((directory / "checkpoints").glob("update_*.pt"))
    previous = checkpoints.load(paths[-1], identity) if paths else None
    if previous:
        if previous.get("draws_per_update", 2) != draw_count:
            raise ValueError("checkpoint draw count differs from frozen batch")
        consumed = []
        for receipt_path in sorted((directory / "updates").glob("*.json")):
            receipt = json.loads(receipt_path.read_text())
            if receipt["update"] <= previous["update"]:
                consumed += receipt["draws"]
        if uniform_f18 and any("training_window" not in draw for draw in consumed):
            raise ValueError("uniform F18 resume requires saved window evidence")
        # Each rank checks a disjoint share; any failure stops every rank.
        share = consumed[runtime.rank :: runtime.world] if runtime else consumed
        failure = None
        try:
            training_windows.verify_consumed_inputs(share, manifest_key="input_manifest_sha256", windows=uniform_f18)
        except ValueError as error:
            failure = error
        if runtime.any(failure is not None) if runtime else failure is not None:
            raise failure or ValueError("another rank found a changed consumed training input")
    lineage = previous["lineage"] if previous else {}
    if fresh and previous is None:
        from scripts.prune.score import export_pruned

        if args.arm == "dense_fresh":
            fixed_mask = torch.ones(48, 32)
        else:
            source_arm = "energy" if args.arm == "energy_fresh" else "learned_joint"
            frozen_path = (
                root
                / f"training/{source_arm}/replicate_{args.replicate_seed}/checkpoints/update_{selection_end:04d}.pt"
            )
            frozen = checkpoints.load(frozen_path)
            if frozen["identity"]["study"] != study["identity"]:
                raise ValueError("fresh mask comes from another study")
            fixed_mask = frozen["mask"].cpu()
            if args.arm == "energy_counts_fresh":
                ranking = json.loads((root / "calibration/ranking.json").read_text())
                if ranking["study"] != study["identity"]:
                    raise ValueError("fixed-count energy ranking belongs to another study")
                per_block_counts = [[int(row.sum())] for row in fixed_mask]
                fixed_mask = select_legal(torch.tensor(ranking["scores"]), int(fixed_mask.sum()), per_block_counts)
            lineage["mask_source"] = {"path": str(frozen_path), "sha256": provenance.file_sha256(frozen_path)}
        source_path = Path(study["components"]["transformer"]["path"])
        model_path = root / f"exports/{args.arm}/replicate_{args.replicate_seed}/compact_base.safetensors"
        if not model_path.exists() and (runtime is None or runtime.primary):
            records.publish(
                model_path,
                lambda p: export_pruned.export(
                    source_path,
                    {f"{b}.attn1": row.tolist() for b, row in enumerate(fixed_mask)},
                    p,
                    model_key="2.5",
                    mode="compact",
                ),
            )
        if runtime:
            torch.distributed.barrier()
        gates.decision = fixed_mask.to(current.device)
        lineage.update(
            {
                "fresh_adapters": True,
                "fresh_optimizer": True,
                "full_source_initializer_seed": args.replicate_seed + 10000,
            }
        )
    if args.branch_prefix:
        if previous:
            raise ValueError("prefix branch must start a new training directory")
        previous = checkpoints.load(args.branch_prefix)
        lineage = checkpoints.verify_prefix(previous, identity["prefix"])
        lineage.update(
            {
                "source_checkpoint": str(args.branch_prefix.resolve()),
                "source_sha256": provenance.file_sha256(args.branch_prefix),
            }
        )
        prefix_updates = args.branch_prefix.parent.parent / "updates"
        lineage["prefix_receipts"] = {
            str(p.resolve()): provenance.file_sha256(p)
            for p in sorted(prefix_updates.glob("*.json"))
            if json.loads(p.read_text())["update"] <= soft_end
        }
    start = previous["update"] if previous else (selection_end if fresh else 0)
    compact = args.phase == "compact_recovery"
    if compact and not fresh and (previous is None or start < selection_end):
        raise ValueError("compact recovery requires the completed selection checkpoint")
    model_path = (
        root / f"exports/{args.arm}/replicate_{args.replicate_seed}/compact_base.safetensors"
        if compact
        else Path(study["components"]["transformer"]["path"])
    )
    if not compact and start >= selection_end:
        return
    with current.transformer(model_path) as model, torch.enable_grad():
        model.train().requires_grad_(False)
        model.velocity_model.set_gradient_checkpointing(True)
        attached = adapters.attach(model, args.replicate_seed, config["rank"])
        frozen_parameters = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        ap = adapters.parameters(attached)
        ao = optimizer(ap, config["adapter_lr"], config)
        go = optimizer([gates.alpha], config["gate_lr"], config) if scores is None and not compact else None
        if fresh and previous is None:
            adapters.load_state(
                attached,
                adapters.transfer(
                    adapters.fresh_source_state(args.replicate_seed, rank=config["rank"]), gates.decision.cpu()
                ),
            )
            save_checkpoint(
                directory / f"checkpoints/update_{selection_end:04d}.pt",
                identity=identity,
                update=selection_end,
                phase="compact",
                gates=gates,
                adapters=attached,
                adapter_optimizer=ao,
                lineage=lineage,
            )
        if previous:
            state = previous["adapters"]
            opt_state = previous["adapter_optimizer"]
            if compact and start == selection_end and not fresh:
                state = adapters.transfer(state, previous["mask"])
                names = [f"{name}.{k}" for name in attached for k in ("A", "B")]
                opt_state = adapters.transfer_optimizer(opt_state, names, previous["mask"])
                lineage = {
                    **lineage,
                    "selection_sha256": provenance.file_sha256(paths[-1]),
                    "compact_base_sha256": provenance.file_sha256(model_path),
                }
            adapters.load_state(attached, state)
            ao.load_state_dict(opt_state)
            gates.alpha.data.copy_(previous["alpha"])
            gates.decision = previous["mask"].to(current.device) if previous["mask"] is not None else None
            if go and previous["gate_optimizer"]:
                go.load_state_dict(previous["gate_optimizer"])
            checkpoints.restore_rng(previous["rng"])
        sources = {s["relative_dir"]: s for s in study["splits"]["learning"]}
        end = (compact_end + 1000 if args.arm == "learned_sequential" else compact_end) if compact else selection_end
        live = hooks.attach_head_provider(model, gates.provider) if not compact else hooks.MaskAttachments()
        from scripts.prune.learn.step import TrainingStep

        if runtime:
            gates.alpha.requires_grad_(go is not None)
        training = TrainingStep(model, gates, config, args.arm)
        if runtime:
            training = runtime.wrap_step(training)
        frozen_parameters = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        stop = [False]
        old = signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
        phase = previous["phase"] if previous else "soft"
        try:
            if not previous and not fresh:
                save_checkpoint(
                    directory / "checkpoints/update_0000.pt",
                    identity=identity,
                    update=0,
                    phase="soft",
                    gates=gates,
                    adapters=attached,
                    adapter_optimizer=ao,
                    gate_optimizer=go,
                )
            for update in range(start + 1, end + 1):
                previous_mask = gates.decision.detach().clone() if gates.decision is not None else None
                requested_stop = stop[0] or time.monotonic() >= deadline
                if runtime.any(requested_stop) if runtime else requested_stop:
                    if update > start + 1:
                        save_checkpoint(
                            directory / f"checkpoints/update_{update - 1:04d}.pt",
                            identity=identity,
                            update=update - 1,
                            phase=phase,
                            gates=gates,
                            adapters=attached,
                            adapter_optimizer=ao,
                            gate_optimizer=go,
                            lineage=lineage,
                        )
                    raise TimeoutError("training bound reached; last checkpoint is resumable")
                phase = (
                    "compact"
                    if compact
                    else (
                        "fixed"
                        if scores is not None
                        else ("soft" if update <= soft_end or args.arm == "learned_soft_only" else "binary")
                    )
                )
                ramp = min(update / 300, 1.0)
                if phase == "fixed":
                    ramp_removed = nearest_removed(args.removed_heads * ramp, allowed, heads=config["heads"])
                    gates.begin_update("fixed", config["blocks"] * config["heads"] - ramp_removed, scores, ramp)
                elif phase == "binary":
                    gates.begin_update("binary", kept)
                else:
                    gates.phase = phase
                phase_start = selection_end if compact else (soft_end if phase == "binary" else 0)
                for group in ao.param_groups:
                    group["lr"] = config["adapter_lr"] * min((update - phase_start) / config["warmup_updates"], 1.0)
                ao.zero_grad(set_to_none=True)
                if go:
                    go.zero_grad(set_to_none=True)
                started = time.monotonic()
                scalars = []
                for device in devices:
                    torch.cuda.reset_peak_memory_stats(device)
                draw_update = update - selection_end if compact and args.arm == "learned_sequential" else update
                selected = protocol.update_draws(draws, draw_update, config, runtime.rank if runtime else None)
                if runtime:
                    runtime.mask(gates)
                accumulation = 1 if runtime else len(selected)
                for draw in selected:
                    sigma = study_sigmas(study)[draw["sigma_index"]]
                    grid, video, c0, capture, eps, inputs, manifest = training_input_case(
                        study, current, sources[draw["view"]], draw["noise_seed"], sigma
                    )
                    regions = None
                    if args.arm in ("learned_person", "learned_small_regions"):
                        source = sources[draw["view"]]
                        matte, person_grid, _ = foreground.person_coverage(
                            Path(source["resolved_dir"]), source["capture_encode_record"], replay=False
                        )
                        regions = {"person": person_grid.reshape(1, -1).to(current.device)}
                        if args.arm == "learned_small_regions":
                            tracks = study["region_tracks"]["views"][draw["view"]]
                            for name in ("face", "left_hand", "right_hand"):
                                region_grid, _ = foreground.region_coverage(
                                    tracks[name], matte, source["capture_encode_record"], schema=tracks["schema"]
                                )
                                regions[name] = region_grid.reshape(1, -1).to(current.device)
                    regions = training_windows.slice_regions(regions, manifest.get("training_window"))
                    loss, official, penalty = training(
                        video,
                        eps,
                        capture,
                        grid.tokens_per_latent_frame,
                        update,
                        phase,
                        regions,
                        accumulation,
                    )
                    loss.backward()
                    scalars.append(
                        {
                            "official": float(official),
                            "penalty": float(penalty.detach()),
                            "total_loss": float(loss.detach()) * accumulation,
                            "sigma": sigma,
                            "sigma_index": draw["sigma_index"],
                            "input_manifest": str(inputs / "manifest.json"),
                            "input_manifest_sha256": provenance.file_sha256(inputs / "manifest.json"),
                            "epsilon_sha256": manifest["videos"][0]["artifacts"]["epsilon_sha256"],
                            **({"training_window": manifest["training_window"]} if uniform_f18 else {}),
                        }
                    )
                    del loss, official, capture, eps, grid, video, c0
                local_scalars = scalars
                if runtime:
                    runtime.reduce_gradients(ap + ([gates.alpha] if go else []))
                    scalars = runtime.draws(scalars)
                anorm = float(torch.nn.utils.clip_grad_norm_(ap, 1.0, error_if_nonfinite=True))
                sparse_gradient = torch.zeros_like(gates.alpha)
                if phase == "soft" and args.arm != "learned_no_sparsity":
                    sparse_gradient = losses.sparsity_gradient(
                        gates.alpha,
                        update=update,
                        phase=phase,
                        lambda_max=config["lambda_max"],
                        lambda_warmup=config.get("lambda_warmup", 300),
                        lambda_start_multiplier=config.get("lambda_start_multiplier", 0.0),
                        kind=config.get("sparsity_penalty", "mean_sigmoid"),
                        logit_offset=config.get("sparsity_logit_offset", 3.0),
                    )
                sparse_norm = float(sparse_gradient.norm())
                velocity_norm = (
                    float((gates.alpha.grad - sparse_gradient).norm()) if go and gates.alpha.grad is not None else 0.0
                )
                # Per-update gate trace: total logit gradient after DDP averaging and before clipping.
                gate_grad = gates.alpha.grad.detach().clone() if go and gates.alpha.grad is not None else None
                gnorm = (
                    float(torch.nn.utils.clip_grad_norm_([gates.alpha], 1.0, error_if_nonfinite=True)) if go else 0.0
                )
                if args.arm != "learned_sequential" or compact:
                    ao.step()
                if go:
                    go.step()
                gate_trace = (
                    torch.stack([gates.alpha.detach(), gate_grad, sparse_gradient.to(gate_grad)]).float().cpu()
                    if gate_grad is not None
                    else None
                )
                if any(p._version != version for p, version in frozen_parameters):
                    raise ValueError("a frozen pretrained tensor changed during learning")
                if not compact and scores is None and (update % 100 == 0 or update in (soft_end, selection_end)):
                    gates.decision = select_legal(gates.alpha, kept, allowed).to(current.device)
                memory = []
                for device in devices:
                    if device.type != "cuda":
                        continue
                    torch.cuda.synchronize(device)
                    free, _ = torch.cuda.mem_get_info(device)
                    memory.append(
                        {
                            "device": str(device),
                            "peak_allocated": torch.cuda.max_memory_allocated(device),
                            "peak_reserved": torch.cuda.max_memory_reserved(device),
                            "free": free,
                        }
                    )
                    if free < config["headroom_bytes"]:
                        raise RuntimeError(f"training device headroom below frozen minimum: {device}")
                if runtime:
                    memory = [item for rank_memory in runtime.gather(memory) for item in rank_memory]
                    runtime.mask(gates)
                rank_metrics = (
                    runtime.gather(
                        {
                            "rank": runtime.rank,
                            "sigma": local_scalars[0]["sigma"],
                            "official": local_scalars[0]["official"],
                            "penalty": local_scalars[0]["penalty"],
                            "total_loss": local_scalars[0]["total_loss"],
                            "seconds": time.monotonic() - started,
                        }
                    )
                    if runtime
                    else []
                )
                receipt = {
                    "study": study["identity"],
                    "update": update,
                    "phase": phase,
                    "draws": scalars,
                    "adapter_gradient_norm": anorm,
                    "logit_gradient_norm": gnorm,
                    "velocity_logit_gradient_norm": velocity_norm,
                    "sparsity_logit_gradient_norm": sparse_norm,
                    "sparsity_penalty_kind": config.get("sparsity_penalty", "mean_sigmoid"),
                    "sparsity_logit_offset": config.get("sparsity_logit_offset", 3.0),
                    "sparsity_lambda": losses.sparsity_coefficient(
                        update,
                        phase,
                        lambda_max=config["lambda_max"],
                        lambda_warmup=config.get("lambda_warmup", 300),
                        lambda_start_multiplier=config.get("lambda_start_multiplier", 0.0),
                    )
                    if phase == "soft" and args.arm != "learned_no_sparsity"
                    else 0.0,
                    "legal_mask_changes": int((gates.decision - previous_mask).abs().sum())
                    if previous_mask is not None and gates.decision is not None
                    else 0,
                    "gates": gates.snapshot(),
                    "seconds": time.monotonic() - started,
                    "peak_bytes": torch.cuda.max_memory_allocated(current.device),
                    "devices": memory,
                    **({"rank_metrics": rank_metrics} if runtime else {}),
                }
                receipt_path = directory / f"updates/{update:04d}.json"
                if receipt_path.exists():
                    old_receipt = json.loads(receipt_path.read_text())
                    fixed = [k for k in receipt if k not in ("seconds", "peak_bytes", "devices", "rank_metrics")]
                    if any(old_receipt.get(k) != receipt[k] for k in fixed):
                        raise ValueError("replayed update differs from immutable progress evidence")
                elif runtime is None or runtime.primary:
                    records.publish_json(receipt_path, receipt)
                if gate_trace is not None and (runtime is None or runtime.primary):
                    # Rows: logits after the step, total gradient, sparsity gradient. Replays must match bytes.
                    records.publish_tensor(directory / f"gate_trace/{update:04d}.pt", gate_trace)
                if update == end or update % 100 == 0 or update in (soft_end, selection_end, compact_end):
                    save_checkpoint(
                        directory / f"checkpoints/update_{update:04d}.pt",
                        identity=identity,
                        update=update,
                        phase=phase,
                        gates=gates,
                        adapters=attached,
                        adapter_optimizer=ao,
                        gate_optimizer=go,
                        lineage=lineage,
                    )
                    if not compact:
                        monitor(study, current, model, gates, args.arm, args.replicate_seed, update, kept, scores)
                print(
                    f"{args.arm} update={update} phase={phase} velocity_mse={sum(r['official'] for r in scalars) / draw_count:.6g}",
                    flush=True,
                )
                if args.max_updates and update - start >= args.max_updates:
                    if update % 100:
                        save_checkpoint(
                            directory / f"checkpoints/update_{update:04d}.pt",
                            identity=identity,
                            update=update,
                            phase=phase,
                            gates=gates,
                            adapters=attached,
                            adapter_optimizer=ao,
                            gate_optimizer=go,
                            lineage=lineage,
                        )
                    return
        finally:
            signal.signal(signal.SIGTERM, old)
            live.detach_all()
    if runtime is None or runtime.primary:
        records.publish_json(
            directory / f"{args.phase}_completion.json",
            {"identity": identity, "update": end, "status": "complete"},
        )


def feasibility(study, current, deadline):
    config = study["config"]
    devices = getattr(current, "execution_devices", (current.device,))
    root = Path(study["root"])
    cases = [next(s for s in study["splits"]["learning"] if s["n_latent_frames"] == length) for length in (18, 28)]
    results = []
    with current.transformer(Path(study["components"]["transformer"]["path"])) as model:
        model.eval().requires_grad_(False)
        for source in cases:
            rows = []
            for sigma in study_sigmas(study):
                if time.monotonic() >= deadline:
                    raise TimeoutError("native feasibility bound reached")
                grid, video, c0, capture, eps, _, _ = input_case(study, current, source, 42, sigma)
                started = time.monotonic()
                native, _ = model(video=video, audio=None, perturbations=None)
                torch.cuda.synchronize(current.device)
                teacher_seconds = time.monotonic() - started
                velocity, clean = ltx_adapter.velocity_forward(model, video)
                if not torch.equal(clean, native):
                    raise ValueError("native velocity conversion differs from ordinary wrapper")
                gates = HeadGates().to(current.device)
                gates.begin_update("all_one")
                gates.microbatch()
                with hooks.attach_head_provider(model, gates.provider):
                    all_one, _ = ltx_adapter.velocity_forward(model, video)
                if not torch.equal(all_one, velocity):
                    raise ValueError("all-one live provider differs from native velocity")
                terms = losses.velocity_mse(velocity, eps, capture, grid.tokens_per_latent_frame)
                destination = (
                    Path(study["root"])
                    / "checks/feasibility"
                    / f"F{source['n_latent_frames']}"
                    / f"sigma_{study_sigmas(study).index(sigma)}"
                )
                records.publish_tensor(destination / "native_velocity.pt", velocity)
                records.publish_tensor(destination / "native_clean.pt", common.with_clean_prefix(clean, c0))
                gates.begin_update("soft")
                gates.microbatch()
                with hooks.attach_head_provider(model, gates.provider):
                    near_velocity, near_clean = ltx_adapter.velocity_forward(model, video)
                near_difference = float((near_clean.float() - clean.float()).abs().max())
                records.publish_tensor(
                    destination / "initialized_soft_clean.pt", common.with_clean_prefix(near_clean, c0)
                )
                rows.append(
                    {
                        "sigma": sigma,
                        "native_parity": True,
                        "all_one_parity": True,
                        "official_loss": float(terms["official"]),
                        "initialized_soft_max_abs": near_difference,
                        "teacher_seconds": teacher_seconds,
                        "clean_frame_equal": torch.equal(clean[:, : len(c0[0])], c0),
                    }
                )
            results.append({"frames": source["n_latent_frames"], "rows": rows})
            records.publish_json(
                root / f"checks/feasibility/F{source['n_latent_frames']}/controls.json",
                {"study": study["identity"], "rows": rows},
            )
            print(f"F{source['n_latent_frames']} all declared native controls passed", flush=True)
        del native, velocity, clean, all_one, near_velocity, near_clean, terms
        # Verify zero-adapter training parity before bounded ten-update backwards.
        attached = adapters.attach(model, 42, config["rank"])
        frozen_parameters = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        ap = adapters.parameters(attached)
        ao = optimizer(ap, config["adapter_lr"], config)
        gates = HeadGates().to(current.device)
        go = optimizer([gates.alpha], config["gate_lr"], config)
        model.velocity_model.set_gradient_checkpointing(True)
        for source in cases:
            for update in range(1, 11):
                if time.monotonic() >= deadline:
                    raise TimeoutError("native feasibility bound reached")
                with torch.enable_grad():
                    (root / "checks").mkdir(exist_ok=True)
                    (root / "checks/feasibility_progress.json").write_text(
                        json.dumps(
                            {
                                "frames": source["n_latent_frames"],
                                "update": update,
                                "sigma": study_sigmas(study)[(update - 1) % len(study_sigmas(study))],
                                "phase": "soft" if update <= 5 else "binary",
                            }
                        )
                        + "\n"
                    )
                    grid, video, c0, capture, eps, _, _ = input_case(
                        study,
                        current,
                        source,
                        42 + update,
                        study_sigmas(study)[(update - 1) % len(study_sigmas(study))],
                    )
                    # Parity is an inference control. Retaining its full-clip adapter
                    # graph defeats checkpointing before the actual backward test.
                    # Only the first update has zero adapter deltas, so only it runs the control.
                    if update == 1:
                        with torch.no_grad():
                            model.eval()
                            for module in attached.values():
                                module.enabled = False
                            baseline_v, baseline_z = ltx_adapter.velocity_forward(model, video)
                            for module in attached.values():
                                module.enabled = True
                            adapter_v, adapter_z = ltx_adapter.velocity_forward(model, video)
                            if not torch.equal(adapter_v, baseline_v) or not torch.equal(adapter_z, baseline_z):
                                raise ValueError("zero-adapter native control failed")
                            del adapter_v, adapter_z, baseline_v, baseline_z
                    model.train()
                    gates.begin_update("soft" if update <= 5 else "binary", 1384)
                    gate = gates.microbatch()
                    ao.zero_grad(set_to_none=True)
                    go.zero_grad(set_to_none=True)
                    for device in devices:
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.monotonic()
                    with hooks.attach_head_provider(model, gates.provider):
                        velocity, _ = ltx_adapter.velocity_forward(model, video)
                        terms = losses.velocity_mse(velocity, eps, capture, grid.tokens_per_latent_frame)
                        loss, _ = losses.phase_loss(
                            terms,
                            gate,
                            update=update,
                            phase=gates.phase,
                            accumulation=2,
                            lambda_max=config["lambda_max"],
                            lambda_warmup=config.get("lambda_warmup", 300),
                            lambda_start_multiplier=config.get("lambda_start_multiplier", 0.0),
                            logits=gates.alpha,
                            sparsity_kind=config.get("sparsity_penalty", "mean_sigmoid"),
                            logit_offset=config.get("sparsity_logit_offset", 3.0),
                        )
                        loss.backward()
                        del velocity, loss, terms, gate
                        gate = gates.microbatch()
                        velocity, _ = ltx_adapter.velocity_forward(model, video)
                        terms = losses.velocity_mse(velocity, eps, capture, grid.tokens_per_latent_frame)
                        loss, _ = losses.phase_loss(
                            terms,
                            gate,
                            update=update,
                            phase=gates.phase,
                            accumulation=2,
                            lambda_max=config["lambda_max"],
                            lambda_warmup=config.get("lambda_warmup", 300),
                            lambda_start_multiplier=config.get("lambda_start_multiplier", 0.0),
                            logits=gates.alpha,
                            sparsity_kind=config.get("sparsity_penalty", "mean_sigmoid"),
                            logit_offset=config.get("sparsity_logit_offset", 3.0),
                        )
                        loss.backward()
                    for device in devices:
                        torch.cuda.synchronize(device)
                    backward_seconds = time.monotonic() - started
                    adapter_norm = float(torch.nn.utils.clip_grad_norm_(ap, 1.0, error_if_nonfinite=True))
                    gate_norm = float(torch.nn.utils.clip_grad_norm_([gates.alpha], 1.0, error_if_nonfinite=True))
                    if not adapter_norm > 0 or not gate_norm > 0:
                        raise ValueError("native adapter or gate gradient is zero")
                    ao.step()
                    go.step()
                    if any(p._version != version for p, version in frozen_parameters):
                        raise ValueError("native feasibility changed a frozen pretrained tensor")
                    memory = [
                        {
                            "device": str(device),
                            "peak_bytes": torch.cuda.max_memory_allocated(device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                            "free_bytes": torch.cuda.mem_get_info(device)[0],
                        }
                        for device in devices
                    ]
                    free = min(row["free_bytes"] for row in memory)
                    if free < config["headroom_bytes"]:
                        raise MemoryError("native backward lacks declared 2 GiB headroom")
                    results.append(
                        {
                            "frames": source["n_latent_frames"],
                            "update": update,
                            "loss": float(loss.detach()),
                            "peak_bytes": torch.cuda.max_memory_allocated(current.device),
                            "free_bytes": free,
                            "forward_backward_seconds": backward_seconds,
                            "devices": memory,
                            "adapter_gradient_norm": adapter_norm,
                            "gate_gradient_norm": gate_norm,
                        }
                    )
                    records.publish_json(
                        root / f"checks/feasibility/F{source['n_latent_frames']}/update_{update:02d}.json",
                        {"study": study["identity"], **results[-1]},
                    )
                    print(
                        f"F{source['n_latent_frames']} update={update}/10 phase={gates.phase} seconds={backward_seconds:.1f} min_free_GiB={free / 2**30:.2f}",
                        flush=True,
                    )
                    del velocity, loss, terms, gate, grid, video, c0, capture, eps
            # Reset paired zero-delta state for the second length.
            adapters.load_state(attached, adapters.fresh_source_state(42, rank=config["rank"]))
            gates.alpha.data.fill_(3.0)
            ao = optimizer(ap, config["adapter_lr"], config)
            go = optimizer([gates.alpha], config["gate_lr"], config)
    records.publish_json(
        Path(study["root"]) / "checks/feasibility.json",
        {
            "study": study["identity"],
            "passed": True,
            "speed_layout_passed": False,
            "results": results,
            "remaining": "head and FFN execution probes must establish speed before speed training",
        },
    )


def probe_layouts(config):
    """Seeded execution probes; identities are unrelated to quality selection."""
    rng = torch.Generator().manual_seed(42 + 30000)
    block_order = torch.randperm(48, generator=rng).tolist()
    head_order = [torch.randperm(32, generator=rng).tolist() for _ in range(48)]
    layouts = {"all_one": {f"{b}.attn1": [1] * 32 for b in range(48)}}
    for name, reductions in (("38x4", [4] * 38), ("19x8", [8] * 19), ("mixed", [8] * 10 + [4] * 18)):
        masks = {f"{b}.attn1": [1] * 32 for b in range(48)}
        for block, count in zip(block_order, reductions):
            for h in head_order[block][:count]:
                masks[f"{block}.attn1"][h] = 0
        layouts[name] = masks
    channels = [torch.randperm(16384, generator=rng).tolist() for _ in range(48)]
    for width in config["ffn_probe_widths"]:
        mask = {}
        for block in range(48):
            values = torch.zeros(16384, dtype=torch.int64)
            values[channels[block][:width]] = 1
            mask[f"{block}.ff"] = values.tolist()
        layouts[f"ffn_{width}"] = mask
    return layouts


def timing_decision(arms, order):
    summary = timing.bracket(arms, order, "median_ms")
    return {
        "speedup": summary["speedup"],
        "outside_drift_ms": summary["outside_drift"],
        "drift_fraction": summary["drift_fraction"],
        "saving_ms": summary["saving"],
        "passed": summary["speedup"] >= 1.05 and summary["saving"] > 2 * summary["outside_drift"],
    }


def execution_probes(study, current, deadline):
    from safetensors import safe_open

    from scripts.prune.score import export_pruned

    root, config = Path(study["root"]), study["config"]
    source_path = Path(study["components"]["transformer"]["path"])
    cases = [next(s for s in study["splits"]["learning"] if s["n_latent_frames"] == length) for length in (18, 28)]
    sigma = study_sigmas(study)[0]
    evidence = {}
    for label, masks in probe_layouts(config).items():
        output = root / f"exports/execution_probes/{label}.safetensors"
        if not output.exists():
            records.publish(
                output, lambda p: export_pruned.export(source_path, masks, p, model_key="2.5", mode="compact")
            )
        with safe_open(output, framework="pt", device="cpu") as handle:
            params = sum(torch.Size(handle.get_slice(k).get_shape()).numel() for k in handle.keys())
        rows = []
        for case in cases:
            grid, video, c0, _, _, _, _ = input_case(study, current, case, 42, sigma)
            expected = {}
            for arm, path in (("A", source_path), ("B", output)):
                tensor_path = root / f"checks/execution_probes/{label}/{case['n_latent_frames']}_{arm}.pt"
                if not tensor_path.exists():
                    with current.transformer(path) as model:
                        prediction, _ = model(video=video, audio=None, perturbations=None)
                        records.publish_tensor(
                            tensor_path,
                            grid.unpatchify_block(common.with_clean_prefix(prediction, c0), grid.latent_frames),
                        )
                expected[arm] = tensor_path
            a, b = [torch.load(expected[k], map_location="cpu", weights_only=True) for k in ("A", "B")]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite timing probe predictions")
            functional_path = root / f"checks/execution_probes/{label}/{case['n_latent_frames']}_functional.pt"
            if not functional_path.exists():
                with current.transformer(source_path) as model:
                    tensors = {name: torch.tensor(values) for name, values in masks.items()}
                    with (
                        hooks.attach_head_masks(model, tensors, requires_grad=False),
                        hooks.attach_ffn_masks(model, tensors, requires_grad=False),
                    ):
                        prediction, _ = model(video=video, audio=None, perturbations=None)
                    records.publish_tensor(
                        functional_path,
                        grid.unpatchify_block(common.with_clean_prefix(prediction, c0), grid.latent_frames),
                    )
            functional = torch.load(functional_path, map_location="cpu", weights_only=True)
            if not torch.isfinite(functional).all():
                raise ValueError("nonfinite functional timing probe")
            parity = {
                "max_abs": float((functional.float() - b.float()).abs().max()),
                "bit_exact": torch.equal(functional, b),
                "dense_to_compact_max_abs": float((a.float() - b.float()).abs().max()),
                "reference": "functional_mask",
            }
            if label == "all_one" and not parity["bit_exact"]:
                raise ValueError("all-one compact export changed native output")
            for order in ("ABA", "BAB"):
                result_path = root / f"timing/execution_probes/{label}/{case['n_latent_frames']}_{order}.json"
                if result_path.exists():
                    saved = json.loads(result_path.read_text())
                    if saved["study"] != study["identity"]:
                        raise ValueError("timing probe study changed")
                    rows.append(saved)
                    continue
                repetitions = config["timing_repeats"]
                while True:
                    times = []
                    for i, arm in enumerate(order):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("execution probe bound reached")
                        path = source_path if arm == "A" else output
                        times.append(
                            timing.measure_arm(
                                current,
                                path,
                                expected[arm],
                                grid,
                                video,
                                c0,
                                warmup=config["timing_warmups"],
                                repeats=repetitions,
                                label=f"{arm}{i}",
                            )
                        )
                    result = timing_decision(times, order)
                    if result["drift_fraction"] > 0.02 and repetitions < 20:
                        repetitions = 20
                        continue
                    break
                saved = {
                    "study": study["identity"],
                    "layout": label,
                    "frames": case["n_latent_frames"],
                    "order": order,
                    "repeats": repetitions,
                    "arms": times,
                    "parity": parity,
                    **result,
                }
                records.publish_json(result_path, saved)
                rows.append(saved)
        evidence[label] = {
            "rows": rows,
            "checkpoint_bytes": output.stat().st_size,
            "total_parameters": params,
            "passed_both_lengths_orders": all(r["passed"] for r in rows),
        }
    records.publish_json(
        root / "checks/execution_probes.json",
        {
            "study": study["identity"],
            "layouts": evidence,
            "speed_layout_passed": any(evidence[k]["passed_both_lengths_orders"] for k in ("38x4", "19x8", "mixed")),
        },
    )


def monitor(study, current, model, gates, arm, seed, update, kept, scores=None):
    """Monitoring cannot advance optimizer RNG or retain its temporary gate graph."""
    rng = checkpoints.rng_state()
    original = gates.phase, gates.decision, gates.current
    try:
        return _monitor(study, current, model, gates, arm, seed, update, kept, scores)
    finally:
        gates.phase, gates.decision, gates.current = original
        checkpoints.restore_rng(rng)


def _monitor(study, current, model, gates, arm, seed, update, kept, scores=None):
    """Paired quick validation and hardening evidence, with unchanged adapters."""
    from scripts.prune.evaluate.learned_heads import capture_metrics, paired_metrics

    root = Path(study["root"])
    gt_only = study["config"].get("monitor_reference") == "gt_capture"
    runtime = getattr(current, "distributed", None)
    cache = None if gt_only else TeacherReferences(root / "teacher/references")
    actors = study["splits"]["validation_actors"][:4]
    cases = [next(s for s in study["splits"]["validation"] if s["actor"] == actor) for actor in actors]
    ddp = runtime is not None and runtime.uses_ddp
    if ddp:
        cases = [source for index, source in enumerate(cases) if index % runtime.world == runtime.rank]
    original = gates.phase, gates.decision, gates.current
    rows = []
    final_mask = select_legal(gates.alpha if scores is None else scores, kept, gates.allowed).to(current.device)
    with torch.no_grad():
        for source in cases:
            for i in monitor_sigma_indices(study):
                sigma = study_sigmas(study)[i]
                grid, video, c0, capture, eps, inputs, manifest = input_case(study, current, source, 2042, sigma)
                ref_id = (
                    None if gt_only else reference_identity(study, current, grid, video, eps, inputs, manifest, sigma)
                )
                teacher = None if gt_only else cache.read(ref_id)
                outputs = {}
                for state in ("trained", "hardened"):
                    gates.phase = original[0] if state == "trained" else "compact"
                    gates.decision = original[1] if state == "trained" else final_mask
                    gates.microbatch()
                    velocity, clean = ltx_adapter.velocity_forward(model, video)
                    clean = common.with_clean_prefix(clean, c0)
                    directory = (
                        root
                        / f"checks/{arm}/replicate_{seed}/monitor_{update:04d}"
                        / hashlib.sha256(source["relative_dir"].encode()).hexdigest()
                    )
                    if gt_only:
                        checks = capture_metrics(
                            velocity, clean, capture, eps, prefix_tokens=grid.tokens_per_latent_frame
                        )
                    else:
                        checks = paired_metrics(
                            velocity,
                            clean,
                            teacher["velocity"].to(current.device),
                            teacher["clean"].to(current.device),
                            capture,
                            eps,
                            sigma=sigma,
                            prefix_tokens=grid.tokens_per_latent_frame,
                        )
                    path = directory / f"sigma_{i}_{state}.pt"
                    if runtime is None or runtime.primary or ddp:
                        records.publish_tensor(path, grid.unpatchify_block(clean, grid.latent_frames))
                    if runtime and not ddp:
                        torch.distributed.barrier()
                    outputs[state] = {"metrics": checks, "path": str(path), "sha256": provenance.file_sha256(path)}
                rows.append(
                    {
                        "view": source["relative_dir"],
                        "actor": source["actor"],
                        "sigma": sigma,
                        "states": outputs,
                        "input_manifest_sha256": provenance.file_sha256(inputs / "manifest.json"),
                        "epsilon_sha256": manifest["videos"][0]["artifacts"]["epsilon_sha256"],
                        "context_sha256": study["text_context"]["tensor_sha256"],
                        "source": source,
                        "reference": "gt_capture" if gt_only else "original_teacher",
                        "teacher_key": None if gt_only else records.digest(ref_id),
                        **({"rank": runtime.rank} if ddp else {}),
                    }
                )
    gates.phase, gates.decision, gates.current = original
    if ddp:
        rows = [row for rank_rows in runtime.gather(rows) for row in rank_rows]
    if runtime is None or runtime.primary:
        records.publish_json(
            root / f"checks/{arm}/replicate_{seed}/monitor_{update:04d}.json",
            {
                "study": study["identity"],
                "update": update,
                "rows": rows,
                "hardening_is_model_change": True,
            },
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument(
        "--phase",
        choices=(
            "feasibility",
            "execution_probes",
            "energy_calibration",
            "teacher_references",
            "selection",
            "compact_recovery",
        ),
        required=True,
    )
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--arm", choices=ARMS, default="learned_joint")
    p.add_argument("--replicate-seed", type=int, default=42)
    p.add_argument("--removed-heads", type=int, default=152)
    p.add_argument("--gpu-id", type=int, required=True)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--branch-prefix", type=Path)
    p.add_argument(
        "--quality-research", action="store_true", help="Explicit quality pilot if timing has not established speed."
    )
    args = p.parse_args()
    numerics.deterministic_policy()
    torch.set_num_threads(4)
    # Ranks share the pinned component files; one full hash per launch is enough.
    study = load_study(args.study, verify_components=int(os.environ.get("RANK", "0")) == 0)
    if args.replicate_seed not in study["config"]["replicate_seeds"]:
        p.error("replicate seed is not frozen")
    distributed = None
    if protocol.is_ddp(study["config"]) and args.phase in ("selection", "compact_recovery"):
        from scripts.prune.learn.ddp import DDPRuntime, Session

        distributed = DDPRuntime(study["config"])
        args.gpu_id = distributed.local_rank
    current = open_current(study, args.gpu_id)
    if distributed:
        current = Session(current, distributed)
    deadline = time.monotonic() + study["config"]["wall_time_hours"] * 3600
    random.seed(args.replicate_seed)
    np.random.seed(args.replicate_seed)
    torch.manual_seed(args.replicate_seed)
    started = time.monotonic()
    try:
        if args.phase == "feasibility":
            feasibility(study, current, deadline)
        elif args.phase == "execution_probes":
            execution_probes(study, current, deadline)
        elif args.phase == "energy_calibration":
            energy_phase(study, current, deadline)
        elif args.phase == "teacher_references":
            teacher_phase(study, current, args.split, deadline)
        else:
            training_phase(study, current, args, deadline)
        records.publish_json(
            Path(study["root"])
            / f"provenance/phase_receipts/{args.phase}_rank{args.gpu_id}_{provenance.run_id()}.json",
            {
                "study": study["identity"],
                "phase": args.phase,
                "arm": args.arm,
                "replicate_seed": args.replicate_seed,
                "status": "bounded_partial" if args.max_updates else "returned_without_error",
                "elapsed_seconds": time.monotonic() - started,
                "gpu_hours": (time.monotonic() - started)
                * len(getattr(current, "execution_devices", (current.device,)))
                / 3600,
                "gpu_id": args.gpu_id,
                "gpu_name": torch.cuda.get_device_name(current.device),
                "scope": "resident phase wall time; includes monitoring; session setup recorded separately",
            },
        )
    except Exception as exc:
        records.publish_json(
            Path(study["root"]) / f"logs/{args.phase}_{provenance.run_id()}.failure.json",
            {
                "study": study["identity"],
                "phase": args.phase,
                "error": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "gpu_id": args.gpu_id,
            },
        )
        raise
    finally:
        if distributed:
            distributed.close()


if __name__ == "__main__":
    main()
