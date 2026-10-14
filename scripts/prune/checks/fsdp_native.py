"""Native four-rank FSDP parity, full-clip backward and exact resume acceptance."""

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.prune.core import ltx_adapter, provenance
from scripts.prune.data.learned_head_study import load_study, study_sigmas
from scripts.prune.learn import adapters, checkpoints, heads, losses
from scripts.prune.learn.distributed import Runtime, Session
from scripts.prune.learn.gates import HeadGates
from scripts.prune.score import hooks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    heads.numerical_policy()
    runtime = Runtime()
    torch.set_num_threads(4)
    study = load_study(args.study)
    current = Session(heads.open_current(study, runtime.local_rank), runtime)
    config, root = study["config"], Path(study["root"])
    started = time.monotonic()
    rows = []
    with current.transformer(Path(study["components"]["transformer"]["path"])) as model, torch.enable_grad():
        model.eval().requires_grad_(False)
        baseline_reference = study["config"].get("native_baseline_study")
        if baseline_reference and runtime.primary:
            reference_path = Path(__file__).resolve().parents[4] / baseline_reference / "configs/frozen_study.json"
            reference = json.loads(reference_path.read_text())
            reference_id = reference.pop("identity")
            if checkpoints.digest(reference) != reference_id:
                raise ValueError("serial native baseline source study changed")
            for key in ("components", "splits", "text_context", "software"):
                if study[key] != reference[key]:
                    raise ValueError(f"serial native baseline inputs differ: {key}")
            for key in ("sigmas", "target", "numerics", "objective", "prompt"):
                if study["config"][key] != reference["config"][key]:
                    raise ValueError(f"serial native baseline protocol differs: {key}")
            dependencies = (
                "packages/ltx-core/",
                "packages/ltx-pipelines/src/ltx_pipelines/utils/",
                "scripts/prune/core/",
                "scripts/prune/data/whole_clip.py",
                "scripts/prune/learn/heads.py",
                "scripts/onestep_avatar/model/",
                "scripts/onestep_avatar/corpus/dataset.py",
            )
            for name, sha in study["code"]["files"].items():
                if name.startswith(dependencies) and reference["code"]["files"].get(name) != sha:
                    raise ValueError(f"serial baseline numerical dependency changed: {name}")
            pins = {}
            for frames in (18, 28):
                for index in range(len(study_sigmas(study))):
                    original = Path(reference["root"]) / f"checks/native_baseline/F{frames}/sigma_{index}/velocity.pt"
                    value = torch.load(original, map_location="cpu", weights_only=True)
                    if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                        raise ValueError("invalid saved serial baseline tensor")
                    pins[str(original)] = provenance.file_sha256(original)
                    checkpoints.publish_tensor(
                        root / f"checks/native_baseline/F{frames}/sigma_{index}/velocity.pt", value
                    )
            checkpoints.publish_json(
                root / "provenance/native_baseline_reuse.json",
                {
                    "source_study": reference_id,
                    "source_study_file_sha256": provenance.file_sha256(reference_path),
                    "validated": "same stock numerical dependencies, components, source clips, noise seeds and sigmas",
                    "records": pins,
                },
            )
        elif not baseline_reference:
            if runtime.primary:
                with torch.no_grad():
                    for frames in (18, 28):
                        source = next(s for s in study["splits"]["learning"] if s["n_latent_frames"] == frames)
                        for index, sigma in enumerate(study_sigmas(study)):
                            grid, video, c0, capture, eps, _, _ = heads.input_case(study, current, source, 42, sigma)
                            stock, _ = model(video=video, audio=None, perturbations=None)
                            velocity, clean = ltx_adapter.velocity_forward(model, video)
                            if not torch.equal(stock, clean):
                                raise ValueError("native velocity conversion differs from ordinary unsharded wrapper")
                            checkpoints.publish_tensor(
                                root / f"checks/native_baseline/F{frames}/sigma_{index}/velocity.pt", velocity
                            )
                            del stock, velocity, clean, grid, video, c0, capture, eps
        torch.distributed.barrier()
        attached = adapters.attach(model, 42, config["rank"])
        ap = adapters.parameters(attached)
        gates = HeadGates().to(current.device)
        live = hooks.attach_head_provider(model, gates.provider)
        model = runtime.wrap(model, ap)
        if runtime.primary:
            checkpoints.publish_json(root / "provenance/fsdp_precision_inventory.json", runtime.inventory)
        frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        model.velocity_model.set_gradient_checkpointing(True)
        for frames in (18, 28):
            source = next(s for s in study["splits"]["learning"] if s["n_latent_frames"] == frames)
            gates.alpha.data.fill_(3)
            adapters.load_state(attached, adapters.fresh_source_state(42, rank=config["rank"]))
            ao = heads.optimizer(ap, config["adapter_lr"], config)
            go = heads.optimizer([gates.alpha], config["gate_lr"], config)
            gates.begin_update("all_one")
            gates.microbatch()
            model.eval()
            with torch.no_grad():
                for index, sigma in enumerate(study_sigmas(study)):
                    grid, video, c0, capture, eps, inputs, manifest = heads.input_case(
                        study, current, source, 42, sigma
                    )
                    velocity, clean = ltx_adapter.velocity_forward(model, video)
                    baseline = root / f"checks/native_baseline/F{frames}/sigma_{index}/velocity.pt"
                    if not torch.equal(velocity.cpu(), torch.load(baseline, map_location="cpu", weights_only=True)):
                        raise ValueError("FSDP all-one/zero-adapter velocity differs from saved native prediction")
                    if not torch.equal(
                        clean[:, : grid.tokens_per_latent_frame], capture[:, : grid.tokens_per_latent_frame]
                    ):
                        raise ValueError("FSDP parity control clean frame differs")
                    record = {
                        "study": study["identity"],
                        "frames": frames,
                        "sigma": sigma,
                        "native_parity": True,
                        "all_one_parity": True,
                        "clean_frame_equal": True,
                        "input_manifest_sha256": provenance.file_sha256(inputs / "manifest.json"),
                    }
                    if runtime.primary:
                        checkpoints.publish_json(root / f"checks/fsdp_native/{frames}/control_{index}.json", record)
                    del velocity, clean, grid, video, c0, capture, eps
            model.train()
            directory = root / f"checks/fsdp_native/{frames}"
            ckpt_identity = {"study": study["identity"], "frames": frames}

            def update(number):
                phase = "soft" if number <= 5 else "binary"
                gates.begin_update(phase, 1384 if phase == "binary" else None)
                runtime.mask(gates)
                ao.zero_grad(set_to_none=True)
                go.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(current.device)
                begin = time.monotonic()
                official = []
                for _ in range(2):
                    g = gates.microbatch()
                    grid, video, c0, capture, eps, _, _ = heads.input_case(
                        study, current, source, 42 + number, study_sigmas(study)[(number - 1) % len(study_sigmas(study))]
                    )
                    velocity, clean = ltx_adapter.velocity_forward(model, video)
                    terms = losses.velocity_mse(velocity, eps, capture, grid.tokens_per_latent_frame)
                    loss, _ = losses.phase_loss(
                        terms,
                        g,
                        update=number,
                        phase=phase,
                        logits=gates.alpha,
                        lambda_max=config["lambda_max"],
                        lambda_warmup=config.get("lambda_warmup", 300),
                        lambda_start_multiplier=config.get("lambda_start_multiplier", 0.0),
                        sparsity_kind=config.get("sparsity_penalty", "mean_sigmoid"),
                        logit_offset=config.get("sparsity_logit_offset", 3.0),
                    )
                    loss.backward()
                    official.append(float(terms["official"].detach()))
                    del velocity, clean, terms, loss, grid, video, c0, capture, eps
                runtime.reduce_gradients(ap + [gates.alpha])
                anorm = float(torch.nn.utils.clip_grad_norm_(ap, 1, error_if_nonfinite=True))
                gnorm = float(torch.nn.utils.clip_grad_norm_([gates.alpha], 1, error_if_nonfinite=True))
                if anorm <= 0 or gnorm <= 0:
                    raise ValueError("zero native FSDP trainable gradients")
                ao.step()
                go.step()
                if any(p._version != version for p, version in frozen):
                    raise ValueError("native FSDP frozen parameter version changed")
                torch.cuda.synchronize(current.device)
                free, _ = torch.cuda.mem_get_info(current.device)
                memory = runtime.gather(
                    {
                        "rank": runtime.rank,
                        "peak_bytes": torch.cuda.max_memory_allocated(current.device),
                        "free_bytes": free,
                        "allocated_bytes": torch.cuda.memory_allocated(current.device),
                        "reserved_bytes": torch.cuda.memory_reserved(current.device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(current.device),
                    }
                )
                if any(r["free_bytes"] < config["headroom_bytes"] for r in memory):
                    if runtime.primary:
                        checkpoints.publish_json(
                            directory / f"memory_failure_{number:02d}.json",
                            {"study": study["identity"], "frames": frames, "update": number, "devices": memory},
                        )
                        print("native memory failure " + json.dumps(memory), flush=True)
                    raise RuntimeError("native FSDP rank lacks two GiB headroom")
                runtime.agree(checkpoints.tensor_hash(gates.alpha))
                return {
                    "frames": frames,
                    "update": number,
                    "phase": phase,
                    "official_loss": sum(official) / 2,
                    "adapter_gradient_norm": anorm,
                    "gate_gradient_norm": gnorm,
                    "devices": memory,
                    "seconds": time.monotonic() - begin,
                }

            first_state = None
            for number in range(1, 11):
                record = update(number)
                if number == 1:
                    runtime.save(
                        directory / "resume.pt",
                        identity=ckpt_identity,
                        update=1,
                        phase="soft",
                        gates=gates,
                        adapters=attached,
                        adapter_optimizer=ao,
                        gate_optimizer=go,
                    )
                    first_state = checkpoints.load(directory / "resume.pt", ckpt_identity)
                elif number == 2:
                    expected = adapters.state_dict(attached)
                    expected_alpha = gates.alpha.detach().clone()
                    adapters.load_state(attached, first_state["adapters"])
                    gates.alpha.data.copy_(first_state["alpha"])
                    ao.load_state_dict(first_state["adapter_optimizer"])
                    go.load_state_dict(first_state["gate_optimizer"])
                    checkpoints.restore_rng(first_state["rng"])
                    update(number)
                    if any(
                        not torch.equal(v, expected[k]) for k, v in adapters.state_dict(attached).items()
                    ) or not torch.equal(gates.alpha, expected_alpha):
                        raise ValueError("native FSDP resumed next update differs")
                rows.append(record)
                if runtime.primary:
                    checkpoints.publish_json(directory / f"update_{number:02d}.json", record)
                    print(json.dumps(record), flush=True)
        live.detach_all()
    if runtime.primary:
        checkpoints.publish_json(
            root / "checks/feasibility.json",
            {
                "study": study["identity"],
                "passed": True,
                "method": "four_rank_fsdp",
                "reference": "saved_unsharded_same_inputs",
                "results": rows,
                "controls": 2 * len(study_sigmas(study)),
                "exact_native_resume": True,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    runtime.close()


if __name__ == "__main__":
    main()
