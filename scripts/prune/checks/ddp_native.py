"""Bounded native acceptance for uniform-F18 DDP before sustained training."""

import argparse
import time
from pathlib import Path

import torch

from scripts.prune.core import ltx_adapter, provenance
from scripts.prune.data import training_windows
from scripts.prune.data.learned_head_study import load_study, study_sigmas
from scripts.prune.learn import adapters, checkpoints, heads
from scripts.prune.learn.ddp import DDPRuntime
from scripts.prune.learn.distributed import Session
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.step import TrainingStep
from scripts.prune.score import hooks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    heads.numerical_policy()
    torch.set_num_threads(4)
    study = load_study(args.study)
    cfg = study["config"]
    root = Path(study["root"])
    runtime = DDPRuntime(cfg)
    current = Session(heads.open_current(study, runtime.local_rank), runtime)
    groups = {length: [s for s in study["splits"]["learning"] if s["n_latent_frames"] == length] for length in (18, 28)}
    length = 18 if runtime.rank % 2 == 0 else 28
    source = groups[length][runtime.rank // 2]
    noise_seed = 42000 + runtime.rank
    while length > 18 and training_windows.bounds(length, noise_seed)[0] == 0:
        noise_seed += runtime.world
    identity = {"study": study["identity"], "phase": "ddp_native"}
    output = root / "checks/ddp_native"
    started = time.monotonic()
    try:
        with current.transformer(Path(study["components"]["transformer"]["path"])) as model, torch.enable_grad():
            model.eval().requires_grad_(False)
            case = heads.training_input_case(study, current, source, noise_seed, study_sigmas(study)[0])
            grid, video, c0, capture, eps, inputs, manifest = case
            if grid.latent_frames != 18 or not torch.equal(video.latent[:, : grid.tokens_per_latent_frame], c0):
                raise ValueError("native DDP crop or clean condition differs")
            with torch.no_grad():
                native, _ = ltx_adapter.velocity_forward(model, video)
            attached = adapters.attach(model, 42, cfg["rank"])
            ap = adapters.parameters(attached)
            gates = HeadGates(cfg["blocks"], cfg["heads"]).to(current.device)
            live = hooks.attach_head_provider(model, gates.provider)
            gates.phase = "all_one"
            gates.microbatch()
            with torch.no_grad():
                control, _ = ltx_adapter.velocity_forward(model, video)
            if not torch.equal(native, control):
                raise ValueError("native zero-adapter all-one control differs")
            control_record = {
                "rank": runtime.rank,
                "source": source["relative_dir"],
                "training_window": manifest["training_window"],
                "all_one_bit_identical": True,
                "input_manifest_sha256": provenance.file_sha256(inputs / "manifest.json"),
            }
            del native, control, case, grid, video, c0, capture, eps
            model.train()
            model.velocity_model.set_gradient_checkpointing(True)
            ao = heads.optimizer(ap, cfg["adapter_lr"], cfg)
            go = heads.optimizer([gates.alpha], cfg["gate_lr"], cfg)
            training = runtime.wrap_step(TrainingStep(model, gates, cfg, "learned_joint"))
            frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]

            def state_hash():
                return checkpoints.digest(
                    {
                        "alpha": checkpoints.tensor_hash(gates.alpha),
                        "adapters": {k: checkpoints.tensor_hash(v) for k, v in adapters.state_dict(attached).items()},
                    }
                )

            def update(number):
                phase = "soft" if number <= 3 else "binary"
                gates.begin_update(phase, cfg["blocks"] * cfg["heads"] - 152 if phase == "binary" else None)
                ao.zero_grad(set_to_none=True)
                go.zero_grad(set_to_none=True)
                for group in ao.param_groups:
                    group["lr"] = cfg["adapter_lr"] * min(number / cfg["warmup_updates"], 1)
                sigma = study_sigmas(study)[(number + runtime.rank - 1) % len(study_sigmas(study))]
                grid, video, c0, y, eps, inputs, manifest = heads.training_input_case(
                    study, current, source, noise_seed, sigma
                )
                torch.cuda.reset_peak_memory_stats(current.device)
                loss, official, penalty = training(video, eps, y, grid.tokens_per_latent_frame, number, phase, None, 1)
                loss.backward()
                runtime.reduce_gradients(ap + [gates.alpha])
                anorm = float(torch.nn.utils.clip_grad_norm_(ap, 1, error_if_nonfinite=True))
                gnorm = float(torch.nn.utils.clip_grad_norm_([gates.alpha], 1, error_if_nonfinite=True))
                if min(anorm, gnorm) <= 0:
                    raise ValueError("missing native DDP gradients")
                ao.step()
                go.step()
                torch.cuda.synchronize(current.device)
                if any(p._version != version for p, version in frozen):
                    raise ValueError("native DDP changed frozen weights")
                free, _ = torch.cuda.mem_get_info(current.device)
                if free < cfg["headroom_bytes"]:
                    raise MemoryError("native DDP lacks frozen headroom")
                digest = state_hash()
                runtime.agree(digest)
                record = {
                    "rank": runtime.rank,
                    "update": number,
                    "phase": phase,
                    "sigma": sigma,
                    "official": float(official),
                    "penalty": float(penalty),
                    "adapter_gradient_norm": anorm,
                    "gate_gradient_norm": gnorm,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(current.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(current.device),
                    "free_bytes": free,
                    "trainable_state_sha256": digest,
                }
                return record

            rows = [update(1)]
            saved = output / "update_0001.pt"
            runtime.save(
                saved,
                identity=identity,
                update=1,
                phase="soft",
                gates=gates,
                adapters=attached,
                adapter_optimizer=ao,
                gate_optimizer=go,
            )
            second = update(2)
            state = checkpoints.load(saved, identity)
            adapters.load_state(attached, state["adapters"])
            gates.alpha.data.copy_(state["alpha"])
            gates.decision = state["mask"]
            ao.load_state_dict(state["adapter_optimizer"])
            go.load_state_dict(state["gate_optimizer"])
            checkpoints.restore_rng(state["rng"])
            replay = update(2)
            if any(second[key] != replay[key] for key in ("trainable_state_sha256", "official", "penalty")):
                raise ValueError("native DDP resumed update differs")
            rows += [second, update(3), update(4)]
            controls = runtime.gather(control_record)
            updates = runtime.gather(rows)
            live.detach_all()
            if runtime.primary:
                checkpoints.publish_json(
                    root / "checks/feasibility.json",
                    {
                        "study": study["identity"],
                        "passed": True,
                        "method": runtime.feasibility_method,
                        "world_size": runtime.world,
                        "training_window_mode": "uniform_f18",
                        "controls": controls,
                        "updates": updates,
                        "exact_native_resume": True,
                        "elapsed_seconds": time.monotonic() - started,
                        "quality_accepted": False,
                    },
                )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
