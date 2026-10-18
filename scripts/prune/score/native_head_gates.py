"""Collect native gates on the frozen F18 training stream using physical GPUs 4-6."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from scripts.onestep_avatar.corpus.dataset import load_training_master
from scripts.prune.core import ltx_adapter, preflight, session
from scripts.prune.core.provenance import file_sha256
from scripts.prune.data import native_gate_study, training_windows, whole_clip
from scripts.prune.learn.checkpoints import publish_json, tensor_hash
from scripts.prune.score.native_gate_accumulator import Accumulator, atomic_json, digest
from scripts.prune.score.native_gate_observer import NativeGateObserver


def current_for(study: dict, gpu: int) -> session.Session:
    if gpu not in study["gpus"]:
        raise ValueError("native gate collection is authorized only on physical GPUs 4-6")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(gpu)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    software = study["source"]["software"]
    if (
        software["torch"] != str(torch.__version__)
        or software["cuda"] != torch.version.cuda
        or software["python"] != platform.python_version()
    ):
        raise ValueError("native collection software differs from the training input authority")
    paths = study["components"]
    model = preflight.check(
        "2.5",
        gpu_id=gpu,
        min_free_gb=44,
        transformer_path=paths["transformer"]["path"],
        text_encoder_path=paths["text_encoder"]["path"],
        video_vae_path=paths["video_vae"]["path"],
    )
    if not model.caps.apply_gated_attention or (model.caps.num_layers, model.caps.num_heads) != (48, 32):
        raise ValueError("native gate model metadata differs")
    if file_sha256(study["context"]["path"]) != study["context"]["bytes_sha256"]:
        raise ValueError("native context bytes changed")
    context = torch.load(study["context"]["path"], map_location="cpu", weights_only=True)
    if tensor_hash(context) != study["context"]["tensor_sha256"]:
        raise ValueError("native context tensor changed")
    return session.Session(
        model, torch.device(f"cuda:{gpu}"), "prune.score.native_head_gates", context.to(f"cuda:{gpu}")
    )


def input_case(study: dict, current: session.Session, draw: dict) -> tuple:
    source = study["source"]
    member = next(s for s in source["splits"]["learning"] if s["relative_dir"] == draw["view"])
    original = native_gate_study.saved_input_root(source, draw["view"], draw["noise_seed"])
    key = hashlib.sha256(draw["view"].encode()).hexdigest()
    target = Path(study["root"]) / f"inputs/{key}/seed_{draw['noise_seed']}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / f"seed_{draw['noise_seed']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists() and (original / "manifest.json").exists():
            temporary = target.with_name(f"{target.name}.reuse-{os.getpid()}")
            shutil.copytree(original, temporary, copy_function=os.link)
            temporary.rename(target)
        manifest = whole_clip.prepare_score_inputs(
            target,
            member,
            current=current,
            sigmas=source["config"]["sigmas"],
            seed=draw["noise_seed"],
        )
    sigma = source["config"]["sigmas"][draw["sigma_index"]]
    grid, video, c0, row = whole_clip.build_input(
        target, manifest, view=member["resolved_dir"], sigma=sigma, current=current
    )
    capture, _ = load_training_master(Path(member["capture"]))
    capture = grid.patchify(capture.unsqueeze(0).to(current.device))
    epsilon = whole_clip.load_epsilon(target, row).to(current.device)
    case = training_windows.crop_case(
        (grid, video, c0, capture, epsilon, target, manifest), current, draw["noise_seed"], sigma
    )
    receipt = {
        "draw_index": draw["microbatch"],
        "step": draw["microbatch"] // 4 + 1,
        "logical_rank": draw["microbatch"] % 4,
        "sigma_index": study["source_sigma_indices"].index(draw["sigma_index"]),
        "source_sigma_index": draw["sigma_index"],
        "sigma": sigma,
        "view": draw["view"],
        "actor": member["actor"],
        "noise_seed": draw["noise_seed"],
        "frames": 18,
        "tokens_per_frame": case[0].tokens_per_latent_frame,
        "input_manifest": str(target / "manifest.json"),
        "input_sha256": file_sha256(target / "manifest.json"),
        "epsilon_sha256": row["artifacts"]["epsilon_sha256"],
        "capture_sha256": member["capture_latent_sha256"],
        "context_sha256": study["context"]["tensor_sha256"],
        "training_window": case[-1]["training_window"],
        "input_tensor_sha256": tensor_hash(case[1].latent),
    }
    # Compare exact consumed training evidence, including nonzero-start crop coordinates.
    historical = Path(source["root"]) / f"training/updates/{receipt['step']:04d}.json"
    if historical.exists():
        row = json.loads(historical.read_text())["draws"][receipt["logical_rank"]]
        if any(row[k] != draw[k] for k in draw) or row["training_window"] != receipt["training_window"]:
            raise ValueError("native observation does not reproduce the historical training draw/crop")
        if row["input_sha256"] != receipt["input_sha256"]:
            raise ValueError("native observation input manifest differs from training")
        receipt["historical_training_match"] = True
    else:
        receipt["historical_training_match"] = False
    return case, receipt


def weight_hash(model: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor_hash(tensor).encode())
    return result.hexdigest()


def acceptance(
    study: dict, current: session.Session, model: torch.nn.Module, observer: NativeGateObserver, draws: list
) -> None:
    members = {s["relative_dir"]: s for s in study["source"]["splits"]["learning"]}
    chosen = next(
        d
        for d in draws
        if d["sigma_index"] == 4
        and training_windows.bounds(members[d["view"]]["n_latent_frames"], d["noise_seed"])[0] > 0
    )
    started = time.monotonic()
    case, receipt = input_case(study, current, chosen)
    grid, video = case[:2]
    before = weight_hash(model)
    unobserved = ltx_adapter.velocity_forward(model, video)
    observer.begin(grid.latent_frames, grid.tokens_per_latent_frame)
    observed = ltx_adapter.velocity_forward(model, video)
    values = observer.finish()
    if not all(torch.equal(a, b) for a, b in zip(unobserved, observed, strict=True)):
        raise ValueError("native observer changes model output")
    if weight_hash(model) != before or tensor_hash(video.latent) != receipt["input_tensor_sha256"]:
        raise ValueError("native observer changes frozen inputs or weights")
    if values["histogram"].shape != (48, 32, 2048):
        raise ValueError("native acceptance histogram geometry differs")
    publish_json(
        Path(study["root"]) / f"provenance/acceptance_gpu_{current.device.index}.json",
        {
            "study": study["identity"],
            "passed": True,
            "gpu": current.device.index,
            "draw": receipt,
            "bit_identical_velocity": True,
            "bit_identical_clean": True,
            "max_abs": 0.0,
            "unchanged_weight_sha256": before,
            "modules": 48,
            "heads": 1536,
            "seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(current.device),
        },
    )


def worker(study: dict, gpu: int) -> None:  # noqa: PLR0912, PLR0915
    # The checkpoint transaction retains input, observation and publication in one ordered owner.
    root = Path(study["root"])
    slot = study["gpus"].index(gpu)
    draws = json.loads((root / "configs/draws.json").read_text())
    current = current_for(study, gpu)
    status = root / f"monitoring/gpu_{gpu}.json"
    atomic_json(status, {"status": "loading_model", "gpu": gpu, "pid": os.getpid(), "updated": time.time()})
    with current.transformer() as model, torch.inference_mode():
        model.eval().requires_grad_(False)
        versions = [(p, p._version) for p in model.parameters()]
        with NativeGateObserver(model) as observer:
            if not (root / f"provenance/acceptance_gpu_{gpu}.json").exists():
                acceptance(study, current, model, observer, draws)
            for step in range(100, study["steps"] + 1, 100):
                final = root / f"workers/gpu_{gpu}/step_{step:04d}.npz"
                expected = [
                    i
                    for i in range((step - 100) * 4, step * 4)
                    if i % len(study["gpus"]) == slot and draws[i]["sigma_index"] in study["source_sigma_indices"]
                ]
                if final.exists():
                    saved = Accumulator.load(final, study["identity"])
                    if [r["draw_index"] for r in saved.receipts] != expected:
                        raise ValueError("completed worker shard draw identities differ")
                    continue
                partial = final.with_name(f"step_{step:04d}.partial.npz")
                acc = (
                    Accumulator.load(partial, study["identity"])
                    if partial.exists()
                    else Accumulator(len(study["sigmas"]), study["identity"])
                )
                seen = {r["draw_index"] for r in acc.receipts}
                if not seen <= set(expected):
                    raise ValueError("partial worker state contains out-of-window draws")
                try:
                    for index in expected:
                        if index in seen:
                            continue
                        started = time.monotonic()
                        case, receipt = input_case(study, current, draws[index])
                        grid, video = case[:2]
                        observer.begin(grid.latent_frames, grid.tokens_per_latent_frame)
                        velocity, clean = ltx_adapter.velocity_forward(model, video)
                        values = observer.finish()
                        if not torch.isfinite(velocity).all() or not torch.isfinite(clean).all():
                            raise ValueError("nonfinite native output")
                        if any(p._version != version for p, version in versions):
                            raise ValueError("native observation changed frozen parameter versions")
                        payload = hashlib.sha256()
                        for name in sorted(values):
                            payload.update(name.encode())
                            payload.update(values[name].tobytes())
                        receipt["statistics_sha256"] = payload.hexdigest()
                        publish_json(root / f"draws/{index:05d}.json", receipt)
                        acc.add(index, receipt["sigma_index"], values, receipt)
                        if len(acc.receipts) % 20 == 0:
                            acc.save(partial)
                        progress = {
                            "status": "collecting",
                            "gpu": gpu,
                            "pid": os.getpid(),
                            "updated": time.time(),
                            "draw_index": index,
                            "logical_step": index // 4 + 1,
                            "window_endpoint": step,
                            "completed_in_window": len(acc.receipts),
                            "seconds": time.monotonic() - started,
                        }
                        atomic_json(status, progress)
                        print(json.dumps(progress), flush=True)
                        del case, grid, video, velocity, clean, values
                    acc.save(final)
                    partial.unlink(missing_ok=True)
                    publish_json(
                        final.with_suffix(".json"),
                        {
                            "study": study["identity"],
                            "sha256": file_sha256(final),
                            "indices": expected,
                            "step": step,
                            "gpu": gpu,
                        },
                    )
                except BaseException:
                    if acc.receipts:
                        acc.save(partial)
                    raise
    atomic_json(status, {"status": "complete", "gpu": gpu, "pid": os.getpid(), "updated": time.time()})


def complete_window(study: dict, step: int) -> bool:
    root = Path(study["root"])
    return all((root / f"workers/gpu_{gpu}/step_{step:04d}.json").exists() for gpu in study["gpus"])


def commit_window(study: dict, step: int, cumulative: Accumulator) -> None:
    root = Path(study["root"])
    window = Accumulator(len(study["sigmas"]), study["identity"])
    for gpu in study["gpus"]:
        path = root / f"workers/gpu_{gpu}/step_{step:04d}.npz"
        pin = json.loads(path.with_suffix(".json").read_text())
        if pin["study"] != study["identity"] or file_sha256(path) != pin["sha256"]:
            raise ValueError("worker milestone bytes changed")
        window.merge(Accumulator.load(path, study["identity"]))
    window.require_indices([i for i in study["selected_draw_indices"] if (step - 100) * 4 <= i < step * 4])
    if not np.all(window.counts == 40):
        raise ValueError("milestone sigma coverage is not balanced")
    cumulative.merge(window)
    cumulative.require_indices([i for i in study["selected_draw_indices"] if i < step * 4])
    directory = root / f"milestones/step_{step:04d}"
    window.save(directory / "window.npz")
    cumulative.save(directory / "cumulative.npz")
    publish_json(
        directory / "complete.json",
        {
            "study": study["identity"],
            "step": step,
            "draws": len(cumulative.receipts),
            "scheduled_draws": step * 4,
            "skipped_draws": step * 4 - len(cumulative.receipts),
            "per_sigma": cumulative.counts.tolist(),
            "cumulative_sha256": file_sha256(directory / "cumulative.npz"),
            "window_sha256": file_sha256(directory / "window.npz"),
            "receipt_chain_sha256": digest(cumulative.receipts),
        },
    )


def supervise(study: dict) -> None:  # noqa: PLR0912, PLR0915
    # This bounded lifecycle owns only its children and the exact milestone commit sequence.
    from scripts.prune.score.native_gate_analysis import analyze, build_report  # noqa: PLC0415 -- CPU owner only

    root = Path(study["root"])
    handles, processes = [], []
    started = time.time()
    try:
        for gpu in study["gpus"]:
            log = root / f"logs/gpu_{gpu}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a")
            command = [
                sys.executable,
                "-m",
                "scripts.prune.score.native_head_gates",
                "--phase",
                "worker",
                "--study",
                str(root / "configs/frozen_study.json"),
                "--gpu-id",
                str(gpu),
            ]
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ | {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"},
            )
            handles.append(handle)
            processes.append(process)
        atomic_json(
            root / "monitoring/owner.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "started": started,
                "study": study["identity"],
                "children": [{"gpu": g, "pid": p.pid} for g, p in zip(study["gpus"], processes, strict=True)],
            },
        )
        # Create the sketch after launch, before the first statistics milestone.
        build_report(root)
        cumulative = Accumulator(len(study["sigmas"]), study["identity"])
        step = 100
        while step <= study["steps"]:
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"native observation worker failed: {[p.pid for p in failed]}")
            if complete_window(study, step):
                path = root / f"milestones/step_{step:04d}/complete.json"
                if path.exists():
                    pin = json.loads(path.read_text())
                    data = path.with_name("cumulative.npz")
                    if pin["study"] != study["identity"] or file_sha256(data) != pin["cumulative_sha256"]:
                        raise ValueError("completed milestone changed on resume")
                    cumulative = Accumulator.load(data, study["identity"])
                    cumulative.require_indices([i for i in study["selected_draw_indices"] if i < step * 4])
                else:
                    commit_window(study, step, cumulative)
                analyze(root, step)
                build_report(root)
                print(f"Validated and reported milestone {step}, {step * 160 // 100} observed draws", flush=True)
                step += 100
            else:
                progress = []
                for gpu in study["gpus"]:
                    path = root / f"monitoring/gpu_{gpu}.json"
                    if path.exists():
                        row = json.loads(path.read_text())
                        progress.append(row)
                        if time.time() - row["updated"] > 1800 and row["status"] != "complete":
                            raise TimeoutError(f"GPU {gpu} made no recorded progress for 30 minutes")
                atomic_json(
                    root / "monitoring/status.json",
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "updated": time.time(),
                        "validated_step": step - 100,
                        "validated_draws": (step - 100) * 160 // 100,
                        "scheduled_draws": (step - 100) * 4,
                        "next_milestone": step,
                        "workers": progress,
                    },
                )
                if all(p.poll() == 0 for p in processes):
                    raise RuntimeError("workers ended without the next complete milestone")
                build_report(root)
                time.sleep(20)
        for process in processes:
            process.wait(timeout=60)
        atomic_json(
            root / "monitoring/status.json",
            {
                "status": "complete",
                "validated_step": study["steps"],
                "validated_draws": study["steps"] * 160 // 100,
                "scheduled_draws": study["steps"] * 4,
                "updated": time.time(),
                "seconds": time.time() - started,
            },
        )
        build_report(root)
    except BaseException:
        atomic_json(root / "monitoring/failure.json", {"error": traceback.format_exc(), "updated": time.time()})
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "run", "worker", "report"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--combined", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-id", type=int, choices=(4, 5, 6))
    parser.add_argument("--steps", type=int, default=1300)
    args = parser.parse_args()
    if args.phase == "prepare":
        if any(getattr(args, name) is None for name in ("source", "combined", "plan", "output")):
            parser.error("prepare needs --source, --combined, --plan and --output")
        prepared = native_gate_study.prepare(args.source, args.combined, args.plan, args.output, args.steps)
        print(f"Prepared native study {prepared['identity']}", flush=True)
        return
    if args.study is None:
        parser.error("this phase needs --study")
    if args.phase == "report":
        from scripts.prune.score.native_gate_analysis import build_report  # noqa: PLC0415 -- CPU report phase

        build_report(args.study.resolve().parent.parent)
        return
    study = native_gate_study.load(args.study)
    if args.phase == "worker":
        if args.gpu_id is None:
            parser.error("worker needs --gpu-id")
        try:
            worker(study, args.gpu_id)
        except BaseException:
            atomic_json(
                Path(study["root"]) / f"monitoring/failure_gpu_{args.gpu_id}.json",
                {"error": traceback.format_exc(), "updated": time.time()},
            )
            raise
    else:
        supervise(study)


if __name__ == "__main__":
    main()
