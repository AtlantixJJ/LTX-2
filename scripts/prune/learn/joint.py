"""V3 native acceptance and joint selection with explicit G1/G2 review gates.

See doc/learn/joint.md. This owner never starts a historical study.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, provenance, session
from scripts.prune.data import joint_study as ffn_study
from scripts.prune.data import training_windows, whole_clip
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import adapters, checkpoints, heads
from scripts.prune.learn import joint_training as ffn_training
from scripts.prune.learn.joint_training import JointStep


class Runtime:
    def __init__(self, config, *, testing=False):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        expected = config["test_gpu_ids"] if testing else config["gpu_ids"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible != ",".join(map(str, expected)) or self.world != len(expected):
            raise ValueError("FFN launch must use the exact frozen physical GPU list and rank count")
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=self.device)

    @property
    def primary(self):
        return self.rank == 0

    def gather(self, value):
        result = [None] * self.world
        dist.all_gather_object(result, value)
        return result

    def wrap(self, step):
        initial = checkpoints.digest(
            {k: checkpoints.tensor_hash(v) for k, v in adapters.state_dict(step.attached).items()}
        )
        if len(set(self.gather(initial))) != 1:
            raise ValueError("FFN ranks initialized different adapters")
        return DistributedDataParallel(
            step, device_ids=[self.local_rank], broadcast_buffers=False, gradient_as_bucket_view=True
        )

    def rng(self):
        return self.gather(checkpoints.rng_state([self.local_rank]))


def mutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def open_current(study, runtime):
    if (
        study["software"]["torch"] != str(torch.__version__)
        or study["software"]["cuda"] != torch.version.cuda
        or study["software"]["python"] != platform.python_version()
    ):
        raise ValueError("FFN runtime software differs from its frozen study")
    current = session.open_session(
        argparse.Namespace(model="2.5", gpu_id=runtime.local_rank, seed=42),
        script="prune.learn.ffn",
        prompt="",
        transformer_path=Path(study["components"]["transformer"]["path"]),
    )
    if checkpoints.tensor_hash(current.context) != study["text_context"]["tensor_sha256"]:
        raise ValueError("FFN context differs from the frozen study")
    for role, path in (
        ("transformer", current.model.paths.transformer()),
        ("text_encoder", current.model.paths.text_encoder()),
        ("video_vae", current.model.paths.video_vae()),
    ):
        if Path(path).resolve() != Path(study["components"][role]["path"]).resolve():
            raise ValueError("FFN resolved components differ")
    return current


def input_case(study, current, source, seed, sigma, *, crop=False):
    # Hard links retain exact saved CUDA noise without making the new run depend on old output paths.
    key = hashlib.sha256(source["relative_dir"].encode()).hexdigest()
    target = Path(study["root"]) / f"inputs/{key}/seed_{seed}"
    original = Path(study["parent"]["input_root"]) / f"inputs/{key}/seed_{seed}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / f"reuse_{seed}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists() and (original / "manifest.json").is_file():
            temporary = target.with_name(target.name + f".reuse-{os.getpid()}")
            shutil.copytree(original, temporary, copy_function=os.link)
            temporary.rename(target)
    return (heads.training_input_case if crop else heads.input_case)(study, current, source, seed, sigma)


def memory(runtime, minimum):
    torch.cuda.synchronize(runtime.device)
    free, _ = torch.cuda.mem_get_info(runtime.device)
    row = dict(
        rank=runtime.rank,
        allocated=torch.cuda.max_memory_allocated(runtime.device),
        reserved=torch.cuda.max_memory_reserved(runtime.device),
        free=free,
    )
    rows = runtime.gather(row)
    if any(item["free"] < minimum for item in rows):
        raise RuntimeError("joint head/FFN device headroom is below the frozen two-GiB minimum")
    return rows


def save(step, ao, go, study, runtime, update, path, chain):
    rng = runtime.rng()
    if runtime.primary:
        payload = ffn_training.state(
            step, ao, go, study=study["identity"], update=update, world_size=runtime.world, rng=rng, input_chain=chain
        )
        ffn_training.save(path, payload)
    dist.barrier()


def native(study, current, runtime, *, testing):
    cfg, root = study["config"], Path(study["root"])
    directory = root / ("checks/native_test" if testing else "checks/native_formal")
    seed_all(cfg["seed"])
    length = 18 if runtime.rank % 2 == 0 else 28
    source = next(s for s in study["splits"]["learning"] if s["n_latent_frames"] == length)
    noise_seed = 42000 + runtime.rank
    while length > 18 and training_windows.bounds(length, noise_seed)[0] == 0:
        noise_seed += runtime.world
    fixture = dict(
        cfg, policy=dict(interval=2, late=1, earliest=4, limit=6, recovery=1, alignment=128), recovery_updates=1
    )
    if cfg["arm"] == "budget_joint_v3":
        fixture.update(budget_ramp_start=0, budget_ramp_end=1)
    started = time.monotonic()
    with current.transformer() as model, torch.enable_grad():
        model.eval().requires_grad_(False)
        case = input_case(study, current, source, noise_seed, cfg["sigmas"][0], crop=True)
        grid, video, c0, _, _, _, manifest = case
        if grid.latent_frames != 18 or not torch.equal(video.latent[:, : grid.tokens_per_latent_frame], c0):
            raise ValueError("FFN native crop or c0 differs")
        with torch.no_grad():
            original, _ = ltx_adapter.velocity_forward(model, video)
        step = JointStep(model, fixture)
        step.configure(0, all_one=True)
        with torch.no_grad():
            step.masks()
            control, _ = ltx_adapter.velocity_forward(model, video)
        if not torch.equal(original, control):
            raise ValueError("FFN all-one/zero-adapter native parity failed")
        del original, control
        model.train()
        model.velocity_model.set_gradient_checkpointing(True)
        # Exercise returnable zero and permanent zero separately in both families.
        if cfg["arm"] != "budget_joint_v3":
            for gates in (step.heads,) if cfg["arm"] == "heads_clip_reset_v3" else (step.heads, step.ffn):
                gates.family.kept[0, :2] = False
                gates.family.permanent[0, 1] = True
                gates.family.reset_summary(gates.family.kept.float())
        step.policy.reset(step.heads.alpha, step.ffn.alpha, [])
        ao, go = ffn_training.optimizers(step)
        wrapped = runtime.wrap(step)
        # DDP's initial broadcast writes replicas before the first optimizer step.
        frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        if runtime.primary:
            print("FFN native: all-one parity passed; starting checkpointed backward", flush=True)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        first = ffn_training.update(step, wrapped, case, 1, ao, go)
        if not first["adapter_gradient_norm"] > 0 or (
            cfg["arm"] != "heads_clip_reset_v3" and not first["ffn_gradient_norm"] > 0
        ):
            raise ValueError("FFN native score or adapter gradient is zero")
        if not first["head_gradient_norm"] > 0:
            raise ValueError("native joint head gradient is zero")
        averaged = {}
        for name, gates in (("heads", step.heads), ("channels", step.ffn)):
            grad = gates.alpha.grad
            if (
                cfg["arm"] != "budget_joint_v3"
                and (name == "heads" or cfg["arm"] != "heads_clip_reset_v3")
                and (grad[0, 0] == 0 or grad[0, 1] != 0)
            ):
                raise ValueError("native returnable/permanent gradient check failed")
            digests = runtime.gather(checkpoints.tensor_hash(grad))
            if len(set(digests)) != 1:
                raise ValueError("native gate gradients differ between ranks")
            averaged[name] = digests[0]
        save(step, ao, go, study, runtime, 1, directory / "update_0001.pt", "native")
        if runtime.primary:
            print("FFN native: first joint update saved; checking exact resume", flush=True)
        second = ffn_training.update(step, wrapped, case, 2, ao, go)
        uninterrupted = ffn_training.state(
            step,
            ao,
            go,
            study=study["identity"],
            update=2,
            world_size=runtime.world,
            rng=runtime.rng(),
            input_chain="native",
        )
        expected = unchanged_state_hash(uninterrupted)
        del uninterrupted
        payload = ffn_training.read(directory / "update_0001.pt", study=study["identity"], world_size=runtime.world)
        ffn_training.restore(step, ao, go, payload, rank=runtime.rank)
        del payload
        replay = ffn_training.update(step, wrapped, case, 2, ao, go)
        repeated = ffn_training.state(
            step,
            ao,
            go,
            study=study["identity"],
            update=2,
            world_size=runtime.world,
            rng=runtime.rng(),
            input_chain="native",
        )
        if unchanged_state_hash(repeated) != expected or second != replay:
            raise ValueError("joint head/FFN native checkpoint resume is not bit-identical")
        del repeated
        save(step, ao, go, study, runtime, 2, directory / "update_0002.pt", "native")
        # Exact resume crosses a real event; the following disposable tail
        # explicitly freezes its binary result for recovery verification.
        if cfg["arm"] == "budget_joint_v3":
            step.policy.heads.kept = step.heads.decision.bool().cpu()
            step.policy.channels.kept = step.ffn.decision.bool().cpu()
        step.policy.selection_end = 2
        step.policy.reset(step.heads.alpha, step.ffn.alpha, go.items)
        step.policy.align(step.heads.alpha, step.ffn.alpha, go.items)
        frozen_mask = step.policy.channels.kept.clone()
        third = ffn_training.update(step, wrapped, case, 3, ao, go)
        if (
            third["ffn_gradient_norm"] != 0
            or third["head_gradient_norm"] != 0
            or not third["adapter_gradient_norm"] > 0
        ):
            raise ValueError("joint FFN fixed-mask recovery gradients differ")
        if not torch.equal(frozen_mask, step.policy.channels.kept) or any(p._version != v for p, v in frozen):
            raise ValueError("FFN native recovery changed a mask or frozen backbone")
        del case, grid, video, c0, frozen_mask
        # Full F28 sources are a distinct inference-memory check from F18 optimizer crops.
        full = input_case(study, current, source, noise_seed, cfg["sigmas"][-1])
        with torch.no_grad():
            step.masks()
            _, clean = ltx_adapter.velocity_forward(model, full[1])
            clean = common.with_clean_prefix(clean, full[2])
        if not torch.equal(clean[:, : full[0].tokens_per_latent_frame], full[2]):
            raise ValueError("FFN full-source monitor changed c0")
        rows = memory(runtime, cfg["headroom_bytes"])
        evidence = runtime.gather(
            dict(
                rank=runtime.rank,
                source=source["relative_dir"],
                source_frames=length,
                window=manifest.get("training_window"),
                first=first,
                recovery=third,
            )
        )
        if runtime.primary:
            print("FFN native: exact resume and fixed recovery passed", flush=True)
            checkpoints.publish_json(
                directory / "acceptance.json",
                dict(
                    study=study["identity"],
                    passed=True,
                    testing=testing,
                    world_size=runtime.world,
                    physical_gpus=cfg["test_gpu_ids"] if testing else cfg["gpu_ids"],
                    all_one_bit_identical=True,
                    resume_bit_identical=True,
                    fixed_recovery=True,
                    returnable_nonzero_permanent_zero=cfg["arm"] != "budget_joint_v3",
                    averaged_gate_gradients=averaged,
                    code=study["code"]["sha256"],
                    ranks=evidence,
                    memory=rows,
                    seconds=time.monotonic() - started,
                ),
            )
        step.close()
        del full, clean, wrapped, step, ao, go, model, frozen
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def monitor(study, current, runtime, step, update, *, tag=None):
    suffix = f"_{tag}" if tag else ""
    saved_rng = checkpoints.rng_state([runtime.local_rank])
    saved = (step.heads.phase, step.heads.current, step.ffn.current, step.model.training)
    rows = []
    try:
        step.model.eval()
        actors = study["splits"]["validation_actors"][:4]
        sources = [next(s for s in study["splits"]["validation"] if s["actor"] == actor) for actor in actors]
        with torch.no_grad():
            for index, source in enumerate(sources):
                if index % runtime.world != runtime.rank:
                    continue
                for sigma_index in study["config"]["monitor_sigma_indices"]:
                    sigma = study["config"]["sigmas"][sigma_index]
                    grid, video, c0, capture, epsilon, inputs, manifest = input_case(
                        study, current, source, 2042, sigma
                    )
                    step.masks()
                    _, clean = ltx_adapter.velocity_forward(step.model, video)
                    clean = common.with_clean_prefix(clean, c0)
                    prefix = grid.tokens_per_latent_frame
                    mse = float((clean[:, prefix:].float() - capture[:, prefix:].float()).square().mean())
                    path = (
                        Path(study["root"]) / f"monitor/update_{update:04d}{suffix}/{source['actor']}_{sigma_index}.pt"
                    )
                    checkpoints.publish_tensor(path, clean)
                    rows.append(
                        dict(
                            actor=source["actor"],
                            view=source["relative_dir"],
                            sigma=sigma,
                            capture_mse=mse,
                            prediction=str(path),
                            prediction_sha256=provenance.file_sha256(path),
                            input_manifest=str(inputs / "manifest.json"),
                            input_sha256=provenance.file_sha256(inputs / "manifest.json"),
                            noise_sha256=manifest["videos"][0]["artifacts"]["epsilon_sha256"],
                        )
                    )
                    del grid, video, c0, capture, epsilon, clean
        gathered = [row for rank_rows in runtime.gather(rows) for row in rank_rows]
        result = dict(
            study=study["identity"],
            update=update,
            cases=sorted(gathered, key=lambda r: (r["actor"], r["sigma"])),
            capture_mse=sum(row["capture_mse"] for row in gathered) / len(gathered),
        )
        if runtime.primary:
            checkpoints.publish_json(Path(study["root"]) / f"monitor/update_{update:04d}{suffix}.json", result)
        return result
    finally:
        step.heads.phase, step.heads.current, step.ffn.current, train_mode = saved
        step.model.train(train_mode)
        checkpoints.restore_rng(saved_rng)


def verify_consumed(study, update, expected_chain):
    chain = "initial"
    for number in range(1, update + 1):
        row = json.loads((Path(study["root"]) / f"training/updates/{number:04d}.json").read_text())
        for item in row["draws"]:
            path = Path(item["input_manifest"])
            if provenance.file_sha256(path) != item["input_sha256"]:
                raise ValueError("consumed FFN input manifest changed")
            whole_clip.load_epsilon(path.parent, whole_clip.load_manifest(path.parent)["videos"][0])
            training_windows.validate_saved_window(path.parent, item["training_window"])
        chain = checkpoints.digest(dict(previous=chain, draws=row["draws"]))
    if chain != expected_chain:
        raise ValueError("consumed FFN input chain differs from checkpoint")


def train(study, current, runtime, *, max_updates=None, pilot=False):
    cfg, root = study["config"], Path(study["root"])
    if not pilot:
        prereq_record = study.get("prerequisites")
        if not prereq_record:
            raise ValueError("formal V3 training requires pinned P0/P1/P2 acceptance")
        ffn_study.verify_files([prereq_record])
        prerequisite = json.loads(Path(prereq_record["path"]).read_text())
        if not prerequisite.get("passed") or not prerequisite.get("checks") or not all(prerequisite["checks"].values()):
            raise ValueError("formal V3 prerequisites are unresolved")
        ffn_study.verify_files(prerequisite["source_receipts"])
    acceptance_path = root / ("checks/native_test/acceptance.json" if pilot else "checks/native_formal/acceptance.json")
    acceptance = json.loads(acceptance_path.read_text())
    if not (
        acceptance["passed"]
        and acceptance["study"] == study["identity"]
        and acceptance["world_size"] == (3 if pilot else cfg["world_size"])
        and acceptance["physical_gpus"] == (cfg["test_gpu_ids"] if pilot else cfg["gpu_ids"])
    ):
        raise ValueError("formal FFN training requires its exact four-rank native acceptance")
    for review_path in (root / "reviews").glob("G*.json"):
        review = json.loads(review_path.read_text())
        if review.get("triggers"):
            raise ValueError(f"review gate remains pending: {review_path}")
    if pilot and (max_updates is None or max_updates > 50):
        raise ValueError("disposable three-rank pilot is bounded to 50 updates")
    seed_all(cfg["seed"])
    total = cfg["selection_updates"] + cfg["recovery_updates"]
    end = min(total, max_updates) if max_updates is not None else total
    deadline = time.monotonic() + cfg["wall_time_hours"] * 3600
    draws = json.loads(Path(study["draws"]["path"]).read_text())
    sources = {s["relative_dir"]: s for s in study["splits"]["learning"]}
    previous = sorted((root / "training/checkpoints").glob("update_*.pt"))
    payload = ffn_training.read(previous[-1], study=study["identity"], world_size=runtime.world) if previous else None
    start, chain = (payload["update"], payload["input_chain"]) if payload else (0, "initial")
    if start >= end:
        if runtime.primary:
            print(f"FFN already at update {start}; requested endpoint {end}", flush=True)
        return
    if runtime.primary and payload:
        verify_consumed(study, start, chain)
    dist.barrier()
    stopped = [False]
    old_handler = signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    try:
        with current.transformer() as model, torch.enable_grad():
            model.train()
            model.velocity_model.set_gradient_checkpointing(True)
            step = JointStep(model, cfg)
            ao, go = ffn_training.optimizers(step)
            if payload:
                ffn_training.restore(step, ao, go, payload, rank=runtime.rank)
                del payload
            else:
                step.configure(0)
            wrapped = runtime.wrap(step)
            frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
            if not previous:
                # Native monitor uses all-one clipped gates before any update.
                save(step, ao, go, study, runtime, 0, root / "training/checkpoints/update_0000.pt", chain)
                monitor(study, current, runtime, step, 0)
            for number in range(start + 1, end + 1):
                started = time.monotonic()
                draw = draws[(number - 1) * 4 + runtime.rank]
                sigma = cfg["sigmas"][draw["sigma_index"]]
                case = input_case(study, current, sources[draw["view"]], draw["noise_seed"], sigma, crop=True)
                torch.cuda.reset_peak_memory_stats(runtime.device)
                metrics = ffn_training.update(step, wrapped, case, number, ao, go)
                if runtime.primary:
                    gradient_path = root / f"training/gradients/{number:04d}.pt"
                    payload_grad = dict(
                        study=study["identity"],
                        update=number,
                        dtype="float32",
                        units="d_official_velocity_MSE/d_executed_gate",
                        tensors=step.task_gradients,
                        hashes={k: checkpoints.tensor_hash(v) for k, v in step.task_gradients.items()},
                    )
                    ffn_training.publish_record(gradient_path, payload_grad)
                    if step.trace is not None and not step.frozen:
                        ffn_training.publish_record(root / f"training/traces/{number:04d}.pt", step.trace)
                    if step.event:
                        ffn_training.publish_record(root / f"training/events/{number:04d}.pt", step.event)
                if any(p._version != version for p, version in frozen):
                    raise ValueError("joint FFN training changed a frozen backbone tensor")
                inputs, manifest = case[-2:]
                evidence = dict(
                    draw,
                    sigma=sigma,
                    input_manifest=str(inputs / "manifest.json"),
                    input_sha256=provenance.file_sha256(inputs / "manifest.json"),
                    training_window=manifest["training_window"],
                )
                del case
                all_draws, rank_metrics = runtime.gather(evidence), runtime.gather(metrics)
                chain = checkpoints.digest(dict(previous=chain, draws=all_draws))
                mem = memory(runtime, cfg["headroom_bytes"])
                row = dict(
                    study=study["identity"],
                    update=number,
                    phase="recovery" if step.frozen and number > step.policy.selection_end else "selection",
                    draws=all_draws,
                    rank_metrics=rank_metrics,
                    metrics={k: sum(r[k] for r in rank_metrics) / runtime.world for k in metrics},
                    ffn=step.ffn.snapshot(),
                    heads=step.heads.snapshot(),
                    pilot=pilot,
                    projection=step.projection,
                    optimizer_damping=step.optimizer_damping,
                    head_phase=step.heads.phase,
                    memory=mem,
                    seconds=time.monotonic() - started,
                )
                if runtime.primary:
                    path = root / f"training/updates/{number:04d}.json"
                    if path.exists():
                        before = json.loads(path.read_text())
                        scientific = lambda r: {k: v for k, v in r.items() if k not in ("memory", "seconds")}
                        if scientific(before) != scientific(row):
                            raise ValueError("replayed FFN optimizer receipt differs")
                    else:
                        checkpoints.publish_json(path, row)
                    mutable_json(root / "monitoring/status.json", dict(status="running", owner_pid=os.getpid(), **row))
                    print(
                        f"FFN update={number}/{total} phase={row['phase']} loss={row['metrics']['official']:.8f} kept={row['ffn']['retained_per_block'][0]} seconds={row['seconds']:.1f}",
                        flush=True,
                    )
                stop = any(runtime.gather(stopped[0] or time.monotonic() >= deadline))
                review = None
                if step.event and step.policy.recovering:
                    # Preserve the pre-alignment binary result and score it.
                    pre = monitor(study, current, runtime, step, number, tag="pre_alignment")
                    ids = step.policy.align(step.heads.alpha, step.ffn.alpha, go.items)
                    post = monitor(study, current, runtime, step, number, tag="aligned")
                    baseline = json.loads((root / "monitor/update_0000.json").read_text())["capture_mse"]
                    parity_record = study.get("prerequisites", {})
                    parity = Path(parity_record.get("path", root / "checks/prerequisites.json"))
                    prereq = json.loads(parity.read_text()) if parity.is_file() else {}
                    triggers = []
                    if max(pre["capture_mse"], post["capture_mse"]) / baseline > 1.03:
                        triggers.append("monitor_above_three_percent")
                    if not prereq.get("passed", False):
                        triggers.append("prerequisites_unresolved")
                    review = dict(
                        gate="G2",
                        update=number,
                        triggers=triggers,
                        restored=ids,
                        selection_end=number,
                        budget=int(step.policy.heads.kept.sum()) * 704
                        + int(step.policy.channels.kept.sum())
                        - len(ids),
                        pre=pre["capture_mse"],
                        post=post["capture_mse"],
                    )
                    stop = stop or bool(triggers)
                if number == 100:
                    result = monitor(study, current, runtime, step, number)
                    baseline = json.loads((root / "monitor/update_0000.json").read_text())["capture_mse"]
                    triggers = []
                    if (step.policy.channels.kept.sum(1) <= 0.6 * cfg["ffn_width"]).any():
                        triggers.append("block_lost_forty_percent")
                    if runtime.primary:
                        last = [torch.load(root / f"training/events/{i:04d}.pt", weights_only=False) for i in (50, 100)]
                        capped = all(
                            e.get("families", {}).get("channels", {}).get("global_cap_bound", False) for e in last
                        )
                    else:
                        capped = None
                    capped = runtime.gather(capped)[0]
                    if capped:
                        triggers.append("channel_cap_both_events")
                    if result["capture_mse"] / baseline > 1.03:
                        triggers.append("monitor_above_three_percent")
                    review = dict(gate="G1", update=number, triggers=triggers, monitor=result["capture_mse"])
                    stop = stop or bool(triggers)
                if review and runtime.primary:
                    checkpoints.publish_json(root / f"reviews/{review['gate']}_{number:04d}.json", review)
                if (
                    number % cfg["checkpoint_interval"] == 0
                    or number in (cfg["selection_updates"], end)
                    or step.event is not None
                    or stop
                ):
                    save(
                        step,
                        ao,
                        go,
                        study,
                        runtime,
                        number,
                        root / f"training/checkpoints/update_{number:04d}.pt",
                        chain,
                    )
                if pilot and step.event and number == 50:
                    saved_h = step.heads.alpha.detach().clone()
                    saved_c = step.ffn.alpha.detach().clone()
                    saved_hp = step.policy.heads.permanent.clone()
                    saved_cp = step.policy.channels.permanent.clone()
                    with torch.no_grad():
                        step.heads.alpha.copy_(
                            step.event["families"]["heads"]["pre_scores"].to(step.heads.alpha.device)
                        )
                        step.ffn.alpha.copy_(step.event["families"]["channels"]["pre_scores"].to(step.ffn.alpha.device))
                    # Pre-event permanent masks exclude newly permanent removals.
                    for family in (step.policy.heads, step.policy.channels):
                        family.permanent &= family.final.eq(0)
                    monitor(study, current, runtime, step, number, tag="fractional_before_reset")
                    with torch.no_grad():
                        step.heads.alpha.copy_(saved_h)
                        step.ffn.alpha.copy_(saved_c)
                    step.policy.heads.permanent = saved_hp
                    step.policy.channels.permanent = saved_cp
                if (number % cfg["monitor_interval"] == 0 or number == end) and number != 100:
                    monitor(study, current, runtime, step, number)
                if step.policy.final_update == number:
                    total = end = number
                    stop = False
                    break
                if stop:
                    break
            else:
                stop = False
            if runtime.primary:
                status = "complete" if end == total and not stop else "stopped"
                mutable_json(
                    root / "monitoring/owner.json",
                    dict(
                        study=study["identity"],
                        status=status,
                        update=number if start < end else start,
                        total=total,
                        pid=os.getpid(),
                    ),
                )
            step.close()
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--phase", choices=("native", "formal", "train", "pilot"), default="formal")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    if args.testing and args.phase not in ("native", "pilot"):
        raise ValueError("GPUs 4–6 are reserved for disposable native tests")
    if args.phase == "pilot" and not args.testing:
        raise ValueError("pilot uses the three-rank testing device list")
    heads.numerical_policy()
    torch.set_num_threads(4)
    study = ffn_study.load(args.study, verify_inputs=int(os.environ.get("RANK", "0")) == 0)
    runtime = Runtime(study["config"], testing=args.testing)
    current = open_current(study, runtime)
    try:
        if args.phase in ("native", "formal"):
            native(study, current, runtime, testing=args.testing)
        if args.phase in ("train", "formal", "pilot"):
            train(study, current, runtime, max_updates=args.max_updates, pilot=args.phase == "pilot")
    except BaseException as exc:
        if runtime.primary:
            mutable_json(
                Path(study["root"]) / "monitoring/failure.json",
                dict(error=repr(exc), phase=args.phase, pid=os.getpid()),
            )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
