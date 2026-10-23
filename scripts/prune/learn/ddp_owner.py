"""Share native acceptance, monitoring and distributed training across mask families.

FFN, joint clipped selection and compact recovery supply numerical owners and
family hooks. This module owns exact frozen device/rank validation, reused
inputs, consumed-draw checks, update publication and checkpoint/monitor order.
Release dead CUDA cache blocks before checking configured physical headroom.
See doc/learn/ddp_owner.md for hook order and resume/failure semantics."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
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
from scripts.prune.core import ltx_adapter, provenance, records, session
from scripts.prune.core.records import unchanged_state_hash
from scripts.prune.data import training_windows
from scripts.prune.learn import adapters, checkpoints, heads, visualization


class Runtime:
    """Exact frozen device list, one NCCL group and small CPU record collectives."""

    def __init__(self, config, *, testing=False):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        expected = config["test_gpu_ids"] if testing else config["gpu_ids"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible != ",".join(map(str, expected)) or self.world != len(expected):
            raise ValueError("launch must use the exact frozen physical GPU list and rank count")
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=self.device)
        if self.any(torch.cuda.mem_get_info(self.device)[0] < 44 * 2**30):
            raise ValueError("each frozen DDP device needs at least 44 GiB free before model loading")

    @property
    def primary(self):
        return self.rank == 0

    def gather(self, value):
        result = [None] * self.world
        dist.all_gather_object(result, value)
        return result

    def any(self, value):
        return any(self.gather(bool(value)))

    def agree(self, value):
        if len(set(self.gather(value))) != 1:
            raise ValueError("distributed ranks disagree on state")

    def wrap(self, step):
        initial = records.digest({k: records.tensor_hash(v) for k, v in adapters.state_dict(step.attached).items()})
        if len(set(self.gather(initial))) != 1:
            raise ValueError("ranks initialized different adapters")
        return DistributedDataParallel(
            step, device_ids=[self.local_rank], broadcast_buffers=False, gradient_as_bucket_view=True
        )

    def rng(self):
        return self.gather(checkpoints.rng_state([self.local_rank]))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def open_current(study, runtime, *, script):
    return session.open_frozen(study, runtime.local_rank, script=script)


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
    # Release dead cached blocks before checking physical device headroom. Without
    # this, a no-grad full-source forward leaves its activations reserved and
    # `mem_get_info` reports the allocator cache as used memory.
    torch.cuda.empty_cache()
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


def save(family, step, ao, go, study, runtime, update, path, chain):
    rng = runtime.rng()
    if runtime.primary:
        payload = family.state.state(
            step, ao, go, study=study["identity"], update=update, world_size=runtime.world, rng=rng, input_chain=chain
        )
        family.state.save(path, payload)
    dist.barrier()


def monitor(study, current, runtime, step, update, *, tag=None):
    """Score four fixed validation actors without changing RNG, gate state or train mode."""
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
                    records.publish_tensor(path, clean)
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
            records.publish_json(Path(study["root"]) / f"monitor/update_{update:04d}{suffix}.json", result)
        return result
    finally:
        step.heads.phase, step.heads.current, step.ffn.current, train_mode = saved
        step.model.train(train_mode)
        checkpoints.restore_rng(saved_rng)


def verify_consumed(study, update, expected_chain, runtime=None):
    """Replay the input chain through `update` and check every consumed input once.

    With a runtime, each rank checks a disjoint share and any failure stops all ranks.
    """
    failure = None
    try:
        chain, consumed = "initial", []
        for number in range(1, update + 1):
            row = json.loads((Path(study["root"]) / f"training/updates/{number:04d}.json").read_text())
            consumed += row["draws"]
            chain = records.digest(dict(previous=chain, draws=row["draws"]))
        if chain != expected_chain:
            raise ValueError("consumed input chain differs from checkpoint")
        share = consumed[runtime.rank :: runtime.world] if runtime else consumed
        training_windows.verify_consumed_inputs(share, manifest_key="input_sha256")
    except Exception as error:
        # Missing files and malformed receipts must join the same failure vote.
        # Otherwise healthy ranks wait in a collective until the NCCL timeout.
        failure = error
    if runtime.any(failure is not None) if runtime else failure is not None:
        raise failure or ValueError("another rank found a changed consumed input")


def _state_digest(state, step, ao, go, study, runtime):
    payload = state.state(
        step,
        ao,
        go,
        study=study["identity"],
        update=2,
        world_size=runtime.world,
        rng=runtime.rng(),
        input_chain="native",
    )
    return unchanged_state_hash(payload)


def native(study, current, runtime, family, *, testing):
    """Bounded native acceptance: all-one parity, gradients, exact resume and fixed-mask recovery."""
    cfg, root = study["config"], Path(study["root"])
    state = family.state
    directory = root / ("checks/native_test" if testing else "checks/native_formal")
    seed_all(cfg["seed"])
    length = 18 if runtime.rank % 2 == 0 else 28
    source = next(s for s in study["splits"]["learning"] if s["n_latent_frames"] == length)
    noise_seed = training_windows.nonzero_crop_seed(length, 42000 + runtime.rank, runtime.world)
    fixture = family.native_fixture(cfg)
    started = time.monotonic()
    with current.transformer() as model, torch.enable_grad():
        model.eval().requires_grad_(False)
        case = input_case(study, current, source, noise_seed, cfg["sigmas"][0], crop=True)
        grid, video, c0, _, _, _, manifest = case
        if grid.latent_frames != 18 or not torch.equal(video.latent[:, : grid.tokens_per_latent_frame], c0):
            raise ValueError("native crop or c0 differs")
        with torch.no_grad():
            original, _ = ltx_adapter.velocity_forward(model, video)
        step = state.JointStep(model, fixture)
        step.configure(0, all_one=True)
        with torch.no_grad():
            step.masks()
            control, _ = ltx_adapter.velocity_forward(model, video)
        if not torch.equal(original, control):
            raise ValueError("all-one/zero-adapter native parity failed")
        del original, control
        model.train()
        model.velocity_model.set_gradient_checkpointing(True)
        family.native_prepare(step, cfg)
        ao, go = state.optimizers(step)
        wrapped = runtime.wrap(step)
        # DDP's initial broadcast writes replicas before the first optimizer step.
        frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        if runtime.primary:
            print("native: all-one parity passed; starting checkpointed backward", flush=True)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        first = state.update(step, wrapped, case, 1, ao, go)
        extra = family.native_first_checks(step, first, cfg, runtime)
        save(family, step, ao, go, study, runtime, 1, directory / "update_0001.pt", "native")
        if runtime.primary:
            print("native: first update saved; checking exact resume", flush=True)

        second = state.update(step, wrapped, case, 2, ao, go)
        expected = _state_digest(state, step, ao, go, study, runtime)
        payload = state.read(directory / "update_0001.pt", study=study["identity"], world_size=runtime.world)
        state.restore(step, ao, go, payload, rank=runtime.rank)
        del payload
        replay = state.update(step, wrapped, case, 2, ao, go)
        if _state_digest(state, step, ao, go, study, runtime) != expected or second != replay:
            raise ValueError("native checkpoint resume is not bit-identical")
        save(family, step, ao, go, study, runtime, 2, directory / "update_0002.pt", "native")
        fixed_mask = family.native_freeze(step, go)
        frozen_mask = fixed_mask().clone()
        third = state.update(step, wrapped, case, 3, ao, go)
        if (
            third["ffn_gradient_norm"] != 0
            or third["head_gradient_norm"] != 0
            or not third["adapter_gradient_norm"] > 0
        ):
            raise ValueError("fixed-mask recovery gradients differ")
        if not torch.equal(frozen_mask, fixed_mask()) or any(p._version != v for p, v in frozen):
            raise ValueError("native recovery changed a mask or frozen backbone")
        del case, grid, video, c0, frozen_mask
        # Full F28 sources are a distinct inference-memory check from F18 optimizer crops.
        full = input_case(study, current, source, noise_seed, cfg["sigmas"][-1])
        with torch.no_grad():
            step.masks()
            _, clean = ltx_adapter.velocity_forward(model, full[1])
            clean = common.with_clean_prefix(clean, full[2])
        if not torch.equal(clean[:, : full[0].tokens_per_latent_frame], full[2]):
            raise ValueError("full-source monitor changed c0")
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
            print("native: exact resume and fixed recovery passed", flush=True)
            records.publish_json(
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
                    code=study["code"]["sha256"],
                    ranks=evidence,
                    memory=rows,
                    seconds=time.monotonic() - started,
                    **extra,
                ),
            )
        step.close()
        del full, clean, wrapped, step, ao, go, model, frozen
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


class Context:
    """State one update loop shares with its family hooks."""

    def __init__(self, **values):
        self.__dict__.update(values)


def train(study, current, runtime, family, *, max_updates=None, pilot=False):
    cfg, root = study["config"], Path(study["root"])
    state = family.state
    family.check_start(study, root, cfg, pilot=pilot, max_updates=max_updates)
    acceptance = json.loads(
        (root / ("checks/native_test" if pilot else "checks/native_formal") / "acceptance.json").read_text()
    )
    if not (
        acceptance["passed"]
        and acceptance["study"] == study["identity"]
        and acceptance["world_size"] == (len(cfg["test_gpu_ids"]) if pilot else cfg["world_size"])
        and acceptance["physical_gpus"] == (cfg["test_gpu_ids"] if pilot else cfg["gpu_ids"])
    ):
        raise ValueError("training requires its exact native acceptance")
    seed_all(cfg["seed"])
    total = cfg["selection_updates"] + cfg["recovery_updates"]
    end = min(total, max_updates) if max_updates is not None else total
    deadline = time.monotonic() + cfg["wall_time_hours"] * 3600
    draws = json.loads(Path(study["draws"]["path"]).read_text())
    # The frozen batch is world_size draws; a smaller pilot consumes the first ranks' draws.
    stride = cfg["world_size"]
    sources = {s["relative_dir"]: s for s in study["splits"]["learning"]}
    previous = sorted((root / "training/checkpoints").glob("update_*.pt"))
    payload = state.read(previous[-1], study=study["identity"], world_size=runtime.world) if previous else None
    start, chain = (payload["update"], payload["input_chain"]) if payload else (0, "initial")
    if start >= end:
        if runtime.primary:
            print(f"already at update {start}; requested endpoint {end}", flush=True)
        return
    if payload:
        verify_consumed(study, start, chain, runtime)
    dist.barrier()
    stopped = [False]
    old_handler = signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    try:
        with current.transformer() as model, torch.enable_grad():
            model.train()
            model.velocity_model.set_gradient_checkpointing(True)
            step = state.JointStep(model, cfg)
            ao, go = state.optimizers(step)
            if payload:
                state.restore(step, ao, go, payload, rank=runtime.rank)
                del payload
            else:
                step.configure(0)
            wrapped = runtime.wrap(step)
            frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
            ctx = Context(
                study=study, current=current, runtime=runtime, step=step, root=root, cfg=cfg, go=go, pilot=pilot
            )
            if not previous:
                save(family, step, ao, go, study, runtime, 0, root / "training/checkpoints/update_0000.pt", chain)
                monitor(study, current, runtime, step, 0)
            for number in range(start + 1, end + 1):
                started = time.monotonic()
                draw = draws[(number - 1) * stride + runtime.rank]
                sigma = cfg["sigmas"][draw["sigma_index"]]
                case = input_case(study, current, sources[draw["view"]], draw["noise_seed"], sigma, crop=True)
                torch.cuda.reset_peak_memory_stats(runtime.device)
                visualize = visualization.arm(step, cfg, number, primary=runtime.primary)
                metrics = state.update(step, wrapped, case, number, ao, go)
                preview = visualization.snapshot(step, case, study=study["identity"], update=number, **draw, sigma=sigma)
                if runtime.primary:
                    family.after_update(ctx, number)
                if any(p._version != version for p, version in frozen):
                    raise ValueError("training changed a frozen backbone tensor")
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
                chain = records.digest(dict(previous=chain, draws=all_draws))
                mem = memory(runtime, cfg["headroom_bytes"])
                row = dict(
                    study=study["identity"],
                    update=number,
                    draws=all_draws,
                    rank_metrics=rank_metrics,
                    metrics={k: sum(r[k] for r in rank_metrics) / runtime.world for k in metrics},
                    memory=mem,
                    seconds=time.monotonic() - started,
                    **family.receipt_fields(ctx, number),
                )
                if runtime.primary:
                    path = root / f"training/updates/{number:04d}.json"
                    if path.exists():
                        before = json.loads(path.read_text())

                        def scientific(r):
                            return {k: v for k, v in r.items() if k not in ("memory", "seconds")}

                        if scientific(before) != scientific(row):
                            raise ValueError("replayed optimizer receipt differs")
                    else:
                        records.publish_json(path, row)
                    records.write_json(
                        root / "monitoring/status.json", dict(status="running", owner_pid=os.getpid(), **row)
                    )
                    print(
                        f"{family.label} update={number}/{total} phase={row['phase']} loss={row['metrics']['official']:.8f} "
                        f"kept={row['ffn']['retained_per_block'][0]} seconds={row['seconds']:.1f}",
                        flush=True,
                    )
                stop = any(runtime.gather(stopped[0] or time.monotonic() >= deadline))
                stop = family.reviews(ctx, number) or stop
                if number % cfg["checkpoint_interval"] == 0 or family.checkpoint_due(ctx, number, end) or stop:
                    save(
                        family,
                        step,
                        ao,
                        go,
                        study,
                        runtime,
                        number,
                        root / f"training/checkpoints/update_{number:04d}.pt",
                        chain,
                    )
                family.after_checkpoint(ctx, number)
                if visualize:
                    visualization.save(
                        current, preview, root / f"training/visualizations/update_{number:04d}", runtime=runtime
                    )
                if family.monitor_due(ctx, number, end):
                    monitor(study, current, runtime, step, number)
                if family.final_update(ctx) == number:
                    total = end = number
                    stop = False
                    break
                if stop:
                    break
            else:
                stop = False
            if runtime.primary:
                status = "complete" if end == total and not stop else "stopped"
                records.write_json(
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


def run(family, args, load_study):
    """Shared CLI flow: numerical policy, frozen study, runtime, then native and/or training phases."""
    from scripts.prune.core import numerics

    numerics.deterministic_policy()
    torch.set_num_threads(4)
    study = load_study(args.study, verify_inputs=int(os.environ.get("RANK", "0")) == 0)
    runtime = Runtime(study["config"], testing=args.testing)
    current = open_current(study, runtime, script=family.script)
    try:
        if args.phase in ("native", "formal"):
            native(study, current, runtime, family, testing=args.testing)
        if args.phase in ("train", "formal", "pilot"):
            train(study, current, runtime, family, max_updates=args.max_updates, pilot=args.phase == "pilot")
    except BaseException as exc:
        if runtime.primary:
            records.write_json(
                Path(study["root"]) / "monitoring/failure.json",
                dict(error=repr(exc), phase=args.phase, pid=os.getpid()),
            )
        raise
    finally:
        dist.destroy_process_group()
