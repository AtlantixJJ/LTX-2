"""Run both real numerical families through the shared owner on four CPU ranks.

The model uses checkpointed native-style attention and FFN boundaries. Only CUDA
placement, memory accounting and full-source monitor rendering are substituted.
Compare an uninterrupted three-update run with a serialized one-plus-two resume.
"""

import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ltx_core.model.transformer.model import X0Model
from scripts.prune.core import records
from scripts.prune.learn import checkpoints, ddp_owner, ffn, joint, visualization
from scripts.prune.tests.test_ffn_learning import Velocity, case, config


class CPURuntime(ddp_owner.Runtime):
    def __init__(self, rank):
        self.rank = self.local_rank = rank
        self.world = 4
        self.device = torch.device("cpu")

    def wrap(self, step):
        wrapped = DistributedDataParallel(step, broadcast_buffers=False)

        def average(_, bucket):
            # Gloo can change floating-point reduction order when DDP rebuilds
            # buckets after startup. Pin rank order for this exact-state CPU
            # control. Separate four-rank tests exercise the stock reducer.
            values = [torch.empty_like(bucket.buffer()) for _ in range(self.world)]
            dist.all_gather(values, bucket.buffer())
            result = values[0].clone()
            for value in values[1:]:
                result.add_(value)
            future = torch.futures.Future()
            future.set_result(result.div_(self.world))
            return future

        wrapped.register_comm_hook(None, average)
        return wrapped

    def rng(self):
        return self.gather(checkpoints.rng_state())


class Current:
    device = torch.device("cpu")

    @contextmanager
    def transformer(self):
        torch.manual_seed(42)
        model = X0Model(Velocity())
        model.velocity_model.set_gradient_checkpointing = lambda enabled: None
        yield model


def worker(rank, store, directory, family_name):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=4)
    runtime = CPURuntime(rank)
    family = ffn if family_name == "ffn" else joint
    cfg = config()
    cfg.update(
        world_size=4,
        gpu_ids=[0, 1, 2, 3],
        test_gpu_ids=[4, 5, 6],
        sigmas=[0.5],
        wall_time_hours=1,
        headroom_bytes=0,
        checkpoint_interval=1,
        monitor_interval=50,
        visualization_interval=2,
    )
    if family is joint:
        cfg.update(
            arm="joint_clip_reset_v3",
            rho=1 / 704,
            policy=dict(interval=2, late=1, earliest=4, limit=6, recovery=2, alignment=4),
        )
    state = copy.copy(
        SimpleNamespace(
            **{
                k: getattr(family.state, k)
                for k in ("JointStep", "optimizers", "update", "state", "save", "read", "restore", "publish_record")
                if hasattr(family.state, k)
            }
        )
    )
    constructor = state.JointStep
    state.JointStep = lambda model, config: constructor(
        model, config, blocks=1, heads=4, **({"allowed": [[2, 4]]} if family is ffn else {})
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(family, "state", state)
        patch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
        patch.setattr(ddp_owner, "memory", lambda *_: [dict(rank=r, free=1) for r in range(4)])
        patch.setattr(ddp_owner, "monitor", lambda *_, **__: None)

        def preview(current, snapshot, destination):
            assert runtime.primary and snapshot["settings"]["update"] == 2
            assert all(not value.requires_grad for value in snapshot["latents"].values())
            torch.rand(5)  # The shared preview owner must restore this RNG draw.
            records.publish_json(destination / "fixture.json", snapshot["settings"])

        patch.setattr(visualization, "_write", preview)

        def inputs(study, current, source, seed, sigma, **kwargs):
            value = list(case(rank=rank, number=seed))
            value[0] = SimpleNamespace(
                tokens_per_latent_frame=1,
                latent_frames=6,
                fps=30,
                unpatchify_block=lambda tokens, frames: tokens.transpose(1, 2).reshape(1, 4, frames, 1, 1),
            )
            value[-2:] = [Path(directory), {"training_window": {"seed": seed}}]
            # The owner appends /manifest.json to its returned input directory.
            return tuple(value)

        patch.setattr(ddp_owner, "input_case", inputs)
        # Input checks have their own real-file tests. This fixture tests cursor,
        # chain, checkpoint and update ownership against actual DDP numerical state.
        patch.setattr(ddp_owner.training_windows, "verify_consumed_inputs", lambda *_, **__: None)
        study = dict(identity="tiny-owner", config=cfg, splits={"learning": [dict(relative_dir="view")]})
        study["draws"] = {"path": str(Path(directory) / "draws.json")}
        prerequisite = Path(directory) / "prerequisites.json"
        study["prerequisites"] = dict(path=str(prerequisite), sha256=ddp_owner.provenance.file_sha256(prerequisite))
        for branch in ("continuous", "resumed"):
            study["root"] = str(Path(directory) / branch)
            if branch == "resumed":
                ddp_owner.train(study, Current(), runtime, family, max_updates=1)
            ddp_owner.train(study, Current(), runtime, family, max_updates=3)
            dist.barrier()
            owner = records.read_json(Path(study["root"]) / "monitoring/owner.json")
            assert owner["status"] == "stopped" and owner["update"] == 3
            if runtime.primary:
                assert (Path(study["root"]) / "training/visualizations/update_0002/fixture.json").is_file()
            # A bounded no-op continuation must not execute a fourth update.
            ddp_owner.train(study, Current(), runtime, family, max_updates=3)
        if rank == 0:
            roots = [Path(directory) / b for b in ("continuous", "resumed")]
            payloads = [
                state.read(r / "training/checkpoints/update_0003.pt", study="tiny-owner", world_size=4) for r in roots
            ]
            assert records.unchanged_state_hash(payloads[0]) == records.unchanged_state_hash(payloads[1])
            assert payloads[0]["draw_cursor"] == 12
            chain = "initial"
            for number in range(1, 4):
                receipts = [records.read_json(r / f"training/updates/{number:04d}.json") for r in roots]
                for receipt in receipts:
                    receipt.pop("seconds")
                    receipt.pop("memory")
                assert records.digest(receipts[0]) == records.digest(receipts[1])
                assert [d["noise_seed"] for d in receipts[0]["draws"]] == list(
                    range((number - 1) * 4 + 1, number * 4 + 1)
                )
                chain = records.digest(dict(previous=chain, draws=receipts[0]["draws"]))
            assert chain == payloads[0]["input_chain"]
        dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("family", ["ffn", "joint"])
def test_shared_owner_three_updates_and_exact_four_rank_resume(tmp_path, family):
    for name, value in (
        ("manifest.json", {}),
        ("prerequisites.json", dict(passed=True, checks={"fixture": True}, source_receipts=[])),
        ("draws.json", [dict(view="view", noise_seed=i + 1, sigma_index=0) for i in range(12)]),
    ):
        records.publish_json(tmp_path / name, value)
    for branch in ("continuous", "resumed"):
        records.publish_json(
            tmp_path / branch / "checks/native_formal/acceptance.json",
            dict(passed=True, study="tiny-owner", world_size=4, physical_gpus=[0, 1, 2, 3]),
        )
    mp.start_processes(
        worker, args=(str(tmp_path / "store"), str(tmp_path), family), nprocs=4, start_method="fork", join=True
    )


def failure_worker(rank, store, root, failure):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=4)
    runtime = CPURuntime(rank)
    with pytest.MonkeyPatch.context() as patch:

        def verify(*args, **kwargs):
            if rank == 1:
                raise FileNotFoundError("rank-local missing epsilon")

        patch.setattr(ddp_owner.training_windows, "verify_consumed_inputs", verify)
        expected = FileNotFoundError if rank == 1 or failure == "receipt" else ValueError
        with pytest.raises(expected):
            ddp_owner.verify_consumed({"root": root}, 0 if failure == "input" else 1, "initial", runtime)
    dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["input", "receipt"])
def test_resume_missing_file_stops_all_ranks(tmp_path, failure):
    mp.start_processes(
        failure_worker, args=(str(tmp_path / "store"), str(tmp_path), failure), nprocs=4, start_method="fork", join=True
    )
