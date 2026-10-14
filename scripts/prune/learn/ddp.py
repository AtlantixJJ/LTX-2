"""Full BF16 replicas with DDP-managed FP32 adapters and head gates."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.distributed import Runtime


class DDPRuntime(Runtime):
    uses_ddp = True
    feasibility_method = "uniform_f18_ddp"

    def __init__(self, config):
        protocol.validate(config)
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        if self.world != protocol.draws_per_update(config) or not 0 <= self.rank < self.world:
            raise ValueError("DDP rank count differs from frozen batch")
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=self.device)

    def wrap_step(self, step):
        params = [p for p in step.parameters() if p.requires_grad]
        self.agree(checkpoints.digest({str(i): checkpoints.tensor_hash(p) for i, p in enumerate(params)}))
        return DistributedDataParallel(
            step,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    def reduce_gradients(self, parameters):
        # DDP has already averaged these gradients during backward. Do not reduce twice.
        if self.any(any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters)):
            raise ValueError("missing or nonfinite DDP gradient")

    def draws(self, rows):
        records = self.gather(rows)
        if any(len(rank_rows) != 1 for rank_rows in records):
            raise ValueError("DDP requires exactly one consumed draw per rank")
        return [row for rank_rows in records for row in rank_rows]

    def save(self, path, **kwargs):
        super().save(path, **kwargs, draws_per_update=self.world)
