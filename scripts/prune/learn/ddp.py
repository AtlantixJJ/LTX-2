"""Full BF16 replicas with DDP-managed FP32 adapters and head gates.

Each rank holds a complete model and consumes one frozen draw per update.
DDP averages gradients once during backward; `reduce_gradients` only checks them.
`gather`, `any` and `agree` coordinate small CPU records across ranks.
`save` gathers every rank's RNG and publishes one checkpoint from rank zero.
`Session` delegates to the native session and adds the runtime as `distributed`.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.prune.core import records
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.ddp_owner import Runtime


class DDPRuntime(Runtime):
    uses_ddp = True
    feasibility_method = "uniform_f18_ddp"

    def __init__(self, config):
        protocol.validate(config)
        super().__init__(config)
        if self.world != protocol.draws_per_update(config) or not 0 <= self.rank < self.world:
            raise ValueError("DDP rank count differs from frozen batch")

    def wrap_step(self, step):
        params = [p for p in step.parameters() if p.requires_grad]
        self.agree(records.digest({str(i): records.tensor_hash(p) for i, p in enumerate(params)}))
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

    def mask(self, gates):
        if gates.decision is not None:
            dist.broadcast(gates.decision, src=0)

    def draws(self, rows):
        values = self.gather(rows)
        if any(len(rank_rows) != 1 for rank_rows in values):
            raise ValueError("DDP requires exactly one consumed draw per rank")
        return [row for rank_rows in values for row in rank_rows]

    def save(self, path, **kwargs):
        rng = self.gather(checkpoints.rng_state([self.local_rank]))
        self.agree(records.tensor_hash(kwargs["gates"].alpha))
        from scripts.prune.learn.adapters import state_dict

        self.agree(records.digest({k: records.tensor_hash(v) for k, v in state_dict(kwargs["adapters"]).items()}))
        if self.primary:
            checkpoints.save(
                path, **kwargs, rng_override={"world_size": self.world, "ranks": rng}, draws_per_update=self.world
            )
        dist.barrier()

    def close(self):
        dist.destroy_process_group()


class Session:
    """Session delegation keeps native ownership and only adds rank coordination."""

    def __init__(self, current, runtime):
        self.current, self.distributed = current, runtime
        self.device = current.device
        self.execution_devices = (current.device,)

    def __getattr__(self, name):
        return getattr(self.current, name)
