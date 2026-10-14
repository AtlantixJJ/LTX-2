"""Four-rank frozen-weight FSDP with small replicated FP32 trainable state."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import TransformerArgs
from scripts.prune.learn import checkpoints

_NATIVE_ARGS_REGISTERED = False


def assignment(rank, world=4):
    if world != 4 or not 0 <= rank < world:
        raise ValueError("the frozen two-clip update requires exactly four ranks")
    return rank % 2


class Runtime:
    uses_ddp = False
    feasibility_method = "four_rank_fsdp"

    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        assignment(self.rank, self.world)
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=self.device)

    @property
    def primary(self):
        return self.rank == 0

    def gather(self, value):
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def any(self, value):
        flag = torch.tensor(int(value), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def agree(self, value):
        if len(set(self.gather(value))) != 1:
            raise ValueError("distributed ranks disagree on state")

    def wrap(self, model, parameters, block_type=BasicAVTransformerBlock):
        global _NATIVE_ARGS_REGISTERED

        if not _NATIVE_ARGS_REGISTERED:
            torch.export.register_dataclass(TransformerArgs)
            torch.export.register_dataclass(Modality)
            _NATIVE_ARGS_REGISTERED = True
        self.agree(checkpoints.digest({str(i): checkpoints.tensor_hash(p) for i, p in enumerate(parameters)}))
        precise = [
            (name, p)
            for name, p in model.velocity_model.named_parameters()
            if not p.requires_grad and p.dtype != torch.bfloat16
        ]
        if (
            any(p.dtype != torch.float32 for _, p in precise)
            or sum(p.numel() * p.element_size() for _, p in precise) > 64 * 2**20
        ):
            raise ValueError("unsupported frozen precision inventory for replicated native tables")
        self.inventory = {
            "world_size": self.world,
            "native_args_pytree_registered": True,
            "adapter_master_dtype": sorted({str(p.dtype) for p in parameters}),
            "adapter_bytes": sum(p.numel() * p.element_size() for p in parameters),
            "replicated_frozen": [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in precise],
            "replicated_frozen_bytes": sum(p.numel() * p.element_size() for _, p in precise),
        }
        self.agree(checkpoints.digest(self.inventory))
        model.velocity_model = FSDP(
            model.velocity_model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=ModuleWrapPolicy({block_type}),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=False,
            device_id=self.device,
            use_orig_params=True,
            sync_module_states=True,
            ignored_states=[*parameters, *(p for _, p in precise)],
            limit_all_gathers=True,
        )
        torch.cuda.empty_cache()
        return model

    def reduce_gradients(self, parameters):
        parameters = list(parameters)
        if self.any(any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters)):
            raise ValueError("missing or nonfinite replicated gradient")
        flat = torch.cat([p.grad.reshape(-1).float() for p in parameters])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world)
        cursor = 0
        for p in parameters:
            p.grad.copy_(flat[cursor : cursor + p.numel()].reshape_as(p))
            cursor += p.numel()

    def mask(self, gates):
        if gates.decision is not None:
            dist.broadcast(gates.decision, src=0)

    def draws(self, rows):
        records = self.gather(rows)
        if records[0] != records[2] or records[1] != records[3]:
            raise ValueError("duplicate-rank draw or loss evidence differs")
        return records[0] + records[1]

    def save(self, path, **kwargs):
        rng = self.gather(checkpoints.rng_state([self.local_rank]))
        self.agree(checkpoints.tensor_hash(kwargs["gates"].alpha))
        from scripts.prune.learn.adapters import state_dict

        self.agree(
            checkpoints.digest({k: checkpoints.tensor_hash(v) for k, v in state_dict(kwargs["adapters"]).items()})
        )
        if self.primary:
            checkpoints.save(path, **kwargs, rng_override={"world_size": self.world, "ranks": rng})
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
