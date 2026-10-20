"""Frozen batch semantics and phase budgets for serial and uniform-F18 DDP head learning.

DDP uses one independent draw per rank, one local backward and one automatic
gradient average. A serial study consumes `global_unique_clips` draws per update
(two by default) with gradient accumulation. The frozen config owns the soft,
binary and compact budgets; binary refinement can be zero updates.
Progressive policies and four-rank FSDP are retired; their configs fail here.
"""

DDP_METHOD = "ddp_full_replicas"


def is_ddp(config):
    return config.get("distributed_training") == DDP_METHOD


def draws_per_update(config):
    count = config.get("global_unique_clips", 2)
    if type(count) is not int or count < 1:
        raise ValueError("global_unique_clips must be a positive integer")
    return count


def validate(config):
    if config.get("progressive_policy") is not None:
        raise ValueError("progressive head-gate policies are retired from the live tree")
    method = config.get("distributed_training")
    if method not in (None, DDP_METHOD):
        raise ValueError(f"unsupported distributed training method: {method}")
    count = draws_per_update(config)
    if is_ddp(config):
        world = len(config["gpu_ids"])
        if (
            world not in (2, 4)
            or config.get("uniform_f18") is not True
            or count != world
            or config.get("accumulation") != 1
            or config.get("rank_draw_assignment") != list(range(world))
        ):
            raise ValueError("DDP requires uniform F18, one draw per rank and accumulation one on two or four GPUs")
    return count


def update_draws(draws, update, config, rank=None):
    count = draws_per_update(config)
    selected = draws[(update - 1) * count : update * count]
    if len(selected) != count:
        raise ValueError("frozen schedule lacks complete update draws")
    if rank is None:
        return selected
    if not is_ddp(config) or not 0 <= rank < count:
        raise ValueError("rank exceeds frozen draw assignment")
    return [selected[rank]]


def phase_budget(config):
    """Global update boundaries: soft end, selection end and compact end."""
    if config.get("progressive_policy") is not None:
        raise ValueError("progressive head-gate policies are retired from the live tree")
    soft, binary, compact = (config[k] for k in ("soft_updates", "binary_updates", "compact_updates"))
    if any(type(v) is not int or v < 0 for v in (soft, binary, compact)) or soft < 1 or compact < 1:
        raise ValueError("phase budgets must be non-negative integers with soft and compact phases")
    return soft, soft + binary, soft + binary + compact


def recovery_end(config, arm):
    """Final recovery update: the compact end, plus 1000 sequential adapter updates for that arm."""
    _, _, compact_end = phase_budget(config)
    return compact_end + 1000 if arm == "learned_sequential" else compact_end
