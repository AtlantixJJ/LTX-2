"""Frozen batch semantics and phase budgets for legacy two-draw FSDP and uniform-F18 DDP.

DDP uses one independent draw per rank, one local backward and one automatic
gradient average. Older presets keep their two-draw schedules unchanged.
Progressive studies use exactly 500 soft updates and no binary or compact phase.
They validate configurable hardening interval, original-head fraction and regret threshold.
"""

DDP_METHOD = "ddp_full_replicas"
FSDP_METHOD = "fsdp_frozen_replicated_adapters"


def is_ddp(config):
    return config.get("distributed_training") == DDP_METHOD


def draws_per_update(config):
    count = config.get("global_unique_clips", 2)
    if type(count) is not int or count < 1:
        raise ValueError("global_unique_clips must be a positive integer")
    return count


def validate(config):
    if config.get("progressive_policy") is not None:
        from scripts.prune.learn.progressive import validate_config

        validate_config(config["progressive_policy"])
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
    elif config.get("distributed_training") == FSDP_METHOD and count != 2:
        raise ValueError("legacy FSDP requires two unique draws")
    return count


def update_draws(draws, update, config, rank=None):
    count = draws_per_update(config)
    selected = draws[(update - 1) * count : update * count]
    if len(selected) != count:
        raise ValueError("frozen schedule lacks complete update draws")
    if rank is None:
        return selected
    index = rank if is_ddp(config) else rank % 2
    if not 0 <= index < count:
        raise ValueError("rank exceeds frozen draw assignment")
    return [selected[index]]


def phase_budget(config):
    """Global update boundaries: soft end, selection end and compact end.

    The frozen config owns the budget. Binary refinement can be zero updates.
    """
    soft, binary, compact = (config[k] for k in ("soft_updates", "binary_updates", "compact_updates"))
    progressive = config.get("progressive_policy") is not None
    if progressive and (soft, binary, compact) != (500, 0, 0):
        raise ValueError("progressive study requires 500 soft and no binary or compact updates")
    if any(type(v) is not int or v < 0 for v in (soft, binary, compact)) or soft < 1 or (compact < 1 and not progressive):
        raise ValueError("phase budgets must be non-negative integers with soft and compact phases")
    return soft, soft + binary, soft + binary + compact
