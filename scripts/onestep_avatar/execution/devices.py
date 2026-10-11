"""Own the shared GPU pool and fixed four-rank training topology without models.

Inputs: complete direct device-memory observations, active owned device IDs and
the requested training/single-device topology. Validate strict integer IDs and
memory values and an optional explicit pool; filter pool order by idle memory and active ownership; return
the first four free devices for training or the first one for other work.
Outputs: a complete selected tuple, or None when too few devices are free.
This policy grants no ownership: the registry locks and rechecks before launch.
One active attempt per original owner preserves targeted descendant containment.
"""

from collections.abc import Iterable

GPU_POOL = tuple(range(8))
TRAIN_PROCESSES = 4


def validate_pool(pool: tuple[int, ...]) -> None:
    """Reject empty, repeated or out-of-range physical device pools."""
    if (not isinstance(pool, tuple) or not pool
            or any(type(gpu) is not int or gpu not in GPU_POOL for gpu in pool)
            or len(set(pool)) != len(pool)):
        raise ValueError("GPU pool requires unique allowed physical devices")


def validate_inventory(memory: dict[int, int]) -> None:
    """Missing rows or non-integer occupancy cannot be interpreted as idle."""
    if (
        not isinstance(memory, dict)
        or not set(GPU_POOL).issubset(memory)
        or any(type(gpu) is not int or type(used) is not int or used < 0 for gpu, used in memory.items())
    ):
        raise ValueError("incomplete direct GPU memory inventory")


def validate_selection(gpus: tuple[int, ...], *, training: bool | None = None) -> None:
    """Permit one GPU or a complete four-device training set, never partial ranks."""
    expected = (1, TRAIN_PROCESSES) if training is None else (TRAIN_PROCESSES,) if training else (1,)
    if (
        not isinstance(gpus, tuple)
        or len(gpus) not in expected
        or any(type(gpu) is not int or gpu not in GPU_POOL for gpu in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("device selection requires unique allowed GPUs and one or four devices")


def select_devices(
    memory: dict[int, int], *, training: bool, busy: Iterable[int] = (), pool: tuple[int, ...] = GPU_POOL
) -> tuple[int, ...] | None:
    """Choose a complete ordered subset; active or non-idle devices remain excluded."""
    validate_inventory(memory)
    validate_pool(pool)
    busy = tuple(busy)
    if type(training) is not bool or any(type(gpu) is not int or gpu not in GPU_POOL for gpu in busy):
        raise ValueError("GPU selection has an invalid topology or owned-device inventory")
    free = tuple(gpu for gpu in pool if memory[gpu] < 1024 and gpu not in busy)
    count = TRAIN_PROCESSES if training else 1
    return free[:count] if len(free) >= count else None
