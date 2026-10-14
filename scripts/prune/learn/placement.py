"""Place complete native video blocks on multiple devices without changing math.

Transport frozen TransformerArgs with differentiable copies. Hooks also run
during checkpoint recomputation. Audio, causal caches and duplicate devices
are unsupported. This context owns placement only, not model lifetime.
"""

from contextlib import contextmanager
from dataclasses import fields, replace

import torch

from ltx_core.model.transformer.transformer_args import TransformerArgs


def move_args(value, device):
    if value is None:
        return None
    if not isinstance(value, TransformerArgs):
        raise TypeError("placement requires native TransformerArgs")
    if value.kv_caches is not None or value.self_attn_kv_cache is not None or value.kv_write:
        raise ValueError("placement does not support causal caches")

    def move(item):
        if isinstance(item, torch.Tensor):
            return item.to(device=device)
        if isinstance(item, tuple):
            return tuple(move(v) for v in item)
        if item is None or isinstance(item, (bool, int, float)):
            return item
        raise TypeError(f"unsupported TransformerArgs field: {type(item)}")

    return replace(value, **{field.name: move(getattr(value, field.name)) for field in fields(value)})


def input_hook(device):
    def hook(module, args, kwargs):
        audio = args[1] if len(args) > 1 else kwargs.get("audio")
        if audio is not None:
            raise ValueError("placement requires video-only execution")
        if args:
            return (move_args(args[0], device), *args[1:]), kwargs
        return args, {**kwargs, "video": move_args(kwargs.get("video"), device)}

    return hook


@contextmanager
def partition(model, devices):
    devices = tuple(torch.device(d) for d in devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("placement needs distinct devices")
    blocks = model.velocity_model.transformer_blocks
    if len(blocks) % len(devices):
        raise ValueError("block count must divide evenly across devices")
    primary = devices[0]
    original = [next(block.parameters()).device for block in blocks]
    if any(device != primary for device in original):
        raise ValueError("native model must initially reside on the primary device")
    handles = []
    try:
        per_device = len(blocks) // len(devices)
        for index, block in enumerate(blocks):
            owner = devices[index // per_device]
            block.to(owner)
            handles.append(block.register_forward_pre_hook(input_hook(owner), with_kwargs=True))

        def output_hook(module, args, kwargs, output):
            video, audio = output
            if audio is not None:
                raise ValueError("placement requires video-only output")
            return move_args(video, primary), None

        handles.append(blocks[-1].register_forward_hook(output_hook, with_kwargs=True))
        for device in devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        yield model
    finally:
        for handle in handles:
            handle.remove()
        # The enclosing Session releases all parameter storage. Moving the
        # trained model back to one GPU here could itself exhaust that GPU.


class PlacedSession:
    """Apply frozen block placement to every resident model, including recovery."""

    def __init__(self, current, devices, reference_root=None):
        self.current = current
        self.device = current.device
        self.execution_devices = tuple(torch.device(f"cuda:{d}") for d in devices)
        self.reference_root = reference_root
        if self.execution_devices[0] != self.device:
            raise ValueError("placement primary differs from Session device")
        if len(devices) != 4 or devices != [0, 1, 2, 3] or torch.cuda.device_count() != 4:
            raise ValueError("placement requires exactly four visible GPUs")

    def __getattr__(self, name):
        return getattr(self.current, name)

    @contextmanager
    def transformer(self, *args, **kwargs):
        with self.current.transformer(*args, **kwargs) as model:
            with partition(model, self.execution_devices):
                yield model
