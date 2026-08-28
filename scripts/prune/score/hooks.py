"""Non-destructive head/FFN mask hooks used by pruning calibration.

The masks are attached at the two executed branch boundaries.  They deliberately
live outside checkpoint surgery: scoring can change a mask many times while the
loaded checkpoint remains immutable.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


class MaskAttachments(dict[str, torch.Tensor]):
    """Named mask parameters plus the removable PyTorch hook handles."""

    def __init__(self) -> None:
        super().__init__()
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def detach_all(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "MaskAttachments":
        return self

    def __exit__(self, *_: object) -> None:
        self.detach_all()


def _core(model):
    """Accept either ``X0Model`` or its underlying LTX model."""
    return getattr(model, "velocity_model", model)


def iter_video_attention(model) -> Iterator[tuple[str, object]]:
    core = _core(model)
    for layer, block in enumerate(core.transformer_blocks):
        for kind in ("attn1", "attn2"):
            yield f"{layer}.{kind}", getattr(block, kind)


def iter_video_ffn(model) -> Iterator[tuple[str, object]]:
    core = _core(model)
    for layer, block in enumerate(core.transformer_blocks):
        yield f"{layer}.ff", block.ff


def detach_all(attachments: MaskAttachments | None) -> None:
    """Remove all hooks, accepting ``None`` for simple ``finally`` blocks."""
    if attachments is not None:
        attachments.detach_all()


def attach_head_masks(model, initial: dict[str, torch.Tensor] | None = None, *, requires_grad: bool = True) -> MaskAttachments:
    """Attach one multiplicative ``(heads,)`` mask before every attention output projection."""
    attached = MaskAttachments()
    for name, attn in iter_video_attention(model):
        value = torch.ones(attn.heads, device=attn.to_out[0].weight.device, dtype=torch.float32)
        if initial is not None and name in initial:
            source = initial[name].detach().to(device=value.device, dtype=value.dtype)
            if source.shape != value.shape:
                raise ValueError(f"{name}: mask {tuple(source.shape)} != heads {tuple(value.shape)}")
            value.copy_(source)
        mask = value.requires_grad_(requires_grad)

        def hook(_mod, args, *, mask=mask, attn=attn):
            (x,) = args
            b, t, width = x.shape
            expected = attn.heads * attn.dim_head
            if width != expected:
                raise ValueError(f"attention activation width {width} != {expected}")
            masked = x.reshape(b, t, attn.heads, attn.dim_head) * mask.to(dtype=x.dtype).view(1, 1, -1, 1)
            return (masked.reshape(b, t, width),)

        attached[name] = mask
        attached.handles.append(attn.to_out[0].register_forward_pre_hook(hook))
    return attached


def attach_ffn_masks(model, initial: dict[str, torch.Tensor] | None = None, *, requires_grad: bool = True) -> MaskAttachments:
    """Attach one multiplicative intermediate-channel mask before every FFN output projection."""
    attached = MaskAttachments()
    for name, ff in iter_video_ffn(model):
        inner = ff.net[2].weight.shape[1]
        value = torch.ones(inner, device=ff.net[2].weight.device, dtype=torch.float32)
        if initial is not None and name in initial:
            source = initial[name].detach().to(device=value.device, dtype=value.dtype)
            if source.shape != value.shape:
                raise ValueError(f"{name}: mask {tuple(source.shape)} != FFN width {tuple(value.shape)}")
            value.copy_(source)
        mask = value.requires_grad_(requires_grad)

        def hook(_mod, args, *, mask=mask):
            (x,) = args
            if x.shape[-1] != mask.numel():
                raise ValueError(f"FFN activation width {x.shape[-1]} != mask width {mask.numel()}")
            return (x * mask.to(dtype=x.dtype).view(1, 1, -1),)

        attached[name] = mask
        attached.handles.append(ff.net[2].register_forward_pre_hook(hook))
    return attached


def collect_activations(model, which: str, callback) -> MaskAttachments:
    """Call ``callback(name, activation, module)`` at the specified prune boundary."""
    if which not in {"head", "ffn"}:
        raise ValueError("which must be 'head' or 'ffn'")
    attached = MaskAttachments()
    iterator = iter_video_attention(model) if which == "head" else iter_video_ffn(model)
    for name, module in iterator:
        boundary = module.to_out[0] if which == "head" else module.net[2]

        def hook(_mod, args, *, name=name, module=module):
            callback(name, args[0], module)

        attached.handles.append(boundary.register_forward_pre_hook(hook))
    return attached
