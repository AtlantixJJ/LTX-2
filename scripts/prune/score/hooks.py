"""Non-destructive head/FFN mask hooks used by pruning calibration.

The masks are attached at the two executed branch boundaries.  They deliberately
live outside checkpoint surgery: scoring can change a mask many times while the
loaded checkpoint remains immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

import torch

from scripts.prune.data import whole_clip


def read_mask_artifact(path: str | Path, *, model_key: str, fingerprint: str,
                       widths: dict[str, int], expected_task: str | None = None,
                       baseline: dict | None = None) -> tuple[dict[str, list[float]], str]:
    """Validate a score report before applying its mask to a checkpoint.

    A report may contain all attention masks, all FFN masks, or both. Omitted
    kinds mean unpruned; a partial kind is an error rather than an implicit mask.
    """
    raw = Path(path).read_bytes()
    report = json.loads(raw)
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("model_key") != model_key or (
        provenance.get("transformer_fingerprint") != fingerprint
    ):
        raise ValueError(f"{path}: mask model key or transformer fingerprint differs from the active checkpoint")
    if expected_task is not None and provenance.get("task") != expected_task:
        raise ValueError(f"{path}: mask task {provenance.get('task')!r} != {expected_task!r}")
    if expected_task == "whole_clip_d0":
        views = provenance.get("calibration_views")
        sigmas = provenance.get("sigmas")
        if (report.get("candidate_format") != "whole_clip_d0_mask_v1" or
                provenance.get("attention") != "full_bidirectional" or
                provenance.get("objective") != "white" or
                provenance.get("conditioning") != "clean_capture_frame_0" or
                not isinstance(views, list) or not views or
                any(not isinstance(view, str) or not view for view in views) or
                not isinstance(sigmas, list) or not sigmas or
                any(type(sigma) not in (int, float) or not math.isfinite(sigma) or not 0 < sigma <= 1
                    for sigma in sigmas) or
                not isinstance(provenance.get("baseline_manifest"), str) or
                not provenance.get("baseline_manifest") or
                not isinstance(provenance.get("text_context"), dict)):
            raise ValueError(f"{path}: incomplete native D0 mask provenance")
        whole_clip.validate_native_provenance(provenance, baseline)
    iterative = report.get("iterative")
    masks = (iterative.get("masks") if isinstance(iterative, dict) and isinstance(iterative.get("masks"), dict)
             else report.get("masks"))
    if not isinstance(masks, dict) or not masks:
        raise ValueError(f"{path}: no mask dictionary")
    if set(masks) - set(widths):
        raise ValueError(f"{path}: unknown mask keys {sorted(set(masks) - set(widths))[:4]}")
    for suffixes in ((".attn1", ".attn2"), (".ff",)):
        expected = {name for name in widths if name.endswith(suffixes)}
        present = expected & set(masks)
        if present and present != expected:
            raise ValueError(f"{path}: incomplete mask family; missing {sorted(expected - present)[:4]}")
    for name, values in masks.items():
        if not isinstance(values, list) or len(values) != widths[name] or not values:
            raise ValueError(f"{path}: {name} has wrong mask width")
        if any(type(value) not in (int, float) or value not in (0, 1) for value in values):
            raise ValueError(f"{path}: {name} must contain finite binary values")
        if not any(values):
            raise ValueError(f"{path}: {name} would remove the entire branch")
    return masks, hashlib.sha256(raw).hexdigest()


def require_native_heldout_scope(path: str | Path, *, view: str, sigmas: list[float]) -> None:
    """Reject calibration-subject reuse and sigma substitution in D0 validation."""
    provenance = json.loads(Path(path).read_text())["provenance"]
    if provenance.get("task") != "whole_clip_d0":
        raise ValueError("held-out check requires a native D0 mask")
    if whole_clip.actor_identity(view) in {whole_clip.actor_identity(v) for v in provenance["calibration_views"]}:
        raise ValueError("held-out actor was used to calibrate this mask")
    if any(sigma not in provenance["sigmas"] for sigma in sigmas):
        raise ValueError("held-out sigma was not included in mask calibration")


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
