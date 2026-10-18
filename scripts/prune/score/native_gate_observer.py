"""Observe native gate logits without changing attention or projection outputs.

The projection hook reconstructs the native BF16 expression, then reduces detached
values on CPU. No token tensor survives the callback. The mirrored design explains
why this measures native modulation rather than learned pruning scores.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ltx_core.model.transformer.ops import PytorchGatedAttention
from scripts.prune.score.hooks import iter_video_attention

BINS = 2048
CUTOFFS = (0.1, 0.5, 1.5, 1.9)


def moments(values: np.ndarray, prefix: str = "") -> dict[str, np.ndarray]:
    """Population moments; sums stay mergeable without quantile approximation."""
    values = values.astype(np.float64)
    return {
        prefix + "sum": values.sum(axis=0),
        prefix + "square_sum": np.square(values).sum(axis=0),
        prefix + "minimum": values.min(axis=0),
        prefix + "maximum": values.max(axis=0),
    }


def summarize(logits: np.ndarray, gates: np.ndarray, frames: int, per_frame: int) -> dict[str, np.ndarray]:
    """Reduce one block and draw, separating the local clean first frame."""
    if logits.shape != gates.shape or gates.ndim != 2 or gates.shape[0] != frames * per_frame or frames < 2:
        raise ValueError("native gate token/frame geometry differs")
    if not np.isfinite(gates).all() or not np.isfinite(logits).all() or (gates < 0).any() or (gates > 2).any():
        raise ValueError("nonfinite or out-of-range native gates")
    generated, clean = gates[per_frame:], gates[:per_frame]
    result = moments(generated, "gate_") | moments(logits[per_frame:], "logit_")
    result.update(moments(clean, "clean_gate_"))
    result.update(moments(logits[:per_frame], "clean_logit_"))
    heads = gates.shape[1]
    indices = np.minimum((generated.astype(np.float64) * (BINS / 2)).astype(np.int64), BINS - 1)
    indices += np.arange(heads, dtype=np.int64)[None, :] * BINS
    result["histogram"] = np.bincount(indices.ravel(), minlength=heads * BINS).reshape(heads, BINS)
    for key, values in (("threshold_counts", generated), ("clean_threshold_counts", clean)):
        result[key] = np.stack(
            [
                (values < CUTOFFS[0]).sum(0),
                (values < CUTOFFS[1]).sum(0),
                (values > CUTOFFS[2]).sum(0),
                (values > CUTOFFS[3]).sum(0),
            ]
        )
    framed = gates.reshape(frames, per_frame, heads)
    result["frame_gate_sum"] = framed.astype(np.float64).sum(axis=1)
    result["frame_high_count"] = (framed > 1.5).sum(axis=1)
    return result


class NativeGateObserver:
    """One observation per complete native video self-attention block."""

    def __init__(self, model: torch.nn.Module, *, blocks: int = 48, heads: int = 32):
        self.model = model
        self.blocks = blocks
        self.heads = heads
        self.handles: list[Any] = []
        self.values: dict[int, dict[str, np.ndarray]] = {}
        self.frames = self.per_frame = 0
        self.active = False

    def begin(self, frames: int, per_frame: int) -> None:
        if self.active:
            raise ValueError("unfinished native observation")
        self.values = {}
        self.frames, self.per_frame = frames, per_frame
        self.active = True

    def finish(self) -> dict[str, np.ndarray]:
        if set(self.values) != set(range(self.blocks)):
            raise ValueError("native gate observation lacks exact block coverage")
        self.active = False
        return {key: np.stack([self.values[b][key] for b in range(self.blocks)]) for key in self.values[0]}

    def hook_for(self, block: int) -> Callable:
        def observe(_module: torch.nn.Module, _args: tuple, logits: torch.Tensor) -> None:
            if not self.active:
                return
            if block in self.values:
                raise ValueError("duplicate native gate observation")
            if logits.shape != (1, self.frames * self.per_frame, self.heads):
                raise ValueError("native gate batch or head geometry differs")
            # Keep the exact original expression/dtype; casting before sigmoid changes BF16 gates.
            gates = 2.0 * torch.sigmoid(logits.detach())
            self.values[block] = summarize(
                logits.detach()[0].float().cpu().numpy(),
                gates[0].float().cpu().numpy(),
                self.frames,
                self.per_frame,
            )
            # A projection hook returning None preserves the original output object.

        return observe

    def __enter__(self) -> NativeGateObserver:
        try:
            names = []
            for name, attention in iter_video_attention(self.model):
                if not name.endswith(".attn1"):
                    continue
                block = int(name.split(".")[0])
                if (
                    attention.to_gate_logits is None
                    or attention.heads != self.heads
                    or attention.active_head_indices is not None
                    or type(attention.gated_attention_function) is not PytorchGatedAttention
                ):
                    raise ValueError("observer requires full-width native PytorchGatedAttention")
                names.append(block)
                self.handles.append(attention.to_gate_logits.register_forward_hook(self.hook_for(block)))
            if names != list(range(self.blocks)):
                raise ValueError("observer model block inventory differs")
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False
