"""Persistent AR-geometry calibration states (Phase 1).

Each record contains the fully patchified state presented to the transformer,
its teacher x0 target, and enough metadata to audit its clip/family/sigma.
Records are CPU ``torch.save`` files: all importance estimators load the exact
same noisy tensors, rather than independently re-noising clips.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ltx_core.types import LatentState
from scripts.prune.core import refine_core, refine_task

# Bumped from 1 when ``fps`` joined ChunkStateMeta and CTX_LATENT_FRAMES went 4 -> 1.
# A format-1 cache was built at the wrong window geometry and with a hardcoded 24 fps,
# so it is not merely missing a field -- its tensors are states the deployed refiner
# never sees. Rebuild rather than migrate.
RECORD_FORMAT = 2


@dataclass(frozen=True)
class ChunkStateMeta:
    clip: str
    split: str
    family: str                 # on_policy | renoised
    sigma: float
    step_index: int
    chunk_latent_frames: int
    context_latent_frames: int
    seed: int
    fps: float                  # part of RoPE, not metadata -- see refine_core.build_tools


def make_state(l_init, ctx_latent, sigma: float, video_tools, seed: int, device: torch.device) -> LatentState:
    """Build the deployed window's frozen-context state.

    A thin wrapper over :func:`refine_core.make_window_state` -- the same call
    ``scripts/vae_refine_sliding_window.py`` makes -- that additionally asserts the
    carryover is the deployed width. ``ctx_latent`` is injected at latent index 1;
    index 0 stays fresh because a causal VAE's first latent frame represents the
    standalone first pixel frame and has no predecessor in the carryover.
    """
    if ctx_latent.shape[2] != refine_task.CTX_LATENT_FRAMES:
        raise ValueError(
            f"expected {refine_task.CTX_LATENT_FRAMES} frozen latent frames, got {ctx_latent.shape[2]}"
        )
    return refine_core.make_window_state(l_init, ctx_latent, sigma, video_tools, seed, device, l_init.dtype)


def chunk_token_mask(state: LatentState, meta: ChunkStateMeta) -> torch.Tensor:
    """The tokens the deployed AR refiner actually predicts: the ``n_new`` chunk frames.

    ``state.denoise_mask`` is **not** that set, despite plan §2 asserting it is.
    ``make_state`` freezes the carryover at latent index 1, which leaves latent
    frame 0 -- the causal VAE's standalone single-pixel keyframe -- fresh as well.
    So ``denoise_mask`` selects ``{frame 0} u {the n_new chunk frames}``, and at
    ``n_new=1`` the keyframe is *half* the fresh-token mass. Scoring head/FFN
    importance on that set spends half the calibration signal on reconstructing a
    keyframe the AR refiner never emits.

    Tokens are laid out frame-major and contiguous (``VideoConditionByLatentIndex``
    relies on the same fact to locate its injection slice), so the chunk is exactly
    the trailing ``n_new`` frames' worth of tokens.
    """
    total_latent_frames = 1 + meta.context_latent_frames + meta.chunk_latent_frames
    tokens = state.denoise_mask.shape[1]
    if tokens % total_latent_frames:
        raise ValueError(
            f"{tokens} tokens is not divisible by {total_latent_frames} latent frames; "
            "the frame-major token layout this mask depends on does not hold"
        )
    per_frame = tokens // total_latent_frames
    mask = torch.zeros_like(state.denoise_mask)
    mask[:, tokens - meta.chunk_latent_frames * per_frame :] = 1.0
    # The chunk must be a strict subset of what is actually being denoised; if it is
    # not, the geometry assumption above is wrong and every score built on it is too.
    if not torch.all(mask * state.denoise_mask.to(mask.dtype) == mask):
        raise ValueError("derived chunk tokens are not all fresh -- token layout assumption violated")
    return mask


def _cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.detach().to(device="cpu").contiguous() if value is not None else None


def save_record(path: str | Path, state: LatentState, x0_star: torch.Tensor, meta: ChunkStateMeta) -> Path:
    """Atomically persist a state/target pair; refuses accidental target mismatch."""
    path = Path(path)
    if state.latent.shape != x0_star.shape:
        raise ValueError(f"state tokens {tuple(state.latent.shape)} != x0* {tuple(x0_star.shape)}")
    if state.denoise_mask.sum().item() == 0:
        raise ValueError("calibration state has no fresh tokens")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": RECORD_FORMAT,
            "meta": asdict(meta),
            "state": {
                "latent": _cpu(state.latent), "denoise_mask": _cpu(state.denoise_mask),
                "positions": _cpu(state.positions), "clean_latent": _cpu(state.clean_latent),
                "attention_mask": _cpu(state.attention_mask), "keyframes_mask": _cpu(state.keyframes_mask),
                "frozen": state.frozen,
            },
            "x0_star": _cpu(x0_star),
        },
        tmp,
    )
    tmp.replace(path)
    return path


def load_record(path: str | Path, device: torch.device | str = "cpu") -> tuple[LatentState, torch.Tensor, ChunkStateMeta]:
    """Load a record created by :func:`save_record`, restoring a ``LatentState``."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format") != RECORD_FORMAT:
        raise ValueError(
            f"{path}: calibration record format {payload.get('format')!r}, expected {RECORD_FORMAT}. "
            "Format 1 caches were built at the pre-parity geometry (4 frozen latent frames, 24 fps) "
            "and cannot be migrated -- rebuild with `teacher.py --build-calibration`."
        )
    s = payload["state"]
    state = LatentState(
        latent=s["latent"], denoise_mask=s["denoise_mask"], positions=s["positions"],
        clean_latent=s["clean_latent"], attention_mask=s["attention_mask"],
        keyframes_mask=s["keyframes_mask"], frozen=s["frozen"],
    )
    return state, payload["x0_star"], ChunkStateMeta(**payload["meta"])


def write_index(root: str | Path, records: list[Path], *, provenance: dict, extra: dict | None = None) -> Path:
    """Write an inspectable manifest alongside opaque tensor records."""
    root = Path(root)
    entries = []
    for p in records:
        state, _, meta = load_record(p)
        # Record both token counts: the gap between them is exactly the index-0
        # keyframe, and it is what tells a later reader which mask a score used.
        entries.append({
            "path": str(p.relative_to(root)),
            **asdict(meta),
            "tokens": int(state.denoise_mask.shape[1]),
            "fresh_tokens": int(state.denoise_mask.sum().item()),
            "chunk_tokens": int(chunk_token_mask(state, meta).sum().item()),
        })
    index = {"format": RECORD_FORMAT, "provenance": provenance, "records": entries, **(extra or {})}
    path = root / "index.json"
    path.write_text(json.dumps(index, indent=2))
    return path


def iter_records(root: str | Path, split: str | None = None):
    """Yield sorted records, optionally restricted to calibration/held_out."""
    for path in sorted(Path(root).glob("*.pt")):
        _, _, meta = load_record(path)
        if split is None or meta.split == split:
            yield path
