"""Decoding a token latent to pixels -- one implementation.

Both callers previously built their own ``VideoDecoder`` + ``gpu_model``
resident-decoder boilerplate and their own copy of the token-latent restore
path; that construction now lives only in ``session.Session.decoder()``.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.helpers import post_process_latent

from scripts.prune.session import DTYPE, Session


def _token_tools(session: Session, state, token_latent: torch.Tensor) -> VideoLatentTools:
    """Reconstruct the unpatchifier geometry from the serialized token positions.

    The fps literal passed below is inert: ``VideoLatentTools.unpatchify`` only
    reads ``self.patchifier``/``self.target_shape``, never ``self.fps`` -- fps
    only feeds RoPE position construction, which decode never recomputes (the
    state's own saved positions are used as-is). Not one of the two real fps
    bugs plan §S5 fixed.
    """
    positions = state.positions
    if positions.shape != (token_latent.shape[0], 3, token_latent.shape[1], 2):
        raise ValueError(f"unexpected position shape {tuple(positions.shape)} for tokens {tuple(token_latent.shape)}")
    frames, height, width = (int(torch.unique(positions[0, axis, :, 0]).numel()) for axis in range(3))
    if frames * height * width != token_latent.shape[1]:
        raise ValueError(f"position grid {(frames, height, width)} does not cover {token_latent.shape[1]} tokens")
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1),
        VideoLatentShape(token_latent.shape[0], token_latent.shape[-1], frames, height, width),
        24.0,
        scale_factors=session.model.scale_factors,
    )


def decode_latent(session: Session, latent: torch.Tensor, decoder) -> torch.Tensor:
    """A dense ``(B,C,F,H,W)`` latent -> ``[F,H,W,C]`` float pixels in ``[0,1]``.

    The phase1_gates rollout path.
    """
    decoded = torch.cat(
        list(decoder.decode_video(latent.to(device=session.device, dtype=DTYPE), None, None)), dim=0
    ).float()
    return decoded.clamp(0, 1).cpu()


def decode_token_latent(session: Session, state, token_latent: torch.Tensor, decoder) -> torch.Tensor:
    """Token-space x0 -> ``[F,C,H,W]`` float pixels in ``[0,1]``, conditioning restored.

    head_ablation_eval's channel-first convention; ``decode_latent``'s is
    channel-last. ``metrics._as_bchw`` accepts either layout, so the two are
    free to keep their own rather than being forced to agree.
    """
    restored = post_process_latent(token_latent, state.denoise_mask, state.clean_latent)
    tools = _token_tools(session, state, restored)
    latent = tools.unpatchify(tools.clear_conditioning(replace(state, latent=restored))).latent
    return decode_latent(session, latent, decoder).permute(0, 3, 1, 2)
