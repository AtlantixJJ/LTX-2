"""Decode dense latents through the session-owned video VAE.

Convert the decoded tensor to the requested frame/image format and release the
session decoder on exit. This helper performs no denoising or model selection."""

from __future__ import annotations

import torch

from scripts.prune.core.session import DTYPE, Session


def decode_latent(
    session: Session,
    latent: torch.Tensor,
    decoder,  # noqa: ANN001
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A dense ``(B,C,F,H,W)`` latent -> ``[F,H,W,C]`` float pixels in ``[0,1]``."""
    decoded = torch.cat(
        list(decoder.decode_video(latent.to(device=session.device, dtype=DTYPE), None, generator)), dim=0
    ).float()
    return decoded.clamp(0, 1).cpu()
