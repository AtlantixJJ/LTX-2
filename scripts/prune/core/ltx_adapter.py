"""Isolate access to private upstream model, transformer and VAE interfaces.

velocity_forward calls the native velocity model once and applies the stock
clean-prediction conversion. Transformer and VAE contexts delegate loading and
cleanup to upstream owners. Keep private upstream imports here so an upstream
interface change has one review point. tests/test_ltx_adapter.py enforces it.

Pinned at LTX-2 commit 9a1c49b as the upstream interface reference.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

import torch

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.utils import to_denoised
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, VideoDecoder
from ltx_pipelines.utils.gpu_model import gpu_model


def velocity_forward(model: torch.nn.Module, video: Modality) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native velocity and the stock clean conversion from one forward.

    Only the timestep-based X0 wrapper is supported. The caller restores c0
    and controls autograd. Never derive velocity from rounded clean outputs.
    """
    if not isinstance(model, X0Model):
        raise ValueError("velocity forward requires the supported X0Model wrapper")
    velocity, audio = model.velocity_model(video=video, audio=None, perturbations=None)
    if velocity is None or audio is not None:
        raise ValueError("expected a video-only velocity prediction")
    return velocity, to_denoised(video.latent, velocity, video.timesteps)


def transformer_ctx(stage: DiffusionStage, **kwargs: object) -> AbstractContextManager:
    """The resident-transformer context a built ``DiffusionStage`` yields."""
    return stage._transformer_ctx(**kwargs)


@contextmanager
def video_decoder(video_vae_path: str, dtype: torch.dtype, device: torch.device) -> Iterator[torch.nn.Module]:
    """A resident video decoder, built and freed as a context manager."""
    holder = VideoDecoder(video_vae_path, dtype, device)
    with torch.no_grad(), gpu_model(holder._decoder_builder.build(device=device, dtype=dtype).eval()) as decoder:
        yield decoder


@contextmanager
def video_encoder(video_vae_path: str, dtype: torch.dtype, device: torch.device) -> Iterator[torch.nn.Module]:
    """A resident video encoder, built and freed as a context manager."""
    conditioner = ImageConditioner(video_vae_path, dtype, device)
    with torch.no_grad(), gpu_model(conditioner._build_encoder()) as encoder:
        yield encoder
