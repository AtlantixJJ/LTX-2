"""Every use of an underscore-prefixed ``ltx_core`` / ``ltx_pipelines`` symbol.

Pinned at LTX-2 commit 9a1c49b (last upstream merge, 2026-08-25). When that
pin moves, this is the ONE file that breaks -- nothing else in
``scripts/prune`` may import a private name; ``tests/test_ltx_adapter.py``
enforces it.

  DiffusionStage._transformer_ctx                                -> transformer_ctx
  VideoDecoder._decoder_builder                                  -> video_decoder
  ImageConditioner._build_encoder                                -> video_encoder
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager

import torch

from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, VideoDecoder
from ltx_pipelines.utils.gpu_model import gpu_model


def transformer_ctx(stage: DiffusionStage, **kwargs: object) -> AbstractContextManager:
    """The resident-transformer context a built ``DiffusionStage`` yields."""
    return stage._transformer_ctx(**kwargs)


@contextmanager
def video_decoder(video_vae_path: str, dtype: torch.dtype, device: torch.device):
    """A resident video decoder, built and freed as a context manager."""
    holder = VideoDecoder(video_vae_path, dtype, device)
    with torch.no_grad(), gpu_model(holder._decoder_builder.build(device=device, dtype=dtype).eval()) as decoder:
        yield decoder


@contextmanager
def video_encoder(video_vae_path: str, dtype: torch.dtype, device: torch.device):
    """A resident video encoder, built and freed as a context manager."""
    conditioner = ImageConditioner(video_vae_path, dtype, device)
    with torch.no_grad(), gpu_model(conditioner._build_encoder()) as encoder:
        yield encoder
