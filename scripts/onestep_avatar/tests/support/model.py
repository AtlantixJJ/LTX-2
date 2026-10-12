"""Shared CPU builders; never collected as test modules."""

from __future__ import annotations

import torch

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.model.causal import CausalGeometry
from scripts.onestep_avatar.model.common import ClipGrid

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)

CHANNELS = 8

EDGE = 64

LATENT_FRAMES = 7

FPS = 30.0

SIGMA0 = 0.725

DEVICE = torch.device("cpu")

CONTEXT_TOKENS = 3

CONTEXT_DIM = 16


def _model(*, prompt_adaln: bool = False, weight_scale: float = 0.05) -> LTXModel:
    """A 2-layer LTXModel with every parameter deterministically initialised.

    ``LTXModel`` allocates several parameters with ``torch.empty``; left as they come, a CPU
    test would compare two forwards through uninitialised memory and could pass or NaN at
    random.
    """
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_layers=2,
        cross_attention_dim=8,  # == inner_dim: caption_projection maps the text dim into it
        caption_projection=torch.nn.Linear(CONTEXT_DIM, 8),
        cross_attention_adaln=prompt_adaln,
        use_prompt_adaln_single=prompt_adaln,
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * weight_scale)
    return model.eval()


def _geometry(context_latent_frames: int = 16) -> CausalGeometry:
    return CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=context_latent_frames)


def _grid(geometry: CausalGeometry, latent_frames: int = LATENT_FRAMES) -> ClipGrid:
    return ClipGrid.build(
        latent_frames, EDGE, EDGE, FPS, geometry, device=DEVICE, dtype=torch.float32, latent_channels=CHANNELS
    )


def _context() -> torch.Tensor:
    return torch.randn(1, CONTEXT_TOKENS, CONTEXT_DIM, generator=torch.Generator().manual_seed(1))
