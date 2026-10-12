"""Shared saved-input builders and fixtures; never a collected test module."""

import json
from pathlib import Path

import torch

from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import hashing
from scripts.onestep_avatar.experiments import stock_parity
from scripts.onestep_avatar.model import common


class Geometry:
    scale_factors = SpatioTemporalScaleFactors(time=8, height=32, width=32)


class ControlledX0(torch.nn.Module):
    num_blocks = 1

    def forward(
        self, video: Modality, audio: Modality | None = None, perturbations: object = None
    ) -> tuple[torch.Tensor, None]:
        del perturbations
        assert audio is None
        # Depend on current state, text and noise levels so a mismatched call matters.
        denoised = video.latent.float() * 0.31 + video.timesteps * 0.17 + video.context.float().mean() * 0.2
        denoised = torch.where(video.timesteps == 0, video.latent.float(), denoised)
        return denoised.to(video.latent.dtype), None


def inputs() -> tuple[common.ClipGrid, torch.Tensor, torch.Tensor]:
    grid = common.ClipGrid.build(
        3, 64, 64, 30.0, Geometry(), device=torch.device("cpu"), dtype=torch.bfloat16, latent_channels=128
    )
    image = torch.randn(1, 128, 1, 2, 2, generator=torch.Generator().manual_seed(81)).bfloat16()
    context = torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(82)).bfloat16()
    return grid, image, context


def reference_case(tmp_path: Path) -> tuple[common.ClipGrid, torch.Tensor, torch.Tensor, torch.Tensor, dict, dict]:
    grid, image, context = inputs()
    levels = torch.tensor([1.0, 0.8670830726623535, 0.6315688490867615, 0.1, 0.0])
    outputs, traces, noise, _ = stock_parity.sample_paths(ControlledX0(), context, grid, image, levels, 42)
    for name, value in outputs.items():
        torch.save(grid.unpatchify_block(value, 3), tmp_path / (name + ".pt"))
    torch.save(traces, tmp_path / "traces.pt")
    torch.save(noise, tmp_path / "noise.pt")
    protocol = dict(
        stock_parity.PROTOCOL,
        software=stock_parity.software.capture(
            "evaluation", "bidirectional", decoder=True, extra_sources=stock_parity.EXTRA_SOURCES
        ),
        schedule=levels.tolist(),
        seed=42,
        frames=3,
        fps=30.0,
        prompt="checked prompt",
        input_files={},
    )
    record = {
        "protocol": protocol,
        "raw": stock_parity.compare_paths(outputs, traces),
        "decoded": {"stock_repeat": {"unequal_elements": 0}},
        "output_files": {path.name: stock_parity.sha256(path) for path in tmp_path.glob("*.pt")},
        "noise_tensor_sha256": hashing.tensor_sha256(noise),
        "image_tensor_sha256": hashing.tensor_sha256(image),
        "text_tensor_sha256": hashing.tensor_sha256(context),
    }
    (tmp_path / "result.json").write_text(json.dumps(record))
    return grid, image, context, levels, protocol, record
