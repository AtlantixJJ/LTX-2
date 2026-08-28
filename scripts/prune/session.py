"""Shared bootstrap for pruning entry points."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_pipelines.utils.blocks import DiffusionStage, VideoDecoder
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.gpu_model import gpu_model

from scripts.prune import artifacts, preflight, prompt_cache, refine_core, refine_task
from scripts.prune.model_registry import SUPPORTED_MODELS, RefinerModel

DTYPE = torch.bfloat16


def add_model_args(parser) -> None:
    parser.add_argument("--model", default="2.5", choices=SUPPORTED_MODELS)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)


def add_record_args(parser, *, default_split: str = "calibration") -> None:
    parser.add_argument("--states", type=Path, default=None, help="Calibration cache root; default is model calibration/.")
    parser.add_argument("--split", choices=("calibration", "held_out"), default=default_split)
    parser.add_argument("--max-records", type=int, default=None, help="Cap to a balanced, strided sample; omit for all.")


def add_geometry_args(parser) -> None:
    parser.add_argument("--window-frames", type=int, default=refine_task.WINDOW_FRAMES)
    parser.add_argument("--overlap-frames", type=int, default=refine_task.OVERLAP_FRAMES)


@dataclass(frozen=True)
class Session:
    model: RefinerModel
    device: torch.device
    script: str
    context: object
    denoiser: object
    sigmas: torch.Tensor

    @property
    def key(self) -> str:
        return self.model.key

    @property
    def out_root(self) -> Path:
        return artifacts.root(self.key)

    def states_root(self, override: Path | None = None) -> Path:
        return override or artifacts.calibration(self.key)

    def geometry(self, window_frames: int | None = None, overlap_frames: int | None = None) -> refine_core.WindowGeometry:
        return refine_core.WindowGeometry(
            window_frames=window_frames or refine_task.WINDOW_FRAMES,
            overlap_frames=overlap_frames or refine_task.OVERLAP_FRAMES,
            scale_factors=self.model.scale_factors,
        )

    @contextmanager
    def transformer(self, transformer_path: Path | None = None):
        stage = DiffusionStage.from_checkpoint(
            str(transformer_path or self.model.paths.transformer()),
            DTYPE,
            self.device,
            model_configurator=LTXVideoOnlyModelConfigurator,
            scale_factors=self.model.scale_factors,
        )
        try:
            with torch.no_grad(), stage._transformer_ctx() as transformer:
                yield transformer
        finally:
            del stage
            torch.cuda.empty_cache()

    @contextmanager
    def decoder(self):
        holder = VideoDecoder(self.model.paths.video_vae(), DTYPE, self.device)
        with torch.no_grad(), gpu_model(holder._decoder_builder.build(device=self.device, dtype=DTYPE).eval()) as decoder:
            yield decoder

    def stamp(self, **extra) -> dict:
        from scripts.prune import provenance

        return provenance.stamp(self.model, self.device, script=self.script, **extra)


def open_session(args, *, script: str, sampler: str = "euler", transformer_path: Path | None = None) -> Session:
    model = preflight.check(args.model, sampler=sampler, gpu_id=args.gpu_id, transformer_path=transformer_path)
    device = torch.device(f"cuda:{args.gpu_id}")
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    sigmas = torch.tensor(refine_task.schedule_for(model.sigmas, refine_task.K_STEP), dtype=torch.float32, device=device)
    return Session(
        model=model,
        device=device,
        script=script,
        context=context,
        denoiser=SimpleDenoiser(context, None),
        sigmas=sigmas,
    )
