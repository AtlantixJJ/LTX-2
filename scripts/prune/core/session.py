"""Shared bootstrap for pruning entry points."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_pipelines.utils.blocks import DiffusionStage
from scripts.prune.core import artifacts, ltx_adapter, preflight, records
from scripts.prune.core.model_registry import SUPPORTED_MODELS, RefinerModel
from scripts.prune.data import prompt_cache

DTYPE = torch.bfloat16
DEFAULT_PROMPT = "a high quality, sharp, detailed video with fine texture and natural lighting"


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="2.5", choices=SUPPORTED_MODELS)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    """Text prompt selection; the default keeps every saved run reproducible."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", default=None, help="Prompt text; default DEFAULT_PROMPT.")
    group.add_argument("--prompt-file", type=Path, default=None, help="UTF-8 file holding the prompt text.")


def resolve_prompt(args: argparse.Namespace) -> str:
    """The prompt text selected by ``add_prompt_args`` (``DEFAULT_PROMPT`` when neither flag is set)."""
    if getattr(args, "prompt_file", None) is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    if getattr(args, "prompt", None) is not None:
        return args.prompt
    return DEFAULT_PROMPT


@dataclass(frozen=True)
class Session:
    model: RefinerModel
    device: torch.device
    script: str
    context: object

    @property
    def key(self) -> str:
        return self.model.key

    @property
    def out_root(self) -> Path:
        return artifacts.root(self.key)

    @contextmanager
    def transformer(
        self,
        transformer_path: Path | None = None,
        *,
        video_tools: object | None = None,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
    ) -> Iterator[torch.nn.Module]:
        """The resident transformer, optionally with ``loras`` fused into its weights.

        LoRAs fuse at load (``loader/fuse_loras.py``), so the built transformer is an ordinary
        one -- there is no adapter left at inference and nothing downstream needs to know. An
        empty tuple uses the unmodified checkpoint loading path. Sampling schedules
        are supplied by the caller; a Session has no window geometry or default schedule.
        """
        stage = DiffusionStage.from_checkpoint(
            str(transformer_path or self.model.paths.transformer()),
            DTYPE,
            self.device,
            loras=tuple(loras),
            model_configurator=LTXVideoOnlyModelConfigurator,
            scale_factors=self.model.scale_factors,
        )
        try:
            with torch.no_grad(), ltx_adapter.transformer_ctx(stage, video_tools=video_tools) as transformer:
                yield transformer
        finally:
            del stage
            torch.cuda.empty_cache()

    @contextmanager
    def decoder(self) -> Iterator[torch.nn.Module]:
        with ltx_adapter.video_decoder(self.model.paths.video_vae(), DTYPE, self.device) as decoder:
            yield decoder

    def stamp(self, **extra) -> dict:
        from scripts.prune.core import provenance  # noqa: PLC0415 -- defer the session/artifact provenance edge

        return provenance.stamp(self.model, self.device, script=self.script, **extra)


def open_session(
    args: argparse.Namespace,
    *,
    script: str,
    transformer_path: Path | None = None,
    prompt: str | None = None,
) -> Session:
    """Open a session from parsed `add_model_args` options."""
    return open_model(args.model, args.gpu_id, script=script, transformer_path=transformer_path, prompt=prompt)


def open_model(
    model_key: str,
    gpu_id: int,
    *,
    script: str,
    transformer_path: Path | None = None,
    prompt: str | None = None,
) -> Session:
    """Check the device and components, then load or build the selected prompt context."""
    model = preflight.check(model_key, gpu_id=gpu_id, transformer_path=transformer_path)
    device = torch.device(f"cuda:{gpu_id}")
    context = prompt_cache.get_or_build(model, DEFAULT_PROMPT if prompt is None else prompt, DTYPE, device)
    return Session(
        model=model,
        device=device,
        script=script,
        context=context,
    )


def open_frozen(study: dict, gpu_id: int, *, script: str) -> Session:
    """Open a 2.5 empty-text session on a frozen study's transformer and verify it.

    The frozen software versions, the actual context tensor and every resolved
    component path must equal the study's pins.
    """
    software = study["software"]
    if (
        software["torch"] != str(torch.__version__)
        or software["cuda"] != torch.version.cuda
        or software["python"] != platform.python_version()
    ):
        raise ValueError("runtime software differs from frozen study")
    current = open_model(
        "2.5", gpu_id, script=script, prompt="", transformer_path=Path(study["components"]["transformer"]["path"])
    )
    if records.tensor_hash(current.context) != study["text_context"]["tensor_sha256"]:
        raise ValueError("actual empty context differs from frozen study")
    for role, path in (
        ("transformer", current.model.paths.transformer()),
        ("text_encoder", current.model.paths.text_encoder()),
        ("video_vae", current.model.paths.video_vae()),
    ):
        if Path(path).resolve() != Path(study["components"][role]["path"]).resolve():
            raise ValueError("resolved component differs from frozen study")
    return current
