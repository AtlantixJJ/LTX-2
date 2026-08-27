"""``--model {2.3,2.5}`` -> ``ModelPaths`` + sigmas + sampler + probed ``ModelCaps``.

Single place that turns a generation key into everything downstream needs, so
every scripts/prune/* script and vae_refine_sliding_window.py agree on what
"2.3" and "2.5" mean. See plans/2026-08-26-refiner-head-ffn-pruning.md §4.

Downstream code should always go through :func:`resolve` and its accessors
(``RefinerModel.paths.transformer()`` etc.), never a bare checkpoint string --
that is what lets one script serve both generations without a version branch.

``ModelCaps`` fields are read straight from the checkpoint's own metadata with
the *same* defaults ``LTXModelConfigurator``/``LTXVideoOnlyModelConfigurator``
use (packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py),
so ``ModelCaps`` always describes the model that would actually be built --
never a value asserted from the plan's own tables. That distinction already
caught one wrong assumption in the plan: LTX-2.3's checkpoint declares
``frequencies_precision=float64`` too (not float32 as originally assumed), so
``double_precision_rope`` is True on both generations, not just 2.5.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from ltx_core.types import SpatioTemporalScaleFactors
from ltx_pipelines.distilled import should_use_ancestral_sampler
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, detect_model_version
from ltx_pipelines.utils.model_paths import ModelPaths

from scripts.prune import geometry

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../LTX-2
WORKSPACE_ROOT = REPO_ROOT.parent
CKPT_ROOT = Path(os.environ.get("LTX_CHECKPOINTS", WORKSPACE_ROOT / "checkpoints"))

SUPPORTED_MODELS = ("2.3", "2.5")
SAMPLER_CHOICES = ("euler", "ancestral", "auto")


@dataclass(frozen=True)
class ModelCaps:
    """Transformer capabilities probed from checkpoint metadata (never hardcoded).

    Field defaults mirror ``LTXModelConfigurator.from_metadata`` /
    ``LTXVideoOnlyModelConfigurator.from_metadata`` exactly, so a missing config
    key resolves to the same value the real model build would use.
    """

    num_layers: int
    num_heads: int
    head_dim: int
    cross_attention_dim: int
    ff_inner_dim: int
    ff_bias: bool
    apply_gated_attention: bool
    cross_attention_adaln: bool
    latent_channels: int
    double_precision_rope: bool
    use_prompt_adaln_single: bool
    causal_temporal_positioning: bool
    use_keyframes_abs_pos_embedding: bool


def probe_caps(transformer_path: str) -> ModelCaps:
    """Read ``ModelCaps`` from a transformer checkpoint's safetensors metadata."""
    with safe_open(transformer_path, framework="pt") as f:
        meta = f.metadata() or {}
    cfg = json.loads(meta.get("config", "{}")).get("transformer", {})
    if not cfg:
        raise ValueError(f"{transformer_path}: no 'transformer' block in checkpoint metadata['config']")
    num_heads = cfg.get("num_attention_heads", 32)
    head_dim = cfg.get("attention_head_dim", 128)
    dim = num_heads * head_dim
    return ModelCaps(
        num_layers=cfg.get("num_layers", 48),
        num_heads=num_heads,
        head_dim=head_dim,
        cross_attention_dim=cfg.get("cross_attention_dim", 4096),
        ff_inner_dim=4 * dim,  # FeedForward hardcodes mult=4; no config key exists yet (§10.1).
        ff_bias=cfg.get("ff_bias", True),
        apply_gated_attention=cfg.get("apply_gated_attention", False),
        cross_attention_adaln=cfg.get("cross_attention_adaln", False),
        latent_channels=cfg.get("in_channels", 128),
        double_precision_rope=cfg.get("frequencies_precision", False) == "float64",
        use_prompt_adaln_single=cfg.get("use_prompt_adaln_single", True),
        causal_temporal_positioning=cfg.get("causal_temporal_positioning", False),
        use_keyframes_abs_pos_embedding=cfg.get("use_keyframes_abs_pos_embedding", False),
    )


@dataclass(frozen=True)
class RefinerModel:
    key: str  # "2.3" | "2.5" -- namespaces every downstream artifact path
    version: tuple[int, ...]
    paths: ModelPaths
    sigmas: list[float]
    stepper_kind: str  # "euler" | "ancestral"
    caps: ModelCaps
    # Probed from the VAE block list, never the literal (8, 32, 32) -- see
    # scripts/prune/geometry.py and plan §4 decision 2. Pass to every
    # DiffusionStage.from_checkpoint / VideoLatentShape.from_pixel_shape call so
    # the state builder and the stage cannot silently disagree about the grid.
    scale_factors: SpatioTemporalScaleFactors
    scale_factors_source: str


def _default_paths(key: str) -> dict[str, Path]:
    if key == "2.3":
        return {
            "transformer": CKPT_ROOT / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors",
            "text_encoder": CKPT_ROOT / "google" / "gemma-3-12b-it-qat-q4_0-unquantized",
        }
    if key == "2.5":
        root = CKPT_ROOT / "LTX-2.5"
        return {
            "transformer": root / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
            "text_encoder": root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            "video_vae": root / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
        }
    raise ValueError(f"unknown model key {key!r}; expected one of {SUPPORTED_MODELS}")


def _download_hint(key: str, missing: list[tuple[str, Path]]) -> str:
    lines = [f"Missing {key} component file(s):"]
    for name, path in missing:
        lines.append(f"  {name}: {path}")
    lines.append("")
    if key == "2.3":
        lines.append(f"hf download Lightricks/LTX-2.3 {' '.join(p.name for _, p in missing)} \\")
        lines.append(f"    --local-dir {CKPT_ROOT / 'LTX-2.3'}")
    else:
        rel = [str(p.relative_to(CKPT_ROOT / "LTX-2.5")) for _, p in missing]
        lines.append("hf download Lightricks/LTX-2.5 \\")
        lines.append("    " + " \\\n    ".join(rel) + " \\")
        lines.append(f"    --local-dir {CKPT_ROOT / 'LTX-2.5'}")
    return "\n".join(lines)


def _require_files(key: str, paths: ModelPaths) -> None:
    missing: list[tuple[str, Path]] = []
    for name, value in (
        ("transformer", paths.transformer_path),
        ("text_encoder", paths.text_encoder_path),
        ("video_vae", paths.video_vae_path),
    ):
        if value is None:
            continue
        p = Path(value)
        if not p.exists():
            missing.append((name, p))
    if missing:
        raise SystemExit(_download_hint(key, missing))


def resolve(
    key: str = "2.5",
    *,
    sampler: str = "euler",
    transformer_path: str | Path | None = None,
    text_encoder_path: str | Path | None = None,
    video_vae_path: str | Path | None = None,
) -> RefinerModel:
    """Resolve a generation key (plus optional per-component overrides) to a
    :class:`RefinerModel`. Per-component paths always override the registry default
    for that key -- e.g. ``--transformer-path`` lets a caller point at an exported
    pruned checkpoint while everything else (text encoder, video VAE, sigmas) still
    comes from the ``key``'s defaults.
    """
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"unknown --model {key!r}; expected one of {SUPPORTED_MODELS}")
    if sampler not in SAMPLER_CHOICES:
        raise ValueError(f"unknown --sampler {sampler!r}; expected one of {SAMPLER_CHOICES}")

    defaults = _default_paths(key)
    transformer = str(transformer_path) if transformer_path is not None else str(defaults["transformer"])
    text_encoder = str(text_encoder_path) if text_encoder_path is not None else str(defaults.get("text_encoder", ""))
    text_encoder = text_encoder or None

    if key == "2.3":
        video_vae = str(video_vae_path) if video_vae_path is not None else None  # defaults to the monolith itself
        paths = ModelPaths.from_monolith(transformer, text_encoder, video_vae_path=video_vae)
    else:
        video_vae = str(video_vae_path) if video_vae_path is not None else str(defaults["video_vae"])
        paths = ModelPaths.from_split(transformer_path=transformer, text_encoder_path=text_encoder, video_vae_path=video_vae)

    _require_files(key, paths)
    probed = geometry.probe_scale_factors(paths.video_vae(), paths.transformer())
    version = detect_model_version(paths.transformer())
    if sampler == "auto":
        kind = "ancestral" if should_use_ancestral_sampler(paths.transformer()) else "euler"
    else:
        kind = sampler
    return RefinerModel(
        key=key,
        version=version,
        paths=paths,
        sigmas=[float(v) for v in DISTILLED_SIGMA_VALUES],
        stepper_kind=kind,
        caps=probe_caps(paths.transformer()),
        scale_factors=probed.factors,
        scale_factors_source=probed.source,
    )
