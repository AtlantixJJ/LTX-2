"""Which transformer file a run loads -- the one backbone resolver for train, probe and deploy.

``model_registry.resolve("2.5")`` defaults to the **distilled** transformer. Training a dev
adapter, probing it and deploying it must all load the *same* file, and an adapter must be
refused on any other one, so the variant is resolved here once and recorded by **file
identity** (name + ``provenance.checkpoint_fingerprint``) rather than by ``model_key`` alone:
``model_key=2.5`` names a generation, not a set of weights.

The model version (``--model``) and the backbone variant (``--variant``) are separate axes.
This module reuses the registry's existing per-component ``transformer_path`` override rather
than adding another loader, so text encoder, VAE, sigmas and scale factors still come from the
version's defaults.
"""

from __future__ import annotations

from pathlib import Path

VARIANTS = ("distilled", "dev")
DEFAULT_VARIANT = "distilled"

# The dev transformer sits beside the registry's distilled default. Only 2.5 has one here.
DEV_TRANSFORMER = {"2.5": "ltx-2.5-22b-dev-transformer-bf16.safetensors"}


def transformer_path(key: str, variant: str, explicit: str | Path | None = None) -> Path:
    """The transformer file for ``(key, variant)``; ``explicit`` overrides it (pruned exports)."""
    from scripts.prune.core import model_registry  # noqa: PLC0415 -- keeps this module import-light

    if variant not in VARIANTS:
        raise ValueError(f"unknown backbone variant {variant!r}; expected one of {VARIANTS}")
    if explicit is not None:
        return Path(explicit)
    distilled = Path(model_registry._default_paths(key)["transformer"])  # noqa: SLF001 -- the registry's own default
    if variant == "distilled":
        return distilled
    if key not in DEV_TRANSFORMER:
        raise SystemExit(f"no dev transformer is registered for model {key!r}")
    return distilled.with_name(DEV_TRANSFORMER[key])


def resolve(key: str, variant: str, explicit: str | Path | None = None):  # noqa: ANN201 -- RefinerModel
    """``model_registry.resolve`` with the variant's transformer as the component override."""
    from scripts.prune.core import model_registry  # noqa: PLC0415

    path = transformer_path(key, variant, explicit)
    if not path.is_file():
        raise SystemExit(f"{variant} transformer for model {key} does not exist: {path}")
    return model_registry.resolve(key, transformer_path=path)


def identity(path: str | Path, variant: str, key: str) -> dict[str, str]:
    """The base-weights identity an adapter is stamped with and later checked against."""
    from scripts.prune.core import provenance  # noqa: PLC0415

    path = Path(path)
    return {
        "model_key": key,
        "base_variant": variant,
        "base_transformer_file": path.name,
        "base_transformer_fingerprint": provenance.checkpoint_fingerprint(path),
    }
