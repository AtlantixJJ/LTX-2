"""Export structural width masks as self-describing safetensors checkpoints.

Validate binary branch widths, plan coupled V/output/gate and FFN slicing on
meta tensors, then stream payloads. Q/K remain full for original normalization.
The four storage/execution modes and every runtime config key are defined in
doc/score/export_pruned.md. The CLI validates native provenance; direct export
callers must supply their own provenance checks.

CLI: python -m scripts.prune.score.export_pruned --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import json
import resource
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.prune.core import model_registry, preflight, provenance
from scripts.prune.score import hooks, safetensors_stream

PREFIX = "model.diffusion_model.transformer_blocks"


def checkpoint_mask_widths(source: str | Path) -> dict[str, int]:
    """Inspect safetensors shapes without materializing the checkpoint's 40 GB of weights."""
    with safe_open(source, framework="pt", device="cpu") as handle:
        config = json.loads((handle.metadata() or {}).get("config", "{}")).get("transformer", {})
        layers = int(config.get("num_layers", 48))
        head_dim = int(config.get("attention_head_dim", 128))
        widths = {}
        for layer in range(layers):
            for kind in ("attn1", "attn2"):
                size = handle.get_slice(f"{PREFIX}.{layer}.{kind}.to_q.weight").get_shape()[0]
                if size % head_dim:
                    raise ValueError(f"{layer}.{kind}: Q width is not divisible by head dim")
                widths[f"{layer}.{kind}"] = size // head_dim
            widths[f"{layer}.ff"] = handle.get_slice(f"{PREFIX}.{layer}.ff.net.0.proj.weight").get_shape()[0]
    return widths


def _indices(keep: list[int], head_dim: int) -> torch.Tensor:
    return (torch.tensor(keep, dtype=torch.long)[:, None] * head_dim + torch.arange(head_dim)[None]).reshape(-1)


def _keep(mask: list[float], label: str) -> list[int]:
    out = [i for i, value in enumerate(mask) if value != 0]
    if not out:
        raise ValueError(f"{label}: refusing to export an empty branch")
    return out


def _slice_heads(sd: dict[str, torch.Tensor], layer: int, kind: str, keep: list[int], d: int) -> None:
    idx = _indices(keep, d)
    b = f"{PREFIX}.{layer}.{kind}"
    # Q/K RMSNorm spans *all original heads*. Keep full Q/K projections and
    # their normalization weights, then select retained heads after norm in
    # PytorchPreAttention. Slicing Q/K here changes the surviving head values.
    for proj in ("to_v",):
        sd[f"{b}.{proj}.weight"] = sd[f"{b}.{proj}.weight"][idx]
        key = f"{b}.{proj}.bias"
        if key in sd:
            sd[key] = sd[key][idx]
    sd[f"{b}.to_out.0.weight"] = sd[f"{b}.to_out.0.weight"][:, idx]
    for suffix in ("weight", "bias"):
        key = f"{b}.to_gate_logits.{suffix}"
        if key in sd:
            sd[key] = sd[key][keep]


def _slice_ffn(sd: dict[str, torch.Tensor], layer: int, keep: list[int]) -> None:
    b = f"{PREFIX}.{layer}.ff"
    index = torch.tensor(keep, dtype=torch.long)
    sd[f"{b}.net.0.proj.weight"] = sd[f"{b}.net.0.proj.weight"][index]
    key = f"{b}.net.0.proj.bias"
    if key in sd:
        sd[key] = sd[key][index]
    key = f"{b}.net.2.weight"
    sd[key] = sd[key][:, index]


def export(
    source: str | Path,
    masks: dict,
    output: str | Path,
    *,
    model_key: str,
    provenance_block: dict | None = None,
    mode: str = "masked_full",
) -> Path:
    """Perform checkpoint-space surgery; masks are keyed ``'0.attn1'`` / ``'0.ff'``."""
    tensors, metadata, transforms = stream_plan(
        source, masks, model_key=model_key, provenance_block=provenance_block, mode=mode
    )
    safetensors_stream.write(source, output, tensors, metadata, transforms=transforms)
    return Path(output)


def stream_plan(source, masks, *, model_key, provenance_block=None, mode="masked_full"):
    """Use the existing surgery on meta tensors; build independent per-tensor slices."""
    tensors, metadata = safetensors_stream.inspect(source)
    export_tensors(
        tensors,
        metadata,
        masks,
        widths=checkpoint_mask_widths(source),
        model_key=model_key,
        provenance_block=provenance_block,
        mode=mode,
    )
    transforms = {}
    if mode in ("compact", "compact_faithful"):
        d = json.loads(metadata["config"])["transformer"].get("attention_head_dim", 128)
        for name, values in masks.items():
            block, kind = name.split(".")
            keep = _keep(values, name)
            if len(keep) == len(values):
                continue
            prefix = f"{PREFIX}.{block}.{kind}"
            if kind == "ff":
                indices = torch.tensor(keep, dtype=torch.long)
                axes = {"net.0.proj.weight": 0, "net.0.proj.bias": 0, "net.2.weight": 1}
            else:
                indices = _indices(keep, d)
                axes = {"to_v.weight": 0, "to_v.bias": 0, "to_out.0.weight": 1}
                for suffix in ("weight", "bias"):
                    key = f"{prefix}.to_gate_logits.{suffix}"
                    if key in tensors:
                        transforms[key] = partial(torch.index_select, dim=0, index=torch.tensor(keep))
            for suffix, axis in axes.items():
                key = f"{prefix}.{suffix}"
                if key in tensors:
                    transforms[key] = partial(torch.index_select, dim=axis, index=indices)
    return tensors, metadata, transforms


def load(source):
    """All tensors and metadata of one safetensors checkpoint, read once."""
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        # safe_open exposes keys() but does not implement iteration.
        sd = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
    return sd, metadata


def export_tensors(  # noqa: PLR0912 -- four explicit export formats
    sd, metadata, masks, *, widths, model_key, provenance_block=None, mode="masked_full"
):
    """Apply checkpoint-space surgery to loaded tensors and their config metadata in place."""
    if mode not in ("masked_full", "sparse", "compact", "compact_faithful"):
        raise ValueError(f"unknown export mode: {mode}")
    if set(masks) - set(widths):
        raise ValueError(f"unknown mask keys: {sorted(set(masks) - set(widths))[:4]}")
    for name, values in masks.items():
        if len(values) != widths[name] or not values or not any(values) or any(value not in (0, 1) for value in values):
            raise ValueError(f"{name}: invalid binary mask or width")
    config_all = json.loads(metadata.get("config", "{}"))
    config = config_all.setdefault("transformer", {})
    layers, d = int(config.get("num_layers", 48)), int(config.get("attention_head_dim", 128))
    a1, a2, ffn, a1_indices, a2_indices = [], [], [], [], []
    a1_active, a2_active, ffn_active = [], [], []
    for layer in range(layers):
        for kind, kept_widths, identities in (("attn1", a1, a1_indices), ("attn2", a2, a2_indices)):
            key = f"{layer}.{kind}"
            original = sd[f"{PREFIX}.{layer}.{kind}.to_q.weight"].shape[0] // d
            keep = _keep(masks.get(key, [1.0] * original), key)
            if mode in ("compact", "compact_faithful"):
                if len(keep) < original:
                    _slice_heads(sd, layer, kind, keep, d)
                kept_widths.append(len(keep) if mode == "compact" else original)
                identities.append(keep if mode == "compact" and len(keep) < original else None)
                if mode == "compact_faithful":
                    (a1_active if kind == "attn1" else a2_active).append(keep)
            else:
                kept_widths.append(original)
                # Full-width Q/K already use the original RoPE order. An
                # identity list would gather frequencies in every layer.
                identities.append(None)
                (a1_active if kind == "attn1" else a2_active).append(keep)
        key = f"{layer}.ff"
        original = sd[f"{PREFIX}.{layer}.ff.net.0.proj.weight"].shape[0]
        keep = _keep(masks.get(key, [1.0] * original), key)
        if mode in ("compact", "compact_faithful"):
            _slice_ffn(sd, layer, keep)
            ffn.append(len(keep) if mode == "compact" else original)
            if mode == "compact_faithful":
                ffn_active.append(keep)
        else:
            ffn.append(original)
            ffn_active.append(keep)
    config.update(
        {
            "per_layer_video_attn1_heads": a1,
            "per_layer_video_attn2_heads": a2,
            "per_layer_ff_inner_dim": ffn,
            "per_layer_video_attn1_rope_head_indices": a1_indices,
            "per_layer_video_attn2_rope_head_indices": a2_indices,
            "video_pruning_preserve_qk_norm": True,
            "video_pruning_shape_faithful": mode == "compact_faithful",
            "pruning": {"task": "whole_clip_d0", "model_key": model_key, "mode": mode, **(provenance_block or {})},
        }
    )
    if mode != "compact":
        config.update(
            {
                "per_layer_video_attn1_active_head_indices": a1_active,
                "per_layer_video_attn2_active_head_indices": a2_active,
                "per_layer_video_ffn_active_channels": ffn_active,
                "video_pruning_select_active_heads": mode == "sparse",
            }
        )
    metadata["config"] = json.dumps(config_all)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--masks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--transformer-path")
    p.add_argument("--mode", choices=("masked_full", "sparse", "compact", "compact_faithful"), default="masked_full")
    args = p.parse_args()
    model = preflight.check(args.model, transformer_path=args.transformer_path)
    source = model.paths.transformer()
    stamp = provenance.stamp(model)
    masks, mask_sha256 = hooks.read_mask_artifact(
        args.masks,
        model_key=model.key,
        fingerprint=stamp["transformer_fingerprint"],
        widths=checkpoint_mask_widths(source),
        expected_task="whole_clip_d0",
    )
    path = export(
        source,
        masks,
        args.output,
        model_key=model.key,
        mode=args.mode,
        provenance_block={
            **stamp,
            "task": "whole_clip_d0",
            "source_transformer_fingerprint": stamp["transformer_fingerprint"],
            "mask_sha256": mask_sha256,
            "masks": str(args.masks),
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(path),
                "fingerprint": provenance.checkpoint_fingerprint(path),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            }
        )
    )


if __name__ == "__main__":
    main()
