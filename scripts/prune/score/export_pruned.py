"""Export structured refiner pruning masks as a self-describing safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.prune.core import model_registry, preflight, provenance
from scripts.prune.score import hooks

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


def _slice_ffn(sd: dict[str, torch.Tensor], layer: int, keep: list[int], fitted: torch.Tensor | None = None) -> None:
    b = f"{PREFIX}.{layer}.ff"
    index = torch.tensor(keep, dtype=torch.long)
    sd[f"{b}.net.0.proj.weight"] = sd[f"{b}.net.0.proj.weight"][index]
    key = f"{b}.net.0.proj.bias"
    if key in sd:
        sd[key] = sd[key][index]
    key = f"{b}.net.2.weight"
    if fitted is not None:
        expected = (sd[key].shape[0], len(keep))
        if tuple(fitted.shape) != expected:
            raise ValueError(f"{b}: fitted weight {tuple(fitted.shape)} != {expected}")
        sd[key] = fitted.to(dtype=sd[key].dtype, device="cpu").contiguous()
    else:
        sd[key] = sd[key][:, index]


def export(source: str | Path, masks: dict, output: str | Path, *, model_key: str,
           reconstruction: dict[str, torch.Tensor] | None = None, provenance_block: dict | None = None,
           mode: str = "masked_full") -> Path:
    """Perform checkpoint-space surgery; masks are keyed ``'0.attn1'`` / ``'0.ff'``."""
    source, output = Path(source), Path(output)
    if mode not in ("masked_full", "sparse", "compact", "compact_faithful"):
        raise ValueError(f"unknown export mode: {mode}")
    if reconstruction and mode != "compact":
        raise ValueError("reconstruction requires compact export")
    widths = checkpoint_mask_widths(source)
    if set(masks) - set(widths):
        raise ValueError(f"unknown mask keys: {sorted(set(masks) - set(widths))[:4]}")
    for name, values in masks.items():
        if len(values) != widths[name] or not values or not any(values) or any(value not in (0, 1) for value in values):
            raise ValueError(f"{name}: invalid binary mask or width")
    if reconstruction:
        for name, fitted in reconstruction.items():
            if name not in widths or not name.endswith(".ff") or name not in masks:
                raise ValueError(f"{name}: reconstruction has no matching FFN mask")
            kept = sum(value == 1 for value in masks[name])
            with safe_open(source, framework="pt", device="cpu") as handle:
                output_width = handle.get_slice(f"{PREFIX}.{name}.net.2.weight").get_shape()[0]
            if tuple(fitted.shape) != (output_width, kept):
                raise ValueError(f"{name}: reconstruction shape {tuple(fitted.shape)} != {(output_width, kept)}")
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        sd = {key: handle.get_tensor(key) for key in handle.keys()}
    config_all = json.loads(metadata.get("config", "{}"))
    config = config_all.setdefault("transformer", {})
    layers, d = int(config.get("num_layers", 48)), int(config.get("attention_head_dim", 128))
    a1, a2, ffn, a1_indices, a2_indices = [], [], [], [], []
    a1_active, a2_active, ffn_active = [], [], []
    for layer in range(layers):
        for kind, widths, identities in (("attn1", a1, a1_indices), ("attn2", a2, a2_indices)):
            key = f"{layer}.{kind}"
            original = sd[f"{PREFIX}.{layer}.{kind}.to_q.weight"].shape[0] // d
            keep = _keep(masks.get(key, [1.0] * original), key)
            if mode in ("compact", "compact_faithful"):
                if len(keep) < original:
                    _slice_heads(sd, layer, kind, keep, d)
                widths.append(len(keep) if mode == "compact" else original)
                identities.append(keep if mode == "compact" and len(keep) < original else None)
                if mode == "compact_faithful":
                    (a1_active if kind == "attn1" else a2_active).append(keep)
            else:
                widths.append(original)
                # Full-width Q/K already use the original RoPE order. An
                # identity list would gather frequencies in every layer.
                identities.append(None)
                (a1_active if kind == "attn1" else a2_active).append(keep)
        key = f"{layer}.ff"
        original = sd[f"{PREFIX}.{layer}.ff.net.0.proj.weight"].shape[0]
        keep = _keep(masks.get(key, [1.0] * original), key)
        if mode in ("compact", "compact_faithful"):
            _slice_ffn(sd, layer, keep, None if reconstruction is None else reconstruction.get(key))
            ffn.append(len(keep) if mode == "compact" else original)
            if mode == "compact_faithful":
                ffn_active.append(keep)
        else:
            ffn.append(original)
            ffn_active.append(keep)
    config.update({"per_layer_video_attn1_heads": a1, "per_layer_video_attn2_heads": a2,
                   "per_layer_ff_inner_dim": ffn, "per_layer_video_attn1_rope_head_indices": a1_indices,
                   "per_layer_video_attn2_rope_head_indices": a2_indices,
                   "video_pruning_preserve_qk_norm": True,
                   "video_pruning_shape_faithful": mode == "compact_faithful",
                   "pruning": {"task": "vae-refiner", "model_key": model_key, "mode": mode,
                               **(provenance_block or {})}})
    if mode != "compact":
        config.update({"per_layer_video_attn1_active_head_indices": a1_active,
                       "per_layer_video_attn2_active_head_indices": a2_active,
                       "per_layer_video_ffn_active_channels": ffn_active,
                       "video_pruning_select_active_heads": mode == "sparse"})
    metadata["config"] = json.dumps(config_all)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(output), metadata=metadata)
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--masks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--transformer-path")
    p.add_argument("--mode", choices=("masked_full", "sparse", "compact", "compact_faithful"), default="masked_full")
    p.add_argument("--historical-k2-mask", action="store_true",
                   help="Explicitly accept an older mask without native D0 task provenance")
    p.add_argument("--reconstruction-state", type=Path,
                   help="torch.save mapping '<layer>.ff' -> fitted fp32 (4096, kept_channels) projection.")
    args = p.parse_args()
    model = preflight.check(args.model, transformer_path=args.transformer_path)
    source = model.paths.transformer()
    stamp = provenance.stamp(model)
    masks, mask_sha256 = hooks.read_mask_artifact(
        args.masks, model_key=model.key, fingerprint=stamp["transformer_fingerprint"],
        widths=checkpoint_mask_widths(source),
        expected_task=None if args.historical_k2_mask else "whole_clip_d0",
    )
    reconstruction = (torch.load(args.reconstruction_state, map_location="cpu", weights_only=True)
                      if args.reconstruction_state else None)
    path = export(source, masks, args.output, model_key=model.key, reconstruction=reconstruction,
                  mode=args.mode,
                  provenance_block={**stamp, "task": "historical_k2" if args.historical_k2_mask else "whole_clip_d0",
                                    "source_transformer_fingerprint": stamp["transformer_fingerprint"],
                                    "mask_sha256": mask_sha256, "masks": str(args.masks),
                                    "reconstruction_state": (str(args.reconstruction_state)
                                                             if args.reconstruction_state else None)})
    print(json.dumps({"checkpoint": str(path), "fingerprint": provenance.checkpoint_fingerprint(path),
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)}))


if __name__ == "__main__":
    main()
