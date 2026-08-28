"""Export structured refiner pruning masks as a self-describing safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.prune import model_registry, preflight, provenance

PREFIX = "model.diffusion_model.transformer_blocks"


def _indices(keep: list[int], head_dim: int) -> torch.Tensor:
    return (torch.tensor(keep, dtype=torch.long)[:, None] * head_dim + torch.arange(head_dim)[None]).reshape(-1)


def _keep(mask: list[float], label: str) -> list[int]:
    out = [i for i, value in enumerate(mask) if value != 0]
    if not out:
        raise ValueError(f"{label}: refusing to export an empty branch")
    return out


def _slice_heads(sd: dict[str, torch.Tensor], layer: int, kind: str, keep: list[int], d: int) -> None:
    idx = _indices(keep, d); b = f"{PREFIX}.{layer}.{kind}"
    for proj in ("to_q", "to_k", "to_v"):
        sd[f"{b}.{proj}.weight"] = sd[f"{b}.{proj}.weight"][idx]
        key = f"{b}.{proj}.bias"
        if key in sd: sd[key] = sd[key][idx]
    for norm in ("q_norm", "k_norm"):
        sd[f"{b}.{norm}.weight"] = sd[f"{b}.{norm}.weight"][idx]
    sd[f"{b}.to_out.0.weight"] = sd[f"{b}.to_out.0.weight"][:, idx]
    for suffix in ("weight", "bias"):
        key = f"{b}.to_gate_logits.{suffix}"
        if key in sd: sd[key] = sd[key][keep]


def _slice_ffn(sd: dict[str, torch.Tensor], layer: int, keep: list[int], fitted: torch.Tensor | None = None) -> None:
    b = f"{PREFIX}.{layer}.ff"; index = torch.tensor(keep, dtype=torch.long)
    sd[f"{b}.net.0.proj.weight"] = sd[f"{b}.net.0.proj.weight"][index]
    key = f"{b}.net.0.proj.bias"
    if key in sd: sd[key] = sd[key][index]
    key = f"{b}.net.2.weight"
    if fitted is not None:
        expected = (sd[key].shape[0], len(keep))
        if tuple(fitted.shape) != expected:
            raise ValueError(f"{b}: fitted weight {tuple(fitted.shape)} != {expected}")
        sd[key] = fitted.to(dtype=sd[key].dtype, device="cpu").contiguous()
    else:
        sd[key] = sd[key][:, index]


def export(source: str | Path, masks: dict, output: str | Path, *, model_key: str,
           reconstruction: dict[str, torch.Tensor] | None = None, provenance_block: dict | None = None) -> Path:
    """Perform checkpoint-space surgery; masks are keyed ``'0.attn1'`` / ``'0.ff'``."""
    source, output = Path(source), Path(output)
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        sd = {key: handle.get_tensor(key) for key in handle.keys()}
    config_all = json.loads(metadata.get("config", "{}")); config = config_all.setdefault("transformer", {})
    layers, d = int(config.get("num_layers", 48)), int(config.get("attention_head_dim", 128))
    a1, a2, ffn, a1_indices, a2_indices = [], [], [], [], []
    for layer in range(layers):
        for kind, widths, identities in (("attn1", a1, a1_indices), ("attn2", a2, a2_indices)):
            key = f"{layer}.{kind}"; original = sd[f"{PREFIX}.{layer}.{kind}.to_q.weight"].shape[0] // d
            keep = _keep(masks.get(key, [1.0] * original), key)
            _slice_heads(sd, layer, kind, keep, d); widths.append(len(keep)); identities.append(keep)
        key = f"{layer}.ff"; original = sd[f"{PREFIX}.{layer}.ff.net.0.proj.weight"].shape[0]
        keep = _keep(masks.get(key, [1.0] * original), key)
        _slice_ffn(sd, layer, keep, None if reconstruction is None else reconstruction.get(key)); ffn.append(len(keep))
    config.update({"per_layer_video_attn1_heads": a1, "per_layer_video_attn2_heads": a2,
                   "per_layer_ff_inner_dim": ffn, "per_layer_video_attn1_rope_head_indices": a1_indices,
                   "per_layer_video_attn2_rope_head_indices": a2_indices,
                   "pruning": {"task": "vae-refiner", "model_key": model_key, **(provenance_block or {})}})
    metadata["config"] = json.dumps(config_all)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(output), metadata=metadata)
    return output


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--masks", required=True); p.add_argument("--output", required=True); p.add_argument("--transformer-path")
    p.add_argument("--reconstruction-state", type=Path,
                   help="torch.save mapping '<layer>.ff' -> fitted fp32 (4096, kept_channels) projection.")
    args = p.parse_args(); model = preflight.check(args.model, transformer_path=args.transformer_path)
    masks = json.loads(Path(args.masks).read_text()); masks = masks.get("masks", masks)
    reconstruction = torch.load(args.reconstruction_state, map_location="cpu", weights_only=True) if args.reconstruction_state else None
    path = export(model.paths.transformer(), masks, args.output, model_key=model.key, reconstruction=reconstruction,
                  provenance_block=provenance.stamp(model, masks=str(args.masks), reconstruction_state=str(args.reconstruction_state) if args.reconstruction_state else None))
    print(json.dumps({"checkpoint": str(path), "fingerprint": provenance.checkpoint_fingerprint(path)}))


if __name__ == "__main__": main()
