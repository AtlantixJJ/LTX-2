"""Score dense-export pruning units on whole-clip bidirectional D0 capture inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.prune.core import provenance, session
from scripts.prune.data import whole_clip
from scripts.prune.score import estimators, hooks


def _sample_indices(tokens: int, tokens_per_frame: int, stride: int, device: torch.device) -> torch.Tensor:
    """Uniform spatial samples on every generated latent frame; omit clean frame zero."""
    if tokens % tokens_per_frame or tokens <= tokens_per_frame or stride < 1:
        raise ValueError("invalid token grid or stride")
    frame_starts = torch.arange(tokens_per_frame, tokens, tokens_per_frame, device=device)
    spatial = torch.arange(0, tokens_per_frame, stride, device=device)
    return (frame_starts[:, None] + spatial[None, :]).reshape(-1)


def _weight_norms(model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:  # noqa: ANN001
    head, ffn = {}, {}
    for name, attn in hooks.iter_video_attention(model):
        width = attn.heads * attn.dim_head
        weight = attn.to_out[0].weight.detach()
        if weight.shape[1] != width:
            raise ValueError(f"{name}: projection width does not match heads")
        shaped = weight.float().reshape(weight.shape[0], attn.heads, attn.dim_head)
        head[name] = shaped.square().sum((0, 2)).sqrt().cpu()
    for name, block in hooks.iter_video_ffn(model):
        ffn[name] = block.net[2].weight.detach().float().square().sum(0).sqrt().cpu()
    return head, ffn


def score(baseline: Path, *, gpu_id: int, calibration_views: list[str],  # noqa: PLR0915
          sigma_levels: list[float], head_fraction: float, ffn_fraction: float,
          spatial_stride: int, output: Path) -> dict:
    base = whole_clip.load_manifest(baseline)
    rows = whole_clip.records(base)
    if not calibration_views or not sigma_levels:
        raise ValueError("at least one calibration view and sigma are needed")
    if len(set(calibration_views)) != len(calibration_views) or len(set(sigma_levels)) != len(sigma_levels):
        raise ValueError("calibration views and sigma levels must be distinct")
    keys = [(view, sigma) for view in calibration_views for sigma in sigma_levels]
    if any(key not in rows for key in keys):
        raise ValueError("calibration view/sigma is absent from the saved baseline")
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} does not exist")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < 44 * 2**30:
        raise ValueError(f"GPU {gpu_id} needs at least 44 GiB free")
    args = argparse.Namespace(model="2.5", gpu_id=gpu_id, seed=base["seed"])
    current = session.open_session(args, script="prune.score.whole_clip_d0_scores",
                                   prompt=base["text_context"]["prompt"])
    checkpoint = Path(base["model"]["transformer_path"])
    if provenance.checkpoint_fingerprint(checkpoint) != base["model"]["transformer_fingerprint"]:
        raise ValueError("baseline checkpoint changed since saved rollout")
    contributions: dict[str, torch.Tensor] = {}
    runs = []
    with current.transformer(checkpoint) as model:
        head_norm, ffn_norm = _weight_norms(model)
        head_squares = {name: torch.zeros_like(value) for name, value in head_norm.items()}
        ffn_squares = {name: torch.zeros_like(value) for name, value in ffn_norm.items()}
        count = 0
        indices = None

        def head_hook(name, activation, attn) -> None:  # noqa: ANN001
            assert indices is not None
            sample = activation[:, indices].reshape(-1, attn.heads, attn.dim_head).float()
            head_squares[name].add_(sample.square().mean((0, 2)).cpu())

        def ffn_hook(name, activation, _block) -> None:  # noqa: ANN001
            assert indices is not None
            sample = activation[:, indices].float()
            ffn_squares[name].add_(sample.square().mean((0, 1)).cpu())

        with hooks.collect_activations(model, "head", head_hook), hooks.collect_activations(model, "ffn", ffn_hook):
            for view, sigma in keys:
                grid, modality, c0, b = whole_clip.build_input(
                    baseline, base, view=view, sigma=sigma, current=current,
                )
                indices = _sample_indices(
                    modality.latent.shape[1], grid.tokens_per_latent_frame, spatial_stride, device,
                )
                with torch.no_grad():
                    prediction, _ = model(video=modality, audio=None, perturbations=None)
                    prediction = grid.unpatchify_block(
                        torch.cat((c0, prediction[:, c0.shape[1]:]), dim=1), grid.latent_frames,
                    )
                recorded = torch.load(whole_clip.latent_path(baseline, b), map_location="cpu", weights_only=True)
                max_abs = float((recorded.float() - prediction.cpu().float()).abs().max())
                if max_abs > 0.02:
                    raise ValueError(f"scoring forward differs from saved baseline for {view}, {sigma}: {max_abs}")
                runs.append({"view": view, "sigma": sigma, "sample_tokens": int(indices.numel()),
                             "saved_rollout_max_abs": max_abs})
                count += 1
                del grid, modality, c0, prediction, recorded, indices
                indices = None
                torch.cuda.empty_cache()
                print(f"Scored {view} sigma={sigma}", flush=True)
        contributions = {
            **{name: estimators.rms_projection_scores(head_squares[name], head_norm[name], count)
               for name in head_squares},
            **{name: estimators.rms_projection_scores(ffn_squares[name], ffn_norm[name], count)
               for name in ffn_squares},
        }
    masks = {**estimators.fractional_masks({k: contributions[k] for k in head_squares}, head_fraction),
             **estimators.fractional_masks({k: contributions[k] for k in ffn_squares}, ffn_fraction)}
    report = {
        "candidate_format": "whole_clip_d0_mask_v1",
        "provenance": whole_clip.native_provenance(baseline, base, calibration_views, sigma_levels),
        "method": "sampled post-activation RMS times output-projection column norm; per-layer/branch allocation",
        "head_fraction": head_fraction, "ffn_fraction": ffn_fraction,
        "spatial_token_stride": spatial_stride, "runs": runs,
        "scores": {name: value.tolist() for name, value in contributions.items()},
        "masks": masks,
        "removed_heads": sum(len(v) - sum(v) for k, v in masks.items() if not k.endswith(".ff")),
        "removed_ffn_channels": sum(len(v) - sum(v) for k, v in masks.items() if k.endswith(".ff")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--view", action="append", required=True, help="Calibration capture view; repeat")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.725, 0.909375, 1.0])
    parser.add_argument("--head-fraction", type=float, default=0.10)
    parser.add_argument("--ffn-fraction", type=float, default=0.10)
    parser.add_argument("--spatial-stride", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(
        args.baseline, gpu_id=args.gpu_id,
        calibration_views=args.view, sigma_levels=args.sigmas,
        head_fraction=args.head_fraction, ffn_fraction=args.ffn_fraction,
        spatial_stride=args.spatial_stride, output=args.output,
    )
    print(json.dumps({"output": str(args.output), "runs": len(result["runs"]),
                      "removed_heads": result["removed_heads"],
                      "removed_ffn_channels": result["removed_ffn_channels"]}), flush=True)


if __name__ == "__main__":
    main()
