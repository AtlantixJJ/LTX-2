"""Evaluate functionally removed attention heads against the unpruned refiner.

This is the Phase 2 candidate gate before Phase 4 exports narrower tensors: a
zero head mask at ``to_out[0]`` is numerically equivalent to deleting that head
from the output projection, while preserving the original module shape.  It
records chunk-token latent deviation, source-target T0 change, and (by default)
an aligned source | unpruned | masked MP4 for the first evaluated record.

Example:
    conda run -n ltx python -m scripts.prune.head_ablation_eval --model 2.5 --gpu-id 0 \
        --remove-head 7.attn2:14 --split held_out --max-records 8
"""

from __future__ import annotations

import sys
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.blocks import VideoDecoder
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import post_process_latent

from scripts.prune import artifacts, chunk_states, hooks, losses, metrics, model_registry, provenance, records, refine_task, session
from scripts.prune.model_registry import WORKSPACE_ROOT
from scripts.prune.session import DTYPE


def _parse_head(value: str) -> tuple[str, int]:
    try:
        name, index_text = value.rsplit(":", 1)
        layer_text, kind = name.split(".", 1)
        if kind not in {"attn1", "attn2"}:
            raise ValueError
        int(layer_text)
        index = int(index_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("head must have form '<layer>.attn1|attn2:<head>', e.g. 7.attn2:14") from exc
    return name, index


def _mask_for(transformer, removed: list[tuple[str, int]], device: torch.device) -> dict[str, torch.Tensor]:
    attention = dict(hooks.iter_video_attention(transformer))
    masks: dict[str, torch.Tensor] = {}
    for name, head in removed:
        if name not in attention:
            raise ValueError(f"unknown attention module {name!r}")
        if not 0 <= head < attention[name].heads:
            raise ValueError(f"{name}: head {head} is outside [0, {attention[name].heads})")
        masks.setdefault(name, torch.ones(attention[name].heads, device=device))[head] = 0
    return masks


def _tools(model, state, token_latent: torch.Tensor) -> VideoLatentTools:
    """Reconstruct the unpatchifier geometry from the serialized token positions."""
    positions = state.positions
    if positions.shape != (token_latent.shape[0], 3, token_latent.shape[1], 2):
        raise ValueError(f"unexpected position shape {tuple(positions.shape)} for tokens {tuple(token_latent.shape)}")
    frames, height, width = (int(torch.unique(positions[0, axis, :, 0]).numel()) for axis in range(3))
    if frames * height * width != token_latent.shape[1]:
        raise ValueError(f"position grid {(frames, height, width)} does not cover {token_latent.shape[1]} tokens")
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1),
        VideoLatentShape(token_latent.shape[0], token_latent.shape[-1], frames, height, width),
        24.0,
        scale_factors=model.scale_factors,
    )


def _decode_token_latent(model, state, token_latent: torch.Tensor, decoder, device: torch.device) -> torch.Tensor:
    """Restore frozen tokens, unpatchify, and decode a token-space x0 prediction."""
    restored = post_process_latent(token_latent, state.denoise_mask, state.clean_latent)
    tools = _tools(model, state, restored)
    latent = tools.unpatchify(tools.clear_conditioning(replace(state, latent=restored))).latent
    decoded = torch.cat(list(decoder.decode_video(latent.to(device=device, dtype=DTYPE), None, None)), dim=0).float()
    return decoded.clamp(0, 1).cpu().permute(0, 3, 1, 2)


def _mean(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    session.add_model_args(parser)
    session.add_record_args(parser, default_split="held_out")
    parser.add_argument("--remove-head", type=_parse_head, action="append", required=True)
    parser.add_argument("--no-render", action="store_true", help="Skip the first-record latent/video artifacts.")
    args = parser.parse_args()

    s = session.open_session(args, script="head_ablation_eval")
    model, device = s.model, s.device
    states_root = s.states_root(args.states)
    paths = records.select(states_root, split=args.split, limit=args.max_records)

    rows: list[dict] = []
    render_data = None
    with s.transformer() as transformer:
        masks = _mask_for(transformer, args.remove_head, device)
        for path in paths:
            state, target, meta = chunk_states.load_record(path, device)
            chunk_mask = chunk_states.chunk_token_mask(state, meta)
            baseline_result, _ = s.denoiser(transformer, state, None, s.sigmas, meta.step_index)
            if baseline_result is None:
                raise RuntimeError("unpruned denoiser returned no video result")
            with hooks.attach_head_masks(transformer, masks, requires_grad=False):
                candidate_result, _ = s.denoiser(transformer, state, None, s.sigmas, meta.step_index)
            if candidate_result is None:
                raise RuntimeError("masked denoiser returned no video result")
            baseline, candidate = baseline_result.denoised, candidate_result.denoised
            base_t0 = float(losses.rel_l2(baseline, target, state, chunk_mask))
            candidate_t0 = float(losses.rel_l2(candidate, target, state, chunk_mask))
            row = {
                "record": path.name,
                "clip": meta.clip,
                "split": meta.split,
                "family": meta.family,
                "step_index": meta.step_index,
                "chunk_latent_frames": meta.chunk_latent_frames,
                "baseline_t0_rel_l2": base_t0,
                "candidate_t0_rel_l2": candidate_t0,
                "candidate_minus_baseline_t0": candidate_t0 - base_t0,
                "candidate_vs_baseline_rel_l2": float(losses.rel_l2(candidate, baseline, state, chunk_mask)),
                "candidate_vs_baseline_max_abs": float(((candidate - baseline).abs() * chunk_mask).max()),
            }
            rows.append(row)
            print(
                f"[ablation] {path.name}: output ΔrelL2={row['candidate_vs_baseline_rel_l2']:.3e}, "
                f"T0 Δ={row['candidate_minus_baseline_t0']:.3e}",
                flush=True,
            )
            if render_data is None and not args.no_render:
                render_data = (path.name, state, target.cpu(), baseline.cpu(), candidate.cpu(), meta.fps)
    output = artifacts.run_dir(s.key, "head-ablation", script="head_ablation_eval", argv=sys.argv[1:])
    result = {
        "provenance": s.stamp(),
        "removed_heads": [{"name": name, "head": head} for name, head in args.remove_head],
        "records": rows,
        "summary": {
            "records": len(rows),
            "candidate_vs_baseline_rel_l2_mean": _mean(rows, "candidate_vs_baseline_rel_l2"),
            "candidate_minus_baseline_t0_mean": _mean(rows, "candidate_minus_baseline_t0"),
            "candidate_vs_baseline_max_abs_max": max(row["candidate_vs_baseline_max_abs"] for row in rows),
        },
    }
    torch.save({"removed_heads": result["removed_heads"], "record": render_data[0] if render_data else None,
                "target": render_data[2] if render_data else None, "baseline": render_data[3] if render_data else None,
                "candidate": render_data[4] if render_data else None}, output / "latents.pt")

    if render_data is not None:
        record, state, target, baseline, candidate, fps = render_data
        holder = VideoDecoder(model.paths.video_vae(), DTYPE, device)
        with torch.no_grad(), gpu_model(holder._decoder_builder.build(device=device, dtype=DTYPE).eval()) as decoder:
            source_px = _decode_token_latent(model, state, target.to(device), decoder, device)
            baseline_px = _decode_token_latent(model, state, baseline.to(device), decoder, device)
            candidate_px = _decode_token_latent(model, state, candidate.to(device), decoder, device)
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        grid = metrics.t3_grid([(record, source_px, baseline_px, candidate_px)], figures / "ablation_grid.png")
        # meta.fps, not the metrics.t3_video default: this is the review MP4's playback
        # rate for a real corpus clip, and 41 of 44 clips are 30 fps, not 24.
        video = metrics.t3_video(source_px, baseline_px, candidate_px, figures / "ablation_comparison.mp4", fps=fps)
        (figures / "INDEX.md").write_text(
            "# Head ablation artifacts\n\n"
            "- `ablation_grid.png`: VAE source target | unpruned x0 | masked-head x0.\n"
            "- `ablation_comparison.mp4`: aligned source target | unpruned | masked-head output.\n"
        )
        result["artifacts"] = {"grid": str(grid), "video": str(video), "record": record}

    path = output / "head_ablation_eval.json"
    path.write_text(json.dumps(result, indent=2))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
