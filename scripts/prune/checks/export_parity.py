"""Compare a structural export with its functional mask on frozen states and two windows."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch

from scripts.prune.core import artifacts, provenance, refine_task, session
from scripts.prune.data import chunk_states, corpus, records
from scripts.prune.evaluate import phase1_gates
from scripts.prune.score import hooks


def _difference(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"shape_match": False, "source_shape": list(a.shape), "export_shape": list(b.shape)}
    delta = a.float() - b.float()
    return {"shape_match": True, "max_abs": float(delta.abs().max()),
            "rel_l2": float(delta.square().sum().sqrt() / a.float().square().sum().sqrt().clamp_min(1e-12))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    session.add_model_args(p)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--exported-checkpoint", type=Path, required=True)
    p.add_argument("--states", type=Path)
    p.add_argument("--video", type=Path, help="Explicit source when the historical corpus is unavailable.")
    p.add_argument("--expected-source-sha256", help="Required content pin for --video.")
    p.add_argument("--max-abs", type=float, default=0.02)
    args = p.parse_args()
    if args.video is not None:
        if not args.video.is_file() or not args.expected_source_sha256:
            p.error("--video requires an existing file and --expected-source-sha256")
        if provenance.file_sha256(args.video) != args.expected_source_sha256:
            p.error("external source SHA256 differs")
    elif args.expected_source_sha256:
        p.error("--expected-source-sha256 requires --video")
    source = session.open_session(args, script="export_parity")
    exported = session.open_session(args, script="export_parity", transformer_path=args.exported_checkpoint)
    geometry = source.geometry()
    path = records.select(source.states_root(args.states), split="held_out", limit=1)[0]
    state, _, meta = chunk_states.load_record(path, source.device)
    clip = args.video or corpus.pick_clip(geometry, 2, key=source.key, prefer="held_out", longest=True)
    if corpus.frame_count(clip) < geometry.window_frames + geometry.stride_frames:
        p.error(f"{clip} is too short for two windows")
    windows = geometry.plan(corpus.frame_count(clip))[:2]
    latents, _, fps = phase1_gates._encode_windows(source.model, clip, windows, source.device)
    sigmas = refine_task.schedule_for(source.model.sigmas, refine_task.K_STEP)
    with source.transformer() as transformer:
        torch.cuda.reset_peak_memory_stats(source.device)
        widths = {name: attention.heads for name, attention in hooks.iter_video_attention(transformer)}
        widths.update({name: ff.net[2].weight.shape[1] for name, ff in hooks.iter_video_ffn(transformer)})
        masks, mask_sha256 = hooks.read_mask_artifact(
            args.masks, model_key=source.key,
            fingerprint=source.stamp()["transformer_fingerprint"], widths=widths
        )
        head_masks = {name: torch.tensor(values, device=source.device) for name, values in masks.items()
                      if not name.endswith(".ff")}
        ffn_masks = {name: torch.tensor(values, device=source.device) for name, values in masks.items()
                     if name.endswith(".ff")}
        source_times: list[dict] = []
        with ExitStack() as stack:
            if head_masks:
                stack.enter_context(hooks.attach_head_masks(transformer, head_masks, requires_grad=False))
            if ffn_masks:
                stack.enter_context(hooks.attach_ffn_masks(transformer, ffn_masks, requires_grad=False))
            source_result, _ = source.denoiser(transformer, state, None, source.sigmas, meta.step_index)
            source_windows = phase1_gates._rollout(
                transformer, source.denoiser, latents, geometry, sigmas, fps,
                seed=args.seed, device=source.device, timing_rows=source_times
            )
        source_output = source_result.denoised.cpu()
        source_peak_gib = torch.cuda.max_memory_allocated(source.device) / (1024 ** 3)
    with exported.transformer(args.exported_checkpoint) as transformer:
        torch.cuda.reset_peak_memory_stats(exported.device)
        export_times: list[dict] = []
        export_result, _ = exported.denoiser(transformer, state, None, exported.sigmas, meta.step_index)
        export_windows = phase1_gates._rollout(
            transformer, exported.denoiser, latents, geometry, sigmas, fps,
            seed=args.seed, device=exported.device, timing_rows=export_times
        )
        export_output = export_result.denoised.cpu()
        export_peak_gib = torch.cuda.max_memory_allocated(exported.device) / (1024 ** 3)
    comparisons = [_difference(source_output, export_output)]
    comparisons.extend(_difference(a, b) for a, b in zip(source_windows, export_windows, strict=True))
    passed = all(row.get("shape_match") and row["max_abs"] <= args.max_abs for row in comparisons)
    out = artifacts.run_dir(source.key, "export-parity", script="export_parity", argv=sys.argv[1:])
    report = {"provenance": source.stamp(), "exported_fingerprint": exported.stamp()["transformer_fingerprint"],
              "mask_sha256": mask_sha256, "record": path.name, "clip": str(clip),
              "geometry": geometry.as_dict(), "max_abs_tolerance": args.max_abs,
              "comparisons": comparisons, "source_times": source_times,
              "export_times": export_times, "source_peak_allocated_gib": source_peak_gib,
              "export_peak_allocated_gib": export_peak_gib, "pass": passed}
    (out / "export_parity.json").write_text(json.dumps(report, indent=2))
    print(out / "export_parity.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
