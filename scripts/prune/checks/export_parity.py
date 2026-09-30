"""Compare a native D0 structural export with its functional mask on saved inputs."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar import causal_core
from scripts.prune.core import artifacts, provenance, refine_task, session
from scripts.prune.data import chunk_states, corpus, records, whole_clip
from scripts.prune.evaluate import phase1_gates
from scripts.prune.score import export_pruned, hooks


def _difference(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"shape_match": False, "source_shape": list(a.shape), "export_shape": list(b.shape)}
    delta = a.float() - b.float()
    return {"shape_match": True, "max_abs": float(delta.abs().max()),
            "rel_l2": float(delta.square().sum().sqrt() / a.float().square().sum().sqrt().clamp_min(1e-12))}


def _historical_main() -> int:  # noqa: PLR0915
    p = argparse.ArgumentParser(
        description="Historical k2 functional-mask versus export parity on frozen records and two windows",
    )
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


def _forward(model, grid: causal_core.ClipGrid, modality: Modality, c0: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    with torch.no_grad():
        prediction, _ = model(video=modality, audio=None, perturbations=None)
        return grid.unpatchify_block(causal_core.with_clean_prefix(prediction, c0), grid.latent_frames).cpu()


def _native_main() -> int:  # noqa: PLR0915
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="Saved baseline whole-clip D0 rollout directory")
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--exported-checkpoint", type=Path, required=True)
    p.add_argument("--view", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", required=True)
    p.add_argument("--gpu-id", type=int, required=True)
    p.add_argument("--max-abs", type=float, default=0.02)
    args = p.parse_args()
    if args.max_abs < 0 or len(set(args.sigmas)) != len(args.sigmas):
        p.error("max-abs must be nonnegative and sigma levels distinct")
    baseline = whole_clip.load_manifest(args.baseline)
    rows = whole_clip.records(baseline)
    if any((args.view, sigma) not in rows for sigma in args.sigmas):
        p.error("view/sigma pair absent from the saved baseline")
    source_fingerprint = baseline["model"]["transformer_fingerprint"]
    source_path = Path(baseline["model"]["transformer_path"])
    if provenance.checkpoint_fingerprint(source_path) != source_fingerprint:
        raise ValueError("baseline checkpoint changed since saved D0 rollout")
    masks, mask_sha = hooks.read_mask_artifact(
        args.masks, model_key=baseline["model"]["model_key"], fingerprint=source_fingerprint,
        widths=export_pruned.checkpoint_mask_widths(source_path), expected_task=whole_clip.TASK, baseline=baseline,
    )
    hooks.require_native_heldout_scope(args.masks, view=args.view, sigmas=args.sigmas, baseline=baseline)
    with safe_open(args.exported_checkpoint, framework="pt", device="cpu") as handle:
        metadata = json.loads((handle.metadata() or {}).get("config", "{}"))
    pruning = metadata.get("transformer", {}).get("pruning", {})
    if (pruning.get("mask_sha256") != mask_sha or
            pruning.get("source_transformer_fingerprint") != source_fingerprint or
            pruning.get("task") != whole_clip.TASK):
        raise ValueError("exported checkpoint does not carry this native D0 mask and source")
    model_args = argparse.Namespace(model="2.5", gpu_id=args.gpu_id, seed=baseline["seed"])
    source = session.open_session(model_args, script="export_parity_d0",
                                  prompt=baseline["text_context"]["prompt"])
    exported = session.open_session(model_args, script="export_parity_d0",
                                    transformer_path=args.exported_checkpoint,
                                    prompt=baseline["text_context"]["prompt"])
    if source.stamp()["transformer_fingerprint"] != source_fingerprint:
        raise ValueError("baseline checkpoint changed since saved D0 rollout")
    with source.transformer() as model:
        heads = {name: torch.tensor(values, device=source.device) for name, values in masks.items()
                 if not name.endswith(".ff")}
        ffns = {name: torch.tensor(values, device=source.device) for name, values in masks.items()
                if name.endswith(".ff")}
        functional = {}
        torch.cuda.reset_peak_memory_stats(source.device)
        with ExitStack() as stack:
            if heads:
                stack.enter_context(hooks.attach_head_masks(model, heads, requires_grad=False))
            if ffns:
                stack.enter_context(hooks.attach_ffn_masks(model, ffns, requires_grad=False))
            for sigma in args.sigmas:
                grid, modality, c0, _ = whole_clip.build_input(
                    args.baseline, baseline, view=args.view, sigma=sigma, current=source,
                )
                functional[sigma] = _forward(model, grid, modality, c0)
                del grid, modality, c0
        source_peak = torch.cuda.max_memory_allocated(source.device) / 2**30
    comparisons = []
    with exported.transformer(args.exported_checkpoint) as model:
        torch.cuda.reset_peak_memory_stats(exported.device)
        for sigma in args.sigmas:
            grid, modality, c0, _ = whole_clip.build_input(
                args.baseline, baseline, view=args.view, sigma=sigma, current=exported,
            )
            measured = _forward(model, grid, modality, c0)
            comparisons.append({"sigma": sigma, **_difference(functional[sigma], measured)})
            del grid, modality, c0, measured
        export_peak = torch.cuda.max_memory_allocated(exported.device) / 2**30
    passed = all(row.get("shape_match") and row["max_abs"] <= args.max_abs for row in comparisons)
    out = artifacts.run_dir(source.key, "d0-export-parity", script="export_parity", argv=sys.argv[1:])
    result = {"task": whole_clip.TASK, "provenance": source.stamp(),
              "baseline_manifest": str((args.baseline / "manifest.json").resolve()),
              "exported_fingerprint": exported.stamp()["transformer_fingerprint"],
              "mask_sha256": mask_sha, "view": args.view, "sigmas": args.sigmas,
              "max_abs_tolerance": args.max_abs, "comparisons": comparisons,
              "functional_peak_allocated_gib": source_peak,
              "export_peak_allocated_gib": export_peak, "pass": passed}
    (out / "export_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(out / "export_parity.json")
    return 0 if passed else 1


def main() -> int:
    if "--historical-k2" in sys.argv[1:]:
        sys.argv.remove("--historical-k2")
        return _historical_main()
    return _native_main()


if __name__ == "__main__":
    raise SystemExit(main())
