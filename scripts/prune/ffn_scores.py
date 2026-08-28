"""Phase 3 structured FFN-channel scoring and training-free reconstruction."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch

from scripts.prune import chunk_states, hooks, losses, lstsq, prune_schedule
from scripts.prune import model_registry, preflight, prompt_cache, provenance, refine_task
from scripts.prune.model_registry import WORKSPACE_ROOT
from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.denoisers import SimpleDenoiser

DTYPE = torch.bfloat16


@torch.no_grad()
def channel_rms(model, records: list[Path], denoiser, sigmas: torch.Tensor, device: torch.device,
                active_masks: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Fresh-chunk RMS of post-GELU channels, collected from frozen records."""
    sq = {name: torch.zeros(ff.net[2].weight.shape[1], device=device, dtype=torch.float32) for name, ff in hooks.iter_video_ffn(model)}
    count = {name: 0 for name in sq}
    token_mask: torch.Tensor | None = None

    def observe(name, activation, _ff):
        if token_mask is None:
            raise RuntimeError("FFN hook ran without a chunk token mask")
        values = activation[token_mask[..., 0].bool()].float()
        sq[name].add_(values.square().sum(0))
        count[name] += values.shape[0]

    with hooks.attach_ffn_masks(model, active_masks, requires_grad=False), hooks.collect_activations(model, "ffn", observe):
        for path in records:
            state, _, meta = chunk_states.load_record(path, device)
            token_mask = chunk_states.chunk_token_mask(state, meta)
            result, _ = denoiser(model, state, None, sigmas, meta.step_index)
            if result is None:
                raise RuntimeError("video denoiser unexpectedly returned no result")
    return {name: (value / max(count[name], 1)).sqrt().cpu() for name, value in sq.items()}


def channel_scores(model, rms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Activation-aware output-channel score ``RMS(a_j) * ||W_out[:, j]||``."""
    result = {}
    for name, ff in hooks.iter_video_ffn(model):
        result[name] = rms[name].float() * ff.net[2].weight.detach().float().norm(dim=0).cpu()
    return result


def masks_from_scores(scores: dict[str, torch.Tensor], target_sparsity: float, *, min_keep: int = 1) -> dict[str, torch.Tensor]:
    """Per-layer structured allocation, while keeping each FFN executable.

    A raw global ranking can erase almost an entire layer because FFN activation
    scales differ wildly across AdaLN gates.  Until branch-contribution ratios
    are collected, equal *fractional* allocation is the safe non-degenerate
    baseline; it still chooses channels independently within every layer.
    """
    if not 0 <= target_sparsity < 1:
        raise ValueError("target_sparsity must be in [0, 1)")
    masks = {}
    for name, values in scores.items():
        take = min(round(values.numel() * target_sparsity), values.numel() - min_keep)
        mask = torch.ones_like(values, dtype=torch.float32)
        if take:
            mask[torch.argsort(values)[:take]] = 0
        masks[name] = mask
    return masks


@torch.no_grad()
def reconstruct_ffn_projection(model, name: str, keep: torch.Tensor, records: list[Path], denoiser, sigmas: torch.Tensor,
                               device: torch.device, ridge: float = 1e-4) -> torch.Tensor:
    """Fit and install dense-equivalent reconstructed FFN output weights."""
    ff = dict(hooks.iter_video_ffn(model))[name]
    keep = keep.to(device=device, dtype=torch.long)
    acc, add = lstsq.ffn_accumulator(ff, keep)
    token_mask: torch.Tensor | None = None

    def observe(observed, activation, _ff):
        if observed == name:
            if token_mask is None:
                raise RuntimeError("FFN reconstruction hook ran without a chunk token mask")
            add(activation, token_mask)

    with hooks.collect_activations(model, "ffn", observe):
        for path in records:
            state, _, meta = chunk_states.load_record(path, device)
            token_mask = chunk_states.chunk_token_mask(state, meta)
            result, _ = denoiser(model, state, None, sigmas, meta.step_index)
            if result is None:
                raise RuntimeError("video denoiser unexpectedly returned no result")
    fitted = acc.solve(ridge).to(dtype=ff.net[2].weight.dtype)
    expanded = torch.zeros_like(ff.net[2].weight)
    expanded[:, keep] = fitted
    ff.net[2].weight.copy_(expanded)
    return fitted


@torch.no_grad()
def masked_t0(model, masks: dict[str, torch.Tensor], records: list[Path], denoiser, sigmas: torch.Tensor,
              device: torch.device) -> float:
    """Cheap real-state T0 for a candidate FFN mask, before structural export."""
    values = []
    with hooks.attach_ffn_masks(model, masks, requires_grad=False):
        for path in records:
            state, target, meta = chunk_states.load_record(path, device)
            result, _ = denoiser(model, state, None, sigmas, meta.step_index)
            if result is None:
                raise RuntimeError("video denoiser unexpectedly returned no result")
            values.append(float(losses.rel_l2(result.denoised, target, state, chunk_states.chunk_token_mask(state, meta))))
    return sum(values) / len(values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    ap.add_argument("--gpu-id", type=int, default=0); ap.add_argument("--states", type=Path)
    ap.add_argument("--split", choices=("calibration", "held_out"), default="calibration")
    ap.add_argument("--max-records", type=int); ap.add_argument("--target-sparsity", type=float, default=0.5)
    ap.add_argument("--evaluate-sparsities", type=float, nargs="*", default=(),
                    help="Real masked T0 sweep, performed before any structural export.")
    ap.add_argument("--iterative-rounds", type=int, default=0,
                    help="Run iterative masked recollection/pruning and persist its mask history.")
    ap.add_argument("--reconstruct-layers", type=int, nargs="*", default=(),
                    help="Layer numbers to ridge-reconstruct for the selected target mask (real activation solve).")
    ap.add_argument("--save-reconstruction", type=Path,
                    help="Write fitted projections for export_pruned --reconstruction-state.")
    args = ap.parse_args(); model = preflight.check(args.model, sampler="euler", gpu_id=args.gpu_id)
    root = args.states or (WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "calibration")
    paths = list(chunk_states.iter_records(root, args.split)); paths = paths[:args.max_records] if args.max_records else paths
    if not paths: raise SystemExit(f"No {args.split} records under {root}")
    device = torch.device(f"cuda:{args.gpu_id}"); context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    sigmas = torch.tensor(refine_task.schedule_for(model.sigmas, refine_task.K_STEP), dtype=torch.float32, device=device)
    stage = DiffusionStage.from_checkpoint(model.paths.transformer(), DTYPE, device, model_configurator=LTXVideoOnlyModelConfigurator, scale_factors=model.scale_factors)
    with stage._transformer_ctx() as transformer:
        denoiser = SimpleDenoiser(context, None)
        rms = channel_rms(transformer, paths, denoiser, sigmas, device); scores = channel_scores(transformer, rms)
        masks = masks_from_scores(scores, args.target_sparsity)
        evaluation = {str(s): masked_t0(transformer, masks_from_scores(scores, s), paths, denoiser, sigmas, device)
                      for s in args.evaluate_sparsities}
        iterative = None
        if args.iterative_rounds:
            def rescore(active):
                return channel_scores(transformer, channel_rms(transformer, paths, denoiser, sigmas, device, active))
            iterative_masks, history = prune_schedule.iterative_ffn_masks(
                transformer, target_sparsity=args.target_sparsity, rounds=args.iterative_rounds, rescore=rescore,
            )
            iterative = {"rounds": args.iterative_rounds, "masks": {k: v.tolist() for k, v in iterative_masks.items()},
                         "history": history, "masked_t0_rel_l2": masked_t0(transformer, iterative_masks, paths, denoiser, sigmas, device)}
        reconstruction = None
        reconstruction_tensors: dict[str, torch.Tensor] = {}
        if args.reconstruct_layers:
            fitted = {}
            for layer in args.reconstruct_layers:
                name = f"{layer}.ff"
                if name not in masks:
                    raise ValueError(f"unknown FFN layer {layer}")
                keep = torch.nonzero(masks[name] != 0).flatten()
                naive_t0 = masked_t0(transformer, {name: masks[name]}, paths, denoiser, sigmas, device)
                projection = reconstruct_ffn_projection(transformer, name, keep, paths, denoiser, sigmas, device)
                fitted[name] = list(projection.shape)
                reconstruction_tensors[name] = projection.cpu()
                fitted[name + ".isolated_t0"] = [naive_t0, masked_t0(transformer, {name: masks[name]}, paths, denoiser, sigmas, device)]
            reconstruction = {"layers": list(args.reconstruct_layers), "fitted_shapes": fitted,
                              "masked_t0_rel_l2_after_all_layers_masked": masked_t0(transformer, masks, paths, denoiser, sigmas, device)}
    out = WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / provenance.run_id("ffn-scores"); out.mkdir(parents=True, exist_ok=True)
    report = {"provenance": provenance.stamp(model, device, script="ffn_scores"), "records": [x.name for x in paths],
              "target_sparsity": args.target_sparsity, "scores": {k: v.tolist() for k,v in scores.items()}, "masks": {k: v.tolist() for k,v in masks.items()},
              "kept": {k: int(v.sum()) for k,v in masks.items()}, "masked_t0_rel_l2": evaluation, "iterative": iterative,
              "reconstruction": reconstruction}
    if args.save_reconstruction:
        if not reconstruction_tensors:
            raise ValueError("--save-reconstruction requires --reconstruct-layers")
        args.save_reconstruction.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reconstruction_tensors, args.save_reconstruction)
        report["reconstruction_state"] = str(args.save_reconstruction)
    (out / "ffn_scores.json").write_text(json.dumps(report)); print(out / "ffn_scores.json"); return 0


if __name__ == "__main__": raise SystemExit(main())
