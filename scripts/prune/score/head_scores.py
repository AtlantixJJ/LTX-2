"""Phase 2 attention-head ablation and importance estimators.

All methods consume the frozen Phase-1 records and score only their AR chunk
tokens.  They never modify model weights.  Results are namespaced by checkpoint
fingerprint and include leave-one-out validation, so rankings cannot silently be
reused for another generation or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_pipelines.utils.helpers import modality_from_latent_state
from scripts.prune.core import artifacts, refine_task, session
from scripts.prune.data import chunk_states, records
from scripts.prune.score import hooks, losses, prune_schedule

METHODS = ("contribution", "michel", "gauss_newton")


def _sigmas(model, device: torch.device) -> torch.Tensor:
    return torch.tensor(refine_task.schedule_for(model.sigmas, refine_task.K_STEP), device=device, dtype=torch.float32)


def _run(denoiser, transformer, state, sigmas: torch.Tensor, step_index: int):
    result, _ = denoiser(transformer, state, None, sigmas, step_index)
    if result is None:
        raise RuntimeError("video denoiser unexpectedly returned no result")
    return result.denoised


@torch.no_grad()
def _run_stg_self_skip(denoiser, transformer, state, sigmas: torch.Tensor, step_index: int, layer: int):
    """Use the shipped STG perturbation path for the §7.2d self-attn pre-pass."""
    sigma = sigmas[step_index].expand(state.latent.shape[0])
    video = modality_from_latent_state(state, denoiser.v_context, sigma)
    perturbations = BatchedPerturbationConfig(
        [PerturbationConfig([Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=[layer])])],
        transformer.num_blocks, device=state.latent.device, dtype=state.latent.dtype,
    )
    pred, _ = transformer(video=video, audio=None, perturbations=perturbations)
    if pred is None:
        raise RuntimeError("video transformer unexpectedly returned no result")
    return pred


def _empty_scores(model, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(attn.heads, device=device, dtype=torch.float32) for name, attn in hooks.iter_video_attention(model)}


@torch.no_grad()
def contribution_scores(model, paths: list[Path], denoiser, sigmas: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    """Exact post-gate/pre-output-projection contribution norms (§7.2a)."""
    acc = _empty_scores(model, device)
    count = dict.fromkeys(acc, 0)
    token_mask: torch.Tensor | None = None

    def callback(name, x, attn):
        # Avoid one giant B,T,H,Dout tensor: process heads independently.  The
        # deployment geometry is small, but this also makes a bad record fail safe.
        if token_mask is None:
            raise RuntimeError("head contribution hook ran without a task-token mask")
        if token_mask.shape != x.shape[:2] + (1,):
            raise ValueError(f"task-token mask {tuple(token_mask.shape)} != activation {tuple(x.shape)}")
        xh = x.detach().reshape(x.shape[0], x.shape[1], attn.heads, attn.dim_head).float()
        w = attn.to_out[0].weight.detach().float().reshape(-1, attn.heads, attn.dim_head)
        m = token_mask.to(device=x.device, dtype=torch.float32)
        for h in range(attn.heads):
            y = xh[:, :, h] @ w[:, h].T
            acc[name][h] += (y.square() * m).sum()
        count[name] += int(m.sum().item())

    with hooks.collect_activations(model, "head", callback):
        for path in paths:
            state, _, meta = chunk_states.load_record(path, device)
            token_mask = chunk_states.chunk_token_mask(state, meta)
            _run(denoiser, model, state, sigmas, meta.step_index)
    raw = {name: (value / max(count[name], 1)).sqrt().cpu() for name, value in acc.items()}
    return _normalize(raw)


def _freeze_weights(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@contextmanager
def _gradient_checkpointing(model):
    """Enable block recomputation for mask VJPs on 49-GB A6000s.

    A full 48-block backward at even the one-chunk geometry does not fit beside
    the 2.5 video-only weights without checkpointing.  The implementation only
    activates this path in training mode, so preserve the caller's mode and turn
    it on only around score collection; there is no optimizer or weight update.
    """
    core = getattr(model, "velocity_model", model)
    previous_training = model.training
    core.set_gradient_checkpointing(True)
    model.train(True)
    try:
        yield
    finally:
        core.set_gradient_checkpointing(False)
        model.train(previous_training)


def michel_scores(model, paths: list[Path], denoiser, sigmas: torch.Tensor, device: torch.device,
                  initial: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """``E |d x0_loss / d xi_h|`` with per-layer L2 normalization (§7.2b)."""
    _freeze_weights(model)
    # enable_grad, not ambient: session.transformer() wraps its whole yield in
    # torch.no_grad() (every other caller of it never backprops), so the VJP this
    # estimator needs has to explicitly re-open autograd rather than assume it.
    with torch.enable_grad(), _gradient_checkpointing(model), hooks.attach_head_masks(model, initial, requires_grad=True) as masks:
        acc = {name: torch.zeros_like(mask) for name, mask in masks.items()}
        for path in paths:
            state, target, meta = chunk_states.load_record(path, device)
            for mask in masks.values():
                mask.grad = None
            pred = _run(denoiser, model, state, sigmas, meta.step_index)
            loss = losses.x0_loss(pred, target, state, chunk_states.chunk_token_mask(state, meta))
            loss.backward()
            for name, mask in masks.items():
                if mask.grad is None:
                    raise RuntimeError(f"no mask gradient for {name}")
                acc[name].add_(mask.grad.abs())
        return _normalize({name: value.cpu() for name, value in acc.items()})


def gauss_newton_scores(model, paths: list[Path], denoiser, sigmas: torch.Tensor, device: torch.device,
                        projections: int = 8, seed: int = 42,
                        initial: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Hutchinson estimate of ``||df/dxi_h||²`` using chunk-masked Rademacher VJPs."""
    if projections < 1:
        raise ValueError("projections must be positive")
    _freeze_weights(model)
    generator = torch.Generator(device=device).manual_seed(seed)
    # enable_grad: see the matching comment in michel_scores above.
    with torch.enable_grad(), _gradient_checkpointing(model), hooks.attach_head_masks(model, initial, requires_grad=True) as masks:
        acc = {name: torch.zeros_like(mask) for name, mask in masks.items()}
        for path in paths:
            state, _, meta = chunk_states.load_record(path, device)
            token_mask = chunk_states.chunk_token_mask(state, meta).to(device=device, dtype=torch.float32)
            for _ in range(projections):
                for mask in masks.values():
                    mask.grad = None
                pred = _run(denoiser, model, state, sigmas, meta.step_index)
                # Unit-norm output direction only over task tokens.
                direction = torch.empty_like(pred.float()).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                direction.mul_(token_mask).div_(token_mask.sum().mul(pred.shape[-1]).sqrt().clamp_min(1))
                (pred.float() * direction).sum().backward()
                for name, mask in masks.items():
                    if mask.grad is None:
                        raise RuntimeError(f"no mask gradient for {name}")
                    acc[name].add_(mask.grad.square())
        return _normalize({name: (value / (len(paths) * projections)).sqrt().cpu() for name, value in acc.items()})


def _normalize(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value / value.norm().clamp_min(torch.finfo(value.dtype).eps) for name, value in raw.items()}


def score(model, states: list[Path], denoiser, sigmas: torch.Tensor, method: str, device: torch.device, **kwargs) -> dict[str, torch.Tensor]:
    """Shared estimator interface; values are keyed ``'<layer>.attn1|attn2'``."""
    if method == "contribution":
        return contribution_scores(model, states, denoiser, sigmas, device)
    if method == "michel":
        return michel_scores(model, states, denoiser, sigmas, device, kwargs.get("initial"))
    if method == "gauss_newton":
        return gauss_newton_scores(model, states, denoiser, sigmas, device, kwargs.get("projections", 8), kwargs.get("seed", 42), kwargs.get("initial"))
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


@torch.no_grad()
def layer_ablation(model, paths: list[Path], denoiser, sigmas: torch.Tensor, device: torch.device) -> dict[str, float]:
    """Exact complete-attention removal deltas for both attn1 and attn2 (§7.2d)."""
    baseline = []
    for path in paths:
        state, target, meta = chunk_states.load_record(path, device)
        pred = _run(denoiser, model, state, sigmas, meta.step_index)
        baseline.append(float(losses.rel_l2(pred, target, state, chunk_states.chunk_token_mask(state, meta))))
    base = sum(baseline) / len(baseline)
    out: dict[str, float] = {}
    for name, attn in hooks.iter_video_attention(model):
        values = []
        if name.endswith(".attn1"):
            # This is intentionally STG rather than a hook: it verifies the
            # existing production skip machinery before fine-grained scoring.
            layer = int(name.split(".", 1)[0])
            for path in paths:
                state, target, meta = chunk_states.load_record(path, device)
                pred = _run_stg_self_skip(denoiser, model, state, sigmas, meta.step_index, layer)
                values.append(float(losses.rel_l2(pred, target, state, chunk_states.chunk_token_mask(state, meta))))
        else:
            # There is no cross-attention STG enum for video text attention; the
            # equivalent narrow shim is an all-zero post-attention head mask.
            initial = {name: torch.zeros(attn.heads, device=device)}
            with hooks.attach_head_masks(model, initial, requires_grad=False):
                for path in paths:
                    state, target, meta = chunk_states.load_record(path, device)
                    pred = _run(denoiser, model, state, sigmas, meta.step_index)
                    values.append(float(losses.rel_l2(pred, target, state, chunk_states.chunk_token_mask(state, meta))))
        out[name] = sum(values) / len(values) - base
    return out


@torch.no_grad()
def leave_one_out(model, paths: list[Path], denoiser, sigmas: torch.Tensor, device: torch.device,
                  candidates: list[tuple[str, int]]) -> list[dict]:
    """Measure exact individual head ΔT0 for correlation validation."""
    rows = []
    for path in paths:
        state, target, meta = chunk_states.load_record(path, device)
        base = float(losses.rel_l2(_run(denoiser, model, state, sigmas, meta.step_index), target, state,
                                   chunk_states.chunk_token_mask(state, meta)))
        for name, head in candidates:
            initial = {name: torch.ones(next(a.heads for n, a in hooks.iter_video_attention(model) if n == name), device=device)}
            initial[name][head] = 0
            with hooks.attach_head_masks(model, initial, requires_grad=False):
                pred = _run(denoiser, model, state, sigmas, meta.step_index)
                value = float(losses.rel_l2(pred, target, state, chunk_states.chunk_token_mask(state, meta)))
            rows.append({"record": path.name, "head": name, "index": head, "delta_t0": value - base})
    return rows


def _serializable(scores: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    return {name: value.detach().cpu().tolist() for name, value in scores.items()}


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Dependency-free Spearman rho; ties receive their average rank."""
    if len(x) < 2:
        return None
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0
            for position in order[start:end]:
                out[position] = rank
            start = end
        return out
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    denom = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return numerator / denom if denom else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    session.add_model_args(ap)
    session.add_record_args(ap)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--gauss-newton-projections", type=int, default=8)
    ap.add_argument("--target-sparsity", type=float, default=None,
                    help="Run iterative Michel/Gauss--Newton pruning to this global head sparsity.")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--iterative-method", choices=("michel", "gauss_newton"), default="michel")
    ap.add_argument("--ablate-layers", action="store_true")
    ap.add_argument("--validate-heads", type=int, default=0, help="Random heads for exact leave-one-out ΔT0.")
    args = ap.parse_args()
    s = session.open_session(args, script="head_scores")
    root = s.states_root(args.states)
    paths = records.select(root, split=args.split, limit=args.max_records)
    report = {"provenance": s.stamp(), "split": args.split,
              "records": [p.name for p in paths], "methods": {}}
    with s.transformer() as transformer:
        for method in args.methods:
            report["methods"][method] = _serializable(score(transformer, paths, s.denoiser, s.sigmas, method, s.device,
                                                              projections=args.gauss_newton_projections, seed=args.seed))
        if args.target_sparsity is not None:
            def rescore(active):
                return score(transformer, paths, s.denoiser, s.sigmas, args.iterative_method, s.device,
                             projections=args.gauss_newton_projections, seed=args.seed, initial=active)
            masks, history = prune_schedule.iterative_head_masks(
                transformer, target_sparsity=args.target_sparsity, rounds=args.rounds, rescore=rescore,
            )
            report["iterative"] = {"method": args.iterative_method, "target_sparsity": args.target_sparsity,
                                   "masks": _serializable(masks), "history": history}
        if args.ablate_layers:
            report["layer_ablation_delta_t0"] = layer_ablation(transformer, paths, s.denoiser, s.sigmas, s.device)
        if args.validate_heads:
            all_heads = [(name, h) for name, attn in hooks.iter_video_attention(transformer) for h in range(attn.heads)]
            permutation = torch.randperm(len(all_heads), generator=torch.Generator().manual_seed(args.seed))
            picked = [all_heads[i] for i in permutation[:args.validate_heads].tolist()]
            loo = leave_one_out(transformer, paths, s.denoiser, s.sigmas, s.device, picked)
            report["leave_one_out"] = loo
            # Aggregate repeat records per head, then correlate exact ablation ΔT0
            # with every estimator computed in this invocation.
            exact: dict[tuple[str, int], list[float]] = {}
            for row in loo:
                exact.setdefault((row["head"], row["index"]), []).append(row["delta_t0"])
            correlations = {}
            for method, scores in report["methods"].items():
                estimated, observed = [], []
                for (name, index), values in exact.items():
                    estimated.append(scores[name][index])
                    observed.append(sum(values) / len(values))
                correlations[method] = {"spearman_rho": _spearman(estimated, observed), "heads": len(observed)}
            report["validation_spearman"] = correlations
    output = artifacts.run_dir(s.key, "head-scores", script="head_scores", argv=sys.argv[1:])
    (output / "head_scores.json").write_text(json.dumps(report, indent=2))
    print(output / "head_scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
