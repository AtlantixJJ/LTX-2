"""Record the Phase-1 2.5 Euler-versus-ancestral decision on cached states."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep, EulerDiffusionStep
from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import post_process_latent

from scripts.prune import chunk_states, losses, model_registry, preflight, prompt_cache, provenance, refine_task

DTYPE = torch.bfloat16


def _run(transformer, state, context, sigmas: torch.Tensor, *, ancestral: bool, seed: int):
    """Run a cached initial state with a deterministic noise stream per sampler."""
    denoiser = SimpleDenoiser(context, None)
    stepper = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0) if ancestral else EulerDiffusionStep()
    generator = torch.Generator(device=state.latent.device).manual_seed(seed)
    for i in range(len(sigmas) - 1):
        result, _ = denoiser(transformer, state, None, sigmas, i)
        denoised = post_process_latent(result.denoised, state.denoise_mask, state.clean_latent)
        kwargs = {"noise": torch.randn(state.latent.shape, device=state.latent.device, dtype=state.latent.dtype, generator=generator)} if ancestral else {}
        latent = stepper.step(state.latent, denoised, sigmas, i, **kwargs)
        # Re-freeze the carryover context AFTER the step, not just the prediction
        # before it. The ancestral step rescales every token by alpha_next/alpha_down
        # and adds sigma_up*noise on top -- at the k2 tail that is a 0.77x rescale plus
        # sigma~0.38 of noise applied to the frozen context as well. Without this the
        # A/B does not compare samplers, it compares "context intact" against "context
        # destroyed", and ancestral loses for a reason that has nothing to do with
        # sampling. Plain Euler is unaffected: on frozen tokens sample == denoised ==
        # clean, so its update is already the identity there and this is a no-op.
        state = replace(state, latent=post_process_latent(latent, state.denoise_mask, state.clean_latent))
    return state.latent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--states", type=Path, default=None)
    ap.add_argument("--max-states", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    model = preflight.check(args.model, sampler="euler", gpu_id=args.gpu_id)
    root = args.states or (model_registry.WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "calibration")
    # Only step-0 on-policy records: the A/B has to run the whole k2 trajectory from
    # its start, and a step-1 record is already half-way down a Euler-stepped one.
    # Read each record once -- `load_record` deserializes the tensors, so the obvious
    # double-call in a comprehension filter costs a full extra pass over the cache.
    candidates = [(p, chunk_states.load_record(p)[2]) for p in chunk_states.iter_records(root)]
    paths = [p for p, m in candidates if m.family == "on_policy" and m.step_index == 0][: args.max_states]
    if not paths:
        raise SystemExit(
            f"No on-policy step-0 records in {root}. Run "
            f"`python -m scripts.prune.teacher --model {model.key} --gpu-id N --build-calibration` first."
        )
    device = torch.device(f"cuda:{args.gpu_id}")
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)
    sigmas = torch.tensor(refine_task.schedule_for(model.sigmas, refine_task.K_STEP), device=device)
    stage = DiffusionStage.from_checkpoint(model.paths.transformer(), DTYPE, device, model_configurator=LTXVideoOnlyModelConfigurator, scale_factors=model.scale_factors)
    rows = []
    with torch.no_grad(), stage._transformer_ctx() as transformer:
        for i, path in enumerate(paths):
            state, target, meta = chunk_states.load_record(path, device)
            euler = _run(transformer, state.clone(), context, sigmas, ancestral=False, seed=args.seed + i)
            ancestral = _run(transformer, state.clone(), context, sigmas, ancestral=True, seed=args.seed + i)
            rows.append({"record": path.name, "clip": meta.clip, "euler_t0": float(losses.rel_l2(euler, target, state)), "ancestral_t0": float(losses.rel_l2(ancestral, target, state))})
    e = sum(r["euler_t0"] for r in rows) / len(rows); a = sum(r["ancestral_t0"] for r in rows) / len(rows)
    result = {"provenance": provenance.stamp(model, device, script="sampler_ab"), "states": rows, "euler_t0_mean": e, "ancestral_t0_mean": a, "chosen_sampler": "euler" if e <= a else "ancestral", "decision_metric": "mean T0 relative L2 vs frozen teacher"}
    out = model_registry.WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key / "sampler_ab.json"
    out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
