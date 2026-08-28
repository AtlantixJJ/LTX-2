"""Record the Phase-1 2.5 Euler-versus-ancestral decision on cached states."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep, EulerDiffusionStep
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import post_process_latent
from scripts.prune.core import artifacts, session
from scripts.prune.data import chunk_states, records
from scripts.prune.score import losses


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
    session.add_model_args(ap)
    ap.add_argument("--states", type=Path, default=None)
    ap.add_argument("--max-states", type=int, default=12)
    args = ap.parse_args()
    s = session.open_session(args, script="sampler_ab")
    root = s.states_root(args.states)
    # Only step-0 on-policy records: the A/B has to run the whole k2 trajectory from
    # its start, and a step-1 record is already half-way down a Euler-stepped one.
    # Read each record once -- `load_record` deserializes the tensors, so the obvious
    # double-call in a comprehension filter costs a full extra pass over the cache.
    paths = records.select(root, family="on_policy", step_index=0, limit=args.max_states)
    rows = []
    with s.transformer() as transformer:
        for i, path in enumerate(paths):
            state, target, meta = chunk_states.load_record(path, s.device)
            euler = _run(transformer, state.clone(), s.context, s.sigmas, ancestral=False, seed=args.seed + i)
            ancestral = _run(transformer, state.clone(), s.context, s.sigmas, ancestral=True, seed=args.seed + i)
            rows.append({"record": path.name, "clip": meta.clip, "euler_t0": float(losses.rel_l2(euler, target, state)), "ancestral_t0": float(losses.rel_l2(ancestral, target, state))})
    e = sum(r["euler_t0"] for r in rows) / len(rows)
    a = sum(r["ancestral_t0"] for r in rows) / len(rows)
    result = {"provenance": s.stamp(), "states": rows, "euler_t0_mean": e, "ancestral_t0_mean": a, "chosen_sampler": "euler" if e <= a else "ancestral", "decision_metric": "mean T0 relative L2 vs source target"}
    out = artifacts.gate(s.key, "sampler_ab")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
