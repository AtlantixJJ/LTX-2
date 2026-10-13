"""Official FP32 velocity reductions, excluding the supplied clean frame.

Inputs are native patchifier tokens [B,N,C]. The caller supplies the exact
first-frame token count. Region weights use the same token order. Empty
person coverage fails; absent face/hands contribute zero and stay labeled.
Each accumulated microbatch divides the complete loss by the accumulation count.
"""

from __future__ import annotations

import torch


def velocity_target(epsilon, capture):
    if epsilon.shape != capture.shape or not torch.isfinite(epsilon).all() or not torch.isfinite(capture).all():
        raise ValueError("invalid saved velocity target inputs")
    return epsilon.float() - capture.float()


def velocity_mse(velocity, epsilon, capture, prefix_tokens, regions=None):
    if velocity.shape != capture.shape or velocity.ndim != 3 or not 0 < prefix_tokens < velocity.shape[1]:
        raise ValueError("invalid velocity token geometry")
    if not torch.isfinite(velocity).all():
        raise ValueError("nonfinite native velocity")
    error = (velocity.float() - velocity_target(epsilon, capture)).square()[:, prefix_tokens:]
    result = {"official": error.mean(), "coverage": {}}
    for name, weights in (regions or {}).items():
        if (
            weights.shape != velocity.shape[:2]
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or (weights > 1).any()
        ):
            raise ValueError("invalid region coverage")
        w = weights[:, prefix_tokens:].float()
        mass = w.sum()
        result["coverage"][name] = float(mass.detach())
        if mass <= 0:
            if name == "person":
                raise ValueError("empty person coverage")
            result[name] = error.sum() * 0
        else:
            result[name] = (error * w.unsqueeze(-1)).sum() / (mass * error.shape[-1])
    hands = [result[k] for k in ("left_hand", "right_hand") if result["coverage"].get(k, 0) > 0]
    result["hands"] = torch.stack(hands).mean() if hands else error.sum() * 0
    return result


def phase_loss(terms, gates, *, update, phase, lambda_max=0.001, person=0.0, small_region=0.0, accumulation=2):
    if update < 1 or accumulation < 1 or min(lambda_max, person, small_region) < 0:
        raise ValueError("invalid loss coefficients or update")
    official = terms["official"]
    regional = official * 0
    if person:
        if terms["coverage"].get("person", 0) <= 0:
            raise ValueError("person loss requires coverage")
        regional = person * terms["person"]
    if small_region:
        regional = regional + small_region * (terms.get("face", official * 0) + terms["hands"])
    penalty = lambda_max * min(update / 300, 1.0) * gates.mean() if phase == "soft" else official * 0
    return (official + regional + penalty) / accumulation, penalty
