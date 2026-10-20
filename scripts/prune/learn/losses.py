"""Official FP32 velocity reductions, excluding the supplied clean frame.

Inputs are native patchifier tokens [B,N,C]. The caller supplies the exact
first-frame token count. Region weights use the same token order. Empty
person coverage fails; absent face/hands contribute zero and stay labeled.
Each accumulated microbatch divides the complete loss by the accumulation count.
The frozen penalty form selects mean sigmoid gates or mean relu(logit + offset).
The latter stops sparsity pressure at logit <= -offset; it does not clamp logits.
The shared coefficient owner applies the frozen ramp duration and start multiplier
to both the loss and its analytic gradient. Defaults retain the old 300-step ramp.
"""

from __future__ import annotations

import math

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


def sparsity_value(gates, *, logits=None, kind="mean_sigmoid", logit_offset=3.0):
    if kind == "mean_sigmoid":
        return gates.float().mean()
    if kind != "relu_logits" or logits is None or logits.shape != gates.shape:
        raise ValueError("logit sparsity requires raw logits with the gate shape")
    if not torch.isfinite(logits).all() or not torch.isfinite(torch.tensor(logit_offset)):
        raise ValueError("invalid raw logits or sparsity offset")
    return torch.relu(logits.float() + logit_offset).mean()


def sparsity_coefficient(update, phase, *, lambda_max=0.001, lambda_warmup=300, lambda_start_multiplier=0.0):
    """One coefficient for loss, analytic gradients and saved update receipts."""
    if (
        update < 1
        or not math.isfinite(lambda_max)
        or lambda_max < 0
        or not math.isfinite(lambda_warmup)
        or lambda_warmup <= 0
        or not math.isfinite(lambda_start_multiplier)
        or not 0 <= lambda_start_multiplier <= 1
    ):
        raise ValueError("invalid sparsity update, coefficient or ramp")
    if phase != "soft":
        return 0.0
    ramp = lambda_start_multiplier + (1 - lambda_start_multiplier) * min(update / lambda_warmup, 1.0)
    return lambda_max * ramp


def sparsity_gradient(
    logits,
    *,
    update,
    phase,
    lambda_max=0.001,
    lambda_warmup=300,
    lambda_start_multiplier=0.0,
    kind="mean_sigmoid",
    logit_offset=3.0,
):
    """Unclipped analytic logit gradient of the complete per-update penalty."""
    coefficient = sparsity_coefficient(
        update,
        phase,
        lambda_max=lambda_max,
        lambda_warmup=lambda_warmup,
        lambda_start_multiplier=lambda_start_multiplier,
    )
    if phase != "soft":
        return torch.zeros_like(logits)
    alpha = logits.detach().float()
    if kind == "mean_sigmoid":
        p = alpha.sigmoid()
        derivative = p * (1 - p)
    elif kind == "relu_logits":
        derivative = (alpha + logit_offset > 0).float()
    else:
        raise ValueError("unknown sparsity penalty form")
    return coefficient * derivative / alpha.numel()


def phase_loss(
    terms,
    gates,
    *,
    update,
    phase,
    lambda_max=0.001,
    lambda_warmup=300,
    lambda_start_multiplier=0.0,
    person=0.0,
    small_region=0.0,
    accumulation=2,
    logits=None,
    sparsity_kind="mean_sigmoid",
    logit_offset=3.0,
):
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
    coefficient = sparsity_coefficient(
        update,
        phase,
        lambda_max=lambda_max,
        lambda_warmup=lambda_warmup,
        lambda_start_multiplier=lambda_start_multiplier,
    )
    penalty = (
        coefficient * sparsity_value(gates, logits=logits, kind=sparsity_kind, logit_offset=logit_offset)
        if phase == "soft"
        else official * 0
    )
    return (official + regional + penalty) / accumulation, penalty
