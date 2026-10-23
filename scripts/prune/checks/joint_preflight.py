"""Measure optimizer sensitivity and native one-block channel-ablation costs.

The p0 CLI phase replays correlated seeded gradients and optional saved task
gradients across scales; synthetic observations stay labeled. The p1 phase
changes one block and fraction per matched validation case, supports sharding
and saves immutable predictions and cost deltas. Test devices come from frozen
config. Review requirements are defined in doc/VALIDATION.md.

CLI: python -m scripts.prune.checks.joint_preflight --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.prune.core import numerics, preflight, records
from scripts.prune.core.provenance import file_sha256


def replay(gradients, start, epsilon, penalty=0):
    alpha = torch.nn.Parameter(torch.full(gradients.shape[1:], float(start)))
    optimizer = torch.optim.Adam([alpha], lr=0.01, eps=epsilon)
    values = [alpha.detach().clone()]
    for gradient in gradients:
        alpha.grad = gradient + penalty
        optimizer.step()
        values.append(alpha.detach().clone())
    return torch.stack(values)


def p0(output, *, traces=None, samples=1048576):
    torch.set_num_threads(4)
    output = Path(output)
    generator = torch.Generator().manual_seed(42)
    noise = torch.randn(100, samples, generator=generator)
    # Saved seed and exact gradient bytes make this CPU replay independently repeatable.
    noise_path = output / "checks/p0/seeded_gradients.pt"
    if not noise_path.exists():
        records.publish_tensor(noise_path, noise)
    rows = []
    for start in (0, 1):
        for epsilon in (1e-8, 1e-12):
            for penalty in (0.0, 0.01 / 1536, 0.01 / 1536 / 704):
                base = None
                for scale in (1e-9, 1e-8, 1e-7, 1e-6):
                    values = replay(noise[:50] * scale, start, epsilon, penalty)
                    if base is None:
                        base = values
                    displacement = values[-1] - start
                    rows.append(
                        dict(
                            start=start,
                            epsilon=epsilon,
                            penalty=penalty,
                            scale=scale,
                            raw_mean=float(displacement.mean()),
                            raw_std=float(displacement.std()),
                            executed_mean=float((values[-1].clamp(0, 1) - start).mean()),
                            max_path_difference_from_smallest_scale=float((values - base).abs().max()),
                        )
                    )
    hits = []
    for name, threshold, persistent, epsilon, population in (
        ("heads", 0.25, 0.15, 1e-8, 1536),
        ("channels", 0.35, 0.25, 1e-12, 786432),
    ):
        trajectories = [replay(noise[i : i + 50] * 1e-6, 1, epsilon) for i in (0, 50)]
        for direction in ("drop", "rise"):
            qualified = []
            for values in trajectories:
                movement = 1 - values if direction == "drop" else values - 1
                final, peak = movement[-1], movement.max(0).values
                giveback = (peak - final) / peak.clamp_min(1e-30)
                late = movement[-1] - movement[-11]
                clean = (giveback <= 0.1) & (late > 0)
                qualified.append(clean & (final >= persistent))
            strong = clean & (final >= threshold)
            hits.append(
                dict(
                    family=name,
                    direction=direction,
                    samples=samples,
                    population=population,
                    strong_hits=int(strong.sum()),
                    persistent_hits=int((qualified[0] & qualified[1]).sum()),
                    strong_expected=float(strong.float().mean() * population),
                    zero_hit_upper_95=3 * population / samples if not strong.any() else None,
                    persistent_expected=float((qualified[0] & qualified[1]).float().mean() * population),
                )
            )
    correlation = dict(
        status="not run yet", reason="native channel traces required; moments do not establish correlation"
    )
    if traces:
        paths = sorted(Path(traces).glob("*.pt"))
        if len(paths) < 50:
            raise ValueError("P0 temporal replay requires 50 complete gradient traces")
        saved = [
            torch.load(p, map_location="cpu", weights_only=False)["tensors"]["channels"].flatten() for p in paths[:50]
        ]
        matrix = torch.stack(saved)
        # Center each channel over time; sample contiguous five-update blocks
        # with replacement. This removes mean drift, but retains within-block
        # temporal covariance. The block length remains a sensitivity choice.
        ids = torch.randperm(matrix.shape[1], generator=generator)[:samples]
        centered = matrix[:, ids] - matrix[:, ids].mean(0)
        starts = torch.randint(0, 46, (10,), generator=generator)
        indices = (starts[:, None] + torch.arange(5)).flatten()
        values = replay(centered[indices], 1, 1e-12)
        correlation = dict(
            status="complete",
            centering="per-channel temporal mean",
            block_length=5,
            replacement=True,
            selected_channels=ids.tolist(),
            block_starts=starts.tolist(),
            raw_std=float((values[-1] - 1).std()),
            source_hashes={str(p): file_sha256(p) for p in paths[:50]},
        )
    result = dict(
        format="joint_p0_v3",
        seed=42,
        gradient_input=dict(path=str(noise_path), sha256=file_sha256(noise_path)),
        rows=rows,
        noise_reference=hits,
        correlation=correlation,
        limitations="Independent Gaussian reference is not a false-removal rate or a native prediction.",
    )
    records.publish_json(output / "checks/p0/replay.json", result)
    return result


def p1(study, gpu, shard, shards, *, timing_only=False):
    from scripts.onestep_avatar.model import common
    from scripts.prune.core import ltx_adapter, session
    from scripts.prune.learn import losses
    from scripts.prune.learn.ddp_owner import input_case
    from scripts.prune.score import hooks

    numerics.deterministic_policy()
    torch.set_num_threads(4)
    gpu = preflight.testing_gpu(study["config"], gpu)
    current = session.open_model(
        "2.5",
        gpu,
        script="prune.checks.joint_preflight",
        prompt="",
        transformer_path=Path(study["components"]["transformer"]["path"]),
    )
    root = Path(study["root"])
    sources = [
        next(s for s in study["splits"]["validation"] if s["actor"] == a)
        for a in study["splits"]["validation_actors"][:4]
    ]
    generator = torch.Generator().manual_seed(42)
    orders = [torch.randperm(16384, generator=generator) for _ in range(48)]
    mask = torch.ones(48, 16384, device=current.device)

    def provider(name):
        return mask[int(name.split(".")[0])]

    def predict(model, case):
        velocity, clean = ltx_adapter.velocity_forward(model, case[1])
        clean = common.with_clean_prefix(clean, case[2])
        prefix = case[0].tokens_per_latent_frame
        return clean.cpu(), dict(
            velocity_mse=float(losses.velocity_mse(velocity, case[4], case[3], prefix)["official"]),
            capture_mse=float((clean[:, prefix:].float() - case[3][:, prefix:].float()).square().mean()),
        )

    completed = 0
    with current.transformer() as model, torch.no_grad(), hooks.attach_ffn_provider(model, provider):
        for source in sources:
            for si in study["config"]["monitor_sigma_indices"]:
                case = input_case(study, current, source, 2042, study["config"]["sigmas"][si])
                mask.fill_(1)
                predict(model, case)  # Warm the exact geometry before estimating duration.
                torch.cuda.synchronize(current.device)
                started = time.monotonic()
                baseline, native = predict(model, case)
                torch.cuda.synchronize(current.device)
                seconds = time.monotonic() - started
                estimate = dict(
                    warm_forward_seconds=seconds,
                    forwards=1152,
                    estimated_seconds=1152 * seconds,
                    shards=shards,
                    per_shard_seconds=1152 * seconds / shards,
                )
                estimate_path = root / f"checks/p1/estimate_shard_{shard}.json"
                if not estimate_path.exists():
                    records.publish_json(estimate_path, estimate)
                if timing_only:
                    return estimate
                for b in range(shard, 48, shards):
                    for fraction in (0.1, 0.5):
                        count = int(16384 * fraction)
                        target = root / f"checks/p1/{source['actor']}_{si}_{b}_{count}.pt"
                        if target.exists():
                            continue
                        mask.fill_(1)
                        ids = orders[b][:count]
                        mask[b, ids.to(mask.device)] = 0
                        clean, metrics = predict(model, case)
                        row = dict(
                            study=study["identity"],
                            actor=source["actor"],
                            source=source["relative_dir"],
                            sigma=study["config"]["sigmas"][si],
                            block=b,
                            fraction=fraction,
                            count=count,
                            mask_ids=ids,
                            native=native,
                            ablated=metrics,
                            prediction=clean,
                            native_prediction=baseline,
                            input_manifest=str(case[-2] / "manifest.json"),
                            input_sha256=file_sha256(case[-2] / "manifest.json"),
                            delta={k: metrics[k] - native[k] for k in metrics},
                            delta_per_channel={k: (metrics[k] - native[k]) / count for k in metrics},
                        )
                        records.publish(target, lambda p: torch.save(row, p))
                        completed += 1
                        print(
                            f"P1 shard={shard} completed={completed} actor={source['actor']} sigma={si} block={b} removed={count}",
                            flush=True,
                        )
                del case
    return dict(complete=True, observations=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p0", "p1"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "p0":
        result = p0(args.output, traces=args.traces)
    else:
        from scripts.prune.data.joint_study import load

        result = p1(load(args.study), args.gpu_id, args.shard, args.shards, timing_only=args.timing_only)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
