"""Evaluate saved binary head/FFN checkpoints with matched native D0 evidence."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

from scripts.onestep_avatar import media
from scripts.onestep_avatar.corpus.dataset import load_training_master
from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, numerics, provenance, records, session
from scripts.prune.data import ffn_study, foreground
from scripts.prune.evaluate import timing
from scripts.prune.evaluate.media_common import (
    pixel_metrics,
    region_lpips,
    saved_video,
    source_pixels,
    verify_comparison,
)
from scripts.prune.learn import ffn_training, heads


def key(source, seed, sigma):
    return records.digest(dict(view=source["relative_dir"], seed=seed, sigma=sigma))[:24]


def cases(study, split):
    sources = study["splits"][split]
    if split == "validation":
        sources = [
            next(s for s in sources if s["actor"] == actor) for actor in study["splits"]["validation_actors"][:4]
        ]
    return [
        (s, seed, study["config"]["sigmas"][i])
        for s in sources
        for seed in study["config"]["evaluation_seeds"]
        for i in study["config"]["monitor_sigma_indices"]
    ]


def current_for(study, gpu):
    if gpu not in study["config"]["test_gpu_ids"]:
        raise ValueError("FFN evaluation uses physical GPUs 4–6")
    numerics.deterministic_policy()
    return session.open_frozen(study, gpu, script="prune.evaluate.ffn")


def configure_saved(model, study, checkpoint, *, disable_adapters=False):
    step = ffn_training.JointStep(model, study["config"])
    ao, go = ffn_training.optimizers(step)
    state = ffn_training.read(checkpoint, study=study["identity"])
    ffn_training.restore(step, ao, go, state, restore_rng=False)
    if not state["frozen"]:
        raise ValueError("scientific FFN evaluation requires a frozen binary checkpoint")
    if disable_adapters:
        for adapter in step.attached.values():
            adapter.enabled = False
    del ao, go, state
    return step


def infer(study, gpu, split, arm, checkpoint=None):
    if arm != "native" and checkpoint is None:
        raise ValueError("trained FFN evaluation requires a checkpoint")
    if arm not in ("native", "selection", "final", "mask_only"):
        raise ValueError("unknown FFN evaluation arm")
    current = current_for(study, gpu)
    root = Path(study["root"]) / f"evaluation/{split}/{arm}"
    identity = dict(
        study=study["identity"],
        arm=arm,
        checkpoint_sha256=provenance.file_sha256(checkpoint) if checkpoint else None,
        schedules=[[study["config"]["sigmas"][i], 0.0] for i in study["config"]["monitor_sigma_indices"]],
        source_split=split,
        seeds=study["config"]["evaluation_seeds"],
    )
    records.publish_json(root / "identity.json", identity)
    with current.transformer() as model, torch.no_grad():
        model.eval()
        step = configure_saved(model, study, checkpoint, disable_adapters=arm == "mask_only") if checkpoint else None
        run_cases(
            study,
            current,
            model,
            root,
            identity,
            cases(study, split),
            label=f"FFN evaluation arm={arm}",
            before_forward=step.masks if step else None,
        )
        if step:
            step.close()
            del step


def run_cases(
    study,
    current,
    model,
    root,
    identity,
    case_list,
    *,
    label,
    before_forward=None,
    input_case=None,
    expected_input=None,
):
    """Predict each held-out case once with a fixed model; reuse only verified saved receipts.

    `before_forward` rebuilds live gates before each forward. `expected_input(dest)`
    returns the input-manifest hash that a paired native receipt pinned for this case.
    """
    input_case = input_case or heads.input_case
    rows = []
    for source, seed, sigma in case_list:
        dest = root / key(source, seed, sigma)
        receipt = dest / "case.json"
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if row["identity"] != identity or provenance.file_sha256(row["prediction"]) != row["prediction_sha256"]:
                raise ValueError(f"saved evaluation case changed: {receipt}")
            if provenance.file_sha256(row["input_manifest"]) != row["input_sha256"]:
                raise ValueError(f"saved evaluation input changed: {receipt}")
            rows.append(row)
            continue
        grid, video, c0, capture, epsilon, inputs, manifest = input_case(study, current, source, seed, sigma)
        if before_forward:
            before_forward()
        velocity, clean = ltx_adapter.velocity_forward(model, video)
        clean = common.with_clean_prefix(clean, c0)
        prefix = grid.tokens_per_latent_frame
        if not torch.equal(clean[:, :prefix], c0) or not torch.isfinite(clean).all():
            raise ValueError("evaluation clean condition or finite output failed")
        input_sha = provenance.file_sha256(inputs / "manifest.json")
        if expected_input is not None and expected_input(dest) != input_sha:
            raise ValueError("held-out inputs differ from the saved native baseline")
        mse = float((clean[:, prefix:].float() - capture[:, prefix:].float()).square().mean())
        target = epsilon.float() - capture.float()
        velocity_mse = float((velocity[:, prefix:].float() - target[:, prefix:]).square().mean())
        tensor = dest / "prediction.pt"
        records.publish_tensor(tensor, clean)
        row = dict(
            identity=identity,
            actor=source["actor"],
            view=source["relative_dir"],
            seed=seed,
            sigma=sigma,
            schedule=[sigma, 0.0],
            capture_mse=mse,
            velocity_mse=velocity_mse,
            prediction=str(tensor),
            prediction_sha256=provenance.file_sha256(tensor),
            input_manifest=str(inputs / "manifest.json"),
            input_sha256=input_sha,
            noise_sha256=manifest["videos"][0]["artifacts"]["epsilon_sha256"],
            context=study["text_context"],
            capture_sha256=source["capture_latent_sha256"],
        )
        records.publish_json(receipt, row)
        rows.append(row)
        print(f"{label} cases={len(rows)}/{len(case_list)} capture_mse={mse:.8f}", flush=True)
        del grid, video, c0, capture, epsilon, velocity, clean, target
    records.publish_json(
        root / "results.json",
        dict(
            identity=identity,
            rows=rows,
            mean_capture_mse=statistics.mean(r["capture_mse"] for r in rows),
            status="complete_numeric_only",
            cases=len(rows),
        ),
    )
    return rows


def references(current, decoder, metric, source):
    """Decode the capture once per source; build its RGB reference, GT matte weights and VAE ceiling."""
    capture, _ = load_training_master(Path(source["capture"]))
    vae = media.decode(current, capture.unsqueeze(0), decoder, 42).mul(255).round().to(torch.uint8)
    rgb = source_pixels(source)
    matte, _, matte_identity = foreground.person_coverage(
        Path(source["resolved_dir"]), source["capture_encode_record"], replay=True
    )
    weights = torch.nn.functional.interpolate(
        torch.from_numpy(matte).float().unsqueeze(1) / 255, size=vae.shape[-2:], mode="area"
    )[:, 0]
    return dict(
        shape=tuple(capture.shape),
        vae=vae,
        rgb=rgb,
        weights=weights,
        matte_identity=matte_identity,
        ceiling=region_lpips(metric, vae, rgb, weights, current.device),
    )


def compare_case(current, decoder, metric, refs, rows, titles, *, question, dest, source, sigma, seed):
    """Decode saved predictions, score them against the capture VAE and save one synchronized comparison."""
    channels, frames, height, width = refs["shape"]
    pixels = {}
    for arm, row in rows.items():
        if provenance.file_sha256(row["prediction"]) != row["prediction_sha256"]:
            raise ValueError(f"decoded prediction hash differs: {arm}")
        tokens = torch.load(row["prediction"], map_location="cpu", weights_only=True).to(current.device)
        # Patch size one: tokens (1, F*H*W, C) are a transpose of the dense latent.
        latent = tokens.transpose(1, 2).reshape(1, channels, frames, height, width).contiguous()
        pixels[arm] = media.decode(current, latent, decoder, 42).mul(255).round().to(torch.uint8)
        del tokens, latent
    vae, weights = refs["vae"], refs["weights"]
    errors = {
        arm: dict(
            person=region_lpips(metric, value, vae, weights, current.device),
            frame=region_lpips(metric, value, vae, torch.ones_like(weights), current.device),
            pixels=pixel_metrics(value, vae, weights),
        )
        for arm, value in pixels.items()
    }
    roles = [("capture_rgb", "Capture RGB", refs["rgb"]), ("capture_vae", "Capture VAE", vae)]
    roles += [(arm, title, pixels[arm]) for arm, title in titles]
    shown = tuple(range(len(vae)))
    panels = [
        media.Panel(role, title, value, shown, value="RGB" if role == "capture_rgb" else "VAE decoding")
        for role, title, value in roles
    ]
    combined, rendering = media.render_panels(
        panels,
        question=question,
        layout="comparison",
        fps=source["fps"],
        panel_size=(384, 384),
        viewing_width=480,
        poster_frame=min(30, len(shown) - 1),
        common_settings=dict(schedule=[sigma, 0.0], seed=seed, actor=source["actor"]),
    )
    rendering = verify_comparison(media.save_render(combined, rendering, dest))
    records.publish_json(dest / "rendering.verified.json", rendering)
    for role, _, value in roles:
        saved_video(value, dest / f"{role}.mp4", source["fps"])
    return errors


def decoded(study, gpu, split):
    current = current_for(study, gpu)
    root = Path(study["root"])
    selected = []
    for source in study["splits"][split]:
        if source["actor"] not in {s["actor"] for s in selected}:
            selected.append(source)
        if len(selected) == 4:
            break
    import lpips

    metric = lpips.LPIPS(net="alex", verbose=False).to(current.device).eval()
    summaries = []
    arms = ("native", "selection", "final")
    titles = [("native", "Native"), ("selection", "Before recovery"), ("final", "After recovery")]
    with current.decoder() as decoder:
        for source in selected:
            refs = None
            for index in study["config"]["monitor_sigma_indices"]:
                sigma, seed = study["config"]["sigmas"][index], study["config"]["evaluation_seeds"][0]
                dest = root / f"media/{split}/{key(source, seed, sigma)}"
                if (dest / "results.json").exists():
                    summaries.append(json.loads((dest / "results.json").read_text()))
                    continue
                rows = {
                    arm: json.loads(
                        (root / f"evaluation/{split}/{arm}/{key(source, seed, sigma)}/case.json").read_text()
                    )
                    for arm in arms
                }
                if len({row["input_sha256"] for row in rows.values()}) != 1:
                    raise ValueError("decoded FFN arms use different native input bytes")
                refs = refs or references(current, decoder, metric, source)
                errors = compare_case(
                    current,
                    decoder,
                    metric,
                    refs,
                    rows,
                    titles,
                    question="Does recovery preserve the person after pruning?",
                    dest=dest,
                    source=source,
                    sigma=sigma,
                    seed=seed,
                )
                row = dict(
                    actor=source["actor"],
                    view=source["relative_dir"],
                    sigma=sigma,
                    seed=seed,
                    metrics=errors,
                    vae_to_rgb_ceiling=refs["ceiling"],
                    matte_identity=refs["matte_identity"],
                    rendering=str(dest / "rendering.json"),
                    input_sha256=rows["native"]["input_sha256"],
                )
                records.publish_json(dest / "results.json", row)
                summaries.append(row)
                print(f"FFN decoded cases={len(summaries)}", flush=True)
                gc.collect()
            del refs
    records.publish_json(
        root / f"media/{split}/results.json", dict(study=study["identity"], cases=summaries, arms=list(arms))
    )


def benchmark(study, gpu, checkpoint, candidate):
    current = current_for(study, gpu)
    source = study["splits"]["validation"][0]
    case = heads.input_case(study, current, source, 2042, study["config"]["sigmas"][0])
    orders = []
    for order in (("masked", "compact", "masked"), ("compact", "masked", "compact")):
        rows = []
        for arm in order:
            with current.transformer(candidate if arm == "compact" else None) as model, torch.no_grad():
                step = configure_saved(model, study, checkpoint) if arm == "masked" else None
                if step:
                    for adapter in step.attached.values():
                        adapter.base.weight.copy_(adapter.merged_weight())
                        adapter.enabled = False
                torch.cuda.reset_peak_memory_stats(current.device)
                row = timing.timed(model, case[1], step, current.device)
                row["arm"] = arm
                rows.append(row)
                if step:
                    step.close()
                    del step
                del model
            gc.collect()
            torch.cuda.empty_cache()
        orders.append(
            dict(order=list(order), rows=rows, drift=abs(rows[2]["median_seconds"] / rows[0]["median_seconds"] - 1))
        )
    speedups = [
        timing.bracket(b["rows"], "ABA" if b["rows"][0]["arm"] == "masked" else "BAB", "median_seconds")["speedup"]
        for b in orders
    ]
    parity = json.loads(Path(candidate).with_suffix(".parity.json").read_text())
    result = dict(
        study=study["identity"],
        orders=orders,
        speedups=speedups,
        parity_passed=parity["passed"],
        speed_qualified=parity["passed"] and min(speedups) >= 1.05 and max(o["drift"] for o in orders) < 0.02,
        checkpoint_bytes=Path(candidate).stat().st_size,
        scope="complete native transformer, excluding VAE and input preparation",
    )
    records.publish_json(Path(study["root"]) / "timing/results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--stage", required=True, choices=("infer", "media", "benchmark"))
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--arm", choices=("native", "selection", "final", "mask_only"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    study = ffn_study.load(args.study)
    torch.set_num_threads(4)
    if args.stage == "infer":
        infer(study, args.gpu_id, args.split, args.arm, args.checkpoint)
    elif args.stage == "media":
        decoded(study, args.gpu_id, args.split)
    else:
        benchmark(study, args.gpu_id, args.checkpoint, args.candidate)


if __name__ == "__main__":
    main()
