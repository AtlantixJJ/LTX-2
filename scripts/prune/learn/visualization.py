"""Capture actual training forwards and publish matched VAE previews.

Only the primary rank retains the first draw at each configured update.
Predictions are detached before backward; decoding uses no training graph.
See doc/learn/visualization.md for the layout, memory and RNG contracts.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ltx_pipelines.utils.helpers import tiling_config_for_vae
from scripts.onestep_avatar import media
from scripts.onestep_avatar.model import common
from scripts.prune.core import provenance, records, session
from scripts.prune.learn import checkpoints


def interval(config):
    value = config.get("visualization_interval", 100)
    if type(value) is not int or value < 0:
        raise ValueError("visualization_interval must be a nonnegative integer")
    return value


def arm(step, config, update, *, primary=True):
    """Request the first local forward only; return the shared save decision."""
    every = interval(config)
    due = every > 0 and update > 0 and update % every == 0
    step = getattr(step, "module", step)
    step.capture_visualization = due and primary
    step.visualization_prediction = None
    return due


def capture_prediction(step, clean):
    if getattr(step, "capture_visualization", False):
        step.visualization_prediction = clean.detach().to(device="cpu", copy=True)
        step.capture_visualization = False


def snapshot(step, case, **settings):
    """Copy the matching input before its grid and training tensors are freed."""
    step = getattr(step, "module", step)
    prediction = getattr(step, "visualization_prediction", None)
    if prediction is None:
        return None
    step.visualization_prediction = None
    grid, video, c0, capture, _, inputs, manifest = case
    prediction = common.with_clean_prefix(prediction, c0.detach().cpu())
    values = dict(noisy=video.latent.detach().cpu(), gt=capture.detach().cpu(), predicted_x0=prediction)
    if any(v.shape != capture.shape or not torch.isfinite(v).all() for v in values.values()):
        raise ValueError("training visualization has invalid latent values or geometry")
    return dict(
        latents={name: grid.unpatchify_block(value, grid.latent_frames).clone() for name, value in values.items()},
        settings=dict(
            **settings,
            fps=grid.fps,
            input_manifest=str(inputs / "manifest.json"),
            input_manifest_sha256=provenance.file_sha256(inputs / "manifest.json"),
            training_window=manifest.get("training_window"),
            prediction_weights="training forward before the optimizer update",
            sample_selection="first draw on rank zero",
            frame_coordinates="local decoded training window",
        ),
    )


def _write(current, preview, destination):
    from ltx_trainer.video_utils import save_video

    destination = Path(destination)
    settings, latents = preview["settings"], preview["latents"]
    completed = destination / "rendering.json"
    if completed.exists():
        saved = records.read_json(completed)
        if any(saved["common_settings"].get(key) != value for key, value in settings.items()):
            raise ValueError("replayed training visualization has different sample settings")
        for role, latent in latents.items():
            recorded = saved["training_latents"][role]
            stored = torch.load(recorded["latent"], map_location="cpu", weights_only=True)
            if records.tensor_hash(stored) != records.tensor_hash(latent):
                raise ValueError("replayed training visualization has different latent tensors")
            if provenance.file_sha256(recorded["video"]) != recorded["video_sha256"]:
                raise ValueError("saved training visualization video changed")
        if any(provenance.file_sha256(item["path"]) != item["sha256"] for item in saved["outputs"].values()):
            raise ValueError("saved training visualization comparison changed")
        return  # Completed previews retain their original memory-dependent tiling.
    frames = (latents["gt"].shape[2] - 1) * current.model.scale_factors.time + 1
    # The official selector accounts for decoder halos, weights and free memory.
    # Select before loading the VAE, with the training transformer still resident.
    scales = current.model.scale_factors
    tiling = tiling_config_for_vae(
        current.model.paths.video_vae(),
        height=latents["gt"].shape[3] * scales.height,
        width=latents["gt"].shape[4] * scales.width,
        num_frames=frames,
        device=current.device,
    )
    frame_ids = tuple(range(frames))
    titles = dict(noisy="Noisy latent", gt="GT latent", predicted_x0="Predicted x0")
    panels, artifacts = [], {}
    with torch.inference_mode(), current.decoder() as decoder:
        for role, latent in latents.items():
            records.publish_tensor(destination / f"{role}.pt", latent)
            generator = torch.Generator(device=current.device).manual_seed(42)
            chunks = []
            for chunk in decoder.decode_video(latent.to(device=current.device, dtype=session.DTYPE), tiling, generator):
                if not torch.isfinite(chunk).all():
                    raise ValueError("training visualization decoder returned nonfinite pixels")
                chunks.append(chunk.clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8))
            pixels = torch.cat(chunks).permute(0, 3, 1, 2).contiguous()
            if len(pixels) != frames:
                raise ValueError("training visualization decoder changed the frame count")
            path = destination / f"{role}.mp4"
            records.publish(
                path, lambda p, pixels=pixels: save_video(pixels, p, fps=settings["fps"], video_format="FCHW")
            )
            artifacts[role] = dict(
                video=str(path),
                video_sha256=provenance.file_sha256(path),
                latent=str(destination / f"{role}.pt"),
                latent_sha256=records.tensor_hash(latent),
            )
            panels.append(media.Panel(role, titles[role], pixels, frame_ids, value="VAE decode"))
    combined, rendering = media.render_panels(
        panels,
        question="What does the model predict for this training sample?",
        layout="comparison",
        fps=settings["fps"],
        viewing_width=480,
        poster_frame=frames // 2,
        common_settings=dict(**settings, decoder_tiling=asdict(tiling), decoder_seed=42),
    )
    rendering["training_latents"] = artifacts
    media.save_render(combined, rendering, destination)


def save(current, preview, destination, *, runtime=None):
    """Decode on rank zero, restore all RNG, then let all ranks continue together."""
    failure = None
    if preview is not None:
        rng = checkpoints.rng_state([current.device.index] if current.device.type == "cuda" else [])
        try:
            _write(current, preview, destination)
        except Exception as error:
            failure = error
        finally:
            checkpoints.restore_rng(rng)
            if current.device.type == "cuda":
                torch.cuda.empty_cache()
    failed = runtime.any(failure is not None) if runtime else failure is not None
    if failed:
        raise failure or RuntimeError("rank zero failed to save the training visualization")
