"""Continuous F18 training windows from validated full-source input bundles.

The opt-in uniform_f18 config keeps all F18 latents and uniformly chooses one
18-frame span in longer clips. A separate RNG keyed by the draw's noise seed
makes duplicate ranks and resume agree without consuming optimizer RNG state.
Each window has local positions and a clean first latent, including when the
source start is nonzero. Full source manifests remain unchanged; the draw
receipt binds the actual window and its capture/noise tensor hashes.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from scripts.onestep_avatar.corpus.dataset import load_training_master
from scripts.onestep_avatar.model import common
from scripts.prune.core.provenance import file_sha256, verified_sha256
from scripts.prune.core.records import tensor_hash
from scripts.prune.data import whole_clip

LATENT_FRAMES = 18


def enabled(config):
    value = config.get("uniform_f18", False)
    if type(value) is not bool:
        raise ValueError("uniform_f18 must be a boolean")
    return value


def bounds(frames, seed):
    if type(frames) is not int or frames < LATENT_FRAMES or type(seed) is not int:
        raise ValueError("uniform F18 requires at least 18 latent frames and an integer draw seed")
    start = random.Random(f"uniform_f18:{seed}").randrange(frames - LATENT_FRAMES + 1)
    return start, start + LATENT_FRAMES


def nonzero_crop_seed(frames, seed, stride):
    """First seed in `seed, seed+stride, ...` whose crop of a longer source starts after zero."""
    while frames > LATENT_FRAMES and bounds(frames, seed)[0] == 0:
        seed += stride
    return seed


def window_record(frames, seed, per_frame, capture, epsilon):
    start, stop = bounds(frames, seed)
    return {
        "mode": "uniform_f18",
        "source_latent_frames": frames,
        "latent_start": start,
        "latent_stop": stop,
        "latent_frames": LATENT_FRAMES,
        "noise_seed": seed,
        "tokens_per_latent_frame": per_frame,
        "position_policy": "window_local_first_keyframe",
        "capture_tensor_sha256": tensor_hash(capture),
        "epsilon_tensor_sha256": tensor_hash(epsilon),
    }


def crop_case(case, current, seed, sigma):
    full_grid, full_video, _, capture, epsilon, root, manifest = case
    start, stop = bounds(full_grid.latent_frames, seed)
    per_frame = full_grid.tokens_per_latent_frame
    if capture.shape != epsilon.shape or capture.shape[1] != full_grid.latent_frames * per_frame:
        raise ValueError("training window source token geometry differs")
    lo, hi = start * per_frame, stop * per_frame
    capture, epsilon = capture[:, lo:hi].contiguous(), epsilon[:, lo:hi].contiguous()
    spatial = full_grid.tools.target_shape
    scales = current.model.scale_factors
    grid = common.ClipGrid.build(
        LATENT_FRAMES,
        spatial.height * scales.height,
        spatial.width * scales.width,
        full_grid.fps,
        current.model,
        device=capture.device,
        dtype=capture.dtype,
        latent_channels=capture.shape[-1],
    )
    c0 = capture[:, :per_frame]
    noisy = common.with_clean_prefix(common.mix_block_noise(capture, epsilon, sigma), c0)
    video = common.block_modality(
        grid,
        noisy,
        full_video.context,
        sigma,
        token_slices=[grid.token_span(0, LATENT_FRAMES)],
        attention_mask=None,
        clean_prefix_tokens=per_frame,
        sigma_dtype=torch.float32,
    )
    record = window_record(full_grid.latent_frames, seed, per_frame, capture, epsilon)
    return grid, video, c0, capture, epsilon, root, {**manifest, "training_window": record}


def slice_regions(regions, record):
    if regions is None or record is None:
        return regions
    lo = record["latent_start"] * record["tokens_per_latent_frame"]
    hi = record["latent_stop"] * record["tokens_per_latent_frame"]
    expected = record["source_latent_frames"] * record["tokens_per_latent_frame"]
    if any(value.ndim != 2 or value.shape[1] != expected for value in regions.values()):
        raise ValueError("region weights do not cover the full source clip")
    return {name: value[:, lo:hi].contiguous() for name, value in regions.items()}


def validate_saved_window(root, record, *, capture_hashes=None):
    """Verify a consumed crop against the original pinned capture and noise.

    `capture_hashes` maps capture paths to already verified SHA-256 values, so a
    caller that checks many draws hashes each shared capture file once.
    """
    root = Path(root)
    manifest = whole_clip.load_manifest(root)
    row = manifest["videos"][0]
    path = Path(row["artifacts"]["capture"])
    known = None if capture_hashes is None else capture_hashes.get(str(path))
    if known is None:
        known = file_sha256(path)
        if capture_hashes is not None:
            capture_hashes[str(path)] = known
    if known != row["artifacts"]["capture_sha256"]:
        raise ValueError("training window capture source changed")
    capture, fps = load_training_master(path)
    epsilon = whole_clip.load_epsilon(root, row)
    channels, frames, height, width = capture.shape
    per_frame = height * width
    if fps != row["artifacts"]["fps"] or epsilon.shape != (1, frames * per_frame, channels):
        raise ValueError("training window source geometry or fps changed")
    start, stop = bounds(frames, manifest["seed"])
    clean = capture[:, start:stop].reshape(channels, -1).T.unsqueeze(0).contiguous()
    noise = epsilon[:, start * per_frame : stop * per_frame].contiguous()
    if record != window_record(frames, manifest["seed"], per_frame, clean, noise):
        raise ValueError("consumed training window changed")


def verify_consumed_inputs(draws, *, manifest_key, windows=True):
    """Check every consumed input manifest, saved epsilon and training crop.

    Each epsilon is hashed and loaded once; each shared capture is hashed once.
    `manifest_key` names the receipt field that pins the input manifest bytes.
    """
    capture_hashes = {}
    for draw in draws:
        path = Path(draw["input_manifest"])
        if verified_sha256(path) != draw[manifest_key]:
            raise ValueError("consumed training input changed since checkpoint")
        if windows:
            validate_saved_window(path.parent, draw["training_window"], capture_hashes=capture_hashes)
        else:
            whole_clip.load_epsilon(path.parent, whole_clip.load_manifest(path.parent)["videos"][0])
