"""Decode the D0 GT-renoise sanity probe at the distilled refiner's four levels.

D0 is deliberately not deployable: it noises the capture latent itself.  That makes it a
useful capacity control, but only if its review artifact uses that exact state rather than a
guide-noised approximation.  This script runs a fixed three-window chain at each of the first
four values in the distilled sigma grid and writes one portable MP4 per level:

    ground-truth capture | frozen base | D0 LoRA checkpoint

The fixed chain and seeds make a sequence of checkpoints directly comparable.  It is an
offline checkpoint probe, so no VAE is resident while FSDP training is stepping.

Run from ``LTX-2`` in the ``ltx`` environment::

python -m scripts.onestep_avatar.visualize_d0 \
    --subset ../expr/onestep_avatar/windows/prelim2.json \
    --precomputed ../expr/onestep_avatar/precomputed \
    --checkpoint ../expr/onestep_avatar/runs/test/checkpoints/lora_weights_step_00900.safetensors \
    --output ../expr/onestep_avatar/runs/test/probes/step_00900 --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from scripts.onestep_avatar.train import Chain, ChainStore
from scripts.prune.core import refine_core, refine_task
from scripts.prune.core.session import DTYPE, add_model_args, open_session
from scripts.prune.evaluate.decode import decode_latent
from scripts.prune.evaluate.metrics import t3_video


# This is the distilled *refiner* schedule's non-trivial levels, not the first entries of the
# full nine-value generation table.  The latter are all nearly pure noise and made the prior
# D0 probe uninformative.  Keep training and checkpoint review on these exact levels.
#
# sigma=0.0 is deliberately excluded: `_run_d0_chain` steps every level through
# `refine_core.run_schedule(..., torch.tensor([sigma, 0.0]))`, which converts the model's
# denoised prediction back into a velocity via `to_velocity(sample, sigma, denoised)` --
# `(sample - denoised) / sigma`, undefined at sigma=0.0 (`ltx_core.utils.to_velocity` raises
# "Sigma can't be 0.0" there; this crashed the D0 probe in a silent retry loop for hours before
# being caught). It is also not an informative probe point: sigma=0.0 adds no noise, so the
# "denoised" state IS the input and there is no step to take -- the same reason train.py's
# `training_sigmas` now refuses to train that level (see train.py's docstring there).
PROBE_SIGMAS = (0.909375, 0.725, 0.421875)


def _chain(store: ChainStore, index: int) -> Chain:
    if index < 0 or index >= len(store):
        raise SystemExit(f"--chain-index {index} is outside [0, {len(store) - 1}]")
    chain = store[index]
    if not chain.seed_is_clip_start:
        raise SystemExit(
            "the D0 visual probe must start at a clip boundary so its first carry is not a "
            "teacher-forced GT fragment; choose a chain whose seed_is_clip_start is true"
        )
    return chain


def _run_d0_chain(transformer, denoiser, chain: Chain, geometry, sigma: float, *, device, latent_channels: int, seed: int):  # noqa: ANN001
    """Exact D0 AR forward: ``z_y`` is both the noising source and the target reference.

    This intentionally does not call :mod:`onestep_core`: that module refuses D0 to protect
    deployment from accidentally accepting an arm that needs the unavailable capture latent.
    ``make_window_state`` / ``run_schedule`` are nevertheless the shared deployed primitives,
    so conditioning slots, RoPE and carryover remain identical to training.
    """
    carry = None
    outputs = []
    for i, window in enumerate(chain.windows):
        _, _, height, width = window.z_y.shape
        tools = refine_core.tools_for_window(
            geometry,
            height * geometry.scale_factors.height,
            width * geometry.scale_factors.width,
            window.fps,
            latent_channels=latent_channels,
        )
        z_y = window.z_y.unsqueeze(0).to(device=device, dtype=DTYPE)
        state = refine_core.make_window_state(z_y, carry, sigma, tools, seed + window.index, device, DTYPE)
        final = refine_core.finalize(
            refine_core.run_schedule(
                transformer, denoiser, state, torch.tensor([sigma, 0.0], device=device),
            ),
            tools,
        )
        outputs.append(final)
        if i + 1 < len(chain.windows):
            carry = refine_core.carry_from(final, geometry).detach()
    return outputs


def _stitch(decoded: list[torch.Tensor], overlap_frames: int) -> torch.Tensor:
    """Keep the first window then each later non-overlap tail, as the rollout review does."""
    return torch.cat([decoded[0], *(part[overlap_frames:] for part in decoded[1:])], dim=0)


def _decode_chain(session, decoder, latents: list[torch.Tensor], overlap_frames: int) -> torch.Tensor:  # noqa: ANN001
    return _stitch([decode_latent(session, latent, decoder) for latent in latents], overlap_frames)


def _target_latents(chain: Chain, device: torch.device) -> list[torch.Tensor]:
    return [window.z_y.unsqueeze(0).to(device=device, dtype=DTYPE) for window in chain.windows]


def _checkpoint_name(path: Path | None) -> str:
    return "frozen_base" if path is None else path.stem.replace("lora_weights_", "")


def generate_checkpoint(args: argparse.Namespace, checkpoint: Path | None, chain: Chain) -> tuple[object, dict[float, list[torch.Tensor]]]:
    session = open_session(args, script="onestep_avatar.visualize_d0")
    geometry = refine_task.deployed_geometry(session.model.scale_factors)
    denoiser = SimpleDenoiser(session.context, None)
    loras = ()
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")
        loras = (LoraPathStrengthAndSDOps(str(checkpoint), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)

    # The VAE is held only after all four model outputs have been calculated, avoiding the
    # transformer+decoder coexistence that §7.4 explicitly excludes from the train loop.
    outputs: dict[float, list[torch.Tensor]] = {}
    with session.transformer(loras=loras) as transformer:
        for sigma in PROBE_SIGMAS:
            outputs[sigma] = _run_d0_chain(
                transformer, denoiser, chain, geometry, sigma,
                device=session.device, latent_channels=session.model.caps.latent_channels, seed=args.seed,
            )

    # CPU tensors keep this result cheap while the next checkpoint's transformer is loaded.
    return session, {sigma: [latent.cpu() for latent in latents] for sigma, latents in outputs.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--precomputed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[], help="Repeat for every 200-step checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "held_out"), default="train")
    parser.add_argument("--chain-index", type=int, default=0, help="Index within --split; must begin at clip start.")
    add_model_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subset = json.loads(args.subset.read_text())
    store = ChainStore(subset, args.precomputed, split=args.split, loss_mask_kind="none", with_anchor=False)
    chain = _chain(store, args.chain_index)
    args.output.mkdir(parents=True, exist_ok=True)
    sessions_and_outputs = [generate_checkpoint(args, checkpoint, chain) for checkpoint in (None, *args.checkpoint)]
    session, base = sessions_and_outputs[0]
    geometry = refine_task.deployed_geometry(session.model.scale_factors)
    written = []
    # Decode once, after all transformer passes.  Every checkpoint video therefore gives the
    # requested before/after comparison in one frame-aligned artifact: GT | frozen base | LoRA.
    with session.decoder() as decoder:
        target = _decode_chain(session, decoder, _target_latents(chain, session.device), geometry.overlap_frames)
        base_pixels = {sigma: _decode_chain(session, decoder, latents, geometry.overlap_frames) for sigma, latents in base.items()}
        for checkpoint, (_unused_session, outputs) in zip((None, *args.checkpoint), sessions_and_outputs, strict=True):
            for sigma, latents in outputs.items():
                candidate = _decode_chain(session, decoder, latents, geometry.overlap_frames)
                path = args.output / f"{_checkpoint_name(checkpoint)}_sigma_{sigma:.6f}.mp4"
                t3_video(target, base_pixels[sigma], candidate, path, fps=chain.windows[0].fps)
                written.append(path)
    (args.output / "manifest.json").write_text(json.dumps({
        "kind": "d0_gt_renoise_probe",
        "source": chain.source,
        "windows": [window.index for window in chain.windows],
        "fps": chain.windows[0].fps,
        "seed": args.seed,
        "sigmas": list(PROBE_SIGMAS),
        "layout": "ground_truth_capture | frozen_base_generated | checkpoint_generated",
        "videos": [str(path.name) for path in written],
    }, indent=2) + "\n")
    for path in written:
        print(path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
