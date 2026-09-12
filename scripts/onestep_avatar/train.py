"""SS7.1's bespoke autoregressive LoRA loop for the one-step avatar renderer.

``plans/2026-09-10-ltx25-one-step-argavatar-lora.md`` SS7.1 explains why this is not a
``ltx-trainer`` strategy: ``Trainer._training_step`` runs exactly one transformer forward
per step and the strategy interface does not own the forward, so a ``K``-window AR chain
cannot be expressed as one. Only the *step* is ours -- model loading, LoRA injection,
FSDP preparation and checkpoint plumbing are all reused from ``ltx_trainer``.

Three things this loop does that the shared trainer cannot:

1. **The noisy state is built from a different latent than the loss target.** The init is
   the ARGAvatar guide ``z_g`` and the target is the capture ``z_y`` (SS3). When
   ``z_g == z_y`` the target reduces exactly to ``eps - z_y``, the ordinary flow-matching
   target -- ``tests/test_train.py`` pins that, so this is a strict generalisation of what
   the trainer already does rather than a parallel objective.
2. **The carryover is the model's own previous output**, not the ground truth. The
   trainer's ``_apply_intrinsic_condition`` substitutes ``clean_latents``, which is exactly
   the teacher forcing SS4.4 removes: the deployed rollout feeds the model its own error, so
   training that never sees it drifts (measured on ``k2`` at -48.98 dB / 100 chunks).
3. **The window state is the deployed one.** Every forward goes through
   ``refine_core.make_window_state``, the same call the deployed refiner makes -- so RoPE
   positions, the index-0 causal keyframe, the frozen carryover at index 1 and the
   denoise mask are identical to inference by construction, not by two implementations
   agreeing.

sigma_0 is fixed (SS4.2, default 0.725): the distilled checkpoint is a deterministic map on a
9-point grid, not a continuum, so there is no sampler in this loop at all. sigma_0, ``K`` and
the subset hash go into the checkpoint metadata, because a fixed-sigma adapter loaded at
another sigma or run multi-step fails silently (SS9 risk 13).

Run from ``LTX-2`` in the ``ltx`` env. Two or three GPUs is a preliminary-scale run -- drop
the rank rather than the chain length, since ``K`` is what the loop exists to exercise::

    accelerate launch --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml \\
      -m scripts.onestep_avatar.train \\
      --subset ../expr/onestep_avatar/windows/t2.json \\
      --precomputed ../expr/onestep_avatar/precomputed \\
      --output ../expr/onestep_avatar/runs/prelim --lora-rank 8 --steps 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft.utils.other import fsdp_auto_wrap_policy
from safetensors.torch import save_file

from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core.tools import VideoLatentTools
from ltx_core.utils import to_denoised
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_trainer.model_loader import load_transformer
from scripts.prune.core import model_registry, refine_core, refine_task
from scripts.prune.data import prompt_cache

LOGGER = logging.getLogger("onestep_avatar.train")
DTYPE = torch.bfloat16

# SS4.2: the deployed operating point, with a validated k2 baseline. NOT swept -- the distilled
# grid admits only {0.421875, 0.725, 0.909375}, and 0.725 is the one where the base model
# already moves texture by about the right amount (28.59 dB output-vs-input on 2.3).
DEFAULT_SIGMA0 = 0.725

# Two named target sets. "attn" is the trainer's own default projection set; "attn_ffn" adds
# the feed-forward projections, which is the A2 sweep's second axis (SS A2 "attn-only vs
# attn+FFN"). Named here rather than passed as a free list so a run's arm is one word in the
# checkpoint metadata.
LORA_TARGETS = {
    "attn": ["to_k", "to_q", "to_v", "to_out.0"],
    "attn_ffn": ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
}


@dataclass(frozen=True)
class Window:
    """One precomputed training window: the guide init, the capture target, and provenance."""

    z_g: torch.Tensor  # [C, F, H, W] -- the noising source (ARGAvatar render)
    z_y: torch.Tensor  # [C, F, H, W] -- the loss target (real capture)
    fps: float
    index: int
    source: str
    loss_mask: torch.Tensor | None  # [F, H, W] latent-resolution subject coverage, or None
    z0_base: torch.Tensor | None  # [C, F, H, W] frozen-base one-step output, for the anchor


@dataclass(frozen=True)
class Chain:
    """``K`` consecutive windows of one source -- SS4.4's training sample."""

    source: str
    split: str
    actor: str
    seed_is_clip_start: bool
    windows: list[Window]


def _load_record(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


class ChainStore:
    """Reads ``windows.py``'s frozen subset against ``precompute.py``'s output tree.

    Lazy per chain: a chain is ~3 x 3 x 1.05 MB of bf16 latents, and a T3 subset is tens of
    thousands of windows, so nothing is held resident. The subset JSON is the only thing
    parsed up front -- it is also the only place the split lives, so a held-out actor cannot
    leak into training by a path convention.
    """

    def __init__(
        self,
        subset: dict,
        precomputed: Path,
        *,
        split: str,
        loss_mask_kind: str,
        with_anchor: bool,
    ) -> None:
        self.subset = subset
        self.root = precomputed
        self.loss_mask_kind = loss_mask_kind
        self.with_anchor = with_anchor
        self.chains = [chain for chain in subset["chains"] if chain["split"] == split]
        if not self.chains:
            raise SystemExit(f"subset has no chains in split {split!r}")
        self.sources = {record["relative_dir"]: record for record in subset["sources"]}

    def __len__(self) -> int:
        return len(self.chains)

    def _window(self, source: str, index: int) -> Window:
        rel = Path(source) / f"window_{index:04d}.pt"
        init = _load_record(self.root / "init_latents" / rel)
        target = _load_record(self.root / "target_latents" / rel)
        if init["latents"].shape != target["latents"].shape:
            raise ValueError(f"{rel}: guide {tuple(init['latents'].shape)} != capture {tuple(target['latents'].shape)}")
        if init["fps"] != target["fps"]:
            raise ValueError(f"{rel}: guide fps {init['fps']} != capture fps {target['fps']}")

        loss_mask = None
        if self.loss_mask_kind != "none":
            record = _load_record(self.root / "loss_masks" / rel)
            loss_mask = _combine_masks(record, self.loss_mask_kind)

        z0_base = None
        if self.with_anchor:
            z0_base = _load_record(self.root / "base_denoised" / rel)["latents"]

        return Window(
            z_g=init["latents"],
            z_y=target["latents"],
            fps=float(init["fps"]),
            index=index,
            source=source,
            loss_mask=loss_mask,
            z0_base=z0_base,
        )

    def __getitem__(self, i: int) -> Chain:
        chain = self.chains[i]
        return Chain(
            source=chain["source"],
            split=chain["split"],
            actor=chain["actor"],
            seed_is_clip_start=bool(chain["seed_is_clip_start"]),
            windows=[self._window(chain["source"], index) for index in chain["windows"]],
        )


def _combine_masks(record: dict, kind: str) -> torch.Tensor:
    """Turn the two stored coverage grids into the one mask the loss weights by.

    Both are kept separately on disk on purpose (SS4.3 row 1 names the render's alpha, but the
    *target* is the capture, and the two disagree by exactly the SSB1 IoU gap). Which
    disagreement region the loss should cover is a training decision, so it is made here:

    * ``render``      -- where the model is asked to paint something.
    * ``capture``     -- where the target is meaningful. Excludes the render's spurious limbs.
    * ``union``       -- both, so the model is also taught to *remove* what is not there.
    * ``intersection``-- neither disputed region: the conservative reading of "silhouette
      mismatch would otherwise be learned as signal", at the cost of never learning the
      silhouette itself.
    """
    render, capture = record["render_alpha"].float(), record["capture_mask"].float()
    if kind == "render":
        return render
    if kind == "capture":
        return capture
    if kind == "union":
        return torch.maximum(render, capture)
    if kind == "intersection":
        return torch.minimum(render, capture)
    raise ValueError(f"unknown loss mask kind {kind!r}")


def _as_token_weights(mask_5d: torch.Tensor, tools: VideoLatentTools) -> torch.Tensor:
    """Latent-grid coverage ``[1, 1, F, H, W]`` -> per-token weights ``[1, seq, 1]``.

    Patchified through the model's own patchifier rather than a reshape, so the mask lands on
    the same tokens the latent does for any patch size.
    """
    return tools.patchifier.patchify(mask_5d).mean(dim=-1, keepdim=True)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted MSE over tokens, normalised by the weight mass.

    Normalising (rather than averaging over all tokens) keeps the loss scale independent of
    how much of the frame the subject occupies, so a wide crop and a tight one contribute
    comparably instead of the tight one dominating.
    """
    error = (pred.float() - target.float()).pow(2)
    weighted = error * weights
    return weighted.sum() / weights.expand_as(error).sum().clamp(min=1e-8)


def one_window_forward(
    transformer: torch.nn.Module,
    context: torch.Tensor,
    window: Window,
    carry: torch.Tensor | None,
    geometry: refine_core.WindowGeometry,
    *,
    sigma0: float,
    seed: int,
    device: torch.device,
    latent_channels: int,
    guide_mode: str = "d1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object, object]:
    """One guided-init forward -> ``(z0_tokens, target_tokens, token_weights, state, tools)``.

    ``state`` and ``tools`` come back because the caller needs them after the loss: the
    carryover is unpatchified through the same tools, and the anchor term patchifies its
    cached target with them. Rebuilding either would be a second producer of the token
    layout.

    The state is built by ``refine_core.make_window_state`` -- the deployed call -- with
    ``l_init`` set to the *guide*. ``GaussianNoiser`` then computes
    ``lerp(z_g, eps, sigma_0) = (1 - sigma_0) z_g + sigma_0 eps`` and restores the
    conditioned slots from ``clean_latent``, which is where ``carry`` has been written. So
    the init is SS3's ``x_sigma0`` and the carryover is ours, not the ground truth.
    """
    _, _, height, width = window.z_g.shape
    tools = refine_core.tools_for_window(
        geometry,
        height * geometry.scale_factors.height,
        width * geometry.scale_factors.width,
        window.fps,
        latent_channels=latent_channels,
    )
    z_g = window.z_g.unsqueeze(0).to(device=device, dtype=DTYPE)
    extra = ()
    if guide_mode == "d2":
        # SS4.1's hybrid: the SAME guide, a second time, as clean tokens appended at
        # timestep 0. At scale factor 1 they land on the target's own RoPE positions, so the
        # model gets a pixel-aligned copy of the guide that sigma_0's noise has NOT degraded --
        # which is the whole point, since the init's copy is degraded and therefore carries a
        # weaker constraint. Costs ~2.3x attention (2T tokens instead of T).
        extra = (
            VideoConditionByReferenceLatent(
                latent=z_g, downscale_factor=1, temporal_scale_factor=1, strength=1.0
            ),
        )
    state = refine_core.make_window_state(
        z_g, carry, sigma0, tools, seed, device, DTYPE, extra_conditionings=extra
    )

    sigma = torch.tensor(sigma0, device=device, dtype=DTYPE)
    modality = modality_from_latent_state(state, context, sigma.expand(state.latent.shape[0]))
    velocity, _ = transformer(video=modality, audio=None, perturbations=None)
    # SS2.2: the transformer natively emits velocity and `to_denoised` is an exact algebraic
    # identity, so predicting z0 here costs nothing and makes the loss directly comparable
    # with the capture latent. At fixed sigma_0, velocity MSE == x0 MSE / sigma_0**2 (SS3).
    z0_tokens = to_denoised(state.latent, velocity, modality.timesteps)

    z_y = window.z_y.unsqueeze(0).to(device=device, dtype=DTYPE)
    target_tokens = tools.patchifier.patchify(z_y)

    # The frozen carryover and the causal keyframe carry denoise_mask 0; they are conditioning,
    # not prediction, so they must not contribute to the loss (the trainer's own rule).
    weights = state.denoise_mask.float()
    if weights.dim() == 2:
        weights = weights.unsqueeze(-1)
    # D2 appends its reference tokens AFTER the target's, so the target is the leading T
    # tokens -- the opposite end from `flexible._apply_reference_condition`, which prepends and
    # therefore slices `[:, -target_len:]`. Slicing is a no-op under D1 (the counts are equal),
    # so there is one code path rather than a branch.
    n_target = target_tokens.shape[1]
    z0_tokens = z0_tokens[:, :n_target]
    weights = weights[:, :n_target]
    if window.loss_mask is not None:
        coverage = window.loss_mask.to(device=device, dtype=torch.float32)
        weights = weights * _as_token_weights(coverage.unsqueeze(0).unsqueeze(0), tools)
    return z0_tokens, target_tokens, weights, state, tools


def train_chain(
    transformer: torch.nn.Module,
    context: torch.Tensor,
    chain: Chain,
    geometry: refine_core.WindowGeometry,
    accelerator: Accelerator,
    *,
    sigma0: float,
    seed: int,
    anchor_weight: float,
    latent_channels: int,
    guide_mode: str = "d1",
) -> dict[str, float]:
    """SS4.4's chain: ``K`` forwards and ``K`` backwards, one optimizer step, detach between.

    Detaching the carryover is what keeps peak activation memory at a *single* window rather
    than ``K`` of them -- the reason AR training does not raise the SS8.1 memory budget at all.
    It also means window ``i``'s loss never backpropagates into window ``i-1``'s forward,
    which is correct: the carryover is an input the deployed model is handed, not something
    this step gets to optimise through.
    """
    device = accelerator.device
    carry = None
    if not chain.seed_is_clip_start:
        # The chain seed takes the GT carryover (SS4.4). A clip's very first window has no
        # predecessor at deployment either, so it takes none.
        #
        # It is the GT at the DESTINATION slot, not `carry_from` of this window. `carry_from`
        # takes a window's LAST latent frame, which is right when the value comes from the
        # previous window's output -- at a 16-frame stride, window w-1's last latent frame and
        # window w's latent frame 1 are the same master frame (2w+1). Applied to window w's own
        # latents it would instead pick master frame 2w+3: two latent frames into the future,
        # a seed no rollout ever sees.
        idx = refine_core.CARRYOVER_LATENT_IDX
        n = geometry.context_latent_frames
        gt = chain.windows[0].z_y.unsqueeze(0).to(device=device, dtype=DTYPE)
        carry = gt[:, :, idx : idx + n].contiguous()

    totals = {"loss": 0.0, "mse": 0.0, "anchor": 0.0}
    k = len(chain.windows)
    for i, window in enumerate(chain.windows):
        z0_tokens, target_tokens, weights, state, tools = one_window_forward(
            transformer,
            context,
            window,
            carry,
            geometry,
            sigma0=sigma0,
            seed=seed + window.index,
            device=device,
            latent_channels=latent_channels,
            guide_mode=guide_mode,
        )
        mse = masked_mse(z0_tokens, target_tokens, weights)
        loss = mse
        anchor = torch.zeros((), device=device)
        if anchor_weight > 0.0:
            if window.z0_base is None:
                raise ValueError("--anchor-weight > 0 but the subset has no base_denoised/ products")
            # SS4.3 row 2 / SS2.3(3): the risk here is ERODING sharpness Phi already has, not
            # failing to synthesise it. Pulling toward the frozen model's own output on the
            # same input is the cheapest thing that targets that directly.
            base_tokens = tools.patchifier.patchify(window.z0_base.unsqueeze(0).to(device=device, dtype=DTYPE))
            anchor = masked_mse(z0_tokens, base_tokens, weights)
            loss = loss + anchor_weight * anchor

        accelerator.backward(loss / k)
        totals["loss"] += float(loss.detach()) / k
        totals["mse"] += float(mse.detach()) / k
        totals["anchor"] += float(anchor.detach()) / k

        if i + 1 < k:
            z0_latent = refine_core.finalize(replace(state, latent=z0_tokens), tools)
            carry = refine_core.carry_from(z0_latent, geometry).detach()
        del z0_tokens, target_tokens, weights, state, tools, loss, mse, anchor
    return totals


def build_transformer(
    model: model_registry.RefinerModel, args: argparse.Namespace, accelerator: Accelerator
) -> torch.nn.Module:
    """Load the frozen bf16 backbone and inject LoRA -- ``ltx_trainer``'s own plumbing."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Loading straight onto the GPU, not via host RAM: the full bf16 checkpoint is 42 GB and
    # three ranks staging it on the host would need 126 GB of a machine that has ~139 GB free
    # here. FSDP shards in place afterwards, so the 42 GB is transient and fits a 49 GB card.
    init_device = args.init_device if args.init_device != "cuda" else f"cuda:{local_rank}"
    transformer = load_transformer(
        checkpoint_path=model.paths.transformer(), device=init_device, dtype=DTYPE, video_only=True
    )
    transformer.requires_grad_(False)
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGETS[args.lora_target],
            lora_dropout=0.0,
            init_lora_weights=True,
        ),
    )
    if accelerator.distributed_type == DistributedType.FSDP:
        # FSDP needs one dtype per flat parameter, and PEFT makes the adapters fp32 against a
        # bf16 base. This policy wraps the trainable leaves separately, which is what lets the
        # base stay bf16 instead of being promoted to a full fp32 host copy before sharding.
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(transformer)
    transformer.get_base_model().set_gradient_checkpointing(not args.no_gradient_checkpointing)
    return transformer


def checkpoint_metadata(
    args: argparse.Namespace, subset: dict, model: model_registry.RefinerModel, step: int
) -> dict[str, str]:
    """SS7.1 / SS9 risk 13: a fixed-sigma adapter must not be loadable off-condition.

    sigma_0 and ``K`` are recorded so ``refine_task``'s future ``ONE_STEP`` schedule can refuse
    a checkpoint whose sigma disagrees or a multi-step run, both of which would otherwise fail
    silently -- extra steps are extrapolation for a map that was distilled to a fixed grid.
    """
    return {
        "onestep_avatar_sigma0": repr(args.sigma0),
        "onestep_avatar_chain_length": str(subset["chain_length"]),
        "onestep_avatar_schedule": "ONE_STEP",
        "onestep_avatar_subset_sha256": hashlib.sha256(
            json.dumps(subset["sources"], sort_keys=True).encode()
        ).hexdigest(),
        "onestep_avatar_loss_mask": args.loss_mask,
        "onestep_avatar_guide_mode": args.guide_mode,
        "onestep_avatar_anchor_weight": repr(args.anchor_weight),
        "model_key": model.key,
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "lora_target": args.lora_target,
        "step": str(step),
    }


def save_lora(
    transformer: torch.nn.Module,
    accelerator: Accelerator,
    out_dir: Path,
    step: int,
    metadata: dict[str, str],
) -> Path | None:
    """Gather and write the adapter in the trainer's own ComfyUI-compatible layout."""
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(transformer)
    if not accelerator.is_main_process:
        return None
    unwrapped = accelerator.unwrap_model(transformer, keep_torch_compile=False)
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    state_dict = get_peft_model_state_dict(unwrapped, state_dict=state_dict if is_fsdp else None)
    state_dict = {f"diffusion_model.{k.replace('base_model.model.', '', 1)}": v for k, v in state_dict.items()}
    state_dict = {k: v.to(torch.bfloat16).contiguous() for k, v in state_dict.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lora_weights_step_{step:05d}.safetensors"
    save_file(state_dict, path, metadata=metadata)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=Path, required=True, help="windows.py's frozen subset JSON")
    p.add_argument("--precomputed", type=Path, required=True, help="precompute.py --output-root")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    p.add_argument("--split", choices=("train", "held_out"), default="train")
    p.add_argument("--lora-rank", type=int, default=8, help="2-3 GPU preliminary runs drop this, never K")
    p.add_argument("--lora-alpha", type=int, default=None, help="default: equal to --lora-rank")
    p.add_argument("--lora-target", choices=sorted(LORA_TARGETS), default="attn")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss-mask", choices=("none", "render", "capture", "union", "intersection"), default="none")
    p.add_argument(
        "--guide-mode",
        choices=("d1", "d2"),
        default="d1",
        help="SS4.1. d1: the guide reaches the model only as the noised init -- cheapest, and "
        "what refine_core deploys today. d2: ALSO as clean reference tokens at timestep 0, "
        "~2.3x attention. The measured subject-interior r of 0.89-0.93 against a 0.6 threshold "
        "(SS0.3) says d2 is the arm to lead with; the default stays d1 so the arm is always "
        "explicit in the command line and in the checkpoint metadata.",
    )
    p.add_argument("--anchor-weight", type=float, default=0.0, help="SS4.3 row 2; needs base_denoised/")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--init-device", default="cuda", help="'cuda' (default, avoids host-RAM staging) or 'cpu'")
    p.add_argument("--dry-run", action="store_true", help="report the plan and the data shapes, load no model")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915 -- one linear training script.
    args = parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    subset = json.loads(args.subset.read_text())
    if subset.get("kind") != "one_step_argavatar_window_chains":
        raise SystemExit(f"{args.subset} is not a windows.py subset")
    model = model_registry.resolve(args.model)
    store = ChainStore(
        subset,
        args.precomputed,
        split=args.split,
        loss_mask_kind=args.loss_mask,
        with_anchor=args.anchor_weight > 0.0,
    )

    if args.dry_run:
        chain = store[0]
        print(  # noqa: T201 -- CLI's requested plan.
            json.dumps(
                {
                    "chains": len(store),
                    "chain_length": subset["chain_length"],
                    "first_chain": {
                        "source": chain.source,
                        "actor": chain.actor,
                        "seed_is_clip_start": chain.seed_is_clip_start,
                        "window_shape": list(chain.windows[0].z_g.shape),
                        "fps": chain.windows[0].fps,
                    },
                    "sigma0": args.sigma0,
                    "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "target": args.lora_target},
                },
                indent=2,
            )
        )
        return 0

    # No explicit mixed_precision: the accelerate config decides, and the 2/3-GPU configs are
    # copies of the trainer's own, so this loop runs under the same policy the shipped trainer
    # does rather than a second one of its own.
    accelerator = Accelerator()
    device = accelerator.device
    world, rank = accelerator.num_processes, accelerator.process_index

    geometry = refine_task.deployed_geometry(model.scale_factors)
    context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)

    transformer = build_transformer(model, args, accelerator)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    # Counted BEFORE `prepare`: FSDP with `use_orig_params=True` reshapes each parameter to
    # this rank's shard in place, so the same expression afterwards reports total/world_size
    # and reads like a model half the size.
    trainable_total = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    if accelerator.is_main_process:
        LOGGER.info(
            "trainable params: %s total, %s per rank across %d (%s rank %d, alpha %d)",
            f"{trainable_total:,}",
            f"{trainable_total // world:,}",
            world,
            args.lora_target,
            args.lora_rank,
            args.lora_alpha,
        )

    # Chains are sharded by rank rather than by an accelerate DataLoader: a sample here is a
    # variable-length chain of tensors, not a collatable batch, and FSDP is data-parallel over
    # ranks, so a deterministic stride is both simpler and reproducible with no sampler state.
    # Every rank runs the SAME number of steps, so the shard is truncated to the common length.
    per_rank = len(store) // world
    if per_rank == 0:
        raise SystemExit(f"{len(store)} chains cannot be split across {world} ranks")
    order = list(range(len(store)))

    args.output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (args.output / "config.json").write_text(json.dumps({**vars(args), "world_size": world}, indent=2, default=str))
    log_path = args.output / f"metrics_rank{rank}.jsonl"
    log_file = log_path.open("a")

    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    started = time.time()
    while step < args.steps:
        epoch_order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        shard = epoch_order[rank * per_rank : (rank + 1) * per_rank]
        for chain_index in shard:
            if step >= args.steps:
                break
            lr = args.lr * min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = lr

            chain = store[chain_index]
            totals = train_chain(
                transformer,
                context,
                chain,
                geometry,
                accelerator,
                sigma0=args.sigma0,
                # Seeded per (run, window) so eps is reproducible and the same window always
                # gets the same noise -- which is also what makes a cached frozen-base output
                # (the anchor term) correspond to this exact input.
                seed=args.seed * 100003 + chain_index * 101,
                anchor_weight=args.anchor_weight,
                latent_channels=model.caps.latent_channels,
                guide_mode=args.guide_mode,
            )
            grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "rank": rank,
                    "lr": lr,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "elapsed_s": round(time.time() - started, 1),
                    "source": chain.source,
                    **{k: round(v, 6) for k, v in totals.items()},
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if accelerator.is_main_process:
                    LOGGER.info(
                        "step %d/%d loss %.5f mse %.5f anchor %.5f lr %.2e %.1fs",
                        step, args.steps, totals["loss"], totals["mse"], totals["anchor"], lr,
                        time.time() - started,
                    )
            if step % args.save_every == 0 or step == args.steps:
                path = save_lora(
                    transformer, accelerator, args.output / "checkpoints", step,
                    checkpoint_metadata(args, subset, model, step),
                )
                if path is not None:
                    LOGGER.info("saved %s", path)

    log_file.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        LOGGER.info("done: %d steps in %.1f min", step, (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
