"""Sliding-window K-step diffusion refinement for very long SAM3DGS videos.

Walks a long video in overlapping windows, VAE-encoding + K-step-refining +
VAE-decoding each one, and stitches them into one continuous output.

Window geometry is specified in LATENT frames, not pixel frames: `--window-latent-num`
(default 5) is the number of regular latent frames a window newly denoises, and
`--overlap-latent-num` (default 2) is how many of the previous window's trailing latent
frames it carries over. Every window also holds one extra latent frame for its own
index-0 keyframe (see below), so a window totals `window_latent_num + 1` latent frames.
Pixel-frame quantities (`window_frames = time_scale * window_latent_num + 1`,
`overlap_frames = time_scale * overlap_latent_num + 1`) are derived once in `main()`
against the VAE's probed temporal scale factor and never taken directly from the CLI --
they exist only because `refine_core.WindowGeometry` (shared with `scripts/prune/`)
still speaks pixel frames: reading source video, sizing the VAE encode/decode, and
walking `total_frames` in fixed strides all happen in that space.

Window-to-window continuity: the overlap is carried at the LATENT level, not patched up
afterward in pixels. Each window's last `overlap_latent_num` *refined latent* frames are
frozen (via VideoConditionByLatentIndex, strength=1.0) into the start of the next window
(at latent index 1, never 0 -- see `refine_core.CARRYOVER_LATENT_IDX`), so the next
window continues from the previous window's literal output instead of re-deriving a
fresh guess at the same source content. That fresh-guess re-derivation is what caused a
visible appearance shift at every boundary in an earlier version of this script: each
window renoises from its own local frame 0 (different RoPE position, different point in
the same fixed noise draw for "the same" source frames), so two independently-refined
windows can genuinely diverge in generated appearance.

The stitcher (`Stitcher.add`) does not cross-fade the resulting pixel overlap -- it just
drops window i's first `overlap_frames` decoded pixels (which duplicate window i-1's
tail) and appends the rest. Because `WindowGeometry.plan` tiles at a fixed stride
(`stride_frames = window_frames - overlap_frames`), this is exactly contiguous: window
i-1's appended range always ends exactly where window i's begins, for every window
including the last, with no gap and no duplicated frame -- see the derivation in
`Stitcher`'s docstring. The tradeoff is that any small discontinuity from the causal
VAE decoder's own receptive-field effects at that boundary is no longer smoothed over;
it is now visible verbatim in the kept window's decode.

Memory/robustness: only ever holds one batch of windows' tensors in memory (not the
whole video), and periodically flushes finalized frames to a segment .mp4 under
<out>/segments/, then rebuilds <out>/decode_full.mp4 via an ffmpeg concat (stream
copy, no re-encode) so there is always a playable, up-to-date file on disk. Each
window's raw decode is cached to <out>/window_cache/ and its refined latent to
<out>/latent_cache/, so an interrupted run resumes without redoing GPU work for
already-finished windows.

--encode-mode: "per-window" (default) VAE-encodes each diffusion window's own
overlapping pixel range independently, so every window's local frame 0 is a fresh
single-pixel causal keyframe (deliberate -- see refine_core.CARRYOVER_LATENT_IDX),
at the cost of reading and encoding every overlap region twice. "latent-global"
VAE-encodes the whole video once into one continuous master latent and assembles each
window out of it, which skips the per-window pixel read and encoder pass.

Both modes give a window the SAME token structure -- a genuine single-pixel keyframe in
slot 0, then regular latent frames -- because a window's keyframe cannot be sliced out of
the master (every master frame after index 0 spans time_scale pixels, so none of them is a
single-pixel encode). latent-global therefore encodes each window's own first frame
separately, one pixel frame per window, and prepends it. That matters beyond tidiness:
the 2.5 distilled checkpoint sets use_keyframes_abs_pos_embedding, so the model applies a
learned keyframe embedding to whatever VideoLatentTools marks as slot 0 -- handing it a
regular 8-pixel frame there is a mislabel it actively consumes.

The master itself is built from OVERLAPPING tiles (--encode-chunk-frames,
--encode-overlap-frames) that are blended, not abutting chunks that are concatenated. The
distinction is not cosmetic: every tile boundary re-seeds the causal VAE's replicate pad,
so an abutting concat runs one latent frame long per boundary and silently desynchronises
every later window. See plan_encode_tiles.

The two modes are still not bit-identical (the master's regular frames carry a little
residual tile-blend error that a per-window encode does not), which is what
scripts/prune/checks/method_parity.py gates against -- so leave the default alone if that
gate matters to you.

Usage:
  conda run -n ltx python3 scripts/vae_refine_sliding_window.py \
      --video /path/to/long_video.mp4 \
      --output-dir expr/sam3dgs_vae_refine/<video-id>/k2_longform \
      --k-step k2 --gpu-id 7
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch

torch.cuda.init()
torch.set_grad_enabled(False)  # this script never backprops; every call below relies on this
                                # instead of its own torch.no_grad()
import decord  # noqa: E402
from einops import rearrange  # noqa: E402

decord.bridge.set_bridge("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-core" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))
sys.path.append(str(REPO_ROOT / "packages" / "ltx-pipelines" / "src"))

from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator  # noqa: E402
from ltx_core.model.video_vae.video_vae import map_temporal_slice  # noqa: E402
from ltx_core.tiling import compute_trapezoidal_mask_1d, split_temporal_causal  # noqa: E402
from ltx_pipelines.utils.blocks import DiffusionStage  # noqa: E402
from ltx_pipelines.utils.denoisers import SimpleDenoiser  # noqa: E402
from ltx_trainer.video_utils import save_video  # noqa: E402

from scripts.prune.core import (  # noqa: E402
    geometry,
    ltx_adapter,
    model_registry,
    preflight,
    refine_core,
    refine_task,
)
from scripts.prune.data import prompt_cache  # noqa: E402
from scripts.prune.evaluate.timing import StageTimer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vae_refine_sliding_window")

DTYPE = torch.bfloat16

Latents = dict[int, torch.Tensor]  # window index -> latent
Rows = dict[int, dict]  # window index -> its profile.json row

WINDOW_LATENT_NUM = 5  # regular latent frames per window; pixel frames = time_scale * this + 1,
                       # +1 latent frame total for the index-0 keyframe (6 latent frames at t=8)
OVERLAP_LATENT_NUM = 2  # whole latent frames of carryover; pixel overlap = time_scale * this + 1
ENCODE_TILE_FRAMES = 1200  # --encode-mode latent-global: pixel frames per master-encode tile
ENCODE_OVERLAP_FRAMES = 128  # ... and how many of them are real context re-read from the previous tile.
                             # Measured on the 2.5 video VAE, not inherited: a tile's replicate-pad
                             # error is 56% at its own slot 0, 12% two latent frames in, and only
                             # reaches the bf16 noise floor (~1.5e-3) around slot 14-16. So ltx_core's
                             # 16-frame floor leaves ~12% error at a seam; 128 px (16 latent frames)
                             # puts the blended master within ~2x the floor for ~1.1x the pixel
                             # reads, and 192 buys almost nothing more.
MIN_ENCODE_OVERLAP_FRAMES = 16  # ltx_core's own floor (video_vae._validate_overlap) for a conv-VAE encode
FLUSH_EVERY_FRAMES = 24 * 5  # per user: update the on-disk video every 720 finalized frames


@dataclass(frozen=True)
class EncodeTile:
    """One tile of the whole-video master encode for --encode-mode latent-global.

    ``weights`` is the per-latent-frame blend weight of this tile's contribution, and its
    leading entry is exactly 0.0 on every tile after the first -- that slot is the causal
    VAE's fabricated keyframe, and it is discarded rather than written into the master.
    """

    latent_start: int
    latent_end: int
    pixel_start: int
    pixel_stop: int
    weights: torch.Tensor  # (latent_end - latent_start,), in [0, 1]

    @property
    def latent_frames(self) -> int:
        return self.latent_end - self.latent_start


def plan_encode_tiles(
    total_frames: int, tile_frames: int, overlap_frames: int, time_scale: int
) -> list[EncodeTile]:
    """OVERLAPPING, grid-aligned tiles whose blend is ONE continuous master latent.

    This is ``ltx_core``'s own causal encode tiling (``split_temporal_causal`` +
    ``map_temporal_slice``, exactly what ``VideoEncoder.tiled_encode`` uses), applied to
    intervals we walk ourselves so each tile can be streamed from ``decord`` instead of
    sliced out of one in-memory copy of the whole video.

    It replaces an earlier *abutting* chunk walk, which was wrong: every chunk boundary
    re-seeds the causal VAE's replicate pad, so each chunk after the first spent one latent
    frame on a 1-pixel keyframe and the concatenation ran one frame long per boundary. The
    window slicer indexes the master as ``start // time_scale``, so from the second chunk on
    every window read content ``time_scale - 1`` pixels early, compounding per boundary.

    The fix is not to remove the pad (it is inside every ``CausalConv3d``; there is no
    cross-call state to remove it with) but to feed it REAL pixels and throw the result away:
    tile ``k`` starts at pixel ``latent_start * time_scale`` -- a genuine source frame, the
    last one covered by latent frame ``latent_start`` -- and that slot's blend weight is 0,
    with the next ``overlap`` frames cross-faded against the previous tile. Every latent
    frame the master keeps is therefore computed with real preceding context, and master
    frame ``m`` means what a single whole-video encode would mean, which is what makes
    ``start // time_scale`` correct again.

    Raises ``SystemExit`` (not ``AssertionError``) on an illegal tile geometry: these are
    user-supplied CLI values, same convention as ``geometry.check_window_rules``.
    """
    if tile_frames % time_scale != 0 or tile_frames < 2 * time_scale:
        raise SystemExit(
            f"--encode-chunk-frames {tile_frames} must be a multiple of the probed VAE temporal scale "
            f"factor {time_scale} and at least {2 * time_scale}. Note this is NOT the window's "
            f"`F %% {time_scale} == 1` rule: a master-encode tile is addressed on the latent grid, so it "
            f"is sized in whole latent frames. Nearest valid: "
            f"{max(2 * time_scale, time_scale * (tile_frames // time_scale))}."
        )
    if overlap_frames % time_scale != 0:
        raise SystemExit(
            f"--encode-overlap-frames {overlap_frames} must be a multiple of the probed VAE temporal "
            f"scale factor {time_scale}."
        )
    if overlap_frames < MIN_ENCODE_OVERLAP_FRAMES:
        raise SystemExit(
            f"--encode-overlap-frames {overlap_frames} is below ltx_core's own floor of "
            f"{MIN_ENCODE_OVERLAP_FRAMES} for a conv-VAE encode (video_vae._validate_overlap: "
            "'needs enough overlap to discard symmetric-pad edge artifacts')."
        )
    if overlap_frames >= tile_frames:
        raise SystemExit(
            f"--encode-overlap-frames {overlap_frames} must be smaller than --encode-chunk-frames "
            f"{tile_frames}."
        )

    latent_frames = (total_frames - 1) // time_scale + 1
    tile_latent = tile_frames // time_scale
    overlap_latent = overlap_frames // time_scale
    # Same clamp TileSizeConfig.to_splitters applies before handing sizes to the splitter.
    tile_latent = max(2, overlap_latent + 1, tile_latent)

    tiles: list[EncodeTile] = []
    for interval in split_temporal_causal(tile_latent, overlap_latent)(latent_frames).intervals:
        pixels, _ = map_temporal_slice(
            interval.start, interval.end, interval.left_ramp, interval.right_ramp, time_scale
        )
        tiles.append(
            EncodeTile(
                latent_start=interval.start,
                latent_end=interval.end,
                pixel_start=pixels.start,
                pixel_stop=pixels.stop,
                weights=compute_trapezoidal_mask_1d(
                    interval.end - interval.start, interval.left_ramp, interval.right_ramp, True
                ),
            )
        )
    return tiles


@dataclass(frozen=True)
class MasterLatent:
    """The whole video as ONE continuous latent, plus every window's own keyframe.

    ``latent`` is what a single end-to-end VAE encode of the clip would produce: frame 0
    covers pixel 0 alone (the causal keyframe) and frame ``m`` covers pixels
    ``[t*m - t + 1, t*m]``. So master frame ``start // t`` is exactly the frame whose last
    covered pixel is ``start``, and a window starting at ``start`` takes frames
    ``[start // t + 1, ...)`` as its regular latent frames.

    ``keyframes`` is ``(1, C, num_windows, H, W)`` -- window ``i``'s index-0 causal keyframe.
    It CANNOT be sliced out of ``latent``: every master frame after index 0 spans ``t`` pixels,
    so none of them is the single-pixel encode a window's slot 0 has to be. It is encoded
    separately, one real source frame per window. The VAE encoder is causal, so latent frame 0
    is a function of pixel frame 0 alone -- encoding that one frame reproduces exactly what a
    full per-window encode puts in slot 0 (see ``--encode-mode per-window``).
    """

    latent: torch.Tensor
    keyframes: torch.Tensor
    window_starts: list[int]
    keyframe_source: str = "encoded"  # "encoded" | "master-slice"; see window()

    def window(self, index: int, latent_frames: int, time_scale: int) -> torch.Tensor:
        """Window ``index``'s initial latent: its slot-0 keyframe + its regular frames.

        ``keyframe_source="encoded"`` (default) puts this window's own genuine single-pixel
        keyframe in slot 0, making the window structurally identical to what
        ``--encode-mode per-window`` produces -- same token classes, same
        ``VideoLatentTools`` keyframe mask semantics.

        ``keyframe_source="master-slice"`` instead takes master frame ``start // t``, a
        regular ``t``-pixel block, as slot 0. That is what this mode did before window
        keyframes were encoded separately. It is kept only as the A/B lever: the checkpoint
        sets ``use_keyframes_abs_pos_embedding``, so the model applies a learned KEYFRAME
        embedding to slot 0 either way -- under "master-slice" it is applied to a token that
        is not one. Slot 0's pixels never reach the stitched output (the stitcher drops the
        window's first ``overlap_frames``), so this changes the output only through
        conditioning and through the causal decode of the later slots.
        """
        first = self.window_starts[index] // time_scale
        stop = first + latent_frames
        if stop > self.latent.shape[2]:
            raise RuntimeError(
                f"window {index} needs master latent frames [{first + 1}:{stop}) but the master latent "
                f"has only {self.latent.shape[2]}. The master and the window plan disagree -- they were "
                "built from different --max-total-frames or a different --model."
            )
        if self.keyframe_source == "master-slice":
            return self.latent[:, :, first:stop]
        return torch.cat([self.keyframes[:, :, index : index + 1], self.latent[:, :, first + 1 : stop]], dim=2)


def _encode_window_keyframes(
    vr: decord.VideoReader,
    starts: list[int],
    encoder,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Encode each window's first pixel frame on its own -> ``(1, C, num_windows, H, W)``.

    Batched over the batch axis (not the temporal one): every element is an independent
    1-frame causal encode, which is what makes each result a genuine keyframe rather than a
    slice of a longer clip. Costs one pixel frame per window -- for the default 24-frame
    stride that is ~4% on top of the master encode.
    """
    pieces: list[torch.Tensor] = []
    for offset in range(0, len(starts), batch_size):
        group = starts[offset : offset + batch_size]
        frames = torch.cat([refine_core.read_pixel_window(vr, s, s + 1, device, DTYPE)[0] for s in group], dim=0)
        latents = encoder.tiled_encode(frames, None)  # (B, C, 1, H, W)
        pieces.append(rearrange(latents.float().cpu(), "b c 1 h w -> 1 c b h w"))
    return torch.cat(pieces, dim=2)


def build_master_latent(
    vr: decord.VideoReader,
    video_path: Path,
    total_frames: int,
    tile_frames: int,
    overlap_frames: int,
    window_starts: list[int],
    keyframe_source: str,
    model,
    device: torch.device,
    cache_path: Path,
) -> MasterLatent:
    """One-time whole-video VAE encode for --encode-mode latent-global, cached to disk.

    Tiling here bounds VAE-encoder activation memory, and -- unlike the earlier abutting-chunk
    version -- it does NOT change what the master means. Tiles overlap by ``overlap_frames``
    real pixels and the pad-contaminated head of each one is blended away by
    ``plan_encode_tiles``' weights, so the blended result is a single continuous latent on the
    global grid. That is what makes ``MasterLatent.window`` able to index it arithmetically.

    Accumulation is fp32: the blend is a weighted sum, and accumulating it in bf16 would
    quantise the ramp.
    """
    time_scale = model.scale_factors.time
    meta = {
        # Bumped when the on-disk meaning changes. format 1 was an abutting-chunk concat that
        # ran one latent frame long per chunk boundary; its cache key is a subset of this one,
        # so the bump is what stops a stale format-1 file being silently reused here.
        "format": 2,
        "video": str(video_path.resolve()),
        "video_vae": str(model.paths.video_vae()),
        "total_frames": total_frames,
        "tile_frames": tile_frames,
        "overlap_frames": overlap_frames,
        "time_scale": time_scale,
        "window_starts": window_starts,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("meta") == meta:
            logger.info(f"Loaded cached master latent from {cache_path} ({cached['latent'].shape[2]} latent frames)")
            return MasterLatent(cached["latent"], cached["keyframes"], window_starts, keyframe_source)
        logger.info(f"Cached master latent at {cache_path} is stale for this run -- rebuilding.")

    tiles = plan_encode_tiles(total_frames, tile_frames, overlap_frames, time_scale)
    latent_frames = (total_frames - 1) // time_scale + 1
    logger.info(
        f"Building master latent (latent-global): {total_frames} native frames -> {latent_frames} latent "
        f"frames from {len(tiles)} overlapping encode tiles of up to {tile_frames} frames "
        f"({overlap_frames}-frame real-context overlap), plus {len(window_starts)} window keyframes"
    )

    accumulator: torch.Tensor | None = None
    weights = torch.zeros(1, 1, latent_frames, 1, 1, dtype=torch.float32)
    with ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder:
        for index, tile in enumerate(tiles):
            with StageTimer("encode_tile", device) as timer:
                norm, _ = refine_core.read_pixel_window(vr, tile.pixel_start, tile.pixel_stop, device, DTYPE)
                piece = encoder.tiled_encode(norm, None).float().cpu()
            if piece.shape[2] != tile.latent_frames:
                raise AssertionError(
                    f"encode tile {index} produced {piece.shape[2]} latent frames, expected "
                    f"{tile.latent_frames} for pixels [{tile.pixel_start}:{tile.pixel_stop})"
                )
            if accumulator is None:
                accumulator = torch.zeros(
                    piece.shape[0], piece.shape[1], latent_frames, piece.shape[3], piece.shape[4],
                    dtype=torch.float32,
                )
            tile_weights = tile.weights.view(1, 1, -1, 1, 1)
            accumulator[:, :, tile.latent_start : tile.latent_end] += piece * tile_weights
            weights[:, :, tile.latent_start : tile.latent_end] += tile_weights
            logger.info(
                f"  tile {index + 1}/{len(tiles)} pixels [{tile.pixel_start}:{tile.pixel_stop}) -> latent "
                f"[{tile.latent_start}:{tile.latent_end}) in {timer.elapsed_s:.1f}s "
                f"(peak {timer.peak_alloc_gb:.1f}GB)"
            )

        with StageTimer("encode_keyframes", device) as timer:
            keyframes = _encode_window_keyframes(vr, window_starts, encoder, device)
        logger.info(f"  {len(window_starts)} window keyframes in {timer.elapsed_s:.1f}s")

    if float(weights.min()) <= 0.0:
        raise AssertionError(
            "the encode tiling left some master latent frames with zero total blend weight -- "
            "plan_encode_tiles produced a non-covering plan"
        )
    master_latent = (accumulator / weights).to(DTYPE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "latent": master_latent, "keyframes": keyframes.to(DTYPE)}, cache_path)
    logger.info(f"Master latent built: {tuple(master_latent.shape)}, cached to {cache_path}")
    return MasterLatent(master_latent, keyframes.to(DTYPE), window_starts, keyframe_source)


@dataclass
class WindowRefiner:
    """Encode -> K-step refine -> decode, one batch of windows at a time.

    The three phases are separate passes over the batch rather than one pass per
    window, because the ~44GB transformer and the VAE cannot be resident together on
    a 49GB card. Splitting them lets each heavyweight component be built ONCE per
    batch instead of once per window, which is the dominant per-window cost (the
    checkpoint build, not the denoise compute). Windows are still refined one at a
    time -- DiffusionStage hardcodes batch=1 -- so batching changes nothing about
    the result, only how often the transformer reloads.
    """

    model: object
    device: torch.device
    windows: list[tuple[int, int]]
    geometry: refine_core.WindowGeometry
    tools: object
    sigmas: torch.Tensor
    context: torch.Tensor
    seed: int
    vr: decord.VideoReader
    master_latent: MasterLatent | None
    pixel_dir: Path
    latent_dir: Path

    def __post_init__(self) -> None:
        self.stage = DiffusionStage.from_checkpoint(
            self.model.paths.transformer(), DTYPE, self.device,
            model_configurator=LTXVideoOnlyModelConfigurator, scale_factors=self.model.scale_factors,
        )
        self.denoiser = SimpleDenoiser(self.context, None)
        # Window i's refined latent, kept in memory for window i+1's carryover (latents are
        # tiny, ~1-4MB each) and mirrored to latent_dir so a resumed run can recover the
        # carryover from a window an earlier session finished.
        self.latents: dict[int, torch.Tensor] = {}

    # --- per-window caches -------------------------------------------------

    def pixel_path(self, index: int) -> Path:
        return self.pixel_dir / f"win_{index:04d}.pt"

    def latent_path(self, index: int) -> Path:
        return self.latent_dir / f"win_{index:04d}_latent.pt"

    def cached_pixels(self, index: int) -> torch.Tensor | None:
        path = self.pixel_path(index)
        return torch.load(path, map_location="cpu").float() / 255.0 if path.exists() else None

    def _carry_for(self, index: int) -> torch.Tensor | None:
        """The previous window's frozen-overlap conditioning, from memory or the cache."""
        if index == 0:
            return None
        previous = self.latents.get(index - 1)
        if previous is None:
            path = self.latent_path(index - 1)
            if not path.exists():
                raise RuntimeError(
                    f"Window {index - 1}'s refined latent is unavailable (neither in memory nor at {path}) "
                    f"-- can't build window {index}'s frozen-overlap conditioning. Windows must be "
                    "processed in order from a consistent cache."
                )
            previous = torch.load(path, map_location="cpu")
        # refine_core.make_window_state injects this at latent_idx=1, NOT 0 -- see
        # refine_core.CARRYOVER_LATENT_IDX for why the keyframe slot stays fresh.
        return refine_core.carry_from(previous.to(device=self.device, dtype=DTYPE), self.geometry)

    def run(self, todo: list[int]) -> tuple[Latents, list[dict]]:
        """Refine `todo` (a batch of window indices) into decoded pixels + profile rows.

        Each phase's heavyweight component is built once for the whole batch and freed
        before the next phase starts -- they cannot be resident together on a 49GB card.
        """
        rows: Rows = {
            index: {
                "window_index": index,
                "start": self.windows[index][0],
                "end": self.windows[index][1],
                "batch_size": len(todo),
            }
            for index in todo
        }
        encoded: Latents = {}
        refined: Latents = {}
        decoded: Latents = {}

        # --- Phase A: encode ---------------------------------------------------
        logger.info(f"Encoding windows {todo[0]}-{todo[-1]}")
        if self.master_latent is not None:
            # No pixel read, no VAE-encoder build/forward at all: every window's initial latent is
            # assembled from the once-built master -- its own precomputed keyframe in slot 0, then
            # its regular frames sliced straight out of the master. Window i's pixel start is
            # always a multiple of time_scale (stride_frames = time_scale * chunk_latent_frames),
            # which is what makes that slice pure arithmetic. See MasterLatent.window.
            time_scale = self.model.scale_factors.time
            with StageTimer("slice", self.device) as timer:
                for index in todo:
                    encoded[index] = self.master_latent.window(
                        index, self.geometry.latent_frames, time_scale
                    ).to(device=self.device, dtype=DTYPE)
            for index in todo:
                rows[index] |= {"encoder_build_s": 0.0, "encode_s": timer.elapsed_s / len(todo)}
        else:
            with ExitStack() as stack:
                with StageTimer("encoder_build", self.device) as build:
                    encoder = stack.enter_context(
                        ltx_adapter.video_encoder(self.model.paths.video_vae(), DTYPE, self.device)
                    )
                for index in todo:
                    start, end = self.windows[index]
                    with StageTimer("encode", self.device) as timer:
                        norm, _ = refine_core.read_pixel_window(self.vr, start, end, self.device, DTYPE)
                        encoded[index] = encoder.tiled_encode(norm, None)
                    rows[index] |= {
                        "encoder_build_s": build.elapsed_s,
                        "encoder_build_peak_alloc_gb": build.peak_alloc_gb,
                        "encode_s": timer.elapsed_s,
                        "encode_peak_alloc_gb": timer.peak_alloc_gb,
                    }

        # --- Phase B: refine ---------------------------------------------------
        logger.info("Denoising")
        with ExitStack() as stack:
            with StageTimer("transformer_build", self.device) as build:
                transformer = stack.enter_context(ltx_adapter.transformer_ctx(self.stage, video_tools=self.tools))
            for index in todo:
                carry = self._carry_for(index)
                with StageTimer("refine", self.device) as timer:
                    latent = refine_core.refine_window(
                        transformer, self.denoiser, encoded[index], carry,
                        self.sigmas, self.tools, self.seed, self.device, DTYPE,
                    )
                refined[index] = latent
                self.latents[index] = latent.to(torch.bfloat16).cpu()
                torch.save(self.latents[index], self.latent_path(index))
                rows[index] |= {
                    "transformer_build_s": build.elapsed_s,
                    "transformer_build_peak_alloc_gb": build.peak_alloc_gb,
                    "carryover_latent_frames": 0 if carry is None else carry.shape[2],
                    "refine_s": timer.elapsed_s,
                    "refine_peak_alloc_gb": timer.peak_alloc_gb,
                }

        # --- Phase C: decode ---------------------------------------------------
        logger.info("Decoding")
        with ExitStack() as stack:
            with StageTimer("decoder_build", self.device) as build:
                decoder = stack.enter_context(
                    ltx_adapter.video_decoder(self.model.paths.video_vae(), DTYPE, self.device)
                )
            for index in todo:
                with StageTimer("decode", self.device) as timer:
                    frames = torch.cat(list(decoder.decode_video(refined[index], None, None)), dim=0)
                decoded[index] = frames.cpu().float()
                torch.save((decoded[index].clamp(0, 1) * 255.0).to(torch.uint8), self.pixel_path(index))
                rows[index] |= {
                    "decoder_build_s": build.elapsed_s,
                    "decoder_build_peak_alloc_gb": build.peak_alloc_gb,
                    "decode_s": timer.elapsed_s,
                    "decode_peak_alloc_gb": timer.peak_alloc_gb,
                }

        for index in todo:
            row = rows[index]
            logger.info(
                f"window {index + 1}/{len(self.windows)} batch={len(todo)} frames {self.windows[index]} "
                f"encode={row['encode_s']:.1f}s refine={row['refine_s']:.1f}s decode={row['decode_s']:.1f}s"
            )

        return decoded, [rows[index] for index in todo]


class Stitcher:
    """Copies each window's non-overlapping frames and flushes them to segment .mp4s.

    The first window contributes its full frame range; every later window contributes
    only its non-overlapping tail (``decoded[overlap_prev:]``) -- the overlap itself is
    taken entirely from the earlier window's decode, not blended. Never holds more than
    one flush interval of pixels: frames are appended to the current segment, written
    out on ``flush()``, and stream-copy-concatenated into one always-playable
    ``decode_full.mp4``.
    """

    def __init__(self, segments_dir: Path, full_path: Path, fps: float) -> None:
        self.segments_dir = segments_dir
        self.full_path = full_path
        self.fps = fps
        self.buffer: list[torch.Tensor] = []
        self.buffered = 0
        self.flushed_to = 0  # first native-frame index not yet flushed

    def add(self, decoded: torch.Tensor, overlap_prev: int) -> None:
        self._append(decoded[overlap_prev:])

    def _append(self, frames: torch.Tensor) -> None:
        self.buffer.append(frames)
        self.buffered += frames.shape[0]

    def flush(self) -> None:
        if not self.buffer:
            return
        chunk = torch.cat(self.buffer, dim=0)
        end = self.flushed_to + chunk.shape[0]
        seg_path = self.segments_dir / f"seg_{self.flushed_to:06d}_{end:06d}.mp4"
        save_video(rearrange(chunk, "f h w c -> f c h w"), seg_path, fps=self.fps, video_format="FCHW")
        self._rebuild_full()
        logger.info(f"Flushed frames [{self.flushed_to}:{end}) -> {seg_path.name}, rebuilt {self.full_path.name}")
        self.flushed_to = end
        self.buffer = []
        self.buffered = 0

    def _rebuild_full(self) -> None:
        segments = sorted(self.segments_dir.glob("seg_*.mp4"))
        if not segments:
            return
        list_path = self.segments_dir / "_concat_list.txt"
        list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
        tmp_out = self.full_path.with_suffix(".tmp.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(tmp_out)],
            check=True,
            capture_output=True,
        )
        tmp_out.replace(self.full_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k-step", default=refine_task.K_STEP)
    ap.add_argument("--gpu-id", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=refine_task.REFINE_PROMPT)
    ap.add_argument(
        "--model", default="2.5", choices=model_registry.SUPPORTED_MODELS,
        help="Generation to refine with (see scripts/prune/core/model_registry.py). Per-component "
        "flags below always override this generation's default path for that component.",
    )
    ap.add_argument("--sampler", default="euler", choices=model_registry.SAMPLER_CHOICES)
    ap.add_argument(
        "--transformer-path", type=Path, default=None,
        help="Override this --model's transformer checkpoint.",
    )
    ap.add_argument(
        "--text-encoder-path", type=Path, default=None,
        help="Override this --model's text encoder (gemma root dir for 2.3, gemma4 file for 2.5).",
    )
    ap.add_argument("--video-vae-path", type=Path, default=None, help="Override this --model's video VAE checkpoint.")
    ap.add_argument(
        "--window-latent-num", type=int, default=WINDOW_LATENT_NUM,
        help="Regular (non-keyframe) latent frames per window. Pixel-frame window length "
        "(time_scale * window_latent_num + 1) is derived from the model's probed temporal scale "
        "factor, not passed directly -- the window's total latent frame count is this plus 1 for "
        "the index-0 keyframe (e.g. 5 -> 6 latent frames, 41 pixel frames at t=8).",
    )
    ap.add_argument(
        "--overlap-latent-num", type=int, default=OVERLAP_LATENT_NUM,
        help="Whole latent frames of carryover between consecutive windows. The pixel-frame overlap "
        "this implies (time_scale * overlap_latent_num + 1) is derived from the model's probed "
        "temporal scale factor, not passed directly.",
    )
    ap.add_argument(
        "--encode-mode", default="per-window", choices=["per-window", "latent-global"],
        help="'per-window' (default, unchanged behavior): every diffusion window independently "
        "VAE-encodes its own overlapping pixel range -- bit-comparable with existing cached "
        "results and what scripts/prune/checks/method_parity.py's gate checks against. "
        "'latent-global': VAE-encode the whole video exactly once into one continuous latent "
        "(see --encode-chunk-frames), then every diffusion window is a slice of it -- no "
        "per-window pixel read or VAE-encoder work, but only the video's true first pixel frame "
        "is a real single-pixel keyframe (see module docstring for the trade-off).",
    )
    ap.add_argument(
        "--encode-chunk-frames", type=int, default=ENCODE_TILE_FRAMES,
        help="--encode-mode latent-global only: pixel-frame tile size for the one-time master "
        "encode. Must be a multiple of the probed VAE temporal scale factor -- NOT the window's "
        "F %% t == 1 rule, because a master-encode tile is addressed on the latent grid. Purely "
        "bounds VAE-encoder activation memory: tiles overlap by --encode-overlap-frames and are "
        "blended, so unlike --window-latent-num this has no effect on the master latent's "
        "contents. Pick it as large as the GPU allows.",
    )
    ap.add_argument(
        "--encode-overlap-frames", type=int, default=ENCODE_OVERLAP_FRAMES,
        help="--encode-mode latent-global only: pixel frames of REAL preceding context each master "
        "encode tile re-reads from its predecessor. Every tile's leading latent frame is the causal "
        "VAE's replicate-pad keyframe and is discarded (blend weight 0); the rest of the overlap "
        "cross-fades the residual pad contamination away. Must be a multiple of the temporal scale "
        f"factor and at least {MIN_ENCODE_OVERLAP_FRAMES} (ltx_core's own floor) -- but that floor is far "
        "too small for this VAE: measured contamination is 56%% at a tile's own slot 0, 12%% two latent "
        "frames in, and only reaches the bf16 noise floor around 14-16 latent frames, hence the "
        f"{ENCODE_OVERLAP_FRAMES}-frame default.",
    )
    ap.add_argument(
        "--keyframe-source", default="encoded", choices=["encoded", "master-slice"],
        help="--encode-mode latent-global only: what goes in each window's latent slot 0. "
        "'encoded' (default) is the window's own genuine single-pixel causal keyframe, encoded "
        "separately -- structurally what --encode-mode per-window produces. 'master-slice' is the "
        "master's regular t-pixel frame at that position, which the model then receives the learned "
        "keyframe embedding on. Kept as the A/B lever; see MasterLatent.window.",
    )
    ap.add_argument("--flush-every-frames", type=int, default=FLUSH_EVERY_FRAMES)
    ap.add_argument("--max-windows", type=int, default=None, help="Process at most this many windows (testing).")
    ap.add_argument(
        "--max-total-frames", type=int, default=None,
        help="Only plan windows within the first N native frames of the source video (testing on a slice).",
    )
    ap.add_argument(
        "--batch-windows", type=int, default=2,
        help="Windows per 'resident transformer' batch -- see WindowRefiner. Larger values amortize "
        "the checkpoint build over more windows but bound resume granularity: an interruption "
        "mid-batch redoes that batch's un-cached windows. Set to 1 for a rebuild per window.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    model = preflight.check(
        args.model,
        sampler=args.sampler,
        gpu_id=args.gpu_id,
        transformer_path=args.transformer_path,
        text_encoder_path=args.text_encoder_path,
        video_vae_path=args.video_vae_path,
    )
    if model.stepper_kind == "ancestral":
        # EulerAncestralDiffusionStep needs a per-step noise draw (eta=1.0 renoises after every
        # step); refine_core's step loop doesn't supply one. Plan §4 decision 1 defaults the
        # refiner to Euler on both generations and defers the ancestral A/B to Phase 1 (needs the
        # noise-injecting loop, mirroring ltx_pipelines.utils.samplers._ancestral_euler_denoising_loop).
        raise SystemExit(
            f"--model {args.model} --sampler {args.sampler} resolved to the ancestral stepper, which "
            "isn't wired into this script's step loop yet (Phase 1 item per plan §4 decision 1). "
            "Pass --sampler euler explicitly."
        )
    # The window grid is checked against the VAE's PROBED temporal scale factor rather than a
    # literal 8 (plan §4 decision 2), which is why this runs after the model resolve. The rule
    # itself is unchanged: each window's own latent frame 0 is a single-pixel causal keyframe
    # rather than a full temporal block, so the carried-over overlap needs the same F%t==1 grid
    # for it to land on whole *regular* latent frames of the next window.
    window_frames = model.scale_factors.time * args.window_latent_num + 1
    overlap_frames = model.scale_factors.time * args.overlap_latent_num + 1
    geometry.check_window_rules(window_frames, overlap_frames, model.scale_factors)
    window_geometry = refine_core.WindowGeometry(
        window_frames=window_frames, overlap_frames=overlap_frames, scale_factors=model.scale_factors
    )
    # NB: master-encode tiles deliberately do NOT go through geometry.check_window_rules --
    # a window is a pixel span that starts with a keyframe (F % t == 1), a tile is a latent-grid
    # interval (F % t == 0). plan_encode_tiles validates them on their own rules.
    logger.info(
        f"model={model.key} (version {model.version}) sampler={model.stepper_kind} "
        f"scale_factors={tuple(model.scale_factors)} (from {model.scale_factors_source})"
    )

    device = torch.device(f"cuda:{args.gpu_id}")
    out_dir = args.output_dir
    segments_dir = out_dir / "segments"
    pixel_dir = out_dir / "window_cache"
    latent_dir = out_dir / "latent_cache"
    for path in (out_dir, segments_dir, pixel_dir, latent_dir):
        path.mkdir(parents=True, exist_ok=True)

    vr = decord.VideoReader(str(args.video))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr) if args.max_total_frames is None else min(len(vr), args.max_total_frames)
    windows = window_geometry.plan(total_frames)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    logger.info(
        f"{args.video.name}: {total_frames} native frames -> {len(windows)} windows of {window_frames} "
        f"({args.window_latent_num} + 1 keyframe latent frames; overlap {overlap_frames} = "
        f"{args.overlap_latent_num} latent frames, stride {window_geometry.stride_frames})"
    )
    (out_dir / "window_plan.json").write_text(
        json.dumps(
            {"total_frames": total_frames, "fps": fps, "geometry": window_geometry.as_dict(), "windows": windows},
            indent=2,
        )
    )
    if args.dry_run:
        print(f"Dry run: {len(windows)} windows planned, written to {out_dir / 'window_plan.json'}")
        return 0

    profile_path = out_dir / "profile.json"
    profile: list[dict] = json.loads(profile_path.read_text()) if profile_path.exists() else []
    done_indices = {row["window_index"] for row in profile}

    sigmas = torch.tensor(
        refine_task.schedule_for(model.sigmas, args.k_step), dtype=torch.float32, device=device
    )
    # One prompt for every window, encoded once and cached on disk by (model, prompt hash),
    # so a resumed or repeated run never rebuilds the text encoder.
    context = prompt_cache.get_or_build(model, args.prompt, DTYPE, device)

    master_latent = None
    if args.encode_mode == "latent-global":
        master_latent = build_master_latent(
            vr, args.video, total_frames, args.encode_chunk_frames, args.encode_overlap_frames,
            [start for start, _ in windows], args.keyframe_source, model, device,
            latent_dir / "master_latent.pt",
        )

    # Geometry is identical for every window (fixed window_frames, one source video), so the
    # tools DiffusionStage would otherwise rebuild per call are built once. fps is the clip's
    # own, never a constant: VideoLatentTools divides the temporal position axis by it, so it
    # is part of RoPE. See refine_core.build_tools.
    probe, _ = refine_core.read_pixel_window(vr, *windows[0], device, DTYPE)
    tools = refine_core.tools_for_window(window_geometry, int(probe.shape[-2]), int(probe.shape[-1]), fps)
    del probe

    refiner = WindowRefiner(
        model=model, device=device, windows=windows, geometry=window_geometry, tools=tools,
        sigmas=sigmas, context=context, seed=args.seed, vr=vr, master_latent=master_latent,
        pixel_dir=pixel_dir, latent_dir=latent_dir,
    )
    stitcher = Stitcher(segments_dir, out_dir / "decode_full.mp4", fps)

    batch_size = max(1, args.batch_windows)
    for batch_start in range(0, len(windows), batch_size):
        batch = list(range(batch_start, min(batch_start + batch_size, len(windows))))
        decoded: Latents = {}
        todo: list[int] = []
        for index in batch:
            cached = refiner.cached_pixels(index) if index in done_indices else None
            if cached is not None:
                decoded[index] = cached
            else:
                todo.append(index)
        if todo:
            fresh, rows = refiner.run(todo)
            decoded.update(fresh)
            profile.extend(rows)
            profile_path.write_text(json.dumps(profile, indent=2))

        for index in batch:
            # Fixed-stride tiling (WindowGeometry.plan) makes every pairwise overlap exactly
            # window_geometry.overlap_frames -- no need to re-derive it from window boundaries.
            overlap_prev = window_geometry.overlap_frames if index > 0 else 0
            stitcher.add(decoded[index], overlap_prev)
            if stitcher.buffered >= args.flush_every_frames or index == len(windows) - 1:
                stitcher.flush()

    stitcher.flush()
    logger.info(f"Done. Full stitched video at {stitcher.full_path} ({stitcher.flushed_to} frames finalized).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
