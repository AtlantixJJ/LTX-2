# LTX-2 Codebase Annotation

Technical reference for AI agents. For workflow, commands, and rules, see [CLAUDE.md](../CLAUDE.md).

LTX-2 is a `uv` monorepo with four packages (`packages/ltx-core`, `packages/ltx-pipelines`,
`packages/ltx-trainer`, `packages/ltx-kernels`) plus a top-level `scripts/` directory of
standalone research probes. This is a single hub with no spokes: every folder below is
tightly coupled through one shared `LTXModel`/`Modality`/`LatentState` contract that crosses
package boundaries constantly (inference pipelines and the trainer both build the transformer
the same way and call it with the same `Modality` objects), so splitting into spokes would
duplicate that contract rather than isolate it. `packages/ltx-kernels` is the closest thing to
a self-contained unit (a narrow Python API in front of CUDA/Triton kernels), but it is still
directly wired into `ltx-core/quantization` module ops, so it stays in the hub too.

## Architecture

### Data Flow Spine

**Inference** (`packages/ltx-pipelines/src/ltx_pipelines/*.py`, e.g. distilled.py,
ti2vid_two_stages.py):

```
raw video/audio/text prompt
  -> PromptEncoder: GemmaTextEncoder.encode() -> EmbeddingsProcessor
       -> caption embeddings  video ctx [B, L, cross_attention_dim=4096]
                               audio ctx [B, L, audio_cross_attention_dim=2048]
     (packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py:31,
      packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py:96)
  -> ImageConditioner/AudioConditioner: VideoEncoder/AudioEncoder (VAE, tiled)
       -> normalized latent  video [B, 128, F', H', W']  (F'=(F-1)/8+1, H'=H/32, W'=W/32)
                              audio [B, 8, T', 16]        (mel_bins=16)
     (packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py:256,328;
      packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py:189,248)
  -> ConditioningItem.apply_to(LatentState, LatentTools) for each conditioning
       (VideoConditionByLatentIndex/ByKeyframeIndex/ByMask/ByReferenceLatent,
        AudioConditionByReferenceLatent, TemporalRegionMask) -- appends or overwrites
       tokens in LatentState{latent, clean_latent, denoise_mask, positions, attention_mask}
     (packages/ltx-core/src/ltx_core/conditioning/item.py:10, .../types/*.py)
  -> VideoLatentPatchifier(patch_size=1).patchify / AudioPatchifier(patch_size=1).patchify
       -> flattened token sequence [B, N, C]; N = frames*height*width (video)
     (packages/ltx-core/src/ltx_core/components/patchifiers.py:27,287)
  -> GaussianNoiser(generator): lerp(latent, noise, noise_scale) blended by denoise_mask
     (packages/ltx-core/src/ltx_core/components/noisers.py:32)
  -> Modality(latent, sigma, timesteps=denoise_mask*sigma, positions, context, context_mask,
       enabled, attention_mask)  -- one per stream, video and audio
     (packages/ltx-core/src/ltx_core/model/transformer/modality.py:10;
      packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:263)
  -> LTXModel.forward(video: Modality|None, audio: Modality|None, perturbations)
       -> TransformerArgsPreprocessor.prepare -> TransformerArgs per stream
       -> 48x BasicAVTransformerBlock (video self-attn, audio self-attn, text cross-attn,
          audio<->video cross-attn, feed-forward, AdaLN modulation via scale_shift_table)
       -> _process_output projects back to channel space
     (packages/ltx-core/src/ltx_core/model/transformer/model.py:423,403;
      packages/ltx-core/src/ltx_core/model/transformer/transformer.py:253)
  -> Denoiser (SimpleDenoiser | GuidedDenoiser | FactoryGuidedDenoiser) runs one batched
       transformer call across cond/uncond/perturbed/isolated passes, combines via Guider
       (CFGGuider | STGGuider | LtxAPGGuider | MultiModalGuider) -> DenoisedLatentResult
     (packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py:184;
      packages/ltx-core/src/ltx_core/components/guiders.py:11-337)
  -> DiffusionStepProtocol.step (Euler | Res2s | EulerCfgPp) advances LatentState across the
       sigma schedule (euler_denoising_loop / res2s_audio_video_denoising_loop)
     (packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py:35,308)
  -> unpatchify -> VideoDecoder.tiled_decode / AudioDecoder -> Vocoder
       -> pixel video [B, 3, F, H, W] in [-1, 1]  |  waveform Audio(waveform, sampling_rate)
     (packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py:811,904;
      packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py:496)
  -> media_io.encode_video / encode_audio -> .mp4 / .wav
     (packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py:333,422)
```

**Training** (`packages/ltx-trainer/src/ltx_trainer/trainer.py`, `training_strategies/*.py`):

```
PrecomputedDataset  (.precomputed/ latents + text/embeddings written once by
  packages/ltx-trainer/scripts/process_dataset.py -- text encoder, video VAE, audio VAE,
  and mask phases all run offline; the trainer itself never loads Gemma/VAE weights except
  inside validation)
  (packages/ltx-trainer/src/ltx_trainer/datasets.py:88; packages/ltx-trainer/scripts/process_dataset.py:54)
  -> raw batch dict {video/audio latents, condition embeddings, masks}
  -> TrainingStrategy.prepare_training_inputs(batch, timestep_sampler)
       [FlexibleStrategy (recommended) | TextToVideoStrategy | VideoToVideoStrategy
        (both deprecated, warned in packages/ltx-trainer/src/ltx_trainer/training_strategies/__init__.py:59)]
       -- patchify latents, TimestepSampler.sample() sigma, apply noise:
          (1-sigma)*clean + sigma*noise, build conditioning masks
          (first_frame/prefix/suffix/spatial_crop/mask/reference), compute velocity
          targets = noise - clean, build per-token timesteps and positions
       -> ModelInputs(video, audio, targets, loss_masks)
     (packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py:84,104;
      packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py:260,278; packages/ltx-trainer/src/ltx_trainer/training_strategies/text_to_video.py:79,131,138,256)
  -> LTXModel.forward(video=inputs.video, audio=inputs.audio, perturbations=None)
       -> (pred_video, pred_audio) velocity predictions -- same call as inference, no
          guidance/perturbation on the training path
  -> TrainingStrategy.compute_loss(pred, inputs) -- masked MSE, mean over seq/channels,
       normalized by mask density, per-sample [B]
     (packages/ltx-trainer/src/ltx_trainer/training_strategies/text_to_video.py:267)
  -> accelerator.backward(loss.mean()); clip_grad_norm_; optimizer.step(); scheduler.step()
     (packages/ltx-trainer/src/ltx_trainer/trainer.py:350 _training_step)
  -> every validation.interval steps: ValidationRunner (self-contained: caches prompt
       embeddings and conditioning media once, loads VAE decoder/vocoder on CPU, runs the
       same Euler denoising loop as inference, decodes, saves, optionally logs to W&B)
     (packages/ltx-trainer/src/ltx_trainer/validation_runner.py:156,790)
  -> every checkpoints.interval steps: _save_checkpoint (LoRA state dict or full model) +
       _save_training_state (RNG/optimizer/scheduler/ConfigFingerprint, for resume)
     (packages/ltx-trainer/src/ltx_trainer/trainer.py:925,995)
```

### Model Class Hierarchy

```
torch.nn.Module
├── LTXModel                         (packages/ltx-core/src/ltx_core/model/transformer/model.py:41)   -- the transformer
│     Owns: video/audio input & output projections, caption cross-attn projections,
│           48x BasicAVTransformerBlock, AdaLayerNormSingle
│     Extension points: forward(video, audio, perturbations); set_gradient_checkpointing
│     Built by: LTXModelConfigurator | LTXVideoOnlyModelConfigurator |
│               LTXAudioOnlyModelConfigurator (from_config classmethods select which
│               streams exist, not subclasses of LTXModel)
├── X0Model / LegacyX0Model          (packages/ltx-core/src/ltx_core/model/transformer/model.py:505,472)
│     Wraps a built LTXModel to convert its timestep-conditioned denoising into the
│     explicit-sigma API pipelines/denoisers call (`transformer(video=..., audio=...)`)
├── BasicAVTransformerBlock          (packages/ltx-core/src/ltx_core/model/transformer/transformer.py:86)
│     Owns: video/audio self-attn, text cross-attn, audio<->video cross-attn, FeedForward,
│           per-block scale_shift_table(s) for AdaLN
│     forward(...) -> (video_x, audio_x); the repeated unit, 48 instances in LTXModel
├── BlockStreamingWrapper            (packages/ltx-core/src/ltx_core/block_streaming/wrapper.py:15)
│     Wraps a model exposing `blocks_attr`; loads block weights from disk/pinned CPU on
│     demand via pre/post forward hooks so models larger than GPU memory still run
├── BatchSplitAdapter                (packages/ltx-core/src/ltx_core/batch_split.py:44)
│     Wraps a model to process an oversized batch in <=max_batch_size chunks sequentially
├── VideoEncoder / VideoDecoder      (packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py:144,557)
│     Video VAE; 128 latent channels, patch_size_hw=4 (space-to-depth), causal_decoder
│     option, spatial+temporal tiled encode/decode for large videos
├── AudioEncoder / AudioDecoder      (packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py:59,276)
│     Audio VAE; mel-spectrogram <-> 8-channel latent
├── Vocoder / VocoderWithBWE         (packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py:292)
│     BigVGAN-v2-style mel -> waveform, alias-free (Snake/SnakeBeta) activations
├── GemmaTextEncoder                 (packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py:12)
│     Wraps Gemma3ForConditionalGeneration; also does T2V/I2V prompt enhancement
├── EmbeddingsProcessor              (packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py:50)
│     Projects Gemma hidden states -> video/audio caption embeddings via
│     Embeddings1DConnector (FeatureExtractorV1 or V2, version-detected from config)
└── LatentUpsampler                  (packages/ltx-core/src/ltx_core/model/upsampler/model.py:10)
      2x spatial latent upsampler (SpatialRationalResampler for non-2x rational scales)

Not nn.Module -- orchestrators, one per generation mode (packages/ltx-pipelines/src/ltx_pipelines/):
  DistilledPipeline, TI2VidOneStagePipeline, TI2VidTwoStagesPipeline,
  TI2VidTwoStagesHQPipeline, A2VidPipelineTwoStage, KeyframeInterpolationPipeline,
  LipDubPipeline, RetakePipeline, T2AOneStagePipeline
  Each composes reusable blocks from utils/blocks.py: DiffusionStage, PromptEncoder,
  ImageConditioner, AudioConditioner, VideoUpsampler, VideoDecoder, AudioDecoder --
  every block builds its model lazily on __call__ and frees it on exit
  (packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:2, "Blocks build a model on
  each __call__, use it, then free GPU memory").

Trainer (packages/ltx-trainer/src/ltx_trainer/):
  LtxvTrainer                        (packages/ltx-trainer/src/ltx_trainer/trainer.py:89) -- owns the full lifecycle: model
    loading, LoRA setup, dataloader, optimizer/scheduler, accelerate, train loop,
    validation, checkpointing, W&B/Hub push
  TrainingStrategy (ABC)             (training_strategies/base_strategy.py)
    ├── FlexibleStrategy             (packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py:241)  -- recommended;
    │     config-driven conditioning (first_frame/prefix/suffix/spatial_crop/mask/
    │     reference/video_to_audio/audio_to_video), any subset of video+audio generated
    ├── TextToVideoStrategy          (packages/ltx-trainer/src/ltx_trainer/training_strategies/text_to_video.py:61)  -- deprecated
    └── VideoToVideoStrategy         (packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py:51)  -- deprecated,
          IC-LoRA reference-video conditioning
```

### Multi-GPU Variant Architecture (`ltx_core/multigpu`, `ltx_pipelines/multigpu`)

Every single-GPU building block has a multi-GPU counterpart that wraps it rather than
replacing it:

- **Sequence parallelism**: `SequenceParallelBuilder` wraps a `SingleGPUModelBuilder`,
  attaches `All2AllAttention`/`MaskedAll2AllAttention` module ops, produces a
  `SequenceParallelModelWrapper` that shards video tokens across ranks and redistributes
  Q/K/V via IPC-based CUDA `all2all` kernels (`packages/ltx-kernels/csrc/all2all/`).
  (packages/ltx-core/src/ltx_core/multigpu/transformer/sequence_parallel.py:262;
   packages/ltx-pipelines/src/ltx_pipelines/multigpu/sp_builder.py:25)
- **Tiled data parallelism**: `TiledDataParallelBuilder` wraps the same builder pattern to
  produce a `TiledDataParallelModelWrapper` that splits the *latent* into spatial tiles,
  one tile group per rank, then all-reduces.
  (packages/ltx-core/src/ltx_core/multigpu/transformer/tiled_data_parallel.py:30)
- **Distributed VAE decode**: `DistributedDecoderBuilder` -> `DistributedVideoDecoder`
  partitions decode tiles across ranks and gathers on a driver rank.
  (packages/ltx-core/src/ltx_core/multigpu/vae/distributed_decoder.py:152)
- **Gemma parallelization**: `AccelerateGemmaBuilder`/`BatchParallelGemmaBuilder` either run
  Gemma via `device_map="auto"` on one source rank and broadcast, or replicate Gemma on
  every rank and batch-split the prompt list.
  (packages/ltx-pipelines/src/ltx_pipelines/multigpu/gemma_builders.py:34;
   packages/ltx-core/src/ltx_core/multigpu/gemma/batch_parallel_wrapper.py:26)
- **Orchestration**: `MGPUController` spawns one worker process per GPU (`torch.elastic`),
  rank 0 relays each job over NCCL broadcast, all ranks run the runner's `__call__`
  generator in SPMD lockstep, results stream back through a queue.
  (packages/ltx-pipelines/src/ltx_pipelines/multigpu/controller.py:144,339)
- **LoRA hot-swap under sharding**: `TransformerWeightTracker` keeps a `ShardedSD` backup of
  clean weights (sharded across ranks by a stable hash of the key) so switching LoRAs
  mid-session resets-then-refuses without re-reading the checkpoint from disk.
  (packages/ltx-pipelines/src/ltx_pipelines/multigpu/weight_tracker.py:48)

## Data Contracts

### Shapes & Naming Conventions

```
# Named dims
B = batch    F = video frames (pixel space)    F' = video frames (latent space)
H, W = pixel height/width    H', W' = latent height/width    N = token count
T' = audio latent frames    L = text sequence length    C = channel/embed dim

# Video VAE latent space (packages/ltx-core/src/ltx_core/types.py:19,39)
128 latent channels, scale factors (time=8, height=32, width=32):
  F' = (F - 1) // 8 + 1        (frames % 8 == 1 is the pixel-space constraint)
  H' = H // 32, W' = W // 32   (H, W must be multiples of 32)
Encoder path: patchify(patch_size_hw=4, space-to-depth) -> conv_in -> down_blocks
  -> conv_norm_out (PixelNorm) -> conv_out -> PerChannelStatistics.normalize()
  (packages/ltx-core/src/ltx_core/model/video_vae/ops.py:6,80)
`0.0` in normalized latent space is the per-channel dataset mean, not black/empty pixels.

# Audio VAE latent space (packages/ltx-core/src/ltx_core/types.py:100,135)
8 latent channels, mel_bins=16, sample_rate=16000, hop_length=160,
audio_latent_downsample_factor=4 -> latents_per_second = 16000/160/4 = 25/s.

# Transformer token space -- distinct from VAE latent channels
Both video and audio streams are projected to a shared per-stream embed width inside the
transformer via input/output linear layers set up in LTXModel._init_video/_init_audio
(default in_channels/out_channels=128 video, audio_in_channels/out_channels=128 audio,
cross_attention_dim=4096 video-text, audio_cross_attention_dim=2048 audio-text) --
the VAE's native 8-channel audio latent is NOT the transformer's audio token width.
(packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py:49,50,61,62)

# Two distinct "patchify" operations -- do not conflate them
1. VAE-internal patchify/unpatchify (packages/ltx-core/src/ltx_core/model/video_vae/ops.py:6) -- space-to-depth on PIXELS,
   patch_size_hw=4, happens inside VideoEncoder/VideoDecoder.
2. Transformer-level VideoLatentPatchifier/AudioPatchifier (packages/ltx-core/src/ltx_core/components/patchifiers.py:11)
   -- flattens the (already-VAE-encoded) LATENT grid into a token sequence for attention,
   patch_size=1 in every call site actually used in this repo
   (packages/ltx-trainer/src/ltx_trainer/validation_runner.py:143-144; base_strategy.py uses patch_size=1 too).

# LatentState field naming (packages/ltx-core/src/ltx_core/types.py:186)
latent          -- current (possibly noisy) tokens, what the transformer denoises
clean_latent    -- ground-truth / conditioning tokens, blended in via denoise_mask
denoise_mask    -- 1.0 = fully noised/generated, 0.0 = fully clean/frozen, in between =
                   partial conditioning strength
positions       -- RoPE position grid per token
attention_mask  -- optional; None means full attention

# Modality dataclass (packages/ltx-core/src/ltx_core/model/transformer/modality.py:10)
latent, sigma, timesteps (= denoise_mask * sigma, per-token not per-batch), positions,
context (text embeddings), enabled: bool = True, context_mask, attention_mask
```

### Forward Pass Output Format

```python
# LTXModel.forward(video: Modality | None, audio: Modality | None,
#                   perturbations: BatchedPerturbationConfig | None) -> tuple[Tensor|None, Tensor|None]
# (packages/ltx-core/src/ltx_core/model/transformer/model.py:423)
(
    denoised_video,   # [B, N_video, video_out_channels=128] -- velocity prediction over
                      #   video tokens in transformer token space; None if video Modality
                      #   absent/disabled (LTXVideoOnlyModelConfigurator omits audio, and
                      #   vice versa for LTXAudioOnlyModelConfigurator)
    denoised_audio,   # [B, N_audio, audio_out_channels=128] -- velocity prediction over
                      #   audio tokens; None under the same absent/disabled rule
)
# Consumers unpatchify this back to VAE latent shape, then blend with the clean latent
# using denoise_mask before VAE decode:
#   result = denoised * denoise_mask + clean.float() * (1 - denoise_mask)
#   (packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:250;
#    packages/ltx-trainer/src/ltx_trainer/validation_runner.py:1066)

# Denoiser-level wrapper, one per modality (packages/ltx-pipelines/src/ltx_pipelines/utils/types.py:45)
DenoisedLatentResult = {
    denoised: Tensor,       # the guidance-combined prediction actually used to step
    # + raw per-pass outputs (cond/uncond/perturbed) kept for guider bookkeeping
}

# Training-side equivalent contract (packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py:53)
ModelInputs = {
    video: Modality | None, audio: Modality | None,   # transformer call inputs
    targets: {video: Tensor, audio: Tensor},           # velocity targets = noise - clean
    loss_masks: {video: Tensor, audio: Tensor},         # which tokens count toward loss
}

# Trainer step output (packages/ltx-trainer/src/ltx_trainer/trainer.py:82 TrainingStepOutput)
{ loss: Tensor[B],   # per-sample, .detach()'d before logging (packages/ltx-trainer/src/ltx_trainer/trainer.py:261)
  sigma: Tensor[B] }  # per-sample sigma actually sampled, for SigmaBucketTracker
```

### Conditioning Token Convention

Every `ConditioningItem.apply_to(latent_state, latent_tools)` implementation follows the
same shape: it patchifies its own media, computes RoPE positions relative to the target
grid, and either **overwrites** a slice of the existing sequence (`VideoConditionByLatentIndex`
-- in-place at `latent_idx`) or **appends** new tokens at the end
(`VideoConditionByKeyframeIndex`, `VideoConditionByReferenceLatent`,
`AudioConditionByReferenceLatent`) while extending `attention_mask` via
`conditioning/mask_utils.py:build_attention_mask` (block structure: existing tokens keep
their mask, new tokens get full self-attention plus a `strength`-scaled cross-attention
block to/from the noisy tokens). `VideoConditionByMask` instead **interpolates** between
existing and conditioning tokens per-pixel via a patchified mask.
(packages/ltx-core/src/ltx_core/conditioning/mask_utils.py:136,186,195,202,206)

## Configuration

### Config Loading Chain

Two independent config systems -- do not confuse them:

1. **Model config (inference-time)**: embedded as a JSON string in each `.safetensors`
   checkpoint's metadata under the key `"config"`
   (`SafetensorsModelStateDictLoader.metadata` -> `json.loads(meta["config"])`,
   packages/ltx-core/src/ltx_core/loader/sft_loader.py:63). Every `ModelConfigurator.from_config(config: dict)`
   classmethod (`LTXModelConfigurator`, `VideoEncoderConfigurator`, `AudioEncoderConfigurator`,
   `GemmaTextEncoderConfigurator`, `LatentUpsamplerConfigurator`, `VocoderConfigurator`, ...)
   pulls its own keys via `config.get(key, default)` -- every key has a coded default, so a
   checkpoint missing a key silently falls back rather than erroring. **Version handling is
   automatic**: LTX-2 vs LTX-2.3 (feature extractor V1/V2, caption-projection location,
   vocoder legacy/modern) is detected from which keys are present in the config, not from an
   explicit version flag the caller sets (packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py:78;
   packages/ltx-core/src/ltx_core/model/audio_vae/model_configurator.py:53,60).
2. **Trainer config (training-time)**: YAML under `packages/ltx-trainer/configs/*.yaml`
   (one file per conditioning recipe: t2v_lora.yaml, i2v_lora.yaml, v2a_lora.yaml,
   video_extend_lora.yaml, video_inpainting_lora.yaml, video_outpainting_lora.yaml,
   av2av_ic_lora.yaml, etc.), parsed into a strict Pydantic `LtxTrainerConfig`
   (`ConfigDict(extra="forbid")` at every level, so an unknown key is a hard error, not a
   silent no-op) with ten top-level sections: `model`, `lora`, `training_strategy`,
   `optimization`, `acceleration`, `data`, `validation`, `checkpoints`, `flow_matching`,
   `hub`, `wandb`.
   (packages/ltx-trainer/src/ltx_trainer/config.py:13,14,766)

### Key Config Sections Reference

**`training_strategy`** -- discriminated union on the `name` key
(packages/ltx-trainer/src/ltx_trainer/config.py:307,313,316):

| `name` | Status | Conditioning |
|---|---|---|
| `flexible` | recommended | config-driven: `video`/`audio` blocks each set `is_generated: bool` + optional `conditions: [...]` list, each with its own `type` discriminator (`first_frame`, `prefix`, `suffix`, `spatial_crop`, `mask`, `reference`, `video_to_audio`, `audio_to_video`) and a `probability` |
| `text_to_video` | deprecated (warns) | fixed first-frame-only conditioning, `first_frame_conditioning_p` default `0.1` |
| `video_to_video` | deprecated (warns) | IC-LoRA reference-video only; forces `training_mode: lora` (packages/ltx-trainer/src/ltx_trainer/config.py:830) |

Selected `FlexibleStrategyConfig` condition keys and their required constraints
(packages/ltx-trainer/src/ltx_trainer/config.py:27-160):

| Condition type | Key gotcha |
|---|---|
| `prefix` | `num_frames % 8 == 1` required; exactly one of `video`/`audio` set |
| `suffix` | `num_frames % 8 == 0` required (note: different from prefix!); exactly one of `video`/`audio` set |
| `spatial_crop` | `spatial_region: (y1,x1,y2,x2)` tuple, latent-space divisible-by-32 rules apply |
| `mask` | exactly one of `video`/`audio` set |
| `reference` | exactly one of `video`/`audio` set, carries its own downscale/temporal-scale factors |
| `video_to_audio` / `audio_to_video` | frozen cross-modal conditioning; `LtxTrainerConfig` rejects a validation sample that mixes both frozen types in one sample (packages/ltx-trainer/src/ltx_trainer/config.py:215) |

**Shared recipe defaults across every shipped `configs/*.yaml`** (all files converge on the
same values unless a recipe overrides them): LoRA `rank=32, alpha=32, dropout=0.0`,
`target_modules: ["to_k","to_q","to_v","to_out.0"]`, `optimizer_type: adamw`,
`scheduler_type: linear`, `mixed_precision_mode: bf16`, `enable_gradient_checkpointing: true`,
`timestep_sampling_mode: shifted_logit_normal`, `checkpoints.precision: bfloat16`, `seed: 42`.
configs/t2v_lora_low_vram.yaml is the outlier: `rank=16`, `optimizer_type: adamw8bit`,
`quantization: int8-quanto`.

**`ValidationConfig` / `PrefixConditionConfig` divisibility rules** (packages/ltx-trainer/src/ltx_trainer/config.py:203,489): video
dimensions must satisfy `width % 32 == 0`, `height % 32 == 0`, `frames % 8 == 1` -- the same
constraint as the VAE's own pixel-space requirement, checked twice (once at config-validation
time, once again inside the pipeline/`ValidationRunner`).

**`AccelerationConfig`** (packages/ltx-trainer/src/ltx_trainer/config.py:379): `mixed_precision_mode: "no"|"fp16"|"bf16"|None`,
`quantization` (routes through `packages/ltx-trainer/src/ltx_trainer/quantization.py`'s
`optimum-quanto`-based `quantize_model`, options `qint2/qint4/qint8/qfloat8/qfloat8_e4m3fnuz`),
`load_text_encoder_in_8bit` (routes through gemma_8bit.py's bitsandbytes loader).

**Accelerate launch configs** (`packages/ltx-trainer/configs/accelerate/*.yaml`): ddp.yaml
(plain multi-GPU DDP, `mixed_precision: bf16`), ddp_compile.yaml (adds
`dynamo_backend: INDUCTOR`), fsdp.yaml / fsdp_compile.yaml (`fsdp_transformer_layer_cls_to_wrap:
BasicAVTransformerBlock` -- FSDP wraps at the transformer-block granularity, matching the
class hierarchy above).

## Training Internals

### Module Freezing Pattern

- **LoRA mode** (`model.training_mode: "lora"`, the default in every shipped recipe except
  full fine-tunes): `_setup_lora` injects a `peft.LoraConfig` into the transformer;
  `_collect_trainable_params` collects only the LoRA adapter parameters.
  `target_modules` default `["to_k","to_q","to_v","to_out.0"]` -- video-stream attention
  projections; per-recipe configs extend this with `audio_attn1.*`/`audio_attn2.*`/`audio_ff.*`
  / cross-modal `video_to_audio_attn.*` module names for audio-touching recipes.
  (packages/ltx-trainer/src/ltx_trainer/trainer.py:452,433)
- **Full mode**: the entire `LTXModel` is trainable; `video_to_video` strategy explicitly
  forbids this (`packages/ltx-trainer/src/ltx_trainer/config.py:830`, always requires `lora`).
- **Always frozen, never trainable**: `GemmaTextEncoder`,
  `EmbeddingsProcessor`, `VideoEncoder`/`VideoDecoder`, `AudioEncoder`/`AudioDecoder`,
  `Vocoder`. These aren't even loaded during the training step -- process_dataset.py
  precomputes their outputs to `.precomputed/` once, offline; the trainer's main loop only
  ever calls `LTXModel.forward`. They're loaded again, on CPU, only inside `ValidationRunner`
  for periodic sampling.
  (packages/ltx-trainer/src/ltx_trainer/validation_runner.py:411)

### Checkpoint System

- **Weights**: `_save_checkpoint` writes either the full model `state_dict` or, in LoRA mode,
  `peft.get_peft_model_state_dict()` as `.safetensors`, at `checkpoints.precision`
  (`bfloat16` or `float32`). `_cleanup_checkpoints` enforces `keep_last_n` retention.
  (packages/ltx-trainer/src/ltx_trainer/trainer.py:925,985)
- **Load path always uses `strict=False, assign=True`** in every weight-loading call site
  (`SingleGPUModelBuilder`, `StreamingModelBuilder`, block-streaming non-block weights) --
  see Common Pitfalls, this is the single most consequential silent-failure surface in the
  loader stack.
- **Training state** (separate from weights): `TrainingState` (Pydantic) bundles
  `global_step`, `ConfigFingerprint{optimizer_type, scheduler_type, training_mode, lora_rank}`,
  `RngStates{torch_state, cuda_state}`, optional `lr_scheduler_state_dict`/
  `optimizer_state_dict`/`wandb_run_id`. `to_save_dict` recurses nested `BaseModel`s and drops
  `None` fields before `torch.save`; `from_save_dict` reconstructs with Pydantic validation.
  `checkpoints.save_training_state: "full"|"minimal"|"off"` controls whether
  optimizer/scheduler state is included at all.
  (packages/ltx-trainer/src/ltx_trainer/training_state.py:23,37,48;
   packages/ltx-trainer/src/ltx_trainer/config.py:694)
- **Resume**: `_resolve_resume_state` / `_restore_training_state` compare the saved
  `ConfigFingerprint` against the current run's config before restoring optimizer/scheduler/
  RNG state -- an optimizer-type or LoRA-rank change between runs is caught here rather than
  producing a shape-mismatched optimizer state.
  (packages/ltx-trainer/src/ltx_trainer/trainer.py:509,572)

### Training Loop Architecture

```python
# packages/ltx-trainer/src/ltx_trainer/trainer.py:119 train(), :350 _training_step()
for step in range(config.optimization.steps):
    batch = next(dataloader)                                    # PrecomputedDataset
    inputs = strategy.prepare_training_inputs(batch, timestep_sampler)   # -> ModelInputs
    pred_video, pred_audio = transformer(video=inputs.video, audio=inputs.audio,
                                          perturbations=None)      # no guidance in training
    output = TrainingStepOutput(loss=strategy.compute_loss(pred, inputs), sigma=...)
    accelerator.backward(output.loss.mean() / gradient_accumulation_steps)
    if accumulation_boundary:
        clip_grad_norm_(trainable_params, optimization.max_grad_norm)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
    sigma_tracker.update(output.sigma.tolist(), output.loss.detach().tolist())  # 4 sigma buckets
    if step % 200 == 0: log_gpu_memory()                          # MEMORY_CHECK_INTERVAL
    if step % validation.interval == 0: _run_validation(progress)  # offloads optimizer first
    if step % checkpoints.interval == 0: _save_checkpoint(); _save_training_state()
```

- `optimization.enable_gradient_checkpointing` (default `true` in every recipe) calls
  `LTXModel.set_gradient_checkpointing(True)` -- trades transformer-block recompute for
  activation memory.
  (packages/ltx-core/src/ltx_core/model/transformer/model.py:351)
- `_offloaded_optimizer_state` is a context manager that moves optimizer state off-GPU for
  the duration of validation sampling if VRAM is tight
  (`acceleration.offload_optimizer_during_validation` in the config).
  (packages/ltx-trainer/src/ltx_trainer/trainer.py:807)
- `SigmaBucketTracker` buckets per-element loss by sampled sigma (default 4 equal-width
  buckets over `[0,1]`) so `wandb`/console logs surface whether loss is concentrated at
  low-noise or high-noise timesteps.
  (packages/ltx-trainer/src/ltx_trainer/sigma_tracker.py:10,41)

### Precision & Device Rules

| Component | Precision | Reason |
|---|---|---|
| `LTXModel` (transformer) | `bf16` (`acceleration.mixed_precision_mode`) | default training/inference dtype everywhere |
| Video/Audio VAE encode+decode | `bf16` | matches pipeline dtype; `DTYPE = torch.bfloat16` repeated across every `scripts/visualize_*.py` probe |
| `Vocoder` | forced `float32` on MPS (`vocoder_dtype = torch.float32 if device.type == "mps" else dtype`); explicit `fp32_ctx` wrapper elsewhere in the causal STFT path | alias-free `Snake`/`SnakeBeta` activations and STFT bases are numerically unstable outside fp32 on some backends |
| Quantized transformer linears | `float8_e4m3fn` weights (quantization/fp8_cast.py, fp8_scaled_mm.py) or blockwise FP8/FP6 (`ltx_core/quantization/blockwise`, backed by `ltx_kernels`) | opt-in memory/throughput tradeoff, selected via `AccelerationConfig.quantization` |
| LoRA fusion aggregation | `bf16` by default (`bf16_fuse_rule`), forced `fp32` on MPS (`device_fuse_rule`) | bf16 delta accumulation is unreliable on MPS's matmul path |
| `RngStates` | CPU tensor always present, CUDA tensor optional | reproducible resume without requiring a CUDA device to be present |
| Blockwise-quantized `Linear` | `in_features`/`out_features` must be divisible by 128 (4/8 for fp6) | hard `assert` in ltx_kernels/blockwise/linear.py -- see Common Pitfalls |

## Reference

### File Quick Reference

One representative row per worklist folder at minimum; files most likely to be the entry
point for a change in that folder are listed first.

| Path | Purpose | Key classes/functions |
|---|---|---|
| `pyproject.toml` | uv workspace root, ruff config | `[tool.uv.sources]`, `members = ["packages/*"]` |
| `scripts/visualize_vae.py` | Deep VAE diagnostic: PCA projections, reconstruction, tiled encode/decode probes | `build_stage_defs`, `encode_tail_stages_tiled`, `compute_psnr` |
| `scripts/vae_motion_probe.py` | Benchmarks VAE PSNR across motion types/frame counts | `load_video`, `main` |
| `scripts/visualize_spatial_upscaler.py` | Probes LTX-2.3 spatial latent upscaler vs bicubic | branch detection via safetensors metadata |
| `scripts/visualize_two_stage_upscale_refine.py` | Compares 4 VAE decode paths at 1024x768 | `ImageConditioner`, `VideoUpsampler`, `SimpleDenoiser` refinement |
| `packages/ltx-core/pyproject.toml` | ltx-core package metadata | version `1.1.7` |
| `packages/ltx-core/src/ltx_core/types.py` | Video/Audio latent shape math, `LatentState` | `VideoLatentShape`, `AudioLatentShape`, `LatentState` |
| `packages/ltx-core/src/ltx_core/tiling.py` | Overlapping-tile splitting + trapezoidal blend masks | `create_tiles`, `TileCountConfig` |
| `packages/ltx-core/src/ltx_core/modality_tiling.py` | Tile/blend a `Modality`'s token sequence | `VideoModalityTilingHelper` |
| `packages/ltx-core/src/ltx_core/block_streaming/wrapper.py` | Streams transformer blocks in from disk/pinned CPU | `BlockStreamingWrapper` |
| `packages/ltx-core/src/ltx_core/block_streaming/builder.py` | Builds a `BlockStreamingWrapper` from checkpoint | `StreamingModelBuilder` |
| `packages/ltx-core/src/ltx_core/components/guiders.py` | CFG/STG/APG guidance implementations | `CFGGuider`, `STGGuider`, `MultiModalGuider` |
| `packages/ltx-core/src/ltx_core/components/patchifiers.py` | Latent <-> token sequence conversion | `VideoLatentPatchifier`, `AudioPatchifier` |
| `packages/ltx-core/src/ltx_core/components/schedulers.py` | Sigma schedule generation | `LTX2Scheduler`, `LinearQuadraticScheduler`, `BetaScheduler` |
| `packages/ltx-core/src/ltx_core/conditioning/item.py` | `ConditioningItem` protocol | `apply_to(latent_state, latent_tools)` |
| `packages/ltx-core/src/ltx_core/conditioning/mask_utils.py` | Attention-mask construction for appended tokens | `build_attention_mask`, `update_attention_mask` |
| `packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py` | IC-LoRA reference-video conditioning | `VideoConditionByReferenceLatent` |
| `packages/ltx-core/src/ltx_core/guidance/perturbations.py` | STG attention-skip masks, batched across samples | `PerturbationType`, `BatchedPerturbationConfig` |
| `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py` | Single-GPU model build + LoRA fuse | `SingleGPUModelBuilder`, see Common Pitfalls |
| `packages/ltx-core/src/ltx_core/loader/fuse_loras.py` | LoRA delta aggregation and fusion into base weights | `aggregate_lora_products`, `apply_loras` |
| `packages/ltx-core/src/ltx_core/loader/sft_loader.py` | Reads `.safetensors` weights + embedded JSON config | `SafetensorsModelStateDictLoader` |
| `packages/ltx-core/src/ltx_core/model/model_protocol.py` | `ModelConfigurator`/model forward protocols | `ModelConfigurator[ModelType]` |
| `packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py` | Audio VAE encoder/decoder | `AudioEncoder`, `AudioDecoder`, `encode_audio`, `decode_audio` |
| `packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py` | BigVGAN-style mel->waveform vocoder | `Vocoder`, `VocoderWithBWE`, `Snake`/`SnakeBeta` |
| `packages/ltx-core/src/ltx_core/model/common/normalization.py` | `NormType`/`PixelNorm` shared by VAE and transformer | `build_normalization_layer` |
| `packages/ltx-core/src/ltx_core/model/transformer/model.py` | The transformer itself | `LTXModel`, `X0Model`, `LegacyX0Model` |
| `packages/ltx-core/src/ltx_core/model/transformer/transformer.py` | The repeated dual-stream block | `BasicAVTransformerBlock`, `TransformerConfig` |
| `packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py` | Config-dict -> `LTXModel` construction, all defaults | `LTXModelConfigurator.from_config` |
| `packages/ltx-core/src/ltx_core/model/transformer/attention.py` | Backend-selectable attention (SDPA/FlashAttention/MPS) | `Attention`, `automatic_attention` |
| `packages/ltx-core/src/ltx_core/model/transformer/rope.py` | Rotary position embeddings, split/interleaved modes | `apply_rotary_emb`, `precompute_freqs_cis` |
| `packages/ltx-core/src/ltx_core/model/transformer/compiling.py` | `torch.compile` wrapping for the transformer | `compile_transformer` |
| `packages/ltx-core/src/ltx_core/model/upsampler/model.py` | 2x spatial latent upsampler | `LatentUpsampler`, `upsample_video` |
| `packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py` | Video VAE encoder/decoder, tiled encode/decode | `VideoEncoder`, `VideoDecoder`, `tiled_decode` |
| `packages/ltx-core/src/ltx_core/model/video_vae/memory_efficient_decode.py` | In-place/chunked decode to cut peak VRAM | `enable_memory_efficient_decode` |
| `packages/ltx-core/src/ltx_core/multigpu/sharded_sd.py` | Sharded weight backup/restore across ranks | `ShardedSD` |
| `packages/ltx-core/src/ltx_core/multigpu/gemma/broadcast_wrapper.py` | Base for distributed Gemma encoding | `BroadcastGemmaWrapper` |
| `packages/ltx-core/src/ltx_core/multigpu/transformer/sequence_parallel.py` | Sequence-parallel transformer wrapper | `SequenceParallelModelWrapper` |
| `packages/ltx-core/src/ltx_core/multigpu/transformer/attention.py` | All2All-based distributed attention | `AttentionManager`, `All2AllAttention` |
| `packages/ltx-core/src/ltx_core/multigpu/vae/distributed_decoder.py` | Distributed tiled VAE decode | `DistributedVideoDecoder` |
| `packages/ltx-core/src/ltx_core/quantization/fp8_cast.py` | Naive FP8 downcast + stochastic-rounding upcast | `Fp8CastLinear`, `build_policy` |
| `packages/ltx-core/src/ltx_core/quantization/fp8_scaled_mm.py` | Scaled FP8 matmul linear layer | `FP8Linear` |
| `packages/ltx-core/src/ltx_core/quantization/blockwise/_impl.py` | Blockwise FP8/FP6 quantization, kernel-backed | `BlockwiseFP8LTXModelConfigurator` |
| `packages/ltx-core/src/ltx_core/text_encoders/gemma/config.py` | Gemma3 architecture config dataclasses | `GEMMA3_CONFIG_FOR_LTX` |
| `packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py` | Hidden states -> video/audio caption embeddings | `Embeddings1DConnector` |
| `packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py` | Orchestrates connector + masking | `EmbeddingsProcessor`, `EmbeddingsProcessorOutput` |
| `packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py` | Gemma wrapper, prompt enhancement | `GemmaTextEncoder.encode/enhance_t2v/enhance_i2v` |
| `packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py` | Builds Gemma on meta device, V4/V5 rotary population | `GemmaTextEncoderConfigurator` |
| `packages/ltx-kernels/setup.py` | Compiles CUDA extensions (all2all, blockwise GEMM, ops) | — |
| `packages/ltx-kernels/csrc/all2all/all2all.cpp` | IPC-based multi-GPU All2All primitive | `All2All` class |
| `packages/ltx-kernels/csrc/blockwise/api.cpp` | Python bindings for blockwise FP8/FP6 GEMM | `PYBIND11_MODULE` |
| `packages/ltx-kernels/csrc/ops/rms_norm_rope_cuda.cu` | Fused RMSNorm + RoPE CUDA kernel | `rms_norm_rope_cuda` |
| `packages/ltx-kernels/csrc/ops/fp6_pack.cu` | FP6 bit-pack/unpack kernels | `fp6_pack_kernel` |
| `packages/ltx-kernels/src/ltx_kernels/blockwise/linear.py` | Python-facing blockwise-quantized `Linear` | see 128-alignment asserts in Common Pitfalls |
| `packages/ltx-kernels/src/ltx_kernels/all_to_all.py` | Python bridge to the C++ All2All primitive | — |
| `packages/ltx-kernels/csrc/all2all/cuda/api.cuh` | Host-side launch declarations for the All2All/AllGather CUDA kernels | — |
| `packages/ltx-kernels/csrc/blockwise/kernels/geforce/gemm.cu` | SM89 (GeForce) blockwise FP8 GEMM kernel, CuTe-based | `namespace sm89` |
| `packages/ltx-kernels/csrc/blockwise/kernels/deep_gemm/include/deep_gemm/common/sm90_utils.cuh` | SM90 (Hopper) tensor-core intrinsics: MMA templates, TMA copy, GMMA descriptors | `namespace deep_gemm::sm90` |
| `packages/ltx-kernels/csrc/blockwise/kernels/deep_gemm/include/deep_gemm/common/types.hpp` | GEMM operation-mode enum shared across `deep_gemm` variants | `enum class GemmType` |
| `packages/ltx-kernels/csrc/blockwise/kernels/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` | The actual SM90 FP8 blockwise GEMM kernel implementation | `sm90_fp8_gemm_1d2d_impl` |
| `packages/ltx-kernels/csrc/include/cuda/utils.cuh` | Low-level non-caching memory/PTX-atomic/barrier primitives shared by all2all + blockwise kernels | — |
| `packages/ltx-kernels/csrc/ops/include/fast_hadamard_transform.h` | Parameter structs for the fused Hadamard-transform kernel family | `struct HadamardParamsBase` |
| The remaining `csrc/blockwise/kernels/{deep_gemm/include/deep_gemm/{,common,impls},geforce}` and `csrc/{include/cuda,ops/include}` folders | Internal CUDA/C++ template headers (scheduling, epilogue indexing, warp reduction, static-switch macros) consumed only by the `.cu`/`.cpp` files above | not directly Python-visible; per the guide's "non-Python kernel" rule, treat as a black box behind the `packages/ltx-kernels/csrc/blockwise/api.cpp`/`packages/ltx-kernels/csrc/ops/ops_api.cpp` PyBind11 bindings unless doing kernel-level CUDA work |
| `packages/ltx-pipelines/CLAUDE.md` | Pipeline maintenance contract (read before editing any pipeline class) | — |
| `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` | Primary fast inference entry point | `DistilledPipeline` |
| `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py` | Two-stage T2V/I2V with CFG + distilled refine | `TI2VidTwoStagesPipeline` |
| `packages/ltx-pipelines/src/ltx_pipelines/lipdub.py` | IC-LoRA + reference-audio lip dub pipeline | `LipDubPipeline` |
| `packages/ltx-pipelines/src/ltx_pipelines/retake.py` | Regenerate a time window of an existing video | `RetakePipeline` |
| `packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py` | Keyframe-conditioned interpolation | `KeyframeInterpolationPipeline` |
| `packages/ltx-pipelines/src/ltx_pipelines/iclora_utils.py` | Shared IC-LoRA helpers (metadata, mask downsample) | `read_lora_reference_downscale_factor` |
| `packages/ltx-pipelines/src/ltx_pipelines/multigpu/controller.py` | Multi-GPU SPMD job controller | `MGPUController`, `Stream` |
| `packages/ltx-pipelines/src/ltx_pipelines/multigpu/fleet.py` | Worker process fleet + wire protocol | `_RunnersFleet`, `_Job` |
| `packages/ltx-pipelines/src/ltx_pipelines/multigpu/weight_tracker.py` | Distributed LoRA hot-swap | `TransformerWeightTracker` |
| `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py` | Reusable lazy-build/auto-free pipeline blocks | `DiffusionStage`, `PromptEncoder`, `VideoDecoder` |
| `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py` | Guided-denoising batched transformer call | `SimpleDenoiser`, `GuidedDenoiser`, `FactoryGuidedDenoiser` |
| `packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py` | Denoising loops (Euler/res2s/CFG++) | `euler_denoising_loop`, `res2s_audio_video_denoising_loop` |
| `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py` | Latent<->Modality conversion, conditioning application | `modality_from_latent_state`, `state_with_conditionings` |
| `packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py` | Video/audio encode/decode, color conversion | `encode_video`, `decode_video_from_file` |
| `packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` | Sigma schedules, `PipelineParams`, version detection | `DISTILLED_SIGMA_VALUES`, `detect_params` |
| `packages/ltx-trainer/AGENTS.md` / `CLAUDE.md` | Trainer architecture + config-table maintenance contract | — |
| `packages/ltx-trainer/configs/*.yaml` | One recipe per conditioning mode (see Configuration) | — |
| `packages/ltx-trainer/configs/accelerate/*.yaml` | DDP/DDP+compile/FSDP/FSDP+compile launch configs | — |
| `packages/ltx-trainer/scripts/train.py` | Training CLI entry point | `main(config_path)` |
| `packages/ltx-trainer/scripts/process_dataset.py` | Full offline preprocessing pipeline (text/video/audio/masks) | `preprocess_dataset` |
| `packages/ltx-trainer/scripts/process_videos.py` | Video/audio -> latent encoding, resolution bucketing | `compute_latents`, `compute_video_masks` |
| `packages/ltx-trainer/scripts/process_captions.py` | Caption -> text embedding precompute | `compute_captions_embeddings` |
| `packages/ltx-trainer/scripts/caption_videos.py` | Auto-captioning via Qwen-Omni/Gemini | `QwenOmniCaptioner`, `GeminiFlashCaptioner` |
| `packages/ltx-trainer/scripts/decode_latents.py` | Decode precomputed latents back to media (debugging) | `LatentsDecoder` |
| `packages/ltx-trainer/scripts/compute_reference.py` | Canny-edge reference videos for IC-LoRA | `compute_reference` |
| `packages/ltx-trainer/src/ltx_trainer/trainer.py` | Full training lifecycle | `LtxvTrainer` |
| `packages/ltx-trainer/src/ltx_trainer/config.py` | All Pydantic config models + cross-field validation | `LtxTrainerConfig` |
| `packages/ltx-trainer/src/ltx_trainer/training_state.py` | Resumable training-state serialization | `TrainingState`, `ConfigFingerprint` |
| `packages/ltx-trainer/src/ltx_trainer/validation_runner.py` | Self-contained validation sampling during training | `ValidationRunner` |
| `packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py` | Flow-matching timestep sampling | `UniformTimestepSampler`, `ShiftedLogitNormalTimestepSampler` |
| `packages/ltx-trainer/src/ltx_trainer/sigma_tracker.py` | Per-sigma-bucket loss tracking | `SigmaBucketTracker` |
| `packages/ltx-trainer/src/ltx_trainer/quantization.py` | Training-time optimum-quanto quantization | `quantize_model` |
| `packages/ltx-trainer/src/ltx_trainer/datasets.py` | Precomputed-latents dataset loader | `PrecomputedDataset`, `DummyDataset` |
| `packages/ltx-trainer/src/ltx_trainer/model_loader.py` | Unified component loading for the trainer | `load_model`, `LtxModelComponents` |
| `packages/ltx-trainer/src/ltx_trainer/training_strategies/__init__.py` | Strategy dispatch factory | `get_training_strategy` |
| `packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py` | Shared strategy base, position/timestep helpers | `TrainingStrategy`, `ModelInputs` |
| `packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py` | Recommended config-driven strategy | `FlexibleStrategy`, `FlexibleStrategyConfig` |

### Helper Module Reference

| Function | Contract | Gotcha |
|---|---|---|
| `rms_norm(x, weight=None, eps=1e-6)` (`packages/ltx-core/src/ltx_core/utils.py:7`) | `[..., C] -> [..., C]`, wraps `F.rms_norm` | — |
| `check_config_value(config, key, expected)` (`packages/ltx-core/src/ltx_core/utils.py:15`) | raises with a descriptive message on mismatch | used to validate checkpoint config against code assumptions |
| `to_velocity` / `to_denoised` (`packages/ltx-core/src/ltx_core/utils.py:21,39`) | flow-matching conversions, both take `sigma` | division by `sigma`; caller must guard `sigma != 0` |
| `to_vae_range` / `from_vae_range` (`packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py:98,103`) | `[0,1] <-> [-1,1]` | inverse of each other; VAE always expects `[-1,1]` |
| `compute_trapezoidal_mask_1d` / `compute_rectangular_mask_1d` (`packages/ltx-core/src/ltx_core/tiling.py:11,47`) | `(length,) -> [0,1]` blend weight | trapezoidal ramps at tile edges; rectangular is a hard cut |
| `align_resolution` (`packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py:153`) | rounds requested resolution to the VAE's divisor (`64` two-stage / `32` one-stage) | must match the divisor the pipeline was constructed with |
| `_conform_latent_length` (`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:38`) | pads/crops a latent tensor to a required frame count | silently changes content length; check call site expectations |

### Dataset Notes

- `PrecomputedDataset` reads from `<preprocessed_data_root>/.precomputed/` -- process_dataset.py
  is a 4-phase pipeline: (1) caption embeddings, (2) video latents + optional auto-audio
  extraction, (2b) video masks aligned to latent grid, (3) audio latents for `audio`/
  `reference_audio` roles, (4) audio masks. Every phase is independently resumable (`_is_done`
  checks for existing output files before re-encoding).
  (packages/ltx-trainer/scripts/process_dataset.py:54,139,190,199,232)
- Dataset columns are auto-detected against `_KNOWN_ROLES = {video, audio, reference_video,
  reference_audio, video_mask, audio_mask, caption}` with legacy aliases
  `media_path -> video`, `ref_media_path -> reference_video`.
  (packages/ltx-trainer/scripts/process_dataset.py:50,51,268)
- Reference-video preprocessing uses *scaled* resolution buckets
  (`compute_scaled_resolution_buckets(buckets, reference_downscale_factor)`) -- the reference
  video is encoded at a lower resolution than the target, matching what IC-LoRA conditioning
  expects at inference time (see iclora_utils.py).

### Common Pitfalls

**[SILENT BUG] Every weight-load call site uses `strict=False, assign=True`.**
`SingleGPUModelBuilder._load_model_weights` (packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:67,83),
`StreamingModelBuilder._load_non_block_weights` (packages/ltx-core/src/ltx_core/block_streaming/builder.py:430).
A renamed, misspelled, or dropped key in an `SDOps` rename map does not raise -- the tensor is
silently left at its meta-device/random-init value instead of the checkpoint's weight. When
adding a new `SDOps` mapping or renaming a module, diff the loaded state-dict keys against the
checkpoint's key list by hand; do not trust a clean `build()` call as proof the mapping is
correct.

**[SILENT BUG] `zip(..., strict=False)` in two list-pairing call sites.**
`packages/ltx-trainer/scripts/caption_videos.py:314` (existing captions vs. paths) and `packages/ltx-trainer/src/ltx_trainer/hf_hub_utils.py:153` (validation
prompts vs. generated videos) both truncate silently to the shorter list on a length mismatch
instead of raising. If you change how either list is built, verify the lengths still match.

**[GRAD BUG] `.detach()`/`@torch.no_grad()` in diagnostic scripts and trainer logging is
load-bearing, not decorative.** `scripts/visualize_vae.py` (lines 387,531,537,542-544,548,593,
622,629,640,655,701,716), `scripts/visualize_two_stage_upscale_refine.py` (lines 194,284,344),
`scripts/visualize_spatial_upscaler.py:318`, `scripts/vae_latent_boundary_removal.py`
(lines 155,163,185) detach every intermediate tensor before it's stashed for PCA/PSNR
visualization -- removing one leaks the encoder/decoder's autograd graph into a script that has
no optimizer, growing GPU memory unboundedly over a run. In trainer.py (lines 261,273,389,391)
and `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py:319`, `.detach()` on `sigma`/`loss` before `.cpu().tolist()`
keeps the training graph from retaining a reference through the logging/tracking path.

**[GRAD BUG] `ValidationRunner`'s three top-level entry points are `@torch.no_grad()`**
(`packages/ltx-trainer/src/ltx_trainer/validation_runner.py:154,248,296`) -- if you factor sampling logic out into a helper called
from *outside* one of these three methods, the new call site will build a full autograd graph
during validation unless it's wrapped separately.

**[CRASH] Patchifier/RoPE shape asserts fire on malformed `VideoLatentShape`/`AudioLatentShape`
construction, not on weights.** `assert frames > 0` / `height > 0` / `width > 0` /
`batch_size > 0` (`packages/ltx-core/src/ltx_core/components/patchifiers.py:89-92`); `assert len(timesteps.shape) == 1`
(`packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py:32`); `assert len(indices_grid.shape) == 4` and
`indices_grid.shape[-1] == 2` (`packages/ltx-core/src/ltx_core/model/transformer/rope.py:148,149`). If you hit one of these, the bug
is almost always upstream in how a shape/duration was computed, not in the transformer itself.

**[CRASH] Blockwise-quantized `Linear` requires 128-divisible dims (4/8 for fp6).**
`packages/ltx-kernels/src/ltx_kernels/blockwise/linear.py:159,160,218-221`. A custom head, LoRA target,
or any module with a non-128-aligned `in_features`/`out_features` will build and run fine
unquantized, then crash the instant `AccelerationConfig.quantization` selects a blockwise FP8/FP6
policy.

**[CRASH] blur_downsample.py/pixel_shuffle.py dimensionality asserts.**
`assert dims in (2, 3)` (`packages/ltx-core/src/ltx_core/model/upsampler/blur_downsample.py:16`), `assert dims in [1, 2, 3]`
(`packages/ltx-core/src/ltx_core/model/upsampler/pixel_shuffle.py:27`), plus `stride`/`kernel_size` positivity and parity asserts
(`packages/ltx-core/src/ltx_core/model/upsampler/blur_downsample.py:17-20`) -- these guard the spatial upsampler's internal dimensionality
argument, not the video's actual `H`/`W`/`F`; passing `dims=` the number of *spatial* axes
(not batch/channel) is the usual mistake.

**[CRASH] Multi-GPU lifecycle-ordering asserts.** `MGPUController` requires `start()` (which
populates `self._channels`/`self._fleet`/`self._num_gpus`) before any `stream()` call
(`packages/ltx-pipelines/src/ltx_pipelines/multigpu/controller.py:245,246,304,346-348`); `BroadcastGemmaWrapper._broadcast_str`
asserts the NCCL broadcast actually produced a result (`packages/ltx-core/src/ltx_core/multigpu/gemma/broadcast_wrapper.py:80`);
`BatchedPerturbationConfig.any_in_batch`/`all_in_batch` require the CPU mirror to exist
(`packages/ltx-core/src/ltx_core/guidance/perturbations.py:117,121`, only true when built via the default host-then-device
path, not `from_masks(..., cpu_mirror=None)`). A `None` or assertion failure here almost always
means a required setup call was skipped, not a data problem.

**[CRASH] Shape-equality guards across a resample/decode boundary.** `assert x.shape[1] == 2`
(stereo waveform, `packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py:410`), `assert residual.shape == skip.shape`
(`packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py:627`), `assert upsampled_latent.shape == original_latent.shape`
(`scripts/visualize_two_stage_upscale_refine.py:344` -- the spatial upsampler's 2x output must
exactly match the tiled-encoder path's own upsampled shape; a `TilingConfig` mismatch between
the two paths trips this before any numeric comparison happens),
`assert seq_len % self.num_learnable_registers == 0`
(`packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py:140`), `assert isinstance(builder,
StreamingModelBuilder)` (`packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:400` -- only a streaming builder can
back a streaming context manager), `assert (f_pix - 1) % (f_lat - 1) == 0`
(`packages/ltx-pipelines/src/ltx_pipelines/iclora_utils.py:67`, reference-video pixel/latent frame-count alignment for
IC-LoRA), `assert n_pos_dims == len(max_pos)` (`packages/ltx-core/src/ltx_core/model/transformer/rope.py:134`, RoPE position-grid
dimensionality must match the configured `positional_embedding_max_pos` length).

### New Variant Checklist

Adding a new **pipeline** (a new generation mode, e.g. a new conditioning combination):

1. Create `packages/ltx-pipelines/src/ltx_pipelines/<name>.py`, a class following the
   `__init__(checkpoint paths, LoRAs, quantization, ...)` / `__call__(prompt, ...) -> (video,
   audio)` / `main()` CLI shape every existing pipeline uses (see distilled.py for the
   simplest example).
2. Compose it from utils/blocks.py building blocks (`DiffusionStage`, `PromptEncoder`,
   `ImageConditioner`/`AudioConditioner`, `VideoUpsampler`, `VideoDecoder`, `AudioDecoder`) --
   do not hand-build a `SingleGPUModelBuilder` directly; the blocks own lazy-build/auto-free
   lifecycle.
3. Register the class in ltx_pipelines/__init__.py's `_EXPORTS` lazy-import map and `__all__`.
4. Add a CLI arg parser via utils/args.py's `new_video_gen_arg_parser` factory family.
5. Update `packages/ltx-pipelines/README.md`'s pipeline table and
   `packages/ltx-pipelines/CLAUDE.md`'s maintenance contract per its own instructions.
6. Update this hub's Model Class Hierarchy pipeline list and File Quick Reference.

Adding a new **training conditioning type** (extending `FlexibleStrategy`):

1. Add a new `IntrinsicConditionBase` subclass with a `Literal["your_type"]` discriminator in
   training_strategies/flexible.py (mirror `PrefixConditionConfig`/`MaskConditionConfig`).
2. Add the corresponding `ValidationCondition` variant in config.py and update
   `_condition_targets_video`/`_condition_targets_audio`.
3. Wire encoding in `ValidationRunner._encode_sample_conditions` and application in
   `_apply_video_conditionings`/`_apply_audio_conditionings` (validation_runner.py) --
   these must exactly mirror what `FlexibleStrategy._process_modality` does for training,
   or validation samples will not represent what the model was actually trained on.
4. Add a `configs/<name>_lora.yaml` recipe following the shared-defaults pattern above.
5. Update the Key Config Sections Reference table in this hub.

### Removing a Feature Checklist

1. Liveness-grep this annotation for every reference to the feature (`grep -rn <name> docs/ANNOTATION.md`).
2. Remove from Spine -> Class Hierarchy -> Shapes & Naming -> Forward Output -> Configuration ->
   Module Freezing -> Common Pitfalls -> File Quick Reference, in that order (later sections
   often reference earlier ones by name).
3. Re-run `python3 ../skills/create-annotation/scripts/check_annotation.py docs/ANNOTATION.md .`
   from `LTX-2/` to confirm nothing dangles.
