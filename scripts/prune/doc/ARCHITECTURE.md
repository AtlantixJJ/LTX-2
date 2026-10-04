# Whole-clip pruning architecture

`data.whole_clip` validates baseline manifests, reconstructs capture/noise/context
inputs and verifies paired candidates. `score.whole_clip_d0_scores` observes
native forwards through `score.hooks`, computes `score.estimators` statistics and
writes a pinned mask. `score.export_pruned` turns that mask into a checkpoint.
`score.export_depth` separately binds intact-block removal to native saved
calibration inputs, streams retained payloads into a shorter checkpoint and
records original/compact block mappings with exact resident-video counts.
`score.token_sampling` owns deterministic generated-frame sample indices and
their hashes. `score.ffn_reconstruction` fits a bounded local ridge correction
and verifies calibration caches; changed fitted weights need separate model
validation and are not faithful width-mask exports.

`checks.export_parity` compares functional masking with checkpoint execution.
`evaluate.whole_clip_d0` compares baseline and candidate directions, capture
fidelity and decoded media. `evaluate.bench_whole_clip_d0` measures warmed,
synchronized full-transformer latency and memory with bracketed baseline arms.

`core.session` owns model and decoder lifetime, dtype and prompt selection;
`core.artifacts` owns output paths; `core.provenance` owns content identity.
`core.model_registry`, `geometry` and `ltx_adapter` own checkpoint resolution,
VAE scales and private upstream access. `data.prompt_cache` owns cached text.

Avatar scripts share bootstrap, checkpoint, prompt, dense decoding and video
utilities. Avatar one-step schedule validation lives in
`onestep_avatar.sampling`; corpus coverage lives in `precompute.CaptureGeometry`.
Prune imposes no avatar block geometry or schedule.
