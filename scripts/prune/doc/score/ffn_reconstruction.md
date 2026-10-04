# `score/ffn_reconstruction.py` — bounded local ridge and calibration caches

## Interface and objective

`fit_output_projection(X, Y, W0, ridge_lambda=..., max_samples=512,
output_chunk=256, max_memory_bytes=1<<30)` accepts retained post-activation
features `[N,K]`, teacher projection targets `[N,O]` and sliced source output
weight `[O,K]` on CPU. It solves for a correction around W0:

$$
\min_\Delta \frac{1}{N}\|X(W_0+\Delta)^T-Y\|_F^2
  +\lambda\|\Delta\|_F^2.
$$

The dual solve is
`Delta.T = X.T @ solve(X @ X.T + N*lambda*I, Y - X @ W0.T)`.
Cholesky factorization and output-channel chunks use FP64. The helper forms an
`N×N` matrix, never a `K×K` matrix. A conservative memory estimate includes
caller tensors, result, FP64 features, factorization and chunk headroom; it fails
before allocation when the declared cap is exceeded. Lambda must be positive
and finite; rank-deficient features remain supported through ridge damping.

The returned `weight` is FP64 and the diagnostics record sample count, caps,
lambda, solve precision and local calibration MSE before/after. These statistics
are not whole-model or held-out improvement. Bias remains unchanged: Y excludes
source bias. Production collection must label targets as
`fp32_source_ffn_projection_without_bias_v1`, computed from sampled BF16 features
and original weights in FP32. Native BF16 reductions/rounding are a separate
numerical question; no native BF16 parity claim follows from an FP64 fit.

## Calibration cache contract

`validate_cache_manifest(path, baseline=None, ...)` checks the separate
`whole_clip_d0_ffn_reconstruction_samples_v1` schema before loading CPU tensors.
`load_calibration_cache` verifies expected tensor metadata on the meta device,
then loads finite FP32 `retained_features` and `teacher_output` only. It checks:

- Complete native D0 provenance and immutable calibration-manifest content.
- An unchanged source checkpoint and full native width-mask hash/distribution.
- One declared FFN branch and exact retained indices in source order; production
  retained width is aligned to 128 by default.
- Every calibration view/sigma exactly once, equal per-case quotas, contiguous
  payload row slices and the total sample cap.
- Actual capture SHA256, schema-2 BF16 latent F/H/W/fps, saved relative epsilon
  SHA256 and BF16 token geometry.
- Deterministic sampler metadata and integer hashes. Each case uses the
  `midpoint_subsample_v1` bounded quota from its full generated-frame selection;
  clean c0 is never selected.
- Payload file hash/byte cap, expected shape/dtype, allocation cap and finite
  values; source/mask/manifest/payload identity is checked again after loading.

The JSON fields are `cache_format`, `target`, `selection`, `provenance`,
`source_checkpoint`, `mask_path`, `mask_sha256`, `branch`, `retained_indices`,
`payload`, `payload_sha256`, and `cases`. Each case supplies `view`, `sigma`,
the complete `token_sampling.sampling_record`, `token_indices`,
`token_indices_sha256`, and `row_slice=[start,end]`. Paths other than the native
epsilon field are explicit paths. Cache validation binds recorded identities
and scope; it cannot certify that arbitrary cached features/targets were
actually collected from the declared model. A future collector must reproduce
the saved native forward before accepting its cache.

## Scope and verification

No GPU collector, ridge-fitted checkpoint exporter or model improvement has
been run by this implementation. Callers must supply W0 from the pinned source
projection and retained source-order columns. Fitted weights define a changed
model, separate from the unchanged faithful width-mask gate. Validate its
in-memory/saved execution and held-out full model quality independently.

`test_ffn_reconstruction.py` compares the dual result with an independent primal
solve, correlated-feature synthetic recovery, zero/rank-deficient features,
identity/chunk controls and numerical/memory rejection. Native tiny checkpoint
caches exercise scope, mask/source/capture/noise/sampler/payload mutations and
alignment/sample caps. The synthetic local recovery test demonstrates algebra,
not observed improvement in LTX avatars.
