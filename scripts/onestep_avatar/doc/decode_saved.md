# `decode_saved.py`

## Objective

Render saved latent outputs and controlled decoder comparisons without transformer inference, for reproducible presentation and RGB evidence.

## Data flow

A jobs JSON binds absolute latent paths, SHA-256, optional prefix length, decode seed, FPS and comparison IDs. The session decoder and public `media.decode` produce frames. Output contains MP4, sampled PNG, and `manifest.json` with source/input/output hashes, VAE identity, runtime and uncompressed per-frame difference metrics.

## Organization logic

Capture the shared decoding software manifest before reading jobs; check it
before opening the decoder and before each manifest publication. Record it in
every decoded row and comparison. Reuse requires a current saved manifest in
addition to the existing file/VAE checks. If a cached row's software is absent
or stale, preserve it and require a fresh destination instead of overwriting
historical evidence. Read-only reuse returns false for such a row.

Native decoding settings come from `media.native_decoder_settings`; queue
verification uses that same identity rather than maintaining a second copy.
Session setup uses `media.open_decoder_session`, which performs shared CUDA
preflight and uses a null text context. It prepares no prompt embeddings and
opens neither a text encoder nor a transformer.

`parse_args(argv=None)` exposes the saved-decoder argument contract without
reading jobs or opening a model. Queue validation reuses this parser; `main`
then performs the existing decoding steps with its checked arguments.

Decode whole outputs consistently. A fresh identical generator is owned by `media.decode`. The job producer only selects saved tensors and lengths; it does not own rollout, noising, conditioning or cache logic. Named comparison jobs retain CPU decoder outputs until pixel differences are measured.

Before checking reuse, verify the source hash, load the saved tensor, and apply
the requested encoded-frame prefix. Calculate `media.decode_key` from the saved
file hash, full VAE hash, effective tensor dimensions, native decoder method,
seed, and actual settings (bf16 conversion, no tiling, fresh generator, Torch
version). A standalone result is reusable only if this key, complete job,
producer hash, MP4 hash, and every recorded PNG hash match. Older manifests
without a key are redecoded. A changed VAE or missing poster therefore causes
a decode even when the saved tensor and MP4 are unchanged. Comparison jobs
always decode to obtain uncompressed pixels for their measurements.

## Invariants

No transformer context is opened. Input hashes must match before rendering. No full rollout is reconstructed. Numerical differences use decoder float frames before video encoding or PNG quantization. PNG differences have an explicit 20× gain. Output quantization is documented.

## Gotchas

Length changes decoder context and noise shape together. Foreground is a derived union threshold mask, not the corpus mask or a training mask. Comparison-required jobs are redecoded together to ensure all numerical inputs exist in memory; these jobs do not claim a cached numerical result. Saved frames are illustrative samples, while numerical curves cover every shared frame.

## Tests

Check `--help` from the LTX-2 root in `ltx`; run the package suite. Focused reuse
checks must reject changed decoder keys, old manifests, changed MP4s, and
missing or changed PNGs. A decode smoke job must verify input/output hashes,
expected frame count and deterministic repeat comparison. The study owns
actor/config jobs and compares original metric tolerances; this producer remains generic.

## Invocation

From `LTX-2`: `conda run -n ltx python -m scripts.onestep_avatar.decode_saved --jobs /absolute/jobs.json --output /absolute/output --gpu-id 4 --seed 42 --model 2.5`.
