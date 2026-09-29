# `decode_saved.py`

## Objective

Render saved latent outputs and controlled decoder comparisons without transformer inference, for reproducible presentation and RGB evidence.

## Data flow

A jobs JSON binds absolute latent paths, SHA-256, optional prefix length, decode seed, FPS and comparison IDs. The existing session decoder and `visualize_d0._decode` produce frames. Output contains MP4, sampled PNG, and `manifest.json` with source/input/output hashes, VAE identity, runtime and uncompressed per-frame difference metrics.

## Organization logic

Decode whole outputs consistently. A fresh identical generator is owned by `_decode`. The job producer only selects saved tensors and lengths; it does not own rollout, noising, conditioning or cache logic. Named comparison jobs retain CPU decoder outputs until pixel differences are measured. Standalone jobs resume only when input, producer hash and output hash match.

## Invariants

No transformer context is opened. Input hashes must match before rendering. No full rollout is reconstructed. Numerical differences use decoder float frames before video encoding or PNG quantization. PNG differences have an explicit 20× gain. Output quantization is documented.

## Gotchas

Length changes decoder context and noise shape together. Foreground is a derived union threshold mask, not the corpus mask or a training mask. Comparison-required jobs are redecoded together to ensure all numerical inputs exist in memory; these jobs do not claim a cached numerical result. Saved frames are illustrative samples, while numerical curves cover every shared frame.

## Tests

Check `--help` from the LTX-2 root in `ltx`; run the package suite. A decode smoke job must verify input/output hashes, expected frame count and deterministic repeat comparison. The study owns actor/config jobs and compares original metric tolerances; this producer remains generic.

## Invocation

From `LTX-2`: `conda run -n ltx python -m scripts.onestep_avatar.decode_saved --jobs /absolute/jobs.json --output /absolute/output --gpu-id 4 --seed 42 --model 2.5`.
