# `decode_saved.py` — decode saved encodings

## Objective

Decode saved latent outputs to video, and run controlled decoder comparisons.
The module runs no transformer. It gives reproducible presentation media and
RGB evidence.

The module entry calls the shared queue runtime helper. Outside a queue, the
helper calls the public operation unchanged. In a queued child, it measures one
local resource phase and records any declared numerical policy. The queue owns
the launch, the supervision and the proof that no worker remains. This module
starts no child. See [runtime](execution/runtime.md).

## Usage

Humans can use the foreground queue or run this module directly.
See [launch, output and stop commands](usage.md). Agents use the queue.
The queue recipes below describe managed execution.

Start decoding as a queue `decode` job. The queue selects the GPU, and the
child uses local device zero. See
[run guide §8](../configs/README.md#8-decoding-comparison-media-cost-and-plots).

This is the agent path. A person can also run the same module directly, or start the queue in the foreground; see [usage](usage.md).

```json
{"id": "decode_e3", "kind": "decode",
 "arguments": ["--jobs", "/abs/decode_jobs.json", "--output", "/abs/out/decode_e3",
               "--seed", "42", "--model", "2.5"],
 "output": "/abs/out/decode_e3",
 "completion": {"manifest": "/abs/out/decode_e3/manifest.json"}}
```

| Argument | Meaning |
|---|---|
| `--jobs` | Required. The decoder job list (JSON). |
| `--output` | Required. The output directory. |
| `--seed` | The default decode seed. A job can set its own `seed`. |
| `--model {2.3,2.5}` | The VAE of this model version. |
| `--gpu-id` | Local GPU for direct CLI; assigned by the queue in a job. |

The decoder job list has a `jobs` list and an optional `comparisons` list:

```json
{"jobs": [
   {"id": "cached", "latent": "/abs/cached_output.pt", "sha256": "<SHA256>",
    "latent_frames": 7, "fps": 30, "seed": 42, "sample_frames": [0, 24, 48]},
   {"id": "recompute", "latent": "/abs/recompute_output.pt", "sha256": "<SHA256>",
    "comparison_required": true}],
 "comparisons": [{"id": "cached_vs_recompute", "left": "cached", "right": "recompute"}]}
```

- `latent` and `sha256` are required. The other job fields are optional.
- `latent_frames` keeps only that many encoded frames from the start.
- `sample_frames` selects the PNG frames.
- A job that a comparison uses needs `"comparison_required": true`.

Check the queue list on CPU with the queue's `--dry-run`
([queue usage](execution/queue.md#usage)).

Outputs in `<OUT>`: one MP4 for each job, sampled PNG frames, difference PNGs
for each comparison and `manifest.json`.

## Data flow

The job list binds absolute latent paths, SHA-256 values, an optional prefix
length, the decode seed, fps and comparison IDs. The session decoder and the
public `media.decode` make the frames. `manifest.json` records the source,
input and output hashes, the VAE identity, the runtime, and difference metrics
for each frame. The metrics use uncompressed pixels.

## Organization logic

### Software and session

The module records the shared decoding software manifest before it reads the
jobs. It checks the manifest again before it opens the decoder and before each
manifest publication. Every decoded row and comparison records it.

The native decoding settings come from `media.native_decoder_settings`. Queue
verification uses that same identity. It keeps no second copy. The session
comes from `media.open_decoder_session`. It does the shared CUDA check and uses
a null text context. It prepares no prompt embeddings and opens no text encoder
or transformer.

`parse_args(argv=None)` gives the argument contract without reading jobs or
opening a model. Queue validation uses this parser. Then `main` does the
decoding with the checked arguments.

### Decoding

`media.decode` owns a fresh identical generator, so all outputs decode the same
way. The job producer only selects saved tensors and lengths. It owns no
rollout, noising, conditioning or cache logic. Named comparison jobs keep their
CPU decoder outputs in memory until the pixel differences are measured.

### Reuse

For each job, the module does these steps:

1. Verify the source hash and load the saved tensor.
2. Apply the requested encoded-frame prefix.
3. Calculate `media.decode_key` from the saved file hash, the full VAE hash,
   the tensor dimensions, the native decoder method, the seed and the actual
   settings (bf16 conversion, no tiling, a fresh generator, the Torch version).
4. Reuse a saved result only when all of these match: the key, the complete
   job, the producer hash, the MP4 hash and every recorded PNG hash.

Results from older manifests without a key are decoded again. So a changed VAE
or a missing poster causes a new decode, even when the saved tensor and the MP4
are unchanged. Comparison jobs always decode, because their measurements need
uncompressed pixels.

Reuse also requires a current saved software manifest. If a cached row has no
software record, or a stale one, the module keeps it. It then requires a fresh
destination and never overwrites saved evidence. Read-only reuse returns false
for such a row.

## Invariants

- No transformer context opens.
- Input hashes must match before rendering.
- No full rollout is rebuilt.
- Numerical differences use the decoder's float frames, before video encoding
  or PNG quantization.
- Difference PNGs use an explicit 20× gain.
- The output quantization is documented.

## Gotchas

- A length change changes the decoder context and the noise shape together.
- The foreground is a derived union threshold mask. It is not the corpus mask
  or a training mask.
- Jobs that a comparison needs are decoded together, so all numerical inputs
  are in memory. These jobs do not claim a cached numerical result.
- Saved frames are examples. The numerical curves cover every shared frame.

## Tests

Check `--help` from the LTX-2 root in `ltx`, and run the package suite.

- Reuse checks must refuse changed decoder keys, old manifests, changed MP4s
  and missing or changed PNGs.
- A decode smoke job must verify the input and output hashes, the expected
  frame count and a deterministic repeat.

The study owns the actor and configuration jobs and compares the original
metric tolerances. This producer stays generic.
