# `data/prompt_cache.py` — cached video text context

## Objective

Build and verify text conditioning shared by pruning and avatar consumers.

## Usage

A human or agent can run this module directly from `LTX-2` in `ltx`.
Use one free GPU for text encoding. There is no `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.data.prompt_cache \
  --model 2.5 --gpu-id 0 --verify
```

| Argument | Meaning |
|---|---|
| `--model` | Registered key; default 2.5 |
| `--gpu-id` | Encoding device; default 0 |
| `--verify` | Re-encode and compare exact tensor values |

The CLI always uses `session.DEFAULT_PROMPT`. It has no prompt-selection arguments.
Outputs are `../expr/refiner_prune/prompt_cache/prompt_ctx_<key>_<digest>.pt`
and, with verification, `../expr/refiner_prune/<key>/prompt_cache_check.json`.
See [run guide](../usage.md#preflight-and-prompt-cache) for explicit empty-text cache construction.

## Data flow

`get_or_build` loads an existing tensor or runs `PromptEncoder` once.
It stores video encoding on CPU and returns it on the requested device and dtype.
`verify` loads the old cache, forces a fresh encoding and compares with `torch.equal`.

## Organization logic

The cache name contains the model key and the first eight SHA-1 characters of prompt text.
`force=True` replaces the cache.
Verification reports shape, dtype, file SHA-256, exact equality and maximum absolute difference.
The CLI exits 1 when equality fails.

Worked example: empty text and `DEFAULT_PROMPT` use different cache names.
A cache hit avoids a text-encoder load.

## Invariants

- Consumers pass the actual selected text.
- Native D0 reconstruction verifies actual context bytes against the saved baseline.
- Exact equality, rather than a tolerance, defines cache verification.

## Gotchas

The cache name does not include component hashes. A changed encoder needs fresh verification.
Verification replaces the cache even when the old and new values differ.
The CLI cannot construct the empty-text cache used in the worked pruning workflow.

## Tests

Avatar's `test_prompt_and_whole_clip.py` checks prompt selection.
Native cache verification needs a real text-encoder pass on a free GPU.
