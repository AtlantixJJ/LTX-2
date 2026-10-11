# `core/session.py` — model bootstrap and lifetime

## Objective

Select the model and prompt context. Own transformer and decoder lifetimes.

## Data flow

`open_session` runs preflight, selects `cuda:<gpu_id>`, and obtains cached BF16 video context.
It returns a frozen `Session`.
The transformer context builds a video-only `DiffusionStage` with the probed VAE factors.

## Organization logic

`DTYPE=torch.bfloat16` owns weight and latent dtype.
`resolve_prompt` uses the stripped UTF-8 prompt file, explicit text, or `DEFAULT_PROMPT`, in that order.
The parser makes the explicit options mutually exclusive.

`Session.transformer` accepts a checkpoint override, video tools and optional LoRAs.
LoRAs fuse during loading. The context runs under `no_grad` and clears the CUDA cache on exit.
`Session.decoder` delegates VAE lifetime to the adapter.

Worked example: `--prompt ""` selects empty text. Omitting both prompt options selects `DEFAULT_PROMPT`.
Calling `open_session` without its explicit `prompt` argument also selects the default.

## Invariants

- The session defines no schedule or window geometry.
- Callers select model inputs and schedules.
- Native D0 consumers reject LoRA runs, although the shared loader supports LoRAs.
- Sigma and token timesteps follow the float32 [algorithm contract](../core_algorithm.md).

## Gotchas

Pass the result of `resolve_prompt` to `open_session`; prompt selection is not automatic.
A gradient estimator must explicitly enable autograd inside the normally gradient-free context.

## Tests

`tests/test_session.py` checks dtype ownership and GPU lifetime.
`scripts/onestep_avatar/tests/test_prompt_and_whole_clip.py` checks prompt selection.
