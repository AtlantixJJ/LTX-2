# `backbone.py` — which transformer weights a run loads

## Objective

Resolve the LTX transformer file for a `(model version, backbone variant)` pair once, for
training, the probe and deployment alike, and give it an **identity** an adapter can be
checked against. `model_registry.resolve("2.5")` defaults to the distilled transformer; a dev
adapter trained, probed or deployed on that file would be silently wrong.

## Data flow

`transformer_path(key, variant)` returns the registry's distilled default or the dev file
beside it (`DEV_TRANSFORMER`). `resolve(key, variant)` hands that path to the registry's
existing per-component `transformer_path` override, so the text encoder, VAE, sigmas and
scale factors still come from the version's defaults; there is no second loader.
`identity(path, variant, key)` returns `model_key`, `base_variant`, the file name and
`provenance.checkpoint_fingerprint` (size + safetensors header + sampled bytes, fast on 42 GB).

Consumers: `train.py --variant` stamps the identity into every checkpoint
(`onestep_avatar_base_*`) and `config.json`; `visualize_d1.py` resolves `--variant dev` here
and passes the identity to `sampling.check_adapter_conditions` before loading.

## Organization logic

Model version (`--model`) and backbone variant (`--variant`) are separate axes. The file
identity, not `model_key`, is what an adapter is bound to: `model_key=2.5` names a
generation, not a set of weights.

## Invariants

- One spelling of the dev file name (`DEV_TRANSFORMER`); `visualize_d1` no longer carries its own.
- A dev adapter is refused on distilled weights by fingerprint, not by name.

## Gotchas

`train.py --variant` defaults to `distilled` so the historical recipes keep their meaning;
every dev recipe must pass `--variant dev` explicitly.

## Tests

`tests/test_train.py::test_checkpoint_metadata_stamps_base_identity_and_full_subset_hash` and
`::test_condition_reader_refuses_a_dev_adapter_on_distilled_weights`.
