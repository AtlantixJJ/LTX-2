# `data/whole_clip.py` — saved native D0 input contract

## Objective

Reconstruct exactly the noised whole-video input used in a saved baseline D0 run. Calibration uses the baseline alone; a candidate is needed only for a paired comparison.

## Data flow

Read capture tensors and frame rate through `onestep_avatar.dataset.load_training_master`.
The dataset reader checks the continuous-master schema, nonempty floating shape,
and finite positive frame rate. It imports no training CLI. This replaces the
private trainer reader without changing reconstructed input values.

`load_manifest` validates the one-step white-capture, full-bidirectional run, BF16 dtype, finite sigmas in (0,1], integer seed, absent LoRA and unguided single-pass CFG 1/STG 0. Unsupported guidance is rejected rather than silently reconstructed differently. `build_input` checks capture content, BF16 shape, fps, VAE fingerprint and actual prompt-context bytes, loads content-verified finite BF16 epsilon, verifies the token shape and saved geometry, and rebuilds the clean-first-frame modality. It returns the grid, modality, clean first-frame tokens and source row. `verify_pair` adds candidate-versus-baseline setup checks; `verify_saved_noise` compares actual epsilon tensors after independently verifying each file's recorded hash. `latent_path` verifies the saved output hash before returning its path.

`native_provenance` stamps the full calibration-manifest SHA256, seed, VAE,
context, guidance, geometry, dtype and calibration capture/noise identities.
`validate_native_provenance` checks these against the pinned manifest and, when
supplied, the selected baseline. Older masks without these pins need recalibration.
`actor_identity` resolves aliases and uses the bare actor ID from DNARender
`meta.json` above `views/`. Without metadata it recognizes `Part_N/actor_clip`
directory names, normalizing leading zeros. Another clip, Part, view or symlink
of that bare actor cannot serve as held-out validation. Generic fixture names
retain the resolved-directory fallback; existing malformed actor metadata fails
closed. `test_actor_holdout.py` checks this boundary independently of the avatar
producer.

`verify_candidate` checks the saved checkpoint fingerprint and actual safetensors
export task/source identity, then dispatches by the declared pruning family.
Existing width exports retain their mask-hash and complete native-mask checks.
Depth exports require a content-pinned `whole_clip_d0_depth_v1` artifact and
`score.export_depth.verify_export`, including both index mappings, complete
retained tensor inventory, sliced config and exact resident-video accounting.
Unknown families are rejected. This validates architecture/provenance, not
payload equality or numerical parity: run the separate export gate before
accepting output quality or deployment.

## Invariants and checks

Frame 0 is conditioning, not a predicted frame. Sigmas have exact `[sigma, 0]` schedules. Do not derive fps from defaults or replace the saved epsilon with a new draw. The module has no dependence on a particular candidate checkpoint. `test_whole_clip_d0.py` covers manifest rejection; real forward equality to the saved baseline checks the model-facing tensor contract after changes.

Whole-clip attention uses `attention_mask=None`: every token is visible. Do not
materialize an all-ones token-by-token mask; its conversion to attention bias
allocates quadratic temporary tensors and can exhaust a 48 GB device at 145 frames.
Sigma and per-token timesteps remain float32, matching the stock pipeline schedule;
BF16 describes the weights and latent tensors, not a rounded schedule.
