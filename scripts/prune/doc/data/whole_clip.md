# `data/whole_clip.py` — saved native D0 inputs

## Objective

Reconstruct the exact whole-video input used by a saved D0 baseline.
Calibration reads the baseline alone. A paired comparison also reads a candidate.
The [glossary](../glossary.md) defines the saved-input and export terms.

## Data flow

`load_manifest` reads and validates `manifest.json`.
The native contract requires white capture, full bidirectional attention, BF16 latents and unguided single-pass inference.
CFG is 1, STG and rescale are 0. There is no LoRA checkpoint.
Sigmas are distinct finite values in `(0,1]`.
Each view/sigma row has exact schedule `[sigma,0]`.

`build_input` reads the capture through `corpus.dataset.load_training_master`.
It verifies the capture hash, finite BF16 shape, fps and selected VAE fingerprint.
It verifies saved epsilon content, dtype and token shape.
It rebuilds the grid and checks recorded geometry and one-block coverage.

The prompt text hash and actual context-byte hash must match.
Mix capture and saved epsilon, insert clean c0, then assemble the modality.
Use `attention_mask=None`, float32 global sigma and float32 token timesteps.
Return the grid, modality, clean first-frame tokens and manifest row.
`load_case` verifies each small capture/noise file buffer and deserializes that same buffer once.
It returns the reconstructed modality and those verified capture/noise tensors to learning and evaluation callers.

Both public paths use `_reconstruct_input`; no second noising or conditioning implementation exists.
`source_grid` verifies a capture and builds its grid for media without creating noise or a model input.

## Organization logic

`records` indexes rows by view and sigma.
`latent_path` requires one matching D0 artifact and verifies its full file hash.
`load_epsilon` verifies the noise file before loading.
The bytes in its verified buffer are the bytes passed to `torch.load`, even if the source path changes afterward.
`verify_saved_noise` compares actual epsilon tensors after independently checking both hashes.

`native_provenance` pins the calibration manifest SHA-256 and selected distribution.
It includes seed, context, guidance, geometry, dtype, model/VAE fingerprints and calibration capture/noise identities.
`validate_native_provenance` reconstructs that stamp from the pinned manifest.
When given another baseline, it checks matching settings and the presence of all pinned calibration rows.

`actor_identity` resolves filesystem aliases.
It uses DNARender `meta.json` above `views/`, or the normalized bare actor from `Part_N/actor_clip`.
Another clip, Part or view of that actor cannot serve as held-out data.
Malformed existing actor metadata raises. Generic fixtures use the resolved directory fallback.

`verify_pair` checks fixed settings, source identities and row coverage.
It requires different transformer fingerprints.
`verify_candidate` additionally reads the actual export metadata.
Width candidates require the pinned mask and complete native family checks.
The `depth` family requires the pinned artifact, mappings, retained inventory, sliced config and parameter accounting.
Unknown families raise.

The `width_head_ffn_recovered` family validates learned head/FFN envelopes through a deferred import.
`width_joint_readmission_recovered` also dispatches to its joint export validator; see `data.whole_clip.ENVELOPE_FAMILIES`.
It checks final masks, consumed inputs, original actor scope and every merged tensor hash.
Disposable probe envelopes fail candidate qualification. [Verification](../VALIDATION.md) defines the remaining evidence gates.

Worked example: three latent frames at height and width two produce 12 tokens.
Tokens 0–3 are c0. The saved block plan is `[[0,3]]`.
Every token is visible; only the first four have zero token timesteps.

### Capture-only scoring inputs

`prepare_score_inputs` saves input evidence without producing predictions or a baseline rollout.
It verifies audited capture bytes, full shape, fps and the encoding record.
RGB and clip metadata hashes are checked only when their optional pins occur in the source record.
It uses the public corpus reader and the common seeded epsilon helper.
It saves epsilon once and publishes a capture-only input manifest atomically.

A completed input manifest reuses its pinned epsilon. Orphan files without a manifest fail for diagnosis.

Every sigma uses `_reconstruct_input` through `build_input` or the validated `load_case` path.
The input manifest declares `whole_clip_d0_inputs_v1`; it does not claim saved rollout completion.
Shared task and guidance fields come from `core.native_manifest.fields(input_only=True)`.
The tiny dataset-worker tests verify exact saved-noise reuse and first-frame conditioning across sigmas.

### Recovered width candidates

`verify_candidate` dispatches `width_recovered` to `checks.learned_heads` through a deferred import.
The export must pin a matching versioned recovered envelope and full merged tensor inventory.
That validator checks source components, original IDs, geometry wrappers, recovery state, consumed noise and actor separation.
Unknown or incomplete recovered formats fail.
The input owner keeps its original single-seed geometry, FP32 mixing, c0 and guidance rules.

## Invariants

- Capture frame zero is conditioning, not a generated frame.
- Preserve the exact schedule, recorded fps and original saved epsilon.
- Verify actual context bytes, not only prompt text.
- Keep VAE factors and recorded geometry consistent.
- Retain pinned calibration rows in baselines used for candidate validation.

## Gotchas

Different serialization can change file hashes while tensor values remain equal.
The [evidence guide](../VALIDATION.md#g1-reproduce-the-saved-baseline) defines saved-noise comparison.

Full visibility uses `None`, not a dense all-ones token mask.
Converting a dense mask can allocate quadratic attention-bias tensors and exhaust a 48 GB GPU.
BF16 weights do not justify rounding the schedule.

Candidate metadata validation checks identity and architecture.
Use the [export gate](../VALIDATION.md#g3-reproduce-the-functional-export-reference) for numerical agreement.
Use export parity for no-prune controls because paired quality analysis requires different fingerprints.

## Tests

`tests/test_whole_clip_d0.py` checks manifest, noise and cross-task refusals.
`tests/test_actor_holdout.py` checks actor identities and aliases.
`tests/test_input_loading.py` checks one load per tensor, exact modality fields, grid-only media and replacement after a verified read.
`tests/test_export_depth.py` checks depth candidate-family dispatch.
A real baseline forward verifies the reconstructed model input after numerical changes.
