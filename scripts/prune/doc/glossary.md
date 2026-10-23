# Pruning glossary

Use these names in concepts, commands and source docstrings.
Start with the [overview](overview.md), then read the [algorithm](core_algorithm.md) for calculations.
Study labels belong to saved reports; package recipes use the exact configured arm name.

## Inputs and coordinates

- **D0**: add noise to capture latents and predict the clean capture. This is the native pruning task.
- **D1**: add noise to guide latents instead. The baseline producer supports it as a comparison; pruning calibration uses D0.
- **Capture**: the recorded target video, encoded with the selected video VAE.
- **Guide latents**: encoded pose/body-render conditioning used by the avatar corpus and the producer's optional D1 arm.
- **c0**: the first latent frame, inserted without noise as the image condition and restored in the output.
- **Sigma** (`sigma`): noise-mix weight for one direct `[sigma, 0]` step. **Epsilon** (`epsilon`) is the saved noise tensor.
- **GT**: ground truth. `monitor_reference=gt_capture` compares predictions with the recorded capture, using no teacher cache.
- **White objective** (`objective=white`): use the corpus capture composited on a white background, in its recorded crop coordinates.
- **F18 / F28**: 18 or 28 consecutive temporal VAE latents. With temporal factor eight they decode to 137 or 217 pixel frames.
- **Full source**: the complete saved clip before optimizer cropping. `uniform_f18` chooses one continuous 18-latent training window.
- **Held-out actor**: a person excluded from calibration and learning across clips, Parts, views and filesystem aliases.

## Model and selection

- **Head**: one attention head. **FFN channel**: one intermediate feed-forward activation before the down projection.
- **`attn1` / `attn2` / `ff`**: video self-attention, video text cross-attention and video feed-forward branches.
- **Native head gate**: the model's own `to_gate_logits` modulation. It is distinct from a learned pruning score.
- **Functional mask**: zero selected head/channel activations immediately before their output projection, without rewriting weights.
- **Score / pruning gate**: an FP32 trainable value and the amplitude derived from it. The chosen method defines that conversion.
- **Legal count**: an allowed number of kept heads in a block, supplied by `allowed_counts` in the frozen head config.
  The legal selector chooses block counts that meet the global budget, with stable original-ID ties.
- **Original ID**: a unit's position before slicing. It determines retained parameter rows, positional coordinates and optimizer-state transfer.
- **Event**: a joint-selection decision after one configured score-observation window.
- **Give-back**: the part of a score's peak movement reversed before the event. See the [equations](core_algorithm.md#joint-clipped-selection-with-return).
- **Re-admission / return**: changing a removed unit back to kept. A returned unit removed again becomes permanent during selection.
- **Alignment**: round each retained FFN width upward to a configured multiple by restoring removed channels before recovery.
- **LoRA / adapter**: a trainable low-rank update to a frozen projection. Recovery trains these weights with fixed unit choices.
- **DDP**: distributed data parallel execution. Each rank stores a full backbone and contributes one loss to one averaged gradient update.

## Formats and evidence

- **V3**: the joint clipped-score method implemented by `learn.readmission`, using format `joint_clip_reset_v3`.
  It supports the arms `joint_clip_reset_v3`, `heads_clip_reset_v3` and `budget_joint_v3`.
- **V2**: a study-generation label in saved records. Use its recorded format and source snapshot for reproduction.
- **Receipt**: a saved evidence record with its inputs, identity and result. Immutable receipts permit identical replay; mutable status files are separate.
- **Envelope**: export metadata that binds a checkpoint or split bundle to its source study, retained IDs, state and tensor hashes.
- **Candidate family**: a format dispatch name, such as `width`, `depth`, `width_recovered`,
  `width_head_ffn_recovered` or `width_joint_readmission_recovered`.
- **Export parity**: agreement between a declared functional intervention and its exported execution on the same saved input.
- **Split bundle**: compact base weights plus separate adapter tensors and an envelope; its loader restores trained adapter arithmetic.
- **RMS proxy**: sampled activation root mean square multiplied by an output-projection norm.
- **Projected energy**: the mean squared norm of one head's projected vector on specified query tokens; it is not a causal-importance measure.
- **Budget units**: FFN-channel compute units used by `learn.readmission.budget_projection`.
  Its `cost` argument charges one head as 704 channel units; the model has 48 × 32 = 1,536 video self-attention heads.
- **152-head budget**: a `removed_heads` choice in the head preset, leaving 1,384 original heads across all blocks.

## Reading saved study labels

These labels explain existing records. Use descriptive names in new package prose.

- **A / B / C**: plan-specific arms. In the joint clipped study they denote joint clipped selection,
  head-only clipped selection and budget-matched selection. Other plans can reuse these letters.
- **r3**: a saved run suffix for a particular parent input/checkpoint set; it is not an algorithm or a format.
- **P0 / P1 / P2**: optimizer replay, native channel ablations and export-isolation evidence in the joint-study CLI.
  Their legacy flags and receipt keys remain readable. Audit plans can reuse the same labels for other phases.
- **G1 / G2 / G3 / G4 / G5**: the package's [five evidence gates](VALIDATION.md#evidence-gates).
  Saved joint reviews also used G1 for update-100 allocation and G2 for terminal alignment.
  New joint review records use `allocation_review` and `alignment_review` to remove that ambiguity.
- **White capture**: the capture under the white objective; it does not imply unmodified original RGB.
