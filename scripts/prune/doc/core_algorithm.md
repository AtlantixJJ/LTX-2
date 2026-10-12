# Whole-clip pruning algorithm

## Terms and shapes

- **D0** noises capture latents and predicts clean capture latents.
- **D1** noises guide latents. It is an optional producer arm, not the pruning calibration task.
- **c0** is the clean capture's first latent frame.
- **epsilon** is the saved random noise tensor.
- **sigma** is the starting noise level for one direct denoising step.
- **Functional mask** zeros a head or FFN channel at its output-projection input.
- **Export parity** compares that intervention with its saved checkpoint execution.
- **Held-out actor** is an actor excluded from calibration across clips, Parts and views.

Dense video latents have shape `[1,C,F,H,W]`.
With patch size one, tokens have shape `[1,F*H*W,C]`.
A frame occupies `H*W` consecutive tokens.
Token `(f,h,w)` has index `(f*H+h)*W+w`.

Use registered VAE scales. The aligned pixel count is `time*(F-1)+1`.
Preserve the recorded fps and source coordinates.

## Input reconstruction

1. Validate the saved native D0 manifest and requested view/sigma row.
2. Check capture, VAE, epsilon and context identities.
3. Build the same grid from capture shape, fps and registered scales.
4. Mix source and noise in float32:
   `x_sigma=(1-sigma)*capture+sigma*epsilon`.
   Cast the result to BF16.
5. Replace the first `H*W` tokens with `c0`.
6. Set those tokens' timesteps to zero. Keep other token timesteps at sigma.
7. Use float32 global sigma and float32 token timesteps.
8. Set `attention_mask=None` for full bidirectional visibility.
9. Run the complete clip as one block with no cache.
10. Restore exact `c0` in the predicted clean output.

BF16 applies to weights, latents and ordinary context.
It does not mean a BF16 schedule.
The session's transformer wrapper returns clean predictions; do not convert them as velocity a second time.

```mermaid
flowchart LR
  Y("capture tokens") --> N["float32 noise mix"]
  E("saved epsilon") --> N
  N --> C["insert clean c0"]
  I("capture c0") --> C
  C --> M["one bidirectional model call"]
  T("text context and float32 sigma") --> M
  M --> P["restore clean c0"] --> O(["BF16 clean latent"])
  I --> P
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef tensor fill:#dcf5df,stroke:#43854b,color:#173b1c;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class N,C,M,P proc;
  class Y,E,I,T tensor;
  class O out;
```

## Worked calculation

Let `F=3`, `H=W=2`, and `C=128`.
There are 12 tokens. Tokens `[0,4)` contain the clean first frame.
The block plan is `[(0,3)]`.

For a later scalar source 2, epsilon 6 and sigma 0.5, the model input is 4.
For a first-frame scalar source 2, replace that mix with 2 and set its token timestep to zero.
The global sigma remains 0.5.

At sigma 1, capture and guide have zero weight in the noise mix.
With identical epsilon, context and c0, D0 and D1 have identical model inputs.
Equal inputs are a useful control; verify saved outputs within the same numerical policy.

## Calibration and interventions

For each calibration view/sigma, reproduce the saved baseline before accepting scores.
The scorer rejects nonfinite maximum differences and differences above 0.02.
It samples generated-frame tokens and observes activations before output projections.

The default proxy is sampled activation RMS multiplied by the corresponding projection norm.
Average squared activation means equally across calibration cases before taking RMS.
Remove low scores independently in each branch, with stable ties and one retained unit minimum.

Head masks multiply a complete head's channels before `to_out[0]`.
FFN masks multiply post-activation channels before `ff.net[2]`.
Checkpoint export must preserve coupled projections, normalization and positional identity.
Depth parity instead uses an ordered retained block list.

## Metrics and their limits

Quality metrics exclude clean latent frame zero.
The direction is `d=(x_sigma-predicted_x0)/sigma`.
Relative L2 is `norm(d_candidate-d_baseline)/norm(d_baseline)`.
Cosine compares direction angle. Capture MSE compares predicted and capture latents.
These metrics do not prove identity, motion correctness or decoded quality.

Export parity uses maximum absolute latent difference, with default tolerance 0.02.
A passing result proves agreement with the declared intervention, not agreement with the unpruned teacher.

Timing uses warmed, synchronized transformer calls.
Speedup is `baseline_ms/candidate_ms`.
ABA/BAB orders repeat the outside arm to expose drift.
Storage size and parameter counts require their own denominators.

## Diagram legend

Use a left-to-right flow.
Rounded input nodes hold tensors, rectangular nodes hold operations, and output nodes hold saved artifacts.
Use green for tensors, blue for operations and purple for outputs.
This follows the [avatar legend](../../onestep_avatar/doc/core_algorithm.md#7-end-to-end-data-flow).
Diagrams explain data flow. Saved hashes and numerical checks establish evidence.

## MobileWan squared projected energy

The optional `mobilewan_heads` helper uses executed gated head vectors and head-local projection Gram matrices.
It computes mean squared projected vector norm over declared generated tokens.
Average raw energies over sigmas and equal-weight views before dividing by the cohort block mean.
Actor-cluster bootstrap draws retain all views and sigmas, and recompute normalization in each replicate.
Global allocation uses ascending pooled train scores, original block/head ties and one retained head per block.
This helper supplies no native execution evidence yet.
