# MobileWan numerical head statistics

## Objective

Measure squared projected self-attention energy and allocate global removal masks.
This helper supplies no model runner or complete experiment result.

## Data flow

`projected_energy` receives gated activations `[batch,tokens,heads*width]`, projection weights and explicit token indices.
It selects one head's columns and constructs the head-local Gram matrix `W_h.T @ W_h` in float32.
It computes `a_h.T @ Gram_h @ a_h` in bounded token chunks with batched head products.
Each Gram matrix uses the original head-local GEMM layout.
Invalid-energy flags accumulate on the device and are checked after all chunks.
FP32 batched products can differ from scalar products by rounding; native gates validate the revised arithmetic.
It sums energies in float64 and divides by the batch and selected-token counts.
Output bias and other heads do not enter this energy.

`cohort_statistics` receives `[view,sigma,head]` energies for one block and one cohort.
It averages raw energies across sigmas for each view before calculating pooled statistics.
Every view receives equal weight, independent of clip length.
The first result row is pooled. Later rows follow the input sigma order.

## Organization logic

The bootstrap samples sorted actor identities with replacement.
Each sampled actor contributes all its views, with multiplicity.
The replicate mean divides summed energies by the sampled view count.
Each replicate recomputes its own block mean for normalization.
The same actor order, seed and replicate count give paired draws across blocks.
The result records the actor order, RNG and little-endian int64 draw hash.

`global_masks` sorts by score, numeric block index and original head index.
It traverses one order and skips the last remaining head in each block.
Each budget removes `floor(fraction*total+0.5)` heads from that constrained order.
Returned masks are nested, even when callers request budgets out of order.
Retained and removed indices use original head identities.

## Invariants

Activations must already include the native gate. Do not apply it twice.
All heads use exactly the same unique token indices.
Native callers must disable TF32 and keep the paired numerical policy fixed.
Only pooled train scores may enter mask allocation.
The helper does not enforce train membership; the dataset runner must enforce it.
Cross-attention is outside this score and must remain fully retained in complete mask families.

Raw sample standard deviations use `ddof=1` across views.
Normalized dispersion divides raw dispersion by the fixed cohort block mean.
Intervals use actor-cluster percentiles and describe pointwise means.
They do not describe individual-view ranges or seed variability.

## Gotchas

A Gram matrix retains within-head covariance. A product of scalar activation RMS and weight norm does not.
With activation `(1,-1)` and output weight `(1,1)`, the exact energy is zero.
Zero block means, nonfinite energies and infeasible budgets fail explicitly.
For ten heads and removal fraction 0.23, the budget removes two heads.
An otherwise low-ranked final head remains to satisfy the allocation constraint.

## Tests

`tests/test_mobilewan_heads.py` compares independent direct projection with chunked Gram energy.
It checks cancellation, full generated-token coverage at both lengths, independent actor resampling,
normalization inside bootstrap, stable ties, nested budgets and failure cases.
Native transparency, throughput and dataset coverage remain separate gates.
