# Whole-clip structural pruning methods

## Objective

Preserve the native full-bidirectional prediction while reducing measured forward
latency, subject to held-out direction and decoded-quality checks. Every input
uses one clean capture latent frame and one exact `[sigma, 0]` step over the
complete video. Scores exclude frame 0. Calibration and validation actors differ.

## Implemented activation screen

An FFN channel's score is post-activation RMS times its output-column norm:

\[
I_j = \sqrt{\mathbb{E}[a_j^2]}\,\|w_j\|_2.
\]

For a head of width \(d_h\), the factored proxy is

\[
I_h = \sqrt{\mathbb{E}[\|a_h\|^2]/d_h}\,\|W_h\|_F.
\]

The scorer samples space across every generated latent frame and weights each
calibration view/sigma forward equally. It averages squared statistics before
taking the square root. `fractional_masks` removes the lowest scores per branch,
uses stable ties and keeps at least one unit. Actual removal counts reflect
rounding to complete heads/channels.

This local proxy ignores within-head covariance, cancellation between units,
residual scaling and downstream amplification. It is a shortlist heuristic;
held-out ablation and decoded quality establish its usefulness.

`exact_local_head_energy` provides the alternative local score
\(\sqrt{\mathbb{E}\|W_h a_h\|^2}\) on explicitly selected tokens. It retains
within-head covariance but is still not final-output deletion cost. The default
CLI uses the factored proxy.

## Export semantics

Removing an FFN channel couples an input-projection row/bias and output-projection
column. Attention Q/K normalization spans original heads, so exports retain full
Q/K projections and normalization before selecting heads. V, gate and output
columns can be sliced. Retained RoPE identities must match original heads.

`masked_full` applies full-width masks. `sparse` selects active attention heads
with full-width projections. `compact` reduces supported tensor shapes.
`compact_faithful` stores sliced parameters but restores original GEMM geometry
at execution. Shape changes can change BF16 reductions: validate numerical parity
before interpreting output quality. Storage reduction and execution speed are
separate measurements.

## Next experiments

Compare activation ranking with direct grouped ablation on matched native inputs.
Measure actual supported group savings, then allocate using held-out damage and
positive measured time savings. Re-score and remeasure after changes because
latency and quality are not additive. These are proposed experiments, not
implemented selection strategies or established LTX results.
