# Native gate milestone analysis and report

## Objective

`score/native_gate_analysis.py` computes saved-statistics comparisons and builds the Markdown report/sketch.
It imports no torch, model, decoder or queue owner. A report rebuild cannot create missing model evidence.
The [study plan](../../../../../plans/2026-10-18-native-head-gate-statistics-vs-pruning.md)
assigns the experiment questions, figure axes and expectations.

## Data flow

A validated milestone supplies cumulative and disjoint-window NPZ statistics.
Read the predecessor's statistics for convergence and the frozen masks/events for pruning comparisons.
Save numerical analysis, all plotted data and PNG/PDF figures under the milestone.
Build the current report from the latest complete numerical analysis and the narrative config.
Preserve all older milestone snapshots.

## Organization logic

E1 plots block/head means, fixed-head distributions and conditional brief-activation examples.
Examples are selected once, at step 100, and stay fixed.
E2 compares successive cumulative means, histograms and threshold fractions.
It also compares new-window histograms to prior cumulative histograms and plots actor/view coverage.
E3 plots noise-specific head responses and pairwise Spearman rank agreement.
E4 compares kept/removed head distributions, native-score lists and historical pre-reset drops.

Histograms use probability mass, not raw count, before distribution comparison.
Wasserstein-1 distance is the area between fixed-bin cumulative distributions, in native gate units.
Mean stability alone cannot establish distribution stability.
A head/sigma pair needs three consecutive passing comparisons before the practical-stability label appears.
Thus step 200 provides the first comparison; step 400 provides the first possible stable label.
Recent-window fluctuations remain visible even when cumulative changes shrink.

The agreement reader gives half-credit for ROC ties and groups score ties for average precision.
Stable original-ID ordering resolves matched-count selection ties.
The block-matched reference uses 10,000 seeded hypergeometric draws, equivalent to random labels
with each block's removed count preserved. It is a conditional reference, not a quality result.

## Figure organization

Mean-gate heatmaps have head ID on x, block ID on y and one common color scale.
Convergence plots have original source steps on x and gate-unit changes on y.
Noise-response plots use the four exact sigmas on x. Correlation matrices use sigma on both axes.
Pruning distributions use per-head means on x and empirical cumulative head fraction on y.
Pruning-agreement curves use source steps on x and separately labeled AUC/AP/recall/Jaccard panels.
All scientific plots save PNG, PDF and plotted-data JSON.
Each report figure names its current milestone and links its data.

## Invariants

- Only complete retained source prefixes enter analysis.
- Each selected sigma has 40 observations per completed 100-source-step window.
- Source step counts, scheduled records, observed draws and skipped records are distinct.
- Four selected noise groups remain separate; the overall score equally weights their means.
- No result curves appear in a sketch without scientific observations.
- Report validation checks saved local links and scientific figure companions before replacing the report.

## Gotchas

A flat cumulative curve can result from dilution of new samples.
Read disjoint-window distances and coverage beside cumulative curves.
Native-score agreement measures similarity to learned masks, not safe removability or perceptual quality.
Noise-specific draws differ in view/crop/noise; E3 is not a paired sigma intervention.
Repeated views and tokens do not provide independent subject counts.

## Tests

Focused tests compare exact ties, perfect/reversed ranks, known distances and model-free sketches.
The full milestone fixture exercises all figure builders and step-200 convergence from saved synthetic test data.
Synthetic fixtures remain in temporary test directories, outside experiment outputs.
