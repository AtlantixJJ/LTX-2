# Saved native-gate report supplements

## Objective

`score/native_gate_review.py` completes the per-sigma evidence in a native-gate report.
It reads audited statistics only. It does not change the collection snapshot or existing milestone arrays.

## Data flow

Read the passing saved-prefix audit, analysis arrays, frozen examples, mask references and historical events.
Require saved per-head means to equal means reconstructed from the audited sufficient statistics.
Write supplemental figures, original-head records and event correlations in the milestone's `review/` directory.
Update the narrative atomically after all figure companions exist.
The collection owner's existing report generator reads this narrative on its next rebuild.

```mermaid
flowchart LR
  S("audited saved statistics") --> G["per-sigma comparisons"]
  G --> F(["figures and head records"])
  F --> N["atomic narrative update"] --> R(["Markdown report"])
  classDef tensor fill:#dcf5df,stroke:#43854b,color:#173b1c;
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class S tensor;
  class G,N proc;
  class F,R out;
```

## Organization logic

`density_overlays` keeps the first milestone's minimum, median and maximum example IDs.
It overlays the first, preceding and current distributions on the same fixed bins.
For example, step 300 overlays steps 100, 200 and 300 without choosing new heads.
A changed example ID stops the reader with a named error.

`identity_figures` draws each sigma's global-count and within-block-count lists separately.
`head_records` joins the binary masks and native lists by original block/head ID.
The head-record file declares its score-group order: four source sigmas, then the equal-sigma score.

`event_correlations` retains every historical event's eligible population.
It computes the native-mean/drop correlation separately at each sigma and for the equal-sigma mean.
`drop_figures` displays the early, middle and final selection events in separate figures.
For two noise groups with opposite rankings, correlations can be +1 and -1 while the aggregate is undefined.
This is why an aggregate cannot replace noise-specific comparisons.

`sampling_coverage` compares cumulative and disjoint-window actor/view counts after step 100.
The report states exact current per-sigma coverage beside the noise comparison.
It retains repeated views as training-sampler observations, not independent subjects.

`write_narrative` starts from `configs/narrative_design.json` on each rebuild.
It adds current results and captions to `configs/narrative.json` without duplicating previous prose.
Each result names its source prefix. Missing stability labels stay unavailable before step 400.

## Invariants

- No model, torch, decoder, queue or avatar owner is imported.
- A passing matching prefix audit is required before supplemental analysis.
- The native source stream, frozen head examples and saved binary masks remain unchanged.
- Every figure has PNG, PDF and plotted-data JSON companions.
- `review/complete.json` pins the supplemental outputs after publication.
- Historical drop correlations never fabricate a final continuous pruning score.
- Agreement with a saved mask is separate from removal safety or output quality.

## Gotchas

Use a separate saved-reader snapshot while collection runs from frozen code.
The source snapshot used by GPU workers must not change to add report views.
The observer's generated-token histograms retain their original 2,048 bins.
Density displays combine adjacent bins for readability; they do not change the saved distributions.
Full-size PNG and PDF links preserve readable chart labels on narrow screens.
The report narrative can briefly lag a newly published milestone while its independent audit finishes.
Every supplemental caption states its own source step to prevent a false coverage claim.

## Tests

`tests/test_native_gate_review.py` checks opposite sigma rankings and eligible event populations.
It checks original IDs, distinct global/block lists, fixed example rejection and exact saved-mean validation.
An end-to-end saved fixture checks prefix overlays, coverage and repeated narrative rebuilds.
An isolated import control verifies the model-free boundary.
All fixture artifacts stay in temporary test directories.
