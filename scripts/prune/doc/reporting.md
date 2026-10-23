# Pruning report and wording rules

Read the [glossary](glossary.md) for shared terms.

## Scope and references

Apply these rules to pruning reports, figure captions, comparison media and design documents.
The [workspace report rules](../../../../CLAUDE.md#experiment-reports-and-visual-comparisons) define the reader, report structure,
numbers, validity, figures and maintenance. Where this guide differs from them, the workspace rules apply.
This guide adds pruning terms, evidence rules and media conventions.
The [avatar documentation rules](../../onestep_avatar/doc/README.md#documentation-rules) define the shared technical wording.
The [avatar media guide](../../onestep_avatar/doc/media.md#text-inside-the-video) defines comparison titles and labels.

This guide applies those contracts to pruning. Explicit task instructions determine the model, schedule, layout and scientific scope.
Prune's [documentation rules](README.md#documentation-rules) own page structure; the avatar links supply wording and media conventions.

## Wording

Write reports for the reader that the workspace rules define: a researcher who knows the field but not this project.
Use plain technical English. Each sentence states a finding, its evidence, a reason or a limitation.
Keep the words that join evidence to a conclusion, such as `because` and `so`. Short sentences must not drop the reasoning.

Write doc pages, captions and video text in simplified technical English, about 80% to ASD-STE100.
Apply the same wording rules as `onestep_avatar`:

- Write one topic per sentence.
- Keep instructions to 20 words and descriptions to 25 words, where practical.
- Keep paragraphs to six sentences or fewer.
- Use active voice, present tense and the imperative for instructions.
- Keep the articles `a`, `an` and `the`. Do not stack more than three nouns.
- Use one term per concept. Define it before use.
- Explain the reason for each non-obvious rule; sentence length is only one part of clarity.
- Put ordered steps and conditions in lists.
- Preserve exact code names, paths, values and equations.
- Use `must` for a rule and `can` for an option.

Use `video segment` for consecutive encoded frames. Explain `span` when quoting an argument with that name.
Define `c0` as the first-image input with no added noise.
Define sigma, raw energy, normalized score, GT and each comparison arm before using them.
Do not describe projected head energy as causal importance or a percentage of output energy.

Distinguish the original scheduler grid, its eligible tail, selected noise levels and actual model calls.
Round values in report text to two or three significant figures. Use full precision in saved records.
Rounded display values must retain an explicit exact-value source.

## Report structure

Follow the workspace [report structure](../../../../CLAUDE.md#structuring-an-experiment-report):
a title that states the result, a short summary, one results table, brief context, one section per question,
discussion, limitations and an appendix.
Give each section its answer, evidence and interpretation in prose. Do not fill a fixed set of labelled fields per question;
empty fields become filler.

Pruning reports add these rules:

- Each comparison changes one factor against a stated baseline. A recipe comparison names every changed setting and limits its causal claims.
- Report capture error at each noise level beside the pooled value.
  Absolute error grows with noise, so a pooled ratio of means follows the high-noise cases and can hide low-noise losses.
- Claim speed, parity and quality only with the evidence that the [verification guide](VALIDATION.md) requires.
- State the population, weighting and uncertainty method of each descriptive statistic; long definitions go in the appendix.
- The heading, the figure or table and the answer of a section describe the same comparison.
- Mark planned work without evidence as **not run yet**. Keep older run IDs and methods in the appendix.

## Metrics and evidence

Define each metric once, at first use, including its units, observation set and limits.
State the exact denoising schedule and number of model calls.
For uncertainty, identify the estimand, standard-deviation observations, confidence level and independent sampling unit.
Repeated views and sigmas of one actor must not become independent actors.

Prove matched inputs with saved source, context and noise hashes. Equal seeds alone are insufficient.
Include a control that must reproduce the baseline under the recorded numerical policy.
Validate a custom sampling path against the stock path before interpreting its outputs.
Match the stock scheduler call, including its latent argument policy.

Keep latent agreement, decoded quality and execution cost as separate claims.
Include matched capture RGB and capture VAE references when judging identity, action or capture fidelity.
Scope perceptual findings to the clips actually inspected.
When text or conditioning changes, include a mismatched control.
VLM or LLM checklist scores require a ground-truth ceiling under the same scoring procedure.
Treat those scores as supporting evidence and disclose failures to distinguish conditions.

Move invalid or replaced runs to `runs/superseded_<reason>/` with an explanatory README.
Do not cite those runs as current evidence. Keep historical observations in their saved records.
Document a failed or negative result without changing the scientific question.

## Figures, diagrams and video text

Include an image only when it explains setup, supports a conclusion or verifies evidence.
Its caption states that claim first. Put redundant examples in an archive and link its location.

Draw a method diagram only when it shows a data flow that the text cannot; never draw one per section.
Use a small left-to-right Mermaid flow from source to metric, and label the edge changed by the tested factor.
Use the [avatar legend](../../onestep_avatar/doc/core_algorithm.md#7-end-to-end-data-flow).
Render with `npx -y @mermaid-js/mermaid-cli` and inspect the result.

Use one synchronized comparison video per matched case and comparison.
Preserve full source videos. Follow the task's panel layout and keep roles in consistent positions.

Match source frames and playback rate. State shared coverage when lengths differ.
Verify the recorded crop, scale, matte and frame map before interpreting alignment.
Scale and pad panels to equal displayed heights. Preserve aspect ratios and all relevant evidence.

Label missing panels explicitly. Never create replacement evidence during report rebuilding.

Video text uses the same rules as the avatar media:

- Give the video one plain question of 15 words or fewer.
- Give each panel its input or output role.
- Put the changed factor and actual value on a second title line when relevant.
- Put shared settings and long exact schedules in the caption and saved record.
- Show a shared time label only when time helps answer the question.
- Keep paths, hashes, seeds, checkpoint filenames and tensor shapes in records, unless one is the tested factor.
- Put metric scores in figures. Do not label a panel `better` or imply a conclusion in its title.

Save the question, panel order, titles, frame maps, padding, display size and poster frame in rendering records.
Inspect the figures, videos and report at wide and narrow widths.
Check label readability, alignment, distortion and overflow. Link checks alone do not verify layout.

## Saved report ownership and verification

Model execution, reusable metrics, decoding and comparison-media production belong in the LTX-2 package.
Code under `expr/` reads saved artifacts for report sections, captions, plots and validation.
It must not load models, decode latents, start queue jobs or repair missing scientific results.
This matches the [avatar report boundary](../../onestep_avatar/doc/architecture.md#saved-report-ownership).

Write the final report by hand. A CPU publisher can refresh a progress report during a live run.
It must not impose a fixed field template, because validators check facts, not prose layout.
Keep figure scripts with the study output; each rebuilds its figure from saved results as PNG, PDF and plotted-data JSON.
Add every quoted number to the output root's checker, `<output root>/check_reports.py`; for LTX-2.5 the root is `expr/refiner_prune/2.5/`.
The checker recomputes each number from saved results. Run it after each report change. Keep the study plan current.

The checker and any publisher check coverage, identities, metrics, plotted data, references and media links.
They must reject missing evidence before a report is published.

Verify the report boundary with a worked check:

1. Rebuild with complete saved inputs and model, VAE and queue loaders blocked.
2. Remove one required input from the test fixture.
3. Require a named missing-file error and zero model, decoder or queue jobs.

A validation record, schema check or passing unit test does not establish native quality.
Use the [pruning verification gates](VALIDATION.md) for the scientific claims being made.
