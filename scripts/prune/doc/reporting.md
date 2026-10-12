# Pruning report and wording rules

## Scope and references

Apply these rules to pruning reports, figure captions, comparison media and design documents.
The [workspace report rules](../../../../CLAUDE.md#structuring-an-experiment-report) define the shared report structure.
The [avatar documentation rules](../../onestep_avatar/doc/README.md#documentation-rules) define the shared technical wording.
The [avatar media guide](../../onestep_avatar/doc/media.md#text-inside-the-video) defines comparison titles and labels.
This guide applies those contracts to pruning. Explicit task instructions determine the model, schedule, layout and scientific scope.

## Wording

Write in simplified technical English, about 80% to ASD-STE100.
Apply the same wording rules as `onestep_avatar`:

- Write one topic per sentence.
- Keep instructions to 20 words and descriptions to 25 words, where practical.
- Keep paragraphs to six sentences or fewer.
- Use active voice, present tense and the imperative for instructions.
- Keep the articles `a`, `an` and `the`. Do not stack more than three nouns.
- Use one term per concept. Define it before use.
- Put ordered steps and conditions in lists.
- Preserve exact code names, paths, values and equations.
- Use `must` for a rule and `can` for an option.

Use `video segment` for consecutive encoded frames. Explain `span` when quoting an argument with that name.
Define `c0` as the first-image input with no added noise.
Define sigma, raw energy, normalized score, GT and each comparison arm before using them.
Do not describe projected head energy as causal importance or a percentage of output energy.
Distinguish the original scheduler grid, its eligible tail, selected noise levels and actual model calls.
Use full precision in saved records. Rounded display values must retain an explicit exact-value source.

## Report opening and questions

Deliver Markdown unless the user requests another format. Use native image and video links.
Open with these separate parts, in order:

1. A short problem statement.
2. A bulleted **Terms** list for every arm and term used in the report.
3. One line of shared setup.
4. Numbered main findings, each linked to its question.

Organize sections by scientific question. Each experimental sub-question changes one factor against a stated baseline.
Use plain question headings of 15 words or fewer. Use defined terms or everyday language.
Put exact settings in the setup, not in question headings. Avoid new jargon, config names and experiment IDs in headings.
Keep older IDs and methods in an appendix. Mark planned work without evidence as **not run yet**.

Write **Common inputs** and **Default config** once per section.
For each experimental sub-question, give:

1. **Short answer** directly below the heading, in one or two sentences.
2. The baseline and changed variable.
3. **Testing variants**, **Output**, **Expectation**, and **Existing coverage and gaps**.
4. **Results** in prose: fixed setup, observed changes, numbers, sample size and limits.
5. A scientific figure in PNG and PDF, with its plotted-data JSON.
6. An inline poster linked to the synchronized comparison MP4.

Do not add experiment-summary tables or a trailing Conclusion that repeats the Short answer.
Native Markdown grids can align comparison images. They do not replace experimental reasoning.
Descriptive statistics sections state their population, observations, weighting and uncertainty method explicitly.
Do not invent an intervention baseline for a descriptive measurement.
The question, baseline, changed variable, variants, figure and answer must describe the same comparison.

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
Its caption states that purpose. Put redundant examples in an archive and link its location.
Give each experiment a small left-to-right Mermaid flow from source to metric.
Label the edge changed by the tested factor. Use the [avatar legend](../../onestep_avatar/doc/core_algorithm.md#7-end-to-end-data-flow).
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

Keep prose in `configs/narrative.json` and layout in `code/report.py` within the study output.
Rebuild the report from checked saved results. Rebuilding must preserve the specified layout.
Run the saved-artifact validator after each report change. Keep the study plan current.
The validator checks coverage, identities, metrics, plotted data, references and media links.
It must reject missing evidence before publishing a completed report.

Verify the report boundary with a worked check:

1. Rebuild with complete saved inputs and model, VAE and queue loaders blocked.
2. Remove one required input from the test fixture.
3. Require a named missing-file error and zero model, decoder or queue jobs.

A validation record, schema check or passing unit test does not establish native quality.
Use the [pruning verification gates](VALIDATION.md) for the scientific claims being made.
