# prepare_ltx_prompts.py

Run from the LTX-2 repository root in the `ltx` conda environment with
`python -m scripts.ltx_prompts.prepare_ltx_prompts`. Captioner backends, media conversion, and trace
rendering live in `backends.py`, `media.py`, and `traces.py`.
The default `--env-file` is `.env` in the caller's directory; `LTX_ENV_FILE`
or `--env-file` overrides it. Resume context now hashes all four source
modules, invalidating records produced before this source split.

This standalone command is a copy of LTX Trainer's `ltx_trainer/captioning.py`
with a CLI and one extra backend. The original Qwen3-Omni and Gemini captioners
remain selectable. The upstream trainer captioning module is unchanged.

All backends create a source caption, then a LiteLLM text model rewrites it
as an LTX generation prompt following the [official LTX-2 guide](https://ltx.io/blog/prompting-guide-for-ltx-2).
Output JSONL rows contain `media_path`, `source_caption`, `caption` (the LTX
prompt), a source SHA-256, processing context, the model that actually served
each call (`served_models`), and the crop window when one was used
(`subject_box`, `[x0, y0, x1, y1]` in source pixels). Paths are relative to
the output file's directory, matching LTX Trainer's dataset format.

| Backend | Input seen by the captioner | Requirements |
|---|---|---|
| `litellm_vision` (default) | Sampled frames only; no audio | `LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_MODEL_VISION`, `LITELLM_MODEL_TEXT` in the workspace `.env` |
| `qwen_omni` | Full video plus extracted audio | The original Qwen3-Omni vLLM server |
| `gemini_flash` | Full video including audio | The original Gemini credentials |

Run from the LTX-2 repository root with the `ltx` conda environment:

```bash
conda run -n ltx python -m scripts.ltx_prompts.prepare_ltx_prompts /path/to/clips \
  --output /path/to/results/ltx_prompts.jsonl --trace-dir /path/to/results/trace \
  --vision-model litellm/gemma-4-31b-it --text-model litellm/gemma-4-31b-it \
  --subject-crop --max-side 1024 --frames 12
```

## Pin the models

`free_model_vision` and `free_model_text` in the workspace `.env` are LiteLLM
load-balanced groups. On 2026-09-28 the vision group also contained
Llama-3.2-11B-Vision, Kosmos-2, Fuyu-8B, DePlot, CLIP and embedding models, so
one clip could be captioned by a strong model and the next by one that cannot
chat. This explains the earlier empty answers and gross misreadings. Pass
`--vision-model`/`--text-model` with a concrete model or a single-model group,
such as `litellm/gemma-4-31b-it`, which spans NIM, OpenRouter `:free` and AI
Studio. Every call records the deployment that served it, taken from LiteLLM's
`x-litellm-model-id` header, because inside a group `response.model` is just
the group name. NIM's `google/gemma-4-31b-it` accepts at most **4 images
per request** and no video input.

## Frame layout

- `--layout frames` (default) sends at most `--max-images` images. Each image is
  preceded by a text part naming its frame numbers and timestamps. When there
  are more frames than images, consecutive frames share one image as a
  left-to-right strip. Each tile keeps `--max-side` and carries a drawn
  timestamp.
- `--subject-crop` first sends *locate* calls in which the same model returns a
  person box per frame. It then crops every frame to the union of those boxes
  plus a margin, and spends one image slot on a full uncropped context frame.
  A single fixed window keeps travel across the floor visible. The option exists
  because a full-body subject in a wide capture view is about 15% of the frame
  width: without it, small held objects and hand poses cannot be read. If no
  usable box comes back, the run falls back to full frames and records
  `subject_box: null`.
- `--segments N --segment-mode sequential|hierarchical` splits the clip into N
  consecutive segments of `--max-images` frames, overriding `--frames`. Each
  frame gets its own image slot, with one vision call per segment. In
  `hierarchical` mode an overview call first sees the uncropped first frame of
  every segment, and each segment call receives that overview as context. A
  merge-aware rewrite prompt (`SEGMENT_REWRITE_INSTRUCTION`) joins the labelled
  observations into one prompt. The option exists because a single call over
  12–16 frames puts several frames in one image and leads the model to copy the
  previous per-frame line forward.
- `--layout sheet` keeps the original 2×2 contact sheets.

The vision prompt asks for one line per frame (facing direction, held objects,
arms and hands, legs, horizontal position from 0 to 1), then motion,
appearance, setting and camera. Its rules define left and right from the
person's own viewpoint, forbid inventing where an object went, and ask for
colours as seen. `--instruction-file` and `--rewrite-instruction-file` replace
either prompt. The prompt hashes are part of the resume context.

## Multi-view coordination

`--coordinate-views` treats the clips in one directory as simultaneous views of
one performance. For DNA Rendering these are the 8 calibrated cameras. An
optional `views.json` in that directory gives each clip stem's camera placement
(`azimuth_deg`, `elevation_deg`, `direction`). After every view has its own
observations and single-view prompt, one `mv_sheet` call per directory
reconciles all views' observations into a subject sheet. The sheet holds the
person, appearance, held objects, heading and turns in rig azimuth, travel, and
a body-relative action timeline. Its rules: a detail clearly seen by one camera
beats cameras that could not see it; arm motion is described in body terms;
left/right is converted using the camera geometry. Then one `mv_rewrite:<view>`
call per view writes that view's prompt from the sheet, which is authoritative
for what happens, and the view's own observations plus geometry, which are
authoritative for framing. Each record keeps `caption_single_view`, and
`caption` becomes the coordinated prompt. The sheet is stored in
`coordination`. The trace for the directory is `<trace-dir>/mv_<dir>/`.

Code, not the model, does the geometry. `_heading_consensus` converts each
camera's per-frame facing label into an implied rig heading (camera azimuth plus
the relative facing) and takes the circular medoid per time step.
`_heading_phases` merges those steps into heading phases. `_travel_estimate`
solves floor travel by least squares, since a camera at azimuth B sees travel
toward rig direction B−90 as motion toward frame right. The sheet call receives
both as tables, and each view's rewrite receives its own code-computed relative
facing and travel. Without this, Nemotron-3-Ultra spent its whole budget on
per-frame angle arithmetic and returned nothing. With it, the heading consensus
matched hand-checked references even where single cameras were wrong. An
unparseable sheet is resampled up to 3 times.

Split the models by role. The vision model (`--vision-model`, e.g.
`litellm/gemma-4-31b-it`) should only perceive, and a short fixed-field prompt
suits a ~30B model better than one that also asks it to reason. The rewrite
and coordination are text-only reasoning (`--text-model`, `--coord-model`) and
suit a stronger model such as `litellm/nemotron-3-ultra-550b-a55b`.

Reasoning text models need room and time: `--text-max-tokens` (default 16000;
hidden reasoning counts against it) and `--text-timeout` (default 900 s). Use
`--sheet-thinking on` for the subject sheet, where thinking off got a turn
direction wrong, and `--rewrite-thinking off` for the rewrites, which is about
10× faster with the geometry precomputed. Both set
`chat_template_kwargs.enable_thinking`, which NIM Nemotron-3 honours.

`--workers N` processes up to N clips concurrently, each thread with its own
captioner. Multi-view groups coordinate in parallel, with a semaphore limiting
their combined sheet and per-view rewrite calls to N concurrent requests.
Vision observations are cached separately from the text stages.
A record's `vision_context` (vision model, prompt hashes, sampling, crop,
segments and `VISION_STAGE_VERSION`) decides reuse, so changing only the text
model or text prompts reruns just the text stages. `--observations-from
other/ltx_prompts.jsonl` also reuses a matching run's observations; the record
then points at the donor trace in `observations_trace`. Bump
`VISION_STAGE_VERSION` whenever frame sampling, cropping, packing or segment
code changes.

## Trace and report

With `--trace-dir`, each clip gets `<trace-dir>/<clip>/trace.json` and the
exact JPEGs that were base64-encoded into each request
(`callNN_<stage>_imgMM.jpg`, SHA-256 in the trace). For every call the trace
holds the message parts in order, with images replaced by an index into those
files, plus `max_tokens`/`temperature`, every retry attempt, and the full raw
response envelope (`model_dump`). `<trace-dir>/report.md` is a Markdown report of
them. For each call it has the text parts fenced verbatim, the labelled images side by side in one table row,
the raw message content, and a link to each attempt's response envelope
(`callNN_attemptM_response.json`). Add `--ground-truth refs.json` to show a
reference prompt with its key facts beside each result. `--report-only`
rebuilds the report without calling any model.

## Completion

The command saves successful records to `.work.jsonl` after each clip and
publishes the final JSONL only when all selected clips succeed. It reuses a
compatible record on a rerun only when the source hash and the context match:
models, prompt hashes, frames, size, layout, crop setting and producer hash.
`--overwrite` forces regeneration. `--retries` retries 429/5xx errors,
timeouts and empty answers with backoff. Other errors fail the clip.

Sampled frames cannot show fast repeated gestures or events between samples,
and the frames-only backend never hears audio. Review generated prompts before
using them for training. The 2026-09-28 study
(`expr/ltx_prompt_examples_20260928/dna_review/REVIEW.md`, and `MV_REVIEW.md` for all 8 views) measured the
remaining gaps.

## Offline checks

From the LTX-2 repository root:

```bash
conda run -n ltx python -m unittest scripts.ltx_prompts.test_prepare_ltx_prompts
```
