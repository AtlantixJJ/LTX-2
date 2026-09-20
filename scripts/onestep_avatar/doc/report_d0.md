# `report_d0.py` — the D0 handoff record

> **Historical.** This builder encodes a retired run contract: it expects the obsolete σ list
> containing zero (current training refuses σ = 0), separate `frozen_base_sigma_*.mp4` files the
> current probe deliberately does not write, and it hardcodes union-masked loss, which the
> binding unweighted full-frame decision superseded. It cannot report a current run. Keep
> historical reports as they are rather than relabelling old masked runs.

## Objective

Write the completion report for the D0 GT-renoise sanity arm, **after verifying the artifacts
exist** — the checkpoint, the per-rank logs, and the decoded MP4s — rather than describing a
run from memory.

## Data flow

```
runs/<name>/  checkpoint + metrics_rank*.jsonl + probes/*.mp4
        ▼  (each checked before anything is written)
   report markdown / JSON
```

## Organization logic

It is a **handoff record, not an interpretation of visual quality**. The distinction matters
for D0 specifically: D0 is a capacity control at `r = 0`, and its loss is **not comparable**
to guide-conditioned arms. A report that blurred the two would invite exactly the wrong
conclusion — that the render→capture task is nearly solved.

Checking artifacts before writing is the same discipline as `_render_is_complete`: a record
that claims a run finished must be falsifiable against what is on disk.

## Scope

D0 only. A general arm reporter is `report.py`'s job (not written), which would carry the
held-out evaluation table over all baselines.
