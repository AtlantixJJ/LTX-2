# CLAUDE.md — `LTX-2/scripts/onestep_avatar/`

Guidance for Claude Code when working inside this package: the one-step LTX-2.5 avatar
renderer, corpus tooling and model training in **one tree**.

**Read [`doc/README.md`](doc/README.md) before editing anything here** — it carries the
cross-file data flow, and every module has its own design doc beside it.

## One package, two conda envs

The package was split across two directories until 2026-09-15 (`scripts/onestep_avatar/` in
the workspace for the corpus half, this one for the model half) because the ARGAvatar renderer
and the LTX VAE cannot share a process. That reasoning confuses a **runtime** constraint with a
**layout** one: exactly one module imports ARGAvatar, and the split cost four transcribed
copies of shared knowledge plus the tests to pin them. It is one package now.

| Env | Runs |
|---|---|
| `argavatar` | **`build_guidance.py` only** — the one module that imports `scripts.inference.pipeline` |
| `ltx` | everything else, including the tests |

```bash
cd LTX-2                                       # the repo root, not scripts/onestep_avatar
conda run -n ltx       python -m scripts.onestep_avatar.<module> ...
conda run -n argavatar python -m scripts.onestep_avatar.build_guidance ...
conda run -n ltx       python -m pytest scripts/onestep_avatar/tests -q
```

`-m` from the LTX-2 root is mandatory: modules import each other as `scripts.onestep_avatar.*`
and do not touch `sys.path`.

**The namespace-package detail that makes `build_guidance` work.** `LTX-2/scripts/` has no
`__init__.py`, so `scripts` is a PEP 420 namespace package whose `__path__` recomputes when
`sys.path` changes. `build_guidance` inserts the ARGAvatar root at runtime, and
`scripts.inference` (ARGAvatar's portion) then resolves alongside `scripts.onestep_avatar`
(this one) and `scripts.prune`. **Do not add an `__init__.py` to `LTX-2/scripts/`** — it would
turn the namespace into a regular package and break that merge.

## Where this code lives, and why that matters

This package sits inside the **LTX-2 submodule**, but most of it is workspace-specific:
corpus plumbing for DNARendering and ARGAvatar, not anything upstream would want. It lives
here because the model half genuinely cannot leave — `train.py`, `precompute.py` and
`causal_core.py` import `scripts.prune.core` and `ltx_core`/`ltx_trainer`, all LTX-2-resident.

Practical consequence: **corpus-side changes move the LTX-2 submodule pin.** The workspace
`CLAUDE.md` asks for explicit approval before moving that pin, so say so when a change here
needs committing.

## The documentation contract

**Every module has a design doc at `doc/<module>.md`, and updating it is part of the change —
not a write-up afterwards.**

| | |
|---|---|
| **Inline docstrings** answer | *why this line, why this constant, why not the obvious alternative* |
| **`doc/<module>.md`** answers | *what this file is for, what flows through it, how it fits the other files, what breaks if you change it* |

The second is what a reader cannot reconstruct from one file, and it is where this pipeline
has gone wrong before — every past bug has been **two producers of something that must have
one**, which is invisible from inside either producer.

**When to update a doc.** Any change to a module's *objective*, *data flow*, *invariants*, or
*contract with another module*. A pure refactor that moves no tensor and changes no artifact
needs no doc edit; a new flag that changes what lands on disk always does.

**When adding a module**, add `doc/<name>.md` and a row in `doc/README.md`'s file table in the
same commit, keeping the existing section shape — Objective · Data flow · Organization logic ·
Invariants · Gotchas · Tests.

**When a doc and the code disagree, the code is right and the doc is a bug.** Fix it there and
then; a stale design doc is worse than none, because it is trusted.

## The invariants that are not obvious from one file

1. **`causal_core.py` is the ONE rollout implementation.** `train.py`, `onestep_core.py`,
   `visualize_d0.py`, `bench_forward.py` and `windows.py`'s block plan all call it. Never add
   a second "build a block state" path — a train/deploy mismatch must have to be an edit to
   that file rather than a divergence between two that were meant to agree.
2. **One producer per artifact.** The crop box comes from `precompute.py --capture-only`'s
   manifest; `z_y` from the capture pass; the guide and its alpha from `build_guidance.py`;
   the subset from `windows.py`. Readers never recompute and never "reconstruct if missing" —
   they raise with a pointed error.
3. **Shared knowledge has exactly one spelling.** Artifact names live in `dataset.py`, the
   crop box in `geometry.py`, the block plan in `causal_core.py`, the mask codec in
   `mask_video.py`. These were four transcribed pairs before the merge; do not reintroduce a
   copy "to avoid an import".
4. **Both objectives (`bg`, `white`) share every code path**, differing only in which pixels
   were encoded and which filename holds them. An objective is never a second pipeline.
5. **No synthetic pixels ever enter a loss target** — shift-and-cap, never white-pad, and
   exclude a subject that does not fit the canvas.
6. **Masks are stored losslessly** (`mask_video.py`). These mattes are already one generation
   of lossy video from the truth; the pipeline does not add a second.

## Before calling a change done

- `conda run -n ltx python -m pytest scripts/onestep_avatar/tests -q` passes;
- the touched modules' `doc/*.md` reflect the change;
- if the change can move a tensor on the `k2` path, `scripts/prune`'s own rule applies —
  see [`../prune/CLAUDE.md`](../prune/CLAUDE.md).
