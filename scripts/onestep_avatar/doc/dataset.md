# `dataset.py` — corpus layout and the objective → filename map

## Objective

Three jobs, all "one constant (or one function), not a convention repeated at call sites":

1. **Where the corpus is and how a clip is laid out.** `DEFAULT_CORPUS_ROOT`, `ClipRef` and
   its per-view path accessors. T4's scale-out to the full `Processed/` tree is a `root=`
   argument, not a second code path.
2. **Which filename each objective's artifacts use, and the alpha/mask grid they share**
   (SS1.2). Every module reads it from here. It was transcribed into a second module
   (`corpus_names.py`) while the package was split across two trees; consolidating removed
   both the copy and the test that pinned it.
   Also owns `GUIDE_COMPOSITING_VERSION` — the guide's RGB/alpha *contract* version, as
   opposed to *which filename* it lives in. It belongs beside the naming map for the same
   reason: `build_guidance.py`'s `_render_is_complete` reads it the same way it reads
   `render_name`, so one module still owns "how does a reader know this artifact is current".
3. **`atomic_write`, since S1 of the 2026-09-17 cleanup plan.** Not a corpus-layout concern by
   itself, but every writer in the package needs it and this is the one leaf module every
   writer (`precompute.py`, `build_guidance.py`, `mask_video.py`, `windows.py`) already
   imports without creating a cycle — `mask_video.py` cannot import `precompute.py` (which
   imports it back), and a second new file was not worth it for three lines.

## Data flow

Mostly pure path/metadata resolution — reads `meta.json`, opens no video. `atomic_write` is
the one exception: it is generic file I/O, used by every producer in the package.

Three independent resolutions, no I/O beyond `meta.json`:

- corpus root → `ClipRef` → `rgb_path` / `mask_path` / `bbox_path` / `pose3d_path` /
  `refined_pose3d_path` / `view_dir`,
  and `actor_id` / `fps` / `n_frames` / `is_done` from `meta.json`;
- objective → `render_name` / `render_metadata_name` / `guide_bundle_name` /
  `capture_bundle_name`;
- `(destination, write_to)` → `atomic_write` → `write_to(temp)` → `temp.replace(destination)`.

`CaptureManifest` reads the **crop box of record** written by the LTX half. It is a reader
only: nothing here computes a box.

## Organization logic

The objective mapping lives here, next to the corpus layout, because that is what it is —
a fact about where things sit on disk, not a training decision. Putting it in
`build_guidance.py` would make the renderer its owner, and `windows.py` would then import
a renderer to learn a filename.

**Three rules encoded in the mapping:**

- **`bg` is the unsuffixed name.** The 2,034 capture bundles and 19 guide renders already on
  disk were written before the objective existed, and they are `bg` artifacts. Mapping `bg`
  to the names they already have means adding `white` invalidates none of them.
- **Only two artifacts are suffixed** — the guide render and the two latent bundles. The
  render's alpha, the cropped capture matte and the loss-mask grids are
  objective-**independent** (same render, same matte; only what sits behind the subject
  differs), so suffixing them would manufacture two copies of one thing.
- **Both persisted masks are `.mp4`, and the constants carry a `_STEM` as well as a `_NAME`.**
  The stem is what `mask_video.read_mask` takes, so a legacy `.npy` is still found. See
  [mask_video.md](mask_video.md).
- **An unknown objective raises**, rather than falling back to a default. A typo that
  silently resolves to `bg` would train the wrong pair with no error anywhere.
- **`GUIDE_COMPOSITING_VERSION` has no legacy value to grandfather in.** Unlike a
  pre-`objective` sidecar (which correctly infers as `bg`), every render built before this
  field existed used the retired v1 double-alpha formula — there is no historical value that
  means "current", so a missing/old version is always stale (2026-09-18 audit finding F1;
  see [build_guidance.md](build_guidance.md)).

## Invariants

- **`capture_master_latent_frames` is the one producer of a source's latent-frame count.**
  Read it from the stored master, never from the video: a master consolidated out of v1
  per-window slices ends at the last whole window and is short of the video by up to one
  window. `windows.py` derived the number from the video until 2026-09-16 and froze block
  plans one block too long, which `train.py` — planning from the tensor — then refused.


- `actor_id()` returns the **bare** actor id, never `(part, id)`. Actor ids are not globally
  unique across `Part_*`, and the conservative reading is what makes the held-out split
  leak-proof under either interpretation.
- `fps()` is never defaulted — fps scales the temporal RoPE axis.
- `refined_pose3d_path()` is clip-level because the multiview solve refines one shared body
  trajectory; it is never substituted for the per-view pose record, which owns the calibrated
  camera and image-cache fields.
- `CaptureManifest` is the single source of the crop box. Recomputing one "the same way"
  is exactly the desync this file exists to prevent.
- **`atomic_write`'s temp name is one convention** (`.<stem>.tmp.<pid><suffix>`) for every
  writer in the package. Do not hand-roll a second one at a new call site.

## Tests

`tests/test_geometry.py` (golden crop-box values), `tests/test_mask_video.py` (the mask
codec), and `tests/test_dataset.py` (`atomic_write`'s replace-on-success /
cleanup-on-failure contract). The name mapping no longer needs a test of its own: there is
one copy of it.
