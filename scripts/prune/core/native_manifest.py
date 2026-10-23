"""Build shared task, visibility, dtype and guidance fields for saved manifests.

The rollout producer can record guided/dev tasks; capture-only input manifests
use a narrower field set. This helper selects no schedule or source geometry,
so every producer retains explicit ownership of those values."""

import hashlib

from scripts.prune.core.session import DTYPE


def fields(*, cfg=1.0, stg=0.0, stg_blocks=(), rescale=0.0, negative_prompt=None, input_only=False):
    guidance = dict(
        cfg=cfg,
        stg=stg,
        stg_blocks=list(stg_blocks) if stg != 0.0 else [],
        rescale=rescale,
        negative_prompt=negative_prompt,
        passes_per_step=1 + (cfg != 1.0) + (stg != 0.0),
    )
    if not input_only:
        guidance.update(
            negative_prompt_sha256=None
            if negative_prompt is None
            else hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest(),
            execution="sequential passes",
        )
    return dict(
        whole_clip=True,
        trajectory_only=False,
        attention="full_bidirectional",
        latent_dtype=str(DTYPE),
        guidance=guidance,
    )
