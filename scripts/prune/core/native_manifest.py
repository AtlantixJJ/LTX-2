"""Shared whole-clip task and guidance fields for producers and saved readers.

The rollout producer supports guided/dev modes; input-only manifests retain
their narrower historic field set. Both obtain full visibility, dtype and
guidance accounting here. This helper selects no schedule or source geometry.
"""

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
