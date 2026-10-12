"""Shared CPU builders; never collected as test modules."""

from __future__ import annotations

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import subset
from scripts.onestep_avatar.training import checkpoints, config

A = "diffusion_model.block.to_q.lora_A.weight"

B = "diffusion_model.block.to_q.lora_B.weight"


def _contract(mode: str = "bidirectional", teacher: bool = False, *, random_start: bool = False) -> dict:
    options = [
        "--mode",
        mode,
        "--subset",
        "/unused/list",
        "--output",
        "/unused/output",
        "--variant",
        "dev",
        "--objective",
        "white",
        "--lora-rank",
        "2",
    ]
    if mode == "bidirectional":
        options += ["--span-latent-frames", "7"]
    elif teacher:
        options += ["--teacher-forcing"]
    if random_start:
        options += ["--start-policy", "random"]
        if mode == "causal":
            options += ["--blocks-per-sample", "1", "--span-latent-frames", "3"]
    settings = config.parse_settings(options)
    settings.base_identity = {"base_transformer_file": "base.safetensors", "base_transformer_sha256": "a" * 64}
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "corpus_root": "/unused",
        "splits": {"train": ["actor"]},
        "sources": [
            {
                "relative_dir": "actor/view",
                "actor": "actor",
                "split": "train",
                "n_latent_frames": 7,
                "shape": [128, 7, 2, 2],
                "fps": 30,
            }
        ],
        "excluded": {},
    }
    membership["sha256"] = subset.membership_hash(membership)
    plan = config.build_frame_plan(settings, membership, SpatioTemporalScaleFactors(8, 32, 32))
    record = checkpoints.make_contract(settings, membership, plan, step=0)
    record["adapter"]["tensor_shapes"] = {A: [2, 4], B: [4, 2]}
    return record
