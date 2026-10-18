"""Freeze exact-step historical references without loading a model.

Historical studies retain their recorded code pins; current code must not be
mistaken for their code identity. Checkpoint loading verifies bytes and the
saved sigmoid policy. The comparison pins the committed original-ID masks.
"""

import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.learn import checkpoints
from scripts.prune.learn.progressive import ProgressivePolicy

SOURCE_ID = "474c6087fb96dff9078b5fc8b088647ab6f272d6b4cef7945dc3b596eb30c7fb"
REMOVAL_ID = "e0cbebb86d734b56125bb9a23fc65f5cfef6d2620833ca8341277ec030757c73"
CONTINUATION_ID = "77ad0365a4b9cd7cee6131f3873c647feb8a00b5f60fb9a5d6480ae8c3726a9c"
ZERO_SHA = "8736fccb98f652562f83d305f6b9e47b51d1f93a394e6023828d0bb6c3aa9ab9"
TEST_ACTORS = ["51", "800", "801", "178", "791", "195"]


def test_sources(study):
    sources = study["splits"]["test"]
    actors = study["splits"]["test_actors"]
    if (
        actors != TEST_ACTORS
        or len(sources) != 12
        or len({s["relative_dir"] for s in sources}) != 12
        or set(actors) & set(study["splits"]["learning_actors"])
        or any(sum(s["actor"] == actor for s in sources) != 2 for actor in actors)
        or any(
            s["actor"] not in actors or s["n_latent_frames"] != 28 or s["shape"] != [128, 28, 32, 32] for s in sources
        )
    ):
        raise ValueError("fixed test requires the exact six actors, two full F28 views each, and no training actor")
    return sources


def historical_study(path, expected):
    path = Path(path).resolve()
    study = json.loads(path.read_text())
    if (
        study.get("format") != "learned_head_study_v1"
        or study.get("identity") != expected
        or expected != checkpoints.digest({k: v for k, v in study.items() if k != "identity"})
    ):
        raise ValueError("historical frozen study identity differs")
    return study


def ordinary_draws(study):
    pin = study["draws"]["42:ordinary"]
    if file_sha256(pin["path"]) != pin["sha256"]:
        raise ValueError("historical ordinary draws changed")
    return json.loads(Path(pin["path"]).read_text())


def committed_mask(state):
    if state.get("progressive") is None:
        raise ValueError("comparison requires a saved permanent head policy")
    policy = ProgressivePolicy.from_state(state["progressive"])
    ones = getattr(policy, "one", torch.zeros_like(policy.free))
    removed = ~policy.free & ~ones
    if removed.shape != (48, 32):
        raise ValueError("comparison original head layout differs")
    return removed


def checkpoint_record(study, path, update, draws):
    """Verify actual checkpoint bytes, global adapter updates, and draw prefix."""
    state = checkpoints.load(path)
    if (
        type(update) is not int
        or update < 0
        or update % 50
        or state["identity"]["study"] != study["identity"]
        or state["identity"]["arm"] != "learned_progressive"
        or state["identity"]["seed"] != 42
        or state["update"] != update
        or state["draw_cursor"] != 4 * update
        or state.get("draws_per_update") != 4
        or state["identity"]["draws"] != study["draws"]["42:ordinary"]["sha256"]
    ):
        raise ValueError("comparison checkpoint global step, identity, or draw cursor differs")
    moments = state["adapter_optimizer"]["state"]
    if (update == 0 and moments) or (
        update > 0 and (len(moments) != 192 or any(float(v["step"]) != update for v in moments.values()))
    ):
        raise ValueError("comparison adapter optimizer completed updates differ")
    if len(draws) < 4 * update:
        raise ValueError("comparison lacks complete consumed draws")
    removed = committed_mask(state)
    return dict(
        path=str(Path(path).resolve()),
        sha256=file_sha256(path),
        sidecar_sha256=file_sha256(Path(path).with_suffix(".json")),
        study=study["identity"],
        identity=state["identity"],
        phase=state["phase"],
        update=update,
        draw_cursor=state["draw_cursor"],
        consumed_draws_sha256=checkpoints.digest(draws[: 4 * update]),
        progressive_sha256=checkpoints.digest(state["progressive"]),
        alpha_sha256=checkpoints.tensor_hash(state["alpha"]),
        adapter_hashes={k: checkpoints.tensor_hash(v) for k, v in state["adapters"].items()},
        removed=int(removed.sum()),
        retained=int((~removed).sum()),
        removed_per_block=removed.sum(1).tolist(),
        removed_original_ids=removed.nonzero().tolist(),
    )


def freeze_pairs(source, removal_path, continuation_path, source_zero, draws):
    """Return all 17 references only after validating both frozen lineages."""
    removal = historical_study(removal_path, REMOVAL_ID)
    continuation = historical_study(continuation_path, CONTINUATION_ID)
    test_sources(source)
    source_draws = ordinary_draws(source)
    if len(draws) != 3200 or draws != source_draws[:3200]:
        raise ValueError("new pilot must preserve the first 3200 r13 draws")
    for previous, prefix in ((removal, 2000), (continuation, 3200)):
        if any(previous[k] != source[k] for k in ("components", "text_context", "software", "splits")):
            raise ValueError("comparison components, context, software, or splits differ")
        for key in (
            "sigmas",
            "monitor_sigma_indices",
            "numerics",
            "uniform_f18",
            "rank_draw_assignment",
            "global_unique_clips",
            "rank",
            "adapter_lr",
            "gate_lr",
            "betas",
            "adam_epsilon",
            "sparsity_penalty",
            "sparsity_logit_offset",
            "lambda_max",
            "lambda_warmup",
            "lambda_start_multiplier",
        ):
            if previous["config"][key] != source["config"][key]:
                raise ValueError(f"comparison numerical or input policy differs: {key}")
        if ordinary_draws(previous)[:prefix] != draws[:prefix]:
            raise ValueError("comparison ordered draw prefix differs")
    pairs = []
    source_hashes = {k: checkpoints.tensor_hash(v) for k, v in source_zero["adapters"].items()}
    for update in range(0, 801, 50):
        owner = removal if update <= 500 else continuation
        path = Path(owner["root"]) / f"training/learned_progressive/replicate_42/checkpoints/update_{update:04d}.pt"
        record = checkpoint_record(owner, path, update, draws)
        if update == 0 and record["adapter_hashes"] != source_hashes:
            raise ValueError("all 192 initial adapter tensors must match the previous lineage")
        if update in (500, 800) and record["removed"] != {500: 160, 800: 256}[update]:
            raise ValueError("historical permanent-mask sanity count differs")
        pairs.append(dict(update=update, previous=record))
    return dict(
        format="clipped_reset_checkpoint_pairs_v1",
        source_study=source["identity"],
        previous_studies=[
            dict(path=str(Path(p).resolve()), sha256=file_sha256(p)) for p in (removal_path, continuation_path)
        ],
        draws_sha256=checkpoints.digest(draws),
        pair_grid=list(range(0, 801, 50)),
        initial_adapter_hashes=source_hashes,
        pairs=pairs,
    )
