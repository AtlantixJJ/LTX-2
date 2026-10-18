"""Prepare the pinned r13-zero clipped-reset pilot and exact-step comparisons.

This CPU entrypoint hashes source bytes but opens no model. It changes only
the explicit experiment settings, raw-score initialization, and gate policy.
No previous optimizer, progress, or acceptance receipt becomes native evidence.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.checkpoint_pairs import SOURCE_ID, ZERO_SHA, freeze_pairs, historical_study, ordinary_draws
from scripts.prune.data.learned_head_study import code_identity, load_study
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import checkpoints, protocol
from scripts.prune.learn.clipped_reset import DEFAULTS, ClippedResetPolicy
from scripts.prune.learn.heads import optimizer, train_identity


def pilot_config(source):
    cfg = copy.deepcopy(source["config"])
    cfg.update(
        version="clipped-reset-r13zero-f18-rank128-v1",
        alpha_init=1.0,
        soft_updates=500,
        binary_updates=0,
        compact_updates=0,
        replicate_seeds=[42],
        gpu_ids=[0, 1, 2, 3],
        removed_heads=[],
        person_coefficient=0.0,
        small_region_coefficient=0.0,
        progressive_policy=dict(DEFAULTS),
        wall_time_hours=72,
    )
    protocol.validate(cfg)
    if protocol.phase_budget(cfg) != (500, 500, 800):
        raise ValueError("pilot selection or total budget differs")
    return cfg


def initialize(source, cfg):
    """Replace only raw scores and gate state; preserve every other payload."""
    if (
        source["update"] != 0
        or source["draw_cursor"] != 0
        or source["phase"] != "soft"
        or source.get("draws_per_update") != 4
        or source.get("mask") is not None
        or source.get("progressive") is not None
        or not torch.equal(source["alpha"], torch.full((48, 32), 3.0))
        or len(source["adapters"]) != 192
        or source["adapter_optimizer"]["state"]
        or source["gate_optimizer"]["state"]
    ):
        raise ValueError("r13 source is not the untrained checkpoint-zero state")
    if any(not torch.isfinite(v).all() or (k.endswith(".B") and v.any()) for k, v in source["adapters"].items()):
        raise ValueError("source adapter B matrices must be zero and all tensors finite")
    result = copy.deepcopy(source)
    result["alpha"] = torch.ones_like(source["alpha"], dtype=torch.float32)
    result["gate_optimizer"] = optimizer(
        [torch.nn.Parameter(result["alpha"].clone())], cfg["gate_lr"], cfg
    ).state_dict()
    result["progressive"] = ClippedResetPolicy(result["alpha"], cfg["progressive_policy"]).state_dict()
    original, replaced = copy.deepcopy(source), copy.deepcopy(result)
    for state in (original, replaced):
        for key in ("alpha", "gate_optimizer", "progressive"):
            state.pop(key, None)
    if unchanged_state_hash(original) != unchanged_state_hash(replaced):
        raise ValueError("initialization changed adapters, adapter Adam, RNG, or cursors")
    return result


def prepare(previous, removal, continuation, plan, output):
    previous, removal, continuation, plan, output = [
        Path(p).resolve() for p in (previous, removal, continuation, plan, output)
    ]
    if output.exists():
        raise ValueError("clipped-reset preparation requires a fresh output")
    source = historical_study(previous, SOURCE_ID)
    for pin in [source["plan"], *source["files"]]:
        if file_sha256(pin["path"]) != pin["sha256"]:
            raise ValueError(f"source bytes changed: {pin['path']}")
    for pin in source["components"].values():
        if file_sha256(pin["path"]) != pin["file_sha256"]:
            raise ValueError("source component bytes changed")
    source_path = Path(source["root"]) / "training/learned_joint/replicate_42/checkpoints/update_0000.pt"
    if file_sha256(source_path) != ZERO_SHA:
        raise ValueError("r13 initialization checkpoint hash differs")
    source_zero = checkpoints.load(source_path)
    if source_zero["identity"] != train_identity(source, "learned_joint", 42, 152, source["draws"]["42:ordinary"]):
        raise ValueError("r13 initialization training identity differs")
    cfg = pilot_config(source)
    state = initialize(source_zero, cfg)
    draws = ordinary_draws(source)[:3200]
    pairs = freeze_pairs(source, removal, continuation, source_zero, draws)
    # All expensive source checks finish before creating a new output root.
    config_path, draw_path = output / "configs/config.json", output / "draws/replicate_42_ordinary.json"
    checkpoints.publish_json(config_path, cfg)
    checkpoints.publish_json(draw_path, draws)
    frozen_plan = output / "provenance/plan.md"
    checkpoints.publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    pair_path = output / "comparisons/checkpoint_pairs.json"
    checkpoints.publish_json(pair_path, pairs)
    pin = lambda p: dict(path=str(p), sha256=file_sha256(p))
    replaced = {p["path"] for p in source["draws"].values()} | {str(Path(source["root"]) / "configs/config.json")}
    study = dict(source)
    study.update(
        root=str(output),
        config=cfg,
        config_sha256=file_sha256(config_path),
        code=code_identity(),
        plan=pin(frozen_plan),
        draws={"42:ordinary": pin(draw_path)},
        files=[p for p in source["files"] if p["path"] not in replaced]
        + [pin(p) for p in (config_path, draw_path, pair_path)],
        progressive_source=dict(
            study=pin(previous), checkpoint_zero=pin(source_path), draw_prefix_sha256=checkpoints.digest(draws)
        ),
        clipped_reset_protocol=dict(
            serial_gpu_id=6,
            checkpoint_pairs=pin(pair_path),
            fixed_test_actors=source["splits"]["test_actors"],
            noise_seed=2042,
            selection_updates=500,
            recovery_updates=300,
            fixed_test_max_case_forwards=1836,
            wall_time_allowance_hours=72,
            wall_time_scope="training, native acceptance, test inference, media, export and validation",
            endpoint_mse_ratio=1.05,
            lpips_mean_delta_max=0.01,
            lpips_case_flag=0.03,
            export_max_abs=0.02,
            stop_on_quality=False,
        ),
    )
    study.pop("identity")
    study["identity"] = checkpoints.digest(study)
    checkpoints.publish_json(output / "configs/frozen_study.json", study)
    state["identity"] = train_identity(study, "learned_progressive", 42, 0, study["draws"]["42:ordinary"])
    state["lineage"] = dict(source_identity=source_zero["identity"], **study["progressive_source"])
    target = output / "training/learned_progressive/replicate_42/checkpoints/update_0000.pt"

    def write(p):
        with p.open("wb") as stream:
            torch.save(state, stream)

    checkpoints.publish(target, write)
    checkpoints.publish_json(
        target.with_suffix(".json"),
        dict(
            identity=state["identity"],
            update=0,
            phase="soft",
            draws_per_update=4,
            sha256=file_sha256(target),
            tensor_hashes={k: checkpoints.tensor_hash(v) for k, v in state["adapters"].items()},
            alpha_sha256=checkpoints.tensor_hash(state["alpha"]),
            lineage=state["lineage"],
            progressive_sha256=checkpoints.digest(state["progressive"]),
        ),
    )
    restored = checkpoints.load(target, state["identity"])
    if unchanged_state_hash(restored) != unchanged_state_hash(state):
        raise ValueError("published clipped initialization changed")
    receipt = dict(
        study=study["identity"],
        source_checkpoint_sha256=ZERO_SHA,
        exact_draw_prefix=3200,
        initial_adapter_tensors=192,
        initial_B_zero=True,
        paired_checkpoints=17,
        preserved_adapter_optimizer_and_rank_rng=True,
        source_bytes_verified=True,
    )
    checkpoints.publish_json(output / "provenance/preparation.json", receipt)
    (output / "logs").mkdir(exist_ok=True)
    load_study(output / "configs/frozen_study.json", verify_components=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "removal", "continuation", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(args.previous, args.removal, args.continuation, args.plan, args.output), indent=2))


if __name__ == "__main__":
    main()
