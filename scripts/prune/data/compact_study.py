"""Prepare aligned compact weights and a new fixed-width adapter-recovery study.

Read the completed joint checkpoint and its terminal selection scores. Restore
heads to a multiple of four, retain every previously kept unit, fuse source
adapters and slice weights in original order. Verify real compact tensor shapes
before freezing the new draw prefix and source identity. This is a new study,
with fresh recovery optimizers, rather than a continuation of source Adam state.
The compact-study schema retains its dedicated device policy.

CLI: python -m scripts.prune.data.compact_study --help.
Workflows and launch settings: doc/usage.md."""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.prune.core import provenance, records
from scripts.prune.data import study_io
from scripts.prune.learn import joint_training
from scripts.prune.score import export_learned_ffn, export_pruned


def aligned_mask(mask, scores, multiple):
    """Restore the highest removed scores, breaking ties by original unit ID."""
    if mask.ndim != 2 or mask.shape != scores.shape or multiple < 1:
        raise ValueError("Invalid alignment geometry")
    if not torch.isfinite(scores).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Invalid alignment scores or mask")
    result, restored = mask.bool().clone(), []
    for block, row in enumerate(result):
        count = int(row.sum())
        target = ((count + multiple - 1) // multiple) * multiple
        if count == 0 or target > row.numel():
            raise ValueError("Alignment cannot retain a legal width")
        eligible = [i for i, bit in enumerate(row) if not bit]
        eligible.sort(key=lambda i: (-float(scores[block, i]), i))
        chosen = eligible[: target - count]
        row[chosen] = True
        restored.append(chosen)
    return result.float(), restored


def verify_export(path, alignment):
    """Check that the artifact is genuinely compact in the aligned, original-ID order."""
    prefix = export_pruned.PREFIX
    with safe_open(path, framework="pt", device="cpu") as handle:
        config = json.loads(handle.metadata()["config"])["transformer"]
        if (
            config["pruning"]["mode"] != "compact"
            or not config["video_pruning_preserve_qk_norm"]
            or config["video_pruning_shape_faithful"]
            or "per_layer_video_ffn_active_channels" in config
        ):
            raise ValueError("Compact artifact must slice both FFN projections and keep native Q/K norm")
        if (config["per_layer_video_attn1_heads"] != alignment["compact_heads"]
                or config["per_layer_ff_inner_dim"] != alignment["compact_ffn"]):
            raise ValueError("Compact artifact widths differ from the alignment")
        for block, (heads, width) in enumerate(zip(alignment["compact_heads"], alignment["compact_ffn"], strict=True)):
            keep = [i for i, bit in enumerate(alignment["masks"][f"{block}.attn1"]) if bit]
            ids = config["per_layer_video_attn1_rope_head_indices"][block]
            if (ids or list(range(32))) != keep or keep != sorted(keep) or len(keep) != heads:
                raise ValueError(f"block {block}: retained head identities differ")
            base = f"{prefix}.{block}"
            names = ("attn1.to_q.weight", "attn1.to_v.weight", "attn1.to_out.0.weight",
                     "ff.net.0.proj.weight", "ff.net.2.weight")
            shapes = {name: handle.get_slice(f"{base}.{name}").get_shape() for name in names}
            model = shapes["attn1.to_q.weight"][1]
            expected = {"attn1.to_q.weight": [32 * 128, model], "attn1.to_v.weight": [heads * 128, model],
                        "attn1.to_out.0.weight": [model, heads * 128], "ff.net.0.proj.weight": [width, model],
                        "ff.net.2.weight": [model, width]}
            if shapes != expected:
                raise ValueError(f"block {block}: compact tensor shapes differ: {shapes}")
    return dict(mode="compact", qk_norm="native_full_width", ffn_projections="both_sliced",
                heads=sum(alignment["compact_heads"]), ffn=sum(alignment["compact_ffn"]))


def prepare(source_study, plan, output, updates=500):
    started = time.monotonic()
    output, plan = Path(output).resolve(), Path(plan).resolve()
    # The isolated numerical source snapshot may already sit in `source/`.
    if output.exists() and any(item.name != "source" for item in output.iterdir()):
        raise ValueError("Use a fresh compact run directory")
    if updates < 100 or updates % 100:
        raise ValueError("Use a positive multiple of 100 recovery updates")
    parent = study_io.read_frozen(source_study, formats=("joint_study_v3",), verify_code=False)
    root = Path(parent["root"])
    owner = records.read_json(root / "monitoring/owner.json")
    if owner["status"] != "complete" or owner["update"] != 1700:
        raise ValueError("This run requires completed A update 1700")
    checkpoint = root / "training/checkpoints/update_1700.pt"
    state = joint_training.read(checkpoint, study=parent["identity"], world_size=4)
    if not state["frozen"] or not state["policy"]["aligned"]:
        raise ValueError("A masks must be final, binary and aligned")
    event_path = root / "training/events/1500.pt"
    event = torch.load(event_path, map_location="cpu", weights_only=False)
    heads, restored_heads = aligned_mask(state["heads_mask"], event["families"]["heads"]["pre_scores"], 4)
    ffn, restored_ffn = aligned_mask(state["ffn_mask"], event["families"]["channels"]["pre_scores"], 128)
    payload = dict(state, heads_mask=heads, ffn_mask=ffn)
    source = parent["components"]["transformer"]["path"]
    study_io.verify_files(list(parent["components"].values()))
    masks = export_learned_ffn.masks_from_state(payload, export_pruned.checkpoint_mask_widths(source))
    alignment = dict(
        round="up", head_multiple=4, ffn_multiple=128, restored_heads=restored_heads,
        restored_ffn=restored_ffn, source_heads=state["heads_mask"].sum(1).tolist(),
        compact_heads=heads.sum(1).tolist(), source_ffn=state["ffn_mask"].sum(1).tolist(),
        compact_ffn=ffn.sum(1).tolist(), masks=masks,
        ranking="descending final selection pre-reset score; original ID breaks ties",
    )
    records.publish_json(output / "exports/alignment.json", alignment)
    compact = output / "exports/initial_compact.safetensors"
    print("Exporting merged A with genuinely compact attention and both FFN projections", flush=True)
    export_learned_ffn.stream_merge(
        source, state["adapters"], compact, masks=masks, merge_first=True,
        pruning=dict(family="compact_A_retrain_v1", qualification="new_compact_recovery_experiment"),
    )
    export_check = verify_export(compact, alignment)
    preparation = dict(seconds=time.monotonic() - started, compact_bytes=compact.stat().st_size,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    cfg = copy.deepcopy(parent["config"])
    cfg.update(
        format="compact_config_v1", arm="compact_A_lora", selection_updates=0,
        recovery_updates=updates, test_gpu_ids=[4, 5], monitor_interval=100,
        checkpoint_interval=50, visualization_interval=0, wall_time_hours=24,
        warmup_updates=50, minimum_lr=1e-5, alignment_heads=4, alignment_ffn=128,
    )
    old_draws = records.read_json(Path(parent["draws"]["path"]))
    if len(old_draws) < updates * 4:
        raise ValueError("A does not supply the requested paired draw prefix")
    draws = output / "draws/paired_A_prefix.json"
    records.publish_json(draws, old_draws[: updates * 4])
    frozen_plan = output / "provenance/plan.md"
    records.publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    study = {k: copy.deepcopy(parent[k]) for k in (
        "components", "splits", "files", "software", "text_context", "schedule", "scale_factors", "perceptual_metric"
    )}
    study.update(
        format="compact_study_v1", root=str(output), config=cfg, code=study_io.code_identity(),
        plan=dict(path=str(frozen_plan), sha256=provenance.file_sha256(frozen_plan)),
        parent=dict(path=str(Path(source_study).resolve()), sha256=provenance.file_sha256(source_study),
                    identity=parent["identity"], input_root=parent["root"]),
        source_checkpoint=dict(path=str(checkpoint), sha256=provenance.file_sha256(checkpoint)),
        alignment_event=dict(path=str(event_path), sha256=provenance.file_sha256(event_path)),
        alignment=dict(path=str(output / "exports/alignment.json"),
                       sha256=provenance.file_sha256(output / "exports/alignment.json")),
        compact=dict(path=str(compact), sha256=provenance.file_sha256(compact)),
        draws=dict(path=str(draws), sha256=provenance.file_sha256(draws)),
        export_check=export_check, preparation=preparation,
        initialization="A adapters fused into compact BF16 base; fresh zero-output LoRA; fresh AdamW",
        input_model="native components pin saved input construction; all train forwards load compact artifact",
    )
    study["identity"] = records.digest(study)
    records.publish_json(output / "configs/frozen_study.json", study)
    summary = dict(study=study["identity"], heads=int(heads.sum()), ffn=int(ffn.sum()), root=str(output))
    print(json.dumps(summary), flush=True)
    return study


def load(path, *, verify_inputs=True, verify_code=True):
    """Training requires the frozen code; completed-study evaluation records its own executor."""
    study = study_io.read_frozen(path, formats=("compact_study_v1",), verify_code=verify_code)
    cfg = study["config"]
    if cfg["gpu_ids"] != [0, 1, 2, 3] or cfg["test_gpu_ids"] != [4, 5] or cfg["selection_updates"] != 0:
        raise ValueError("Compact frozen device or fixed-width policy differs")
    study_io.verify_files([study[k] for k in (
        "plan", "draws", "parent", "source_checkpoint", "alignment_event", "alignment", "compact"
    )])
    if verify_inputs:
        study_io.verify_files([*study["files"], *study["components"].values()])
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-study", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(8)
    prepare(args.source_study, args.plan, args.output, args.updates)


if __name__ == "__main__":
    main()
