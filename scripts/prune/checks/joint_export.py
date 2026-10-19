"""Isolate native export transitions without changing the 0.02 acceptance limit.

The same saved case passes through functional adapters, FP32 accumulation then
BF16 merged weights, full-width reload and uneven compact reload. Layer hooks
record residual outputs to locate the first divergent block. See design page.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, provenance, session
from scripts.prune.data import joint_study
from scripts.prune.learn import adapters, checkpoints, ffn_training, heads, joint_training
from scripts.prune.learn.ffn import input_case
from scripts.prune.score import export_learned_ffn, export_pruned, hooks


def run(study, output, *, gpu=6):
    if gpu not in (4, 5, 6):
        raise ValueError("export tests use GPUs 4–6")
    heads.numerical_policy()
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    current = session.open_session(
        argparse.Namespace(model="2.5", gpu_id=gpu, seed=42),
        script="prune.checks.joint_export",
        prompt="",
        transformer_path=Path(study["components"]["transformer"]["path"]),
    )
    source = study["components"]["transformer"]["path"]
    case = input_case(study, current, study["splits"]["validation"][0], 2042, joint_study.SIGMAS[0])
    widths = export_pruned.checkpoint_mask_widths(source)
    all_masks = {k: [1] * w for k, w in widths.items()}
    parent_root = Path(study["parent"]["input_root"])
    adapter_checkpoint = parent_root / "training/checkpoints/update_0050.pt"
    learned = ffn_training.read(adapter_checkpoint)["adapters"]
    result = dict(
        study=study["identity"],
        input_manifest=str(case[-2] / "manifest.json"),
        input_sha256=provenance.file_sha256(case[-2] / "manifest.json"),
        tolerance=0.02,
        adapter_checkpoint=str(adapter_checkpoint),
        adapter_sha256=provenance.file_sha256(adapter_checkpoint),
        transitions=[],
        controlled_precision=[],
    )

    def predict(model, label, *, reference=None):
        trace_dir = output / "activation_traces" / label
        trace_dir.mkdir(parents=True, exist_ok=True)
        handles, errors = [], []

        def capture(block_id):
            def hook(module, args, result):
                value = result[0] if isinstance(result, tuple) else result
                value = value.x if hasattr(value, "x") else value
                if not isinstance(value, torch.Tensor):
                    raise ValueError("native block output is not the expected video tensor")
                value = value.detach().cpu()
                if reference:
                    saved = torch.load(
                        output / "activation_traces" / reference / f"{block_id:02d}.pt", weights_only=True
                    )
                    error = float((saved.float() - value.float()).abs().max())
                    errors.append(dict(block=block_id, max_abs=error))
                else:
                    checkpoints.publish_tensor(trace_dir / f"{block_id:02d}.pt", value)

            return hook

        # Full residual outputs share the same 4096 coordinates after surgery.
        for block_id, block in enumerate(model.velocity_model.transformer_blocks):
            handles.append(block.register_forward_hook(capture(block_id)))
        try:
            _, clean = ltx_adapter.velocity_forward(model, case[1])
        finally:
            for handle in handles:
                handle.remove()
        if reference:
            first = next((row for row in errors if row["max_abs"] > 0), None)
            checkpoints.publish_json(
                output / f"{label}_layer_errors.json", dict(reference=reference, errors=errors, first_divergence=first)
            )
        clean = common.with_clean_prefix(clean, case[2]).cpu()
        checkpoints.publish(output / f"{label}.pt", lambda p: torch.save(clean, p))
        return clean

    def compare(a, b, label):
        difference = (a.float() - b.float()).abs()
        error = float(difference.max())
        result["transitions"].append(
            dict(name=label, max_abs=error, passed=bool(torch.isfinite(difference).all() and error <= 0.02))
        )
        checkpoints.publish_json(
            output / "progress.json"
            if not (output / "progress.json").exists()
            else output / f"transition_{len(result['transitions'])}.json",
            result["transitions"][-1],
        )

    def compact_masks(kind):
        masks = {k: list(v) for k, v in all_masks.items()}
        for name, values in masks.items():
            b, family = name.split(".")
            if family == "attn1" and kind in ("head", "joint"):
                values[-1] = 0
            if family == "ff" and kind in ("ffn", "joint"):
                removed = 128 * (1 + int(b) % 3)
                values[-removed:] = [0] * removed
        return masks

    for enabled in (False, True):
        label = "adapters_on" if enabled else "adapters_off"
        with current.transformer() as model, torch.no_grad():
            step = joint_training.JointStep(model, study["config"])
            if enabled:
                adapters.load_state(step.attached, learned)
            step.configure(0, all_one=True)
            step.masks()
            split = predict(model, label + "_split")
            # Controlled per-projection calculation isolates rounding from merge
            # order on the first nonzero adapter. Record FP32 and BF16 arithmetic
            # on the SAME captured activation, rather than blaming dtype by fiat.
            diagnostics = []
            handles = []

            def precision_hook(name, adapter):
                def observe(module, args, output_tensor):
                    x = args[0].detach().reshape(-1, args[0].shape[-1])[:32]
                    base = adapter.base.weight.float()
                    exact = torch.nn.functional.linear(
                        x.float(),
                        base + adapter.B @ adapter.A,
                        None if adapter.base.bias is None else adapter.base.bias.float(),
                    )
                    split32 = (
                        torch.nn.functional.linear(
                            x.float(), base, None if adapter.base.bias is None else adapter.base.bias.float()
                        )
                        + (x.float() @ adapter.A.T) @ adapter.B.T
                    )
                    merged16 = torch.nn.functional.linear(x, adapter.merged_weight(), adapter.base.bias).float()
                    diagnostics.append(
                        dict(
                            name=name,
                            fp32_split_merged_max_abs=float((split32 - exact).abs().max()),
                            bf16_merged_fp32_max_abs=float((merged16 - exact).abs().max()),
                            bf16_split_fp32_max_abs=float(
                                (output_tensor.detach().reshape(-1, output_tensor.shape[-1])[:32].float() - exact)
                                .abs()
                                .max()
                            ),
                        )
                    )

                return observe

            for name, adapter in step.attached.items():
                handles.append(adapter.register_forward_hook(precision_hook(name, adapter)))
            predict(model, label + "_precision_control", reference=label + "_split")
            for handle in handles:
                handle.remove()
            result["controlled_precision"].append(dict(adapters=enabled, projections=diagnostics))
            for adapter in step.attached.values():
                adapter.base.weight.copy_(adapter.merged_weight())
                adapter.enabled = False
            fused = predict(model, label + "_merged", reference=label + "_split")
            compare(split, fused, label + "_split_to_merged")
            step.close()
            del step, model
        gc.collect()
        torch.cuda.empty_cache()
        merged_path = output / f"{label}_merged.safetensors"
        export_learned_ffn.merge(source, learned if enabled else {}, merged_path)
        with current.transformer(merged_path) as model, torch.no_grad():
            reloaded = predict(model, label + "_reloaded", reference=label + "_split")
        del model
        compare(fused, reloaded, label + "_merged_to_reload")
        gc.collect()
        torch.cuda.empty_cache()
        for kind in ("zero", "head", "ffn", "joint"):
            masks = all_masks if kind == "zero" else compact_masks(kind)
            compact_path = output / f"{label}_{kind}.safetensors"
            export_pruned.export(merged_path, masks, compact_path, model_key="2.5", mode="compact")
            with (
                current.transformer(merged_path) as model,
                torch.no_grad(),
                hooks.attach_head_masks(model, {k: torch.tensor(v) for k, v in masks.items()}, requires_grad=False),
                hooks.attach_ffn_masks(model, {k: torch.tensor(v) for k, v in masks.items()}, requires_grad=False),
            ):
                functional = predict(model, label + "_" + kind + "_functional")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            with current.transformer(compact_path) as model, torch.no_grad():
                compact = predict(model, label + "_" + kind + "_compact", reference=label + "_" + kind + "_functional")
            del model
            compare(functional, compact, label + "_" + kind + "_functional_to_compact")
            gc.collect()
            torch.cuda.empty_cache()
            # Preserve predictions, exact mask IDs and hashes; disposable full
            # checkpoint payloads are rebuildable and occupy tens of GiB each.
            checkpoints.publish_json(
                output / f"{label}_{kind}_inventory.json",
                dict(
                    masks=masks,
                    compact_sha256=provenance.file_sha256(compact_path),
                    source_sha256=provenance.file_sha256(merged_path),
                ),
            )
            compact_path.unlink()
        merged_path.unlink()
    result["passed"] = all(row["passed"] for row in result["transitions"])
    checkpoints.publish_json(output / "acceptance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", default=6, type=int)
    args = parser.parse_args()
    run(joint_study.load(args.study), args.output, gpu=args.gpu_id)


if __name__ == "__main__":
    main()
