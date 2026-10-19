"""Localize export differences on exactly one saved attention input.

This diagnostic reads a historical study as input provenance, not as a resumable
training state. It records its own code identity. Native and sliced attention
share captured x and RoPE tensors. See doc/checks/joint_faults.md.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from scripts.prune.core import ltx_adapter, provenance, session
from scripts.prune.data.learned_head_study import code_identity
from scripts.prune.learn import adapters, checkpoints, ffn_training, heads
from scripts.prune.learn.ffn import input_case


def error(a, b):
    difference = (a.float() - b.float()).abs()
    return dict(
        max_abs=float(difference.max()),
        rms=float(difference.square().mean().sqrt()),
        unequal=int((difference != 0).sum()),
        elements=difference.numel(),
    )


def run(study_path, output, gpu=4):
    heads.numerical_policy()
    torch.set_num_threads(4)
    if gpu not in (4, 5, 6):
        raise ValueError("Use physical GPUs 4–6")
    study = json.loads(Path(study_path).read_text())
    identity = study.pop("identity")
    if checkpoints.digest(study) != identity:
        raise ValueError("historical study identity changed")
    study["identity"] = identity
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    current = session.open_session(
        argparse.Namespace(model="2.5", gpu_id=gpu, seed=42),
        script="prune.checks.joint_faults",
        prompt="",
        transformer_path=Path(study["components"]["transformer"]["path"]),
    )
    case = input_case(study, current, study["splits"]["validation"][0], 2042, study["config"]["sigmas"][0])
    captured = {}
    result = dict(
        source_study=identity, code=code_identity(), input_sha256=provenance.file_sha256(case[-2] / "manifest.json")
    )
    with current.transformer() as model, torch.no_grad():
        attn = model.velocity_model.transformer_blocks[0].attn1

        def capture(module, args, kwargs):
            captured.update(x=args[0].detach().clone(), kwargs=kwargs)

        handle = attn.register_forward_pre_hook(capture, with_kwargs=True)
        ltx_adapter.velocity_forward(model, case[1])
        handle.remove()
        x = captured["x"]
        kwargs = captured["kwargs"]
        # Only the captured first attention is needed from this point onward.
        for block in model.velocity_model.transformer_blocks[1:]:
            block.to("cpu")
        torch.cuda.empty_cache()
        keep = list(range(attn.heads - 1))
        index = adapters.channel_indices(keep, attn.dim_head).to(current.device)
        compact = copy.deepcopy(attn)
        for module, axis, ids in (
            (compact.to_v, 0, index),
            (compact.to_out[0], 1, index),
            (compact.to_gate_logits, 0, torch.tensor(keep, device=current.device)),
        ):
            module.weight = nn.Parameter(module.weight.index_select(axis, ids).clone(), requires_grad=False)
            if axis == 0 and module.bias is not None:
                module.bias = nn.Parameter(module.bias.index_select(0, ids).clone(), requires_grad=False)
        compact.heads = len(keep)
        compact.rope_head_indices = keep

        def stages(a):
            q0, k0, v = a.to_q(x), a.to_k(x), a.to_v(x)
            q, k = a.preattention_function(q0, k0, a, kwargs.get("mask"), kwargs.get("pe"), kwargs.get("k_pe"))
            attention = a.attention_function(q, k, v, a.heads)
            gate = a.to_gate_logits(x)
            gated = a.gated_attention_function(x, attention, a)
            return dict(q=q, k=k, v=v, attention=attention, gate=gate, gated=gated)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            native = stages(attn)
            sliced = stages(compact)
        result["head_stages"] = {}
        for key in native:
            selected = native[key][..., keep] if key == "gate" else native[key].index_select(-1, index)
            result["head_stages"][key] = error(selected, sliced[key])
        mask = torch.ones(attn.heads, device=current.device)
        mask[-1] = 0
        mask_activation = native["gated"].reshape(*x.shape[:2], attn.heads, attn.dim_head) * mask.view(1, 1, -1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = attn.to_out[0](mask_activation.flatten(-2))
            b = compact.to_out[0](sliced["gated"])
        result["head_stages"]["out"] = error(a, b)
        # Repeat each projection at FP32 on the same input to separate geometry
        # effects from index/order errors. Do not compare different layer inputs.
        result["fp32_controls"] = {}
        for key, full, small, inp in (
            ("v", attn.to_v, compact.to_v, x),
            ("gate", attn.to_gate_logits, compact.to_gate_logits, x),
            ("out", attn.to_out[0], compact.to_out[0], mask_activation.flatten(-2)),
        ):
            full_out = torch.nn.functional.linear(inp.float(), full.weight.float(), full.bias.float())
            small_in = inp.index_select(-1, index) if key == "out" else inp
            small_out = torch.nn.functional.linear(small_in.float(), small.weight.float(), small.bias.float())
            expected = (
                full_out if key == "out" else full_out[..., keep] if key == "gate" else full_out.index_select(-1, index)
            )
            result["fp32_controls"][key] = error(expected, small_out)
        # Preserve the exact first divergent projection's input and weights for
        # small local kernel experiments without another full model load.
        fixture = dict(
            x=x.cpu(),
            mask_activation=mask_activation.flatten(-2).cpu(),
            keep=keep,
            tensors={name: t.detach().cpu() for name, t in attn.state_dict().items()},
            input_sha256=result["input_sha256"],
        )
        with (output / "first_attention.pt").open("wb") as stream:
            torch.save(fixture, stream)
        learned = ffn_training.read(Path(study["parent"]["input_root"]) / "training/checkpoints/update_0050.pt")[
            "adapters"
        ]
        result["adapter_stages"] = {}
        for kind, base, inp in (("v", attn.to_v, x), ("out", attn.to_out[0], mask_activation.flatten(-2))):
            module = adapters.ProjectionAdapter(base, 128)
            module.A.copy_(learned[f"0.attn1.{kind}.A"])
            module.B.copy_(learned[f"0.attn1.{kind}.B"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                split = module(inp)
                fused = torch.nn.functional.linear(inp, module.merged_weight(), base.bias)
            inp32 = inp.float()
            bias32 = None if base.bias is None else base.bias.float()
            split32 = torch.nn.functional.linear(inp32, base.weight.float(), bias32) + torch.nn.functional.linear(
                torch.nn.functional.linear(inp32, module.A), module.B
            )
            merged32 = torch.nn.functional.linear(inp32, base.weight.float() + module.B @ module.A, bias32)
            result["adapter_stages"][kind] = dict(bf16=error(split, fused), fp32=error(split32, merged32))
    checkpoints.publish_json(output / "diagnosis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "code"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-id", type=int, default=4)
    a = p.parse_args()
    run(a.study, a.output, a.gpu_id)


if __name__ == "__main__":
    main()
