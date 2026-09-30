"""Profile one held-out refiner state in functional-mask and exported-checkpoint modes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from scripts.prune.core import artifacts, provenance, session
from scripts.prune.data import chunk_states, records
from scripts.prune.score import hooks


@contextmanager
def _ranges(transformer):
    """Add profiler scopes around existing calls without replacing any computation."""
    original = []

    def wrap(obj, attr, label):
        function = getattr(obj, attr)

        def timed(*args, **kwargs):
            with record_function(label):
                return function(*args, **kwargs)

        setattr(obj, attr, timed)
        original.append((obj, attr, function))

    for name, attention in hooks.iter_video_attention(transformer):
        prefix = f"layer/{name}"
        wrap(attention, "forward", prefix + "/total")
        for projection in ("to_q", "to_k", "to_v"):
            wrap(getattr(attention, projection), "forward", prefix + "/" + projection)
        wrap(attention.to_out[0], "forward", prefix + "/to_out")
        wrap(attention, "preattention_function", prefix + "/preattention")
        wrap(attention, "attention_function", prefix + "/attention_kernel")
        wrap(attention, "masked_attention_function", prefix + "/masked_attention_kernel")
        if attention.to_gate_logits is not None:
            wrap(attention.to_gate_logits, "forward", prefix + "/gate_projection")
            wrap(attention, "gated_attention_function", prefix + "/gate_apply")
    for name, ff in hooks.iter_video_ffn(transformer):
        wrap(ff, "forward", f"layer/{name}/total")
        wrap(ff.net[0].proj, "forward", f"layer/{name}/input_projection")
        wrap(ff.net[2], "forward", f"layer/{name}/output_projection")
    try:
        yield
    finally:
        for obj, attr, function in reversed(original):
            setattr(obj, attr, function)


def _summarize(profiler) -> dict:
    rows = {}
    for event in profiler.key_averages():
        if event.key.startswith("layer/"):
            rows[event.key] = {"device_ms": event.device_time_total / 1000,
                               "cpu_ms": event.cpu_time_total / 1000, "calls": event.count}
    groups = defaultdict(float)
    for name, row in rows.items():
        part = name.rsplit("/", 1)[-1]
        groups[part] += row["device_ms"]
    return {"groups_device_ms": dict(sorted(groups.items())), "ranges": rows}


def _bench_attentions(transformer, inputs, device, repeats=10):
    """Compare head selection with full attention plus the identical output mask."""
    attention = dict(hooks.iter_video_attention(transformer))
    results = {}
    for name, (args, kwargs) in inputs.items():
        module = attention[name]
        active = module.active_head_indices
        original_selection = module.select_active_heads

        def bench(sparse):
            module.select_active_heads = sparse
            try:
                for _ in range(2):
                    result = module(*args, **kwargs)
                torch.cuda.synchronize(device)
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    result = module(*args, **kwargs)
                    torch.cuda.synchronize(device)
                    times.append((time.perf_counter() - start) * 1000)
                return result.detach(), sorted(times)[len(times) // 2]
            finally:
                module.select_active_heads = original_selection

        sparse_result, sparse_ms = bench(True)
        full_result, full_ms = bench(False)
        results[name] = {"active_heads": len(active), "sparse_ms": sparse_ms, "masked_full_ms": full_ms,
                         "max_abs": float((sparse_result.float() - full_result.float()).abs().max())}
    return results


def _run(current, path, state, meta, masks=None, repeats=5, microbench=False):
    with current.transformer(path) as transformer:
        with ExitStack() as stack:
            captured = {}
            capture_handles = []
            if microbench:
                for name, attention in hooks.iter_video_attention(transformer):
                    if attention.active_head_indices is not None:
                        def capture(_module, values, kwargs, *, name=name):
                            captured.setdefault(name, (values, kwargs))
                        capture_handles.append(attention.register_forward_pre_hook(capture, with_kwargs=True))
            if masks:
                head = {name: torch.tensor(values, device=current.device) for name, values in masks.items()
                        if not name.endswith(".ff")}
                ff = {name: torch.tensor(values, device=current.device) for name, values in masks.items()
                      if name.endswith(".ff")}
                if head:
                    stack.enter_context(hooks.attach_head_masks(transformer, head, requires_grad=False))
                if ff:
                    stack.enter_context(hooks.attach_ffn_masks(transformer, ff, requires_grad=False))
            def forward():
                return current.denoiser(transformer, state, None, current.sigmas, meta.step_index)[0].denoised

            forward()
            for handle in capture_handles:
                handle.remove()
            torch.cuda.synchronize(current.device)
            wall = []
            for _ in range(repeats):
                start = time.perf_counter()
                forward()
                torch.cuda.synchronize(current.device)
                wall.append((time.perf_counter() - start) * 1000)
            with _ranges(transformer), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                output = forward().detach().cpu()
                torch.cuda.synchronize(current.device)
            microbench_results = _bench_attentions(transformer, captured, current.device) if microbench else None
    return output, {"wall_ms": wall, "median_wall_ms": sorted(wall)[len(wall) // 2],
                    "masked_attention_microbench": microbench_results, **_summarize(profiler)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    session.add_model_args(parser)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--exported-checkpoint", type=Path, required=True)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    source = session.open_session(args, script="profile_export")
    exported = session.open_session(args, script="profile_export", transformer_path=args.exported_checkpoint)
    record = records.select(source.states_root(args.states), split="held_out", limit=1)[0]
    state, _, meta = chunk_states.load_record(record, source.device)
    from scripts.prune.score.export_pruned import checkpoint_mask_widths

    masks, mask_sha256 = hooks.read_mask_artifact(
        args.masks, model_key=source.key, fingerprint=source.stamp()["transformer_fingerprint"],
        widths=checkpoint_mask_widths(source.model.paths.transformer())
    )
    original, baseline = _run(source, None, state, meta, masks=masks, repeats=args.repeats)
    compact, candidate = _run(exported, args.exported_checkpoint, state, meta, repeats=args.repeats, microbench=True)
    max_abs = float((original.float() - compact.float()).abs().max())
    report = {"provenance": source.stamp(), "exported_fingerprint": provenance.checkpoint_fingerprint(
        args.exported_checkpoint), "mask_sha256": mask_sha256, "record": record.name,
        "max_abs": max_abs, "baseline": baseline, "candidate": candidate}
    output = artifacts.run_dir(source.key, "export-profile", script="profile_export", argv=sys.argv[1:])
    path = output / "export_profile.json"
    path.write_text(json.dumps(report, indent=2))
    print(path)
    return 0 if max_abs <= 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
