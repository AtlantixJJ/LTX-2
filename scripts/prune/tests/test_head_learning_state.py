import pytest
import torch

from scripts.prune.learn import checkpoints
from scripts.prune.learn.adapters import ProjectionAdapter, load_state
from scripts.prune.learn.gates import HeadGates


def test_feasibility_zero_adapter_control_does_not_retain_graph(tmp_path, monkeypatch):
    """Exercise the S1 control with real trainable adapter tensors before backward."""
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    from scripts.prune.learn import heads

    class BeforeBackward(Exception):
        pass

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 4, bias=False)
            self.velocity_model = SimpleNamespace(set_gradient_checkpointing=lambda enabled: None)
            self.controls = []

        def train(self, mode=True):
            if mode and isinstance(self.projection, ProjectionAdapter):
                raise BeforeBackward
            return super().train(mode)

        def forward(self, video, **kwargs):
            value = self.projection(video.latent)
            if isinstance(self.projection, ProjectionAdapter):
                self.controls.append((torch.is_grad_enabled(), value.requires_grad))
            return value, None

    model = Model()

    @contextmanager
    def resident(*args):
        with torch.no_grad():
            yield model

    def attach(candidate, seed, rank):
        candidate.projection = ProjectionAdapter(candidate.projection, rank=2)
        return {"0.attn1.v": candidate.projection}

    video = SimpleNamespace(latent=torch.randn(1, 2, 4))
    grid = SimpleNamespace(tokens_per_latent_frame=1)
    zero = torch.zeros_like(video.latent)
    monkeypatch.setattr(heads, "input_case", lambda *a: (grid, video, zero[:, :1], zero, zero, None, None))
    monkeypatch.setattr(heads.adapters, "attach", attach)
    monkeypatch.setattr(heads.hooks, "attach_head_provider", lambda *a: nullcontext())
    monkeypatch.setattr(
        heads.ltx_adapter, "velocity_forward", lambda candidate, v: (candidate(video=v)[0], candidate(video=v)[0])
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    current = SimpleNamespace(device="cpu", transformer=resident)
    study = {
        "identity": "tiny-feasibility",
        "root": str(tmp_path),
        "components": {"transformer": {"path": "unused"}},
        "splits": {"learning": [{"n_latent_frames": n} for n in (18, 28)]},
        "config": {
            "adapter_lr": 0.0001,
            "gate_lr": 0.01,
            "betas": [0.9, 0.999],
            "adam_epsilon": 1e-8,
            "sigmas": [1.0],
            "rank": 2,
        },
    }
    with pytest.raises(BeforeBackward):
        heads.feasibility(study, current, float("inf"))
    assert len(model.controls) == 4
    assert all(not grad_enabled and not retained_graph for grad_enabled, retained_graph in model.controls)


def test_resume_identical_optimizer_and_rng(tmp_path):
    torch.manual_seed(42)
    a = ProjectionAdapter(torch.nn.Linear(4, 4), 2)
    g = HeadGates(1, 4, [[2, 4]])
    modules = {"0.attn1.v": a}
    ao = torch.optim.AdamW([a.A, a.B], lr=0.001)
    go = torch.optim.AdamW([g.alpha], lr=0.01)
    identity = {"prefix": "one"}

    def update():
        ao.zero_grad()
        go.zero_grad()
        x = torch.randn(1, 3, 4)
        loss = a(x).square().mean() + g.alpha.sigmoid().mean()
        loss.backward()
        ao.step()
        go.step()

    update()
    path = tmp_path / "update.pt"
    checkpoints.save(
        path,
        identity=identity,
        update=1,
        phase="soft",
        gates=g,
        adapters=modules,
        adapter_optimizer=ao,
        gate_optimizer=go,
    )
    update()
    expected = a.B.detach().clone(), g.alpha.detach().clone(), torch.get_rng_state()
    state = checkpoints.load(path, identity)
    load_state(modules, state["adapters"])
    g.alpha.data.copy_(state["alpha"])
    ao.load_state_dict(state["adapter_optimizer"])
    go.load_state_dict(state["gate_optimizer"])
    checkpoints.restore_rng(state["rng"])
    update()
    assert torch.equal(a.B, expected[0]) and torch.equal(g.alpha, expected[1])
    assert torch.equal(torch.get_rng_state(), expected[2])
    with pytest.raises(ValueError, match="identity"):
        checkpoints.load(path, {"prefix": "changed"})
    with pytest.raises(ValueError, match="700"):
        checkpoints.verify_prefix(state, "one")


@pytest.mark.parametrize("gt_only", [False, True])
def test_real_phase_loop_matches_interrupted_resume(tmp_path, monkeypatch, gt_only):
    import argparse
    import copy
    import json
    import time
    from contextlib import contextmanager
    from types import SimpleNamespace

    from ltx_core.model.transformer.model import X0Model
    from scripts.prune.learn import heads

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads, self.dim_head = 4, 1
            self.to_v = torch.nn.Linear(4, 4)
            self.to_out = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def forward(self, value):
            return self.to_out[0](self.to_v(value))

    class Velocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.attn1, block.attn2 = Attention(), Attention()
            self.transformer_blocks = torch.nn.ModuleList([block])

        def set_gradient_checkpointing(self, _):
            pass

        def forward(self, video, audio, perturbations):
            return self.transformer_blocks[0].attn1(video.latent), None

    torch.manual_seed(5)
    native = X0Model(Velocity())
    base_before = {n: p.clone() for n, p in native.named_parameters()}

    @contextmanager
    def transformer(_):
        yield copy.deepcopy(native)

    current = SimpleNamespace(device=torch.device("cpu"), transformer=transformer)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    monkeypatch.setattr(heads.whole_clip, "load_manifest", lambda _: {"videos": [{}]})
    monkeypatch.setattr(heads.whole_clip, "load_epsilon", lambda *_: torch.ones(1, 6, 4))
    monkeypatch.setattr(heads, "monitor", lambda *args: None)
    config = {
        "blocks": 1,
        "heads": 4,
        "allowed_counts": [2, 4],
        "removed_heads": [2],
        "rank": 2,
        "adapter_lr": 0.001,
        "gate_lr": 0.01,
        "betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "warmup_updates": 50,
        "lambda_max": 0.001,
        "sigmas": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "soft_updates": 700,
        "binary_updates": 300,
        "compact_updates": 500,
    }
    if gt_only:
        config["monitor_reference"] = "gt_capture"
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "manifest.json").write_text("{}")

    def inputs(*args):
        y, e = torch.zeros(1, 6, 4), torch.ones(1, 6, 4)
        value = torch.full_like(y, 0.5)
        value[:, :2] = y[:, :2]
        video = SimpleNamespace(latent=value, timesteps=torch.full((1, 6, 1), 0.5))
        return (
            SimpleNamespace(tokens_per_latent_frame=2),
            video,
            y[:, :2],
            y,
            e,
            input_root,
            {"videos": [{"artifacts": {"epsilon_sha256": "noise"}}]},
        )

    monkeypatch.setattr(heads, "input_case", inputs)
    draws_path = tmp_path / "draws.json"
    draws_path.write_text(json.dumps([{"view": "case", "noise_seed": i, "sigma_index": i % 8} for i in range(8)]))
    paths = []
    for label in ("continuous", "resumed"):
        root = tmp_path / label
        for name, data in (
            ("checks/feasibility.json", {"study": "one", "passed": True}),
            ("checks/execution_probes.json", {"study": "one", "speed_layout_passed": True}),
            ("teacher/validation_completion.json", {"study": "one"}),
        ):
            if gt_only and name != "checks/feasibility.json":
                continue
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data))
        study = {
            "root": str(root),
            "identity": "one",
            "config": config,
            "code": "code",
            "software": {},
            "text_context": {"tensor_sha256": "context"},
            "components": {"transformer": {"path": "unused"}},
            "region_tracks": {"status": "not_verified"},
            "splits": {"learning": [{"relative_dir": "case"}]},
            "draws": {"42:ordinary": {"path": str(draws_path), "sha256": "draws"}},
        }
        args = argparse.Namespace(
            arm="learned_joint",
            phase="selection",
            replicate_seed=42,
            removed_heads=2,
            quality_research=False,
            branch_prefix=None,
            max_updates=2 if label == "continuous" else 1,
        )
        heads.training_phase(study, current, args, time.monotonic() + 30)
        if label == "resumed":
            heads.training_phase(study, current, args, time.monotonic() + 30)
        paths.append(root / "training/learned_joint/replicate_42/checkpoints/update_0002.pt")
    a, b = [checkpoints.load(path) for path in paths]
    traces = [
        [torch.load(p.parent.parent / f"gate_trace/{u:04d}.pt", weights_only=True) for u in (1, 2)] for p in paths
    ]
    for continuous, resumed in zip(*traces, strict=True):
        assert continuous.shape == (3, 1, 4) and torch.equal(continuous, resumed)
    assert torch.equal(traces[0][1][0], a["alpha"])  # trace row 0 is the logit state after the update
    assert torch.equal(a["alpha"], b["alpha"])
    assert all(torch.equal(a["adapters"][k], b["adapters"][k]) for k in a["adapters"])
    assert all(torch.equal(base_before[n], p) for n, p in native.named_parameters())
