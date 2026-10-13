from scripts.prune.learn.heads import probe_layouts, timing_decision


def test_benchmark_refuses_nonfinite_outputs_and_differences(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest
    import torch

    from scripts.prune.evaluate.bench_whole_clip_d0 import _one_arm

    class Model(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, **kwargs):
            return self.value, None

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    grid = SimpleNamespace(latent_frames=2, unpatchify_block=lambda value, *_: value)
    for recorded, value in ((float("nan"), 1.0), (1.0, float("nan")), (3e38, -3e38)):
        expected = tmp_path / "expected.pt"
        tensor = torch.full((1, 4, 2), value)
        tensor[:, :1] = 0
        saved = torch.full((1, 4, 2), recorded)
        saved[:, :1] = 0
        torch.save(saved, expected)
        current = SimpleNamespace(device=torch.device("cpu"), transformer=lambda *_: nullcontext(Model(tensor)))
        with pytest.raises(ValueError, match="nonfinite"):
            _one_arm(
                current,
                tmp_path / "weights",
                expected,
                grid,
                None,
                torch.zeros(1, 1, 2),
                warmup=2,
                repeats=7,
                label="test",
            )


def test_exact_probe_budgets_and_drift_gate():
    masks = probe_layouts({"ffn_probe_widths": [16384, 15872, 15360, 14848]})
    for name in ("38x4", "19x8", "mixed"):
        assert sum(values.count(0) for values in masks[name].values()) == 152
    for width in (16384, 15872, 15360, 14848):
        assert all(sum(values) == width for values in masks[f"ffn_{width}"].values())
    assert timing_decision([{"median_ms": m} for m in (100, 90, 101)], "ABA")["passed"]
    assert not timing_decision([{"median_ms": m} for m in (100, 98, 102)], "ABA")["passed"]
