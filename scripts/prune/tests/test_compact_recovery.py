"""Aligned recovery preserves units, base weights and exact adapter replay."""

from types import SimpleNamespace

import pytest
import torch

from scripts.prune.core import records
from scripts.prune.data.compact_study import aligned_mask
from scripts.prune.learn import adapters, checkpoints, compact_training
from scripts.prune.learn.compact import CompactSession


def test_alignment_restores_highest_scores_without_removing_retained_units():
    mask = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    scores = torch.tensor([[0., 3., 2., 0., 2., 1., -1., -2.], [0.] * 8])
    actual, restored = aligned_mask(mask, scores, 4)
    assert restored == [[1, 2], []]
    assert actual.sum(1).tolist() == [4, 4]
    assert (actual[mask.bool()] == 1).all()


@pytest.mark.parametrize("bad", ["nan", "empty", "binary", "oversized"])
def test_invalid_alignment_is_rejected(bad):
    mask = torch.tensor([[1., 0., 0.]])
    scores = torch.ones_like(mask)
    multiple = 2
    if bad == "nan":
        scores[0, 0] = float("nan")
    elif bad == "empty":
        mask.zero_()
    elif bad == "binary":
        mask[0, 0] = 0.2
    else:
        multiple = 4
    with pytest.raises(ValueError):
        aligned_mask(mask, scores, multiple)


def test_compact_loader_keeps_input_model_and_selects_explicit_artifact():
    native = SimpleNamespace(model=object(), transformer=lambda path, **kwargs: path)
    current = CompactSession(native, "/compact.safetensors")
    assert current.model is native.model
    assert str(current.transformer()) == "/compact.safetensors"
    assert current.transformer("/other.safetensors") == "/other.safetensors"


def test_compact_zero_output_frozen_base_and_exact_adam_resume(monkeypatch, tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, value):
            return self.linear(value)

    def attach(model, seed, rank):
        model.linear = adapters.ProjectionAdapter(model.linear, rank, torch.Generator().manual_seed(seed))
        return {"0.attn1.v": model.linear}

    monkeypatch.setattr(compact_training.adapters, "attach", attach)
    monkeypatch.setattr(compact_training.adapters, "attach_ffn", lambda *args: {})
    monkeypatch.setattr(compact_training.ltx_adapter, "velocity_forward", lambda model, video: (model(video), None))
    torch.manual_seed(18)
    model = Tiny()
    video, capture, epsilon = [torch.randn(1, 5, 4) for _ in range(3)]
    native = model(video).detach().clone()
    cfg = dict(seed=42, head_rank=2, ffn_rank=2, warmup_updates=2,
               recovery_updates=6, adapter_lr=1e-4, minimum_lr=1e-5)
    step = compact_training.JointStep(model, cfg)
    assert torch.equal(step.model(video), native)
    before = step.model.linear.base.weight.detach().clone()
    ao, go = compact_training.optimizers(step)
    case = SimpleNamespace(tokens_per_latent_frame=1), video, None, capture, epsilon, None, None
    first = compact_training.update(step, step, case, 1, ao, go)
    assert first["adapter_gradient_norm"] > 0 and first["penalty"] == 0
    payload = compact_training.state(step, ao, go, study="unit", update=1, world_size=4,
                                     rng=[checkpoints.rng_state()] * 4, input_chain="test")
    path = tmp_path / "update_0001.pt"
    compact_training.save(path, payload)
    second = compact_training.update(step, step, case, 2, ao, go)
    expected = records.unchanged_state_hash(
        dict(adapters=adapters.state_dict(step.attached), optimizer=ao.state_dict())
    )
    restored = compact_training.read(path, study="unit", world_size=4)
    compact_training.restore(step, ao, go, restored)
    assert compact_training.update(step, step, case, 2, ao, go) == second
    actual = records.unchanged_state_hash(dict(adapters=adapters.state_dict(step.attached), optimizer=ao.state_dict()))
    assert actual == expected
    assert torch.equal(step.model.linear.base.weight, before)


def test_compact_learning_rate_reaches_frozen_endpoints():
    cfg = dict(warmup_updates=50, recovery_updates=500, adapter_lr=1e-4, minimum_lr=1e-5)
    assert compact_training.learning_rate(cfg, 1) == 2e-6
    assert compact_training.learning_rate(cfg, 50) == 1e-4
    assert compact_training.learning_rate(cfg, 500) == 1e-5


def _artifact(path, *, heads, keep, width, faithful=False, out_heads=None):
    import json

    from safetensors.torch import save_file

    prefix = "model.diffusion_model.transformer_blocks.0"
    model = 8
    tensors = {
        f"{prefix}.attn1.to_q.weight": torch.zeros(32 * 128, model),
        f"{prefix}.attn1.to_v.weight": torch.zeros(heads * 128, model),
        f"{prefix}.attn1.to_out.0.weight": torch.zeros(model, (out_heads or heads) * 128),
        f"{prefix}.ff.net.0.proj.weight": torch.zeros(width, model),
        f"{prefix}.ff.net.2.weight": torch.zeros(model, width),
    }
    config = dict(transformer=dict(
        pruning=dict(mode="compact"), video_pruning_preserve_qk_norm=True, video_pruning_shape_faithful=faithful,
        per_layer_video_attn1_heads=[heads], per_layer_ff_inner_dim=[width],
        per_layer_video_attn1_rope_head_indices=[keep if heads < 32 else None],
    ))
    save_file(tensors, str(path), metadata={"config": json.dumps(config)})


def test_export_check_requires_compact_shapes_ids_and_native_qk(tmp_path):
    from scripts.prune.data.compact_study import verify_export

    keep = [0, 3, 5, 9]
    alignment = dict(compact_heads=[4], compact_ffn=[256],
                     masks={"0.attn1": [int(i in keep) for i in range(32)]})
    good = tmp_path / "good.safetensors"
    _artifact(good, heads=4, keep=keep, width=256)
    assert verify_export(good, alignment)["heads"] == 4
    for name, kwargs in (("faithful", dict(faithful=True)), ("out", dict(out_heads=32)),
                         ("ids", dict(keep=[0, 3, 5, 10]))):
        bad = tmp_path / f"{name}.safetensors"
        _artifact(bad, **{**dict(heads=4, keep=keep, width=256), **kwargs})
        with pytest.raises(ValueError):
            verify_export(bad, alignment)


def test_media_titles_fit_the_comparison_layout():
    from scripts.onestep_avatar import media
    from scripts.prune.evaluate.compact import MEDIA_QUESTION, MEDIA_TITLES

    titles = [("capture_rgb", "Capture RGB"), ("capture_vae", "Capture VAE"), *MEDIA_TITLES]
    panels = [media.Panel(role, title, None, (0,), value="RGB" if role == "capture_rgb" else "VAE decoding")
              for role, title in titles]
    # Same geometry as ffn.compare_case; raises when a title is wider than its panel.
    media.layout_geometry(panels, question=MEDIA_QUESTION, layout="comparison", panel_size=(384, 384),
                          viewing_width=480)


def test_media_shards_cover_each_test_source_once():
    from scripts.prune.evaluate.compact import media_keys

    sources = [dict(actor=f"a{n}", relative_dir=f"v{n}", n_latent_frames=28) for n in range(6)]
    study = dict(splits=dict(test=sources), config=dict(sigmas=[0.9, 0.7, 0.5], monitor_sigma_indices=[0, 1, 2]))
    keys = media_keys(study)
    assert len(keys) == 18
    assert len({key for _, _, key in keys}) == 18
    shards = [[s for n, s in enumerate(sources) if n % 4 == shard] for shard in range(4)]
    assert sorted(id(s) for part in shards for s in part) == sorted(id(s) for s in sources)
