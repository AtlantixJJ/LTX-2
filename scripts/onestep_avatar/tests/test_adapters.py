"""The deployed unmerged function equals training with the same saved matrices."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, common
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.tests.test_checkpoint_contract import _contract
from scripts.onestep_avatar.training.checkpoints import CONTRACT_KEY


def exported(model):
    return {name.replace('base_model.model.', 'diffusion_model.', 1): value.to(torch.bfloat16).contiguous()
            for name, value in get_peft_model_state_dict(model).items()}


@pytest.fixture
def saved(tmp_path):
    torch.manual_seed(4)
    base = _model().to(dtype=torch.bfloat16)
    state = {name: value.clone() for name, value in base.state_dict().items()}
    model = adapters.attach(base, rank=2, alpha=2, target='attn', init_seed=9)
    for name, value in model.named_parameters():
        if '.lora_B.' in name:
            with torch.no_grad():
                value.fill_(0.125)
    path = tmp_path / 'adapter.safetensors'
    weights = exported(model)
    contract = _contract()
    contract['adapter']['tensor_shapes'] = {name: list(value.shape) for name, value in weights.items()}
    save_file(weights, path, metadata={CONTRACT_KEY: json.dumps(contract)})
    adapters.load_weights(model, path)
    return model, state, path, contract


@pytest.mark.parametrize('zero', [False, True])
def test_unmerged_inference_matches_training_reference(saved, monkeypatch, zero):
    from ltx_core.model.transformer.model import X0Model
    from ltx_trainer import model_loader
    reference, state, path, contract = saved
    if zero:
        weights = exported(reference)
        for name, value in weights.items():
            if '.lora_B.' in name:
                value.zero_()
        save_file(weights, path, metadata={CONTRACT_KEY: json.dumps(contract)})
        adapters.load_weights(reference, path)
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        base = _model().to(dtype=torch.bfloat16)
        base.load_state_dict(state)
        return base

    monkeypatch.setattr(model_loader, 'load_transformer', load)
    session = SimpleNamespace(device=torch.device('cpu'), model=SimpleNamespace(
        paths=SimpleNamespace(transformer=lambda: 'checked-base')),
        transformer=lambda **kw: pytest.fail('unmerged inference fused the adapter'))
    grid = _grid(_geometry())
    source = torch.randn(1, 28, 8).to(dtype=torch.bfloat16)
    context = torch.randn(1, 3, 16).to(dtype=torch.bfloat16)
    modality = common.block_modality(grid, source, context, 0.725, token_slices=[(0, 28)], clean_prefix_tokens=4)
    with adapters.inference_transformer(session, path, contract, adapter_sha256=sha256(path)) as deployed:
        assert isinstance(deployed, X0Model)
        assert not any(p.requires_grad for p in deployed.parameters())
        assert not deployed.training and not deployed.velocity_model.training
        result = common.denoised_from_x0_model(deployed)(modality)
        expected = common.denoised_from_velocity_model(reference.eval())(modality)
        assert torch.equal(result, expected)
        memory = adapters.parameter_memory(deployed)
        assert memory['parameter_dtypes'] == {'base': ['torch.bfloat16'], 'adapter': ['torch.float32']}
        assert memory['adapter_parameter_bytes'] > 0
        bare = _model().to(dtype=torch.bfloat16)
        bare.load_state_dict(state)
        base_result = common.denoised_from_velocity_model(bare.eval())(modality)
        assert torch.equal(result, base_result) if zero else not torch.equal(result, base_result)
    assert calls == [{'checkpoint_path': 'checked-base', 'device': 'cpu', 'dtype': torch.bfloat16, 'video_only': True}]


@pytest.mark.parametrize('defect', ['missing', 'extra', 'shape', 'nonfinite'])
def test_adapter_loader_refuses_incomplete_or_bad_matrices(saved, defect):
    model, _, path, _ = saved
    weights = exported(model)
    name = next(iter(weights))
    if defect == 'missing':
        del weights[name]
    elif defect == 'extra':
        weights['diffusion_model.extra.lora_A.weight'] = torch.zeros(2, 4)
    elif defect == 'shape':
        weights[name] = torch.zeros(1, 1)
    else:
        weights[name].fill_(float('nan'))
    save_file(weights, path)
    with pytest.raises(ValueError, match='inventory|shape|nonfinite'):
        adapters.load_weights(model, path)


@pytest.mark.parametrize('method', [adapters.UNMERGED, adapters.FUSED])
def test_base_uses_native_session_without_an_adapter(method):
    calls = []
    model = object()
    session = SimpleNamespace(transformer=lambda **kw: (calls.append(kw) or nullcontext(model)))
    with adapters.inference_transformer(session, None, None, method=method) as result:
        assert result is model
    assert calls == [{'loras': ()}]


def test_explicit_fusion_and_unmerged_failure_never_switch_methods(saved, monkeypatch):
    from ltx_trainer import model_loader
    _, _, path, contract = saved
    calls = []
    session = SimpleNamespace(device=torch.device('cpu'), model=SimpleNamespace(
        paths=SimpleNamespace(transformer=lambda: 'checked-base')),
        transformer=lambda **kw: (calls.append(kw) or nullcontext('fused')))
    with adapters.inference_transformer(
        session, path, contract, method=adapters.FUSED, adapter_sha256=sha256(path)) as result:
        assert result == 'fused'
    assert len(calls) == 1 and len(calls[0]['loras']) == 1
    calls.clear()
    monkeypatch.setattr(model_loader, 'load_transformer', lambda **kw: (_ for _ in ()).throw(torch.OutOfMemoryError()))
    with pytest.raises(torch.OutOfMemoryError):
        with adapters.inference_transformer(session, path, contract, adapter_sha256=sha256(path)):
            pytest.fail('resource failure yielded a model')
    assert calls == []


@pytest.mark.parametrize('mode', ['bidirectional', 'causal'])
def test_ordinary_mode_samplers_match_the_loaded_training_function(saved, monkeypatch, mode):
    from ltx_trainer import model_loader
    from scripts.onestep_avatar import evaluate, infer
    from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings
    reference, state, path, contract = saved

    def load(**kwargs):
        base = _model().to(dtype=torch.bfloat16)
        base.load_state_dict(state)
        return base

    monkeypatch.setattr(model_loader, 'load_transformer', load)
    session = SimpleNamespace(device=torch.device('cpu'), model=SimpleNamespace(
        paths=SimpleNamespace(transformer=lambda: 'checked-base')))
    grid = _grid(_geometry(context_latent_frames=8))
    generator = torch.Generator().manual_seed(26)
    capture = torch.randn(1, 28, 8, generator=generator).to(dtype=torch.bfloat16)
    guide = torch.randn(1, 28, 8, generator=generator).to(dtype=torch.bfloat16)
    noise = torch.randn(1, 28, 8, generator=generator).to(dtype=torch.bfloat16)
    context = torch.randn(1, 3, 16, generator=generator).to(dtype=torch.bfloat16)
    settings = BidirectionalSettings() if mode == 'bidirectional' else CausalSettings()
    options = dict(mode=mode, mode_settings=settings, guide_mode='d1', schedule=[0.725, 0], seed=42)
    with adapters.inference_transformer(session, path, contract, adapter_sha256=sha256(path)) as deployed:
        output, record = evaluate.sample_case(deployed, context, grid, capture, guide, noise, **options)
        expected, reference_record = evaluate.sample_case(
            reference, context, grid, capture, guide, noise,
            predict_x0=common.denoised_from_velocity_model(reference.eval()), **options)
        product, product_record = infer.generate(deployed, context, grid, guide, capture[:, :4],
            mode=mode, settings=settings, schedule=[0.725, 0], seed=42, epsilon=noise)
    assert torch.equal(output, expected) and torch.equal(output, product)
    assert record['call_counts'] == reference_record['call_counts'] == product_record['call_counts']


@pytest.mark.parametrize('method', [adapters.UNMERGED, adapters.FUSED])
@pytest.mark.parametrize('defect', ['missing_identity', 'changed_metadata', 'changed_tensor'])
def test_adapter_identity_refuses_changed_preflight_before_weights(
    saved: tuple, monkeypatch: pytest.MonkeyPatch, method: str, defect: str
) -> None:
    from ltx_trainer import model_loader  # noqa: PLC0415 -- native loader guard
    from scripts.onestep_avatar.training import checkpoints  # noqa: PLC0415 -- public recheck
    _, _, path, contract = saved
    identity = sha256(path)
    if defect == 'changed_metadata':
        changed = json.loads(json.dumps(contract))
        changed['adapter']['step'] += 1
        from safetensors.torch import load_file  # noqa: PLC0415 -- changed saved payload fixture
        save_file(load_file(str(path)), path, metadata={CONTRACT_KEY: json.dumps(changed)})
    elif defect == 'changed_tensor':
        from safetensors.torch import load_file  # noqa: PLC0415 -- changed saved payload fixture
        weights = load_file(str(path))
        next(iter(weights.values())).add_(0.125)
        save_file(weights, path, metadata={CONTRACT_KEY: json.dumps(contract)})
    else:
        identity = None
    model_loader_calls = []
    monkeypatch.setattr(model_loader, 'load_transformer', lambda **kw: model_loader_calls.append(kw))
    session = SimpleNamespace(transformer=lambda **_kw: pytest.fail('fused/base loader opened changed adapter'))
    with pytest.raises(ValueError, match=r'SHA-256|changed'):
        checkpoints.recheck_adapter(path, contract, identity)
    with (pytest.raises(ValueError, match=r'SHA-256|changed'),
          adapters.inference_transformer(session, path, contract, method=method, adapter_sha256=identity)):
        pytest.fail('changed adapter yielded a transformer')
    assert not model_loader_calls


def test_adapter_recheck_uses_complete_tensor_and_metadata_checks(saved: tuple) -> None:
    from safetensors.torch import load_file  # noqa: PLC0415 -- incomplete inventory fixture

    from scripts.onestep_avatar.training import checkpoints  # noqa: PLC0415 -- public recheck
    _, _, path, contract = saved
    checkpoints.recheck_adapter(path, contract, sha256(path))
    changed = json.loads(json.dumps(contract))
    changed['adapter']['step'] += 1
    with pytest.raises(ValueError, match='contract changed'):
        checkpoints.recheck_adapter(path, changed, sha256(path))
    weights = load_file(str(path))
    weights.pop(next(iter(weights)))
    save_file(weights, path, metadata={CONTRACT_KEY: json.dumps(contract)})
    with pytest.raises(ValueError, match='inventory'):
        checkpoints.recheck_adapter(path, contract, sha256(path))


@pytest.mark.parametrize('method', [adapters.UNMERGED, adapters.FUSED])
def test_adapter_changed_during_backbone_loading_refused_before_matrix_use(
    saved: tuple, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    from safetensors.torch import load_file  # noqa: PLC0415 -- changed payload fixture

    from ltx_trainer import model_loader  # noqa: PLC0415 -- replace only native backbone loading
    _, state, path, contract = saved
    identity = sha256(path)

    def mutate_while_loading(**_kwargs) -> torch.nn.Module:
        weights = load_file(str(path))
        next(iter(weights.values())).add_(0.125)
        save_file(weights, path, metadata={CONTRACT_KEY: json.dumps(contract)})
        base = _model().to(dtype=torch.bfloat16)
        base.load_state_dict(state)
        return base

    monkeypatch.setattr(model_loader, 'load_transformer', mutate_while_loading)
    monkeypatch.setattr(adapters, 'load_weights', lambda *_args: pytest.fail('changed matrices reached loader'))
    session = SimpleNamespace(device=torch.device('cpu'), model=SimpleNamespace(
        paths=SimpleNamespace(transformer=lambda: 'checked-base')),
        transformer=lambda **kwargs: nullcontext(mutate_while_loading(**kwargs)))
    with (pytest.raises(ValueError, match='adapter content changed'),
          adapters.inference_transformer(session, path, contract, method=method, adapter_sha256=identity)):
        pytest.fail('changed adapter yielded a model')


@pytest.mark.parametrize("application", ["product", "evaluation"])
@pytest.mark.parametrize("defect", [None, "metadata", "tensor"])
def test_ordinary_recheck_precedes_text_native_handles_and_writes(
    saved: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, application: str, defect: str | None
) -> None:
    from safetensors.torch import load_file  # noqa: PLC0415 -- post-preflight mutations

    from scripts.onestep_avatar import evaluate, infer  # noqa: PLC0415 -- actual ordinary entry points
    from scripts.prune.core import preflight  # noqa: PLC0415 -- observe first native handle
    from scripts.prune.data import prompt_cache  # noqa: PLC0415 -- text loading must not run

    _, _, path, contract = saved
    checked = {"contract": contract, "adapter_sha256": sha256(path)}
    args = SimpleNamespace(mode="bidirectional", checkpoint=path, dry_run=False, model="2.5",
                           gpu_id=0, output=tmp_path / "ordinary_output", changed_noise_file=None,
                           cfg=1.0, decode=False)
    specification = SimpleNamespace(paths=SimpleNamespace(transformer=lambda: "checked-base"))

    def mutate_after_preflight() -> None:
        if defect is None:
            return
        metadata = json.loads(json.dumps(contract))
        weights = load_file(str(path))
        if defect == "metadata":
            metadata["adapter"]["step"] += 1
        else:
            next(iter(weights.values())).add_(0.125)
        save_file(weights, path, metadata={CONTRACT_KEY: json.dumps(metadata)})

    def prepare_product(_args: SimpleNamespace) -> tuple:
        mutate_after_preflight()
        return specification, None, None, 30, {}, checked

    def prepare_evaluation(_args: SimpleNamespace) -> tuple:
        mutate_after_preflight()
        return specification, [None, path], [(None, 0, {}, [{}, checked])], {}

    handles = []

    def native_preflight(*_args, **_kwargs) -> None:
        handles.append("native_preflight")
        raise RuntimeError("unchanged adapter reached native preflight")

    monkeypatch.setattr(preflight, "check", native_preflight)
    monkeypatch.setattr(prompt_cache, "get_or_build", lambda *_args, **_kwargs: pytest.fail("text encoder opened"))
    monkeypatch.setattr(evaluate.software, "capture", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(evaluate.software, "check_current", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(infer, "parse_args", lambda _argv: args)
    monkeypatch.setattr(infer, "prepare_product", prepare_product)
    monkeypatch.setattr(evaluate, "prepare_evaluation", prepare_evaluation)
    expected = RuntimeError if defect is None else ValueError
    message = "unchanged adapter reached native preflight" if defect is None else "adapter content changed"
    runner = infer.main if application == "product" else evaluate.execute_evaluation
    argument = [] if application == "product" else args
    with pytest.raises(expected, match=message):
        runner(argument)
    assert handles == (["native_preflight"] if defect is None else [])
    assert not args.output.exists()
