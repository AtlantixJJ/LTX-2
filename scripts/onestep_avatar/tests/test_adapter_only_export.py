"""Real PEFT payloads match while controlled FSDP contexts exclude frozen weights."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from peft import get_peft_model_state_dict

from scripts.onestep_avatar.model import adapters
from scripts.onestep_avatar.tests.test_causal_core import _model
from scripts.onestep_avatar.training import checkpoints


@pytest.mark.parametrize('missing', [False, True])
def test_gather_only_adapter_leaves_preserves_actual_peft_payload(
    monkeypatch: pytest.MonkeyPatch, missing: bool,
) -> None:
    model = adapters.attach(_model().bfloat16(), rank=2, alpha=2, target='attn', init_seed=11)
    reference = {name: value.clone() for name, value in get_peft_model_state_dict(model).items()}
    calls = []
    initialized = False

    class Wrapped(torch.nn.Module):
        def __init__(self, leaf: torch.nn.Module) -> None:
            super().__init__()
            self._fsdp_wrapped_module = leaf

        def check_is_root(self) -> bool:
            nonlocal initialized
            initialized = True
            return True

        @property
        def module(self) -> torch.nn.Module:
            return self._fsdp_wrapped_module

        @staticmethod
        @contextmanager
        def summon_full_params(module: torch.nn.Module, **kwargs: bool) -> Iterator[None]:
            assert initialized
            assert kwargs == {'recurse': False, 'writeback': False}
            assert module.module.weight.requires_grad
            calls.append(module)
            yield

        def state_dict(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail('full frozen/model state dictionary requested')

    leaves = [(name, module) for name, module in model.named_modules()
              if isinstance(module, torch.nn.Linear) and ('.lora_A.' in name or '.lora_B.' in name)]
    for index, (name, module) in enumerate(leaves):
        if missing and index == 0:
            continue
        parent, key = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), key, Wrapped(module))
    monkeypatch.setattr(checkpoints, 'FSDP', Wrapped)
    model = Wrapped(model)
    accelerator = SimpleNamespace(is_main_process=True)
    if missing:
        with pytest.raises(ValueError, match='incomplete separately wrapped'):
            checkpoints._fsdp_adapter_state(model, accelerator)
        return
    state = checkpoints._fsdp_adapter_state(model, accelerator)
    actual = get_peft_model_state_dict(model.module, state_dict=state)
    assert len(calls) == len(leaves)
    assert set(actual) == set(reference)
    assert all(torch.equal(actual[name], reference[name]) for name in reference)
    assert all(value.device.type == 'cpu' for value in actual.values())
