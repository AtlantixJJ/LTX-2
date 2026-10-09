"""Shared PEFT construction/loading; see doc/model/adapters.md.

Training retains velocity output; ordinary inference wraps that same unmerged
function in the stock X0Model. Fusion is an explicit research condition only.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from scripts.onestep_avatar.hashing import sha256

LORA_TARGETS = {
    "attn": ["to_k", "to_q", "to_v", "to_out.0"],
    "attn_ffn": ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
}

UNMERGED = 'peft_unmerged_fp32'
FUSED = 'fused_bf16'
METHODS = (UNMERGED, FUSED)


def parameter_memory(model: torch.nn.Module) -> dict:
    """Count resident parameter storage without claiming peak execution memory."""
    counts = {'base_parameter_bytes': 0, 'adapter_parameter_bytes': 0}
    dtypes = {'base': set(), 'adapter': set()}
    for name, parameter in model.named_parameters():
        role = 'adapter' if '.lora_' in name else 'base'
        counts[role + '_parameter_bytes'] += parameter.numel() * parameter.element_size()
        dtypes[role].add(str(parameter.dtype))
    return {**counts, 'parameter_dtypes': {role: sorted(values) for role, values in dtypes.items()}}


def attach(base: torch.nn.Module, *, rank: int, alpha: int, target: str,
           init_seed: int | None = None) -> torch.nn.Module:
    """Construct the same unmerged fp32 adapter for every consumer."""
    if rank < 1 or alpha != rank or target not in LORA_TARGETS:
        raise ValueError('adapter requires positive rank, alpha=rank and a recorded target')
    if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in base.parameters()):
        raise ValueError('unmerged adapter base parameters must be bf16')
    base.requires_grad_(False)
    if init_seed is not None:
        torch.manual_seed(init_seed)
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha,
        target_modules=LORA_TARGETS[target], lora_dropout=0.0, init_lora_weights=True))
    for name, parameter in model.named_parameters():
        if '.lora_' in name and parameter.dtype != torch.float32:
            raise ValueError('PEFT adapter parameters must be fp32')
    return model


def load_weights(
    model: torch.nn.Module, path: Path, *, expected_sha256: str | None = None
) -> None:
    """Reject incomplete adapters and load saved matrices into the shared function."""
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("adapter content changed before matrix loading")
    exported = load_file(str(path))
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("adapter content changed while reading matrices")
    if any(not name.startswith('diffusion_model.') for name in exported):
        raise ValueError('adapter tensors require the exported ComfyUI prefix')
    state = {name.replace('diffusion_model.', 'base_model.model.', 1): value
             for name, value in exported.items()}
    expected = get_peft_model_state_dict(model)
    if set(state) != set(expected):
        raise ValueError('adapter matrix inventory differs from the constructed PEFT model')
    if any(value.shape != expected[name].shape or not torch.isfinite(value).all()
           for name, value in state.items()):
        raise ValueError('adapter matrix shape differs or contains nonfinite values')
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys or any('.lora_' in name for name in result.missing_keys):
        raise ValueError('adapter matrices did not load completely')


@contextmanager
def inference_transformer(session, checkpoint: Path | None, contract: dict | None,
                          *, method: str = UNMERGED, adapter_sha256: str | None = None):
    """Yield native x0 output without changing the selected adapter function."""
    if method not in METHODS:
        raise ValueError('unsupported adapter application method')
    if checkpoint is not None:
        if contract is None or adapter_sha256 is None:
            raise ValueError('adapter loading requires its checked contract and content SHA-256')
        if sha256(checkpoint) != adapter_sha256:
            raise ValueError('adapter content changed since preflight')
    if checkpoint is None or method == FUSED:
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        loras = () if checkpoint is None else (
            LoraPathStrengthAndSDOps(str(checkpoint), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)
        with session.transformer(loras=loras) as model:
            if checkpoint is not None and sha256(checkpoint) != adapter_sha256:
                raise ValueError('adapter content changed while loading')
            yield model
        return
    if contract is None:
        raise ValueError('unmerged inference requires a checked adapter contract')
    from ltx_core.model.transformer.model import X0Model
    from ltx_trainer.model_loader import load_transformer
    base = load_transformer(checkpoint_path=session.model.paths.transformer(),
                            device=str(session.device), dtype=torch.bfloat16, video_only=True)
    model = None
    try:
        settings = contract['adapter']
        model = attach(base, rank=settings['rank'], alpha=settings['alpha'], target=settings['target'])
        if sha256(checkpoint) != adapter_sha256:
            raise ValueError('adapter content changed before matrix loading')
        load_weights(model, checkpoint, expected_sha256=adapter_sha256)
        if sha256(checkpoint) != adapter_sha256:
            raise ValueError('adapter content changed while loading matrices')
        model.get_base_model().set_gradient_checkpointing(False)
        model.requires_grad_(False).eval()
        with torch.no_grad():
            yield X0Model(model).eval()
    finally:
        del model, base
        if session.device.type == 'cuda':
            torch.cuda.empty_cache()
