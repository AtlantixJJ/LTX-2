import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.prune.score.export_learned_heads import masks_from_tensor, merge
from scripts.prune.score.export_pruned import PREFIX


def test_merge_algebra_inventory_and_reload(tmp_path):
    path, output = tmp_path/'base.safetensors', tmp_path/'merged.safetensors'
    v, o = f'{PREFIX}.0.attn1.to_v.weight', f'{PREFIX}.0.attn1.to_out.0.weight'
    tensors = {v: torch.randn(4, 8).to(torch.bfloat16), o: torch.randn(8, 4).to(torch.bfloat16), 'unchanged': torch.randn(3)}
    save_file(tensors, str(path), metadata={'config': json.dumps({'transformer': {'pruning': {}}})})
    state = {'0.attn1.v.A': torch.randn(2, 8), '0.attn1.v.B': torch.randn(4, 2),
             '0.attn1.out.A': torch.randn(2, 4), '0.attn1.out.B': torch.randn(8, 2)}
    merge(path, state, output)
    with safe_open(output, framework='pt', device='cpu') as handle:
        for key, name in ((v, 'v'), (o, 'out')):
            expected = (tensors[key].float()+state[f'0.attn1.{name}.B']@state[f'0.attn1.{name}.A']).to(torch.bfloat16)
            assert torch.equal(handle.get_tensor(key), expected)
        assert torch.equal(handle.get_tensor('unchanged'), tensors['unchanged'])
    mask = masks_from_tensor(torch.tensor([[1., 0., 1., 0.]]), {'0.attn1': 4, '0.attn2': 4})
    assert mask['0.attn2'] == [1]*4
    with pytest.raises(ValueError, match='mask'):
        masks_from_tensor(torch.full((1, 4), .5), {'0.attn1': 4})
