import torch
from scripts.onestep_avatar.tests import test_causal_core as t
from ltx_core.model.transformer.model import LTXModel, LTXModelType

torch.set_num_threads(2)
original_assert = torch.testing.assert_close
def measured_assert(a, b, **kw):
    print('  max_abs=', float((a-b).detach().abs().max()), 'relative_l2=', float((a-b).detach().norm()/b.detach().norm()))
    original_assert(a,b,**kw)
torch.testing.assert_close=measured_assert
for prompt in (False, True):
    def model():
        m = LTXModel(model_type=LTXModelType.VideoOnly, num_attention_heads=2,
            attention_head_dim=4,in_channels=t.CHANNELS,out_channels=t.CHANNELS,
            num_layers=2,cross_attention_dim=8,
            caption_projection=torch.nn.Linear(t.CONTEXT_DIM,8),
            cross_attention_adaln=True,use_prompt_adaln_single=prompt)
        g=torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in m.parameters(): p.copy_(torch.randn(p.shape,generator=g)*.2)
        return m.eval()
    t._model=model
    for sigma in (0.725, 1.0):
        t.SIGMA0 = sigma
        try:
            t.test_cached_rollout_matches_block_causal_full_sequence()
            print('prompt_adaln',prompt,'sigma',sigma,'PASS')
        except AssertionError as e:
            print('prompt_adaln',prompt,'sigma',sigma,'FAIL',str(e))
