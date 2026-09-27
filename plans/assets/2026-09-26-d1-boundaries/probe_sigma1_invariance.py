"""CPU source-invariance check, not a real-checkpoint quality evaluation."""
import json
from pathlib import Path

import torch
from scripts.onestep_avatar import causal_core as c
from scripts.onestep_avatar.tests import test_causal_core as t

torch.set_num_threads(2)
torch.manual_seed(123)
model = t._model()
geometry = t._geometry()
grid = t._grid(geometry)
context = t._context()
shape = (1, grid.latent_frames * grid.tokens_per_latent_frame, t.CHANNELS)
target = torch.randn(shape)
guide = torch.randn(shape) * 3 + 2
c0 = target[:, :grid.tokens_per_latent_frame]
plan = geometry.plan(grid.latent_frames)
eps = [c.epsilon_block(target[:, slice(*grid.token_span(*span))], 42+i) for i,span in enumerate(plan)]
tail = [1., .99375, .9875, .98125, .975, .909375, .725, .421875, 0.]
rows = []
with torch.no_grad():
    for teacher_forcing in (False, True):
        for schedule in ([1., 0.], tail):
            outputs = []
            for source in (target, guide):
                cache = c.BlockCache.allocate(grid, geometry, num_layers=len(model.transformer_blocks),
                    inner_dim=model.inner_dim, device=t.DEVICE, dtype=torch.float32)
                out, _ = c.rollout(c.denoised_from_velocity_model(model), grid, geometry,
                    cache, source, context, 1., first_frame_condition=c0,
                    teacher_forcing=teacher_forcing, teacher_tokens=target if teacher_forcing else None,
                    block_epsilons=eps, schedule=schedule)
                assert torch.equal(out[:, :grid.tokens_per_latent_frame], c0)
                outputs.append(out)
            delta = float((outputs[0]-outputs[1]).abs().max())
            assert delta == 0., delta
            rows.append(dict(teacher_forcing=teacher_forcing, denoise_steps=len(schedule)-1,
                source_difference_max=delta, clean_c0_preserved=True))
result = json.dumps(rows, indent=2)
print(result)
Path(__file__).with_name('sigma1_invariance.json').write_text(result+'\n')
