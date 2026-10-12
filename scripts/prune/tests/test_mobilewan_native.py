"""Stock primitive parity preserves float32 sigmas, timesteps and keyframe identity."""
import torch

from ltx_core.model.transformer.modality import Modality
from scripts.prune.checks.mobilewan_native import stock_prediction


def test_stock_primitive_passes_exact_native_modality():
    source = torch.randn(1, 6, 8, dtype=torch.bfloat16)
    sigma = torch.tensor([0.9489720463752747], dtype=torch.float32)
    timesteps = torch.ones(1, 6, 1) * sigma
    timesteps[:, :2] = 0
    modality = Modality(latent=source, sigma=sigma, timesteps=timesteps,
                        positions=torch.randn(1, 3, 6, 2), context=torch.randn(1, 3, 8),
                        attention_mask=None, keyframes_mask=torch.tensor([[[1.], [1.], [0.], [0.], [0.], [0.]]]))
    calls = []

    def model(*, video, audio, perturbations):
        calls.append(video)
        assert audio is None and perturbations is None
        assert torch.equal(video.latent, modality.latent)
        assert torch.equal(video.sigma, modality.sigma)
        assert torch.equal(video.timesteps, modality.timesteps)
        assert video.sigma.dtype == torch.float32 and video.timesteps.dtype == torch.float32
        assert torch.equal(video.positions, modality.positions)
        assert torch.equal(video.keyframes_mask, modality.keyframes_mask)
        assert video.attention_mask is None
        return video.latent, None

    assert torch.equal(stock_prediction(model, modality), source)
    assert len(calls) == 1
