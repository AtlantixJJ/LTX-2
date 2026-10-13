import pytest
import torch

from scripts.prune.evaluate.learned_heads import actor_bootstrap, paired_metrics


def test_paired_targets_and_actor_clusters():
    y, e = torch.zeros(1, 6, 2), torch.ones(1, 6, 2)
    v = e - y
    terms = paired_metrics(v, y, v, y, y, e, sigma=0.5, prefix_tokens=2)
    assert terms["official_velocity_mse"] == 0 and terms["teacher_latent_mse"] == 0
    result = actor_bootstrap([{"actor": a, "error": value} for a, value in [("1", 1), ("1", 1), ("2", 3)]], "error")
    assert result["actor_mean"] == 2 and result["actor_count"] == 2 and result["case_count"] == 3
    with pytest.raises(ValueError, match="finite"):
        paired_metrics(v * float("nan"), y, v, y, y, e, sigma=0.5, prefix_tokens=2)
