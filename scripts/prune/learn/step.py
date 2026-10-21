"""Register the native model and hook-owned gates inside one trainable step.

DDP sees every trainable parameter and both gate-gradient paths in this forward.
The shared native velocity and loss owners retain all numerical decisions.
Return detached monitoring scalars. Optional previews copy clean predictions to CPU.
The preview owner never retains the clean-output graph or changes the loss.
"""

import torch

from scripts.prune.core import ltx_adapter
from scripts.prune.learn import losses, visualization


class TrainingStep(torch.nn.Module):
    def __init__(self, model, gates, config, arm):
        super().__init__()
        self.model, self.gates = model, gates
        self.config, self.arm = config, arm

    def forward(self, video, epsilon, capture, prefix_tokens, update, phase, regions=None, accumulation=2):
        gates = self.gates.microbatch()
        velocity, clean = ltx_adapter.velocity_forward(self.model, video)
        visualization.capture_prediction(self, clean)
        terms = losses.velocity_mse(velocity, epsilon, capture, prefix_tokens, regions)
        cfg = self.config
        loss, penalty = losses.phase_loss(
            terms,
            gates,
            update=update,
            phase=phase,
            accumulation=accumulation,
            lambda_max=0 if self.arm == "learned_no_sparsity" else cfg["lambda_max"],
            lambda_warmup=cfg.get("lambda_warmup", 300),
            lambda_start_multiplier=cfg.get("lambda_start_multiplier", 0.0),
            logits=self.gates.alpha,
            sparsity_kind=cfg.get("sparsity_penalty", "mean_sigmoid"),
            logit_offset=cfg.get("sparsity_logit_offset", 3.0),
            person=cfg["person_coefficient"] if regions else 0.0,
            small_region=cfg["small_region_coefficient"] if self.arm == "learned_small_regions" else 0.0,
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        return loss, terms["official"].detach(), penalty.detach()
