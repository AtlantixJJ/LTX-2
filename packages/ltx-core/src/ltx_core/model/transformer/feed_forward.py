import torch

from ltx_core.model.transformer.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    def __init__(
        self, dim: int, dim_out: int, mult: int = 4, bias: bool = True, inner_dim: int | None = None
    ) -> None:
        super().__init__()
        # ``mult`` remains the checkpoint-compatible default.  Refiner-pruned
        # checkpoints provide a real per-layer width through ``inner_dim``.
        inner_dim = int(dim * mult) if inner_dim is None else int(inner_dim)
        if inner_dim <= 0:
            raise ValueError(f"inner_dim must be positive, got {inner_dim}")
        project_in = GELUApprox(dim, inner_dim, bias=bias)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
