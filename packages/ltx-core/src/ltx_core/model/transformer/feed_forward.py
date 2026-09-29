import torch

from ltx_core.model.transformer.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    def __init__(
        self, dim: int, dim_out: int, mult: int = 4, bias: bool = True, inner_dim: int | None = None,
        active_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        # ``mult`` remains the checkpoint-compatible default.  Refiner-pruned
        # checkpoints provide a real per-layer width through ``inner_dim``.
        inner_dim = int(dim * mult) if inner_dim is None else int(inner_dim)
        if inner_dim <= 0:
            raise ValueError(f"inner_dim must be positive, got {inner_dim}")
        project_in = GELUApprox(dim, inner_dim, bias=bias)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out, bias=bias))
        if active_channels is not None and (
            not active_channels or len(set(active_channels)) != len(active_channels)
            or any(index < 0 or index >= inner_dim for index in active_channels)
        ):
            raise ValueError("active FFN channels must be unique and within the original width")
        self.active_channels = active_channels if active_channels is not None and len(active_channels) < inner_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_channels is not None:
            activation = self.net[1](self.net[0](x))
            mask = torch.zeros(activation.shape[-1], dtype=activation.dtype, device=activation.device)
            mask[self.active_channels] = 1
            return self.net[2](activation * mask)
        return self.net(x)
