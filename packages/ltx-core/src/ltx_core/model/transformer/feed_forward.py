import torch

from ltx_core.model.transformer.gelu_approx import GELUApprox


class ShapeFaithfulLinear(torch.nn.Linear):
    """Store retained rows/columns, restoring the original GEMM shape at execution.

    This preserves BF16 reduction geometry. It compresses parameters rather than
    claiming less matrix-multiply work; the temporary padded weight is deliberate.
    """

    def __init__(self, in_features: int, out_features: int, *, indices: list[int], axis: int, bias: bool) -> None:
        width = out_features if axis == 0 else in_features
        if (axis not in (0, 1) or not indices or sorted(set(indices)) != indices or
                indices[-1] >= width or indices[0] < 0):
            raise ValueError("retained linear indices must be sorted, unique and within the original width")
        super().__init__(in_features if axis == 0 else len(indices),
                         len(indices) if axis == 0 else out_features, bias=bias)
        self.original_shape = (out_features, in_features)
        self.axis = axis
        # Builders instantiate on meta and materialize only checkpoint tensors.
        # Keep index metadata as Python values so no unmaterialized buffer survives.
        self.retained_indices = tuple(indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor(self.retained_indices, dtype=torch.long, device=self.weight.device)
        weight = self.weight.new_zeros(self.original_shape).index_copy(self.axis, indices, self.weight)
        bias = self.bias
        if bias is not None and self.axis == 0:
            bias = bias.new_zeros(self.original_shape[0]).index_copy(0, indices, bias)
        return torch.nn.functional.linear(x, weight, bias)


class FeedForward(torch.nn.Module):
    def __init__(
        self, dim: int, dim_out: int, mult: int = 4, bias: bool = True, inner_dim: int | None = None,
        active_channels: list[int] | None = None,
        shape_faithful: bool = False,
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
        self.active_channels = (
            active_channels if active_channels is not None and len(active_channels) < inner_dim else None
        )
        if shape_faithful and self.active_channels is not None:
            project_in.proj = ShapeFaithfulLinear(dim, inner_dim, indices=self.active_channels, axis=0, bias=bias)
            self.net[2] = ShapeFaithfulLinear(inner_dim, dim_out, indices=self.active_channels, axis=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_channels is not None:
            activation = self.net[1](self.net[0](x))
            mask = torch.zeros(activation.shape[-1], dtype=activation.dtype, device=activation.device)
            mask[self.active_channels] = 1
            return self.net[2](activation * mask)
        return self.net(x)
