"""Apple MLX golden reference for Group Normalization."""

import mlx.core as mx
import mlx.nn as nn


class GNModel:
  """Group Normalization module wrapper in Apple MLX."""

  def __init__(self, num_groups: int, num_channels: int) -> None:
    """Initialize MLX ``GroupNorm`` layer.

    Args:
        num_groups: Number of channel groups.
        num_channels: Total number of input channels.
    """
    self.gn = nn.GroupNorm(num_groups, num_channels)

  def __call__(self, x: mx.array) -> mx.array:
    """Apply group normalization to ``x``.

    Args:
        x: Input MLX array operand.

    Returns:
        Group-normalized MLX array.
    """
    res: mx.array = self.gn(x)
    return res
