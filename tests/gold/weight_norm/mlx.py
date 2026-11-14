"""Apple MLX golden reference for Weight-Normalized Linear projection."""

import mlx.core as mx
import mlx.nn as nn


class WNModel:
  """Weight-normalized linear layer in Apple MLX."""

  def __init__(self, in_features: int, out_features: int) -> None:
    """Initialize MLX ``WeightNorm``-wrapped ``nn.Linear`` module.

    Args:
        in_features: Number of input linear features.
        out_features: Number of output linear features.
    """
    self.linear = nn.Linear(in_features, out_features)
    self.weight_g = mx.ones((out_features, 1))

  def __call__(self, x: mx.array) -> mx.array:
    """Apply weight-normalized linear transformation to ``x``.

    Args:
        x: Input MLX array operand.

    Returns:
        Projected MLX array output.
    """
    w = self.linear.weight
    norm = mx.sqrt(mx.sum(mx.square(w), axis=1, keepdims=True) + 1e-12)
    w_normed = w * (self.weight_g / norm)
    res: mx.array = mx.matmul(x, mx.transpose(w_normed)) + self.linear.bias
    return res
