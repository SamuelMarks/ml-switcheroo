"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class BNModel:
  """Sentinel `BNModel` fixture class used in MLX tests."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance.

    Args:
        num_features: Input `num_features` parameter value (`int`).
    """
    super().__init__()
    self.bn = nn.BatchNorm(num_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.bn(x)
