"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance.

    Args:
        num_features: The num features (int) for   init  .
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
