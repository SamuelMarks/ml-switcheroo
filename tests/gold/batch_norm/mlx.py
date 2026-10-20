"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance.

    Args:
        num_features: Test argument or fixture.
    """
    super().__init__()
    self.bn = nn.BatchNorm(num_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.bn(x)
