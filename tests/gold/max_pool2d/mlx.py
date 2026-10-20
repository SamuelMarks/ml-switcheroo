"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class MaxPoolModel:
  """Test helper class MaxPoolModel."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Initializes the MaxPoolModel instance.

    Args:
        kernel_size: Test argument or fixture.
        stride: Test argument or fixture.
    """
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.pool(x)
