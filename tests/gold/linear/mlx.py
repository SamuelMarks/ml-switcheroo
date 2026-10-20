"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class Model:
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
