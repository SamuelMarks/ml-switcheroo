"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
    """
    super().__init__()
    self.dropout = nn.Dropout(p)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x)
