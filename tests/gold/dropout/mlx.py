"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: The p (float) for   init  .
    """
    super().__init__()
    self.dropout = nn.Dropout(p)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.dropout(x)
