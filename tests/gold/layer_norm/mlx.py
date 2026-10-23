"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: The normalized shape (int) for   init  .
    """
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.ln(x)
