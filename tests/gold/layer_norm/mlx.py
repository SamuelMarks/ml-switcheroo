"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class LayerNormModel:
  """Sentinel `LayerNormModel` fixture class used in MLX tests."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Input `normalized_shape` parameter value (`int`).
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
