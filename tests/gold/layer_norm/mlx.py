"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Test argument or fixture.
    """
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.ln(x)
