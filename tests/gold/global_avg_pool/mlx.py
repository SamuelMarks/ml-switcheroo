"""Test suite for the Mlx module."""

import mlx.core as mx


class GAPModel:
  """Test helper class GAPModel."""

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return mx.mean(x, axis=(1, 2))
