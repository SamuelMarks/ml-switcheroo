"""Test suite for the Mlx module."""

import mlx.core as mx


class FlattenModel:
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: mx.array = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: The start dim (mx.array) for   init  .
    """
    super().__init__()
    self.start_dim = start_dim

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return mx.flatten(x, start_axis=self.start_dim)
