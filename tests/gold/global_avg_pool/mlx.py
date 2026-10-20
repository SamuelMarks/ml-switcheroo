"""Test suite for the Mlx module."""

import mlx.core as mx


class GAPModel:
  """Test helper class GAPModel."""

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return mx.mean(x, axis=(1, 2))
