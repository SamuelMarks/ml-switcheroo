"""Test suite for the Keras3 module."""

import keras


class GAPModel:
  """Test helper class GAPModel."""

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return keras.ops.mean(x, axis=(1, 2))
