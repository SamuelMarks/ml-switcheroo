"""Test suite for the Keras3 module."""

import keras


class GAPModel:
  """Test helper class GAPModel."""

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return keras.ops.mean(x, axis=(1, 2))
