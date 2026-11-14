"""Test suite for the Keras3 module."""

import keras


class GAPModel:
  """Test double `GAPModel` implementing `call` for keras3 tests."""

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `mean` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return keras.ops.mean(x, axis=(1, 2))
