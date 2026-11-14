"""Test suite for the Keras3 module."""

import keras


class MaxPoolModel:
  """Test double `MaxPoolModel` implementing `call` for keras3 tests."""

  def __init__(self, pool_size: int = 2, strides: int = 2) -> None:
    """Initializes the MaxPoolModel instance.

    Args:
        pool_size: Input `pool_size` parameter value (`int`).
        strides: Input `strides` parameter value (`int`).
    """
    super().__init__()
    self.pool = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `pool` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.pool(x)
